//! A development-only chat provider that drives the local GitHub Copilot CLI.
//!
//! Each call spawns `copilot` in a fresh empty working directory, writes the
//! rendered prompt to its standard input (never through a shell or the
//! argument list, so quoting and `ARG_MAX` cannot corrupt it), and parses the
//! JSONL event stream for the answer, the model that answered, token usage
//! and AI credits.
//!
//! Results produced this way are **not canonical**: the CLI wraps every
//! prompt in its own hidden system prompt and resolves model names to
//! whatever upstream deployment it chooses. Callers must stamp every artifact
//! with [`NON_CANONICAL_BANNER`] and keep these runs out of any comparison
//! that claims a result.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

use crate::ProviderDescriptor;
use crate::chat::{
    Billing, ChatModel, ChatRequest, ChatResponse, Message, MessageRole, ProviderError, Usage,
};
use crate::credits::{CreditLedger, CreditReservation};
use crate::dispatch::DispatchPermit;

/// Stamped on every artifact of a run that used this provider.
pub const NON_CANONICAL_BANNER: &str = "NON-CANONICAL: Copilot CLI development run (hidden harness system prompt, unpinned upstream model)";

/// The descriptor revision: the CLI does not expose a deployment pin.
pub const UNPINNED_REVISION: &str = "unpinned-upstream";

pub const OVERHEAD_REPORTED: &str = "reported-system-segments";
pub const OVERHEAD_DERIVED: &str = "input-minus-keep-estimate";

/// The CLI's per-process credit ceiling cannot go below this.
pub const MIN_PROCESS_CREDITS: u32 = 30;

/// Credit prices used only to size worst-case reservations against the
/// run's cap. Settled spend always comes from the CLI's own report.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CreditRates {
    pub input_per_1k: f64,
    pub output_per_1k: f64,
    /// Expected hidden harness tokens added to every prompt.
    pub overhead_tokens: u64,
}

impl CreditRates {
    pub fn worst_case(&self, keep_input_tokens: u64, max_output_tokens: u64) -> f64 {
        ((keep_input_tokens + self.overhead_tokens) as f64 * self.input_per_1k
            + max_output_tokens as f64 * self.output_per_1k)
            / 1000.0
    }
}

#[derive(Clone, Debug)]
pub struct CopilotConfig {
    /// The CLI binary; `copilot` on `PATH` unless overridden.
    pub program: PathBuf,
    pub model: String,
    /// Passed as `--reasoning-effort` only when set; some models reject it.
    pub reasoning_effort: Option<String>,
    pub timeout: Duration,
    /// Parent of the per-call empty working directories.
    pub workdir_root: PathBuf,
    pub rates: CreditRates,
    /// `--max-ai-credits` for each CLI process (at least 30).
    pub process_credit_ceiling: u32,
    /// Consecutive failed calls after which the provider stops spawning the
    /// CLI for the rest of the run, so a systematic failure (auth, rate
    /// limit) cannot burn the budget.
    pub breaker_threshold: u32,
}

impl CopilotConfig {
    /// Defaults: `KEEP_COPILOT_BIN` or `copilot`; `KEEP_COPILOT_WORKDIR` or
    /// the OS temporary directory.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            program: std::env::var_os("KEEP_COPILOT_BIN")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("copilot")),
            model: model.into(),
            reasoning_effort: None,
            timeout: Duration::from_secs(180),
            workdir_root: std::env::var_os("KEEP_COPILOT_WORKDIR")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir),
            rates: CreditRates::default(),
            process_credit_ceiling: MIN_PROCESS_CREDITS,
            breaker_threshold: 6,
        }
    }
}

/// The CLI's argument list. The prompt is never part of it.
pub fn cli_args(config: &CopilotConfig) -> Vec<String> {
    let mut args = vec![
        "--model".to_owned(),
        config.model.clone(),
        "--output-format".into(),
        "json".into(),
        "--no-custom-instructions".into(),
        "--disable-builtin-mcps".into(),
        // An unknown name disables every tool; an empty list does not.
        "--available-tools=none".into(),
        "--max-ai-credits".into(),
        config
            .process_credit_ceiling
            .max(MIN_PROCESS_CREDITS)
            .to_string(),
        "--no-auto-update".into(),
        "--no-color".into(),
        "--no-ask-user".into(),
    ];
    if let Some(effort) = &config.reasoning_effort {
        args.push("--reasoning-effort".into());
        args.push(effort.clone());
    }
    args
}

/// Flatten chat messages into the single prompt the CLI accepts.
pub fn render_prompt(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|m| match m.role {
            MessageRole::System | MessageRole::User => m.content.clone(),
            MessageRole::Assistant => format!("Assistant: {}", m.content),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Everything Keep reads from one CLI event stream. Identifiers (session,
/// request and message ids) are deliberately not kept.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedOutput {
    pub text: Option<String>,
    pub model: Option<String>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub reasoning_tokens: u64,
    pub nano_credits: Option<u64>,
    pub premium_requests: Option<f64>,
    /// Hidden system-prompt segments plus tool definitions, as reported.
    pub overhead_tokens: Option<u64>,
    /// Whether the stream shows a model call was started (and so may be billed).
    pub model_called: bool,
    pub exit_code: Option<i64>,
    pub errors: Vec<String>,
    pub unparsed_lines: usize,
}

impl ParsedOutput {
    pub fn credits(&self) -> Option<f64> {
        self.nano_credits.map(|n| n as f64 / 1e9)
    }
}

fn u(v: &Value) -> u64 {
    v.as_u64().unwrap_or(0)
}

/// Parse the CLI's `--output-format json` stream. Non-JSON lines are counted
/// and skipped; usage comes from the last usage checkpoint.
pub fn parse_jsonl(stdout: &str) -> ParsedOutput {
    let mut p = ParsedOutput::default();
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(e) = serde_json::from_str::<Value>(line) else {
            p.unparsed_lines += 1;
            continue;
        };
        let kind = e["type"].as_str().unwrap_or_default();
        let d = &e["data"];
        match kind {
            "assistant.message" => {
                if let Some(text) = d["content"].as_str().filter(|t| !t.trim().is_empty()) {
                    p.text = Some(text.to_owned());
                }
                if let Some(m) = d["model"].as_str() {
                    p.model = Some(m.to_owned());
                }
            }
            "model.call_start" => p.model_called = true,
            "model.call_final_result" => {
                p.model_called = true;
                if p.model.is_none() {
                    p.model = d["model"].as_str().map(str::to_owned);
                }
                if let Some(r) = d["result"].as_str().filter(|r| *r != "success") {
                    p.errors.push(format!("model call result: {r}"));
                }
            }
            "session.usage_checkpoint" => {
                p.nano_credits = d["totalNanoAiu"].as_u64().or(p.nano_credits);
                p.premium_requests = d["totalPremiumRequests"].as_f64().or(p.premium_requests);
                if let Some(models) = d["accountingSnapshot"]["modelMetrics"].as_object() {
                    let (mut i, mut o, mut cr, mut cw, mut r) = (0, 0, 0, 0, 0);
                    for m in models.values() {
                        let us = &m["usage"];
                        i += u(&us["inputTokens"]);
                        o += u(&us["outputTokens"]);
                        cr += u(&us["cacheReadTokens"]);
                        cw += u(&us["cacheWriteTokens"]);
                        r += u(&us["reasoningTokens"]);
                    }
                    p.input_tokens = Some(i);
                    p.output_tokens = Some(o);
                    (
                        p.cache_read_tokens,
                        p.cache_write_tokens,
                        p.reasoning_tokens,
                    ) = (cr, cw, r);
                }
                if let Some(convs) = d["promptCacheBreakState"].as_array() {
                    let mut total = 0;
                    let mut seen = false;
                    for c in convs {
                        for m in c["models"].as_object().into_iter().flat_map(|o| o.values()) {
                            seen = true;
                            total += u(&m["tool_tokens"]);
                            total += m["system_segments"]
                                .as_array()
                                .map(|s| s.iter().map(|x| u(&x["tokens"])).sum::<u64>())
                                .unwrap_or(0);
                        }
                    }
                    if seen {
                        p.overhead_tokens = Some(total);
                    }
                }
            }
            "result" => p.exit_code = e["exitCode"].as_i64(),
            k if k.ends_with("error") || k.ends_with(".failed") => {
                let msg = d["message"]
                    .as_str()
                    .or_else(|| d["error"].as_str())
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("{k} event"));
                p.errors.push(msg);
            }
            _ => {}
        }
    }
    p
}

/// Map a failed CLI run to an error. Configuration problems (sign-in, an
/// unavailable model, a rejected flag) are not retryable; rate limits,
/// crashes and unknown failures are.
pub fn classify_failure(exit: Option<i32>, stderr: &str, parsed: &ParsedOutput) -> ProviderError {
    let mut detail: Vec<&str> = parsed.errors.iter().map(String::as_str).collect();
    let stderr = stderr.trim();
    if !stderr.is_empty() {
        detail.push(stderr);
    }
    let message = if detail.is_empty() {
        format!("copilot exited with {exit:?} and printed no error")
    } else {
        detail.join(" | ")
    };
    let lower = message.to_lowercase();
    let any = |needles: &[&str]| needles.iter().any(|n| lower.contains(n));
    if any(&["rate limit", "rate-limit", "429", "too many requests"]) {
        return ProviderError::RateLimited {
            retry_after_ms: None,
        };
    }
    if any(&[
        "not available",
        "does not support",
        "not authenticated",
        "authentication",
        "unauthorized",
        "forbidden",
        "log in",
        "login",
        "sign in",
        "unknown option",
        "unexpected argument",
        "invalid value",
        "quota",
        "credit limit",
        "max-ai-credits",
        "subscription",
    ]) {
        return ProviderError::Config { message };
    }
    if exit == Some(0) {
        return ProviderError::Malformed {
            message: if parsed.errors.is_empty() {
                "copilot returned no assistant message".into()
            } else {
                message
            },
        };
    }
    ProviderError::Transport {
        message: format!("copilot exited with {exit:?}: {message}"),
    }
}

/// Build a response from a parsed stream that contains an answer.
pub fn response_from(parsed: &ParsedOutput, keep_input_tokens: u64) -> Option<ChatResponse> {
    let text = parsed.text.clone()?;
    let reported = parsed.input_tokens.is_some();
    let input = parsed.input_tokens.unwrap_or(0);
    let (overhead, source) = match parsed.overhead_tokens {
        Some(t) => (t, OVERHEAD_REPORTED),
        None => (input.saturating_sub(keep_input_tokens), OVERHEAD_DERIVED),
    };
    Some(ChatResponse {
        text,
        usage: Usage {
            input_tokens: input,
            output_tokens: parsed.output_tokens.unwrap_or(0),
            provider_reported: reported,
        },
        finish_reason: Some("stop".into()),
        billing: Some(Billing {
            ai_credits: parsed.credits().unwrap_or(0.0),
            premium_requests: parsed.premium_requests,
            reported_model: parsed.model.clone(),
            harness_overhead_tokens: overhead,
            overhead_source: source.into(),
            cache_read_tokens: parsed.cache_read_tokens,
            cache_write_tokens: parsed.cache_write_tokens,
            reasoning_tokens: parsed.reasoning_tokens,
        }),
        served_model: None,
        system_fingerprint: None,
    })
}

#[derive(Debug, Default)]
struct Breaker {
    consecutive_failures: u32,
    tripped: Option<String>,
}

pub struct CopilotCliChat {
    config: CopilotConfig,
    ledger: Arc<CreditLedger>,
    gate: Arc<Semaphore>,
    breaker: Mutex<Breaker>,
    descriptor: ProviderDescriptor,
}

/// The outcome of one CLI process, before ledger settlement.
struct Attempt {
    result: Result<ChatResponse, ProviderError>,
    credits: Option<f64>,
    /// Whether the model may have been reached (and so billed).
    may_have_billed: bool,
}

impl CopilotCliChat {
    /// `gate` bounds concurrent CLI processes and may be shared between the
    /// answer and judge models; `ledger` holds the run's credit cap.
    pub fn new(
        config: CopilotConfig,
        _permit: &DispatchPermit,
        ledger: Arc<CreditLedger>,
        gate: Arc<Semaphore>,
    ) -> Result<Self, ProviderError> {
        if config.model.trim().is_empty() {
            return Err(ProviderError::Config {
                message: "copilot-cli needs a model id".into(),
            });
        }
        if config.rates.input_per_1k <= 0.0 || config.rates.output_per_1k <= 0.0 {
            return Err(ProviderError::Config {
                message: "copilot-cli needs positive credit rates to size reservations".into(),
            });
        }
        let descriptor = ProviderDescriptor {
            provider: "copilot-cli".into(),
            model: config.model.clone(),
            revision: UNPINNED_REVISION.into(),
            mock: false,
        };
        Ok(Self {
            config,
            ledger,
            gate,
            breaker: Mutex::new(Breaker::default()),
            descriptor,
        })
    }

    pub fn ledger(&self) -> &Arc<CreditLedger> {
        &self.ledger
    }

    async fn attempt(&self, req: &ChatRequest) -> Attempt {
        let fail = |e: ProviderError, billed: bool| Attempt {
            result: Err(e),
            credits: None,
            may_have_billed: billed,
        };
        if let Err(e) = std::fs::create_dir_all(&self.config.workdir_root) {
            return fail(
                ProviderError::Config {
                    message: format!(
                        "cannot create copilot work directory {}: {e}",
                        self.config.workdir_root.display()
                    ),
                },
                false,
            );
        }
        let dir = match tempfile::Builder::new()
            .prefix("keep-copilot-")
            .tempdir_in(&self.config.workdir_root)
        {
            Ok(d) => d,
            Err(e) => {
                return fail(
                    ProviderError::Config {
                        message: format!("cannot create an empty work directory: {e}"),
                    },
                    false,
                );
            }
        };
        let mut cmd = tokio::process::Command::new(&self.config.program);
        cmd.args(cli_args(&self.config))
            .current_dir(dir.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return fail(
                    ProviderError::Config {
                        message: format!("cannot start {}: {e}", self.config.program.display()),
                    },
                    false,
                );
            }
        };
        let prompt = render_prompt(&req.messages);
        let mut stdin = child.stdin.take().expect("stdin is piped");
        // Write concurrently with reading output so a large prompt cannot
        // deadlock against a full stdout pipe. A CLI that exits before
        // reading its input surfaces through its exit status instead.
        let writer = tokio::spawn(async move {
            let _ = stdin.write_all(prompt.as_bytes()).await;
            let _ = stdin.shutdown().await;
        });
        let output = match tokio::time::timeout(self.config.timeout, child.wait_with_output()).await
        {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                writer.abort();
                return fail(
                    ProviderError::Transport {
                        message: format!("waiting for copilot failed: {e}"),
                    },
                    true,
                );
            }
            // Dropping the child kills it.
            Err(_) => {
                writer.abort();
                return fail(ProviderError::Timeout, true);
            }
        };
        let _ = writer.await;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let parsed = parse_jsonl(&stdout);
        let credits = parsed.credits();
        let ok = output.status.success() && parsed.exit_code.is_none_or(|c| c == 0);
        let result = match (ok, response_from(&parsed, req.estimated_input_tokens)) {
            (true, Some(r)) => Ok(r),
            _ => Err(classify_failure(output.status.code(), &stderr, &parsed)),
        };
        Attempt {
            result,
            credits,
            may_have_billed: parsed.model_called || credits.is_some(),
        }
    }

    fn settle(&self, reservation: CreditReservation, attempt: &Attempt) {
        match (attempt.credits, attempt.may_have_billed) {
            (Some(c), _) => self.ledger.settle(reservation, c),
            (None, true) => self.ledger.keep(reservation),
            (None, false) => self.ledger.release(reservation),
        }
    }

    fn record(&self, result: &Result<ChatResponse, ProviderError>) {
        let mut b = self.breaker.lock().expect("breaker lock");
        match result {
            Ok(_) => b.consecutive_failures = 0,
            Err(e) => {
                b.consecutive_failures += 1;
                if matches!(e, ProviderError::Config { .. })
                    || b.consecutive_failures >= self.config.breaker_threshold.max(1)
                {
                    b.tripped.get_or_insert_with(|| e.to_string());
                }
            }
        }
    }
}

#[async_trait]
impl ChatModel for CopilotCliChat {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let _slot = self
            .gate
            .acquire()
            .await
            .map_err(|_| ProviderError::Config {
                message: "copilot concurrency gate closed".into(),
            })?;
        if let Some(reason) = &self.breaker.lock().expect("breaker lock").tripped {
            return Err(ProviderError::DispatchRefused {
                message: format!(
                    "copilot-cli stopped after a systematic failure; not retrying: {reason}"
                ),
            });
        }
        let worst = self
            .config
            .rates
            .worst_case(req.estimated_input_tokens, u64::from(req.max_output_tokens));
        let reservation = self.ledger.reserve(worst)?;
        let attempt = self.attempt(req).await;
        self.settle(reservation, &attempt);
        self.record(&attempt.result);
        attempt.result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_never_carry_the_prompt_and_effort_is_optional() {
        let mut c = CopilotConfig::new("model-a");
        c.process_credit_ceiling = 5;
        let a = cli_args(&c);
        assert!(a.contains(&"--available-tools=none".to_owned()));
        let i = a.iter().position(|x| x == "--max-ai-credits").unwrap();
        assert_eq!(a[i + 1], "30");
        assert!(!a.iter().any(|x| x == "-p" || x == "--prompt"));
        assert!(!a.iter().any(|x| x == "--reasoning-effort"));
        c.reasoning_effort = Some("low".into());
        assert!(cli_args(&c).ends_with(&["--reasoning-effort".into(), "low".into()]));
    }

    #[test]
    fn classification_separates_configuration_from_transient_failures() {
        let p = ParsedOutput::default();
        assert!(matches!(
            classify_failure(
                Some(1),
                "Error: Model \"x\" from --model flag is not available.",
                &p
            ),
            ProviderError::Config { .. }
        ));
        assert_eq!(
            classify_failure(Some(1), "HTTP 429 Too Many Requests", &p),
            ProviderError::RateLimited {
                retry_after_ms: None
            }
        );
        let e = classify_failure(Some(137), "", &p);
        assert!(e.retryable(), "{e:?}");
        assert!(matches!(
            classify_failure(Some(0), "", &p),
            ProviderError::Malformed { .. }
        ));
    }

    #[test]
    fn worst_case_includes_hidden_overhead() {
        let r = CreditRates {
            input_per_1k: 0.1,
            output_per_1k: 0.5,
            overhead_tokens: 2000,
        };
        assert!((r.worst_case(1000, 100) - 0.35).abs() < 1e-12);
    }
}
