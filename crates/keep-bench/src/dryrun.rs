//! Offline cost and capacity estimate for a manifest.
//!
//! A dry run builds every context exactly as a real run would (with the
//! in-memory index and, if needed, the hash embedder), counts tokens, and
//! prices the worst case: every answer uses its full output allowance and
//! every judge call sees a maximum-length answer. Nothing is dispatched.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::ablation::ArmKind;
use crate::artifacts::Outcome;
use crate::context::chunks;
use crate::manifest::{EmbeddingProvider, LoadedManifest, ModelSpec, ProviderKind};
use crate::prepare::load_dataset;
use crate::prompt;
use crate::runner::{Mode, RunOptions, run};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ArmEstimate {
    pub arm: String,
    pub kind: String,
    pub records: usize,
    pub answer_calls: u64,
    pub judge_calls: u64,
    pub unavailable: usize,
    pub not_run: usize,
    pub excluded: usize,
    pub failed: usize,
    pub answer_input_tokens: u64,
    pub answer_output_tokens_max: u64,
    pub judge_input_tokens: u64,
    pub judge_output_tokens_max: u64,
    pub context_tokens_max: usize,
    /// Mean context tokens over the calls this arm would make.
    #[serde(default)]
    pub context_tokens_mean: f64,
    pub cost_usd_max: f64,
    /// Worst-case Copilot CLI credits, hidden harness overhead included.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credits_max: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

fn credits_max(spec: &ModelSpec, calls: u64, input: u64, output: u64) -> Option<f64> {
    if spec.provider != ProviderKind::CopilotCli {
        return None;
    }
    let c = spec.copilot.as_ref()?;
    Some(
        (input + calls * c.overhead_tokens) as f64 * c.credits_per_1k_input / 1000.0
            + output as f64 * c.credits_per_1k_output / 1000.0,
    )
}

/// Session fact extraction for the k3 and mk3 arms: one call per distinct
/// session not already in the cache.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExtractionEstimate {
    pub model: String,
    /// Distinct sessions (by extraction key) of the run's histories.
    pub sessions: usize,
    pub cached: usize,
    pub calls: u64,
    pub input_tokens: u64,
    /// At the manifest's `extraction.expected_output_tokens` per call.
    pub output_tokens_expected: u64,
    /// At the extract model's `max_output_tokens` per call.
    pub output_tokens_max: u64,
    pub cost_usd_expected: f64,
    pub cost_usd_max: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credits_expected: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credits_max: Option<f64>,
    /// Of `calls`, session summaries to write, for arms with `summaries`;
    /// expected at [`SUMMARY_EXPECTED_OUTPUT_TOKENS`] each.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub summary_calls: u64,
}

fn is_zero_u64(n: &u64) -> bool {
    *n == 0
}

/// Typical summary reply length, for estimates.
pub const SUMMARY_EXPECTED_OUTPUT_TOKENS: u64 = 150;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapCheck {
    pub cap: String,
    pub estimate: f64,
    pub limit: f64,
    pub within: bool,
}

/// Wall-clock floor under the manifest's TPM/RPM pacing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThroughputEstimate {
    pub tokens_per_minute: u64,
    pub requests_per_minute: u64,
    /// Tokens counted against the quota: every remote prompt (with its
    /// reservation margin) plus its whole output allowance.
    pub quota_tokens: u64,
    pub requests: u64,
    pub minutes_by_tokens: f64,
    pub minutes_by_requests: f64,
    /// The larger of the two: no run can finish faster under this quota.
    pub minutes_min: f64,
    /// Requests whose prompt plus output allowance exceeds the whole TPM
    /// quota; the provider may refuse these outright.
    pub oversized_requests: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DryRunReport {
    pub manifest_id: String,
    pub manifest_hash: String,
    pub dataset: String,
    pub questions: usize,
    pub histories: usize,
    pub seeds: usize,
    pub tokenizer: String,
    pub tokenizer_exact: bool,
    pub arms: Vec<ArmEstimate>,
    pub embedding_calls: u64,
    pub embedding_input_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extraction: Option<ExtractionEstimate>,
    pub total_calls: u64,
    pub total_input_tokens: u64,
    pub total_output_tokens_max: u64,
    pub total_cost_usd_max: f64,
    pub caps: Vec<CapCheck>,
    /// Full-context fit over the whole dataset (holdout included), from
    /// lengths only.
    pub full_context_fit_fraction: f64,
    pub full_context_gate_eligible: bool,
    /// Why `run` would refuse remote dispatch as configured; empty when it
    /// would be allowed (given the CLI flag).
    pub dispatch_blockers: Vec<String>,
    pub remote_models: bool,
    pub warnings: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_canonical: Option<String>,
    /// The parity protocol and split, when not Keep's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<String>,
    /// True when the run would score the sealed holdout (`split = "full"`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub consumes_holdout: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub throughput: Option<ThroughputEstimate>,
    /// True when a confirmation manifest was estimated on its sealed holdout.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub sealed_holdout: bool,
}

pub async fn dry_run(lm: &LoadedManifest) -> anyhow::Result<DryRunReport> {
    dry_run_with(lm, false).await
}

/// `unseal` lets a confirmation manifest be estimated on the sealed holdout
/// it would score. Nothing is called or scored; only lengths are counted.
pub async fn dry_run_with(lm: &LoadedManifest, unseal: bool) -> anyhow::Result<DryRunReport> {
    let m = &lm.manifest;
    let out = run(
        lm,
        &RunOptions {
            mode: Mode::Estimate,
            unseal,
            allow_model_calls: false,
            results_root: None,
            spend: None,
        },
    )
    .await?;
    let (tok, _) = m.answer_tokenizer();
    let answer_price = m.models.answer.price();
    let judge_price = m.models.judge.price();
    let judge_template: u64 = if m.protocol.is_keep() {
        prompt::judge_messages("", "", "", false)
            .iter()
            .map(|x| tok.count(&x.content) as u64)
            .sum()
    } else {
        // The longest of the protocol's judge templates.
        let types = crate::protocol::LONGMEMEVAL_TYPES;
        types
            .iter()
            .flat_map(|t| [(*t, "q"), (*t, "q_abs")])
            .map(|(category, question_id)| {
                m.protocol
                    .name
                    .judge_messages(
                        &crate::protocol::JudgeInput {
                            question_id,
                            category,
                            ..Default::default()
                        },
                        "",
                    )
                    .iter()
                    .map(|x| tok.count(&x.content) as u64)
                    .sum::<u64>()
            })
            .max()
            .unwrap_or_default()
    };

    // Remote calls reserve a margin over the counted prompt; price that too.
    let reserve = |spec: &ModelSpec, tokens: u64| {
        if spec.provider == ProviderKind::OpenaiCompatible {
            keep_providers::openai::reserved_input_tokens(tokens)
        } else {
            tokens
        }
    };
    // Majority voting can take up to `votes` judge calls per answer; cached
    // verdicts can take none. Estimate the worst case.
    let judge_votes = m
        .judge_consistency
        .as_ref()
        .map_or(1, |j| u64::from(j.votes));
    let (tpm, rpm) = m.dispatch.pacing()?;
    let paced = tpm.is_some() || rpm.is_some();
    let (tpm, rpm) = (tpm.unwrap_or(0), rpm.unwrap_or(0));
    let mut oversized = 0u64;
    let mut arms: Vec<ArmEstimate> = Vec::new();
    for spec in &m.arms {
        let mut context_total = 0u64;
        let mut e = ArmEstimate {
            arm: spec.name.clone(),
            kind: spec.kind.label().into(),
            ..Default::default()
        };
        for r in out.records.iter().filter(|r| r.arm == spec.name) {
            e.records += 1;
            match &r.outcome {
                Outcome::Estimated => {
                    let p = r.prompt.as_ref().expect("estimated records carry a prompt");
                    e.answer_calls += 1;
                    e.judge_calls += judge_votes;
                    let reserved = reserve(&m.models.answer, p.input_tokens as u64);
                    if tpm > 0 && reserved + u64::from(m.models.answer.max_output_tokens) > tpm {
                        oversized += 1;
                    }
                    e.answer_input_tokens += reserved;
                    e.answer_output_tokens_max += u64::from(m.models.answer.max_output_tokens);
                    let q = tok.count(&r.gold) as u64
                        + u64::from(m.models.answer.max_output_tokens)
                        + judge_template
                        + 64;
                    e.judge_input_tokens += judge_votes * reserve(&m.models.judge, q);
                    e.judge_output_tokens_max +=
                        judge_votes * u64::from(m.models.judge.max_output_tokens);
                    let context = r.context.as_ref().map_or(0, |c| c.tokens);
                    context_total += context as u64;
                    e.context_tokens_max = e.context_tokens_max.max(context);
                }
                Outcome::Unavailable { reason } => {
                    e.unavailable += 1;
                    e.reason.get_or_insert_with(|| reason.clone());
                }
                Outcome::NotRun { reason } => {
                    e.not_run += 1;
                    e.reason.get_or_insert_with(|| reason.clone());
                }
                Outcome::Excluded { .. } => e.excluded += 1,
                Outcome::Failed { error } => {
                    e.failed += 1;
                    e.reason.get_or_insert_with(|| error.clone());
                }
                Outcome::Answered => {}
            }
        }
        if e.answer_calls > 0 {
            e.context_tokens_mean = context_total as f64 / e.answer_calls as f64;
        }
        e.cost_usd_max = answer_price.cost(e.answer_input_tokens, e.answer_output_tokens_max)
            + judge_price.cost(e.judge_input_tokens, e.judge_output_tokens_max);
        let answer_credits = credits_max(
            &m.models.answer,
            e.answer_calls,
            e.answer_input_tokens,
            e.answer_output_tokens_max,
        );
        let judge_credits = credits_max(
            &m.models.judge,
            e.judge_calls,
            e.judge_input_tokens,
            e.judge_output_tokens_max,
        );
        if answer_credits.is_some() || judge_credits.is_some() {
            e.credits_max = Some(answer_credits.unwrap_or(0.0) + judge_credits.unwrap_or(0.0));
        }
        arms.push(e);
    }

    // Embedding volume: every chunk of every history once plus each query.
    let (mut embedding_calls, mut embedding_input_tokens) = (0u64, 0u64);
    if m.arms
        .iter()
        .any(|a| matches!(a.kind, ArmKind::Dense { .. }))
    {
        let full = load_dataset(lm)?;
        let run_histories: HashSet<&str> =
            out.records.iter().map(|r| r.history_id.as_str()).collect();
        let dense_arms = m
            .arms
            .iter()
            .filter(|a| matches!(a.kind, ArmKind::Dense { .. }))
            .count() as u64;
        for h in full
            .histories
            .iter()
            .filter(|h| run_histories.contains(h.id.as_str()))
        {
            let cs = chunks(h, None, m.retrieval.chunking);
            embedding_calls += cs.len().div_ceil(m.embedding.request_batch()) as u64;
            embedding_input_tokens += cs.iter().map(|c| tok.count(&c.text) as u64).sum::<u64>();
        }
        let questions: BTreeMap<&str, &str> = out
            .records
            .iter()
            .map(|r| (r.question_id.as_str(), r.history_id.as_str()))
            .collect();
        for q in full
            .questions
            .iter()
            .filter(|q| questions.contains_key(q.id.as_str()))
        {
            embedding_calls += dense_arms;
            embedding_input_tokens += dense_arms * tok.count(&q.query.text) as u64;
        }
    }
    let embedding_cost = if m.embedding.provider == EmbeddingProvider::OpenaiCompatible {
        embedding_input_tokens as f64 * m.embedding.input_usd_per_mtok / 1e6
    } else {
        0.0
    };

    let extraction = extraction_estimate(lm, &out.records, tok.as_ref())?;
    let (x_calls, x_input, x_output, x_cost, x_credits) = match (&extraction, m.extract_model()) {
        (Some(x), Some(spec)) => (
            if spec.provider == ProviderKind::OpenaiCompatible {
                x.calls
            } else {
                0
            },
            if spec.provider == ProviderKind::OpenaiCompatible {
                x.input_tokens
            } else {
                0
            },
            if spec.provider == ProviderKind::OpenaiCompatible {
                x.output_tokens_max
            } else {
                0
            },
            x.cost_usd_max,
            x.credits_max.unwrap_or(0.0),
        ),
        _ => (0, 0, 0, 0.0, 0.0),
    };

    let remote_answer = m.models.answer.provider == ProviderKind::OpenaiCompatible;
    let remote_judge = m.models.judge.provider == ProviderKind::OpenaiCompatible;
    let remote_embed = m.embedding.provider == EmbeddingProvider::OpenaiCompatible;
    let sum = |f: fn(&ArmEstimate) -> u64| arms.iter().map(f).sum::<u64>();
    // Caps apply to remote calls only.
    let total_calls = if remote_answer {
        sum(|a| a.answer_calls)
    } else {
        0
    } + if remote_judge {
        sum(|a| a.judge_calls)
    } else {
        0
    } + if remote_embed { embedding_calls } else { 0 }
        + x_calls;
    let total_input = if remote_answer {
        sum(|a| a.answer_input_tokens)
    } else {
        0
    } + if remote_judge {
        sum(|a| a.judge_input_tokens)
    } else {
        0
    } + if remote_embed {
        embedding_input_tokens
    } else {
        0
    } + x_input;
    let total_output = if remote_answer {
        sum(|a| a.answer_output_tokens_max)
    } else {
        0
    } + if remote_judge {
        sum(|a| a.judge_output_tokens_max)
    } else {
        0
    } + x_output;
    let total_cost = arms.iter().map(|a| a.cost_usd_max).sum::<f64>() + embedding_cost + x_cost;
    let b = &m.budget;
    let credits: Option<f64> = m
        .uses_copilot()
        .then(|| arms.iter().filter_map(|a| a.credits_max).sum::<f64>() + x_credits);
    let mut caps: Vec<(&str, f64, f64)> = vec![
        ("model calls", total_calls as f64, b.max_model_calls as f64),
        (
            "input tokens",
            total_input as f64,
            b.max_input_tokens as f64,
        ),
        (
            "output tokens (worst case)",
            total_output as f64,
            b.max_output_tokens as f64,
        ),
        ("cost USD (worst case)", total_cost, b.max_cost_usd),
    ];
    if let Some(c) = credits {
        caps.push((
            "AI credits (worst case)",
            c,
            b.max_ai_credits.unwrap_or(0.0),
        ));
    }
    let caps = caps
        .into_iter()
        .map(|(cap, estimate, limit)| CapCheck {
            cap: cap.into(),
            estimate,
            limit,
            within: estimate <= limit,
        })
        .collect();

    // Length-only fit over the full dataset, independent of the split.
    let mask = crate::prepare::compute_fit_mask(lm)?;

    let remote = m.uses_remote_models();
    let mut blockers = Vec::new();
    if remote {
        if !m.dispatch.allow_model_calls {
            blockers.push("the manifest does not set dispatch.allow_model_calls = true".into());
        }
        blockers.extend(m.dispatch_problems());
    }
    let mut warnings = out.meta.warnings.clone();
    let throughput = paced.then(|| {
        let quota_tokens = total_input + total_output;
        let per = |n: u64, limit: u64| {
            if limit == 0 {
                0.0
            } else {
                n as f64 / limit as f64
            }
        };
        let (by_tokens, by_requests) = (per(quota_tokens, tpm), per(total_calls, rpm));
        ThroughputEstimate {
            tokens_per_minute: tpm,
            requests_per_minute: rpm,
            quota_tokens,
            requests: total_calls,
            minutes_by_tokens: by_tokens,
            minutes_by_requests: by_requests,
            minutes_min: by_tokens.max(by_requests),
            oversized_requests: oversized,
        }
    });
    if oversized > 0 {
        warnings.push(format!(
            "{oversized} answer requests need more than the whole {tpm}-token-per-minute quota; the provider may refuse them, and each refusal is recorded as a failed call"
        ));
    }
    if m.uses_mock_models() {
        warnings.push(
            "mock providers are configured: a run would produce plumbing checks, not results"
                .into(),
        );
    }
    Ok(DryRunReport {
        manifest_id: m.id.clone(),
        manifest_hash: lm.hash.to_string(),
        dataset: m.dataset.kind.as_str().into(),
        questions: out.meta.questions_run,
        histories: out.meta.histories_run,
        seeds: m.seeds.len(),
        tokenizer: tok.id().into(),
        tokenizer_exact: tok.exact(),
        arms,
        embedding_calls,
        embedding_input_tokens,
        extraction,
        total_calls,
        total_input_tokens: total_input,
        total_output_tokens_max: total_output,
        total_cost_usd_max: total_cost,
        caps,
        full_context_fit_fraction: mask.fit_fraction,
        full_context_gate_eligible: mask.gate_eligible,
        dispatch_blockers: blockers,
        remote_models: remote,
        warnings,
        non_canonical: m.non_canonical_label().map(str::to_owned),
        protocol: (!m.protocol.is_keep()).then(|| {
            format!(
                "{} (split {})",
                m.protocol.name.as_str(),
                m.protocol.split.as_str()
            )
        }),
        consumes_holdout: m.protocol.split == crate::protocol::SplitScope::Full,
        throughput,
        sealed_holdout: unseal && m.phase == crate::manifest::Phase::Confirmation,
    })
}

/// Price the extraction calls a run would still make: the distinct sessions
/// of the run's histories that the cache does not hold.
fn extraction_estimate(
    lm: &LoadedManifest,
    records: &[crate::artifacts::QuestionRecord],
    tok: &dyn keep_providers::Tokenizer,
) -> anyhow::Result<Option<ExtractionEstimate>> {
    let m = &lm.manifest;
    let (Some(spec), Some(x)) = (m.extract_model(), &m.extraction) else {
        return Ok(None);
    };
    let run_histories: HashSet<&str> = records
        .iter()
        .filter(|r| !matches!(r.outcome, Outcome::Excluded { .. }))
        .map(|r| r.history_id.as_str())
        .collect();
    let extractor = crate::facts::extractor(spec, None);
    let summarizer = (x.summary_tokens > 0).then(|| crate::facts::summarizer(spec, None));
    let full = load_dataset(lm)?;
    let mut seen = HashSet::new();
    let mut e = ExtractionEstimate {
        model: crate::facts::model_id(spec),
        ..Default::default()
    };
    for h in full
        .histories
        .iter()
        .filter(|h| run_histories.contains(h.id.as_str()))
    {
        for s in &h.sessions {
            let input = crate::facts::session_input(s);
            if let Some(summarizer) = &summarizer
                && seen.insert(summarizer.key(&input))
                && !summarizer.is_cached(&input)
            {
                e.summary_calls += 1;
                e.input_tokens += keep_knowledge::session_summary::summary_request(
                    &input,
                    summarizer.max_output_tokens(),
                )
                .messages
                .iter()
                .map(|msg| tok.count(&msg.content) as u64)
                .sum::<u64>();
            }
            if !seen.insert(extractor.key(&input)) {
                continue;
            }
            e.sessions += 1;
            if extractor.is_cached(&input) {
                e.cached += 1;
                continue;
            }
            e.calls += 1;
            let request: u64 = extractor
                .request(&input)
                .messages
                .iter()
                .map(|msg| tok.count(&msg.content) as u64)
                .sum();
            e.input_tokens += if spec.provider == ProviderKind::OpenaiCompatible {
                keep_providers::openai::reserved_input_tokens(request)
            } else {
                request
            };
        }
    }
    e.output_tokens_expected =
        e.calls * x.expected_output_tokens + e.summary_calls * SUMMARY_EXPECTED_OUTPUT_TOKENS;
    e.calls += e.summary_calls;
    e.output_tokens_max = e.calls * u64::from(spec.max_output_tokens);
    let price = spec.price();
    e.cost_usd_expected = price.cost(e.input_tokens, e.output_tokens_expected);
    e.cost_usd_max = price.cost(e.input_tokens, e.output_tokens_max);
    e.credits_expected = credits_max(spec, e.calls, e.input_tokens, e.output_tokens_expected);
    e.credits_max = credits_max(spec, e.calls, e.input_tokens, e.output_tokens_max);
    Ok(Some(e))
}

pub fn markdown(r: &DryRunReport) -> String {
    let banner = r
        .non_canonical
        .as_ref()
        .map(|l| format!("> **{l}**\n\n"))
        .unwrap_or_default();
    let mut out = banner;
    out += &format!(
        "# Dry run: {}\n\nManifest hash `{}`. Dataset {}: {} questions in {} histories ({}), {} seed(s). \
         Tokens counted with {}{}.\n\nNo model or database was called.\n\n",
        r.manifest_id,
        r.manifest_hash,
        r.dataset,
        r.questions,
        r.histories,
        if r.consumes_holdout {
            "full set, holdout included"
        } else if r.sealed_holdout {
            "sealed holdout, confirmation"
        } else {
            "development split"
        },
        r.seeds,
        r.tokenizer,
        if r.tokenizer_exact { "" } else { " (estimate)" }
    );
    if let Some(p) = &r.protocol {
        out += &format!("Scoring protocol: {p}.\n\n");
    }
    out.push_str("| Arm | Kind | Answer calls | Unavailable | Not run | Answer input tokens | Mean context tokens | Max context tokens | Worst-case cost USD | Worst-case AI credits | Note |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
    for a in &r.arms {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.0} | {} | {:.4} | {} | {} |\n",
            a.arm,
            a.kind,
            a.answer_calls,
            a.unavailable,
            a.not_run,
            a.answer_input_tokens,
            a.context_tokens_mean,
            a.context_tokens_max,
            a.cost_usd_max,
            a.credits_max
                .map(|c| format!("{c:.2}"))
                .unwrap_or_else(|| "—".into()),
            a.reason.as_deref().unwrap_or("")
        ));
    }
    if let Some(x) = &r.extraction {
        let credits = |c: Option<f64>| {
            c.map(|c| format!(", {c:.1} AI credits"))
                .unwrap_or_default()
        };
        out.push_str(&format!(
            "\nFact extraction ({}): {} distinct sessions, {} cached, {} calls to make; {} input tokens.              Expected {} output tokens (${:.2}{}); worst case {} (${:.2}{}). Caps below count the worst case.\n",
            x.model,
            x.sessions,
            x.cached,
            x.calls,
            x.input_tokens,
            x.output_tokens_expected,
            x.cost_usd_expected,
            credits(x.credits_expected),
            x.output_tokens_max,
            x.cost_usd_max,
            credits(x.credits_max),
        ));
        if x.summary_calls > 0 {
            out.push_str(&format!(
                "Of those calls, {} write session summaries.\n",
                x.summary_calls
            ));
        }
    }
    out.push_str(&format!(
        "\nEmbedding: {} calls, {} input tokens.\n\nFull-context fit (length only, whole dataset): {:.1}% — {}.\n\n## Caps (remote calls only)\n\n",
        r.embedding_calls,
        r.embedding_input_tokens,
        r.full_context_fit_fraction * 100.0,
        if r.full_context_gate_eligible { "gate-eligible" } else { "below the gate minimum" }
    ));
    let num = |cap: &str, v: f64| {
        if cap.contains("USD") {
            format!("${v:.2}")
        } else if cap.contains("credits") {
            format!("{v:.1}")
        } else {
            format!("{v:.0}")
        }
    };
    for c in &r.caps {
        out.push_str(&format!(
            "- {} {}: estimate {} against cap {}\n",
            if c.within { "within" } else { "OVER" },
            c.cap,
            num(&c.cap, c.estimate),
            num(&c.cap, c.limit)
        ));
    }
    if let Some(t) = &r.throughput {
        out.push_str(&format!(
            "\n## Throughput\n\n{} quota tokens over {} requests at {} tokens/min and {} requests/min: at least {:.0} minutes ({:.1} hours). {} answer requests exceed the whole token quota.\n",
            t.quota_tokens,
            t.requests,
            t.tokens_per_minute,
            t.requests_per_minute,
            t.minutes_min,
            t.minutes_min / 60.0,
            t.oversized_requests
        ));
    }
    if r.remote_models {
        out.push_str("\n## Dispatch\n\n");
        if r.dispatch_blockers.is_empty() {
            out.push_str("Configuration allows dispatch once `--allow-model-calls` is passed.\n");
        } else {
            out.push_str("`run` would refuse remote calls:\n\n");
            for b in &r.dispatch_blockers {
                out.push_str(&format!("- {b}\n"));
            }
        }
    }
    if !r.warnings.is_empty() {
        out.push_str("\n## Warnings\n\n");
        for w in &r.warnings {
            out.push_str(&format!("- {w}\n"));
        }
    }
    out
}
