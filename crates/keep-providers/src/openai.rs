//! A client for OpenAI-compatible `/chat/completions` and `/embeddings`
//! endpoints (vLLM, SGLang, llama.cpp server, hosted APIs and Azure OpenAI).
//!
//! Construction needs a [`DispatchPermit`] and every call is charged to a
//! [`BudgetLedger`] before it is sent. On a failed call the worst-case
//! reservation stays charged, because the provider may still bill for it.
//!
//! When a config pins `served_model`, every reply's `model` field must equal
//! it. The first mismatch is fatal: the client refuses every later call
//! without contacting the endpoint, so one wrong deployment cannot mix model
//! versions into a run.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ProviderDescriptor;
use crate::budget::{BudgetLedger, Price};
use crate::chat::{ChatModel, ChatRequest, ChatResponse, MessageRole, ProviderError, Usage};
use crate::dispatch::DispatchPermit;
use crate::embed::{EmbedResponse, Embedder};
use crate::ratelimit::{RateLimiter, retry_after_ms};

/// How request URLs are formed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ApiStyle {
    /// `{base_url}/chat/completions`, with `model` in the body.
    #[default]
    Openai,
    /// Azure OpenAI v1: `{origin}/openai/v1/chat/completions`, with the
    /// deployment name as `model`. Needs no `api-version`.
    AzureV1,
    /// Classic Azure OpenAI:
    /// `{origin}/openai/deployments/{deployment}/chat/completions?api-version=...`.
    AzureDeployments,
}

impl ApiStyle {
    pub fn is_azure(self) -> bool {
        self != ApiStyle::Openai
    }
}

/// Which header carries the API key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AuthHeader {
    /// `Authorization: Bearer <key>`.
    #[default]
    Bearer,
    /// `api-key: <key>` (Azure OpenAI key authentication).
    ApiKey,
}

#[derive(Clone, Default)]
pub struct OpenAiConfig {
    /// For example `http://127.0.0.1:8000/v1`. For the Azure styles only the
    /// scheme, host and port are used, so a resource endpoint with or without
    /// a trailing `/openai/v1` works.
    pub base_url: String,
    /// Never logged, printed or written to artifacts; `Debug` redacts it.
    pub api_key: Option<String>,
    pub model: String,
    pub revision: String,
    pub price: Price,
    pub timeout: Duration,
    pub api_style: ApiStyle,
    pub auth_header: AuthHeader,
    /// The `api-version` query parameter, if any.
    pub api_version: Option<String>,
    /// The Azure deployment to call. Sent as `model` (v1) or in the path
    /// (classic). `None` sends `model`.
    pub deployment: Option<String>,
    /// The exact `model` every reply must report, e.g.
    /// `gpt-4o-mini-2024-07-18`. `None` skips the check.
    pub served_model: Option<String>,
    /// For reasoning models: sent as `reasoning_effort`, with the output
    /// allowance sent as `max_completion_tokens` instead of `max_tokens`.
    /// `temperature` is then sent only with effort `none`, the one setting
    /// at which these models accept it.
    pub reasoning_effort: Option<String>,
    /// Shared TPM/RPM pacing for every client on the same deployment.
    pub limiter: Option<Arc<RateLimiter>>,
}

impl std::fmt::Debug for OpenAiConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiConfig")
            .field("endpoint", &self.endpoint())
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("model", &self.model)
            .field("revision", &self.revision)
            .field("price", &self.price)
            .field("timeout", &self.timeout)
            .field("reasoning_effort", &self.reasoning_effort)
            .finish()
    }
}

/// Where a model was reached, safe to write into artifacts: the host but
/// never the key or the full URL.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndpointInfo {
    pub api_style: ApiStyle,
    pub auth_header: AuthHeader,
    pub host: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deployment: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_model: Option<String>,
}

impl OpenAiConfig {
    /// Read the endpoint and key from the named environment variables. The
    /// key variable may be unset for local servers that need no key.
    pub fn from_env(
        base_url_env: &str,
        api_key_env: Option<&str>,
        model: &str,
        revision: &str,
        price: Price,
        timeout: Duration,
    ) -> Result<Self, ProviderError> {
        let base_url = std::env::var(base_url_env).map_err(|_| ProviderError::Config {
            message: format!("environment variable {base_url_env} is not set"),
        })?;
        let api_key = api_key_env.and_then(|name| std::env::var(name).ok());
        Ok(Self {
            base_url,
            api_key,
            model: model.into(),
            revision: revision.into(),
            price,
            timeout,
            ..Self::default()
        })
    }

    fn origin(&self) -> Result<String, ProviderError> {
        let url = reqwest::Url::parse(self.base_url.trim()).map_err(|e| ProviderError::Config {
            message: format!("endpoint is not a valid URL: {e}"),
        })?;
        let host = url.host_str().ok_or_else(|| ProviderError::Config {
            message: "endpoint has no host".into(),
        })?;
        Ok(match url.port() {
            Some(port) => format!("{}://{host}:{port}", url.scheme()),
            None => format!("{}://{host}", url.scheme()),
        })
    }

    /// The URL for an API path such as `chat/completions`.
    pub fn url(&self, path: &str) -> Result<String, ProviderError> {
        let mut url = match self.api_style {
            ApiStyle::Openai => format!("{}/{path}", self.base_url.trim().trim_end_matches('/')),
            ApiStyle::AzureV1 => format!("{}/openai/v1/{path}", self.origin()?),
            ApiStyle::AzureDeployments => {
                let deployment =
                    self.deployment
                        .as_deref()
                        .ok_or_else(|| ProviderError::Config {
                            message: "azure-deployments needs a deployment name".into(),
                        })?;
                format!("{}/openai/deployments/{deployment}/{path}", self.origin()?)
            }
        };
        if let Some(v) = &self.api_version {
            url.push_str("?api-version=");
            url.push_str(v);
        } else if self.api_style == ApiStyle::AzureDeployments {
            return Err(ProviderError::Config {
                message: "azure-deployments needs an api-version".into(),
            });
        }
        Ok(url)
    }

    /// The `model` value to send in request bodies, if any.
    fn body_model(&self) -> Option<&str> {
        match self.api_style {
            ApiStyle::AzureDeployments => None,
            _ => Some(self.deployment.as_deref().unwrap_or(&self.model)),
        }
    }

    /// The endpoint's host and pins, for artifacts.
    pub fn endpoint(&self) -> EndpointInfo {
        let host = reqwest::Url::parse(self.base_url.trim())
            .ok()
            .and_then(|u| u.host_str().map(str::to_owned))
            .unwrap_or_else(|| "<invalid>".into());
        EndpointInfo {
            api_style: self.api_style,
            auth_header: self.auth_header,
            host,
            deployment: self.deployment.clone(),
            api_version: self.api_version.clone(),
            served_model: self.served_model.clone(),
        }
    }
}

struct Http {
    config: OpenAiConfig,
    client: reqwest::Client,
    ledger: Arc<BudgetLedger>,
    descriptor: ProviderDescriptor,
    /// The first fatal error; once set, every call is refused unsent.
    halted: Mutex<Option<ProviderError>>,
}

impl Http {
    fn new(config: OpenAiConfig, ledger: Arc<BudgetLedger>) -> Result<Self, ProviderError> {
        // Fail on a malformed endpoint before any call is reserved.
        config.url("chat/completions")?;
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| ProviderError::Config {
                message: e.to_string(),
            })?;
        let descriptor = ProviderDescriptor {
            provider: "openai-compatible".into(),
            model: config.model.clone(),
            revision: config.revision.clone(),
            mock: false,
        };
        Ok(Self {
            config,
            client,
            ledger,
            descriptor,
            halted: Mutex::new(None),
        })
    }

    fn check_halted(&self) -> Result<(), ProviderError> {
        match &*self.halted.lock().expect("halt lock") {
            Some(e) => Err(e.clone()),
            None => Ok(()),
        }
    }

    /// Compare the reply's `model` with the pin. A mismatch halts the client.
    fn check_served(&self, served: Option<&str>) -> Result<(), ProviderError> {
        let Some(expected) = &self.config.served_model else {
            return Ok(());
        };
        if served == Some(expected.as_str()) {
            return Ok(());
        }
        let err = ProviderError::ModelMismatch {
            expected: expected.clone(),
            served: served.unwrap_or("<missing>").to_owned(),
        };
        self.halted
            .lock()
            .expect("halt lock")
            .get_or_insert(err.clone());
        Err(err)
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Value, ProviderError> {
        let url = self.config.url(path)?;
        let mut req = self.client.post(url).json(body);
        if let Some(key) = &self.config.api_key {
            req = match self.config.auth_header {
                AuthHeader::Bearer => req.bearer_auth(key),
                AuthHeader::ApiKey => req.header("api-key", key),
            };
        }
        let resp = req.send().await.map_err(|e| {
            if e.is_timeout() {
                ProviderError::Timeout
            } else {
                ProviderError::Transport {
                    message: e.without_url().to_string(),
                }
            }
        })?;
        let status = resp.status().as_u16();
        let retry_after = retry_after_ms(resp.headers());
        let text = resp.text().await.map_err(|e| ProviderError::Transport {
            message: e.without_url().to_string(),
        })?;
        match status {
            200..=299 => serde_json::from_str(&text).map_err(|e| ProviderError::Malformed {
                message: format!("invalid JSON: {e}"),
            }),
            429 => {
                if let Some(l) = &self.config.limiter {
                    l.pause(retry_after.map(Duration::from_millis));
                }
                Err(ProviderError::RateLimited {
                    retry_after_ms: retry_after,
                })
            }
            _ => Err(ProviderError::Http {
                status,
                body: text.chars().take(500).collect(),
            }),
        }
    }
}

pub struct OpenAiCompatibleChat {
    http: Http,
}

impl OpenAiCompatibleChat {
    pub fn new(
        config: OpenAiConfig,
        _permit: &DispatchPermit,
        ledger: Arc<BudgetLedger>,
    ) -> Result<Self, ProviderError> {
        Ok(Self {
            http: Http::new(config, ledger)?,
        })
    }

    pub fn endpoint(&self) -> EndpointInfo {
        self.http.config.endpoint()
    }
}

fn role_name(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
    }
}

#[async_trait]
impl ChatModel for OpenAiCompatibleChat {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.http.descriptor
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        self.http.check_halted()?;
        let price = self.http.config.price;
        let reservation = self.http.ledger.reserve(
            reserved_input_tokens(request.estimated_input_tokens),
            u64::from(request.max_output_tokens),
            &price,
        )?;
        let messages: Vec<Value> = request
            .messages
            .iter()
            .map(|m| json!({"role": role_name(m.role), "content": m.content}))
            .collect();
        let mut body = json!({ "messages": messages });
        match self.http.config.reasoning_effort.as_deref() {
            None => {
                body["max_tokens"] = json!(request.max_output_tokens);
                body["temperature"] = json!(request.temperature);
            }
            Some(effort) => {
                body["max_completion_tokens"] = json!(request.max_output_tokens);
                body["reasoning_effort"] = json!(effort);
                if effort == "none" {
                    body["temperature"] = json!(request.temperature);
                }
            }
        }
        if let Some(model) = self.http.config.body_model() {
            body["model"] = json!(model);
        }
        if let Some(seed) = request.seed {
            body["seed"] = json!(seed);
        }
        if let Some(l) = &self.http.config.limiter {
            // Azure counts the prompt plus the whole output allowance.
            l.acquire(reservation.input_tokens() + u64::from(request.max_output_tokens))
                .await;
        }
        let value = self.http.post("chat/completions", &body).await?;
        // Settle first: a reply from the wrong model was still billed.
        // Without reported usage, keep the worst case for output.
        let usage = parse_usage(
            &value,
            request.estimated_input_tokens,
            Some(u64::from(request.max_output_tokens)),
        );
        self.http
            .ledger
            .settle(reservation, usage.input_tokens, usage.output_tokens, &price);
        let served_model = value["model"].as_str().map(str::to_owned);
        self.http.check_served(served_model.as_deref())?;
        let choice = &value["choices"][0];
        let text = choice["message"]["content"]
            .as_str()
            .ok_or_else(|| ProviderError::Malformed {
                message: "choices[0].message.content is missing".into(),
            })?
            .to_owned();
        Ok(ChatResponse {
            text,
            usage,
            finish_reason: choice["finish_reason"].as_str().map(str::to_owned),
            billing: None,
            served_model,
            system_fingerprint: value["system_fingerprint"].as_str().map(str::to_owned),
        })
    }
}

/// Input tokens to reserve for a prompt the caller counted as `estimated`.
/// Chat framing adds a few tokens per message that a plain tokenizer count
/// misses, so reserve 5% plus 16 tokens more and settle to the reported
/// usage afterwards.
pub fn reserved_input_tokens(estimated: u64) -> u64 {
    estimated + estimated.div_ceil(20) + 16
}

fn parse_usage(value: &Value, estimated_input: u64, estimated_output: Option<u64>) -> Usage {
    let prompt = value["usage"]["prompt_tokens"].as_u64();
    let completion = value["usage"]["completion_tokens"].as_u64();
    Usage {
        input_tokens: prompt.unwrap_or(estimated_input),
        output_tokens: completion.or(estimated_output).unwrap_or(0),
        provider_reported: prompt.is_some(),
    }
}

pub struct OpenAiCompatibleEmbedder {
    http: Http,
    dimensions: usize,
}

impl OpenAiCompatibleEmbedder {
    pub fn new(
        config: OpenAiConfig,
        dimensions: usize,
        _permit: &DispatchPermit,
        ledger: Arc<BudgetLedger>,
    ) -> Result<Self, ProviderError> {
        Ok(Self {
            http: Http::new(config, ledger)?,
            dimensions,
        })
    }
}

#[async_trait]
impl Embedder for OpenAiCompatibleEmbedder {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.http.descriptor
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbedResponse, ProviderError> {
        let price = self.http.config.price;
        let estimate: u64 = texts
            .iter()
            .map(|t| crate::tokenizer::Tokenizer::count(&crate::ApproxTokenizer, t) as u64)
            .sum();
        let reservation = self.http.ledger.reserve(estimate, 0, &price)?;
        let mut body = json!({"input": texts});
        if let Some(model) = self.http.config.body_model() {
            body["model"] = json!(model);
        }
        let value = self.http.post("embeddings", &body).await?;
        let data = value["data"]
            .as_array()
            .ok_or_else(|| ProviderError::Malformed {
                message: "data array is missing".into(),
            })?;
        let mut vectors = vec![Vec::new(); texts.len()];
        for item in data {
            let index = item["index"].as_u64().unwrap_or(0) as usize;
            let embedding: Vec<f32> = item["embedding"]
                .as_array()
                .ok_or_else(|| ProviderError::Malformed {
                    message: "embedding is missing".into(),
                })?
                .iter()
                .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
                .collect();
            if embedding.len() != self.dimensions || embedding.iter().any(|x| x.is_nan()) {
                return Err(ProviderError::Malformed {
                    message: format!(
                        "embedding has {} dimensions or non-numeric values; expected {}",
                        embedding.len(),
                        self.dimensions
                    ),
                });
            }
            *vectors
                .get_mut(index)
                .ok_or_else(|| ProviderError::Malformed {
                    message: format!("embedding index {index} out of range"),
                })? = embedding;
        }
        if vectors.iter().any(Vec::is_empty) {
            return Err(ProviderError::Malformed {
                message: "provider returned fewer embeddings than inputs".into(),
            });
        }
        let usage = parse_usage(&value, estimate, Some(0));
        self.http
            .ledger
            .settle(reservation, usage.input_tokens, 0, &price);
        Ok(EmbedResponse { vectors, usage })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::BudgetCaps;
    use crate::chat::Message;
    use crate::dispatch::DispatchPolicy;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A one-shot HTTP server on loopback that returns a canned reply. No
    /// external endpoint is contacted.
    async fn canned_server(status: &'static str, body: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 65536];
            let _ = socket.read(&mut buf).await.unwrap();
            let reply = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(reply.as_bytes()).await.unwrap();
        });
        format!("http://{addr}/v1")
    }

    fn permit() -> DispatchPermit {
        DispatchPolicy {
            manifest_allows: true,
            operator_allows: true,
        }
        .permit("test", vec![])
        .unwrap()
    }

    fn config(base_url: String) -> OpenAiConfig {
        OpenAiConfig {
            base_url,
            model: "test-model".into(),
            revision: "r1".into(),
            price: Price {
                input_per_mtok_usd: 1.0,
                output_per_mtok_usd: 1.0,
            },
            timeout: Duration::from_secs(5),
            ..OpenAiConfig::default()
        }
    }

    const SECRET: &str = "sk-test-secret-never-logged";

    /// A loopback server that answers `replies.len()` requests in order and
    /// hands back each raw request (request line, headers and body).
    async fn recording_server(
        replies: Vec<(&'static str, String)>,
    ) -> (String, tokio::sync::mpsc::UnboundedReceiver<String>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            for (status, body) in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut raw = Vec::new();
                let mut buf = vec![0u8; 65536];
                // Read until the headers and the declared body are in.
                loop {
                    let n = socket.read(&mut buf).await.unwrap();
                    raw.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&raw).to_string();
                    if let Some(end) = text.find("\r\n\r\n") {
                        let len = text[..end]
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if raw.len() >= end + 4 + len {
                            break;
                        }
                    }
                    if n == 0 {
                        break;
                    }
                }
                tx.send(String::from_utf8_lossy(&raw).to_string()).unwrap();
                let reply = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(reply.as_bytes()).await.unwrap();
            }
        });
        (format!("http://{addr}"), rx)
    }

    fn reply(model: &str) -> String {
        format!(
            r#"{{"model":"{model}","system_fingerprint":"fp_test","choices":[{{"message":{{"content":"Comet"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":12,"completion_tokens":3}}}}"#
        )
    }

    fn azure(origin: String, style: ApiStyle) -> OpenAiConfig {
        OpenAiConfig {
            api_key: Some(SECRET.into()),
            api_style: style,
            auth_header: AuthHeader::ApiKey,
            deployment: Some("answer-deploy".into()),
            served_model: Some("gpt-4o-mini-2024-07-18".into()),
            model: "gpt-4o-mini".into(),
            ..config(format!("{origin}/openai/v1/"))
        }
    }

    fn request() -> ChatRequest {
        ChatRequest {
            messages: vec![Message::user("hi")],
            max_output_tokens: 50,
            temperature: 0.0,
            seed: Some(1),
            estimated_input_tokens: 10,
        }
    }

    fn caps() -> BudgetCaps {
        BudgetCaps {
            max_calls: 5,
            max_input_tokens: 1000,
            max_output_tokens: 1000,
            max_cost_usd: 1.0,
        }
    }

    #[tokio::test]
    async fn chat_parses_reply_and_settles_reported_usage() {
        let url = canned_server(
            "200 OK",
            r#"{"choices":[{"message":{"content":"Comet"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":3}}"#,
        )
        .await;
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let chat = OpenAiCompatibleChat::new(config(url), &permit(), ledger.clone()).unwrap();
        let resp = chat.complete(&request()).await.unwrap();
        assert_eq!(resp.text, "Comet");
        assert_eq!(resp.usage.input_tokens, 12);
        assert!(resp.usage.provider_reported);
        let used = ledger.usage();
        assert_eq!(
            (used.calls, used.input_tokens, used.output_tokens),
            (1, 12, 3)
        );
    }

    #[tokio::test]
    async fn rate_limits_are_explicit_and_keep_the_reservation() {
        let url = canned_server("429 Too Many Requests", r#"{"error":"slow down"}"#).await;
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let chat = OpenAiCompatibleChat::new(config(url), &permit(), ledger.clone()).unwrap();
        let err = chat.complete(&request()).await.unwrap_err();
        assert_eq!(
            err,
            ProviderError::RateLimited {
                retry_after_ms: None
            }
        );
        assert!(err.retryable());
        assert_eq!(ledger.usage().output_tokens, 50);
    }

    #[tokio::test]
    async fn a_429_reports_retry_after_and_pauses_the_shared_limiter() {
        // The status line carries an extra header line for this test.
        let (origin, _rx) = recording_server(vec![(
            "429 Too Many Requests\r\nretry-after-ms: 1500",
            r#"{"error":{"code":"429"}}"#.into(),
        )])
        .await;
        let limiter = Arc::new(RateLimiter::new(100_000, 600));
        let mut cfg = azure(origin, ApiStyle::AzureV1);
        cfg.limiter = Some(limiter.clone());
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let chat = OpenAiCompatibleChat::new(cfg, &permit(), ledger).unwrap();
        let err = chat.complete(&request()).await.unwrap_err();
        assert_eq!(
            err,
            ProviderError::RateLimited {
                retry_after_ms: Some(1500)
            }
        );
        assert!(err.retry_delay(1) >= Duration::from_millis(1500));
        let stats = limiter.stats();
        assert_eq!((stats.admitted, stats.rejected_429), (1, 1));
    }

    #[tokio::test]
    async fn reasoning_models_get_max_completion_tokens_and_effort() {
        let (origin, mut rx) = recording_server(vec![
            ("200 OK", reply("gpt-4o-mini-2024-07-18")),
            ("200 OK", reply("gpt-4o-mini-2024-07-18")),
        ])
        .await;
        let mut cfg = azure(origin, ApiStyle::AzureV1);
        cfg.reasoning_effort = Some("low".into());
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let chat = OpenAiCompatibleChat::new(cfg.clone(), &permit(), ledger.clone()).unwrap();
        chat.complete(&request()).await.unwrap();
        let raw = rx.recv().await.unwrap();
        assert!(raw.contains(r#""max_completion_tokens":50"#), "{raw}");
        assert!(raw.contains(r#""reasoning_effort":"low""#), "{raw}");
        assert!(!raw.contains("max_tokens\""), "{raw}");
        assert!(!raw.contains("temperature"), "{raw}");

        cfg.reasoning_effort = Some("none".into());
        let chat = OpenAiCompatibleChat::new(cfg, &permit(), ledger).unwrap();
        chat.complete(&request()).await.unwrap();
        let raw = rx.recv().await.unwrap();
        assert!(raw.contains(r#""reasoning_effort":"none""#), "{raw}");
        assert!(raw.contains(r#""temperature":0.0"#), "{raw}");
    }

    #[tokio::test]
    async fn budget_refusal_happens_before_any_network_call() {
        let ledger = Arc::new(BudgetLedger::new(BudgetCaps::default()));
        // An unroutable address: the call must fail on budget, not transport.
        let chat =
            OpenAiCompatibleChat::new(config("http://192.0.2.1:9/v1".into()), &permit(), ledger)
                .unwrap();
        let err = chat.complete(&request()).await.unwrap_err();
        assert!(matches!(err, ProviderError::BudgetExhausted { .. }));
    }

    #[tokio::test]
    async fn azure_v1_sends_the_deployment_as_model_with_an_api_key_header() {
        let (origin, mut rx) =
            recording_server(vec![("200 OK", reply("gpt-4o-mini-2024-07-18"))]).await;
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let chat =
            OpenAiCompatibleChat::new(azure(origin, ApiStyle::AzureV1), &permit(), ledger).unwrap();
        let resp = chat.complete(&request()).await.unwrap();
        assert_eq!(resp.served_model.as_deref(), Some("gpt-4o-mini-2024-07-18"));
        assert_eq!(resp.system_fingerprint.as_deref(), Some("fp_test"));
        let raw = rx.recv().await.unwrap();
        assert!(
            raw.starts_with("POST /openai/v1/chat/completions HTTP/1.1"),
            "{raw}"
        );
        assert!(
            raw.to_ascii_lowercase()
                .contains(&format!("api-key: {SECRET}"))
        );
        assert!(!raw.to_ascii_lowercase().contains("authorization:"));
        assert!(raw.contains(r#""model":"answer-deploy""#), "{raw}");
        let endpoint = chat.endpoint();
        assert_eq!(endpoint.host, "127.0.0.1");
        assert_eq!(endpoint.deployment.as_deref(), Some("answer-deploy"));
    }

    #[tokio::test]
    async fn azure_deployments_style_puts_deployment_and_api_version_in_the_url() {
        let (origin, mut rx) =
            recording_server(vec![("200 OK", reply("gpt-4o-mini-2024-07-18"))]).await;
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let mut cfg = azure(origin, ApiStyle::AzureDeployments);
        cfg.api_version = Some("2024-10-21".into());
        let chat = OpenAiCompatibleChat::new(cfg, &permit(), ledger).unwrap();
        chat.complete(&request()).await.unwrap();
        let raw = rx.recv().await.unwrap();
        assert!(
            raw.starts_with(
                "POST /openai/deployments/answer-deploy/chat/completions?api-version=2024-10-21 HTTP/1.1"
            ),
            "{raw}"
        );
        assert!(!raw.contains(r#""model""#), "{raw}");

        let mut missing = azure("http://127.0.0.1:9".into(), ApiStyle::AzureDeployments);
        missing.api_version = None;
        let ledger = Arc::new(BudgetLedger::new(caps()));
        assert!(matches!(
            OpenAiCompatibleChat::new(missing, &permit(), ledger),
            Err(ProviderError::Config { .. })
        ));
    }

    #[tokio::test]
    async fn bearer_auth_is_the_default() {
        let (origin, mut rx) = recording_server(vec![("200 OK", reply("x"))]).await;
        let mut cfg = config(format!("{origin}/v1"));
        cfg.api_key = Some(SECRET.into());
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let chat = OpenAiCompatibleChat::new(cfg, &permit(), ledger).unwrap();
        chat.complete(&request()).await.unwrap();
        let raw = rx.recv().await.unwrap();
        assert!(
            raw.starts_with("POST /v1/chat/completions HTTP/1.1"),
            "{raw}"
        );
        assert!(raw.contains(&format!("Bearer {SECRET}")), "{raw}");
    }

    #[tokio::test]
    async fn a_wrong_served_model_halts_the_client_and_is_still_charged() {
        // One reply only: a second request would hang, so the test also
        // proves the halted client never sends again.
        let (origin, mut rx) =
            recording_server(vec![("200 OK", reply("gpt-4o-mini-2025-01-01"))]).await;
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let chat =
            OpenAiCompatibleChat::new(azure(origin, ApiStyle::AzureV1), &permit(), ledger.clone())
                .unwrap();
        let err = chat.complete(&request()).await.unwrap_err();
        assert_eq!(
            err,
            ProviderError::ModelMismatch {
                expected: "gpt-4o-mini-2024-07-18".into(),
                served: "gpt-4o-mini-2025-01-01".into(),
            }
        );
        assert!(err.fatal() && !err.retryable());
        let used = ledger.usage();
        assert_eq!(
            (used.calls, used.input_tokens, used.output_tokens),
            (1, 12, 3)
        );
        rx.recv().await.unwrap();

        let again = chat.complete(&request()).await.unwrap_err();
        assert_eq!(again, err);
        assert_eq!(ledger.usage().calls, 1, "a halted client reserves nothing");
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn a_reply_without_a_model_field_fails_the_pin() {
        let (origin, _rx) = recording_server(vec![(
            "200 OK",
            r#"{"choices":[{"message":{"content":"x"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#.into(),
        )])
        .await;
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let chat =
            OpenAiCompatibleChat::new(azure(origin, ApiStyle::AzureV1), &permit(), ledger).unwrap();
        let err = chat.complete(&request()).await.unwrap_err();
        assert!(
            matches!(err, ProviderError::ModelMismatch { ref served, .. } if served == "<missing>")
        );
    }

    #[tokio::test]
    async fn the_key_never_appears_in_errors_or_debug_output() {
        let (origin, _rx) = recording_server(vec![(
            "401 Unauthorized",
            r#"{"error":{"code":"401","message":"Access denied due to invalid subscription key."}}"#.into(),
        )])
        .await;
        let cfg = azure(origin, ApiStyle::AzureV1);
        assert!(!format!("{cfg:?}").contains(SECRET));
        assert!(
            !serde_json::to_string(&cfg.endpoint())
                .unwrap()
                .contains(SECRET)
        );
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let chat = OpenAiCompatibleChat::new(cfg, &permit(), ledger).unwrap();
        let err = chat.complete(&request()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Http { status: 401, .. }));
        assert!(!err.to_string().contains(SECRET));
        assert!(!serde_json::to_string(&err).unwrap().contains(SECRET));
        // A transport failure names neither the key nor the URL.
        let mut down = azure("http://127.0.0.1:1".into(), ApiStyle::AzureV1);
        down.timeout = Duration::from_secs(2);
        let ledger = Arc::new(BudgetLedger::new(caps()));
        let chat = OpenAiCompatibleChat::new(down, &permit(), ledger).unwrap();
        let err = chat.complete(&request()).await.unwrap_err();
        assert!(!err.to_string().contains(SECRET));
        assert!(!err.to_string().contains("127.0.0.1"), "{err}");
    }

    #[test]
    fn azure_endpoints_normalise_to_their_origin() {
        for base in [
            "https://example.invalid",
            "https://example.invalid/",
            "https://example.invalid/openai/v1",
            "https://example.invalid/openai/v1/",
        ] {
            let cfg = OpenAiConfig {
                base_url: base.into(),
                api_style: ApiStyle::AzureV1,
                ..OpenAiConfig::default()
            };
            assert_eq!(
                cfg.url("chat/completions").unwrap(),
                "https://example.invalid/openai/v1/chat/completions"
            );
        }
    }
}
