//! Chat-completion contract shared by answer and judge models.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::ProviderDescriptor;
use crate::budget::BudgetExceeded;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub max_output_tokens: u32,
    pub temperature: f32,
    pub seed: Option<u64>,
    /// The caller's input-token count with the pinned (or approximate)
    /// tokenizer, used for budget reservation before dispatch.
    pub estimated_input_tokens: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// False when the provider did not report usage and the numbers are the
    /// harness's own estimates.
    pub provider_reported: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub text: String,
    pub usage: Usage,
    pub finish_reason: Option<String>,
    /// Harness-reported billing for providers that meter in their own units
    /// (the Copilot CLI's AI credits). `None` for token-priced providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub billing: Option<Billing>,
    /// The `model` the endpoint says served the call, when it reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_model: Option<String>,
    /// The backend configuration fingerprint, when the endpoint reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// What a metered harness reported for one call, kept apart from [`Usage`]
/// so the harness's hidden prompt is never mistaken for Keep's own context.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Billing {
    /// AI credits charged for the call.
    pub ai_credits: f64,
    /// Legacy premium-request count, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub premium_requests: Option<f64>,
    /// The model the harness says answered, which may differ from the one
    /// requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reported_model: Option<String>,
    /// Input tokens the harness added on top of Keep's prompt (its own
    /// system prompt and tool definitions).
    pub harness_overhead_tokens: u64,
    /// `reported-system-segments` when summed from the harness's own segment
    /// counts; `input-minus-keep-estimate` when derived by subtraction.
    pub overhead_source: String,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub reasoning_tokens: u64,
}

/// Every way a provider call can fail. Callers record these as explicit
/// outcomes; none of them is silently converted into an empty answer.
#[derive(Clone, Debug, PartialEq, thiserror::Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderError {
    #[error("dispatch refused: {message}")]
    DispatchRefused { message: String },
    #[error("budget exhausted: {message}")]
    BudgetExhausted { message: String },
    /// HTTP 429. `retry_after_ms` is the wait the provider asked for in a
    /// `retry-after-ms` or `retry-after` header, when it sent one.
    #[error("rate limited (HTTP 429){}", retry_after_ms.map(|ms| format!(", retry after {ms} ms")).unwrap_or_default())]
    RateLimited {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u64>,
    },
    #[error("HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("request timed out")]
    Timeout,
    #[error("transport error: {message}")]
    Transport { message: String },
    #[error("malformed provider response: {message}")]
    Malformed { message: String },
    #[error("missing configuration: {message}")]
    Config { message: String },
    /// The endpoint served a different model than the one pinned. Fatal:
    /// the run must stop rather than mix model versions.
    #[error("served model {served:?} does not match the pinned {expected:?}")]
    ModelMismatch { expected: String, served: String },
}

impl ProviderError {
    /// Whether a retry could plausibly succeed.
    pub fn retryable(&self) -> bool {
        match self {
            ProviderError::RateLimited { .. }
            | ProviderError::Timeout
            | ProviderError::Transport { .. } => true,
            ProviderError::Http { status, .. } => *status >= 500,
            _ => false,
        }
    }

    /// Whether the whole run must stop: no later call can produce a result
    /// that belongs with the others.
    pub fn fatal(&self) -> bool {
        matches!(self, ProviderError::ModelMismatch { .. })
    }

    /// How long to wait before retry number `attempt` (1-based) after this
    /// error: exponential backoff from 1 s, raised to the provider's
    /// `Retry-After` when it asked for longer, and never more than
    /// [`MAX_RETRY_WAIT`].
    pub fn retry_delay(&self, attempt: u32) -> std::time::Duration {
        let backoff = std::time::Duration::from_millis(500 * (1u64 << attempt.clamp(1, 7)));
        let asked = match self {
            ProviderError::RateLimited {
                retry_after_ms: Some(ms),
            } => std::time::Duration::from_millis(*ms),
            _ => std::time::Duration::ZERO,
        };
        backoff.max(asked).min(MAX_RETRY_WAIT)
    }
}

/// The longest single wait between retries.
pub const MAX_RETRY_WAIT: std::time::Duration = std::time::Duration::from_secs(90);

impl From<BudgetExceeded> for ProviderError {
    fn from(value: BudgetExceeded) -> Self {
        ProviderError::BudgetExhausted {
            message: value.to_string(),
        }
    }
}

#[async_trait]
pub trait ChatModel: Send + Sync {
    fn descriptor(&self) -> &ProviderDescriptor;
    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError>;
}
