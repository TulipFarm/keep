//! An optional OpenAI-compatible chat model for Knowledge fact extraction
//! and summaries. Off unless an endpoint is configured.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, anyhow, bail};
use async_trait::async_trait;
use clap::{Args, ValueEnum};
use keep_providers::budget::BudgetUsage;
use keep_providers::openai::{ApiStyle, AuthHeader, OpenAiCompatibleChat, OpenAiConfig};
use keep_providers::{
    BudgetCaps, BudgetLedger, ChatModel, ChatRequest, ChatResponse, DispatchPolicy, Price,
    ProviderDescriptor, ProviderError,
};
use serde_json::{Value, json};
use tokio::sync::Semaphore;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ModelApiStyle {
    /// `{endpoint}/chat/completions` with `model` in the body.
    #[default]
    Openai,
    /// Azure OpenAI v1: `{origin}/openai/v1/chat/completions`, with the
    /// deployment as `model`.
    AzureV1,
    /// Classic Azure OpenAI:
    /// `{origin}/openai/deployments/{deployment}/chat/completions?api-version=...`.
    AzureDeployments,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ModelAuthHeader {
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// `api-key: <key>`, as Azure OpenAI key authentication expects.
    ApiKey,
}

#[derive(Clone, Debug, Default, Args)]
pub struct ModelArgs {
    /// OpenAI-compatible endpoint for model-backed Knowledge extraction and
    /// `knowledge_summarize`, such as `https://api.example.com/v1`. Off
    /// (rule-based extraction) when unset.
    #[arg(long, env = "KEEP_MODEL_ENDPOINT", global = true)]
    pub model_endpoint: Option<String>,
    /// API key for the model endpoint. Prefer KEEP_MODEL_API_KEY_FILE.
    #[arg(
        long,
        env = "KEEP_MODEL_API_KEY",
        hide_env_values = true,
        global = true
    )]
    pub model_api_key: Option<String>,
    /// File holding the API key, such as a mounted secret.
    #[arg(long, env = "KEEP_MODEL_API_KEY_FILE", global = true)]
    pub model_api_key_file: Option<PathBuf>,
    /// Model name sent in requests (for Azure, the model behind the
    /// deployment). Required with an endpoint.
    #[arg(long, env = "KEEP_MODEL_NAME", global = true)]
    pub model_name: Option<String>,
    /// How request URLs are formed.
    #[arg(
        long,
        env = "KEEP_MODEL_API_STYLE",
        value_enum,
        default_value_t,
        global = true
    )]
    pub model_api_style: ModelApiStyle,
    /// The `api-version` query parameter. Required by azure-deployments.
    #[arg(long, env = "KEEP_MODEL_API_VERSION", global = true)]
    pub model_api_version: Option<String>,
    /// Azure deployment name. Required by azure-deployments; sent as
    /// `model` by azure-v1.
    #[arg(long, env = "KEEP_MODEL_DEPLOYMENT", global = true)]
    pub model_deployment: Option<String>,
    /// Header that carries the key. Defaults to api-key for the Azure
    /// styles and bearer otherwise.
    #[arg(long, env = "KEEP_MODEL_AUTH_HEADER", value_enum, global = true)]
    pub model_auth_header: Option<ModelAuthHeader>,
    /// The exact `model` every reply must report. A mismatch stops all
    /// later calls, so one wrong deployment cannot mix model versions.
    #[arg(long, env = "KEEP_MODEL_SERVED_NAME", global = true)]
    pub model_served_name: Option<String>,
    /// `reasoning_effort` for reasoning models.
    #[arg(long, env = "KEEP_MODEL_REASONING_EFFORT", global = true)]
    pub model_reasoning_effort: Option<String>,
    /// Seconds before one model call times out.
    #[arg(
        long,
        env = "KEEP_MODEL_TIMEOUT_SECS",
        default_value_t = 60,
        global = true
    )]
    pub model_timeout_secs: u64,
    /// Model calls in flight at once across this process.
    #[arg(
        long,
        env = "KEEP_MODEL_CONCURRENCY",
        default_value_t = 4,
        global = true
    )]
    pub model_concurrency: usize,
    /// Most output tokens one call may produce.
    #[arg(
        long,
        env = "KEEP_MODEL_REPLY_TOKENS",
        default_value_t = 2048,
        global = true
    )]
    pub model_reply_tokens: u32,
    /// Most model calls this process may make. Calls past any budget fail
    /// with `budget exhausted` until the process restarts.
    #[arg(
        long,
        env = "KEEP_MODEL_BUDGET_CALLS",
        default_value_t = 100_000,
        global = true
    )]
    pub model_budget_calls: u64,
    /// Most input tokens this process may send.
    #[arg(
        long,
        env = "KEEP_MODEL_BUDGET_INPUT_TOKENS",
        default_value_t = 100_000_000,
        global = true
    )]
    pub model_budget_input_tokens: u64,
    /// Most output tokens this process may receive.
    #[arg(
        long,
        env = "KEEP_MODEL_BUDGET_OUTPUT_TOKENS",
        default_value_t = 20_000_000,
        global = true
    )]
    pub model_budget_output_tokens: u64,
    /// Use the model only for `knowledge_summarize`; extraction stays
    /// rule-based.
    #[arg(long, env = "KEEP_NO_MODEL_EXTRACTION", global = true)]
    pub no_model_extraction: bool,
}

/// A configured model and what it may be used for.
#[derive(Clone)]
pub struct ModelSetup {
    pub chat: Arc<dyn ChatModel>,
    /// Identifies the model in evidence labels: name and endpoint host.
    pub id: String,
    pub reply_tokens: u32,
    pub concurrency: usize,
    pub extraction: bool,
    pub budget: Option<Arc<BudgetLedger>>,
    /// Safe to show: style, host and pins, never the key.
    pub endpoint: Value,
}

impl std::fmt::Debug for ModelSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelSetup")
            .field("id", &self.id)
            .field("extraction", &self.extraction)
            .field("concurrency", &self.concurrency)
            .finish_non_exhaustive()
    }
}

impl ModelSetup {
    /// A model built in-process, such as an offline mock in tests.
    pub fn new(chat: Arc<dyn ChatModel>, concurrency: usize) -> Self {
        let d = chat.descriptor();
        let id = format!("{}/{}", d.provider, d.model);
        let concurrency = concurrency.max(1);
        Self {
            chat: Arc::new(Limited::new(chat, concurrency)),
            id,
            reply_tokens: 2048,
            concurrency,
            extraction: true,
            budget: None,
            endpoint: Value::Null,
        }
    }

    pub fn info(&self) -> Value {
        let usage = self.budget.as_ref().map(|b| b.usage()).map(usage_json);
        json!({
            "model": self.id,
            "endpoint": self.endpoint,
            "extraction": self.extraction,
            "concurrency": self.concurrency,
            "usage": usage,
        })
    }
}

fn usage_json(u: BudgetUsage) -> Value {
    json!({ "calls": u.calls, "inputTokens": u.input_tokens, "outputTokens": u.output_tokens })
}

impl ModelArgs {
    /// The configured model, or `None` when no endpoint is set.
    pub fn build(&self) -> anyhow::Result<Option<ModelSetup>> {
        let Some(endpoint) = self
            .model_endpoint
            .as_deref()
            .filter(|e| !e.trim().is_empty())
        else {
            return Ok(None);
        };
        let name = self
            .model_name
            .clone()
            .filter(|n| !n.trim().is_empty())
            .context("KEEP_MODEL_NAME (or --model-name) is required with KEEP_MODEL_ENDPOINT")?;
        let api_key = self.api_key()?;
        if self.model_concurrency == 0 {
            bail!("--model-concurrency must be at least 1");
        }
        if self.model_timeout_secs == 0 {
            bail!("--model-timeout-secs must be at least 1");
        }
        if self.model_reply_tokens == 0 {
            bail!("--model-reply-tokens must be at least 1");
        }
        let api_style = match self.model_api_style {
            ModelApiStyle::Openai => ApiStyle::Openai,
            ModelApiStyle::AzureV1 => ApiStyle::AzureV1,
            ModelApiStyle::AzureDeployments => ApiStyle::AzureDeployments,
        };
        let auth_header = match self.model_auth_header {
            Some(ModelAuthHeader::Bearer) => AuthHeader::Bearer,
            Some(ModelAuthHeader::ApiKey) => AuthHeader::ApiKey,
            None if api_style.is_azure() => AuthHeader::ApiKey,
            None => AuthHeader::Bearer,
        };
        let config = OpenAiConfig {
            base_url: endpoint.trim().to_owned(),
            api_key,
            model: name.clone(),
            revision: self
                .model_served_name
                .clone()
                .unwrap_or_else(|| "unpinned".into()),
            price: Price::default(),
            timeout: Duration::from_secs(self.model_timeout_secs),
            api_style,
            auth_header,
            api_version: self.model_api_version.clone(),
            deployment: self.model_deployment.clone(),
            served_model: self.model_served_name.clone(),
            reasoning_effort: self.model_reasoning_effort.clone(),
            limiter: None,
        };
        let info = config.endpoint();
        let caps = BudgetCaps {
            max_calls: self.model_budget_calls,
            max_input_tokens: self.model_budget_input_tokens,
            max_output_tokens: self.model_budget_output_tokens,
            // No price is configured, so spend is capped by tokens.
            max_cost_usd: f64::MAX,
        };
        let mut problems = Vec::new();
        for (what, v) in [
            ("--model-budget-calls", caps.max_calls),
            ("--model-budget-input-tokens", caps.max_input_tokens),
            ("--model-budget-output-tokens", caps.max_output_tokens),
        ] {
            if v == 0 {
                problems.push(format!("{what} is 0"));
            }
        }
        // Configuring an endpoint is the operator's explicit opt-in.
        let permit = DispatchPolicy {
            manifest_allows: true,
            operator_allows: true,
        }
        .permit("keep-server", problems)
        .map_err(|e| anyhow!("{e}"))?;
        let budget = Arc::new(BudgetLedger::new(caps));
        let chat = OpenAiCompatibleChat::new(config, &permit, budget.clone())
            .map_err(|e| anyhow!("KEEP_MODEL_ENDPOINT: {e}"))?;
        let mut setup = ModelSetup::new(Arc::new(chat), self.model_concurrency);
        setup.id = format!("{name}@{}", info.host);
        setup.reply_tokens = self.model_reply_tokens;
        setup.extraction = !self.no_model_extraction;
        setup.budget = Some(budget);
        setup.endpoint = serde_json::to_value(&info).unwrap_or_default();
        Ok(Some(setup))
    }

    fn api_key(&self) -> anyhow::Result<Option<String>> {
        match (&self.model_api_key, &self.model_api_key_file) {
            (Some(_), Some(_)) => {
                bail!("set KEEP_MODEL_API_KEY or KEEP_MODEL_API_KEY_FILE, not both")
            }
            (Some(k), None) => Ok(Some(k.trim().to_owned()).filter(|k| !k.is_empty())),
            (None, Some(path)) => {
                let k = std::fs::read_to_string(path).with_context(|| {
                    format!("reading KEEP_MODEL_API_KEY_FILE {}", path.display())
                })?;
                let k = k.trim().to_owned();
                if k.is_empty() {
                    bail!("KEEP_MODEL_API_KEY_FILE {} is empty", path.display());
                }
                Ok(Some(k))
            }
            (None, None) => Ok(None),
        }
    }
}

/// A chat model with at most `n` calls in flight.
struct Limited {
    inner: Arc<dyn ChatModel>,
    slots: Semaphore,
}

impl Limited {
    fn new(inner: Arc<dyn ChatModel>, n: usize) -> Self {
        Self {
            inner,
            slots: Semaphore::new(n),
        }
    }
}

#[async_trait]
impl ChatModel for Limited {
    fn descriptor(&self) -> &ProviderDescriptor {
        self.inner.descriptor()
    }

    async fn complete(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let _slot = self
            .slots
            .acquire()
            .await
            .map_err(|_| ProviderError::Config {
                message: "model limiter closed".into(),
            })?;
        self.inner.complete(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(endpoint: &str) -> ModelArgs {
        ModelArgs {
            model_endpoint: Some(endpoint.into()),
            model_name: Some("small-model".into()),
            model_timeout_secs: 5,
            model_concurrency: 2,
            model_reply_tokens: 256,
            model_budget_calls: 10,
            model_budget_input_tokens: 1000,
            model_budget_output_tokens: 1000,
            ..ModelArgs::default()
        }
    }

    #[test]
    fn off_without_an_endpoint() {
        assert!(ModelArgs::default().build().unwrap().is_none());
    }

    #[test]
    fn configuration_problems_are_refused_at_startup() {
        let mut a = args("http://127.0.0.1:9/v1");
        a.model_name = None;
        assert!(
            a.build()
                .unwrap_err()
                .to_string()
                .contains("KEEP_MODEL_NAME")
        );

        let mut a = args("http://127.0.0.1:9/v1");
        a.model_budget_calls = 0;
        assert!(a.build().is_err());

        let mut a = args("http://127.0.0.1:9");
        a.model_api_style = ModelApiStyle::AzureDeployments;
        a.model_deployment = Some("d".into());
        assert!(a.build().unwrap_err().to_string().contains("api-version"));

        let mut a = args("http://127.0.0.1:9/v1");
        a.model_api_key = Some("k".into());
        a.model_api_key_file = Some("/nonexistent".into());
        assert!(a.build().is_err());
    }

    #[test]
    fn the_key_file_is_read_and_never_shown() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("key");
        std::fs::write(&path, "secret-key\n").unwrap();
        let mut a = args("https://models.example.com/v1");
        a.model_api_key_file = Some(path);
        let setup = a.build().unwrap().unwrap();
        assert_eq!(setup.id, "small-model@models.example.com");
        let shown = setup.info().to_string();
        assert!(!shown.contains("secret-key"), "{shown}");
        assert!(shown.contains("models.example.com"), "{shown}");
    }
}
