//! Executes a manifest: builds every arm's context for every question, then
//! answers and judges it once per seed.
//!
//! Remote providers are only constructed after [`DispatchPolicy::permit`]
//! succeeds. Mock providers need no permit and every artifact they produce is
//! marked as a mock.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use keep_core::{ContentHash, canonical_json};
use keep_providers::copilot::{CopilotCliChat, CopilotConfig, CreditRates};
use keep_providers::openai::{
    EndpointInfo, OpenAiCompatibleChat, OpenAiCompatibleEmbedder, OpenAiConfig,
};
use keep_providers::ratelimit::RateLimiter;
use keep_providers::{
    BudgetLedger, ChatModel, ChatRequest, ChatResponse, CreditLedger, CreditUsage, DispatchPolicy,
    Embedder, HashEmbedder, Message, Price, ProviderDescriptor, ProviderError, Tokenizer, Usage,
};
use keep_store::{InMemoryVectorIndex, VectorCapabilities, VectorIndex};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use crate::ablation::{ArmKind, FactOptions, KeepPath};
use crate::artifacts::{
    CallRecord, ContextRecord, JudgeOutcome, Outcome, PromptRecord, QuestionRecord, create_run_dir,
    write_json, write_jsonl,
};
use crate::baselines::{Baselines, Built, DenseBackend, OracleEvidence};
use crate::dataset::annotations::GoldDecision;
use crate::isolation::RunStorage;
use crate::knowledge_arm::{KnowledgeArms, KnowledgeStorage};
use crate::manifest::{
    EmbeddingProvider, LoadedManifest, ModelSpec, Phase, PromptStorage, ProviderKind,
    StorageProfile,
};
use crate::prepare::{Prepared, prepare_scoped};
use crate::{metrics, mocks, report};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    /// Answer and judge with the configured providers.
    Execute,
    /// Build every context and count tokens; call no model and no database.
    Estimate,
    /// Build every context with the manifest's own embedder, reranker and
    /// storage and score evidence recall; call no answer or judge model and
    /// no remote embedder.
    Contexts,
}

/// Which arms and questions a run covers; `None` means all of them. Only
/// development questions can be selected: the holdout stays sealed.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    pub arms: Option<std::collections::BTreeSet<String>>,
    pub questions: Option<std::collections::BTreeSet<String>>,
    /// Allow a protocol with `split = "full"` to score the sealed holdout.
    /// Dry runs (estimate mode) never need it: they call no model.
    pub consume_holdout: bool,
    /// Keep each built context's text in [`RunOutput::context_texts`]
    /// (contexts mode), for `keep-bench export`.
    pub capture_context_text: bool,
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub mode: Mode,
    pub unseal: bool,
    pub allow_model_calls: bool,
    /// Parent directory for run directories. `None` writes nothing.
    pub results_root: Option<PathBuf>,
    /// Cross-run spend cap for USD-metered runs. `None` skips it (tests and
    /// dry runs); the CLI always sets it.
    pub spend: Option<crate::spend::SpendLedger>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitSummary {
    pub fit_fraction: f64,
    pub gate_eligible: bool,
    pub tokenizer: String,
    pub tokenizer_exact: bool,
    /// The mask's decisions checked against the answer provider's reported
    /// prompt counts after the run (executed runs with full-context arms).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reported_check: Option<crate::fitmask::FitVerification>,
}

impl FitSummary {
    /// Whether the fit decisions rest on exact counts: an exact local
    /// tokenizer, or a passing check against provider-reported counts.
    pub fn exact(&self) -> bool {
        self.tokenizer_exact || self.reported_check.as_ref().is_some_and(|v| v.verified)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunMeta {
    pub schema: String,
    pub run_id: String,
    pub harness_version: String,
    pub mode: Mode,
    pub manifest_id: String,
    pub manifest_hash: ContentHash,
    pub manifest_file_hash: ContentHash,
    pub phase: Phase,
    pub dataset: String,
    pub dataset_hash: ContentHash,
    pub gold_track: String,
    pub annotations: usize,
    pub development_histories: usize,
    pub holdout_histories: usize,
    pub histories_run: usize,
    pub questions_run: usize,
    pub subset_size: Option<usize>,
    pub seeds: Vec<u64>,
    pub storage_profile: StorageProfile,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector_backend: Option<VectorCapabilities>,
    pub answer_provider: Option<ProviderDescriptor>,
    pub judge_provider: Option<ProviderDescriptor>,
    pub embedding_provider: Option<ProviderDescriptor>,
    pub tokenizer: String,
    pub tokenizer_exact: bool,
    /// True when any provider used is an offline mock: the numbers are a
    /// plumbing check, not a measurement.
    pub mock: bool,
    pub fit_mask: Option<FitSummary>,
    pub embedding_usage: Usage,
    pub budget_usage: Option<keep_providers::budget::BudgetUsage>,
    /// AI credits charged against `budget.max_ai_credits` (Copilot CLI runs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credit_usage: Option<CreditUsage>,
    /// Set on every run that cannot stand behind a reported result; the
    /// comparison and gate refuse such runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_canonical: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<usize>,
    /// Where the answer model was reached (host, deployment, pins); never
    /// the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_endpoint: Option<EndpointRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_endpoint: Option<EndpointRecord>,
    /// Set when a fatal provider error (such as a served-model mismatch)
    /// stopped the run. Aborted runs cannot enter a comparison or gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aborted: Option<String>,
    /// Session fact extraction for the k3 and mk3 arms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extraction: Option<crate::facts::ExtractStats>,
    /// Session summaries for fact arms with `summaries`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summaries: Option<crate::facts::ExtractStats>,
    pub started_at: String,
    pub finished_at: String,
    pub warnings: Vec<String>,
    /// The scoring protocol, when it is not Keep's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<crate::protocol::ProtocolRecord>,
    /// Client-side TPM/RPM pacing and the HTTP 429s it met.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<keep_providers::ratelimit::RateLimitStats>,
}

/// An endpoint as configured, plus what its replies reported.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EndpointRecord {
    #[serde(flatten)]
    pub endpoint: EndpointInfo,
    pub served_models: std::collections::BTreeSet<String>,
    pub system_fingerprints: std::collections::BTreeSet<String>,
}

impl EndpointRecord {
    fn observe<'a>(endpoint: EndpointInfo, calls: impl Iterator<Item = &'a CallRecord>) -> Self {
        let mut r = Self {
            endpoint,
            served_models: Default::default(),
            system_fingerprints: Default::default(),
        };
        for c in calls {
            r.served_models.extend(c.served_model.clone());
            r.system_fingerprints.extend(c.system_fingerprint.clone());
        }
        r
    }
}

pub struct RunOutput {
    pub meta: RunMeta,
    pub records: Vec<QuestionRecord>,
    pub dir: Option<PathBuf>,
    /// Built context text per (arm, question id), when
    /// [`Selection::capture_context_text`] is set.
    pub context_texts: Vec<ContextText>,
}

/// One built context, as the answer model would have seen it.
#[derive(Clone, Debug, PartialEq)]
pub struct ContextText {
    pub arm: String,
    pub question_id: String,
    pub text: String,
}

struct Providers {
    answer: Option<Arc<dyn ChatModel>>,
    judge: Option<Arc<dyn ChatModel>>,
    /// The session fact extraction model; `None` serves cached replies only.
    extract: Option<Arc<dyn ChatModel>>,
    embedder: Option<Arc<dyn Embedder>>,
    ledger: Option<Arc<BudgetLedger>>,
    credits: Option<Arc<CreditLedger>>,
    answer_endpoint: Option<EndpointInfo>,
    judge_endpoint: Option<EndpointInfo>,
    /// TPM/RPM pacing shared by every `openai-compatible` chat model.
    limiter: Option<Arc<RateLimiter>>,
}

fn copilot_config(m: &ModelSpec) -> anyhow::Result<CopilotConfig> {
    let spec = m
        .copilot
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("copilot-cli model {} has no [copilot] table", m.model))?;
    let mut c = CopilotConfig::new(&m.model);
    c.reasoning_effort = spec.reasoning_effort.clone();
    c.timeout = Duration::from_secs(m.timeout_secs);
    c.rates = CreditRates {
        input_per_1k: spec.credits_per_1k_input,
        output_per_1k: spec.credits_per_1k_output,
        overhead_tokens: spec.overhead_tokens,
    };
    Ok(c)
}

#[cfg(feature = "local-embeddings")]
async fn local_embedder(lm: &LoadedManifest) -> anyhow::Result<Arc<dyn Embedder>> {
    use keep_providers::local_embed::{LocalEmbedConfig, LocalEmbedder, LocalPooling, MODEL_FILES};
    let e = &lm.manifest.embedding;
    let local = e
        .local
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("local-onnx embedding needs [embedding.local]"))?;
    let sha256 = MODEL_FILES
        .iter()
        .map(|f| {
            local
                .sha256
                .get(*f)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("embedding.local.sha256 has no digest for {f}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let cache_dir = std::env::var_os("KEEP_MODEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".keep-models"));
    let config = LocalEmbedConfig {
        repo: e.model.clone(),
        revision: e.revision.clone(),
        sha256: sha256.clone(),
        dimensions: e.dimensions,
        pooling: if local.pooling == "mean" {
            LocalPooling::Mean
        } else {
            LocalPooling::Cls
        },
        max_length: local.max_length,
        cache_dir: cache_dir.clone(),
        device: Default::default(),
        intra_threads: None,
        sessions: None,
    }
    .with_env_overrides()?;
    // Everything besides the descriptor that changes a vector.
    let extra = format!(
        "pooling={};max_length={};files={};device={:?}",
        local.pooling,
        local.max_length,
        sha256.join(","),
        config.device
    );
    let embedder: Arc<dyn Embedder> = Arc::new(LocalEmbedder::load(config).await?);
    let vectors_dir = match std::env::var_os("KEEP_EMBED_CACHE") {
        Some(v) if v == "off" => None,
        Some(v) => Some(PathBuf::from(v)),
        None => Some(cache_dir.join("embeddings")),
    };
    Ok(Arc::new(keep_providers::CachedEmbedder::new(
        embedder,
        &extra,
        vectors_dir.as_deref(),
    )?))
}

#[cfg(not(feature = "local-embeddings"))]
async fn local_embedder(_: &LoadedManifest) -> anyhow::Result<Arc<dyn Embedder>> {
    anyhow::bail!(
        "the local-onnx embedding provider needs a build with `--features local-embeddings`"
    )
}

/// The manifest's cross-encoder, downloaded and verified on first use and
/// memoised so arms sharing a question score each passage once.
#[cfg(feature = "local-embeddings")]
async fn local_reranker(
    spec: &crate::manifest::RerankerSpec,
) -> anyhow::Result<Arc<dyn keep_providers::Reranker>> {
    use keep_providers::local_embed::MODEL_FILES;
    use keep_providers::local_rerank::{LocalRerankConfig, LocalReranker};
    let sha256 = MODEL_FILES
        .iter()
        .map(|f| {
            spec.sha256
                .get(*f)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("reranker.sha256 has no digest for {f}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let cache_dir = std::env::var_os("KEEP_MODEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(".keep-models"));
    let config = LocalRerankConfig {
        repo: spec.model.clone(),
        revision: spec.revision.clone(),
        sha256,
        max_length: spec.max_length,
        ..LocalRerankConfig::ms_marco_minilm_l6_v2(cache_dir)
    }
    .with_env_overrides()?;
    let inner: Arc<dyn keep_providers::Reranker> = Arc::new(LocalReranker::load(config).await?);
    Ok(Arc::new(keep_providers::MemoReranker::new(inner, 1 << 20)))
}

#[cfg(not(feature = "local-embeddings"))]
async fn local_reranker(
    _: &crate::manifest::RerankerSpec,
) -> anyhow::Result<Arc<dyn keep_providers::Reranker>> {
    anyhow::bail!("the local-onnx reranker needs a build with `--features local-embeddings`")
}

fn openai_config(m: &ModelSpec) -> Result<OpenAiConfig, ProviderError> {
    let mut c = OpenAiConfig::from_env(
        m.base_url_env.as_deref().unwrap_or_default(),
        m.api_key_env.as_deref(),
        &m.model,
        &m.revision,
        m.price(),
        Duration::from_secs(m.timeout_secs),
    )?;
    if let Some(name) = &m.api_key_env
        && m.api_style.is_azure()
        && c.api_key.as_deref().is_none_or(str::is_empty)
    {
        return Err(ProviderError::Config {
            message: format!("environment variable {name} is not set"),
        });
    }
    c.api_style = m.api_style;
    c.auth_header = m.auth_header;
    c.api_version = m.api_version.clone();
    c.served_model = m.served_model.clone();
    c.reasoning_effort = m.reasoning_effort.clone();
    if let Some(name) = &m.deployment_env {
        c.deployment = Some(
            std::env::var(name)
                .ok()
                .filter(|v| !v.trim().is_empty())
                .ok_or_else(|| ProviderError::Config {
                    message: format!("environment variable {name} is not set"),
                })?,
        );
    }
    Ok(c)
}

async fn providers(
    lm: &LoadedManifest,
    opts: &RunOptions,
    run_id: &str,
    needs_dense: bool,
) -> anyhow::Result<Providers> {
    let m = &lm.manifest;
    if opts.mode == Mode::Estimate {
        let embedder: Option<Arc<dyn Embedder>> = needs_dense
            .then(|| Arc::new(HashEmbedder::new(m.embedding.dimensions)) as Arc<dyn Embedder>);
        return Ok(Providers {
            answer: None,
            judge: None,
            extract: None,
            embedder,
            ledger: None,
            credits: None,
            answer_endpoint: None,
            judge_endpoint: None,
            limiter: None,
        });
    }
    if opts.mode == Mode::Contexts {
        let embedder: Option<Arc<dyn Embedder>> = match (needs_dense, m.embedding.provider) {
            (false, _) => None,
            (true, EmbeddingProvider::Hash) => {
                Some(Arc::new(HashEmbedder::new(m.embedding.dimensions)))
            }
            (true, EmbeddingProvider::LocalOnnx) => Some(local_embedder(lm).await?),
            (true, EmbeddingProvider::OpenaiCompatible) => anyhow::bail!(
                "contexts runs make no remote calls; this manifest embeds through a remote API"
            ),
        };
        return Ok(Providers {
            answer: None,
            judge: None,
            extract: None,
            embedder,
            ledger: None,
            credits: None,
            answer_endpoint: None,
            judge_endpoint: None,
            limiter: None,
        });
    }
    let mut ledger = None;
    let mut credits = None;
    let permit = if m.uses_remote_models() {
        let policy = DispatchPolicy {
            manifest_allows: m.dispatch.allow_model_calls,
            operator_allows: opts.allow_model_calls,
        };
        let permit = policy.permit(run_id, m.dispatch_problems())?;
        let mut caps = m.budget_caps();
        if let Some(spend) = &opts.spend {
            caps.max_cost_usd = spend.run_cap(caps.max_cost_usd);
        }
        ledger = Some(Arc::new(BudgetLedger::new(caps)));
        if m.uses_copilot() {
            credits = Some(Arc::new(CreditLedger::new(
                m.budget.max_ai_credits.unwrap_or(0.0),
            )));
        }
        Some(permit)
    } else {
        None
    };
    let limiter = match m.dispatch.pacing()? {
        (None, None) => None,
        (tpm, rpm) => Some(Arc::new(RateLimiter::new(
            tpm.unwrap_or(0),
            rpm.unwrap_or(0),
        ))),
    };
    // One gate bounds every Copilot CLI process in the run, answer and judge.
    let gate = Arc::new(Semaphore::new(m.dispatch_concurrency()?));
    type Chat = (Arc<dyn ChatModel>, Option<EndpointInfo>);
    let chat =
        |spec: &ModelSpec, mock: fn() -> keep_providers::mock::FnChat| -> anyhow::Result<Chat> {
            Ok(match spec.provider {
                ProviderKind::Mock => (Arc::new(mock()), None),
                ProviderKind::OpenaiCompatible => {
                    let mut config = openai_config(spec)?;
                    config.limiter = limiter.clone();
                    let client = OpenAiCompatibleChat::new(
                        config,
                        permit.as_ref().expect("permit checked above"),
                        ledger.clone().expect("ledger exists with permit"),
                    )?;
                    let endpoint = client.endpoint();
                    (Arc::new(client), Some(endpoint))
                }
                ProviderKind::CopilotCli => (
                    Arc::new(CopilotCliChat::new(
                        copilot_config(spec)?,
                        permit.as_ref().expect("permit checked above"),
                        credits
                            .clone()
                            .expect("credit ledger exists for copilot runs"),
                        gate.clone(),
                    )?),
                    None,
                ),
            })
        };
    let (answer, answer_endpoint) = chat(&m.models.answer, mocks::answerer)?;
    let (judge, judge_endpoint) = chat(&m.models.judge, mocks::judge)?;
    let extract = match m.extract_model() {
        Some(spec) => Some(chat(spec, mocks::extractor)?.0),
        None => None,
    };
    let embedder: Option<Arc<dyn Embedder>> = if !needs_dense {
        None
    } else {
        Some(match m.embedding.provider {
            EmbeddingProvider::Hash => Arc::new(HashEmbedder::new(m.embedding.dimensions)),
            EmbeddingProvider::OpenaiCompatible => {
                let e = &m.embedding;
                let config = OpenAiConfig::from_env(
                    e.base_url_env.as_deref().unwrap_or_default(),
                    e.api_key_env.as_deref(),
                    &e.model,
                    &e.revision,
                    Price {
                        input_per_mtok_usd: e.input_usd_per_mtok,
                        output_per_mtok_usd: 0.0,
                    },
                    Duration::from_secs(120),
                )?;
                Arc::new(OpenAiCompatibleEmbedder::new(
                    config,
                    e.dimensions,
                    permit.as_ref().expect("permit checked above"),
                    ledger.clone().expect("ledger exists with permit"),
                )?)
            }
            EmbeddingProvider::LocalOnnx => local_embedder(lm).await?,
        })
    };
    Ok(Providers {
        answer: Some(answer),
        judge: Some(judge),
        extract,
        embedder,
        ledger,
        credits,
        answer_endpoint,
        judge_endpoint,
        limiter,
    })
}

/// The manifest's judge, for calibration. Remote judges need the same
/// manifest and operator permission as a run; `spend` only lowers the
/// budget's cost cap (see [`crate::spend::SpendLedger::run_cap`]).
pub async fn judge_model(
    lm: &LoadedManifest,
    allow_model_calls: bool,
    spend: Option<crate::spend::SpendLedger>,
) -> anyhow::Result<(Arc<dyn ChatModel>, Option<Arc<BudgetLedger>>)> {
    let opts = RunOptions {
        mode: Mode::Execute,
        unseal: false,
        allow_model_calls,
        results_root: None,
        spend,
    };
    let p = providers(
        lm,
        &opts,
        &format!("calibrate-{}", lm.manifest.hash().short()),
        false,
    )
    .await?;
    Ok((p.judge.expect("execute mode builds a judge"), p.ledger))
}

async fn vector_index(st: &RunStorage) -> anyhow::Result<Arc<dyn VectorIndex>> {
    Ok(match (st.profile, &st.postgres_url, &st.qdrant_url) {
        (StorageProfile::PostgresVector, Some(url), _) => {
            Arc::new(keep_postgres::PgVectorIndex::connect(url).await?)
        }
        (StorageProfile::PostgresQdrant, _, Some(url)) => Arc::new(keep_qdrant::QdrantIndex::new(
            url,
            st.qdrant_api_key.clone(),
        )),
        _ => Arc::new(InMemoryVectorIndex::new()),
    })
}

fn knowledge_storage(st: &RunStorage) -> KnowledgeStorage {
    match (st.profile, &st.postgres_url, &st.qdrant_url) {
        (StorageProfile::PostgresVector, Some(url), _) => {
            KnowledgeStorage::PostgresVector { url: url.clone() }
        }
        (StorageProfile::PostgresQdrant, Some(pg), Some(q)) => KnowledgeStorage::PostgresQdrant {
            postgres_url: pg.clone(),
            qdrant_url: q.clone(),
            api_key: st.qdrant_api_key.clone(),
        },
        _ => KnowledgeStorage::InMemory,
    }
}

/// Every full-context answer call in `records`, for the post-run fit check.
fn sent_full_context(records: &[QuestionRecord]) -> Vec<crate::fitmask::SentPrompt> {
    records
        .iter()
        .filter(|r| r.arm_kind == ArmKind::FullContext.label())
        .filter_map(|r| {
            let call = r.answer.as_ref()?;
            Some(crate::fitmask::SentPrompt {
                question_id: r.question_id.clone(),
                estimated_tokens: r.prompt.as_ref().map_or(0, |p| p.input_tokens),
                reported_tokens: call
                    .usage
                    .provider_reported
                    .then_some(call.usage.input_tokens),
                failed: call.text.is_none(),
            })
        })
        .collect()
}

/// Call a model, retrying only retryable errors.
async fn call(
    model: &dyn ChatModel,
    req: &ChatRequest,
    max_attempts: u32,
    price: Price,
) -> (Result<ChatResponse, ProviderError>, CallRecord) {
    let started = Instant::now();
    let mut errors = Vec::new();
    let mut attempts = 0;
    let result = loop {
        attempts += 1;
        match model.complete(req).await {
            Ok(r) => break Ok(r),
            Err(e) => {
                errors.push(e.to_string());
                if !e.retryable() || attempts >= max_attempts.max(1) {
                    break Err(e);
                }
                tokio::time::sleep(e.retry_delay(attempts)).await;
            }
        }
    };
    let (text, usage, finish) = match &result {
        Ok(r) => (Some(r.text.clone()), r.usage, r.finish_reason.clone()),
        Err(_) => (None, Usage::default(), None),
    };
    let record = CallRecord {
        provider: model.descriptor().clone(),
        text,
        usage,
        latency_ms: started.elapsed().as_millis() as u64,
        attempts,
        cost_usd: price.cost(usage.input_tokens, usage.output_tokens),
        finish_reason: finish,
        errors,
        billing: result.as_ref().ok().and_then(|r| r.billing.clone()),
        served_model: result.as_ref().ok().and_then(|r| r.served_model.clone()),
        system_fingerprint: result
            .as_ref()
            .ok()
            .and_then(|r| r.system_fingerprint.clone()),
    };
    (result, record)
}

/// One answer-and-judge unit, prepared while contexts were built.
struct Job {
    index: usize,
    messages: Vec<Message>,
    input_tokens: u64,
    seed: u64,
    gold: String,
    question: String,
    question_id: String,
    category: String,
    abstention: bool,
    evidence: Vec<String>,
    included: Vec<String>,
    protocol: crate::protocol::ProtocolName,
    answer_prompt: String,
}

struct JobContext {
    answer: Arc<dyn ChatModel>,
    judge: Arc<dyn ChatModel>,
    answer_spec: ModelSpec,
    judge_spec: ModelSpec,
    max_attempts: u32,
    tokenizer: Arc<dyn Tokenizer>,
    consistency: Option<(
        crate::judge::JudgeConsistencySpec,
        crate::judge::VerdictCache,
    )>,
}

struct JobResult {
    /// A fatal provider error that must stop the run.
    fatal: Option<String>,
    outcome: Outcome,
    answer: CallRecord,
    metrics: Option<metrics::AnswerMetrics>,
    judge: JudgeOutcome,
    judge_call: Option<CallRecord>,
    judge_consistency: Option<crate::judge::JudgeConsistencyRecord>,
    final_answer: Option<String>,
}

async fn execute_job(ctx: &JobContext, mut job: Job) -> JobResult {
    let req = ChatRequest {
        messages: std::mem::take(&mut job.messages),
        max_output_tokens: ctx.answer_spec.max_output_tokens,
        temperature: ctx.answer_spec.temperature,
        seed: Some(job.seed),
        estimated_input_tokens: job.input_tokens,
    };
    let (result, answer) = call(
        ctx.answer.as_ref(),
        &req,
        ctx.max_attempts,
        ctx.answer_spec.price(),
    )
    .await;
    let text = match result {
        Ok(r) => r.text,
        Err(e) => {
            return JobResult {
                fatal: e.fatal().then(|| format!("answer model: {e}")),
                outcome: Outcome::Failed {
                    error: e.to_string(),
                },
                answer,
                metrics: None,
                judge: JudgeOutcome::Skipped,
                judge_call: None,
                judge_consistency: None,
                final_answer: None,
            };
        }
    };
    let raw = text;
    let text = crate::prompt::final_answer(&job.answer_prompt, job.protocol.extract_answer(&raw));
    // Store the scored answer only when it is not the reply as it stands.
    let final_answer = crate::prompt::has_working_notes(&job.answer_prompt).then(|| text.clone());
    let metrics = metrics::score(
        &text,
        &job.gold,
        job.abstention,
        &job.evidence,
        &job.included,
    );
    let jm = job.protocol.judge_messages(
        &crate::protocol::JudgeInput {
            question_id: &job.question_id,
            question: &job.question,
            category: &job.category,
            gold: &job.gold,
            abstention: job.abstention,
        },
        &text,
    );
    let jreq = ChatRequest {
        estimated_input_tokens: jm
            .iter()
            .map(|x| ctx.tokenizer.count(&x.content) as u64)
            .sum(),
        messages: jm,
        max_output_tokens: ctx.judge_spec.max_output_tokens,
        temperature: ctx.judge_spec.temperature,
        seed: Some(job.seed),
    };
    let Some((spec, cache)) = &ctx.consistency else {
        let j = judge_votes(ctx, &job, &jreq, 1).await;
        return JobResult {
            fatal: j.fatal,
            outcome: Outcome::Answered,
            answer,
            metrics: Some(metrics),
            judge: j.judge,
            judge_call: j.first_call,
            judge_consistency: None,
            final_answer,
        };
    };
    let votes = if spec.votes_for(metrics.abstained, metrics.f1) {
        spec.votes
    } else {
        1
    };
    let key_input = crate::judge::KeyInput {
        judge_provider: serde_json::to_value(ctx.judge_spec.provider).expect("provider serialises"),
        judge_model: &ctx.judge_spec.model,
        judge_revision: &ctx.judge_spec.revision,
        judge_prompt: &ctx.judge_spec.prompt_version,
        protocol: job.protocol.as_str(),
        votes: spec.votes,
        vote_on: spec.vote_on,
        question_id: &job.question_id,
        question: &job.question,
        category: &job.category,
        gold: &job.gold,
        abstention: job.abstention,
        answer: crate::judge::normalize_answer(&text),
    };
    let key = key_input.key();
    let judge_once = || judge_votes(ctx, &job, &jreq, votes);
    let (j, source) = if spec.cache {
        cache.judge(&key, judge_once).await
    } else {
        (judge_once().await, crate::judge::VerdictSource::Judged)
    };
    JobResult {
        fatal: j.fatal,
        outcome: Outcome::Answered,
        answer,
        metrics: Some(metrics),
        judge: j.judge,
        judge_call: j.first_call,
        judge_consistency: Some(crate::judge::JudgeConsistencyRecord {
            key,
            source,
            votes: j.votes,
            extra_calls: j.extra_calls,
        }),
        final_answer,
    }
}

/// Judge one answer with up to `target` votes: a second vote, and a third
/// only when the first two do not agree. With one vote this is the plain
/// single judge call every run made before judge consistency existed.
async fn judge_votes(
    ctx: &JobContext,
    job: &Job,
    jreq: &ChatRequest,
    target: u32,
) -> crate::judge::Judgement {
    let mut first_call = None;
    let mut extra_calls = Vec::new();
    let mut votes: Vec<Option<bool>> = Vec::new();
    let mut first_judge = None;
    while votes.is_empty() || crate::judge::needs_another_vote(&votes, target) {
        let mut req = jreq.clone();
        req.seed = Some(job.seed.wrapping_add(votes.len() as u64));
        let (jresult, jrecord) = call(
            ctx.judge.as_ref(),
            &req,
            ctx.max_attempts,
            ctx.judge_spec.price(),
        )
        .await;
        if votes.is_empty() {
            first_call = Some(jrecord);
        } else {
            extra_calls.push(jrecord);
        }
        let (judge, vote) = match jresult {
            Ok(r) => match job.protocol.parse_verdict(&r.text) {
                crate::protocol::Verdict::Correct(c) => {
                    (JudgeOutcome::Judged { correct: c }, Some(c))
                }
                crate::protocol::Verdict::Malformed => {
                    (JudgeOutcome::Malformed { reply: r.text }, None)
                }
            },
            Err(e) => {
                // A fatal error (credits, auth) stops voting and the run.
                if e.fatal() {
                    return crate::judge::Judgement {
                        judge: JudgeOutcome::Failed {
                            error: e.to_string(),
                        },
                        first_call,
                        extra_calls,
                        votes,
                        fatal: Some(format!("judge model: {e}")),
                    };
                }
                (
                    JudgeOutcome::Failed {
                        error: e.to_string(),
                    },
                    None,
                )
            }
        };
        votes.push(vote);
        if first_judge.is_none() {
            first_judge = Some(judge);
        }
    }
    let judge = if votes.len() == 1 {
        first_judge.expect("one vote was cast")
    } else {
        match crate::judge::majority(&votes) {
            Some(correct) => JudgeOutcome::Judged { correct },
            None => JudgeOutcome::Malformed {
                reply: format!("no majority among judge votes {votes:?}"),
            },
        }
    };
    crate::judge::Judgement {
        judge,
        first_call,
        extra_calls,
        votes,
        fatal: None,
    }
}

fn run_id(lm: &LoadedManifest, mode: Mode) -> String {
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ");
    let suffix = match mode {
        Mode::Estimate => "-dry",
        Mode::Contexts => "-contexts",
        Mode::Execute => "",
    };
    format!("{stamp}-{}{suffix}", lm.hash.short())
}

/// Fill the fact extraction cache for the development histories (or the
/// selected questions' histories) without building any context or calling
/// the answer and judge models. Runs, contexts and dry runs then read the
/// cache. Dispatch needs the same permission as `run`.
pub async fn extract(
    lm: &LoadedManifest,
    opts: &RunOptions,
    selection: &Selection,
) -> anyhow::Result<ExtractReport> {
    let m = &lm.manifest;
    let spec = m.extract_model().ok_or_else(|| {
        anyhow::anyhow!("this manifest has no k3 or mk3 arm to extract facts for")
    })?;
    anyhow::ensure!(!opts.unseal, "extraction reads development histories only");
    let run_id = format!("{}-extract", run_id(lm, Mode::Execute));
    let providers = providers(lm, opts, &run_id, false).await?;
    let _spend = match (&opts.spend, &providers.ledger) {
        (Some(spend), Some(ledger)) if m.uses_openai_compatible() => Some(spend.guard(
            &run_id,
            "extract",
            &m.id,
            &lm.hash.to_string(),
            ledger.caps().max_cost_usd,
            ledger.clone(),
        )?),
        _ => None,
    };
    let Prepared { dataset, .. } = prepare_scoped(lm, false, selection.consume_holdout)?;
    let mut seen = std::collections::BTreeSet::new();
    let mut sessions = Vec::new();
    for q in &dataset.questions {
        if selection
            .questions
            .as_ref()
            .is_some_and(|ids| !ids.contains(&q.id))
        {
            continue;
        }
        if !seen.insert(q.history_id.clone()) {
            continue;
        }
        let h = dataset
            .history(&q.history_id)
            .expect("loader checked history ids");
        sessions.extend(h.sessions.iter().map(crate::facts::session_input));
    }
    if let Some(ids) = &selection.questions {
        let found = dataset
            .questions
            .iter()
            .filter(|q| ids.contains(&q.id))
            .count();
        anyhow::ensure!(
            found == ids.len(),
            "{} selected questions are not in the development split",
            ids.len() - found
        );
    }
    let concurrency = m.dispatch_concurrency()?;
    eprintln!(
        "[{}] extracting facts from {} histories ({} sessions) with {}, {concurrency} at a time",
        m.id,
        seen.len(),
        sessions.len(),
        crate::facts::model_id(spec)
    );
    let extractor = Arc::new(crate::facts::extractor(spec, providers.extract.clone()));
    let wants_summaries = m.extraction.as_ref().is_some_and(|x| x.summary_tokens > 0);
    let facts = crate::facts::warm_cache(
        extractor,
        sessions.clone(),
        concurrency,
        m.budget.max_attempts,
    )
    .await;
    let summaries = if wants_summaries && facts.fatal.is_none() {
        let summarizer = Arc::new(crate::facts::summarizer(spec, providers.extract.clone()));
        Some(
            crate::facts::warm_cache(summarizer, sessions, concurrency, m.budget.max_attempts)
                .await,
        )
    } else {
        None
    };
    Ok(ExtractReport { facts, summaries })
}

/// What `keep-bench extract` filled: fact extractions, and session
/// summaries when the manifest's arms read them.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ExtractReport {
    #[serde(flatten)]
    pub facts: crate::facts::ExtractStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summaries: Option<crate::facts::ExtractStats>,
}

impl ExtractReport {
    pub fn failed(&self) -> usize {
        self.facts.failed + self.summaries.as_ref().map_or(0, |s| s.failed)
    }
}

pub async fn run(lm: &LoadedManifest, opts: &RunOptions) -> anyhow::Result<RunOutput> {
    run_selected(lm, opts, &Selection::default()).await
}

/// [`run`] over a subset of the manifest's arms and development questions.
pub async fn run_selected(
    lm: &LoadedManifest,
    opts: &RunOptions,
    selection: &Selection,
) -> anyhow::Result<RunOutput> {
    let started_at = chrono::Utc::now().to_rfc3339();
    let m = &lm.manifest;
    let run_id = run_id(lm, opts.mode);
    if let Some(names) = &selection.arms {
        let unknown: Vec<&String> = names
            .iter()
            .filter(|n| !m.arms.iter().any(|a| &a.name == *n))
            .collect();
        anyhow::ensure!(
            unknown.is_empty(),
            "no such arms in the manifest: {unknown:?}"
        );
    }
    let arms: Vec<&crate::ablation::ArmSpec> = m
        .arms
        .iter()
        .filter(|a| selection.arms.as_ref().is_none_or(|n| n.contains(&a.name)))
        .collect();
    let needs_dense = arms.iter().any(|a| matches!(a.kind, ArmKind::Dense { .. }));
    let needs_knowledge = arms.iter().any(|a| {
        matches!(&a.kind, ArmKind::Keep { path, .. } if a.kind.unavailable_reason().is_none() && path.uses_knowledge())
    });

    // Refuse before touching any data if dispatch is not authorised.
    let providers = providers(lm, opts, &run_id, needs_dense || needs_knowledge).await?;
    // Reserve the whole run budget against the cross-run cap before any
    // call; the guard settles to the charged cost however the run ends.
    let _spend = match (&opts.spend, &providers.ledger) {
        (Some(spend), Some(ledger)) if m.uses_openai_compatible() => Some(spend.guard(
            &run_id,
            "run",
            &m.id,
            &lm.hash.to_string(),
            ledger.caps().max_cost_usd,
            ledger.clone(),
        )?),
        _ => None,
    };
    let non_canonical = m.non_canonical_label().map(str::to_owned);
    let Prepared {
        mut dataset,
        annotations,
        split,
        fit_mask,
        subset_size,
    } = prepare_scoped(
        lm,
        opts.unseal,
        selection.consume_holdout || opts.mode == Mode::Estimate,
    )?;
    let protocol = m.protocol.name;
    let mut context_texts: Vec<ContextText> = Vec::new();
    if let Some(ids) = &selection.questions {
        let missing = ids
            .iter()
            .filter(|id| !dataset.questions.iter().any(|q| &q.id == *id))
            .count();
        anyhow::ensure!(
            missing == 0,
            "{missing} selected questions are not in this run's split"
        );
        dataset.questions.retain(|q| ids.contains(&q.id));
    }

    let (tokenizer, tokenizer_warning) = m.answer_tokenizer();
    let mut warnings = dataset.warnings.clone();
    warnings.extend(tokenizer_warning);
    if !tokenizer.exact() {
        warnings.push(format!(
            "token counts use the {} estimator, not the answer model's tokenizer; fit and budget decisions are provisional",
            tokenizer.id()
        ));
    }
    let storage = if needs_dense || needs_knowledge {
        let st = RunStorage::open(lm, &run_id, opts.mode).await?;
        if let Some(schema) = st.schema() {
            eprintln!("[{}] storing this run in Postgres schema {schema}", m.id);
        }
        Some(st)
    } else {
        None
    };
    let mut vector_caps = None;
    let dense = if let Some(embedder) = providers.embedder.clone().filter(|_| needs_dense) {
        let st = storage.as_ref().expect("opened for dense arms");
        let index = vector_index(st).await?;
        vector_caps = Some(index.capabilities());
        if opts.mode == Mode::Estimate && m.embedding.provider != EmbeddingProvider::Hash {
            warnings.push("dry run ranks dense chunks with the hash embedder; context sizes stay within the same budgets".into());
        }
        Some(DenseBackend::new(
            embedder,
            index,
            st.dense_collection.clone(),
            m.embedding.request_batch(),
        ))
    } else {
        None
    };
    let mut baselines = Baselines::new(
        tokenizer.clone(),
        m.models.answer.clone(),
        m.retrieval.clone(),
        dense,
    );

    let mut knowledge = match providers.embedder.clone().filter(|_| needs_knowledge) {
        Some(embedder) => {
            let st = storage.as_ref().expect("opened for knowledge arms");
            let reranker = match &m.reranker {
                Some(_) if opts.mode == Mode::Estimate => {
                    warnings.push(
                        "dry run ranks k2 and mk2 without the reranker; context sizes stay within the same budgets".into(),
                    );
                    None
                }
                Some(spec) => Some((local_reranker(spec).await?, spec.top_n)),
                None => None,
            };
            let k = KnowledgeArms::new(
                knowledge_storage(st),
                embedder,
                tokenizer.clone(),
                m.retrieval.clone(),
                lm.hash.short(),
                st.knowledge_collection.clone(),
                reranker,
            )
            .await?;
            if vector_caps.is_none() {
                vector_caps = k.vectors_capabilities();
            }
            Some(k)
        }
        None => None,
    };

    let mut aborted: Option<String> = None;
    let mut extraction = None;
    let mut summaries = None;
    // Ingest each scored history once per backend, several at a time,
    // before any arm reads it. Histories are independent, so arms see the
    // same stores they would have built one question at a time.
    {
        let mut seen = std::collections::BTreeSet::new();
        let scored: Vec<&crate::dataset::History> = dataset
            .questions
            .iter()
            .filter(|q| {
                matches!(
                    annotations.decide(m.dataset.gold_track, &q.id, &q.answer),
                    GoldDecision::Score { .. }
                )
            })
            .filter(|q| seen.insert(q.history_id.clone()))
            .map(|q| {
                dataset
                    .history(&q.history_id)
                    .expect("loader checked history ids")
            })
            .collect();
        let workers = m.dispatch_concurrency()?;
        let mut problems = Vec::new();
        let mut walls = Vec::new();
        if let Some(dense) = baselines.dense.as_mut() {
            let started = Instant::now();
            problems.extend(dense.prepare(&scored, m.retrieval.chunking, workers).await);
            walls.push(format!("dense {:.1}s", started.elapsed().as_secs_f64()));
        }
        if let Some(k) = knowledge.as_mut() {
            let started = Instant::now();
            problems.extend(k.prepare(&scored, workers).await);
            walls.push(format!("knowledge {:.1}s", started.elapsed().as_secs_f64()));
            let fact_arms = arms.iter().any(|a| {
                matches!(&a.kind, ArmKind::Keep { path, .. } if a.kind.unavailable_reason().is_none() && path.uses_extracted_facts())
            });
            if let (true, Some(spec), Some(x)) = (fact_arms, m.extract_model(), &m.extraction) {
                let started = Instant::now();
                let extractor = Arc::new(crate::facts::extractor(spec, providers.extract.clone()));
                let sessions = scored
                    .iter()
                    .flat_map(|h| h.sessions.iter().map(crate::facts::session_input))
                    .collect();
                let stats = crate::facts::warm_cache(
                    extractor.clone(),
                    sessions,
                    workers,
                    m.budget.max_attempts,
                )
                .await;
                warnings.extend(crate::facts::stats_warnings(&stats));
                if let Some(e) = &stats.fatal {
                    aborted.get_or_insert_with(|| e.clone());
                }
                extraction = Some(stats);
                k.enable_facts(extractor, x.facts_tokens, x.profile_tokens);
                if arms.iter().any(|a| a.summaries) && aborted.is_none() {
                    let summarizer =
                        Arc::new(crate::facts::summarizer(spec, providers.extract.clone()));
                    let sessions = scored
                        .iter()
                        .flat_map(|h| h.sessions.iter().map(crate::facts::session_input))
                        .collect();
                    let stats = crate::facts::warm_cache(
                        summarizer.clone(),
                        sessions,
                        workers,
                        m.budget.max_attempts,
                    )
                    .await;
                    warnings.extend(crate::facts::summary_warnings(&stats));
                    if let Some(e) = &stats.fatal {
                        aborted.get_or_insert_with(|| e.clone());
                    }
                    summaries = Some(stats);
                    k.enable_summaries(summarizer, x.summary_tokens);
                }
                problems.extend(k.prepare_facts(&scored, workers).await);
                walls.push(format!("facts {:.1}s", started.elapsed().as_secs_f64()));
            }
        }
        if opts.mode != Mode::Estimate && !walls.is_empty() {
            eprintln!(
                "[{}] ingested {} histories with {workers} workers: {}",
                m.id,
                scored.len(),
                walls.join(", ")
            );
            if keep_providers::timing::enabled() {
                eprint!(
                    "{}",
                    keep_providers::timing::render(&keep_providers::timing::take())
                );
            }
        }
        for p in problems {
            eprintln!("[{}] {p}; retrying when an arm needs it", m.id);
        }
    }

    let mut records = Vec::new();
    let mut jobs: Vec<Job> = Vec::new();
    for arm in arms.iter().copied() {
        if opts.mode != Mode::Estimate {
            eprintln!("[{}] building contexts for {}", m.id, arm.name);
        }
        let arm_started = Instant::now();
        for q in &dataset.questions {
            let history = dataset
                .history(&q.history_id)
                .expect("loader checked history ids");
            let base =
                |outcome: Outcome, gold: String, corrected: bool, seed: u64| QuestionRecord {
                    run_id: run_id.clone(),
                    non_canonical: non_canonical.clone(),
                    manifest_hash: lm.hash.clone(),
                    arm: arm.name.clone(),
                    arm_kind: arm.kind.label().into(),
                    diagnostic: arm.kind.is_diagnostic(),
                    seed,
                    question_id: q.id.clone(),
                    history_id: q.history_id.clone(),
                    category: q.category.clone(),
                    abstention: q.abstention,
                    gold,
                    gold_corrected: corrected,
                    fits_full_context: fit_mask.as_ref().and_then(|f| f.fits(&q.id)),
                    outcome,
                    context: None,
                    prompt: None,
                    answer: None,
                    judge: JudgeOutcome::Skipped,
                    judge_call: None,
                    judge_consistency: None,
                    final_answer: None,
                    metrics: None,
                };
            let (gold, corrected) = match annotations.decide(m.dataset.gold_track, &q.id, &q.answer)
            {
                GoldDecision::Excluded { reason } => {
                    for &seed in &m.seeds {
                        records.push(base(
                            Outcome::Excluded {
                                reason: reason.clone(),
                            },
                            q.answer.clone(),
                            false,
                            seed,
                        ));
                    }
                    continue;
                }
                GoldDecision::Score { answer, corrected } => (answer, corrected),
            };
            if let Some(reason) = arm.kind.unavailable_reason() {
                for &seed in &m.seeds {
                    records.push(base(
                        Outcome::NotRun {
                            reason: reason.clone(),
                        },
                        gold.clone(),
                        corrected,
                        seed,
                    ));
                }
                continue;
            }
            let oracle = matches!(arm.kind, ArmKind::OracleEvidence).then(|| OracleEvidence {
                turn_ids: &q.evidence_turn_ids,
                session_ids: &q.evidence_session_ids,
            });
            let t0 = Instant::now();
            let built = match (&arm.kind, knowledge.as_mut()) {
                (
                    ArmKind::Keep {
                        path: path @ (KeepPath::K0 | KeepPath::K1),
                        ..
                    },
                    Some(k),
                ) => k.build(*path, history, &q.query).await,
                (
                    ArmKind::Keep {
                        path: KeepPath::K2,
                        representation,
                    },
                    Some(k),
                ) => {
                    k.build_ranked(
                        KeepPath::K2,
                        *representation,
                        None,
                        history,
                        &q.query,
                        FactOptions::default(),
                    )
                    .await
                }
                (
                    ArmKind::Keep {
                        path: KeepPath::K3,
                        representation,
                    },
                    Some(k),
                ) => {
                    k.build_ranked(
                        KeepPath::K3,
                        *representation,
                        None,
                        history,
                        &q.query,
                        arm.fact_options(),
                    )
                    .await
                }
                (
                    ArmKind::Keep {
                        path: KeepPath::Mk3,
                        representation,
                    },
                    Some(k),
                ) => {
                    k.build_ranked(
                        KeepPath::Mk3,
                        *representation,
                        Some(&mut baselines.memory),
                        history,
                        &q.query,
                        arm.fact_options(),
                    )
                    .await
                }
                (
                    ArmKind::Keep {
                        path: KeepPath::Mk2,
                        representation,
                    },
                    Some(k),
                ) => {
                    k.build_ranked(
                        KeepPath::Mk2,
                        *representation,
                        Some(&mut baselines.memory),
                        history,
                        &q.query,
                        FactOptions::default(),
                    )
                    .await
                }
                (
                    ArmKind::Keep {
                        path: path @ (KeepPath::Mk0 | KeepPath::Mk1),
                        representation,
                    },
                    Some(k),
                ) => {
                    k.build_with_memory(
                        *path,
                        *representation,
                        &mut baselines.memory,
                        history,
                        &q.query,
                    )
                    .await
                }
                _ => baselines.build(&arm.kind, history, &q.query, oracle).await,
            };
            let build_ms = t0.elapsed().as_millis() as u64;
            let bundle = match built {
                Built::Ready(b) => b,
                Built::Unavailable(reason) => {
                    for &seed in &m.seeds {
                        records.push(base(
                            Outcome::Unavailable {
                                reason: reason.clone(),
                            },
                            gold.clone(),
                            corrected,
                            seed,
                        ));
                    }
                    continue;
                }
                Built::Failed(error) => {
                    for &seed in &m.seeds {
                        records.push(base(
                            Outcome::Failed {
                                error: error.clone(),
                            },
                            gold.clone(),
                            corrected,
                            seed,
                        ));
                    }
                    continue;
                }
            };
            // Per-arm prompt versions apply to Keep's own protocol only; a
            // parity protocol always uses its upstream prompt.
            let answer_prompt = if m.protocol.is_keep() {
                arm.answer_prompt(&m.models.answer.prompt_version)
            } else {
                protocol.answer_prompt_version()
            };
            let messages = if m.protocol.is_keep() {
                crate::prompt::answer_messages_for(answer_prompt, &bundle.text, &q.query)
            } else {
                protocol.answer_messages(&bundle.text, &q.query, history)
            };
            let input_tokens: usize = messages.iter().map(|x| tokenizer.count(&x.content)).sum();
            // Arms size their context for Keep's own prompt; a parity
            // protocol's prompt is longer, so check the fit again.
            let limit = m.models.answer.context_limit;
            let output = m.models.answer.max_output_tokens as usize;
            if !m.protocol.is_keep() && input_tokens + output > limit {
                let reason = format!(
                    "{} answer prompt needs {input_tokens} + {output} tokens; the answer model holds {limit}",
                    protocol.as_str()
                );
                for &seed in &m.seeds {
                    records.push(base(
                        Outcome::Unavailable {
                            reason: reason.clone(),
                        },
                        gold.clone(),
                        corrected,
                        seed,
                    ));
                }
                continue;
            }
            let prompt_record = PromptRecord {
                version: answer_prompt.into(),
                sha256: ContentHash::of_bytes(
                    canonical_json(&serde_json::to_value(&messages).expect("messages serialise"))
                        .as_bytes(),
                ),
                messages: (m.artifacts.prompts == PromptStorage::Full).then(|| messages.clone()),
                input_tokens,
                tokenizer: tokenizer.id().into(),
                tokenizer_exact: tokenizer.exact(),
            };
            let context_record = ContextRecord::from_bundle(&bundle, build_ms);

            if opts.mode == Mode::Contexts {
                if selection.capture_context_text {
                    context_texts.push(ContextText {
                        arm: arm.name.clone(),
                        question_id: q.id.clone(),
                        text: bundle.text.clone(),
                    });
                }
                // Contexts do not depend on the seed: one record each.
                let mut rec = base(Outcome::Estimated, gold.clone(), corrected, m.seeds[0]);
                rec.context = Some(context_record.clone());
                rec.prompt = Some(prompt_record.clone());
                rec.metrics = Some(crate::metrics::evidence_only(
                    &q.evidence_turn_ids,
                    &bundle.included_turn_ids,
                ));
                records.push(rec);
                continue;
            }
            for &seed in &m.seeds {
                let mut rec = base(Outcome::Estimated, gold.clone(), corrected, seed);
                rec.context = Some(context_record.clone());
                rec.prompt = Some(prompt_record.clone());
                if opts.mode == Mode::Estimate {
                    records.push(rec);
                    continue;
                }
                jobs.push(Job {
                    index: records.len(),
                    messages: messages.clone(),
                    input_tokens: input_tokens as u64,
                    seed,
                    gold: gold.clone(),
                    question: q.query.text.clone(),
                    question_id: q.id.clone(),
                    category: q.category.clone(),
                    abstention: q.abstention,
                    protocol,
                    answer_prompt: answer_prompt.to_string(),
                    evidence: q.evidence_turn_ids.clone(),
                    included: bundle.included_turn_ids.clone(),
                });
                records.push(rec);
            }
        }
        if keep_providers::timing::enabled() {
            eprintln!(
                "[{}] timings for {}: {:.1}s wall",
                m.id,
                arm.name,
                arm_started.elapsed().as_secs_f64()
            );
            eprint!(
                "{}",
                keep_providers::timing::render(&keep_providers::timing::take())
            );
        }
    }

    // Contexts are built in order above (Keep arms carry state between
    // questions); the model calls are independent, so they run concurrently
    // and land back in their records by index.
    let concurrency = m.dispatch_concurrency()?;
    if !jobs.is_empty() {
        let ctx = Arc::new(JobContext {
            answer: providers
                .answer
                .clone()
                .expect("execute mode has an answer model"),
            judge: providers
                .judge
                .clone()
                .expect("execute mode has a judge model"),
            answer_spec: m.models.answer.clone(),
            judge_spec: m.models.judge.clone(),
            max_attempts: m.budget.max_attempts,
            tokenizer: tokenizer.clone(),
            consistency: m.judge_consistency.clone().map(|spec| {
                let dir = spec.cache.then(crate::judge::VerdictCache::default_dir);
                (spec, crate::judge::VerdictCache::new(dir))
            }),
        });
        let total = jobs.len();
        let gate = Arc::new(Semaphore::new(concurrency));
        let mut set = tokio::task::JoinSet::new();
        let mut done = 0;
        let mut apply = |records: &mut Vec<QuestionRecord>,
                         index: usize,
                         result: JobResult,
                         aborted: &mut Option<String>| {
            done += 1;
            if aborted.is_none() {
                *aborted = result.fatal;
            }
            let rec = &mut records[index];
            rec.outcome = result.outcome;
            rec.answer = Some(result.answer);
            rec.metrics = result.metrics;
            rec.judge = result.judge;
            rec.judge_call = result.judge_call;
            rec.judge_consistency = result.judge_consistency;
            rec.final_answer = result.final_answer;
            if done % 10 == 0 || done == total {
                eprintln!("[{}] {done}/{total} questions answered and judged", m.id);
            }
        };
        // Set by a job before it frees its slot, so the next dispatch sees a
        // fatal error even before the job's result is joined.
        let halted = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut pending = jobs.into_iter();
        for job in pending.by_ref() {
            let slot = gate.clone().acquire_owned().await?;
            // Collect finished jobs before each dispatch so a fatal error
            // stops scheduling. The client also refuses calls once halted.
            while let Some(joined) = set.try_join_next() {
                let (index, result) = joined?;
                apply(&mut records, index, result, &mut aborted);
            }
            if aborted.is_some() || halted.load(std::sync::atomic::Ordering::Acquire) {
                records[job.index].outcome = Outcome::NotRun {
                    reason: "run aborted before this call".into(),
                };
                break;
            }
            let ctx = ctx.clone();
            let halted = halted.clone();
            set.spawn(async move {
                let _slot = slot;
                let index = job.index;
                let result = execute_job(&ctx, job).await;
                if result.fatal.is_some() {
                    halted.store(true, std::sync::atomic::Ordering::Release);
                }
                (index, result)
            });
        }
        for job in pending {
            records[job.index].outcome = Outcome::NotRun {
                reason: "run aborted before this call".into(),
            };
        }
        while let Some(joined) = set.join_next().await {
            let (index, result) = joined?;
            apply(&mut records, index, result, &mut aborted);
        }
        if let Some(reason) = &aborted {
            eprintln!("[{}] run aborted: {reason}", m.id);
        }
    }

    let mut embedding_usage = Usage::default();
    let embedding_provider = baselines
        .dense
        .as_ref()
        .map(|d| d.embedder.descriptor().clone())
        .or_else(|| {
            knowledge
                .is_some()
                .then(|| providers.embedder.as_ref().map(|e| e.descriptor().clone()))
                .flatten()
        });
    let piecemeal = !storage.as_ref().is_some_and(RunStorage::drops_everything);
    if let Some(k) = knowledge.as_mut().filter(|_| piecemeal) {
        warnings.extend(k.cleanup().await);
    }
    if let Some(d) = baselines.dense.as_mut() {
        embedding_usage = d.usage;
        if piecemeal {
            warnings.extend(d.cleanup().await);
        }
    }
    drop(knowledge);
    if let Some(st) = &storage {
        warnings.extend(st.close().await);
    }

    let descriptor = |p: &Option<Arc<dyn ChatModel>>| p.as_ref().map(|x| x.descriptor().clone());
    let answer_provider = descriptor(&providers.answer);
    let judge_provider = descriptor(&providers.judge);
    let mock = [&answer_provider, &judge_provider, &embedding_provider]
        .iter()
        .any(|d| d.as_ref().is_some_and(|d| d.mock));
    let fm = fit_mask.as_ref().map(|f| FitSummary {
        fit_fraction: f.fit_fraction,
        gate_eligible: f.gate_eligible,
        tokenizer: f.tokenizer.clone(),
        tokenizer_exact: f.tokenizer_exact,
        // Mock counts are not a provider's, so they verify nothing.
        reported_check: (opts.mode == Mode::Execute && !f.tokenizer_exact && !mock)
            .then(|| f.verify_reported(&sent_full_context(&records)))
            .filter(|v| v.calls_checked + v.calls_unreported + v.calls_failed > 0),
    });
    let meta = RunMeta {
        schema: "keep-run-v1".into(),
        run_id: run_id.clone(),
        harness_version: env!("CARGO_PKG_VERSION").into(),
        mode: opts.mode,
        manifest_id: m.id.clone(),
        manifest_hash: lm.hash.clone(),
        manifest_file_hash: ContentHash::of_bytes(lm.raw.as_bytes()),
        phase: m.phase,
        dataset: m.dataset.kind.as_str().into(),
        dataset_hash: dataset.source_hash.clone(),
        gold_track: format!("{:?}", m.dataset.gold_track).to_lowercase(),
        annotations: annotations.annotations.len(),
        development_histories: split.development_count,
        holdout_histories: split.holdout_count,
        histories_run: dataset.histories.len(),
        questions_run: dataset.questions.len(),
        subset_size,
        seeds: m.seeds.clone(),
        storage_profile: if opts.mode == Mode::Estimate {
            StorageProfile::InMemory
        } else {
            m.storage.profile
        },
        vector_backend: vector_caps,
        answer_provider,
        judge_provider,
        embedding_provider,
        tokenizer: tokenizer.id().into(),
        tokenizer_exact: tokenizer.exact(),
        mock,
        fit_mask: fm,
        embedding_usage,
        budget_usage: providers.ledger.as_ref().map(|l| l.usage()),
        credit_usage: providers.credits.as_ref().map(|l| l.usage()),
        non_canonical,
        concurrency: (opts.mode == Mode::Execute).then_some(concurrency),
        answer_endpoint: providers
            .answer_endpoint
            .clone()
            .map(|e| EndpointRecord::observe(e, records.iter().filter_map(|r| r.answer.as_ref()))),
        judge_endpoint: providers.judge_endpoint.clone().map(|e| {
            EndpointRecord::observe(e, records.iter().filter_map(|r| r.judge_call.as_ref()))
        }),
        aborted: aborted.clone(),
        extraction,
        summaries,
        started_at,
        finished_at: chrono::Utc::now().to_rfc3339(),
        warnings,
        protocol: m.protocol.record(),
        rate_limit: providers.limiter.as_ref().map(|l| l.stats()),
    };

    let dir = match &opts.results_root {
        Some(root) => {
            let dir = create_run_dir(root, &run_id)?;
            std::fs::write(dir.join("manifest.toml"), &lm.raw)?;
            write_json(&dir.join("manifest.json"), m)?;
            write_json(&dir.join("run.json"), &meta)?;
            write_jsonl(&dir.join("records.jsonl"), &records)?;
            let summary = report::summarize(&meta, m, &records);
            write_json(&dir.join("summary.json"), &summary)?;
            std::fs::write(dir.join("summary.md"), report::markdown(&summary))?;
            Some(dir)
        }
        None => None,
    };
    if let Some(reason) = aborted {
        anyhow::bail!(
            "run aborted: {reason}; artifacts{} are kept for inspection and cannot be compared",
            dir.as_ref()
                .map(|d| format!(" in {}", d.display()))
                .unwrap_or_default()
        );
    }
    Ok(RunOutput {
        meta,
        records,
        dir,
        context_texts,
    })
}
