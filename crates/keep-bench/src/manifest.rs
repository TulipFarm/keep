//! Preregistered run manifests.
//!
//! A manifest fixes everything that can change a result before the run:
//! dataset file and subset, holdout definition, model pins, prompts, seeds,
//! storage and embedding profiles, retrieval settings, arms, budgets and the
//! statistical protocol. Its hash is computed over the parsed, defaulted
//! configuration and stamped on every artifact, so a number can always be
//! traced to the exact configuration that produced it.

use std::path::{Path, PathBuf};

use keep_core::{ContentHash, canonical_json};
use keep_providers::openai::{ApiStyle, AuthHeader};
use serde::{Deserialize, Serialize};

use crate::ablation::{ArmKind, ArmSpec, Chunking};
use crate::dataset::DatasetKind;
use crate::dataset::annotations::GoldTrack;

pub const SCHEMA_VERSION: u32 = 1;

/// The model files a `local-onnx` embedding pins, by repository path.
pub const LOCAL_MODEL_FILES: [&str; 5] = [
    "onnx/model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

/// Values that mark a field as deliberately unfilled.
const PLACEHOLDERS: &[&str] = &["PIN-REQUIRED", "TBD", "TODO", "latest", "main", "CHANGE-ME"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Phase {
    /// Tuning and baseline work on development data. Holdout is sealed.
    Development,
    /// One frozen candidate scored on the sealed holdout.
    Confirmation,
}

/// Whether a run can ever stand behind a reported result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Track {
    /// Eligible for comparisons and the superiority gate.
    #[default]
    Canonical,
    /// Development-only plumbing and exploration. Every artifact is stamped
    /// non-canonical; `compare` refuses it unless `--allow-non-canonical`
    /// asks for an internal comparison.
    Development,
}

impl Track {
    pub fn is_canonical(&self) -> bool {
        *self == Track::Canonical
    }
}

/// Stamped on runs that are development-track but use no Copilot model.
pub const DEVELOPMENT_TRACK_BANNER: &str = "NON-CANONICAL: development-track run";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub id: String,
    #[serde(default)]
    pub description: String,
    pub phase: Phase,
    #[serde(default, skip_serializing_if = "Track::is_canonical")]
    pub track: Track,
    pub dataset: DatasetSpec,
    pub holdout: HoldoutSpec,
    #[serde(default)]
    pub confirmation: Option<ConfirmationSpec>,
    #[serde(default)]
    pub fit_mask: Option<PinnedFile>,
    pub models: Models,
    pub embedding: EmbeddingSpec,
    pub storage: StorageSpec,
    #[serde(default)]
    pub retrieval: RetrievalSpec,
    /// Cross-encoder for the ranked Keep arms (K2, Mk2). Absent means they
    /// rank by fusion alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reranker: Option<RerankerSpec>,
    /// Session fact extraction for the K3 and Mk3 arms. Absent in manifests
    /// that have no such arm, so their hashes do not change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extraction: Option<ExtractionSpec>,
    pub budget: BudgetSpec,
    pub seeds: Vec<u64>,
    #[serde(default)]
    pub statistics: StatisticsSpec,
    #[serde(default)]
    pub artifacts: ArtifactSpec,
    #[serde(default)]
    pub dispatch: DispatchSpec,
    /// Scoring protocol. Absent means Keep's own, and the manifest hashes
    /// exactly as it did before protocols existed.
    #[serde(default, skip_serializing_if = "is_default")]
    pub protocol: crate::protocol::ProtocolSpec,
    /// Cached and optionally majority-voted verdicts. Absent means every
    /// answer is judged once, and the manifest hashes as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_consistency: Option<crate::judge::JudgeConsistencySpec>,
    pub arms: Vec<ArmSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetSpec {
    pub kind: DatasetKind,
    /// Relative paths resolve against the manifest's directory.
    pub path: PathBuf,
    pub sha256: ContentHash,
    /// Optional file listing question ids (one per line) to keep.
    #[serde(default)]
    pub subset: Option<PinnedFile>,
    #[serde(default)]
    pub annotations: Option<PinnedFile>,
    #[serde(default)]
    pub gold_track: GoldTrack,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedFile {
    pub path: PathBuf,
    pub sha256: ContentHash,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HoldoutSpec {
    /// Mixed with each history id to assign it to a split.
    pub salt: String,
    /// Share of independent histories reserved for confirmation.
    pub fraction: f64,
    /// Hash of the sorted holdout history ids; written by `keep-bench seal`.
    #[serde(default)]
    pub commitment: Option<ContentHash>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfirmationSpec {
    /// Hash identifying the frozen candidate (for example the manifest hash
    /// of its last development run plus the engine build).
    pub frozen_candidate: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    /// Deterministic offline stand-in; never a measurement.
    Mock,
    OpenaiCompatible,
    /// The local GitHub Copilot CLI. Development track only: it adds a
    /// hidden system prompt and does not pin the upstream model.
    CopilotCli,
}

/// Settings for `provider = "copilot-cli"`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CopilotSpec {
    /// `--reasoning-effort`; omit for models that reject it.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// AI credits per 1,000 input tokens, used only to size reservations
    /// against `budget.max_ai_credits`. Settled spend is what the CLI reports.
    pub credits_per_1k_input: f64,
    pub credits_per_1k_output: f64,
    /// Expected hidden harness tokens per call (system prompt and tools).
    pub overhead_tokens: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub provider: ProviderKind,
    pub model: String,
    /// Exact weights/deployment revision. Placeholders block dispatch.
    pub revision: String,
    /// `approx-v1` or a pinned tokenizer id. Approximate counts are not
    /// allowed for confirmation runs.
    pub tokenizer: String,
    pub context_limit: usize,
    pub max_output_tokens: u32,
    #[serde(default)]
    pub temperature: f32,
    pub prompt_version: String,
    #[serde(default)]
    pub base_url_env: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub input_usd_per_mtok: f64,
    #[serde(default)]
    pub output_usd_per_mtok: f64,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub copilot: Option<CopilotSpec>,
    /// URL layout for `openai-compatible`: `openai` (default), `azure-v1` or
    /// `azure-deployments`.
    #[serde(default, skip_serializing_if = "is_default")]
    pub api_style: ApiStyle,
    /// `bearer` (default) or `api-key`.
    #[serde(default, skip_serializing_if = "is_default")]
    pub auth_header: AuthHeader,
    /// `api-version` query parameter; required by `azure-deployments`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    /// Environment variable holding the Azure deployment name. Required by
    /// the Azure styles; the name is recorded in results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deployment_env: Option<String>,
    /// The exact `model` every reply must report (for example
    /// `gpt-4o-mini-2024-07-18`). A mismatch aborts the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_model: Option<String>,
    /// For `openai-compatible` reasoning models: sent as `reasoning_effort`
    /// (`none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`), with
    /// the output allowance sent as `max_completion_tokens`. Reasoning tokens
    /// count against `max_output_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

/// Reasoning-effort values the harness passes through.
pub const REASONING_EFFORTS: [&str; 7] =
    ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

fn default_timeout() -> u64 {
    120
}

fn is_default<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

impl ModelSpec {
    pub fn price(&self) -> keep_providers::Price {
        keep_providers::Price {
            input_per_mtok_usd: self.input_usd_per_mtok,
            output_per_mtok_usd: self.output_usd_per_mtok,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Models {
    pub answer: ModelSpec,
    pub judge: ModelSpec,
    /// Reads each session once at ingest and lists its facts (K3, Mk3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract: Option<ModelSpec>,
}

impl Models {
    /// Every configured model with its role name.
    pub fn all(&self) -> Vec<(&'static str, &ModelSpec)> {
        let mut v = vec![("answer", &self.answer), ("judge", &self.judge)];
        if let Some(e) = &self.extract {
            v.push(("extract", e));
        }
        v
    }
}

/// How the K3 and Mk3 arms extract and present session facts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionSpec {
    /// Must equal the extractor's built-in prompt version, so a changed
    /// prompt cannot silently reuse a manifest.
    pub prompt_version: String,
    /// Most tokens the dated facts block may take out of the arm's budget;
    /// the ranked evidence fill gets the rest.
    pub facts_tokens: usize,
    /// Expected reply length per session, used only for estimates.
    pub expected_output_tokens: u64,
    /// Most tokens the Memory preference profile may take, for arms with
    /// `preferences = true`. It is standing context, so at most
    /// `retrieval.standing_tokens`, and comes out of the arm's one budget.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub profile_tokens: usize,
    /// Most tokens the per-session summaries may take, for arms with
    /// `summaries = true`. They come out of the same budget, after the
    /// facts block and before the ranked fill, and never take more than
    /// half of what the facts block leaves.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub summary_tokens: usize,
    /// Must equal the summarizer's built-in prompt version when set; needed
    /// with `summary_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary_prompt_version: Option<String>,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbeddingProvider {
    /// Offline feature hashing; plumbing only.
    Hash,
    OpenaiCompatible,
    /// A local ONNX sentence-embedding model (needs the `local-embeddings`
    /// build feature). Files are pinned by commit and SHA-256.
    LocalOnnx,
}

/// Settings for `provider = "local-onnx"`. `model` is the Hugging Face
/// repository and `revision` its full commit hash.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalEmbeddingSpec {
    /// `cls` or `mean`.
    pub pooling: String,
    pub max_length: usize,
    /// `sha256:<hex>` for each model file, keyed by its repository path.
    pub sha256: std::collections::BTreeMap<String, String>,
}

/// A local cross-encoder, pinned like a local embedding model: `model` is
/// the Hugging Face repository, `revision` its full commit hash and `sha256`
/// a `sha256:<hex>` digest per model file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RerankerSpec {
    pub provider: RerankerProvider,
    pub model: String,
    pub revision: String,
    pub max_length: usize,
    /// How many of the fused results the reranker reorders.
    pub top_n: usize,
    pub sha256: std::collections::BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RerankerProvider {
    LocalOnnx,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingSpec {
    pub provider: EmbeddingProvider,
    pub model: String,
    pub revision: String,
    pub dimensions: usize,
    #[serde(default)]
    pub base_url_env: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub input_usd_per_mtok: f64,
    /// Texts per embedding request. Local ONNX embeds one text per model
    /// run whatever the request size, so it takes a whole history per call
    /// and ignores this; see [`EmbeddingSpec::request_batch`].
    #[serde(default = "default_batch")]
    pub batch_size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local: Option<LocalEmbeddingSpec>,
}

impl EmbeddingSpec {
    /// Texts to send per embedding call. Request size cannot change a local
    /// vector, so local ONNX is unbounded; remote APIs keep `batch_size`.
    pub fn request_batch(&self) -> usize {
        match self.provider {
            EmbeddingProvider::LocalOnnx => usize::MAX,
            _ => self.batch_size.max(1),
        }
    }
}

fn default_batch() -> usize {
    64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StorageProfile {
    /// Process memory; for tests and dry runs.
    InMemory,
    /// Postgres ledger with pgvector vectors.
    PostgresVector,
    /// Postgres ledger with Qdrant vectors.
    PostgresQdrant,
}

impl StorageProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            StorageProfile::InMemory => "in-memory",
            StorageProfile::PostgresVector => "postgres-vector",
            StorageProfile::PostgresQdrant => "postgres-qdrant",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageSpec {
    pub profile: StorageProfile,
    #[serde(default = "default_pg_env")]
    pub postgres_url_env: String,
    #[serde(default = "default_qdrant_env")]
    pub qdrant_url_env: String,
    #[serde(default)]
    pub qdrant_api_key_env: Option<String>,
}

fn default_pg_env() -> String {
    "KEEP_POSTGRES_URL".into()
}
fn default_qdrant_env() -> String {
    "KEEP_QDRANT_URL".into()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalSpec {
    #[serde(default)]
    pub chunking: Chunking,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    /// Standing-context allowance for Keep arms.
    pub standing_tokens: usize,
    /// Shared retained tool-evidence allowance for Keep arms.
    pub evidence_tokens: usize,
    pub max_tool_calls: usize,
}

impl Default for RetrievalSpec {
    fn default() -> Self {
        Self {
            chunking: Chunking::Turn,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            standing_tokens: 4096,
            evidence_tokens: 2000,
            max_tool_calls: 6,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetSpec {
    pub max_model_calls: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub max_cost_usd: f64,
    /// Attempts per call including the first; retries only on retryable errors.
    pub max_attempts: u32,
    /// Run-level cap on AI credits reported by a metered harness (the
    /// Copilot CLI). Required when any model uses it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_ai_credits: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatisticsSpec {
    pub method: String,
    pub bootstrap_resamples: usize,
    pub bootstrap_seed: u64,
    pub confidence: f64,
    pub multiplicity: String,
    /// Minimum mean accuracy gain in percentage points.
    pub min_gain_pp: f64,
    /// Minimum share of scheduled answers that must complete.
    pub min_completion: f64,
    /// Minimum share of questions that must fit full context for the gate.
    pub min_fit_fraction: f64,
}

impl Default for StatisticsSpec {
    fn default() -> Self {
        Self {
            method: "paired-cluster-bootstrap-percentile".into(),
            bootstrap_resamples: 10_000,
            bootstrap_seed: 20_261_112,
            confidence: 0.95,
            multiplicity: "holm".into(),
            min_gain_pp: 2.0,
            min_completion: 0.99,
            min_fit_fraction: 0.8,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PromptStorage {
    /// Store complete prompts in per-question records.
    #[default]
    Full,
    /// Store only prompt hashes (for datasets that forbid redistribution or
    /// very large contexts).
    Hash,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSpec {
    #[serde(default)]
    pub prompts: PromptStorage,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchSpec {
    #[serde(default)]
    pub allow_model_calls: bool,
    /// Questions answered and judged at once, and sessions extracted or
    /// summarised at once. Defaults to 4 for Copilot CLI runs and 1
    /// otherwise; [`CONCURRENCY_ENV`] overrides it for one run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<usize>,
    /// Client-side pacing for a quota shared by every `openai-compatible`
    /// chat model in the run (answer, judge and extract on one deployment):
    /// tokens per minute, counting each prompt plus its whole output
    /// allowance, as Azure OpenAI does. Unset means no pacing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_minute: Option<u64>,
    /// Requests per minute under the same quota.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requests_per_minute: Option<u64>,
}

/// Override `dispatch.tokens_per_minute` for one run. Pacing changes how
/// fast calls go out, not what is measured, so a deployment whose quota was
/// raised needs no new manifest. The pacing used is recorded in `run.json`.
pub const TOKENS_PER_MINUTE_ENV: &str = "KEEP_DISPATCH_TOKENS_PER_MINUTE";
/// Override `dispatch.requests_per_minute` for one run.
pub const REQUESTS_PER_MINUTE_ENV: &str = "KEEP_DISPATCH_REQUESTS_PER_MINUTE";
/// Override `dispatch.concurrency` for one run. Like pacing, it changes how
/// many calls are in flight, not what is measured: every call still goes
/// through the same pacer and budget, and results land in the same order.
/// The concurrency used is recorded in `run.json`.
pub const CONCURRENCY_ENV: &str = "KEEP_DISPATCH_CONCURRENCY";
/// Highest [`CONCURRENCY_ENV`] accepted; pacing, not concurrency, is what
/// holds a run under its quota.
pub const MAX_CONCURRENCY: usize = 1024;

fn positive_env(var: impl Fn(&str) -> Option<String>, name: &str) -> anyhow::Result<Option<u64>> {
    match var(name).filter(|v| !v.trim().is_empty()) {
        None => Ok(None),
        Some(v) => match v.trim().parse::<u64>() {
            Ok(n) if n > 0 => Ok(Some(n)),
            _ => anyhow::bail!("{name} must be a positive whole number, not {v:?}"),
        },
    }
}

impl DispatchSpec {
    /// Tokens and requests per minute, with the environment overrides applied.
    pub fn pacing(&self) -> anyhow::Result<(Option<u64>, Option<u64>)> {
        self.pacing_with(|name| std::env::var(name).ok())
    }

    fn pacing_with(
        &self,
        var: impl Fn(&str) -> Option<String>,
    ) -> anyhow::Result<(Option<u64>, Option<u64>)> {
        let read = |name: &str, manifest: Option<u64>| -> anyhow::Result<Option<u64>> {
            Ok(positive_env(&var, name)?.or(manifest))
        };
        Ok((
            read(TOKENS_PER_MINUTE_ENV, self.tokens_per_minute)?,
            read(REQUESTS_PER_MINUTE_ENV, self.requests_per_minute)?,
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Issue {
    pub error: bool,
    pub field: String,
    pub message: String,
}

/// A parsed manifest together with where it came from.
#[derive(Clone, Debug)]
pub struct LoadedManifest {
    pub manifest: Manifest,
    pub path: PathBuf,
    pub raw: String,
    pub hash: ContentHash,
}

impl LoadedManifest {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read manifest {}: {e}", path.display()))?;
        let manifest: Manifest = toml::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("invalid manifest {}: {e}", path.display()))?;
        let hash = manifest.hash();
        Ok(Self {
            manifest,
            path: path.to_owned(),
            raw,
            hash,
        })
    }

    /// Resolve a manifest-relative path.
    pub fn resolve(&self, p: &Path) -> PathBuf {
        if p.is_absolute() {
            p.to_owned()
        } else {
            self.path.parent().unwrap_or(Path::new(".")).join(p)
        }
    }

    /// Read a pinned file and verify its hash.
    pub fn read_pinned(&self, pinned: &PinnedFile) -> anyhow::Result<Vec<u8>> {
        let path = self.resolve(&pinned.path);
        let bytes = std::fs::read(&path)
            .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
        let actual = ContentHash::of_bytes(&bytes);
        anyhow::ensure!(
            actual == pinned.sha256,
            "{} has hash {actual}, manifest pins {}",
            path.display(),
            pinned.sha256
        );
        Ok(bytes)
    }
}

fn is_placeholder(value: &str) -> bool {
    let v = value.trim();
    v.is_empty()
        || PLACEHOLDERS
            .iter()
            .any(|p| v.eq_ignore_ascii_case(p) || v.contains("PIN-REQUIRED"))
}

impl Manifest {
    /// Hash of the effective configuration (defaults included, key order
    /// irrelevant). Comments and formatting in the TOML do not change it.
    pub fn hash(&self) -> ContentHash {
        let value = serde_json::to_value(self).expect("manifest serialises");
        ContentHash::of_bytes(canonical_json(&value).as_bytes())
    }

    pub fn uses_remote_models(&self) -> bool {
        self.uses_copilot()
            || self.models.answer.provider == ProviderKind::OpenaiCompatible
            || self.models.judge.provider == ProviderKind::OpenaiCompatible
            || self.extract_model().map(|e| e.provider) == Some(ProviderKind::OpenaiCompatible)
            || (self.embedding.provider == EmbeddingProvider::OpenaiCompatible
                && self
                    .arms
                    .iter()
                    .any(|a| matches!(a.kind, ArmKind::Dense { .. })))
    }

    pub fn uses_copilot(&self) -> bool {
        self.models.answer.provider == ProviderKind::CopilotCli
            || self.models.judge.provider == ProviderKind::CopilotCli
            || self.extract_model().map(|e| e.provider) == Some(ProviderKind::CopilotCli)
    }

    /// Whether any arm reads model-extracted session facts.
    pub fn uses_extracted_facts(&self) -> bool {
        self.arms.iter().any(|a| {
            matches!(a.kind, ArmKind::Keep { path, .. }
                if path.uses_extracted_facts() && a.kind.unavailable_reason().is_none())
        })
    }

    /// The extraction model, when an arm needs it.
    pub fn extract_model(&self) -> Option<&ModelSpec> {
        self.models
            .extract
            .as_ref()
            .filter(|_| self.uses_extracted_facts())
    }

    /// Whether any remote call is priced in USD (and so counts against the
    /// cross-run spend ledger).
    pub fn uses_openai_compatible(&self) -> bool {
        self.models.answer.provider == ProviderKind::OpenaiCompatible
            || self.models.judge.provider == ProviderKind::OpenaiCompatible
            || self.extract_model().map(|e| e.provider) == Some(ProviderKind::OpenaiCompatible)
            || (self.embedding.provider == EmbeddingProvider::OpenaiCompatible
                && self.uses_embeddings())
    }

    /// Whether any arm needs an embedder (dense baselines or Knowledge paths).
    pub fn uses_embeddings(&self) -> bool {
        self.arms.iter().any(|a| match &a.kind {
            ArmKind::Dense { .. } => true,
            ArmKind::Keep { path, .. } => {
                a.kind.unavailable_reason().is_none() && path.uses_knowledge()
            }
            _ => false,
        })
    }

    /// The banner every artifact of this run must carry, if it is not
    /// canonical.
    pub fn non_canonical_label(&self) -> Option<&'static str> {
        if self.uses_copilot() {
            Some(keep_providers::copilot::NON_CANONICAL_BANNER)
        } else if self.track == Track::Development {
            Some(DEVELOPMENT_TRACK_BANNER)
        } else {
            None
        }
    }

    /// The tokenizer named by the answer model, or the approximate counter
    /// with a warning when this build cannot provide it.
    pub fn answer_tokenizer(
        &self,
    ) -> (
        std::sync::Arc<dyn keep_providers::Tokenizer>,
        Option<String>,
    ) {
        let id = &self.models.answer.tokenizer;
        match keep_providers::tokenizer_for(id) {
            Some(t) => (t, None),
            None => (
                std::sync::Arc::new(keep_providers::ApproxTokenizer),
                Some(format!(
                    "tokenizer {id:?} is not available in this build; counting with approx-v1"
                )),
            ),
        }
    }

    /// Calls in flight at once, with [`CONCURRENCY_ENV`] applied.
    pub fn dispatch_concurrency(&self) -> anyhow::Result<usize> {
        self.dispatch_concurrency_with(|name| std::env::var(name).ok())
    }

    pub fn dispatch_concurrency_with(
        &self,
        var: impl Fn(&str) -> Option<String>,
    ) -> anyhow::Result<usize> {
        let manifest = self
            .dispatch
            .concurrency
            .unwrap_or(if self.uses_copilot() { 4 } else { 1 });
        let n = match positive_env(var, CONCURRENCY_ENV)? {
            Some(n) if n > MAX_CONCURRENCY as u64 => {
                anyhow::bail!("{CONCURRENCY_ENV} must be at most {MAX_CONCURRENCY}, not {n}")
            }
            Some(n) => n as usize,
            None => manifest,
        };
        Ok(n.max(1))
    }

    pub fn uses_mock_models(&self) -> bool {
        self.models.answer.provider == ProviderKind::Mock
            || self.models.judge.provider == ProviderKind::Mock
            || self.extract_model().map(|e| e.provider) == Some(ProviderKind::Mock)
            || self.embedding.provider == EmbeddingProvider::Hash
    }

    /// The most context a Keep arm may inject: the standing allowance plus
    /// the evidence allowance. Equal-context baselines use this budget.
    pub fn keep_context_tokens(&self) -> usize {
        self.retrieval.standing_tokens + self.retrieval.evidence_tokens
    }

    /// Whether a lexical or dense baseline gets the same context budget as
    /// the combined Memory + Knowledge arms. `None` when there is no
    /// combined arm to compare against.
    pub fn has_equal_context_baseline(&self) -> Option<bool> {
        let combined = self.arms.iter().any(|a| {
            matches!(a.kind, ArmKind::Keep { path, .. } if path.uses_memory() && path.uses_knowledge())
        });
        combined.then(|| {
            self.arms.iter().any(|a| {
                matches!(a.kind,
                    ArmKind::Bm25 { budget_tokens, .. } | ArmKind::Dense { budget_tokens, .. }
                        if budget_tokens >= self.keep_context_tokens())
            })
        })
    }

    pub fn full_context_arm(&self) -> Option<&ArmSpec> {
        self.arms.iter().find(|a| a.kind == ArmKind::FullContext)
    }

    /// Structural checks that need no files.
    pub fn validate(&self) -> Vec<Issue> {
        let mut issues = Vec::new();
        let mut err = |field: &str, message: String| {
            issues.push(Issue {
                error: true,
                field: field.into(),
                message,
            })
        };
        if self.schema_version != SCHEMA_VERSION {
            err(
                "schema_version",
                format!(
                    "unsupported schema version {}; expected {SCHEMA_VERSION}",
                    self.schema_version
                ),
            );
        }
        if self.id.trim().is_empty() {
            err("id", "manifest id is empty".into());
        }
        if !(self.holdout.fraction > 0.0 && self.holdout.fraction < 1.0) {
            err(
                "holdout.fraction",
                "must be strictly between 0 and 1".into(),
            );
        }
        if self.holdout.salt.trim().is_empty() {
            err("holdout.salt", "salt is empty".into());
        }
        if self.holdout.commitment.is_none() {
            err(
                "holdout.commitment",
                "holdout is not sealed; run `keep-bench seal` and record the commitment".into(),
            );
        }
        if self.seeds.is_empty() {
            err("seeds", "at least one seed is required".into());
        }
        let mut unique_seeds = self.seeds.clone();
        unique_seeds.sort_unstable();
        unique_seeds.dedup();
        if unique_seeds.len() != self.seeds.len() {
            err("seeds", "seeds must be distinct".into());
        }
        if self.arms.is_empty() {
            err("arms", "no arms declared".into());
        }
        let mut names: Vec<&str> = self.arms.iter().map(|a| a.name.as_str()).collect();
        names.sort_unstable();
        if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
            err(
                "arms",
                format!("arm name {:?} is used more than once", w[0]),
            );
        }
        if self.full_context_arm().is_none() {
            err(
                "arms",
                "a full-context arm is required; when it cannot fit, it is reported as unavailable"
                    .into(),
            );
        }
        for arm in &self.arms {
            let budget = match arm.kind {
                ArmKind::RecentWindow { budget_tokens }
                | ArmKind::Bm25 { budget_tokens, .. }
                | ArmKind::Dense { budget_tokens, .. }
                | ArmKind::NotesGrep { budget_tokens }
                | ArmKind::FilesystemAgent { budget_tokens, .. } => Some(budget_tokens),
                _ => None,
            };
            if budget == Some(0) {
                err(
                    &format!("arms.{}", arm.name),
                    "budget_tokens must be positive".into(),
                );
            }
            if let ArmKind::Bm25 { top_k: 0, .. } | ArmKind::Dense { top_k: 0, .. } = arm.kind {
                err(
                    &format!("arms.{}", arm.name),
                    "top_k must be positive".into(),
                );
            }
        }
        for (name, m) in [
            ("models.answer", &self.models.answer),
            ("models.judge", &self.models.judge),
        ] {
            if m.max_output_tokens == 0 {
                err(
                    &format!("{name}.max_output_tokens"),
                    "must be positive".into(),
                );
            }
            if m.context_limit <= m.max_output_tokens as usize {
                err(
                    &format!("{name}.context_limit"),
                    "must exceed max_output_tokens".into(),
                );
            }
            if m.input_usd_per_mtok < 0.0 || m.output_usd_per_mtok < 0.0 {
                err(name, "prices cannot be negative".into());
            }
        }
        if self.embedding.dimensions == 0 {
            err("embedding.dimensions", "must be positive".into());
        }
        let s = &self.statistics;
        if !(s.confidence > 0.5 && s.confidence < 1.0) {
            err("statistics.confidence", "must be between 0.5 and 1".into());
        }
        if s.bootstrap_resamples < 1000 {
            err(
                "statistics.bootstrap_resamples",
                "use at least 1000 resamples".into(),
            );
        }
        if s.multiplicity != "holm" {
            err(
                "statistics.multiplicity",
                "only `holm` is implemented".into(),
            );
        }
        if s.method != StatisticsSpec::default().method {
            err(
                "statistics.method",
                format!("only `{}` is implemented", StatisticsSpec::default().method),
            );
        }
        if self.phase == Phase::Confirmation {
            if self
                .confirmation
                .as_ref()
                .is_none_or(|c| is_placeholder(&c.frozen_candidate))
            {
                err(
                    "confirmation.frozen_candidate",
                    "confirmation runs must name the frozen candidate".into(),
                );
            }
            if self.seeds.len() < 3 {
                err(
                    "seeds",
                    "confirmation runs need at least three seeds".into(),
                );
            }
            if self.fit_mask.is_none() {
                err(
                    "fit_mask",
                    "confirmation runs need a frozen fit mask".into(),
                );
            }
            for (name, m) in [
                ("models.answer", &self.models.answer),
                ("models.judge", &self.models.judge),
            ] {
                if m.tokenizer == "approx-v1" {
                    err(
                        &format!("{name}.tokenizer"),
                        "confirmation runs need the model's pinned tokenizer, not an estimate"
                            .into(),
                    );
                }
                if m.provider == ProviderKind::Mock {
                    err(
                        &format!("{name}.provider"),
                        "confirmation runs cannot use mock models".into(),
                    );
                }
            }
            if self.has_equal_context_baseline() == Some(false) {
                err(
                    "arms",
                    format!(
                        "confirmation runs with a combined Memory + Knowledge arm need a bm25 or dense arm with budget_tokens >= {} (standing_tokens + evidence_tokens), so the comparison is at equal context",
                        self.keep_context_tokens()
                    ),
                );
            }
            if self.embedding.provider == EmbeddingProvider::Hash
                && self
                    .arms
                    .iter()
                    .any(|a| matches!(a.kind, ArmKind::Dense { .. }))
            {
                err(
                    "embedding.provider",
                    "confirmation runs cannot use the hash embedder".into(),
                );
            }
        }
        if !self.protocol.name.supports(self.dataset.kind) {
            err(
                "protocol.name",
                format!(
                    "protocol `{}` is not defined for dataset `{}`",
                    self.protocol.name.as_str(),
                    self.dataset.kind.as_str()
                ),
            );
        }
        if self.protocol.split == crate::protocol::SplitScope::Full {
            if self.protocol.is_keep() {
                err(
                    "protocol.split",
                    "split = \"full\" is only for parity protocols".into(),
                );
            }
            if self.phase != Phase::Development {
                err(
                    "protocol.split",
                    "split = \"full\" is a development-phase parity run; confirmation keeps its own holdout rules".into(),
                );
            }
        }
        if self.uses_copilot() && self.track != Track::Development {
            err(
                "track",
                "copilot-cli models are development-only; set track = \"development\"".into(),
            );
        }
        if self.track == Track::Development && self.phase == Phase::Confirmation {
            err(
                "track",
                "a development-track manifest cannot run the confirmation phase".into(),
            );
        }
        for (name, m) in [
            ("models.answer", &self.models.answer),
            ("models.judge", &self.models.judge),
        ] {
            if m.provider == ProviderKind::CopilotCli && m.copilot.is_none() {
                err(
                    &format!("{name}.copilot"),
                    "copilot-cli models need a [copilot] table with credit rates".into(),
                );
            }
        }
        if self.dispatch.concurrency == Some(0) {
            err("dispatch.concurrency", "must be at least 1".into());
        }
        if self.embedding.provider == EmbeddingProvider::LocalOnnx {
            let e = &self.embedding;
            if e.revision.len() != 40 || !e.revision.chars().all(|c| c.is_ascii_hexdigit()) {
                err(
                    "embedding.revision",
                    "local-onnx models must be pinned to a full commit hash".into(),
                );
            }
            match &e.local {
                None => err(
                    "embedding.local",
                    "local-onnx needs pooling, max_length and file digests".into(),
                ),
                Some(l) => {
                    if !matches!(l.pooling.as_str(), "cls" | "mean") {
                        err("embedding.local.pooling", "use `cls` or `mean`".into());
                    }
                    for f in LOCAL_MODEL_FILES {
                        if !l.sha256.contains_key(f) {
                            err(
                                "embedding.local.sha256",
                                format!("missing a digest for {f}"),
                            );
                        }
                    }
                }
            }
        }
        if let Some(r) = &self.reranker {
            if r.revision.len() != 40 || !r.revision.chars().all(|c| c.is_ascii_hexdigit()) {
                err(
                    "reranker.revision",
                    "local-onnx models must be pinned to a full commit hash".into(),
                );
            }
            if r.top_n < 2 {
                err("reranker.top_n", "must be at least 2".into());
            }
            if r.max_length == 0 {
                err("reranker.max_length", "must be positive".into());
            }
            for f in LOCAL_MODEL_FILES {
                if !r.sha256.contains_key(f) {
                    err("reranker.sha256", format!("missing a digest for {f}"));
                }
            }
            if !self
                .arms
                .iter()
                .any(|a| matches!(a.kind, ArmKind::Keep { path, .. } if path.is_ranked()))
            {
                err("reranker", "no k2 or mk2 arm uses it".into());
            }
        }
        let fact_arms = self
            .arms
            .iter()
            .any(|a| matches!(a.kind, ArmKind::Keep { path, .. } if path.uses_extracted_facts()));
        match (&self.extraction, &self.models.extract) {
            (Some(x), Some(_)) => {
                if x.prompt_version != keep_knowledge::session_extract::SESSION_PROMPT_VERSION {
                    err(
                        "extraction.prompt_version",
                        format!(
                            "this build extracts with {:?}",
                            keep_knowledge::session_extract::SESSION_PROMPT_VERSION
                        ),
                    );
                }
                if x.facts_tokens == 0 || x.facts_tokens >= self.retrieval.evidence_tokens {
                    err(
                        "extraction.facts_tokens",
                        "must be positive and smaller than retrieval.evidence_tokens".into(),
                    );
                }
                if !fact_arms {
                    err("extraction", "no k3 or mk3 arm uses it".into());
                }
                if x.facts_tokens >= self.retrieval.evidence_tokens {
                    err(
                        "extraction.facts_tokens",
                        "must be smaller than retrieval.evidence_tokens".into(),
                    );
                }
                if x.profile_tokens > self.retrieval.standing_tokens {
                    err(
                        "extraction.profile_tokens",
                        "the profile is standing context; at most retrieval.standing_tokens".into(),
                    );
                }
                let summary_version = keep_knowledge::session_summary::SUMMARY_PROMPT_VERSION;
                match (&x.summary_prompt_version, x.summary_tokens) {
                    (None, 0) => {}
                    (Some(v), n) if n > 0 && v == summary_version => {}
                    (Some(v), _) if v != summary_version => err(
                        "extraction.summary_prompt_version",
                        format!("this build summarizes with {summary_version:?}"),
                    ),
                    _ => err(
                        "extraction.summary_tokens",
                        "summary_tokens and summary_prompt_version must be set together".into(),
                    ),
                }
                if x.summary_tokens > 0 && !self.arms.iter().any(|a| a.summaries) {
                    err(
                        "extraction.summary_tokens",
                        "no arm has summaries = true".into(),
                    );
                }
            }
            (None, None) if fact_arms => err(
                "extraction",
                "k3 and mk3 arms need [extraction] and [models.extract]".into(),
            ),
            (None, None) => {}
            _ => err(
                "extraction",
                "[extraction] and [models.extract] must be set together".into(),
            ),
        }
        for a in &self.arms {
            let path = match a.kind {
                ArmKind::Keep { path, .. } => Some(path),
                _ => None,
            };
            let facts = path.is_some_and(|p| p.uses_extracted_facts());
            if a.temporal && !facts {
                err(
                    &format!("arms.{}.temporal", a.name),
                    "only k3 and mk3 arms read time references".into(),
                );
            }
            if a.assistant_route && !facts {
                err(
                    &format!("arms.{}.assistant_route", a.name),
                    "only k3 and mk3 arms route questions about earlier replies".into(),
                );
            }
            if a.preferences {
                if path != Some(crate::ablation::KeepPath::Mk3) {
                    err(
                        &format!("arms.{}.preferences", a.name),
                        "only mk3 arms have a Memory preference profile".into(),
                    );
                }
                if self
                    .extraction
                    .as_ref()
                    .is_none_or(|x| x.profile_tokens == 0)
                {
                    err(
                        &format!("arms.{}.preferences", a.name),
                        "needs extraction.profile_tokens".into(),
                    );
                }
            }
            for (on, flag, what) in [
                (a.subjects, "subjects", "set facts about other people apart"),
                (a.summaries, "summaries", "read session summaries"),
                (a.expand, "expand", "expand facts by shared entities"),
                (a.provenance, "provenance", "show source turns beside facts"),
                (
                    a.relative_dates,
                    "relative_dates",
                    "anchor source time phrases",
                ),
                (a.decompose, "decompose", "retrieve question facets"),
                (
                    a.fact_selection,
                    "fact_selection",
                    "select individual ranked facts",
                ),
                (a.focus, "focus", "rank excerpts by what the question asks"),
                (
                    a.verbatim_first,
                    "verbatim_first",
                    "leave out facts the excerpts show",
                ),
            ] {
                if on && !facts {
                    err(
                        &format!("arms.{}.{flag}", a.name),
                        format!("only k3 and mk3 arms {what}"),
                    );
                }
            }
            if a.summaries
                && self
                    .extraction
                    .as_ref()
                    .is_none_or(|x| x.summary_tokens == 0)
            {
                err(
                    &format!("arms.{}.summaries", a.name),
                    "needs extraction.summary_tokens".into(),
                );
            }
            if let Some(v) = &a.answer_prompt
                && !crate::prompt::ANSWER_PROMPT_VERSIONS.contains(&v.as_str())
            {
                err(
                    &format!("arms.{}.answer_prompt", a.name),
                    format!(
                        "this build implements {:?}",
                        crate::prompt::ANSWER_PROMPT_VERSIONS
                    ),
                );
            }
            if a.answer_prompt.is_some() && !self.protocol.is_keep() {
                err(
                    &format!("arms.{}.answer_prompt", a.name),
                    "a parity protocol always uses its upstream answer prompt".into(),
                );
            }
        }
        if let Some(jc) = &self.judge_consistency
            && let Err(e) = jc.validate()
        {
            err("judge_consistency.votes", e);
        }
        if self.dispatch.allow_model_calls {
            for p in self.dispatch_problems() {
                issues.push(Issue {
                    error: true,
                    field: "dispatch".into(),
                    message: p,
                });
            }
        }
        issues
    }

    /// Everything that must be fixed before any remote model may be called.
    pub fn dispatch_problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let models: Vec<(&str, &ModelSpec)> = self
            .models
            .all()
            .into_iter()
            .filter(|(name, _)| *name != "extract" || self.uses_extracted_facts())
            .collect();
        for (name, m) in models {
            if m.provider == ProviderKind::CopilotCli {
                if is_placeholder(&m.model) {
                    problems.push(format!("{name} model is not pinned"));
                }
                if self.track != Track::Development {
                    problems.push(format!(
                        "{name} uses copilot-cli, which is only allowed with track = \"development\""
                    ));
                }
                match &m.copilot {
                    Some(c) if c.credits_per_1k_input > 0.0 && c.credits_per_1k_output > 0.0 => {}
                    _ => problems.push(format!(
                        "{name} copilot credit rates must be positive to size reservations"
                    )),
                }
                continue;
            }
            if m.provider != ProviderKind::OpenaiCompatible {
                continue;
            }
            if let Some(e) = &m.reasoning_effort
                && !REASONING_EFFORTS.contains(&e.as_str())
            {
                problems.push(format!(
                    "{name} reasoning_effort {e:?} is not one of {REASONING_EFFORTS:?}"
                ));
            }
            if is_placeholder(&m.model) {
                problems.push(format!("{name} model is not pinned"));
            }
            if is_placeholder(&m.revision) {
                problems.push(format!("{name} model revision is not pinned"));
            }
            if is_placeholder(&m.tokenizer) || m.tokenizer == "approx-v1" {
                problems.push(format!("{name} tokenizer is not pinned"));
            }
            if m.base_url_env.as_deref().is_none_or(is_placeholder) {
                problems.push(format!("{name} base_url_env is not set"));
            }
            if m.api_style.is_azure() {
                // A hosted, metered endpoint: the key, the deployment, the
                // served-model pin and real prices are all required.
                if m.api_key_env.as_deref().is_none_or(is_placeholder) {
                    problems.push(format!("{name} api_key_env is not set"));
                }
                if m.deployment_env.as_deref().is_none_or(is_placeholder) {
                    problems.push(format!("{name} deployment_env is not set"));
                }
                if m.served_model.as_deref().is_none_or(is_placeholder) {
                    problems.push(format!("{name} served_model is not pinned"));
                }
                if !(m.input_usd_per_mtok > 0.0 && m.output_usd_per_mtok > 0.0) {
                    problems.push(format!("{name} token prices must be positive"));
                }
            }
            if m.api_style == ApiStyle::AzureDeployments
                && m.api_version.as_deref().is_none_or(is_placeholder)
            {
                problems.push(format!("{name} azure-deployments needs api_version"));
            }
        }
        if self.embedding.provider == EmbeddingProvider::OpenaiCompatible {
            if is_placeholder(&self.embedding.model) || is_placeholder(&self.embedding.revision) {
                problems.push("embedding model is not pinned".into());
            }
            if self
                .embedding
                .base_url_env
                .as_deref()
                .is_none_or(is_placeholder)
            {
                problems.push("embedding base_url_env is not set".into());
            }
        }
        let b = &self.budget;
        if self.uses_copilot() && b.max_ai_credits.is_none_or(|c| c.is_nan() || c <= 0.0) {
            problems.push("budget.max_ai_credits must be positive for copilot-cli runs".into());
        }
        if self.uses_openai_compatible()
            && (b.max_model_calls == 0
                || b.max_input_tokens == 0
                || b.max_output_tokens == 0
                || b.max_cost_usd <= 0.0)
        {
            problems.push(
                "budget caps must all be positive (calls, input tokens, output tokens, cost)"
                    .into(),
            );
        }
        if b.max_attempts == 0 {
            problems.push("budget.max_attempts must be at least 1".into());
        }
        problems
    }

    pub fn budget_caps(&self) -> keep_providers::BudgetCaps {
        keep_providers::BudgetCaps {
            max_calls: self.budget.max_model_calls,
            max_input_tokens: self.budget.max_input_tokens,
            max_output_tokens: self.budget.max_output_tokens,
            max_cost_usd: self.budget.max_cost_usd,
        }
    }
}

pub fn has_errors(issues: &[Issue]) -> bool {
    issues.iter().any(|i| i.error)
}

#[cfg(test)]
mod pacing_tests {
    use super::*;

    #[test]
    fn pacing_overrides_come_from_the_environment_and_must_be_positive() {
        let spec = DispatchSpec {
            tokens_per_minute: Some(95_000),
            requests_per_minute: Some(570),
            ..Default::default()
        };
        assert_eq!(
            spec.pacing_with(|_| None).unwrap(),
            (Some(95_000), Some(570))
        );
        let raised = spec
            .pacing_with(|n| (n == TOKENS_PER_MINUTE_ENV).then(|| "950000".into()))
            .unwrap();
        assert_eq!(raised, (Some(950_000), Some(570)));
        for bad in ["0", "-5", "1e6", "lots"] {
            assert!(spec.pacing_with(|_| Some(bad.into())).is_err(), "{bad}");
        }
    }
}
