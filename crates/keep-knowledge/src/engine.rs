//! The Knowledge engine: ingestion, background jobs and the read tools
//! (`search`, `fetch`, `traverse`, `cite`, `index`) on top of a [`Ledger`]
//! and an optional vector projection.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use futures_util::{StreamExt, stream};
use keep_providers::{ApproxTokenizer, Embedder, HashEmbedder, Reranker, Tokenizer, timing};
use keep_store::{Bm25Index, Bm25Params, LexicalIndex, ScoredId, VectorIndex, VectorPoint};
use serde::{Deserialize, Serialize};

use crate::budget::EvidenceAllowance;
use crate::chunking::{DEFAULT_CHUNK_CHARS, chunk_text};
use crate::extract::{
    ALIAS, CheckedClaim, Claim, FactExtractor, RuleExtractor, is_single_valued, validate,
};
use crate::fusion::{DEFAULT_RRF_K, rrf};
use crate::index::{IndexRequest, IndexView, normalize, render};
use crate::ledger::{
    AliasWrite, ApplyOutcome, ChangeSet, EntityWrite, IndexInputs, IngestBatch, IngestReport,
    KnowledgeError, Ledger, Projection, PurgeReport, Result, StoredVector, VECTOR_GRACE_SECONDS,
    VersionWrite, normalize_name,
};
use crate::model::{
    Access, Chunk, Entity, Episode, Evidence, Fact, FactStatus, FactVersion, ItemRef, Job,
    JobCounts, JobKind, JobState, NewJob, Source, SourceKind, Time, TimeBasis, digest_hex,
    digest_id,
};
use crate::resolve::{Incoming, Rival, plan};
use crate::session_extract::{SessionExtractor, claims_for, episode_sessions};
use crate::snapshot::{Snapshot, Snapshots};
use crate::temporal::{AsOf, ValidMatch, chunk_visible, fact_as_of};

/// A job's result: an error message and whether it may be retried.
type JobOutcome = std::result::Result<(), (String, bool)>;

/// Source of "now" for system time. Tests and replays use [`ManualClock`].
pub trait Clock: Send + Sync {
    fn now(&self) -> Time;
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Time {
        Utc::now()
    }
}

pub struct ManualClock(Mutex<Time>);

impl ManualClock {
    pub fn new(t: Time) -> Self {
        Self(Mutex::new(t))
    }
    pub fn set(&self, t: Time) {
        *self.0.lock().expect("clock lock") = t;
    }
    pub fn advance(&self, d: chrono::Duration) {
        *self.0.lock().expect("clock lock") += d;
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Time {
        *self.0.lock().expect("clock lock")
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeConfig {
    /// Vector collection that holds every corpus's points.
    pub collection: String,
    pub chunk_chars: usize,
    pub rrf_k: f32,
    /// Candidates taken from each retriever before fusion.
    pub candidate_pool: usize,
    pub max_traverse_nodes: usize,
    /// Where lexical candidates come from. Absent means
    /// [`LexicalMode::Ledger`], so older configs read and write unchanged.
    #[serde(default, skip_serializing_if = "LexicalMode::is_ledger")]
    pub lexical: LexicalMode,
}

/// The lexical retriever behind [`Knowledge::search`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LexicalMode {
    /// The ledger's own lexical search over readable partitions; time
    /// filtering happens after the candidate pools are cut.
    #[default]
    Ledger,
    /// In-process BM25 over exactly the items the reader can see at the
    /// requested time. Both retrievers drop invisible items before their
    /// pools are cut, and an optional reranker reorders the fused head.
    Bm25(Bm25Params),
}

impl LexicalMode {
    pub fn is_ledger(&self) -> bool {
        matches!(self, LexicalMode::Ledger)
    }
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            collection: "knowledge".into(),
            chunk_chars: DEFAULT_CHUNK_CHARS,
            rrf_k: DEFAULT_RRF_K,
            candidate_pool: 50,
            max_traverse_nodes: 100,
            lexical: LexicalMode::Ledger,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceInput {
    pub id: String,
    /// Visibility label inherited by everything derived from the source.
    pub partition: String,
    pub collection: String,
    pub title: String,
    pub kind: SourceKind,
    /// Small host-defined string map returned with citations, such as a
    /// document URL or id. See [`MAX_METADATA_KEYS`] and related limits.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// Most keys a source's metadata may have.
pub const MAX_METADATA_KEYS: usize = 16;
/// Longest metadata key, in bytes.
pub const MAX_METADATA_KEY_BYTES: usize = 64;
/// Longest metadata value, in bytes.
pub const MAX_METADATA_VALUE_BYTES: usize = 1024;

/// Check a source's metadata against the size limits.
pub fn validate_metadata(m: &BTreeMap<String, String>) -> Result<()> {
    if m.len() > MAX_METADATA_KEYS {
        return Err(KnowledgeError::Invalid(format!(
            "metadata has more than {MAX_METADATA_KEYS} keys"
        )));
    }
    for (k, v) in m {
        if k.trim().is_empty() || k.len() > MAX_METADATA_KEY_BYTES {
            return Err(KnowledgeError::Invalid(format!(
                "metadata keys must be 1 to {MAX_METADATA_KEY_BYTES} bytes"
            )));
        }
        if v.len() > MAX_METADATA_VALUE_BYTES {
            return Err(KnowledgeError::Invalid(format!(
                "metadata values must be at most {MAX_METADATA_VALUE_BYTES} bytes"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpisodeInput {
    pub id: String,
    pub speaker: String,
    pub occurred_at: Option<Time>,
    pub text: String,
}

/// A pointer to an exact span of an episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Citation {
    /// Stable reference, `keep:<evidence or chunk id>`.
    pub reference: String,
    pub source_id: String,
    pub source_title: String,
    pub episode_id: String,
    pub speaker: String,
    pub occurred_at: Option<Time>,
    pub char_start: i32,
    pub char_end: i32,
    pub quote: String,
    /// The source's host-defined metadata, such as a document URL.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub source_metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FactSummary {
    pub status: FactStatus,
    pub valid_from: Option<Time>,
    pub valid_to: Option<Time>,
    pub valid_from_basis: TimeBasis,
    pub valid_match: Option<ValidMatch>,
    pub version: i32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    pub item: ItemRef,
    pub rank: usize,
    pub score: f32,
    pub text: String,
    pub tokens: usize,
    pub source_id: String,
    pub occurred_at: Option<Time>,
    pub fact: Option<FactSummary>,
    pub citation: Citation,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    pub corpus: String,
    pub access: Access,
    pub query: String,
    pub as_of: AsOf,
    pub limit: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub hits: Vec<Hit>,
    pub used_tokens: usize,
    /// Results that matched but did not fit the evidence allowance.
    pub omitted: usize,
    /// Why results may be weaker than usual, e.g. `vector_unavailable`.
    pub degraded: Vec<String>,
    pub method: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fetched {
    Chunk {
        chunk: Chunk,
        citation: Citation,
    },
    Fact {
        fact: Fact,
        summary: FactSummary,
        citations: Vec<Citation>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FetchResult {
    pub item: Fetched,
    pub text: String,
    pub tokens: usize,
    /// Some evidence was left out to stay within the allowance.
    pub truncated: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub fact_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub object_entity: Option<String>,
    pub summary: FactSummary,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Neighbourhood {
    pub nodes: Vec<Entity>,
    pub edges: Vec<Edge>,
    pub truncated: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRun {
    pub succeeded: usize,
    pub retried: usize,
    pub failed: usize,
    /// Claimed but handed back unstarted at a deadline; still pending.
    #[serde(default)]
    pub deferred: usize,
}

/// Jobs one claim takes for [`Knowledge::run_source_jobs`].
pub const SOURCE_JOB_BATCH: usize = 16;
/// Jobs one claim takes for [`Knowledge::run_due_jobs`].
pub const BACKGROUND_JOB_BATCH: usize = 16;
/// Projections embedded and written together. Small, so a deadline is
/// checked often and one caller never holds the embedder for long.
pub const PROJECT_GROUP: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgetReport {
    pub episodes: i64,
    pub chunks: i64,
    pub facts_deleted: i64,
    pub facts_kept: i64,
    pub vectors_deleted: usize,
    /// Vector deletion failed and was queued for retry.
    pub vectors_deferred: usize,
}

pub struct Knowledge {
    ledger: Arc<dyn Ledger>,
    vectors: Option<(Arc<dyn VectorIndex>, String)>,
    embedder: Arc<dyn Embedder>,
    extractor: Arc<dyn FactExtractor>,
    /// When set, extraction jobs read each episode through this model-backed
    /// session extractor instead of `extractor`.
    session_extractor: Option<Arc<SessionExtractor>>,
    /// Extraction jobs run at once within one batch.
    extract_concurrency: usize,
    tokenizer: Arc<dyn Tokenizer>,
    clock: Arc<dyn Clock>,
    config: KnowledgeConfig,
    collection_ready: tokio::sync::OnceCell<()>,
    /// A cross-encoder and how many fused candidates it reorders.
    reranker: Option<(Arc<dyn Reranker>, usize)>,
    snapshots: Arc<Snapshots>,
    /// Sleep between background job batches; see [`Self::with_job_pause`].
    job_pause: Duration,
}

/// Vector namespace for one partition of one corpus. Opaque so labels never
/// reach the index backend.
/// A claim extracted outside the per-episode job queue (for example by a
/// model reading a whole session), with the turns that state it.
#[derive(Clone, Debug, PartialEq)]
pub struct ExternalClaim {
    pub claim: Claim,
    /// Fact text to show and index; defaults to "subject predicate object".
    pub text: Option<String>,
    /// Whether a new value for the same (subject, predicate) replaces the
    /// current one.
    pub single_valued: bool,
    /// Episodes (whole turns) cited as evidence, the anchor first.
    pub episodes: Vec<String>,
}

/// One evidence span to record for a planned claim.
struct Span {
    episode_id: String,
    /// Content hash of the episode when it was read.
    content_hash: String,
    source_id: String,
    start: i32,
    end: i32,
    quote: String,
}

/// A claim ready to plan.
struct Planned<'a> {
    claim: &'a Claim,
    spans: Vec<Span>,
    single_valued: bool,
    text: Option<String>,
    extractor: &'a str,
}

pub fn namespace(corpus: &str, partition: &str) -> String {
    digest_hex(&["ns", corpus, partition])[..32].to_string()
}

/// A source's title and metadata, as citations show them.
type Label = (String, BTreeMap<String, String>);

/// What one read may see: readable partitions and, when the caller passed
/// an allowlist, the sources records must derive from.
struct ReadScope {
    parts: Vec<String>,
    sources: Option<BTreeSet<String>>,
    list: Option<Vec<String>>,
    tags: Option<Vec<String>>,
}

impl ReadScope {
    /// Nothing can match: no readable partition, or an empty allowlist.
    fn is_empty(&self) -> bool {
        self.parts.is_empty() || self.sources.as_ref().is_some_and(BTreeSet::is_empty)
    }

    fn allows_source(&self, id: &str) -> bool {
        self.sources.as_ref().is_none_or(|s| s.contains(id))
    }

    /// The fact with only allowed evidence, or `None` if none is allowed.
    fn restrict<'a>(&self, f: &'a Fact) -> Option<std::borrow::Cow<'a, Fact>> {
        let Some(allowed) = &self.sources else {
            return Some(std::borrow::Cow::Borrowed(f));
        };
        if f.evidence.iter().all(|e| allowed.contains(&e.source_id)) {
            return Some(std::borrow::Cow::Borrowed(f));
        }
        let mut f = f.clone();
        f.evidence.retain(|e| allowed.contains(&e.source_id));
        (!f.evidence.is_empty()).then_some(std::borrow::Cow::Owned(f))
    }
}

/// The opaque vector tag for a source, so a vector search can be restricted
/// to items derived from given sources without storing their ids.
pub fn source_tag(corpus: &str, source_id: &str) -> String {
    digest_hex(&["src", corpus, source_id])[..32].to_string()
}

fn render_date(t: Option<Time>) -> String {
    t.map_or_else(|| "undated".into(), |t| t.date_naive().to_string())
}

fn summary_of(v: &FactVersion, m: Option<ValidMatch>) -> FactSummary {
    FactSummary {
        status: v.status,
        valid_from: v.valid_from,
        valid_to: v.valid_to,
        valid_from_basis: v.valid_from_basis,
        valid_match: m,
        version: v.version,
    }
}

fn render_fact(text: &str, s: &FactSummary) -> String {
    let from = match s.valid_from_basis {
        TimeBasis::Unknown => "unknown".to_string(),
        _ => render_date(s.valid_from),
    };
    let window = match s.valid_to {
        Some(to) => format!("{from} to {}", render_date(Some(to))),
        None => format!("since {from}"),
    };
    format!("Fact: {text} ({window}; {})", s.status.as_str())
}

pub(crate) fn render_chunk(c: &Chunk) -> String {
    format!("[{}] {}: {}", render_date(c.occurred_at), c.speaker, c.text)
}

/// The text a reranker judges: exactly what a hit would show.
fn rerank_text(snap: &Snapshot, i: u32, as_of: AsOf) -> String {
    match snap.chunks.get(i as usize) {
        Some(c) => render_chunk(c),
        None => {
            let f = &snap.facts[i as usize - snap.chunks.len()];
            match fact_as_of(f, as_of) {
                Some(view) => render_fact(&f.text, &summary_of(&view.version, view.valid)),
                None => f.text.clone(),
            }
        }
    }
}

fn charge(out: &mut SearchResult, allowance: &mut EvidenceAllowance, hit: Hit) {
    if allowance.try_charge(hit.tokens) {
        out.used_tokens += hit.tokens;
        out.hits.push(Hit {
            rank: out.hits.len() + 1,
            ..hit
        });
    } else {
        out.omitted += 1;
    }
}

impl Knowledge {
    /// Engine with the offline defaults: rule extractor, 256-d hash
    /// embedder, approximate tokenizer and the system clock.
    pub fn new(ledger: Arc<dyn Ledger>, vectors: Option<(Arc<dyn VectorIndex>, String)>) -> Self {
        Self {
            ledger,
            vectors,
            embedder: Arc::new(HashEmbedder::new(256)),
            extractor: Arc::new(RuleExtractor),
            session_extractor: None,
            extract_concurrency: 1,
            tokenizer: Arc::new(ApproxTokenizer),
            clock: Arc::new(SystemClock),
            config: KnowledgeConfig::default(),
            collection_ready: tokio::sync::OnceCell::new(),
            reranker: None,
            snapshots: Arc::default(),
            job_pause: Duration::ZERO,
        }
    }
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = embedder;
        self
    }
    pub fn with_extractor(mut self, extractor: Arc<dyn FactExtractor>) -> Self {
        self.extractor = extractor;
        self
    }
    /// Extract facts with a model: every episode is read through
    /// `extractor` (see [`episode_sessions`]) and its open-vocabulary facts
    /// cite the whole episode. Up to `concurrency` episodes are read at
    /// once. A model failure fails the job like any other, so it is retried
    /// or recorded as failed, never skipped.
    pub fn with_session_extractor(
        mut self,
        extractor: Arc<SessionExtractor>,
        concurrency: usize,
    ) -> Self {
        self.session_extractor = Some(extractor);
        self.extract_concurrency = concurrency.max(1);
        self
    }
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }
    pub fn with_config(mut self, config: KnowledgeConfig) -> Self {
        self.config = config;
        self
    }
    /// Reorder the first `top_n` fused candidates by `reranker`'s score.
    /// Only [`LexicalMode::Bm25`] search uses it.
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>, top_n: usize) -> Self {
        self.reranker = Some((reranker, top_n));
        self
    }

    /// Sleep `pause` after each group of projections and each batch that
    /// [`Self::run_due_jobs`] runs, so background work yields CPU to
    /// requests. Requests' own jobs never pause.
    pub fn with_job_pause(mut self, pause: Duration) -> Self {
        self.job_pause = pause;
        self
    }

    /// Another engine over the same ledger, vectors, embedder, extractor,
    /// tokenizer and config, with its own clock, so independent corpora can
    /// be ingested concurrently.
    pub fn sharing_storage(&self, clock: Arc<dyn Clock>) -> Self {
        Self {
            ledger: self.ledger.clone(),
            vectors: self.vectors.clone(),
            embedder: self.embedder.clone(),
            extractor: self.extractor.clone(),
            session_extractor: self.session_extractor.clone(),
            extract_concurrency: self.extract_concurrency,
            tokenizer: self.tokenizer.clone(),
            clock,
            config: self.config.clone(),
            collection_ready: tokio::sync::OnceCell::new(),
            reranker: self.reranker.clone(),
            snapshots: self.snapshots.clone(),
            job_pause: self.job_pause,
        }
    }

    pub fn ledger(&self) -> &Arc<dyn Ledger> {
        &self.ledger
    }
    pub fn vectors(&self) -> Option<&Arc<dyn VectorIndex>> {
        self.vectors.as_ref().map(|v| &v.0)
    }
    pub fn config(&self) -> &KnowledgeConfig {
        &self.config
    }
    pub fn now(&self) -> Time {
        self.clock.now()
    }

    /// Short description of the retrieval stack, for run artifacts.
    pub fn method(&self) -> String {
        let vec = match &self.vectors {
            Some((_, backend)) => format!("{backend}:{}", self.embedder.descriptor().model),
            None => "none".into(),
        };
        match self.config.lexical {
            LexicalMode::Ledger => format!(
                "hybrid(lexical={}, vector={vec}, fusion=rrf(k={}))",
                self.ledger.lexical_method(),
                self.config.rrf_k
            ),
            LexicalMode::Bm25(params) => {
                let lexical = Bm25Index::build(params, []).method();
                let rerank = match &self.reranker {
                    Some((r, n)) => {
                        let d = r.descriptor();
                        format!(
                            ", rerank={}:{}@{}(top={n})",
                            d.provider, d.model, d.revision
                        )
                    }
                    None => String::new(),
                };
                format!(
                    "hybrid(lexical={lexical};visible-first, vector={vec};visible-first, fusion=rrf(k={}), pool={}{rerank})",
                    self.config.rrf_k, self.config.candidate_pool
                )
            }
        }
    }

    // ----- ingestion -------------------------------------------------------

    /// Add episodes to a source, creating it if needed. Later batches of
    /// the same source append; re-sending stored episodes is a no-op.
    /// Different metadata, or different content under a stored episode id,
    /// is a [`KnowledgeError::Conflict`].
    pub async fn ingest(
        &self,
        corpus: &str,
        source: SourceInput,
        episodes: Vec<EpisodeInput>,
    ) -> Result<IngestReport> {
        self.ingest_inner(corpus, source, episodes, false).await
    }

    /// Ingest a new revision of a source. Episodes whose id and content are
    /// unchanged are kept with their chunks, vectors and facts; episodes
    /// missing from or changed in this batch are purged with everything
    /// derived only from them, and new ones inserted, in the same ledger
    /// transaction, so reads never see both revisions. Metadata is updated
    /// in place. Sending the current revision again is a no-op, and text
    /// that comes back within [`VECTOR_GRACE_SECONDS`] reuses its stored
    /// vector instead of being embedded again.
    pub async fn ingest_revision(
        &self,
        corpus: &str,
        source: SourceInput,
        episodes: Vec<EpisodeInput>,
    ) -> Result<IngestReport> {
        self.ingest_inner(corpus, source, episodes, true).await
    }

    async fn ingest_inner(
        &self,
        corpus: &str,
        source: SourceInput,
        episodes: Vec<EpisodeInput>,
        replace: bool,
    ) -> Result<IngestReport> {
        validate_metadata(&source.metadata)?;
        for (what, v) in [
            ("corpus", corpus),
            ("source id", &source.id),
            ("partition", &source.partition),
        ] {
            if v.trim().is_empty() {
                return Err(KnowledgeError::Invalid(format!("{what} is empty")));
            }
        }
        let now = self.clock.now();
        let chunking = timing::span("knowledge.chunk").items(episodes.len());
        let mut eps = Vec::new();
        let mut chunks = Vec::new();
        let mut seen = BTreeSet::new();
        for (i, e) in episodes.into_iter().enumerate() {
            if e.id.trim().is_empty() || !seen.insert(e.id.clone()) {
                return Err(KnowledgeError::Invalid(
                    "episode ids must be non-empty and unique".into(),
                ));
            }
            let when = e.occurred_at.map(|t| t.to_rfc3339()).unwrap_or_default();
            let hash = digest_hex(&[&e.speaker, &when, &e.text]);
            for (j, span) in chunk_text(&e.text, self.config.chunk_chars)
                .into_iter()
                .enumerate()
            {
                chunks.push(Chunk {
                    corpus: corpus.into(),
                    id: digest_id("chunk", &[corpus, &e.id, &j.to_string()]),
                    episode_id: e.id.clone(),
                    source_id: source.id.clone(),
                    partition: source.partition.clone(),
                    ordinal: j as i32,
                    char_start: span.start as i32,
                    char_end: span.end as i32,
                    speaker: e.speaker.clone(),
                    text: span.text,
                    occurred_at: e.occurred_at,
                    recorded_at: now,
                });
            }
            eps.push(Episode {
                corpus: corpus.into(),
                id: e.id,
                source_id: source.id.clone(),
                partition: source.partition.clone(),
                ordinal: i as i32,
                speaker: e.speaker,
                occurred_at: e.occurred_at,
                recorded_at: now,
                text: e.text,
                content_hash: hash,
            });
        }
        drop(chunking);
        let _t = timing::span("knowledge.ledger_ingest").items(chunks.len());
        let new_points: BTreeSet<String> = chunks.iter().map(|c| c.id.clone()).collect();
        let report = self
            .ledger
            .ingest(IngestBatch {
                source: Source {
                    corpus: corpus.into(),
                    id: source.id,
                    partition: source.partition,
                    collection: source.collection,
                    title: source.title,
                    kind: source.kind,
                    created_at: now,
                    metadata: source.metadata,
                },
                episodes: eps,
                chunks,
                now,
                replace,
            })
            .await?;
        if let Some(purged) = &report.purged {
            self.snapshots.forget(corpus);
            let stale: Vec<String> = purged
                .point_ids
                .iter()
                .filter(|id| !new_points.contains(*id))
                .cloned()
                .collect();
            self.clean_projection(corpus, purged, &stale, now).await?;
        }
        Ok(report)
    }

    /// After a purge: drop stale vector points (deferred to jobs if the
    /// index is down) and re-project facts that lost a supporting source.
    /// Returns (deleted, deferred) point counts.
    async fn clean_projection(
        &self,
        corpus: &str,
        purged: &PurgeReport,
        stale: &[String],
        now: Time,
    ) -> Result<(usize, usize)> {
        let mut counts = (0, 0);
        let Some((index, _)) = &self.vectors else {
            return Ok(counts);
        };
        if !stale.is_empty() {
            let ns = namespace(corpus, &purged.partition);
            match index
                .delete_points(&self.config.collection, &ns, stale)
                .await
            {
                Ok(()) => counts.0 = stale.len(),
                Err(_) => {
                    let jobs: Vec<NewJob> = stale
                        .iter()
                        .map(|id| NewJob {
                            kind: JobKind::Unproject,
                            target: format!("{ns}\u{1f}{id}"),
                        })
                        .collect();
                    self.ledger.enqueue(corpus, &jobs, now).await?;
                    counts.1 = jobs.len();
                }
            }
        }
        if !purged.kept_fact_ids.is_empty() {
            let jobs: Vec<NewJob> = purged
                .kept_fact_ids
                .iter()
                .map(|id| NewJob {
                    kind: JobKind::Project,
                    target: id.clone(),
                })
                .collect();
            self.ledger.enqueue(corpus, &jobs, now).await?;
        }
        Ok(counts)
    }

    /// Run due jobs until none are left (or `max_rounds` batches ran).
    pub async fn run_jobs(&self, corpus: &str, max_rounds: usize) -> Result<JobRun> {
        self.run_jobs_with_budget(corpus, max_rounds, 64, None)
            .await
    }

    /// Run due jobs in batches of up to `batch_size`, stopping between rounds
    /// once `budget` has elapsed. The first round always runs.
    pub async fn run_jobs_with_budget(
        &self,
        corpus: &str,
        max_rounds: usize,
        batch_size: usize,
        budget: Option<std::time::Duration>,
    ) -> Result<JobRun> {
        self.run_rounds(corpus, max_rounds, batch_size, budget, Duration::ZERO)
            .await
    }

    async fn run_rounds(
        &self,
        corpus: &str,
        max_rounds: usize,
        batch_size: usize,
        budget: Option<std::time::Duration>,
        pause: Duration,
    ) -> Result<JobRun> {
        let mut run = JobRun::default();
        let batch_size = batch_size.max(1);
        let started = Instant::now();
        for round in 0..max_rounds {
            if round > 0 && budget.is_some_and(|b| started.elapsed() >= b) {
                break;
            }
            let t = timing::span("knowledge.jobs_claim");
            let jobs = self
                .ledger
                .claim_jobs(corpus, self.clock.now(), batch_size)
                .await?;
            drop(t.items(jobs.len()));
            if jobs.is_empty() {
                break;
            }
            self.run_batch(corpus, jobs, None, pause, &mut run).await?;
        }
        Ok(run)
    }

    /// Run due jobs derived from one source (see
    /// [`Ledger::claim_source_jobs`]) until `deadline`, for a caller that
    /// must answer in time. Other sources' jobs are left to the background
    /// worker. The deadline is checked before each job, and before each
    /// group of [`PROJECT_GROUP`] projections, so the overrun is at most one
    /// job or group; claimed jobs not started by then are handed back
    /// unstarted and counted as [`JobRun::deferred`].
    pub async fn run_source_jobs(
        &self,
        corpus: &str,
        source_id: &str,
        deadline: Instant,
    ) -> Result<JobRun> {
        let mut run = JobRun::default();
        while Instant::now() < deadline {
            let t = timing::span("knowledge.jobs_claim");
            let jobs = self
                .ledger
                .claim_source_jobs(corpus, source_id, self.clock.now(), SOURCE_JOB_BATCH)
                .await?;
            drop(t.items(jobs.len()));
            if jobs.is_empty() {
                break;
            }
            self.run_batch(corpus, jobs, Some(deadline), Duration::ZERO, &mut run)
                .await?;
        }
        Ok(run)
    }

    /// Run claimed jobs, recording their outcomes. Jobs not started before
    /// `deadline` are released. `pause` sleeps after each projection group,
    /// so a background caller leaves CPU for requests.
    async fn run_batch(
        &self,
        corpus: &str,
        jobs: Vec<Job>,
        deadline: Option<Instant>,
        pause: Duration,
        run: &mut JobRun,
    ) -> Result<()> {
        let past = move || deadline.is_some_and(|d| Instant::now() >= d);
        let (project, other): (Vec<Job>, Vec<Job>) =
            jobs.into_iter().partition(|j| j.kind == JobKind::Project);
        let mut released = Vec::new();
        let ran: Vec<(Job, Option<JobOutcome>)> = stream::iter(other)
            .map(|job| async move {
                if past() {
                    return (job, None);
                }
                let _t = timing::span(match job.kind {
                    JobKind::Extract => "knowledge.extract_episode",
                    _ => "knowledge.unproject",
                })
                .items(1);
                let r = match job.kind {
                    JobKind::Extract => self.extract_episode(corpus, &job.target).await,
                    _ => self.unproject(&job.target).await,
                };
                (job, Some(r.map_err(|e| (e.to_string(), retryable(&e)))))
            })
            .buffered(self.extract_concurrency)
            .collect()
            .await;
        let mut outcomes: Vec<(Job, JobOutcome)> = Vec::new();
        for (job, r) in ran {
            match r {
                Some(r) => outcomes.push((job, r)),
                None => released.push(job),
            }
        }
        let groups: Vec<&[Job]> = project.chunks(PROJECT_GROUP).collect();
        for (i, group) in groups.iter().enumerate() {
            if past() {
                released.extend(group.iter().cloned());
                continue;
            }
            match self.project(corpus, group).await {
                Ok(()) => outcomes.extend(group.iter().map(|j| (j.clone(), Ok(())))),
                Err(e) => {
                    let msg = (e.to_string(), retryable(&e));
                    outcomes.extend(group.iter().map(|j| (j.clone(), Err(msg.clone()))));
                }
            }
            if !pause.is_zero() && i + 1 < groups.len() {
                tokio::time::sleep(pause).await;
            }
        }
        let _t = timing::span("knowledge.jobs_finish").items(outcomes.len());
        let mut done = Vec::new();
        for (job, r) in outcomes {
            match r {
                Ok(()) => done.push(job),
                Err((msg, retry)) => {
                    let now = self.clock.now();
                    match self.ledger.fail_job(&job, &msg, retry, now).await? {
                        JobState::RetryScheduled => run.retried += 1,
                        _ => run.failed += 1,
                    }
                }
            }
        }
        let completed = self.ledger.complete_jobs(&done, self.clock.now()).await?;
        run.succeeded += completed.into_iter().filter(|&ok| ok).count();
        if !released.is_empty() {
            self.ledger
                .release_jobs(&released, self.clock.now())
                .await?;
            run.deferred += released.len();
        }
        if !pause.is_zero() {
            tokio::time::sleep(pause).await;
        }
        Ok(())
    }

    /// Background jobs of one partition by state. See [`Ledger::job_counts`].
    pub async fn job_counts(&self, corpus: &str, partition: &str) -> Result<JobCounts> {
        self.ledger
            .job_counts(corpus, partition, &namespace(corpus, partition), true)
            .await
    }

    /// Like [`Self::job_counts`] without finished jobs, whose `succeeded` is 0.
    /// Cheap enough for every write, however large the corpus has grown.
    pub async fn unfinished_job_counts(&self, corpus: &str, partition: &str) -> Result<JobCounts> {
        self.ledger
            .job_counts(corpus, partition, &namespace(corpus, partition), false)
            .await
    }

    /// Check the ledger answers. For a readiness probe.
    pub async fn probe_ledger(&self) -> Result<()> {
        self.ledger.probe(self.clock.now()).await
    }

    /// Check the vector store answers and holds the collection with the
    /// embedder's dimensions. `None` without a vector store.
    pub async fn probe_vectors(&self) -> Option<Result<()>> {
        let (index, _) = self.vectors.as_ref()?;
        Some(
            index
                .ensure_collection(&self.config.collection, self.embedder.dimensions())
                .await
                .map_err(KnowledgeError::from),
        )
    }

    /// Run due jobs of every corpus that has some, up to `max_corpora`
    /// corpora and `max_rounds` batches of [`BACKGROUND_JOB_BATCH`] each,
    /// pausing as set by [`Self::with_job_pause`]. For a background worker,
    /// so retries, deferred vector deletions and work an ingest left behind
    /// finish without a new request. Also sweeps stored vectors whose grace
    /// period has passed.
    pub async fn run_due_jobs(&self, max_corpora: usize, max_rounds: usize) -> Result<JobRun> {
        let now = self.clock.now();
        self.ledger
            .sweep_vectors(now - chrono::Duration::seconds(VECTOR_GRACE_SECONDS))
            .await?;
        let mut total = JobRun::default();
        for corpus in self.ledger.due_corpora(now, max_corpora).await? {
            let run = self
                .run_rounds(
                    &corpus,
                    max_rounds,
                    BACKGROUND_JOB_BATCH,
                    None,
                    self.job_pause,
                )
                .await?;
            total.succeeded += run.succeeded;
            total.retried += run.retried;
            total.failed += run.failed;
        }
        Ok(total)
    }

    async fn ensure_collection(&self) -> Result<()> {
        let Some((index, _)) = &self.vectors else {
            return Ok(());
        };
        self.collection_ready
            .get_or_try_init(|| async {
                index
                    .ensure_collection(&self.config.collection, self.embedder.dimensions())
                    .await
                    .map_err(KnowledgeError::from)
            })
            .await?;
        Ok(())
    }

    async fn project(&self, corpus: &str, jobs: &[Job]) -> Result<()> {
        let Some((index, _)) = &self.vectors else {
            return Ok(());
        };
        let t = timing::span("knowledge.projection_read").items(jobs.len());
        let targets: Vec<String> = jobs.iter().map(|j| j.target.clone()).collect();
        let mut items: Vec<(String, Projection)> = targets
            .iter()
            .cloned()
            .zip(self.ledger.projections(corpus, &targets).await?)
            .filter_map(|(id, p)| p.map(|p| (id, p)))
            .collect();
        drop(t);
        for _ in 0..3 {
            if items.is_empty() {
                return Ok(());
            }
            self.ensure_collection().await?;
            let vectors = self.vectors_for(corpus, &items).await?;
            let mut by_ns: BTreeMap<String, Vec<VectorPoint>> = BTreeMap::new();
            for ((id, p), vector) in items.iter().zip(vectors) {
                by_ns
                    .entry(namespace(corpus, &p.partition))
                    .or_default()
                    .push(VectorPoint {
                        id: id.clone(),
                        vector,
                        tags: p.sources.iter().map(|x| source_tag(corpus, x)).collect(),
                    });
            }
            for (ns, points) in &by_ns {
                let _t = timing::span("knowledge.vector_upsert").items(points.len());
                index.upsert(&self.config.collection, ns, points).await?;
            }
            // A purge may have run while we embedded; never leave its
            // vectors. A replacement may have changed an item's text or
            // sources; embed it again so a stale vector never wins.
            let _t = timing::span("knowledge.projection_recheck").items(items.len());
            let ids: Vec<String> = items.iter().map(|(id, _)| id.clone()).collect();
            let still = self.ledger.projections(corpus, &ids).await?;
            let mut changed = Vec::new();
            for ((id, p), now) in items.iter().zip(still) {
                match now {
                    None => {
                        index
                            .delete_points(
                                &self.config.collection,
                                &namespace(corpus, &p.partition),
                                std::slice::from_ref(id),
                            )
                            .await?;
                    }
                    Some(now) if now != *p => changed.push((id.clone(), now)),
                    Some(_) => {}
                }
            }
            items = changed;
        }
        if items.is_empty() {
            return Ok(());
        }
        Err(KnowledgeError::Conflict(
            "items kept changing while projecting".into(),
        ))
    }

    /// What a stored vector of `text` is keyed by: the embedder and the
    /// exact text, so another model or changed text never reuses it.
    fn vector_digest(&self, text: &str) -> String {
        let d = self.embedder.descriptor();
        digest_hex(&[
            "vector",
            &d.provider,
            &d.model,
            &d.revision,
            &self.embedder.dimensions().to_string(),
            text,
        ])
    }

    /// Vectors for items to project. Chunks whose text the ledger already
    /// holds a vector for reuse it; the rest are embedded, each distinct
    /// text once, and chunks' new vectors are stored for next time.
    async fn vectors_for(
        &self,
        corpus: &str,
        items: &[(String, Projection)],
    ) -> Result<Vec<Vec<f32>>> {
        let keys: Vec<(String, String)> = items
            .iter()
            .map(|(id, p)| (id.clone(), self.vector_digest(&p.text)))
            .collect();
        let t = timing::span("knowledge.vectors_read").items(keys.len());
        let mut found = self.ledger.stored_vectors(corpus, &keys).await?;
        drop(t);
        let mut texts: Vec<String> = Vec::new();
        let mut slot: HashMap<&str, usize> = HashMap::new();
        for ((_, p), v) in items.iter().zip(&found) {
            if v.is_none() && !slot.contains_key(p.text.as_str()) {
                slot.insert(&p.text, texts.len());
                texts.push(p.text.clone());
            }
        }
        if texts.is_empty() {
            return Ok(found.into_iter().map(Option::unwrap_or_default).collect());
        }
        let t = timing::span("knowledge.embed").items(texts.len());
        let embedded = self.embedder.embed(&texts).await?.vectors;
        drop(t);
        if embedded.len() != texts.len() {
            return Err(KnowledgeError::Provider(
                keep_providers::ProviderError::Malformed {
                    message: format!(
                        "embedder returned {} vectors for {} texts",
                        embedded.len(),
                        texts.len()
                    ),
                },
            ));
        }
        let mut fresh = Vec::new();
        for (((id, p), (_, digest)), v) in items.iter().zip(&keys).zip(found.iter_mut()) {
            if v.is_none() {
                let vector = embedded[slot[p.text.as_str()]].clone();
                fresh.push(StoredVector {
                    point_id: id.clone(),
                    digest: digest.clone(),
                    vector: vector.clone(),
                });
                *v = Some(vector);
            }
        }
        let _t = timing::span("knowledge.vectors_store").items(fresh.len());
        self.ledger
            .store_vectors(corpus, &fresh, self.clock.now())
            .await?;
        Ok(found.into_iter().map(Option::unwrap_or_default).collect())
    }

    async fn unproject(&self, target: &str) -> Result<()> {
        let Some((index, _)) = &self.vectors else {
            return Ok(());
        };
        let (ns, id) = target
            .split_once('\u{1f}')
            .ok_or_else(|| KnowledgeError::Invalid("malformed unproject target".into()))?;
        index
            .delete_points(&self.config.collection, ns, &[id.to_string()])
            .await?;
        Ok(())
    }

    async fn extract_episode(&self, corpus: &str, episode_id: &str) -> Result<()> {
        let Some(ep) = self.ledger.episode(corpus, episode_id).await? else {
            return Ok(());
        };
        if let Some(sx) = &self.session_extractor {
            let _t = timing::span("knowledge.extract_model").items(1);
            let mut facts = Vec::new();
            for session in episode_sessions(&ep) {
                facts.extend(sx.extract(&session).await?.facts);
            }
            let claims = claims_for(&facts);
            if !claims.is_empty() {
                let seen = HashMap::from([(ep.id.clone(), ep.content_hash.clone())]);
                self.apply_claims_seen(corpus, &sx.evidence_label(), &claims, &seen)
                    .await?;
            }
            return Ok(());
        }
        let t = timing::span("knowledge.extract_rules").items(1);
        let claims = validate(&ep, self.extractor.extract(&ep).await?);
        drop(t);
        if claims.is_empty() {
            return Ok(());
        }
        for _ in 0..3 {
            let t = timing::span("knowledge.plan_changes").items(claims.len());
            let change = self.plan_changes(&ep, &claims).await?;
            drop(t);
            if change.is_empty() {
                return Ok(());
            }
            let _t = timing::span("knowledge.ledger_apply").items(change.new_facts.len());
            match self.ledger.apply(change).await? {
                ApplyOutcome::Applied | ApplyOutcome::EpisodeGone => return Ok(()),
                ApplyOutcome::Conflict => continue,
            }
        }
        Err(KnowledgeError::Conflict(
            "facts kept changing while extracting".into(),
        ))
    }

    async fn plan_changes(&self, ep: &Episode, claims: &[CheckedClaim]) -> Result<ChangeSet> {
        let planned: Vec<Planned> = claims
            .iter()
            .map(|c| Planned {
                claim: &c.claim,
                spans: vec![Span {
                    episode_id: ep.id.clone(),
                    content_hash: ep.content_hash.clone(),
                    source_id: ep.source_id.clone(),
                    start: c.start as i32,
                    end: c.end as i32,
                    quote: c.claim.quote.clone(),
                }],
                single_valued: is_single_valued(&c.claim.predicate),
                text: None,
                extractor: self.extractor.id(),
            })
            .collect();
        self.plan_planned(ep, &planned).await
    }

    /// Apply claims extracted outside the per-episode job queue, such as a
    /// model reading a whole session. Each claim cites whole turns by
    /// episode id; claims are planned against the episode they cite first,
    /// in the given order, so a later claim can supersede an earlier one.
    /// `single_valued` replaces the fixed predicate list: a new value for
    /// the same (subject, predicate) closes the old one's window. Returns
    /// the number of claims applied; claims citing no known episode are
    /// skipped.
    pub async fn apply_claims(
        &self,
        corpus: &str,
        extractor: &str,
        claims: &[ExternalClaim],
    ) -> Result<usize> {
        self.apply_claims_seen(corpus, extractor, claims, &HashMap::new())
            .await
    }

    /// [`Self::apply_claims`] for claims extracted from episodes with the
    /// given content hashes: claims citing an episode whose content has
    /// changed since are skipped, and the write is refused if one changes
    /// before it commits.
    async fn apply_claims_seen(
        &self,
        corpus: &str,
        extractor: &str,
        claims: &[ExternalClaim],
        seen: &HashMap<String, String>,
    ) -> Result<usize> {
        let mut episodes: HashMap<String, Option<Episode>> = HashMap::new();
        let mut changed: BTreeSet<String> = BTreeSet::new();
        for c in claims {
            for id in &c.episodes {
                if !episodes.contains_key(id) {
                    let ep = self.ledger.episode(corpus, id).await?;
                    if let (Some(e), Some(h)) = (&ep, seen.get(id))
                        && e.content_hash != *h
                    {
                        changed.insert(id.clone());
                    }
                    episodes.insert(id.clone(), ep);
                }
            }
        }
        // Group by anchor episode, keeping first-seen order.
        let mut groups: Vec<(Episode, Vec<Planned>)> = Vec::new();
        for c in claims {
            if c.episodes.iter().any(|id| changed.contains(id)) {
                continue;
            }
            let found: Vec<&Episode> = c
                .episodes
                .iter()
                .filter_map(|id| episodes[id].as_ref())
                .collect();
            let Some(anchor) = found.first() else {
                continue;
            };
            let spans = found
                .iter()
                .map(|ep| Span {
                    episode_id: ep.id.clone(),
                    content_hash: ep.content_hash.clone(),
                    source_id: ep.source_id.clone(),
                    start: 0,
                    end: ep.text.chars().count() as i32,
                    quote: ep.text.clone(),
                })
                .collect();
            let planned = Planned {
                claim: &c.claim,
                spans,
                single_valued: c.single_valued,
                text: c.text.clone(),
                extractor,
            };
            match groups.iter_mut().find(|(ep, _)| ep.id == anchor.id) {
                Some((_, g)) => g.push(planned),
                None => groups.push(((*anchor).clone(), vec![planned])),
            }
        }
        let mut applied = 0;
        for (ep, planned) in &groups {
            let mut done = false;
            for _ in 0..3 {
                let change = self.plan_planned(ep, planned).await?;
                if change.is_empty() {
                    done = true;
                    break;
                }
                match self.ledger.apply(change).await? {
                    ApplyOutcome::Applied | ApplyOutcome::EpisodeGone => {
                        done = true;
                        break;
                    }
                    ApplyOutcome::Conflict => continue,
                }
            }
            if !done {
                return Err(KnowledgeError::Conflict(
                    "facts kept changing while applying claims".into(),
                ));
            }
            applied += planned.len();
        }
        Ok(applied)
    }

    async fn plan_planned(&self, ep: &Episode, claims: &[Planned<'_>]) -> Result<ChangeSet> {
        let corpus = ep.corpus.as_str();
        let part = ep.partition.as_str();
        let now = self.clock.now();
        let mut cs = ChangeSet {
            corpus: corpus.into(),
            partition: part.into(),
            episode_id: ep.id.clone(),
            episodes: vec![(ep.id.clone(), ep.content_hash.clone())],
            now,
            entities: vec![],
            aliases: vec![],
            new_facts: vec![],
            evidence: vec![],
            versions: vec![],
            jobs: vec![],
        };
        for sp in claims.iter().flat_map(|c| &c.spans) {
            if !cs.episodes.iter().any(|(id, _)| *id == sp.episode_id) {
                cs.episodes
                    .push((sp.episode_id.clone(), sp.content_hash.clone()));
            }
        }
        // normalised name -> (entity id, display name)
        let mut names: HashMap<String, (String, String)> = HashMap::new();
        // Facts touched in this change set, by id, with their latest version.
        let mut touched: HashMap<String, Fact> = HashMap::new();

        for c in claims {
            let claim = c.claim;
            let (subject, subject_name) = self
                .resolve_entity(
                    &mut cs,
                    &mut names,
                    part,
                    &claim.subject,
                    &claim.subject_kind,
                )
                .await?;
            if claim.predicate == ALIAS {
                if normalize_name(&claim.object) != normalize_name(&subject_name) {
                    cs.aliases.push(AliasWrite {
                        entity_id: subject.clone(),
                        alias: claim.object.clone(),
                        source_id: ep.source_id.clone(),
                    });
                    names.insert(
                        normalize_name(&claim.object),
                        (subject.clone(), subject_name.clone()),
                    );
                }
                continue;
            }
            let (object_entity, object_name) = match &claim.object_kind {
                Some(kind) => {
                    let (id, name) = self
                        .resolve_entity(&mut cs, &mut names, part, &claim.object, kind)
                        .await?;
                    (Some(id), name)
                }
                None => (None, claim.object.clone()),
            };
            let object_key = object_entity
                .clone()
                .unwrap_or_else(|| normalize_name(&object_name));
            let text = c.text.clone().unwrap_or_else(|| {
                format!(
                    "{subject_name} {} {object_name}",
                    claim.predicate.replace('_', " ")
                )
            });

            // The same statement may hold for several separate windows
            // (moved away, moved back); each window is its own fact.
            let mut chosen: Option<(String, Option<Fact>)> = None;
            for n in 0..8 {
                let key = if n == 0 {
                    object_key.clone()
                } else {
                    format!("{object_key}#{n}")
                };
                let id = Fact::id_for(corpus, part, &subject, &claim.predicate, &key);
                let existing = match touched.get(&id) {
                    Some(f) => Some(f.clone()),
                    None => self
                        .ledger
                        .facts(corpus, std::slice::from_ref(&id))
                        .await?
                        .pop(),
                };
                match existing {
                    None => {
                        chosen = Some((id, None));
                        break;
                    }
                    Some(f) => {
                        let closed_before = f.current().is_some_and(|v| {
                            v.status == FactStatus::Superseded
                                && v.valid_to
                                    .zip(claim.valid_from)
                                    .is_some_and(|(to, from)| to <= from)
                        });
                        if !closed_before {
                            chosen = Some((id, Some(f)));
                            break;
                        }
                    }
                }
            }
            let Some((fact_id, existing)) = chosen else {
                continue;
            };
            let mut evidence: Vec<Evidence> = Vec::with_capacity(c.spans.len());
            for sp in &c.spans {
                let id = Evidence::id_for(corpus, &fact_id, &sp.episode_id, sp.start, sp.end);
                if evidence.iter().any(|e| e.id == id) {
                    continue;
                }
                evidence.push(Evidence {
                    corpus: corpus.into(),
                    id,
                    fact_id: fact_id.clone(),
                    episode_id: sp.episode_id.clone(),
                    source_id: sp.source_id.clone(),
                    partition: part.into(),
                    char_start: sp.start,
                    char_end: sp.end,
                    quote: sp.quote.clone(),
                    extractor: c.extractor.into(),
                    recorded_at: now,
                });
            }
            if let Some(f) = existing {
                // A new supporting source changes the fact's vector tags.
                let new_source = evidence
                    .iter()
                    .any(|e| !f.evidence.iter().any(|x| x.source_id == e.source_id));
                if new_source
                    && !cs
                        .jobs
                        .iter()
                        .any(|j| j.kind == JobKind::Project && j.target == fact_id)
                {
                    cs.jobs.push(NewJob {
                        kind: JobKind::Project,
                        target: fact_id.clone(),
                    });
                }
                for evidence in evidence {
                    if f.evidence.iter().any(|e| e.id == evidence.id) {
                        continue;
                    }
                    if let Some(nf) = cs.new_facts.iter_mut().find(|x| x.id == fact_id) {
                        nf.evidence.push(evidence);
                    } else if !cs.evidence.iter().any(|e| e.id == evidence.id) {
                        cs.evidence.push(evidence);
                    }
                }
                continue;
            }

            let mut rivals = Vec::new();
            if c.single_valued {
                let mut current: BTreeMap<String, Fact> = self
                    .ledger
                    .current_facts(corpus, part, &subject, &claim.predicate)
                    .await?
                    .into_iter()
                    .map(|f| (f.id.clone(), f))
                    .collect();
                for f in touched.values() {
                    if f.subject == subject && f.predicate == claim.predicate {
                        current.insert(f.id.clone(), f.clone());
                    }
                }
                for f in current.values() {
                    if let Some(v) = f.current() {
                        rivals.push(Rival {
                            fact_id: f.id.clone(),
                            current: v.clone(),
                        });
                    }
                }
            }
            let p = plan(
                &Incoming {
                    fact_id: fact_id.clone(),
                    valid_from: claim.valid_from,
                    basis: claim.basis,
                },
                &rivals,
                now,
            );
            let fact = Fact {
                corpus: corpus.into(),
                id: fact_id.clone(),
                partition: part.into(),
                subject: subject.clone(),
                predicate: claim.predicate.clone(),
                object: object_name,
                object_entity,
                text,
                versions: vec![p.incoming],
                evidence,
            };
            for (rid, expected, v) in p.rival_versions {
                if let Some(nf) = cs.new_facts.iter_mut().find(|x| x.id == rid) {
                    // Created in this same change set: rewrite in place, the
                    // intermediate state was never committed.
                    let first = &mut nf.versions[0];
                    first.valid_to = v.valid_to;
                    first.status = v.status;
                    first.cause = v.cause.clone();
                    touched.insert(rid.clone(), nf.clone());
                } else if let Some(w) = cs.versions.iter_mut().find(|w| w.fact_id == rid) {
                    w.version = FactVersion {
                        version: w.version.version,
                        ..v
                    };
                    if let Some(t) = touched.get_mut(&rid) {
                        *t.versions.last_mut().expect("version") = w.version.clone();
                    }
                } else {
                    let mut f = match touched.get(&rid) {
                        Some(f) => f.clone(),
                        None => self
                            .ledger
                            .facts(corpus, std::slice::from_ref(&rid))
                            .await?
                            .pop()
                            .ok_or_else(|| KnowledgeError::Conflict("rival vanished".into()))?,
                    };
                    if let Some(cur) = f.versions.last_mut() {
                        cur.invalidated_at = Some(now);
                    }
                    f.versions.push(v.clone());
                    touched.insert(rid.clone(), f);
                    cs.versions.push(VersionWrite {
                        fact_id: rid,
                        expected_current: expected,
                        version: v,
                    });
                }
            }
            cs.jobs.push(NewJob {
                kind: JobKind::Project,
                target: fact_id.clone(),
            });
            touched.insert(fact_id, fact.clone());
            cs.new_facts.push(fact);
        }
        Ok(cs)
    }

    async fn resolve_entity(
        &self,
        cs: &mut ChangeSet,
        names: &mut HashMap<String, (String, String)>,
        partition: &str,
        name: &str,
        kind: &str,
    ) -> Result<(String, String)> {
        let norm = normalize_name(name);
        if let Some(hit) = names.get(&norm) {
            return Ok(hit.clone());
        }
        let found = match self
            .ledger
            .find_entity(&cs.corpus, partition, &norm)
            .await?
        {
            Some(e) => (e.id, e.name),
            None => {
                let id = digest_id("ent", &[&cs.corpus, partition, &norm]);
                cs.entities.push(EntityWrite {
                    id: id.clone(),
                    name: name.trim().to_string(),
                    kind: kind.into(),
                });
                (id, name.trim().to_string())
            }
        };
        names.insert(norm, found.clone());
        Ok(found)
    }

    // ----- reads -----------------------------------------------------------

    /// Partitions in the corpus the caller may read. Access fails closed.
    async fn readable(&self, corpus: &str, access: &Access) -> Result<Vec<String>> {
        Ok(self
            .ledger
            .partitions(corpus)
            .await?
            .into_iter()
            .map(|(p, _)| p)
            .filter(|p| access.allows(p))
            .collect())
    }

    /// Readable partitions plus the source allowlist, resolved once per read.
    async fn scope(&self, corpus: &str, access: &Access) -> Result<ReadScope> {
        let parts = self.readable(corpus, access).await?;
        let sources = match &access.collections {
            Some(cols) => {
                let cols: Vec<String> = cols.iter().cloned().collect();
                let in_cols = if parts.is_empty() || cols.is_empty() {
                    Vec::new()
                } else {
                    self.ledger
                        .sources_in_collections(corpus, &parts, &cols)
                        .await?
                };
                Some(
                    in_cols
                        .into_iter()
                        .filter(|id| access.sources.as_ref().is_none_or(|s| s.contains(id)))
                        .collect(),
                )
            }
            None => access.sources.clone(),
        };
        Ok(ReadScope {
            tags: sources
                .as_ref()
                .map(|s| s.iter().map(|id| source_tag(corpus, id)).collect()),
            list: sources.as_ref().map(|s| s.iter().cloned().collect()),
            sources,
            parts,
        })
    }

    /// `f` as this read sees it: only allowed evidence, and versions
    /// rebuilt without facts the read may not see. See
    /// [`crate::resolve::restrict`].
    async fn restricted(&self, scope: &ReadScope, f: &Fact) -> Result<Option<Fact>> {
        let Some(allowed) = &scope.sources else {
            return Ok(Some(f.clone()));
        };
        if !f.versions.iter().any(|v| v.cause.is_some()) {
            return Ok(scope.restrict(f).map(std::borrow::Cow::into_owned));
        }
        let group = self
            .ledger
            .current_facts(&f.corpus, &f.partition, &f.subject, &f.predicate)
            .await?;
        let group: Vec<&Fact> = group.iter().collect();
        Ok(crate::resolve::restrict(f, allowed, &group))
    }

    async fn citation_for_chunk(&self, c: &Chunk) -> Result<Citation> {
        let label = self.source_label(&c.corpus, &c.source_id).await?;
        Ok(chunk_citation(c, label))
    }

    async fn source_label(&self, corpus: &str, source_id: &str) -> Result<Label> {
        Ok(self
            .ledger
            .source(corpus, source_id)
            .await?
            .map(|s| (s.title, s.metadata))
            .unwrap_or_default())
    }

    async fn citation_for_evidence(&self, e: &Evidence) -> Result<Option<Citation>> {
        let Some(ep) = self.ledger.episode(&e.corpus, &e.episode_id).await? else {
            return Ok(None);
        };
        let (title, metadata) = self.source_label(&e.corpus, &e.source_id).await?;
        Ok(Some(Citation {
            reference: format!("keep:{}", e.id),
            source_id: e.source_id.clone(),
            source_title: title,
            episode_id: e.episode_id.clone(),
            speaker: ep.speaker,
            occurred_at: ep.occurred_at,
            char_start: e.char_start,
            char_end: e.char_end,
            quote: e.quote.clone(),
            source_metadata: metadata,
        }))
    }

    /// Hybrid search. Permission filtering happens before ranking: each
    /// retriever only ever sees readable partitions, and every candidate is
    /// re-checked against the ledger before it is shown.
    pub async fn search(
        &self,
        req: &SearchRequest,
        allowance: &mut EvidenceAllowance,
    ) -> Result<SearchResult> {
        let _t = timing::span("knowledge.search").items(1);
        let mut out = SearchResult {
            method: self.method(),
            ..Default::default()
        };
        let scope = self.scope(&req.corpus, &req.access).await?;
        if scope.is_empty() || req.query.trim().is_empty() {
            return Ok(out);
        }
        let parts = &scope.parts;
        let limit = if req.limit == 0 { 10 } else { req.limit };
        let pool = self.config.candidate_pool.max(limit * 4);
        if let LexicalMode::Bm25(params) = self.config.lexical {
            return self
                .search_visible(req, allowance, params, &scope, limit, pool, out)
                .await;
        }

        let lexical: Vec<String> = self
            .ledger
            .lexical(&req.corpus, parts, &req.query, pool, scope.list.as_deref())
            .await?
            .into_iter()
            .map(|h| h.id)
            .collect();
        let vector = match self
            .vector_candidates(&req.corpus, &scope, &req.query, pool)
            .await
        {
            Ok(v) => v,
            Err(_) => {
                out.degraded.push("vector_unavailable".into());
                Vec::new()
            }
        };
        if self.vectors.is_none() {
            out.degraded.push("vector_disabled".into());
        }
        let fused = rrf(&[lexical, vector], self.config.rrf_k);
        let fused: Vec<(ItemRef, f32)> = fused
            .into_iter()
            .take(pool * 2)
            .filter_map(|(id, s)| ItemRef::from_point_id(&id).map(|r| (r, s)))
            .collect();

        let chunk_ids: Vec<String> = fused
            .iter()
            .filter_map(|(r, _)| match r {
                ItemRef::Chunk(id) => Some(id.clone()),
                _ => None,
            })
            .collect();
        let fact_ids: Vec<String> = fused
            .iter()
            .filter_map(|(r, _)| match r {
                ItemRef::Fact(id) => Some(id.clone()),
                _ => None,
            })
            .collect();
        let chunks: HashMap<String, Chunk> = self
            .ledger
            .chunks(&req.corpus, &chunk_ids)
            .await?
            .into_iter()
            .map(|c| (c.id.clone(), c))
            .collect();
        let facts: HashMap<String, Fact> = self
            .ledger
            .facts(&req.corpus, &fact_ids)
            .await?
            .into_iter()
            .map(|f| (f.id.clone(), f))
            .collect();

        for (item, score) in fused {
            if out.hits.len() >= limit {
                break;
            }
            let hit = match &item {
                ItemRef::Chunk(id) => {
                    let Some(c) = chunks.get(id) else { continue };
                    if !req.access.allows(&c.partition)
                        || !scope.allows_source(&c.source_id)
                        || !chunk_visible(c, req.as_of)
                    {
                        continue;
                    }
                    self.chunk_hit(&item, score, c).await?
                }
                ItemRef::Fact(id) => {
                    let Some(f) = facts.get(id) else { continue };
                    if !req.access.allows(&f.partition) {
                        continue;
                    }
                    let Some(f) = self.restricted(&scope, f).await? else {
                        continue;
                    };
                    let Some(hit) = self.fact_hit(&item, score, &f, req.as_of).await? else {
                        continue;
                    };
                    hit
                }
            };
            charge(&mut out, allowance, hit);
        }
        Ok(out)
    }

    async fn chunk_hit(&self, item: &ItemRef, score: f32, c: &Chunk) -> Result<Hit> {
        let label = self.source_label(&c.corpus, &c.source_id).await?;
        Ok(chunk_hit_titled(&*self.tokenizer, item, score, c, label))
    }

    /// `None` when the fact is not visible at `as_of` or its first
    /// evidence is gone.
    async fn fact_hit(
        &self,
        item: &ItemRef,
        score: f32,
        f: &Fact,
        as_of: AsOf,
    ) -> Result<Option<Hit>> {
        let Some(view) = fact_as_of(f, as_of) else {
            return Ok(None);
        };
        let summary = summary_of(&view.version, view.valid);
        let text = render_fact(&f.text, &summary);
        let first = &view.evidence[0];
        let Some(citation) = self.citation_for_evidence(first).await? else {
            return Ok(None);
        };
        let ep = self.ledger.episode(&f.corpus, &first.episode_id).await?;
        Ok(Some(Hit {
            item: item.clone(),
            rank: 0,
            score,
            tokens: self.tokenizer.count(&text),
            text,
            source_id: first.source_id.clone(),
            occurred_at: ep.and_then(|e| e.occurred_at),
            fact: Some(summary),
            citation,
        }))
    }

    /// The corpus snapshot at the current partition generations.
    async fn snapshot(&self, corpus: &str) -> Result<Arc<Snapshot>> {
        let mut key = self.ledger.partitions(corpus).await?;
        key.sort();
        if let Some(s) = self.snapshots.get(corpus, &key) {
            return Ok(s);
        }
        let _t = timing::span("knowledge.snapshot").items(1);
        let ids = self.ledger.point_ids(corpus).await?;
        let (mut chunk_ids, mut fact_ids) = (Vec::new(), Vec::new());
        for id in ids {
            match ItemRef::from_point_id(&id) {
                Some(ItemRef::Chunk(id)) => chunk_ids.push(id),
                Some(ItemRef::Fact(id)) => fact_ids.push(id),
                None => {}
            }
        }
        let chunks = self.ledger.chunks(corpus, &chunk_ids).await?;
        let facts = self.ledger.facts(corpus, &fact_ids).await?;
        let snap = Arc::new(Snapshot::new(key, chunks, facts));
        self.snapshots.put(corpus, snap.clone());
        Ok(snap)
    }

    /// [`LexicalMode::Bm25`] search: rank only what the reader can see,
    /// fuse, optionally rerank the head, then fill the allowance in order.
    #[allow(clippy::too_many_arguments)]
    async fn search_visible(
        &self,
        req: &SearchRequest,
        allowance: &mut EvidenceAllowance,
        params: Bm25Params,
        scope: &ReadScope,
        limit: usize,
        pool: usize,
        mut out: SearchResult,
    ) -> Result<SearchResult> {
        let snap = self.snapshot(&req.corpus).await?;
        let vis = snap.visible(&scope.parts, &req.access, scope.sources.as_ref(), req.as_of);
        let lexical: Vec<String> = snap
            .bm25(params, &vis)
            .search(&req.query, pool)
            .into_iter()
            .map(|h| h.id)
            .collect();
        // Over-fetch by the number of invisible items so filtering cannot
        // empty the pool.
        let vector = match self
            .vector_candidates(&req.corpus, scope, &req.query, pool + vis.hidden)
            .await
        {
            Ok(v) => v
                .into_iter()
                .filter(|id| vis.by_id.contains_key(id))
                .take(pool)
                .collect(),
            Err(_) => {
                out.degraded.push("vector_unavailable".into());
                Vec::new()
            }
        };
        if self.vectors.is_none() {
            out.degraded.push("vector_disabled".into());
        }
        let mut fused: Vec<(u32, f32)> = rrf(&[lexical, vector], self.config.rrf_k)
            .into_iter()
            .filter_map(|(id, s)| vis.by_id.get(&id).map(|&i| (i, s)))
            .collect();
        if let Some((reranker, top_n)) = &self.reranker {
            let head = fused.len().min(*top_n);
            if head > 1 {
                let texts: Vec<String> = fused[..head]
                    .iter()
                    .map(|&(i, _)| rerank_text(&snap, i, req.as_of))
                    .collect();
                let scores = reranker.score(&req.query, &texts).await?;
                let mut order: Vec<usize> = (0..head).collect();
                // Stable: equal scores keep their fused order.
                order.sort_by(|&a, &b| {
                    scores[b]
                        .partial_cmp(&scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let reordered: Vec<(u32, f32)> = order.iter().map(|&j| fused[j]).collect();
                fused.splice(..head, reordered);
            }
        }
        // One title lookup per source rather than per hit.
        let mut titles: HashMap<String, Label> = HashMap::new();
        for (i, score) in fused {
            if out.hits.len() >= limit {
                break;
            }
            let item = ItemRef::from_point_id(snap.id(i)).expect("snapshot ids are point ids");
            let hit = match snap.chunks.get(i as usize) {
                Some(c) => {
                    let title = match titles.get(&c.source_id) {
                        Some(t) => t.clone(),
                        None => {
                            let t = self.source_label(&c.corpus, &c.source_id).await?;
                            titles.insert(c.source_id.clone(), t.clone());
                            t
                        }
                    };
                    chunk_hit_titled(&*self.tokenizer, &item, score, c, title)
                }
                None => {
                    let f = &snap.facts[i as usize - snap.chunks.len()];
                    let f = match &scope.sources {
                        Some(allowed) => match snap.restricted(f, allowed) {
                            Some(f) => std::borrow::Cow::Owned(f),
                            None => continue,
                        },
                        None => std::borrow::Cow::Borrowed(f),
                    };
                    let Some(hit) = self.fact_hit(&item, score, &f, req.as_of).await? else {
                        continue;
                    };
                    hit
                }
            };
            charge(&mut out, allowance, hit);
        }
        Ok(out)
    }

    async fn vector_candidates(
        &self,
        corpus: &str,
        scope: &ReadScope,
        query: &str,
        pool: usize,
    ) -> Result<Vec<String>> {
        let Some((index, _)) = &self.vectors else {
            return Ok(Vec::new());
        };
        self.ensure_collection().await?;
        let q = self
            .embedder
            .embed(&[query.to_string()])
            .await?
            .vectors
            .pop()
            .unwrap_or_default();
        let mut hits: Vec<ScoredId> = Vec::new();
        for p in &scope.parts {
            let ns = namespace(corpus, p);
            hits.extend(match &scope.tags {
                Some(tags) => {
                    index
                        .search_tagged(&self.config.collection, &ns, &q, pool, tags)
                        .await?
                }
                None => index.search(&self.config.collection, &ns, &q, pool).await?,
            });
        }
        Ok(keep_store::rank(hits, pool)
            .into_iter()
            .map(|h| h.id)
            .collect())
    }

    /// Fetch one chunk or fact by id with its evidence, charged against the
    /// allowance. Unknown and unreadable ids are indistinguishable.
    pub async fn fetch(
        &self,
        corpus: &str,
        access: &Access,
        id: &str,
        as_of: AsOf,
        allowance: &mut EvidenceAllowance,
    ) -> Result<FetchResult> {
        let item = ItemRef::from_point_id(id).ok_or(KnowledgeError::NotFound)?;
        let scope = self.scope(corpus, access).await?;
        match item {
            ItemRef::Chunk(id) => {
                let c = self
                    .ledger
                    .chunks(corpus, std::slice::from_ref(&id))
                    .await?
                    .pop()
                    .filter(|c| {
                        access.allows(&c.partition)
                            && scope.allows_source(&c.source_id)
                            && chunk_visible(c, as_of)
                    })
                    .ok_or(KnowledgeError::NotFound)?;
                let text = render_chunk(&c);
                let tokens = self.tokenizer.count(&text);
                if !allowance.try_charge(tokens) {
                    return Err(KnowledgeError::BudgetExhausted);
                }
                let citation = self.citation_for_chunk(&c).await?;
                Ok(FetchResult {
                    item: Fetched::Chunk { chunk: c, citation },
                    text,
                    tokens,
                    truncated: false,
                })
            }
            ItemRef::Fact(id) => {
                let f = self
                    .ledger
                    .facts(corpus, std::slice::from_ref(&id))
                    .await?
                    .pop()
                    .filter(|f| access.allows(&f.partition))
                    .ok_or(KnowledgeError::NotFound)?;
                let f = self
                    .restricted(&scope, &f)
                    .await?
                    .ok_or(KnowledgeError::NotFound)?;
                let view = fact_as_of(&f, as_of).ok_or(KnowledgeError::NotFound)?;
                let summary = summary_of(&view.version, view.valid);
                let mut text = render_fact(&f.text, &summary);
                let mut tokens = self.tokenizer.count(&text);
                if !allowance.try_charge(tokens) {
                    return Err(KnowledgeError::BudgetExhausted);
                }
                let mut citations = Vec::new();
                let mut truncated = false;
                for e in &view.evidence {
                    let Some(c) = self.citation_for_evidence(e).await? else {
                        continue;
                    };
                    let line = format!(
                        "\n  evidence {}: [{}] {}: \"{}\"",
                        c.reference,
                        render_date(c.occurred_at),
                        c.speaker,
                        c.quote
                    );
                    let t = self.tokenizer.count(&line);
                    if !allowance.try_charge(t) {
                        truncated = true;
                        break;
                    }
                    text.push_str(&line);
                    tokens += t;
                    citations.push(c);
                }
                // Only expose the versions and evidence visible to this read.
                let mut fact = f.clone();
                fact.versions
                    .retain(|v| as_of.known_at.is_none_or(|t| v.recorded_at <= t));
                fact.evidence = view.evidence;
                Ok(FetchResult {
                    item: Fetched::Fact {
                        fact,
                        summary,
                        citations,
                    },
                    text,
                    tokens,
                    truncated,
                })
            }
        }
    }

    /// Resolve a chunk, evidence or fact id to stable citations.
    pub async fn cite(&self, corpus: &str, access: &Access, id: &str) -> Result<Vec<Citation>> {
        let scope = self.scope(corpus, access).await?;
        if id.starts_with("ev_") {
            let e = self
                .ledger
                .evidence(corpus, id)
                .await?
                .filter(|e| access.allows(&e.partition) && scope.allows_source(&e.source_id))
                .ok_or(KnowledgeError::NotFound)?;
            return Ok(vec![
                self.citation_for_evidence(&e)
                    .await?
                    .ok_or(KnowledgeError::NotFound)?,
            ]);
        }
        match ItemRef::from_point_id(id).ok_or(KnowledgeError::NotFound)? {
            ItemRef::Chunk(cid) => {
                let c = self
                    .ledger
                    .chunks(corpus, &[cid])
                    .await?
                    .pop()
                    .filter(|c| access.allows(&c.partition) && scope.allows_source(&c.source_id))
                    .ok_or(KnowledgeError::NotFound)?;
                Ok(vec![self.citation_for_chunk(&c).await?])
            }
            ItemRef::Fact(fid) => {
                let f = self
                    .ledger
                    .facts(corpus, &[fid])
                    .await?
                    .pop()
                    .filter(|f| access.allows(&f.partition))
                    .ok_or(KnowledgeError::NotFound)?;
                let f = scope.restrict(&f).ok_or(KnowledgeError::NotFound)?;
                let mut out = Vec::new();
                for e in &f.evidence {
                    if let Some(c) = self.citation_for_evidence(e).await? {
                        out.push(c);
                    }
                }
                Ok(out)
            }
        }
    }

    /// Entities and facts around a starting entity (id, name or alias).
    /// Depth is capped at 2, fan-out per entity at 25, nodes at the
    /// configured maximum.
    pub async fn traverse(
        &self,
        corpus: &str,
        access: &Access,
        start: &str,
        depth: usize,
        fanout: usize,
        as_of: AsOf,
    ) -> Result<Neighbourhood> {
        let scope = self.scope(corpus, access).await?;
        let parts = &scope.parts;
        let mut out = Neighbourhood::default();
        if scope.is_empty() {
            return Ok(out);
        }
        let depth = depth.clamp(1, 2);
        let fanout = fanout.clamp(1, 25);
        let max_nodes = self.config.max_traverse_nodes;
        let norm = normalize_name(start);
        let mut frontier: Vec<Entity> = match self.ledger.entity(corpus, start).await? {
            Some(e) if access.allows(&e.partition) => vec![e],
            _ => {
                self.ledger
                    .entities_named(corpus, parts, &norm, scope.list.as_deref())
                    .await?
            }
        };
        if let Some(list) = &scope.list {
            // Under an allowlist an entity exists only through allowed
            // sources: an alias they state, or a fact they support.
            let mut kept = Vec::new();
            for mut e in frontier {
                e.aliases = self.ledger.aliases_from(corpus, &e.id, list).await?;
                let named = e.aliases.iter().any(|a| normalize_name(a) == norm);
                if named
                    || !self
                        .ledger
                        .facts_about(corpus, parts, &e.id, 1, Some(list))
                        .await?
                        .is_empty()
                {
                    kept.push(e);
                }
            }
            frontier = kept;
        }
        let mut seen: BTreeSet<String> = frontier.iter().map(|e| e.id.clone()).collect();
        let mut edge_ids = BTreeSet::new();
        out.nodes.extend(frontier.iter().cloned());
        for _ in 0..depth {
            let mut next = Vec::new();
            for e in &frontier {
                let facts = self
                    .ledger
                    .facts_about(corpus, parts, &e.id, fanout + 1, scope.list.as_deref())
                    .await?;
                if facts.len() > fanout {
                    out.truncated = true;
                }
                for f in facts.into_iter().take(fanout) {
                    if !access.allows(&f.partition) {
                        continue;
                    }
                    let Some(f) = self.restricted(&scope, &f).await? else {
                        continue;
                    };
                    let Some(view) = fact_as_of(&f, as_of) else {
                        continue;
                    };
                    if edge_ids.insert(f.id.clone()) {
                        out.edges.push(Edge {
                            fact_id: f.id.clone(),
                            subject: f.subject.clone(),
                            predicate: f.predicate.clone(),
                            object: f.object.clone(),
                            object_entity: f.object_entity.clone(),
                            summary: summary_of(&view.version, view.valid),
                        });
                    }
                    for other in [Some(f.subject.clone()), f.object_entity.clone()]
                        .into_iter()
                        .flatten()
                    {
                        if seen.contains(&other) {
                            continue;
                        }
                        if seen.len() >= max_nodes {
                            out.truncated = true;
                            continue;
                        }
                        if let Some(mut n) = self.ledger.entity(corpus, &other).await?
                            && access.allows(&n.partition)
                        {
                            if let Some(list) = &scope.list {
                                n.aliases = self.ledger.aliases_from(corpus, &n.id, list).await?;
                            }
                            seen.insert(other);
                            out.nodes.push(n.clone());
                            next.push(n);
                        }
                    }
                }
            }
            frontier = next;
        }
        if scope.sources.is_some() {
            // Entities exist independently of sources; under an allowlist,
            // show only those an allowed fact connects.
            let linked: BTreeSet<&str> = out
                .edges
                .iter()
                .flat_map(|e| std::iter::once(e.subject.as_str()).chain(e.object_entity.as_deref()))
                .collect();
            let keep: BTreeSet<String> = linked.into_iter().map(String::from).collect();
            out.nodes.retain(|n| keep.contains(&n.id));
        }
        Ok(out)
    }

    /// The book index for what the caller may read.
    pub async fn index(
        &self,
        corpus: &str,
        access: &Access,
        req: &IndexRequest,
    ) -> Result<IndexView> {
        let scope = self.scope(corpus, access).await?;
        for _ in 0..3 {
            let parts: Vec<(String, i64)> = self
                .ledger
                .partitions(corpus)
                .await?
                .into_iter()
                .filter(|(p, _)| access.allows(p))
                .collect();
            let mut built: Vec<(String, IndexInputs)> = Vec::new();
            for (p, generation) in &parts {
                let inputs = match &scope.list {
                    // Filtered views are built per request, never cached.
                    Some(list) => normalize(
                        self.ledger
                            .index_inputs(corpus, p, Some(list.as_slice()))
                            .await?,
                    ),
                    None => self.partition_index(corpus, p, *generation).await?,
                };
                built.push((p.clone(), inputs));
            }
            // If anything was written meanwhile, rebuild rather than serve a
            // view that may include something just erased.
            let after: Vec<(String, i64)> = self
                .ledger
                .partitions(corpus)
                .await?
                .into_iter()
                .filter(|(p, _)| access.allows(p))
                .collect();
            let consistent = after == parts
                && built
                    .iter()
                    .zip(&parts)
                    .all(|((_, b), (_, g))| b.generation == *g);
            if consistent {
                return Ok(render(
                    corpus,
                    &built,
                    req,
                    self.tokenizer.as_ref(),
                    self.clock.now(),
                ));
            }
        }
        Err(KnowledgeError::Conflict(
            "index kept changing while it was read".into(),
        ))
    }

    async fn partition_index(
        &self,
        corpus: &str,
        partition: &str,
        generation: i64,
    ) -> Result<IndexInputs> {
        if let Some((g, body)) = self.ledger.cached_index(corpus, partition).await?
            && g == generation
            && let Ok(inputs) = serde_json::from_str::<IndexInputs>(&body)
            && inputs.generation == generation
        {
            return Ok(inputs);
        }
        let inputs = normalize(self.ledger.index_inputs(corpus, partition, None).await?);
        let body =
            serde_json::to_string(&inputs).map_err(|e| KnowledgeError::Storage(e.to_string()))?;
        self.ledger
            .store_index(corpus, partition, inputs.generation, &body)
            .await?;
        Ok(inputs)
    }

    // ----- forgetting ------------------------------------------------------

    /// Erase a source and everything derived only from it, including vector
    /// points and index entries, and journal the erasure without content.
    pub async fn forget_source(
        &self,
        corpus: &str,
        source_id: &str,
    ) -> Result<Option<ForgetReport>> {
        let now = self.clock.now();
        self.snapshots.forget(corpus);
        let Some(report) = self.ledger.purge_source(corpus, source_id, now).await? else {
            return Ok(None);
        };
        let (vectors_deleted, vectors_deferred) = self
            .clean_projection(corpus, &report, &report.point_ids, now)
            .await?;
        Ok(Some(ForgetReport {
            episodes: report.episodes,
            chunks: report.chunks,
            facts_deleted: report.facts_deleted,
            facts_kept: report.facts_kept,
            vectors_deleted,
            vectors_deferred,
        }))
    }

    /// Remove a whole corpus from the ledger and the vector projection.
    pub async fn delete_corpus(&self, corpus: &str) -> Result<()> {
        let _t = timing::span("knowledge.delete_corpus").items(1);
        self.snapshots.forget(corpus);
        let parts = self.ledger.delete_corpus(corpus).await?;
        if let Some((index, _)) = &self.vectors {
            for p in parts {
                index
                    .delete_namespace(&self.config.collection, &namespace(corpus, &p))
                    .await?;
            }
        }
        Ok(())
    }

    /// Queue re-projection of every chunk and fact, e.g. after replacing the
    /// vector backend. The ledger stays the source of truth.
    pub async fn rebuild_projection(&self, corpus: &str) -> Result<usize> {
        let jobs: Vec<NewJob> = self
            .ledger
            .point_ids(corpus)
            .await?
            .into_iter()
            .map(|target| NewJob {
                kind: JobKind::Project,
                target,
            })
            .collect();
        self.ledger.enqueue(corpus, &jobs, self.clock.now()).await?;
        Ok(jobs.len())
    }
}

fn retryable(e: &KnowledgeError) -> bool {
    match e {
        KnowledgeError::Provider(p) => p.retryable(),
        KnowledgeError::Vector(_) | KnowledgeError::Storage(_) | KnowledgeError::Conflict(_) => {
            true
        }
        _ => false,
    }
}

/// Midnight UTC on a date, for fixtures and tests.
pub fn day(y: i32, m: u32, d: u32) -> DateTime<Utc> {
    use chrono::TimeZone;
    Utc.with_ymd_and_hms(y, m, d, 0, 0, 0)
        .single()
        .expect("valid date")
}

fn chunk_citation(c: &Chunk, (title, metadata): Label) -> Citation {
    Citation {
        reference: format!("keep:{}", c.id),
        source_id: c.source_id.clone(),
        source_title: title,
        episode_id: c.episode_id.clone(),
        speaker: c.speaker.clone(),
        occurred_at: c.occurred_at,
        char_start: c.char_start,
        char_end: c.char_end,
        quote: c.text.clone(),
        source_metadata: metadata,
    }
}

fn chunk_hit_titled(
    tokenizer: &dyn Tokenizer,
    item: &ItemRef,
    score: f32,
    c: &Chunk,
    label: Label,
) -> Hit {
    let text = render_chunk(c);
    Hit {
        item: item.clone(),
        rank: 0,
        score,
        tokens: tokenizer.count(&text),
        text,
        source_id: c.source_id.clone(),
        occurred_at: c.occurred_at,
        fact: None,
        citation: chunk_citation(c, label),
    }
}
