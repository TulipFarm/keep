//! The authoritative store contract.
//!
//! A ledger owns every record and every permission decision; vector indexes
//! are projections rebuilt from it. Writes that depend on what was read
//! (extraction results, version changes) are applied as one [`ChangeSet`]
//! with optimistic checks, so concurrent workers or a purge in between make
//! the write fail cleanly instead of resurrecting deleted data.

use std::collections::BTreeSet;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::model::{
    Chunk, Entity, Episode, ErasureEntry, Evidence, Fact, FactVersion, Job, JobCounts, JobState,
    NewJob, Source, Time,
};

#[derive(Debug, thiserror::Error)]
pub enum KnowledgeError {
    #[error("not found")]
    NotFound,
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("evidence allowance exhausted")]
    BudgetExhausted,
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("vector index error: {0}")]
    Vector(#[from] keep_store::StoreError),
    #[error("provider error: {0}")]
    Provider(#[from] keep_providers::ProviderError),
}

pub type Result<T, E = KnowledgeError> = std::result::Result<T, E>;

/// A source and the episodes and chunks derived from it.
#[derive(Clone, Debug)]
pub struct IngestBatch {
    pub source: Source,
    pub episodes: Vec<Episode>,
    pub chunks: Vec<Chunk>,
    pub now: Time,
    /// Replace an existing source whose metadata or episodes differ: purge
    /// its old content and insert this batch in one atomic step. Without
    /// it such a batch is a [`KnowledgeError::Conflict`].
    pub replace: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestReport {
    pub source_created: bool,
    pub episodes_added: usize,
    pub episodes_unchanged: usize,
    pub chunks_added: usize,
    pub jobs_enqueued: usize,
    /// An earlier revision of the source was purged and replaced.
    #[serde(default)]
    pub replaced: bool,
    /// What the replacement purged, for the engine to clean the projection.
    #[serde(skip)]
    pub purged: Option<PurgeReport>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntityWrite {
    pub id: String,
    pub name: String,
    pub kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AliasWrite {
    pub entity_id: String,
    pub alias: String,
    pub source_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VersionWrite {
    pub fact_id: String,
    /// The version the planner saw as current; the write fails otherwise.
    pub expected_current: i32,
    pub version: FactVersion,
}

/// Everything one extraction job wants to write.
#[derive(Clone, Debug)]
pub struct ChangeSet {
    pub corpus: String,
    pub partition: String,
    /// The episode the change was planned from.
    pub episode_id: String,
    /// Every episode the change cites, with the content hash it was read
    /// at. The write is refused ([`ApplyOutcome::EpisodeGone`]) if any of
    /// them no longer exists or now holds other content, such as after a
    /// source was replaced by a new revision while it was being extracted.
    pub episodes: Vec<(String, String)>,
    pub now: Time,
    pub entities: Vec<EntityWrite>,
    pub aliases: Vec<AliasWrite>,
    /// Facts that must not exist yet (versions[0] plus evidence).
    pub new_facts: Vec<Fact>,
    /// Extra evidence for existing facts; duplicates are ignored.
    pub evidence: Vec<Evidence>,
    pub versions: Vec<VersionWrite>,
    pub jobs: Vec<NewJob>,
}

impl ChangeSet {
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
            && self.aliases.is_empty()
            && self.new_facts.is_empty()
            && self.evidence.is_empty()
            && self.versions.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyOutcome {
    Applied,
    /// A cited episode was purged or replaced meanwhile; nothing was written.
    EpisodeGone,
    /// A fact or version changed meanwhile; re-plan and retry.
    Conflict,
}

/// Per-source totals for the table of contents.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceStat {
    pub id: String,
    pub title: String,
    pub collection: String,
    pub episodes: i64,
    pub facts: i64,
    pub first_at: Option<Time>,
    pub last_at: Option<Time>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntityStat {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub aliases: Vec<String>,
    pub facts: i64,
    /// Sources with evidence for facts about this entity.
    pub sources: Vec<String>,
}

/// Everything the index needs from one partition, read consistently.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexInputs {
    pub generation: i64,
    pub sources: Vec<SourceStat>,
    pub entities: Vec<EntityStat>,
}

/// What a purge removed. Point ids let the engine clean the projection.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PurgeReport {
    pub partition: String,
    pub episodes: i64,
    pub chunks: i64,
    pub facts_deleted: i64,
    pub facts_kept: i64,
    pub point_ids: Vec<String>,
    /// Facts that lost some evidence but survive, to re-project.
    pub kept_fact_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    pub partition: String,
    pub text: String,
    /// Sources the item derives from: a chunk's source, or every source
    /// with evidence for a fact.
    pub sources: Vec<String>,
}

#[async_trait]
pub trait Ledger: Send + Sync {
    fn backend(&self) -> &'static str;

    /// Idempotent: identical episodes are skipped; an episode id reused with
    /// different content, or a source whose metadata changed, is a
    /// [`KnowledgeError::Conflict`] unless [`IngestBatch::replace`] is set. New episodes enqueue extraction, new
    /// chunks enqueue projection, all in the same transaction.
    async fn ingest(&self, batch: IngestBatch) -> Result<IngestReport>;

    async fn source(&self, corpus: &str, id: &str) -> Result<Option<Source>>;
    async fn episode(&self, corpus: &str, id: &str) -> Result<Option<Episode>>;
    /// A source's episodes in order. Empty for an unknown source.
    async fn source_episodes(&self, corpus: &str, source_id: &str) -> Result<Vec<Episode>>;
    async fn chunks(&self, corpus: &str, ids: &[String]) -> Result<Vec<Chunk>>;
    async fn facts(&self, corpus: &str, ids: &[String]) -> Result<Vec<Fact>>;
    async fn evidence(&self, corpus: &str, id: &str) -> Result<Option<Evidence>>;
    async fn entity(&self, corpus: &str, id: &str) -> Result<Option<Entity>>;

    /// Entity in one partition whose normalised name or alias matches.
    async fn find_entity(
        &self,
        corpus: &str,
        partition: &str,
        normalized: &str,
    ) -> Result<Option<Entity>>;
    /// Entities in any of `partitions` whose name or alias matches. With
    /// `sources`, only aliases stated by those sources match or are returned.
    async fn entities_named(
        &self,
        corpus: &str,
        partitions: &[String],
        normalized: &str,
        sources: Option<&[String]>,
    ) -> Result<Vec<Entity>>;
    /// An entity's aliases stated by any of `sources`.
    async fn aliases_from(
        &self,
        corpus: &str,
        entity_id: &str,
        sources: &[String],
    ) -> Result<BTreeSet<String>>;
    /// Facts with a current version whose subject and predicate match.
    async fn current_facts(
        &self,
        corpus: &str,
        partition: &str,
        subject: &str,
        predicate: &str,
    ) -> Result<Vec<Fact>>;
    /// Facts whose subject or object is the entity, newest first. With
    /// `sources`, only facts with evidence from at least one of them.
    async fn facts_about(
        &self,
        corpus: &str,
        partitions: &[String],
        entity_id: &str,
        limit: usize,
        sources: Option<&[String]>,
    ) -> Result<Vec<Fact>>;

    async fn apply(&self, change: ChangeSet) -> Result<ApplyOutcome>;

    /// Full-text candidates (chunk and fact ids) from the given partitions.
    /// With `sources`, only chunks of those sources and facts with evidence
    /// from at least one of them, filtered before the limit.
    async fn lexical(
        &self,
        corpus: &str,
        partitions: &[String],
        query: &str,
        limit: usize,
        sources: Option<&[String]>,
    ) -> Result<Vec<keep_store::ScoredId>>;
    fn lexical_method(&self) -> String;

    /// Text and partition for a chunk or fact id, if it still exists.
    async fn projection(&self, corpus: &str, point_id: &str) -> Result<Option<Projection>>;
    /// [`Ledger::projection`] for many ids, in the same order.
    async fn projections(&self, corpus: &str, ids: &[String]) -> Result<Vec<Option<Projection>>> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            out.push(self.projection(corpus, id).await?);
        }
        Ok(out)
    }
    /// Every chunk and fact id, to rebuild a vector projection from scratch.
    async fn point_ids(&self, corpus: &str) -> Result<Vec<String>>;

    // Jobs.
    async fn enqueue(&self, corpus: &str, jobs: &[NewJob], now: Time) -> Result<()>;
    /// Lease due jobs (pending or retry-scheduled with `next_attempt_at <= now`).
    async fn claim_jobs(&self, corpus: &str, now: Time, limit: usize) -> Result<Vec<Job>>;
    /// Returns false when the lease was lost and the result was discarded.
    async fn complete_job(&self, job: &Job, now: Time) -> Result<bool>;
    /// [`Ledger::complete_job`] for many jobs, in the same order.
    async fn complete_jobs(&self, jobs: &[Job], now: Time) -> Result<Vec<bool>> {
        let mut out = Vec::with_capacity(jobs.len());
        for job in jobs {
            out.push(self.complete_job(job, now).await?);
        }
        Ok(out)
    }
    async fn fail_job(
        &self,
        job: &Job,
        error: &str,
        retryable: bool,
        now: Time,
    ) -> Result<JobState>;
    async fn jobs(&self, corpus: &str) -> Result<Vec<Job>>;
    /// Jobs of one partition by state: extraction of its episodes,
    /// projection of its chunks and facts, and vector removals from its
    /// namespace `ns`. Jobs of other partitions, and of records since
    /// forgotten, are not counted, so the answer reveals nothing outside it.
    /// Without `succeeded`, finished jobs are skipped and counted as 0: they
    /// pile up over a corpus's life, so counting them grows with every write.
    async fn job_counts(
        &self,
        corpus: &str,
        partition: &str,
        ns: &str,
        succeeded: bool,
    ) -> Result<JobCounts>;
    /// Corpora with at least one job due at `now`, for a background worker.
    async fn due_corpora(&self, now: Time, limit: usize) -> Result<Vec<String>>;

    // Book index.
    /// Partition labels with their write generation.
    async fn partitions(&self, corpus: &str) -> Result<Vec<(String, i64)>>;
    /// With `sources`, counts only those sources and the facts and
    /// entities their evidence supports.
    async fn index_inputs(
        &self,
        corpus: &str,
        partition: &str,
        sources: Option<&[String]>,
    ) -> Result<IndexInputs>;
    /// Ids of the sources in `partitions` belonging to any of `collections`.
    async fn sources_in_collections(
        &self,
        corpus: &str,
        partitions: &[String],
        collections: &[String],
    ) -> Result<Vec<String>>;
    async fn cached_index(&self, corpus: &str, partition: &str) -> Result<Option<(i64, String)>>;
    /// Store a built index only if `generation` is still current.
    async fn store_index(
        &self,
        corpus: &str,
        partition: &str,
        generation: i64,
        body: &str,
    ) -> Result<bool>;

    // Forgetting.
    async fn purge_source(
        &self,
        corpus: &str,
        source_id: &str,
        now: Time,
    ) -> Result<Option<PurgeReport>>;
    async fn erasure_journal(&self) -> Result<Vec<ErasureEntry>>;
    /// Remove a whole corpus; returns the partitions it had.
    async fn delete_corpus(&self, corpus: &str) -> Result<Vec<String>>;
}

/// Lowercase, collapse whitespace and drop surrounding punctuation.
pub fn normalize_name(s: &str) -> String {
    s.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

pub const MAX_JOB_ATTEMPTS: i32 = 3;

/// Exponential backoff: 2s, 4s, 8s, ...
pub fn retry_delay(attempts: i32) -> chrono::Duration {
    chrono::Duration::seconds(1i64 << attempts.clamp(1, 10))
}

/// Whether two sources agree on everything a caller sets.
pub(crate) fn same_source(a: &Source, b: &Source) -> bool {
    a.partition == b.partition
        && a.collection == b.collection
        && a.title == b.title
        && a.kind == b.kind
        && a.metadata == b.metadata
}

/// The (id, content hash) pairs of a batch, to compare revisions.
pub(crate) fn episode_set(episodes: &[Episode]) -> std::collections::BTreeSet<(String, String)> {
    episodes
        .iter()
        .map(|e| (e.id.clone(), e.content_hash.clone()))
        .collect()
}
