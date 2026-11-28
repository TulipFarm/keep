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
    /// Replace an existing source whose metadata or episodes differ, in one
    /// atomic step. Episodes whose id and content are unchanged are kept
    /// with everything derived from them; removed or changed episodes are
    /// purged and the new ones inserted. A changed partition or kind purges
    /// the whole old revision. Without it such a batch is a
    /// [`KnowledgeError::Conflict`].
    pub replace: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestReport {
    pub source_created: bool,
    pub episodes_added: usize,
    pub episodes_unchanged: usize,
    pub chunks_added: usize,
    pub jobs_enqueued: usize,
    /// An earlier revision of the source was replaced: its metadata changed
    /// or some of its episodes were purged.
    #[serde(default)]
    pub replaced: bool,
    /// Episodes of the earlier revision purged by the replacement.
    #[serde(default)]
    pub episodes_removed: usize,
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

/// A chunk's vector kept by the ledger so unchanged text is never embedded
/// twice. `digest` covers the embedding model and the exact text.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredVector {
    pub point_id: String,
    pub digest: String,
    pub vector: Vec<f32>,
}

/// How long a chunk's stored vector outlives a revision that removed the
/// chunk, so a client resending the rest of a document after a retried
/// replace reuses it. Forgetting a source or corpus deletes it at once.
pub const VECTOR_GRACE_SECONDS: i64 = 3600;

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
    /// [`Ledger::claim_jobs`] limited to jobs derived from one source:
    /// extraction of its episodes and projection of its chunks and of the
    /// facts it supports. Vector removals are left to other claims.
    async fn claim_source_jobs(
        &self,
        corpus: &str,
        source_id: &str,
        now: Time,
        limit: usize,
    ) -> Result<Vec<Job>>;
    /// Hand leased jobs back unstarted: pending again, due at `now`, with
    /// the claim's attempt not counted. Jobs whose lease was lost are left.
    async fn release_jobs(&self, jobs: &[Job], now: Time) -> Result<()>;
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
    /// Delete up to `limit` jobs in `state` that finished before `before`,
    /// oldest first, and return how many went. A job's finish time is the
    /// end of its last lease (`next_attempt_at`). Only `Succeeded` and
    /// `Failed` jobs are ever deleted; other states delete nothing.
    async fn prune_jobs(&self, state: JobState, before: Time, limit: usize) -> Result<usize>;
    /// Corpora with at least one job due at `now`, for a background worker.
    async fn due_corpora(&self, now: Time, limit: usize) -> Result<Vec<String>>;
    /// Check the ledger answers, for a readiness probe. A backend whose
    /// operations hold a connection for a whole transaction must not make
    /// the probe wait behind one.
    async fn probe(&self, now: Time) -> Result<()> {
        self.due_corpora(now, 1).await.map(|_| ())
    }

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

    // Stored chunk vectors.
    /// Vectors stored for these (chunk point id, digest) pairs, in the same
    /// order; `None` where none is stored with that digest.
    async fn stored_vectors(
        &self,
        corpus: &str,
        keys: &[(String, String)],
    ) -> Result<Vec<Option<Vec<f32>>>>;
    /// Store vectors of chunks that still exist; others are skipped.
    async fn store_vectors(&self, corpus: &str, vectors: &[StoredVector], now: Time) -> Result<()>;
    /// Delete vectors of chunks removed by a revision before `removed_before`.
    async fn sweep_vectors(&self, removed_before: Time) -> Result<usize>;

    // Forgetting.
    /// Delete a source and everything derived from it, stored vectors included.
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

/// How a batch revises a source the ledger already holds.
#[derive(Debug, Default)]
pub(crate) struct Revision {
    /// Purge the whole old revision: its partition or kind changed.
    pub full: bool,
    /// Episodes of the old revision to purge: gone or changed.
    pub removed: BTreeSet<String>,
    /// Title, collection or metadata to update in place.
    pub metadata: bool,
}

/// Compare a batch with the source and (episode id, content hash) pairs
/// the ledger holds. Without `replace`, changed metadata is a conflict and
/// nothing is purged.
pub(crate) fn revision_plan(
    existing: &Source,
    batch: &IngestBatch,
    held: BTreeSet<(String, String)>,
) -> Result<Revision> {
    let src = &batch.source;
    let same_meta = same_source(existing, src);
    if !batch.replace {
        if !same_meta {
            return Err(KnowledgeError::Conflict(
                "source already exists with different metadata".into(),
            ));
        }
        return Ok(Revision::default());
    }
    if existing.partition != src.partition || existing.kind != src.kind {
        return Ok(Revision {
            full: true,
            ..Revision::default()
        });
    }
    let wanted = episode_set(&batch.episodes);
    Ok(Revision {
        full: false,
        removed: held
            .into_iter()
            .filter(|p| !wanted.contains(p))
            .map(|(id, _)| id)
            .collect(),
        metadata: !same_meta,
    })
}

/// The (id, content hash) pairs of a batch, to compare revisions.
pub(crate) fn episode_set(episodes: &[Episode]) -> std::collections::BTreeSet<(String, String)> {
    episodes
        .iter()
        .map(|e| (e.id.clone(), e.content_hash.clone()))
        .collect()
}
