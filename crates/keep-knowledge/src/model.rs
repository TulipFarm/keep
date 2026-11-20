//! Records held in the Knowledge ledger.
//!
//! Everything lives inside a *corpus* (a hard isolation boundary, like a
//! tenant) and carries the *partition* label of the source it came from.
//! A partition is a visibility label: callers present the set of labels they
//! may read, and every derived record (chunk, entity, fact, evidence, index
//! entry) inherits exactly one label from its source. Facts are never merged
//! across partitions, so nothing derived from a restricted source can shape
//! what a caller without access sees.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub type Time = DateTime<Utc>;

/// Deterministic opaque id: `prefix_` plus 24 hex characters of a SHA-256
/// over the parts, NUL-separated so `("ab","c")` and `("a","bc")` differ.
pub fn digest_id(prefix: &str, parts: &[&str]) -> String {
    format!("{prefix}_{}", &digest_hex(parts)[..24])
}

pub fn digest_hex(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p.as_bytes());
        h.update([0u8]);
    }
    hex::encode(h.finalize())
}

/// The visibility labels a caller may read. Empty means nothing: access is
/// never widened to "all" by omission.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Access {
    pub partitions: BTreeSet<String>,
    /// When set, only records derived from these sources are visible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<BTreeSet<String>>,
    /// When set, only records derived from sources in these collections are
    /// visible. Combined with `sources`, a source must satisfy both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collections: Option<BTreeSet<String>>,
}

impl Access {
    pub fn to<I, S>(labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            partitions: labels.into_iter().map(Into::into).collect(),
            sources: None,
            collections: None,
        }
    }

    /// Narrow reads to records derived from these sources. An empty list
    /// shows nothing.
    pub fn only_sources<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.sources = Some(ids.into_iter().map(Into::into).collect());
        self
    }

    /// Narrow reads to records derived from sources in these collections.
    /// An empty list shows nothing.
    pub fn only_collections<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.collections = Some(names.into_iter().map(Into::into).collect());
        self
    }

    pub fn allows(&self, partition: &str) -> bool {
        self.partitions.contains(partition)
    }

    pub fn labels(&self) -> Vec<String> {
        self.partitions.iter().cloned().collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Conversation,
    Document,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceKind::Conversation => "conversation",
            SourceKind::Document => "document",
        }
    }
    pub fn parse(s: &str) -> Self {
        if s == "document" {
            SourceKind::Document
        } else {
            SourceKind::Conversation
        }
    }
}

/// A conversation, document or feed: the unit of visibility and forgetting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub corpus: String,
    pub id: String,
    pub partition: String,
    /// Grouping shown in the table of contents, such as "Support calls".
    pub collection: String,
    pub title: String,
    pub kind: SourceKind,
    pub created_at: Time,
    /// Small host-defined string map, such as a document URL or id.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// One timestamped unit of a source: a message or a document section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub corpus: String,
    pub id: String,
    pub source_id: String,
    pub partition: String,
    /// Position within the source.
    pub ordinal: i32,
    pub speaker: String,
    /// When it happened in the world, if known.
    pub occurred_at: Option<Time>,
    /// When the ledger committed it.
    pub recorded_at: Time,
    pub text: String,
    pub content_hash: String,
}

/// A retrievable slice of an episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub corpus: String,
    pub id: String,
    pub episode_id: String,
    pub source_id: String,
    pub partition: String,
    pub ordinal: i32,
    /// Character offsets into the episode text, half-open.
    pub char_start: i32,
    pub char_end: i32,
    pub speaker: String,
    pub text: String,
    pub occurred_at: Option<Time>,
    pub recorded_at: Time,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub corpus: String,
    pub id: String,
    pub partition: String,
    pub name: String,
    pub kind: String,
    /// Alternative names, each backed by a source that stated it.
    pub aliases: BTreeSet<String>,
}

/// How a fact's start of validity was established.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeBasis {
    /// The source states the date.
    Explicit,
    /// The source said it was true when the episode happened; earlier
    /// validity is unknown.
    Observed,
    /// No usable time; the bound stays open and is labelled unknown.
    Unknown,
}

impl TimeBasis {
    pub fn as_str(self) -> &'static str {
        match self {
            TimeBasis::Explicit => "explicit",
            TimeBasis::Observed => "observed",
            TimeBasis::Unknown => "unknown",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "explicit" => TimeBasis::Explicit,
            "observed" => TimeBasis::Observed,
            _ => TimeBasis::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactStatus {
    Active,
    /// A later fact with the same single-valued predicate closed its
    /// validity window.
    Superseded,
    /// A competing value with no usable time ordering exists; both stay
    /// visible and marked.
    Disputed,
}

impl FactStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            FactStatus::Active => "active",
            FactStatus::Superseded => "superseded",
            FactStatus::Disputed => "disputed",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "superseded" => FactStatus::Superseded,
            "disputed" => FactStatus::Disputed,
            _ => FactStatus::Active,
        }
    }
}

/// One immutable version of a fact. World time (`valid_*`) and system time
/// (`recorded_at`/`invalidated_at`) are independent, both half-open.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FactVersion {
    pub version: i32,
    pub valid_from: Option<Time>,
    pub valid_to: Option<Time>,
    pub valid_from_basis: TimeBasis,
    pub status: FactStatus,
    pub recorded_at: Time,
    /// When a newer version replaced this one in the ledger.
    pub invalidated_at: Option<Time>,
    /// The fact whose evidence caused this version (None for the first).
    pub cause: Option<String>,
}

impl FactVersion {
    pub fn is_current(&self) -> bool {
        self.invalidated_at.is_none()
    }
}

/// An exact span of an episode supporting a fact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub corpus: String,
    pub id: String,
    pub fact_id: String,
    pub episode_id: String,
    pub source_id: String,
    pub partition: String,
    pub char_start: i32,
    pub char_end: i32,
    pub quote: String,
    /// Extractor name and version that proposed the fact.
    pub extractor: String,
    pub recorded_at: Time,
}

impl Evidence {
    pub fn id_for(corpus: &str, fact_id: &str, episode_id: &str, start: i32, end: i32) -> String {
        digest_id(
            "ev",
            &[
                corpus,
                fact_id,
                episode_id,
                &start.to_string(),
                &end.to_string(),
            ],
        )
    }
}

/// Stable identity of a statement; versions record how its validity changed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub corpus: String,
    pub id: String,
    pub partition: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// Set when the object is itself an entity, making the fact a graph edge.
    pub object_entity: Option<String>,
    /// Plain rendering, e.g. "Mira Sato lives in Lisbon".
    pub text: String,
    /// Oldest first.
    pub versions: Vec<FactVersion>,
    pub evidence: Vec<Evidence>,
}

impl Fact {
    pub fn id_for(
        corpus: &str,
        partition: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> String {
        digest_id(
            "fact",
            &[
                corpus,
                partition,
                subject,
                predicate,
                &object.to_lowercase(),
            ],
        )
    }

    pub fn current(&self) -> Option<&FactVersion> {
        self.versions.iter().rev().find(|v| v.is_current())
    }
}

/// A searchable record.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum ItemRef {
    Chunk(String),
    Fact(String),
}

impl ItemRef {
    /// Point id used in the vector projection.
    pub fn point_id(&self) -> String {
        match self {
            ItemRef::Chunk(id) | ItemRef::Fact(id) => id.clone(),
        }
    }
    pub fn from_point_id(id: &str) -> Option<Self> {
        if id.starts_with("chunk_") {
            Some(ItemRef::Chunk(id.into()))
        } else if id.starts_with("fact_") {
            Some(ItemRef::Fact(id.into()))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    /// Extract facts from one episode.
    Extract,
    /// Embed and upsert one chunk or fact into the vector projection.
    Project,
    /// Remove vector points left behind by a purge whose first attempt failed.
    Unproject,
}

impl JobKind {
    pub fn as_str(self) -> &'static str {
        match self {
            JobKind::Extract => "extract",
            JobKind::Project => "project",
            JobKind::Unproject => "unproject",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "extract" => JobKind::Extract,
            "unproject" => JobKind::Unproject,
            _ => JobKind::Project,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Pending,
    Running,
    RetryScheduled,
    Succeeded,
    Failed,
}

impl JobState {
    pub fn as_str(self) -> &'static str {
        match self {
            JobState::Pending => "pending",
            JobState::Running => "running",
            JobState::RetryScheduled => "retry_scheduled",
            JobState::Succeeded => "succeeded",
            JobState::Failed => "failed",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "running" => JobState::Running,
            "retry_scheduled" => JobState::RetryScheduled,
            "succeeded" => JobState::Succeeded,
            "failed" => JobState::Failed,
            _ => JobState::Pending,
        }
    }
}

/// How many jobs of a corpus are in each state. Content-free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobCounts {
    pub pending: i64,
    pub running: i64,
    pub retry_scheduled: i64,
    pub succeeded: i64,
    pub failed: i64,
}

impl JobCounts {
    pub fn add(&mut self, state: JobState, n: i64) {
        match state {
            JobState::Pending => self.pending += n,
            JobState::Running => self.running += n,
            JobState::RetryScheduled => self.retry_scheduled += n,
            JobState::Succeeded => self.succeeded += n,
            JobState::Failed => self.failed += n,
        }
    }

    /// Jobs that have not finished yet and will run again.
    pub fn outstanding(&self) -> i64 {
        self.pending + self.running + self.retry_scheduled
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Job {
    pub id: i64,
    pub corpus: String,
    pub kind: JobKind,
    /// Episode id, item point id, or `namespace\u{1f}point` for unproject.
    pub target: String,
    pub state: JobState,
    pub attempts: i32,
    pub max_attempts: i32,
    pub next_attempt_at: Time,
    /// Fencing token of the current claim; a stale worker cannot finish.
    pub lease: Option<i64>,
    pub last_error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewJob {
    pub kind: JobKind,
    pub target: String,
}

/// A content-free record that something was erased. Ids are stored as
/// digests so the journal can be replayed after a restore without keeping
/// the original identifiers or any text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErasureEntry {
    pub seq: i64,
    pub corpus_digest: String,
    pub target_kind: String,
    pub target_digest: String,
    pub erased_at: Time,
    pub episodes: i64,
    pub chunks: i64,
    pub facts: i64,
}
