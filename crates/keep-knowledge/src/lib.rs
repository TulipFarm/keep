//! Keep's KNOWLEDGE primitive.
//!
//! Knowledge answers "what do my sources say, and since when?". It ingests
//! conversations and documents as timestamped episodes, extracts a
//! conservative set of facts with exact evidence spans, keeps every change
//! as a new immutable version on two time axes (when it was true, and when
//! we learned it), and serves hybrid retrieval plus a book-style index.
//!
//! Every derived record inherits its source's visibility partition, and
//! every read is filtered to the caller's partitions *before* ranking.
//! Forgetting a source removes everything derived only from it, including
//! vector points and index entries, and leaves a content-free journal entry.
//!
//! Authority lives in a [`Ledger`] (in-memory or Postgres). Vector indexes
//! (pgvector or Qdrant) are projections that can be rebuilt from it.

pub mod budget;
pub mod chunking;
pub mod engine;
pub mod extract;
pub mod fusion;
pub mod index;
pub mod ledger;
pub mod memory_ledger;
pub mod model;
pub mod pg_ledger;
pub mod profile;
pub mod query_intent;
pub mod relative_time;
pub mod resolve;
pub mod session_extract;
pub mod session_summary;
mod snapshot;
pub mod subject;
pub mod temporal;
pub mod time_ref;

pub use budget::{DEFAULT_EVIDENCE_TOKENS, EvidenceAllowance, MAX_EVIDENCE_TOKENS};
pub use engine::{
    Citation, Clock, EpisodeInput, ExternalClaim, FetchResult, Fetched, ForgetReport, Hit, JobRun,
    Knowledge, KnowledgeConfig, LexicalMode, ManualClock, Neighbourhood, SearchRequest,
    SearchResult, SourceInput, SystemClock,
};
pub use index::{IndexRequest, IndexView};
pub use ledger::{KnowledgeError, Ledger};
pub use memory_ledger::MemoryLedger;
pub use model::{Access, ItemRef, SourceKind};
pub use pg_ledger::PgLedger;
pub use temporal::AsOf;
