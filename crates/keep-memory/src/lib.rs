//! Keep's MEMORY primitive.
//!
//! Memory is a small notebook of Markdown files that agents and people both
//! edit. Each file is shown as plain Markdown, but every save is stored as an
//! immutable structured revision: the text is cut losslessly into entries
//! with stable identities, descriptive annotations and protected lineage.
//!
//! - [`codec`] splits Markdown into entries and joins them back byte for byte.
//! - [`service::MemoryService`] runs every operation with compare-and-swap,
//!   limits, history, trash, undo, idempotency receipts and forget.
//! - [`ledger`] is the storage contract; [`inmem`] implements it in memory and
//!   the `keep-ledger` crate implements it on Postgres.
//! - [`merge`] replays a stale edit onto newer saves when they touched
//!   different entries.
//! - [`context`] assembles a token-bounded, deterministic prompt block.
//! - [`portable`] exports to and imports from a plain Markdown folder.
//! - [`conformance`] is the behaviour suite every ledger must pass.

pub mod align;
pub mod codec;
pub mod conformance;
pub mod context;
pub mod digest;
pub mod error;
pub mod inmem;
pub mod ledger;
pub mod merge;
pub mod model;
pub mod path;
pub mod portable;
pub mod service;

pub use codec::{EntryKind, EntryMode};
pub use context::{ContextBlock, ContextBudget, ContextRequest, KnowledgeToc, Layer};
pub use error::{LimitKind, MemoryError, Result};
pub use inmem::{InMemoryJournal, InMemoryLedger};
pub use ledger::{ErasureJournal, LedgerTx, MemoryLedger};
pub use model::{Annotation, Confidence, Entry, FileRecord, Head, Revision, TreeRecord};
pub use service::{
    CreateFile, Edit, ForgetSelector, Hunk, ListQuery, ManualClock, MemoryService, MetadataPatch,
    OpCtx, Retention, SequentialIds,
};
