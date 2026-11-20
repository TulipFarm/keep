//! Typed errors. Every failure a caller can act on has its own variant.

use serde::{Deserialize, Serialize};

use crate::model::Head;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    RenderedBytesPerFile,
    FilesPerTree,
    RenderedBytesPerTree,
    EntriesPerFile,
    MetadataBytesPerFile,
    DescriptionBytes,
    PathBytes,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum MemoryError {
    #[error("{what} not found")]
    NotFound { what: String },
    #[error("version conflict: the file changed; current head is {current:?}")]
    VersionConflict { current: Head },
    #[error("tree revision conflict: current tree revision is {current}")]
    TreeConflict { current: u64 },
    #[error("a file already exists at {0}")]
    PathExists(String),
    #[error("invalid path: {0}")]
    InvalidPath(String),
    #[error("invalid text: {0}")]
    InvalidText(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("the text to replace was not found")]
    NoMatch,
    #[error("the text to replace matches {0} times; make it unique or request replace-all")]
    MultipleMatches(usize),
    #[error("limit exceeded: {kind:?} is {actual}, limit {limit}")]
    Limit {
        kind: LimitKind,
        limit: usize,
        actual: usize,
    },
    #[error("idempotency key was already used with a different request")]
    IdempotencyConflict,
    #[error("idempotency key was retired by a forget; its outcome is no longer available")]
    IdempotencyKeyRetired,
    #[error("the file is in the trash")]
    InTrash,
    #[error("the file is not in the trash")]
    NotInTrash,
    #[error("the file stayed in the trash past its retention and was purged")]
    TrashExpired,
    #[error("revision {0} is not retained")]
    RevisionNotRetained(String),
    #[error("nothing to undo")]
    NothingToUndo,
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("merge conflict on entries {entries:?}: {reason}")]
    MergeConflict {
        entries: Vec<String>,
        reason: String,
    },
    #[error("export refused: {paths:?} would become trusted instruction files")]
    InstructionFileRefused { paths: Vec<String> },
    #[error("required context does not fit: {0}")]
    RequiredContextOverBudget(String),
    #[error("budget error: {0}")]
    Budget(String),
    #[error("storage error: {0}")]
    Storage(String),
}

impl MemoryError {
    pub fn not_found(what: impl Into<String>) -> Self {
        MemoryError::NotFound { what: what.into() }
    }

    /// A stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            MemoryError::NotFound { .. } => "NOT_FOUND",
            MemoryError::VersionConflict { .. } => "VERSION_CONFLICT",
            MemoryError::TreeConflict { .. } => "TREE_CONFLICT",
            MemoryError::PathExists(_) => "PATH_EXISTS",
            MemoryError::InvalidPath(_) => "INVALID_PATH",
            MemoryError::InvalidText(_) => "INVALID_TEXT",
            MemoryError::InvalidRequest(_) => "INVALID_REQUEST",
            MemoryError::NoMatch => "NO_MATCH",
            MemoryError::MultipleMatches(_) => "MULTIPLE_MATCHES",
            MemoryError::Limit { .. } => "LIMIT_EXCEEDED",
            MemoryError::IdempotencyConflict => "IDEMPOTENCY_CONFLICT",
            MemoryError::IdempotencyKeyRetired => "IDEMPOTENCY_KEY_RETIRED",
            MemoryError::InTrash => "IN_TRASH",
            MemoryError::NotInTrash => "NOT_IN_TRASH",
            MemoryError::TrashExpired => "TRASH_EXPIRED",
            MemoryError::RevisionNotRetained(_) => "REVISION_NOT_RETAINED",
            MemoryError::NothingToUndo => "NOTHING_TO_UNDO",
            MemoryError::PermissionDenied(_) => "PERMISSION_DENIED",
            MemoryError::MergeConflict { .. } => "MERGE_CONFLICT",
            MemoryError::InstructionFileRefused { .. } => "INSTRUCTION_FILE_REFUSED",
            MemoryError::RequiredContextOverBudget(_) => "REQUIRED_CONTEXT_OVER_BUDGET",
            MemoryError::Budget(_) => "BUDGET",
            MemoryError::Storage(_) => "STORAGE",
        }
    }

    pub fn retryable(&self) -> bool {
        matches!(self, MemoryError::Storage(_))
    }

    pub fn storage(e: impl std::fmt::Display) -> Self {
        MemoryError::Storage(e.to_string())
    }
}

pub type Result<T, E = MemoryError> = std::result::Result<T, E>;
