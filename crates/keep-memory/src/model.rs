//! Persisted MEMORY records.
//!
//! These are the canonical shapes every ledger stores. Visible Markdown is
//! the concatenation of a revision's entry fragments; everything else
//! (identities, annotations, protected lineage) lives beside the text, never
//! inside it.

use std::collections::BTreeSet;

use chrono::{DateTime, Utc};
use keep_core::memory::Category;
use serde::{Deserialize, Serialize};

use crate::codec::{EntryKind, EntryMode};

/// Descriptive confidence. It is a caller's description, never a trust level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    Low,
    Medium,
    High,
}

/// Optional descriptive annotations on an entry. Unknown values stay absent;
/// nothing here is guessed from ingestion time. Callers may edit these.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Annotation {
    /// Source locators, such as `turn:abc` or `knowledge://fact/123`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<Confidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl Annotation {
    pub fn is_empty(&self) -> bool {
        *self == Annotation::default()
    }

    /// Whether any descriptive value mentions `needle`.
    pub fn mentions(&self, needle: &str) -> bool {
        self.sources.iter().any(|s| s.contains(needle))
            || self.note.as_deref().is_some_and(|n| n.contains(needle))
    }
}

/// Server-controlled lineage. Clients cannot set or remove it through
/// Markdown, annotations, sidecars or imports.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lineage {
    pub created_by: String,
    pub updated_by: String,
    /// Source dependencies that restrict who may read this entry.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub dependencies: BTreeSet<String>,
}

/// One entry inside a revision: an exact text fragment plus identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Entry {
    pub entry_id: String,
    pub kind: EntryKind,
    /// The exact visible fragment.
    pub text: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub lineage: Lineage,
    #[serde(default, skip_serializing_if = "Annotation::is_empty")]
    pub annotation: Annotation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Create,
    Edit,
    Annotate,
    Metadata,
    Rename,
    Undo,
    Merge,
    Restore,
    Forget,
    Import,
}

impl ChangeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ChangeKind::Create => "create",
            ChangeKind::Edit => "edit",
            ChangeKind::Annotate => "annotate",
            ChangeKind::Metadata => "metadata",
            ChangeKind::Rename => "rename",
            ChangeKind::Undo => "undo",
            ChangeKind::Merge => "merge",
            ChangeKind::Restore => "restore",
            ChangeKind::Forget => "forget",
            ChangeKind::Import => "import",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "create" => ChangeKind::Create,
            "edit" => ChangeKind::Edit,
            "annotate" => ChangeKind::Annotate,
            "metadata" => ChangeKind::Metadata,
            "rename" => ChangeKind::Rename,
            "undo" => ChangeKind::Undo,
            "merge" => ChangeKind::Merge,
            "restore" => ChangeKind::Restore,
            "forget" => ChangeKind::Forget,
            "import" => ChangeKind::Import,
            _ => return None,
        })
    }
}

/// File metadata captured in every revision, so undo restores it too.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileMeta {
    pub path: String,
    pub category_id: Option<String>,
    pub display_name: String,
    pub description: String,
    pub project_id: Option<String>,
    pub entry_mode: EntryMode,
}

/// An immutable revision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Revision {
    pub revision_id: String,
    pub memory_id: String,
    pub tree_id: String,
    /// Monotonic per file; every successful distinct save advances it.
    pub seq: u64,
    pub parents: Vec<String>,
    pub change: ChangeKind,
    pub actor: String,
    pub harness: Option<String>,
    pub created_at: DateTime<Utc>,
    /// Hash of the visible text, metadata and descriptive annotations.
    pub version: String,
    pub meta: FileMeta,
    pub entries: Vec<Entry>,
}

impl Revision {
    pub fn text(&self) -> String {
        crate::codec::render(self.entries.iter().map(|e| e.text.as_str()))
    }

    pub fn rendered_len(&self) -> usize {
        self.entries.iter().map(|e| e.text.len()).sum()
    }

    /// Serialized size of the structured metadata (entry identities, times,
    /// annotations and lineage), charged against the per-file cap.
    pub fn metadata_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|e| {
                serde_json::to_vec(&(
                    &e.entry_id,
                    e.kind,
                    e.created_at,
                    e.updated_at,
                    &e.lineage,
                    &e.annotation,
                ))
                .map(|v| v.len())
                .unwrap_or(0)
            })
            .sum::<usize>()
            + serde_json::to_vec(&self.meta).map(|v| v.len()).unwrap_or(0)
    }

    /// Does any text or descriptive value of this revision contain `needle`?
    pub fn mentions(&self, needle: &str) -> bool {
        self.text().contains(needle)
            || self.meta.description.contains(needle)
            || self.meta.display_name.contains(needle)
            || self.entries.iter().any(|e| e.annotation.mentions(needle))
    }
}

/// The current state of a file: head pointer plus denormalised metadata for
/// listing. `deleted_at` marks a file in the recoverable trash.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileRecord {
    pub memory_id: String,
    pub tree_id: String,
    pub meta: FileMeta,
    pub head_revision_id: String,
    pub head_seq: u64,
    pub version: String,
    pub size_bytes: usize,
    pub entry_count: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub deleted_at: Option<DateTime<Utc>>,
    /// Every source dependency any entry of this file has ever carried.
    /// Removing text never removes a dependency from this set.
    pub dependencies: BTreeSet<String>,
}

impl FileRecord {
    pub fn head(&self) -> Head {
        Head {
            revision_id: self.head_revision_id.clone(),
            version: self.version.clone(),
        }
    }

    pub fn is_live(&self) -> bool {
        self.deleted_at.is_none()
    }
}

/// The compare-and-swap token for a file: revision id *and* version hash.
/// A hash alone misses A→B→A; a revision id alone is opaque to clients that
/// only kept the text.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Head {
    pub revision_id: String,
    pub version: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TreeRecord {
    pub tree_id: String,
    /// Owning scope, for example `user:alex` or `project:<uuid>`.
    pub scope: String,
    /// Advances on every structural change (create, rename, delete, restore,
    /// category change, forget).
    pub tree_revision: u64,
    /// Advances on every forget.
    pub erasure_generation: u64,
    pub categories: Vec<Category>,
    pub created_at: DateTime<Utc>,
}

/// What a replayed idempotency key returns. Metadata only: never text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptOutcome {
    pub memory_id: Option<String>,
    pub revision_id: Option<String>,
    pub version: Option<String>,
    pub seq: Option<u64>,
    pub tree_revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Receipt {
    pub tree_id: String,
    pub principal: String,
    pub key: String,
    pub operation: String,
    /// Keyed digest of the canonical request; never the request itself.
    /// Cleared when the receipt is retired by a forget.
    pub digest: Option<String>,
    pub memory_id: Option<String>,
    /// `None` once retired.
    pub outcome: Option<ReceiptOutcome>,
    pub created_at: DateTime<Utc>,
}

/// A content-free erasure record. It carries opaque identifiers only, so it
/// can outlive the content it erased and be replayed after a restore.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErasureRecord {
    /// Assigned by the journal.
    pub seq: u64,
    pub tree_id: String,
    pub memory_id: Option<String>,
    /// `file`, `entry` or `text`.
    pub selector: String,
    pub generation: u64,
    /// Revisions that held the erased content.
    pub purged_revisions: Vec<String>,
    pub recorded_at: DateTime<Utc>,
}
