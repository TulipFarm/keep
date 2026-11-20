//! MEMORY record types.
//!
//! Memory is a small, curated notebook. Each file is shown to agents as
//! Markdown, but the canonical record is a structured revision with stable
//! entry identities. The harness only needs the shapes and the limits so that
//! later ablation arms can be described and budgeted; parsing, rendering,
//! compare-and-swap and history live in later crates.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ContentHash;

/// Default size limits for one Memory tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryLimits {
    pub max_rendered_bytes_per_file: usize,
    pub max_files_per_tree: usize,
    pub max_rendered_bytes_per_tree: usize,
    pub max_entries_per_file: usize,
    pub max_metadata_bytes_per_file: usize,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_rendered_bytes_per_file: 8 * 1024,
            max_files_per_tree: 128,
            max_rendered_bytes_per_tree: 512 * 1024,
            max_entries_per_file: 128,
            max_metadata_bytes_per_file: 64 * 1024,
        }
    }
}

/// How a category asks to be treated during context assembly. This is a
/// request, never an access grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BehaviorHint {
    /// Considered for every eligible context bundle.
    Profile,
    /// May be selected when relevant.
    Contextual,
    /// Listed in the directory but only read on explicit request.
    Manual,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Category {
    /// `None` is the uncategorised bucket.
    pub id: Option<String>,
    pub display_name: String,
    pub sort_order: i32,
    pub behavior_hint: BehaviorHint,
}

impl Category {
    /// The default category set for a new tree.
    pub fn defaults() -> Vec<Category> {
        let c = |id: Option<&str>, name: &str, order, hint| Category {
            id: id.map(str::to_owned),
            display_name: name.to_owned(),
            sort_order: order,
            behavior_hint: hint,
        };
        vec![
            c(Some("you"), "You", 0, BehaviorHint::Profile),
            c(Some("topics"), "Topics", 1, BehaviorHint::Contextual),
            c(Some("areas"), "Areas", 2, BehaviorHint::Contextual),
            c(Some("people"), "People", 3, BehaviorHint::Contextual),
            c(Some("projects"), "Projects", 4, BehaviorHint::Contextual),
            c(None, "Other", 5, BehaviorHint::Contextual),
        ]
    }
}

/// A validated, normalised path inside a Memory tree, such as `/profile.md`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct MemoryPath(String);

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum PathError {
    #[error("memory paths must start with '/'")]
    NotAbsolute,
    #[error("memory paths must end in '.md'")]
    NotMarkdown,
    #[error("memory path segment {0:?} is not allowed")]
    BadSegment(String),
    #[error("memory path is longer than {0} bytes")]
    TooLong(usize),
}

impl MemoryPath {
    pub const MAX_LEN: usize = 512;

    pub fn parse(raw: &str) -> Result<Self, PathError> {
        if raw.len() > Self::MAX_LEN {
            return Err(PathError::TooLong(Self::MAX_LEN));
        }
        let rest = raw.strip_prefix('/').ok_or(PathError::NotAbsolute)?;
        if !raw.ends_with(".md") {
            return Err(PathError::NotMarkdown);
        }
        for segment in rest.split('/') {
            let bad = segment.is_empty()
                || segment == "."
                || segment == ".."
                || segment.starts_with('.')
                || segment
                    .chars()
                    .any(|c| c.is_control() || c == '\\' || c == ':');
            if bad {
                return Err(PathError::BadSegment(segment.to_owned()));
            }
        }
        Ok(Self(raw.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for MemoryPath {
    type Error = PathError;
    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl From<MemoryPath> for String {
    fn from(value: MemoryPath) -> Self {
        value.0
    }
}

/// File metadata and current head.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryFile {
    pub memory_id: String,
    pub tree_id: String,
    pub path: MemoryPath,
    pub category_id: Option<String>,
    pub display_name: String,
    pub description: String,
    pub head_revision: String,
}

/// An immutable revision. `text` is the exact visible Markdown.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryRevision {
    pub revision_id: String,
    pub memory_id: String,
    pub parent: Option<String>,
    pub text: String,
    pub text_hash: ContentHash,
    pub entries: Vec<MemoryEntry>,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    Section,
    Bullet,
    Opaque,
}

/// A stable entry inside a revision, addressed by byte span of `text`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub entry_id: String,
    pub kind: EntryKind,
    pub start: usize,
    pub end: usize,
    /// Optional descriptive annotation. Unknown values stay absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotation: Option<EntryAnnotation>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EntryAnnotation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_locators: Vec<String>,
}

impl MemoryRevision {
    /// Check that entries reference in-bounds, ordered, non-overlapping spans
    /// on character boundaries.
    pub fn validate_entries(&self) -> Result<(), String> {
        let mut last_end = 0;
        for entry in &self.entries {
            if entry.start > entry.end || entry.end > self.text.len() {
                return Err(format!("entry {} span is out of bounds", entry.entry_id));
            }
            if entry.start < last_end {
                return Err(format!(
                    "entry {} overlaps the previous entry",
                    entry.entry_id
                ));
            }
            if !self.text.is_char_boundary(entry.start) || !self.text.is_char_boundary(entry.end) {
                return Err(format!("entry {} splits a character", entry.entry_id));
            }
            last_end = entry.end;
        }
        if self.text_hash != ContentHash::of_bytes(self.text.as_bytes()) {
            return Err("text_hash does not match text".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_paths_reject_escapes_and_hidden_segments() {
        assert!(MemoryPath::parse("/profile.md").is_ok());
        assert!(MemoryPath::parse("/projects/p-1/index.md").is_ok());
        assert_eq!(MemoryPath::parse("profile.md"), Err(PathError::NotAbsolute));
        assert_eq!(MemoryPath::parse("/notes.txt"), Err(PathError::NotMarkdown));
        assert!(matches!(
            MemoryPath::parse("/../etc.md"),
            Err(PathError::BadSegment(_))
        ));
        assert!(matches!(
            MemoryPath::parse("/a//b.md"),
            Err(PathError::BadSegment(_))
        ));
        assert!(matches!(
            MemoryPath::parse("/.obsidian/x.md"),
            Err(PathError::BadSegment(_))
        ));
        assert!(matches!(
            MemoryPath::parse("/c:/x.md"),
            Err(PathError::BadSegment(_))
        ));
    }

    #[test]
    fn revision_entry_validation_catches_bad_spans() {
        let text = "# Héllo\n\n- one\n".to_owned();
        let mut rev = MemoryRevision {
            revision_id: "rev_1".into(),
            memory_id: "mem_1".into(),
            parent: None,
            text_hash: ContentHash::of_bytes(text.as_bytes()),
            text,
            entries: vec![MemoryEntry {
                entry_id: "ent_1".into(),
                kind: EntryKind::Section,
                start: 0,
                end: 9,
                annotation: None,
            }],
            created_at: DateTime::<Utc>::UNIX_EPOCH,
        };
        assert!(rev.validate_entries().is_ok());
        rev.entries[0].end = 4; // inside the two-byte 'é'
        assert!(rev.validate_entries().unwrap_err().contains("splits"));
        rev.entries[0].end = 999;
        assert!(
            rev.validate_entries()
                .unwrap_err()
                .contains("out of bounds")
        );
    }
}
