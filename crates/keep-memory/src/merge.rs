//! Disjoint-entry merge for stale text edits.
//!
//! A client that edited revision `base` while someone else saved `theirs`
//! gets a version conflict under plain compare-and-swap. When the two sides
//! touched different entries, the edit can be replayed safely instead:
//!
//! 1. apply the client's edit to `base` and align the result with `base`'s
//!    entries, which says which base entries the client changed, removed, or
//!    inserted new entries after;
//! 2. compare `theirs` with `base` by entry id the same way;
//! 3. if no base entry (or insertion point) is touched by both sides, start
//!    from `theirs` and apply the client's changes entry by entry.
//!
//! Anything else is a [`MemoryError::MergeConflict`] naming the entries, and
//! nothing is written. The merged text then goes through the normal write
//! path, so limits, permissions and lineage rules all still apply.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use chrono::{DateTime, Utc};

use crate::align::{AlignCtx, align};
use crate::codec::{self, EntryMode};
use crate::error::{MemoryError, Result};
use crate::model::Entry;

/// Where an inserted entry goes: after a base entry, or at the start.
const START: &str = "";

#[derive(Default)]
struct Changes {
    /// Base entry ids whose text changed, with the new text.
    modified: BTreeMap<String, String>,
    removed: BTreeSet<String>,
    /// Texts inserted after each anchor, in order.
    inserted: BTreeMap<String, Vec<String>>,
}

impl Changes {
    /// Every base id this side depends on staying as it was.
    fn touched(&self) -> BTreeSet<&str> {
        self.modified
            .keys()
            .chain(self.removed.iter())
            .chain(self.inserted.keys())
            .map(String::as_str)
            .collect()
    }
}

fn diff(base: &[Entry], side: &[(String, String)]) -> Changes {
    let base_text: HashMap<&str, &str> = base
        .iter()
        .map(|e| (e.entry_id.as_str(), e.text.as_str()))
        .collect();
    let mut c = Changes::default();
    let present: BTreeSet<&str> = side.iter().map(|(id, _)| id.as_str()).collect();
    for e in base {
        if !present.contains(e.entry_id.as_str()) {
            c.removed.insert(e.entry_id.clone());
        }
    }
    let mut anchor = START.to_owned();
    for (id, text) in side {
        match base_text.get(id.as_str()) {
            Some(old) => {
                if old != text {
                    c.modified.insert(id.clone(), text.clone());
                }
                anchor = id.clone();
            }
            None => c
                .inserted
                .entry(anchor.clone())
                .or_default()
                .push(text.clone()),
        }
    }
    c
}

/// Merge `ours_text` (the client's edit of `base`) onto `theirs`. Returns the
/// merged visible text.
pub fn merge_text(
    base: &[Entry],
    ours_text: &str,
    theirs: &[Entry],
    mode: EntryMode,
    now: DateTime<Utc>,
) -> Result<String> {
    let mut n = 0u64;
    let mut next_id = || {
        n += 1;
        format!("\u{0}ours-{n}")
    };
    let no_deps = BTreeSet::new();
    let ours: Vec<(String, String)> = align(
        base,
        ours_text,
        mode,
        AlignCtx {
            now,
            actor: "merge",
            next_id: &mut next_id,
            dependencies: &no_deps,
        },
    )
    .into_iter()
    .map(|e| (e.entry_id, e.text))
    .collect();
    let theirs_pairs: Vec<(String, String)> = theirs
        .iter()
        .map(|e| (e.entry_id.clone(), e.text.clone()))
        .collect();
    let ours = diff(base, &ours);
    let theirs_changes = diff(base, &theirs_pairs);

    let overlap: Vec<String> = ours
        .touched()
        .intersection(&theirs_changes.touched())
        .map(|s| {
            if s.is_empty() {
                "(start)".into()
            } else {
                (*s).to_owned()
            }
        })
        .collect();
    if !overlap.is_empty() {
        return Err(MemoryError::MergeConflict {
            entries: overlap,
            reason: "both edits changed the same entries".into(),
        });
    }

    let mut out: Vec<String> = Vec::new();
    let emit_inserts = |anchor: &str, out: &mut Vec<String>| {
        if let Some(texts) = ours.inserted.get(anchor) {
            out.extend(texts.iter().cloned());
        }
    };
    emit_inserts(START, &mut out);
    for (id, text) in &theirs_pairs {
        if ours.removed.contains(id) {
            continue;
        }
        out.push(ours.modified.get(id).unwrap_or(text).clone());
        emit_inserts(id, &mut out);
    }
    let merged = codec::render(out.iter().map(String::as_str));
    // Entries need not end in a newline (the last one of a file, say); if
    // joining moved a boundary, the result is not a clean replay.
    if codec::render(
        codec::parse(&merged, mode)
            .iter()
            .map(|b| &merged[b.start..b.end]),
    ) != merged
    {
        return Err(MemoryError::MergeConflict {
            entries: Vec::new(),
            reason: "the merged text does not split back into the same entries".into(),
        });
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::{AlignCtx, align};

    fn entries(text: &str, prior: &[Entry]) -> Vec<Entry> {
        let mut n = prior.len() as u64 * 100;
        let mut next = || {
            n += 1;
            format!("e{n}")
        };
        let deps = BTreeSet::new();
        align(
            prior,
            text,
            EntryMode::Sections,
            AlignCtx {
                now: Utc::now(),
                actor: "t",
                next_id: &mut next,
                dependencies: &deps,
            },
        )
    }

    const BASE: &str = "# A\none\n\n# B\ntwo\n\n# C\nthree\n";

    #[test]
    fn disjoint_sections_merge() {
        let base = entries(BASE, &[]);
        let theirs = entries(&BASE.replace("one", "ONE"), &base);
        let ours = BASE.replace("three", "THREE") + "\n# D\nfour\n";
        let merged = merge_text(&base, &ours, &theirs, EntryMode::Sections, Utc::now()).unwrap();
        assert_eq!(merged, "# A\nONE\n\n# B\ntwo\n\n# C\nTHREE\n\n# D\nfour\n");
    }

    #[test]
    fn same_section_conflicts() {
        let base = entries(BASE, &[]);
        let theirs = entries(&BASE.replace("two", "2"), &base);
        let ours = BASE.replace("two", "deux");
        let err = merge_text(&base, &ours, &theirs, EntryMode::Sections, Utc::now()).unwrap_err();
        assert!(
            matches!(err, MemoryError::MergeConflict { ref entries, .. } if entries.len() == 1)
        );
    }

    #[test]
    fn removed_by_them_and_edited_by_us_conflicts() {
        let base = entries(BASE, &[]);
        let theirs = entries(&BASE.replace("# B\ntwo\n\n", ""), &base);
        let ours = BASE.replace("two", "deux");
        assert!(merge_text(&base, &ours, &theirs, EntryMode::Sections, Utc::now()).is_err());
    }

    #[test]
    fn inserts_at_the_same_place_conflict() {
        let base = entries(BASE, &[]);
        let theirs = entries(&format!("{BASE}\n# X\nx\n"), &base);
        let ours = format!("{BASE}\n# Y\ny\n");
        assert!(merge_text(&base, &ours, &theirs, EntryMode::Sections, Utc::now()).is_err());
    }
}
