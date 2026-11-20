//! Entry identity alignment.
//!
//! When a file is rewritten as Markdown, the new text is parsed into blocks
//! and each block is matched to an entry of the previous revision so that
//! identities, annotations and lineage survive ordinary edits:
//!
//! 1. a block whose exact text belongs to exactly one unmatched old entry
//!    keeps that entry's identity;
//! 2. otherwise a block whose structural key (heading line or first bullet
//!    line) is unique among both sides keeps that identity;
//! 3. otherwise, between two matched neighbours, a single unmatched old entry
//!    and a single unmatched new block of the same kind are paired.
//!
//! Everything else becomes a new entry. A new entry inherits the protected
//! dependencies of every entry removed in the same edit, so rewriting text
//! can never strip lineage that restricts who may read it.

use std::collections::{BTreeSet, HashMap};

use chrono::{DateTime, Utc};

use crate::codec::{self, Block, EntryMode};
use crate::model::{Annotation, Entry, Lineage};

pub struct AlignCtx<'a> {
    pub now: DateTime<Utc>,
    pub actor: &'a str,
    /// Produces fresh entry ids.
    pub next_id: &'a mut dyn FnMut() -> String,
    /// Dependencies attached to every entry written by this operation.
    pub dependencies: &'a BTreeSet<String>,
}

/// Parse `text` and align the resulting blocks with `old` entries.
pub fn align(old: &[Entry], text: &str, mode: EntryMode, cx: AlignCtx<'_>) -> Vec<Entry> {
    let blocks = codec::parse(text, mode);
    let frag = |b: &Block| &text[b.start..b.end];
    let mut matched_old: Vec<Option<usize>> = vec![None; old.len()];
    let mut matched_new: Vec<Option<usize>> = vec![None; blocks.len()];

    // 1. Unique exact text.
    let mut old_by_text: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, e) in old.iter().enumerate() {
        old_by_text.entry(e.text.as_str()).or_default().push(i);
    }
    let mut new_by_text: HashMap<&str, Vec<usize>> = HashMap::new();
    for (j, b) in blocks.iter().enumerate() {
        new_by_text.entry(frag(b)).or_default().push(j);
    }
    for (t, js) in &new_by_text {
        if let (Some(is), [j]) = (old_by_text.get(t), js.as_slice())
            && let [i] = is.as_slice()
        {
            matched_old[*i] = Some(*j);
            matched_new[*j] = Some(*i);
        }
    }
    // Repeated identical fragments pair up in order.
    for (t, js) in &new_by_text {
        if let Some(is) = old_by_text.get(t)
            && js.len() > 1
        {
            let free_old = is.iter().filter(|i| matched_old[**i].is_none());
            let free_new = js.iter().filter(|j| matched_new[**j].is_none());
            let pairs: Vec<(usize, usize)> = free_old.copied().zip(free_new.copied()).collect();
            for (i, j) in pairs {
                matched_old[i] = Some(j);
                matched_new[j] = Some(i);
            }
        }
    }

    // 2. Unique structural key among unmatched entries.
    let mut old_by_key: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, e) in old.iter().enumerate() {
        if matched_old[i].is_none()
            && let Some(k) = entry_key(e, mode)
        {
            old_by_key.entry(k).or_default().push(i);
        }
    }
    let mut new_by_key: HashMap<String, Vec<usize>> = HashMap::new();
    for (j, b) in blocks.iter().enumerate() {
        if matched_new[j].is_none()
            && let Some(k) = &b.key
        {
            new_by_key.entry(k.clone()).or_default().push(j);
        }
    }
    for (k, js) in &new_by_key {
        if let (Some([i]), [j]) = (old_by_key.get(k).map(Vec::as_slice), js.as_slice())
            && old[*i].kind == blocks[*j].kind
        {
            matched_old[*i] = Some(*j);
            matched_new[*j] = Some(*i);
        }
    }

    // 3. Single-candidate gaps between monotonic anchors.
    let pairs: Vec<(usize, usize)> = matched_new
        .iter()
        .enumerate()
        .filter_map(|(j, m)| m.map(|i| (i, j)))
        .collect();
    let anchors = longest_increasing(&pairs);
    let mut bounds = vec![(None, None)];
    bounds.extend(anchors.iter().map(|(i, j)| (Some(*i), Some(*j))));
    bounds.push((None, None));
    for w in bounds.windows(2) {
        let (lo_i, lo_j) = w[0];
        let (hi_i, hi_j) = w[1];
        let in_range = |x: usize, lo: Option<usize>, hi: Option<usize>, len: usize| {
            lo.is_none_or(|l| x > l) && x < hi.unwrap_or(len)
        };
        let free_old: Vec<usize> = (0..old.len())
            .filter(|i| matched_old[*i].is_none() && in_range(*i, lo_i, hi_i, old.len()))
            .collect();
        let free_new: Vec<usize> = (0..blocks.len())
            .filter(|j| matched_new[*j].is_none() && in_range(*j, lo_j, hi_j, blocks.len()))
            .collect();
        if let ([i], [j]) = (free_old.as_slice(), free_new.as_slice())
            && old[*i].kind == blocks[*j].kind
        {
            matched_old[*i] = Some(*j);
            matched_new[*j] = Some(*i);
        }
    }
    // A single leftover on each side is the same entry, wherever it moved.
    let free_old: Vec<usize> = (0..old.len())
        .filter(|i| matched_old[*i].is_none())
        .collect();
    let free_new: Vec<usize> = (0..blocks.len())
        .filter(|j| matched_new[*j].is_none())
        .collect();
    if let ([i], [j]) = (free_old.as_slice(), free_new.as_slice())
        && old[*i].kind == blocks[*j].kind
    {
        matched_old[*i] = Some(*j);
        matched_new[*j] = Some(*i);
    }

    // Lineage of removed entries flows to every new entry.
    let mut inherited: BTreeSet<String> = cx.dependencies.clone();
    for (i, e) in old.iter().enumerate() {
        if matched_old[i].is_none() {
            inherited.extend(e.lineage.dependencies.iter().cloned());
        }
    }

    let mut out = Vec::with_capacity(blocks.len());
    for (j, b) in blocks.iter().enumerate() {
        let text = frag(b).to_owned();
        match matched_new[j] {
            Some(i) => {
                let mut e = old[i].clone();
                e.kind = b.kind;
                if e.text != text {
                    e.text = text;
                    e.updated_at = cx.now;
                    e.lineage.updated_by = cx.actor.to_owned();
                    e.lineage.dependencies.extend(inherited.iter().cloned());
                }
                e.lineage
                    .dependencies
                    .extend(cx.dependencies.iter().cloned());
                out.push(e);
            }
            None => out.push(Entry {
                entry_id: (cx.next_id)(),
                kind: b.kind,
                text,
                created_at: cx.now,
                updated_at: cx.now,
                lineage: Lineage {
                    created_by: cx.actor.to_owned(),
                    updated_by: cx.actor.to_owned(),
                    dependencies: inherited.clone(),
                },
                annotation: Annotation::default(),
            }),
        }
    }
    out
}

/// The longest subsequence of `(old, new)` pairs (already sorted by `new`)
/// whose `old` indices increase. Files hold at most a few hundred entries,
/// so the quadratic version is fine.
fn longest_increasing(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let mut len = vec![1usize; pairs.len()];
    let mut prev = vec![usize::MAX; pairs.len()];
    for b in 0..pairs.len() {
        for a in 0..b {
            if pairs[a].0 < pairs[b].0 && len[a] + 1 > len[b] {
                len[b] = len[a] + 1;
                prev[b] = a;
            }
        }
    }
    let mut at = (0..pairs.len())
        .max_by_key(|i| (len[*i], usize::MAX - *i))
        .unwrap_or(0);
    let mut out = vec![pairs[at]];
    while prev[at] != usize::MAX {
        at = prev[at];
        out.push(pairs[at]);
    }
    out.reverse();
    out
}

fn entry_key(e: &Entry, mode: EntryMode) -> Option<String> {
    // Re-derive the key from the stored fragment so it matches `parse`.
    codec::parse(&e.text, mode)
        .first()
        .and_then(|b| b.key.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::EntryKind;

    fn run(old: &[Entry], text: &str, deps: &[&str], counter: &mut u32) -> Vec<Entry> {
        let deps: BTreeSet<String> = deps.iter().map(|s| s.to_string()).collect();
        let mut next = || {
            *counter += 1;
            format!("ent_{counter}")
        };
        align(
            old,
            text,
            EntryMode::Sections,
            AlignCtx {
                now: DateTime::UNIX_EPOCH,
                actor: "tester",
                next_id: &mut next,
                dependencies: &deps,
            },
        )
    }

    #[test]
    fn edits_keep_identities_and_moves_follow_text() {
        let mut n = 0;
        let v1 = run(&[], "intro\n# A\none\n# B\ntwo\n# C\nthree\n", &[], &mut n);
        let ids: Vec<_> = v1.iter().map(|e| e.entry_id.clone()).collect();
        assert_eq!(v1.len(), 4);
        assert_eq!(v1[1].kind, EntryKind::Section);

        // Edit B's body, move C before A.
        let v2 = run(
            &v1,
            "intro\n# C\nthree\n# A\none\n# B\ntwo, edited\n",
            &[],
            &mut n,
        );
        let ids2: Vec<_> = v2.iter().map(|e| e.entry_id.clone()).collect();
        assert_eq!(
            ids2,
            vec![
                ids[0].clone(),
                ids[3].clone(),
                ids[1].clone(),
                ids[2].clone()
            ]
        );

        // Rename a heading and change its body: still paired by position.
        let v3 = run(
            &v2,
            "intro\n# C\nthree\n# A2\nuno\n# B\ntwo, edited\n",
            &[],
            &mut n,
        );
        assert_eq!(v3[2].entry_id, ids[1]);
        assert_eq!(
            v3.iter().map(|e| e.text.as_str()).collect::<String>(),
            "intro\n# C\nthree\n# A2\nuno\n# B\ntwo, edited\n"
        );
    }

    #[test]
    fn rewriting_text_never_drops_lineage() {
        let mut n = 0;
        let v1 = run(
            &[],
            "# Private\nfrom a restricted source\n",
            &["source:restricted"],
            &mut n,
        );
        assert!(v1[0].lineage.dependencies.contains("source:restricted"));
        // Replace everything with unrelated text: the new entry inherits it.
        let v2 = run(&v1, "fresh\n\n# Other\nunrelated\n", &[], &mut n);
        assert!(
            v2.iter()
                .all(|e| e.lineage.dependencies.contains("source:restricted"))
        );
    }
}
