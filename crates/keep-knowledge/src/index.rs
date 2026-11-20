//! The book-style index: a table of contents at three zoom levels.
//!
//! - Level 0 is always small enough for a standing context slot (640 tokens
//!   by default): one line per collection with counts and date span.
//! - Level 1 lists the sources in one collection (or every collection),
//!   with their top entities, paged by cursor.
//! - Level 2 is a back-of-book index: entities and aliases with pointers to
//!   the sources that mention them.
//!
//! Each visibility partition is indexed separately and only the partitions
//! a caller may read are merged, so a restricted source's title, its counts
//! and even its existence never shape another caller's view. A partition's
//! index is tagged with the ledger generation it was built from; any write
//! or purge bumps the generation, and a stale copy is rebuilt before use.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use keep_providers::Tokenizer;
use serde::{Deserialize, Serialize};

use crate::ledger::{EntityStat, IndexInputs, SourceStat};
use crate::model::{Time, digest_hex};

pub const LEVEL0_TOKENS: usize = 640;
pub const LEVEL1_TOKENS: usize = 1_000;
pub const LEVEL2_TOKENS: usize = 1_500;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexRequest {
    /// 0, 1 or 2.
    pub level: u8,
    /// Collection handle from a level-0 entry, to zoom into at level 1.
    pub collection: Option<String>,
    /// Name prefix filter at level 2.
    pub prefix: Option<String>,
    /// Entry offset returned as `next_cursor` by the previous page.
    pub cursor: Option<usize>,
    /// Override the level's token budget (never above it).
    pub budget_tokens: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coverage {
    pub collections: usize,
    pub sources: usize,
    pub episodes: i64,
    pub facts: i64,
    pub entities: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexEntry {
    /// Opaque handle to zoom in or fetch: a collection handle, source id or
    /// entity id.
    pub handle: String,
    pub kind: String,
    pub label: String,
    pub sources: usize,
    pub episodes: i64,
    pub facts: i64,
    pub first_at: Option<Time>,
    pub last_at: Option<Time>,
    pub aliases: Vec<String>,
    /// Source ids this entry points to (level 2).
    pub pointers: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexView {
    pub level: u8,
    pub text: String,
    pub tokens: usize,
    pub entries: Vec<IndexEntry>,
    /// Digest of the generations of every partition in view; changes after
    /// any write the caller could see.
    pub generation: String,
    pub coverage: Coverage,
    pub generated_at: Time,
    /// Latest episode time covered.
    pub content_through: Option<Time>,
    pub truncated: bool,
    pub next_cursor: Option<usize>,
}

/// Optional prose summaries for collections. Off by default; nothing in the
/// index depends on it, and any implementation must be given only one
/// partition's entries at a time.
#[async_trait]
pub trait IndexSummarizer: Send + Sync {
    async fn summarize(&self, label: &str, entries: &[IndexEntry]) -> Option<String>;
}

pub struct DisabledSummarizer;

#[async_trait]
impl IndexSummarizer for DisabledSummarizer {
    async fn summarize(&self, _label: &str, _entries: &[IndexEntry]) -> Option<String> {
        None
    }
}

/// Canonical order so builds are byte-identical for the same ledger state.
pub fn normalize(mut inputs: IndexInputs) -> IndexInputs {
    inputs.sources.sort_by(|a, b| {
        a.collection
            .cmp(&b.collection)
            .then_with(|| a.title.cmp(&b.title))
            .then_with(|| a.id.cmp(&b.id))
    });
    for e in &mut inputs.entities {
        e.aliases.sort();
        e.aliases.dedup();
        e.sources.sort();
    }
    inputs.entities.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.id.cmp(&b.id))
    });
    inputs
}

pub fn collection_handle(corpus: &str, collection: &str) -> String {
    format!("col_{}", &digest_hex(&[corpus, collection])[..16])
}

fn date(t: Option<Time>) -> String {
    t.map_or_else(|| "undated".into(), |t| t.date_naive().to_string())
}

fn span(first: Option<Time>, last: Option<Time>) -> String {
    match (first, last) {
        (None, None) => "undated".into(),
        (a, b) if a == b => date(a),
        (a, b) => format!("{}..{}", date(a), date(b)),
    }
}

fn plural(n: impl Into<i64>, word: &str) -> String {
    let n = n.into();
    format!("{n} {word}{}", if n == 1 { "" } else { "s" })
}

/// Merge the accessible partitions into one view.
pub fn render(
    corpus: &str,
    parts: &[(String, IndexInputs)],
    req: &IndexRequest,
    tokenizer: &dyn Tokenizer,
    now: Time,
) -> IndexView {
    let sources: Vec<&SourceStat> = parts.iter().flat_map(|(_, p)| p.sources.iter()).collect();
    let entities: Vec<&EntityStat> = parts.iter().flat_map(|(_, p)| p.entities.iter()).collect();

    let mut collections: BTreeMap<&str, IndexEntry> = BTreeMap::new();
    for s in &sources {
        let e = collections
            .entry(s.collection.as_str())
            .or_insert_with(|| IndexEntry {
                handle: collection_handle(corpus, &s.collection),
                kind: "collection".into(),
                label: s.collection.clone(),
                sources: 0,
                episodes: 0,
                facts: 0,
                first_at: None,
                last_at: None,
                aliases: vec![],
                pointers: vec![],
            });
        e.sources += 1;
        e.episodes += s.episodes;
        e.facts += s.facts;
        e.first_at = min_opt(e.first_at, s.first_at);
        e.last_at = e.last_at.max(s.last_at);
    }
    let coverage = Coverage {
        collections: collections.len(),
        sources: sources.len(),
        episodes: sources.iter().map(|s| s.episodes).sum(),
        facts: sources.iter().map(|s| s.facts).sum(),
        entities: entities.len(),
    };
    let content_through = sources.iter().filter_map(|s| s.last_at).max();
    let mut gens: Vec<String> = parts
        .iter()
        .map(|(p, i)| format!("{p}={}", i.generation))
        .collect();
    gens.sort();
    let gen_refs: Vec<&str> = gens.iter().map(String::as_str).collect();
    let generation = digest_hex(&gen_refs)[..16].to_string();

    let level = req.level.min(2);
    let cap = match level {
        0 => LEVEL0_TOKENS,
        1 => LEVEL1_TOKENS,
        _ => LEVEL2_TOKENS,
    };
    let budget = req.budget_tokens.map_or(cap, |b| b.min(cap));

    let (header, entries, lines): (String, Vec<IndexEntry>, Vec<String>) = match level {
        0 | 1 if req.collection.is_none() || level == 0 => {
            let mut list: Vec<IndexEntry> = collections.into_values().collect();
            // Most recent activity first, then by name.
            list.sort_by(|a, b| b.last_at.cmp(&a.last_at).then(a.label.cmp(&b.label)));
            let lines = list
                .iter()
                .map(|c| {
                    format!(
                        "- {} [{}]: {}, {}, {}, {}",
                        c.label,
                        c.handle,
                        plural(c.sources as i64, "source"),
                        plural(c.episodes, "episode"),
                        plural(c.facts, "fact"),
                        span(c.first_at, c.last_at)
                    )
                })
                .collect();
            let header = format!(
                "Knowledge index, level {level}: {}, {}, {}, {}; through {}.",
                plural(coverage.collections as i64, "collection"),
                plural(coverage.sources as i64, "source"),
                plural(coverage.facts, "fact"),
                plural(coverage.entities as i64, "entity"),
                date(content_through)
            )
            .replace("entitys", "entities");
            (header, list, lines)
        }
        1 => {
            let handle = req.collection.clone().unwrap_or_default();
            let in_col: Vec<&&SourceStat> = sources
                .iter()
                .filter(|s| collection_handle(corpus, &s.collection) == handle)
                .collect();
            let mut list: Vec<IndexEntry> = in_col
                .iter()
                .map(|s| IndexEntry {
                    handle: s.id.clone(),
                    kind: "source".into(),
                    label: s.title.clone(),
                    sources: 1,
                    episodes: s.episodes,
                    facts: s.facts,
                    first_at: s.first_at,
                    last_at: s.last_at,
                    aliases: vec![],
                    pointers: vec![],
                })
                .collect();
            list.sort_by(|a, b| {
                b.last_at
                    .cmp(&a.last_at)
                    .then(a.label.cmp(&b.label))
                    .then(a.handle.cmp(&b.handle))
            });
            let ids: BTreeSet<&str> = in_col.iter().map(|s| s.id.as_str()).collect();
            let mut top: Vec<&&EntityStat> = entities
                .iter()
                .filter(|e| e.sources.iter().any(|s| ids.contains(s.as_str())))
                .collect();
            top.sort_by(|a, b| b.facts.cmp(&a.facts).then(a.name.cmp(&b.name)));
            let top: Vec<String> = top
                .iter()
                .take(8)
                .map(|e| format!("{} ({})", e.name, e.facts))
                .collect();
            let label = in_col.first().map_or_else(
                || "unknown collection".to_string(),
                |s| s.collection.clone(),
            );
            let mut header = format!(
                "Knowledge index, level 1: {label}, {}.",
                plural(list.len() as i64, "source")
            );
            if !top.is_empty() {
                header.push_str(&format!(" Top entities: {}.", top.join(", ")));
            }
            let lines = list
                .iter()
                .map(|s| {
                    format!(
                        "- {} [{}]: {}, {}, {}",
                        s.label,
                        s.handle,
                        plural(s.episodes, "episode"),
                        plural(s.facts, "fact"),
                        span(s.first_at, s.last_at)
                    )
                })
                .collect();
            (header, list, lines)
        }
        _ => {
            let prefix = req.prefix.as_deref().map(str::to_lowercase);
            let mut list: Vec<IndexEntry> = entities
                .iter()
                .filter(|e| {
                    prefix.as_deref().is_none_or(|p| {
                        e.name.to_lowercase().starts_with(p)
                            || e.aliases.iter().any(|a| a.to_lowercase().starts_with(p))
                    })
                })
                .map(|e| IndexEntry {
                    handle: e.id.clone(),
                    kind: e.kind.clone(),
                    label: e.name.clone(),
                    sources: e.sources.len(),
                    episodes: 0,
                    facts: e.facts,
                    first_at: None,
                    last_at: None,
                    aliases: e.aliases.clone(),
                    pointers: e.sources.clone(),
                })
                .collect();
            list.sort_by(|a, b| {
                a.label
                    .to_lowercase()
                    .cmp(&b.label.to_lowercase())
                    .then(a.handle.cmp(&b.handle))
            });
            let lines = list
                .iter()
                .map(|e| {
                    let aka = if e.aliases.is_empty() {
                        String::new()
                    } else {
                        format!(" (also {})", e.aliases.join(", "))
                    };
                    format!(
                        "- {}{aka} [{}]: {} in {}; see {}",
                        e.label,
                        e.handle,
                        plural(e.facts, "fact"),
                        plural(e.sources as i64, "source"),
                        e.pointers.join(", ")
                    )
                })
                .collect();
            let header = format!(
                "Knowledge index, level 2: {}.",
                plural(list.len() as i64, "entity")
            )
            .replace("entitys", "entities");
            (header, list, lines)
        }
    };

    let start = req.cursor.unwrap_or(0).min(lines.len());
    let mut text = header;
    let mut tokens = tokenizer.count(&text);
    let mut shown = Vec::new();
    let mut i = start;
    while i < lines.len() {
        // Leave room for the "+N more" footer.
        let more = format!(
            "\n+{} more; zoom in or page with cursor {i}.",
            lines.len() - i
        );
        let candidate = format!("{text}\n{}", lines[i]);
        let after = tokenizer.count(&candidate);
        let footer_tokens = if i + 1 < lines.len() {
            tokenizer.count(&more)
        } else {
            0
        };
        if after + footer_tokens > budget {
            break;
        }
        text = candidate;
        tokens = after;
        shown.push(entries[i].clone());
        i += 1;
    }
    let truncated = i < lines.len();
    if truncated {
        let footer = format!(
            "\n+{} more; zoom in or page with cursor {i}.",
            lines.len() - i
        );
        if tokenizer.count(&format!("{text}{footer}")) <= budget {
            text.push_str(&footer);
            tokens = tokenizer.count(&text);
        }
    }
    IndexView {
        level,
        text,
        tokens,
        entries: shown,
        generation,
        coverage,
        generated_at: now,
        content_through,
        truncated,
        next_cursor: truncated.then_some(i),
    }
}

fn min_opt(a: Option<Time>, b: Option<Time>) -> Option<Time> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.min(y)),
        (x, None) => x,
        (None, y) => y,
    }
}
