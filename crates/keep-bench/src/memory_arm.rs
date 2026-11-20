//! The Keep Memory (M) arm.
//!
//! A deterministic, model-free curator writes the visible history into a
//! Memory tree, and `memory.context` assembles the standing context block
//! under the manifest's `standing_tokens` budget. The representation arms
//! write byte-identical Markdown and differ only in how it is stored
//! (whole-file raw text, sections, or sections plus bullets), which also
//! decides the units an excerpt can keep together.
//!
//! The curator is a stand-in for a model-driven one: it measures the
//! storage, selection and budget machinery, not curation quality.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use keep_core::knowledge::Role;
use keep_memory::context::LABEL_KNOWLEDGE;
use keep_memory::{
    ContextBudget, ContextRequest, CreateFile, EntryMode, InMemoryJournal, InMemoryLedger,
    KnowledgeToc, ManualClock, MemoryService, OpCtx, SequentialIds,
};
use keep_providers::Tokenizer;
use keep_store::{Bm25Index, Bm25Params, LexicalIndex};

use crate::ablation::Representation;
use crate::baselines::Built;
use crate::context::{BlockTokens, CONTEXT_FORMAT, ContextBundle, Slot, Truncation, render_turn};
use crate::dataset::{History, Query, Session};

pub const CURATOR_METHOD: &str = "rule-curator-v1";
const TREE: &str = "bench";
const SCOPE: &str = "user:bench";

/// Rendered bytes a notes file may reach before the curator starts another.
const FILE_SOFT_BYTES: usize = 7 * 1024;
/// Lines per notes file, so bullet entries stay under the per-file limit.
const FILE_MAX_LINES: usize = 96;
/// Longest single remembered line.
const LINE_MAX_BYTES: usize = 480;
const PROFILE_HEADING: &str = "# Prompt essentials\n";

/// First-person statements the curator treats as standing preferences.
const PROFILE_CUES: &[&str] = &[
    "i prefer",
    "i like",
    "i love",
    "i hate",
    "i dislike",
    "i'm allergic",
    "i am allergic",
    "my favorite",
    "my favourite",
    "my name is",
    "i live in",
    "i work as",
    "i'm vegetarian",
    "i'm vegan",
];

pub fn entry_mode(r: Representation) -> EntryMode {
    match r {
        Representation::Raw => EntryMode::Raw,
        Representation::Structured => EntryMode::Sections,
        Representation::StructuredBullets => EntryMode::SectionsAndBullets,
    }
}

pub fn method(r: Representation, standing_tokens: usize) -> String {
    format!(
        "{CURATOR_METHOD};turns=user+speakers;representation={};context=memory.context;standing_tokens={standing_tokens}",
        serde_json::to_value(r)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default()
    )
}

/// One curated line and the turn it came from, as a byte span of its file.
#[derive(Clone, Debug)]
struct Span {
    start: usize,
    end: usize,
    turn_id: String,
}

#[derive(Default)]
struct Draft {
    path: String,
    text: String,
    lines: usize,
    spans: Vec<Span>,
    updated: Option<DateTime<Utc>>,
}

impl Draft {
    fn new(path: String, heading: &str) -> Self {
        Self {
            path,
            text: heading.to_owned(),
            ..Self::default()
        }
    }

    fn push_line(&mut self, line: &str, turn_id: &str) {
        let start = self.text.len();
        self.text.push_str(line);
        self.spans.push(Span {
            start,
            end: self.text.len(),
            turn_id: turn_id.to_owned(),
        });
        self.lines += 1;
    }
}

struct Curated {
    svc: MemoryService,
    spans: HashMap<String, Vec<Span>>,
    entries: Vec<MemoryEntry>,
    visible_turns: usize,
    remembered_turns: usize,
}

/// One remembered line of the curated tree and the turn it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryEntry {
    pub turn_id: String,
    /// The line as the tree stores it, without its list marker.
    pub line: String,
}

/// Tokens the Knowledge table of contents costs inside the Memory context
/// block, framing included, exactly as `memory.context` charges it.
pub fn toc_block_tokens(toc: &str, tokenizer: &dyn Tokenizer) -> usize {
    let body = serde_json::json!({ "text": toc });
    tokenizer.count(&format!("\n{LABEL_KNOWLEDGE}\n{body}"))
}

/// Curated trees, reused while questions share a history and visibility.
#[derive(Default)]
pub struct MemoryArm {
    cache: HashMap<(String, usize, Representation), Curated>,
}

impl MemoryArm {
    async fn curated(
        &mut self,
        representation: Representation,
        history: &History,
        query: &Query,
    ) -> Result<&Curated, String> {
        let (sessions, _) = history.visible_sessions(query.asked_at);
        let key = (history.id.clone(), sessions.len(), representation);
        if !self.cache.contains_key(&key) {
            // Histories are processed one at a time; keep only the current one.
            self.cache.retain(|k, _| k.0 == history.id);
            let c = curate(&sessions, representation)
                .await
                .map_err(|e| format!("memory curation failed: {e}"))?;
            self.cache.insert(key.clone(), c);
        }
        Ok(&self.cache[&key])
    }

    /// Every line the curator kept for what was visible when `query` was
    /// asked, in tree order (oldest file first).
    pub async fn entries(
        &mut self,
        representation: Representation,
        history: &History,
        query: &Query,
    ) -> Result<Vec<MemoryEntry>, String> {
        Ok(self
            .curated(representation, history, query)
            .await?
            .entries
            .clone())
    }

    pub async fn build(
        &mut self,
        representation: Representation,
        standing_tokens: usize,
        history: &History,
        query: &Query,
        tokenizer: &dyn Tokenizer,
    ) -> Built {
        self.build_with_toc(
            representation,
            standing_tokens,
            history,
            query,
            tokenizer,
            None,
        )
        .await
    }

    /// The Memory context with `toc` offered for its Knowledge contents
    /// slot. The returned bundle splits its standing tokens into the
    /// `memory` block and, when it was included, the `knowledge-toc` block.
    pub async fn build_with_toc(
        &mut self,
        representation: Representation,
        standing_tokens: usize,
        history: &History,
        query: &Query,
        tokenizer: &dyn Tokenizer,
        toc: Option<&dyn KnowledgeToc>,
    ) -> Built {
        let curated = match self.curated(representation, history, query).await {
            Ok(c) => c,
            Err(e) => return Built::Failed(e),
        };
        let budget = match ContextBudget::for_total(standing_tokens) {
            Ok(b) => b,
            Err(e) => return Built::Unavailable(e.to_string()),
        };
        let req = ContextRequest {
            budget,
            query: Some(query.text.clone()),
            knowledge: toc,
            ..Default::default()
        };
        let block = match curated.svc.context(TREE, &req, tokenizer).await {
            Ok(b) => b,
            Err(e) => return Built::Failed(format!("memory context failed: {e}")),
        };

        let mut turns: BTreeSet<&str> = BTreeSet::new();
        for f in &block.included {
            for s in curated.spans.get(&f.path).into_iter().flatten() {
                if f.ranges.iter().any(|&(a, b)| a <= s.start && s.end <= b) {
                    turns.insert(&s.turn_id);
                }
            }
        }
        let included_turn_ids: Vec<String> = turns.into_iter().map(str::to_owned).collect();
        let included_ids: Vec<String> = block.included.iter().map(|f| f.path.clone()).collect();
        let excerpts = block.included.iter().filter(|f| f.excerpt).count();
        let dropped_turns = curated.visible_turns - included_turn_ids.len();
        let truncation = (dropped_turns > 0).then(|| Truncation {
            reason: format!(
                "curator kept {} of {} visible turns; context left out {} files, cut {} to excerpts{}",
                curated.remembered_turns,
                curated.visible_turns,
                block.files_omitted,
                excerpts,
                if block.directory_complete {
                    ""
                } else {
                    "; directory partial"
                }
            ),
            dropped_chunks: block.files_omitted,
            dropped_turns,
            dropped_tokens: 0,
        });
        let toc_tokens = match toc {
            Some(t) if block.knowledge_included => t
                .toc(block.budget.knowledge)
                .map_or(0, |text| toc_block_tokens(&text, tokenizer)),
            _ => 0,
        };
        let mut blocks = vec![BlockTokens::new(
            "memory",
            Slot::Standing,
            block.tokens.saturating_sub(toc_tokens),
            standing_tokens - toc_tokens,
        )];
        if toc.is_some() {
            blocks.push(BlockTokens::new(
                "knowledge-toc",
                Slot::Standing,
                toc_tokens,
                block.budget.knowledge,
            ));
        }
        Built::Ready(ContextBundle {
            format: CONTEXT_FORMAT.into(),
            tokens: block.tokens,
            text: block.text,
            budget_tokens: Some(standing_tokens),
            method: Some(method(representation, standing_tokens)),
            retrieved_ids: included_ids.clone(),
            included_ids,
            included_turn_ids,
            visibility: Default::default(),
            truncation,
            blocks,
        })
    }
}

/// Directory of the Memory profile files the fact arms write.
pub const PREFERENCE_DIR: &str = "/profile/preferences";
/// Heading of the preference profile, in Memory and in the context.
pub const PREFERENCE_HEADING: &str = "# User preferences, habits and opinions";
pub const PREFERENCE_METHOD: &str = "extracted-preferences-v1";

/// One line of the preference profile and the turns that state it.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileLine {
    pub date: Option<DateTime<Utc>>,
    pub text: String,
    pub turns: Vec<String>,
}

/// The preference profile as shown for one question.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProfileBlock {
    pub text: String,
    pub tokens: usize,
    pub turns: BTreeSet<String>,
    pub lines: usize,
}

/// Write `lines` (oldest first) into a Memory tree as dated bullets, a new
/// file each time one passes the per-file size, then read the tree back
/// and show it within `budget` tokens. A repeated statement keeps only its
/// newest line. When everything does not fit, lines are chosen by BM25
/// relevance to `query` (newest first among equals) and shown in date
/// order. Empty when there is nothing to show or nothing fits.
///
/// The block is rendered compactly rather than through `memory.context`,
/// whose framing would take most of a few-hundred-token budget.
pub async fn preference_profile(
    lines: &[ProfileLine],
    budget: usize,
    query: &Query,
    tokenizer: &dyn Tokenizer,
) -> Result<ProfileBlock, String> {
    let mut newest: Vec<&ProfileLine> = Vec::new();
    let mut seen = BTreeSet::new();
    for l in lines.iter().rev() {
        if seen.insert(one_line(&l.text).to_lowercase()) {
            newest.push(l);
        }
    }
    newest.reverse();
    if newest.is_empty() || budget == 0 {
        return Ok(ProfileBlock::default());
    }
    let bullet = |l: &ProfileLine| {
        let date = l
            .date
            .map_or_else(|| "undated".into(), |d| d.date_naive().to_string());
        format!("- [{date}] {}", one_line(&l.text))
    };
    let mut files: Vec<String> = Vec::new();
    for l in &newest {
        let line = format!("{}\n", bullet(l));
        match files.last_mut() {
            Some(f) if f.len() + line.len() <= FILE_SOFT_BYTES => f.push_str(&line),
            _ => files.push(format!("{PREFERENCE_HEADING}\n{line}")),
        }
    }
    let updated = newest.iter().filter_map(|l| l.date).max();
    let clock = Arc::new(ManualClock::new(updated.unwrap_or_else(epoch)));
    let svc = new_service().with_clock(clock);
    let fail = |e: keep_memory::MemoryError| format!("preference profile: {e}");
    svc.ensure_tree(TREE, SCOPE).await.map_err(fail)?;
    let ctx = OpCtx::new(SCOPE).pinning().harness(PREFERENCE_METHOD);
    let mut stored: Vec<String> = Vec::new();
    for (i, text) in files.iter().enumerate() {
        let path = format!("{PREFERENCE_DIR}/{:03}.md", i + 1);
        let req = CreateFile {
            entry_mode: EntryMode::SectionsAndBullets,
            ..CreateFile::new(path.clone(), text.clone())
        };
        svc.create(&ctx, TREE, &req).await.map_err(fail)?;
        let view = svc.view_by_path(TREE, &path).await.map_err(fail)?;
        stored.extend(
            view.text()
                .lines()
                .filter(|l| l.starts_with("- "))
                .map(str::to_owned),
        );
    }
    if stored.len() != newest.len() {
        return Err(format!(
            "preference profile: Memory kept {} of {} lines",
            stored.len(),
            newest.len()
        ));
    }
    let head = tokenizer.count(PREFERENCE_HEADING) + 1;
    let all = format!("{PREFERENCE_HEADING}\n{}", stored.join("\n"));
    let chosen: Vec<usize> = if tokenizer.count(&all) <= budget {
        (0..stored.len()).collect()
    } else {
        let index = Bm25Index::build(
            Bm25Params::default(),
            stored
                .iter()
                .enumerate()
                .map(|(i, l)| (i.to_string(), l.clone())),
        );
        let mut order: Vec<usize> = index
            .search(&query.text, stored.len())
            .into_iter()
            .map(|h| h.id.parse().expect("index"))
            .collect();
        let ranked: BTreeSet<usize> = order.iter().copied().collect();
        order.extend((0..stored.len()).rev().filter(|i| !ranked.contains(i)));
        let mut used = head;
        let mut keep = Vec::new();
        for i in order {
            let cost = tokenizer.count(&stored[i]) + 1;
            if used + cost <= budget {
                used += cost;
                keep.push(i);
            }
        }
        keep.sort_unstable();
        keep
    };
    if chosen.is_empty() {
        return Ok(ProfileBlock::default());
    }
    let mut text = PREFERENCE_HEADING.to_owned();
    let mut out = ProfileBlock::default();
    for &i in &chosen {
        text.push('\n');
        text.push_str(&stored[i]);
        out.turns.extend(newest[i].turns.iter().cloned());
    }
    // Joined counts can exceed the parts; drop the oldest until it fits.
    let mut chosen = chosen;
    while tokenizer.count(&text) > budget && !chosen.is_empty() {
        chosen.remove(0);
        text = std::iter::once(PREFERENCE_HEADING.to_owned())
            .chain(chosen.iter().map(|&i| stored[i].clone()))
            .collect::<Vec<_>>()
            .join("\n");
        out.turns = chosen
            .iter()
            .flat_map(|&i| newest[i].turns.iter().cloned())
            .collect();
    }
    if chosen.is_empty() {
        return Ok(ProfileBlock::default());
    }
    out.tokens = tokenizer.count(&text);
    out.lines = chosen.len();
    out.text = text;
    Ok(out)
}

fn one_line(text: &str) -> String {
    let mut s: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.len() > LINE_MAX_BYTES {
        let mut cut = LINE_MAX_BYTES;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push('…');
    }
    s
}

fn is_profile(text: &str) -> bool {
    let lower = text.to_lowercase();
    PROFILE_CUES.iter().any(|c| lower.contains(c))
}

fn notes_stem(s: &Session) -> String {
    s.date
        .map(|d| format!("notes-{}", d.format("%Y-%m")))
        .unwrap_or_else(|| "notes-undated".into())
}

async fn curate(sessions: &[&Session], r: Representation) -> keep_memory::Result<Curated> {
    let mode = entry_mode(r);
    let mut profile = Draft::new("/profile.md".into(), PROFILE_HEADING);
    let mut notes: Vec<Draft> = Vec::new();
    let mut stems: HashMap<String, usize> = HashMap::new();
    let mut visible_turns = 0;
    let mut remembered: BTreeSet<String> = BTreeSet::new();

    for s in sessions {
        let heading = format!("## {}\n", s.date_raw.as_deref().unwrap_or("Date unknown"));
        let stem = notes_stem(s);
        let mut session_open = false;
        for t in &s.turns {
            visible_turns += 1;
            if matches!(t.role, Role::Assistant | Role::System) || t.text.trim().is_empty() {
                continue;
            }
            let line = format!("- {}\n", one_line(&render_turn(t)));
            remembered.insert(t.id.clone());
            if t.role == Role::User && is_profile(&t.text) {
                profile.push_line(&line, &t.id);
                profile.updated = s.date.or(profile.updated);
                continue;
            }
            let full = notes.last().is_none_or(|d| {
                !d.path.contains(&stem)
                    || d.text.len() + heading.len() + line.len() > FILE_SOFT_BYTES
                    || d.lines + 1 >= FILE_MAX_LINES
            });
            if full {
                let n = stems.entry(stem.clone()).or_insert(0);
                *n += 1;
                let path = if *n == 1 {
                    format!("/topics/{stem}.md")
                } else {
                    format!("/topics/{stem}-{n}.md")
                };
                notes.push(Draft::new(path, ""));
                session_open = false;
            }
            let d = notes.last_mut().expect("a notes file is open");
            if !session_open {
                if !d.text.is_empty() {
                    d.text.push('\n');
                }
                d.text.push_str(&heading);
                d.lines += 1;
                session_open = true;
            }
            d.push_line(&line, &t.id);
            d.updated = s.date.or(d.updated);
        }
    }

    let svc = new_service();
    let limits = *svc.limits();
    // Keep the newest lines of the profile within one file.
    while profile.text.len() > FILE_SOFT_BYTES && !profile.spans.is_empty() {
        let first = profile.spans.remove(0);
        let cut = first.end - first.start;
        profile.text.replace_range(first.start..first.end, "");
        remembered.remove(&first.turn_id);
        for sp in &mut profile.spans {
            sp.start -= cut;
            sp.end -= cut;
        }
    }
    // Evict the oldest notes files until the tree is within its limits.
    let has_profile = !profile.spans.is_empty();
    let reserved = usize::from(has_profile);
    let mut bytes = if has_profile { profile.text.len() } else { 0 };
    let mut keep_from = notes.len();
    for (i, d) in notes.iter().enumerate().rev() {
        if notes.len() - i + reserved > limits.max_files_per_tree
            || bytes + d.text.len() > limits.max_rendered_bytes_per_tree
        {
            break;
        }
        bytes += d.text.len();
        keep_from = i;
    }
    for d in &notes[..keep_from] {
        for sp in &d.spans {
            remembered.remove(&sp.turn_id);
        }
    }
    let kept = &notes[keep_from..];

    let clock = Arc::new(ManualClock::new(epoch()));
    let svc = svc.with_clock(clock.clone());
    svc.ensure_tree(TREE, SCOPE).await?;
    let ctx = OpCtx::new(SCOPE).pinning().harness(CURATOR_METHOD);
    let mut ordered: Vec<&Draft> = kept.iter().collect();
    if has_profile {
        ordered.push(&profile);
    }
    // Creation follows time so recency ranking matches the history.
    ordered.sort_by_key(|d| d.updated.unwrap_or_else(epoch));
    let mut spans = HashMap::new();
    let mut entries = Vec::new();
    let mut now = epoch();
    for d in ordered {
        if let Some(t) = d.updated
            && t > now
        {
            now = t;
            clock.set(now);
        }
        let req = CreateFile {
            entry_mode: mode,
            ..CreateFile::new(d.path.clone(), d.text.clone())
        };
        svc.create(&ctx, TREE, &req).await?;
        spans.insert(d.path.clone(), d.spans.clone());
        entries.extend(d.spans.iter().map(|sp| {
            MemoryEntry {
                turn_id: sp.turn_id.clone(),
                line: d.text[sp.start..sp.end]
                    .trim_start_matches("- ")
                    .trim_end()
                    .to_owned(),
            }
        }));
    }
    Ok(Curated {
        svc,
        spans,
        entries,
        visible_turns,
        remembered_turns: remembered.len(),
    })
}

fn new_service() -> MemoryService {
    MemoryService::new(
        Arc::new(InMemoryLedger::new()),
        Arc::new(InMemoryJournal::new()),
    )
    .with_ids(Arc::new(SequentialIds::default()))
    .with_receipt_key(b"keep-bench-memory-arm".to_vec())
}

fn epoch() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0)
        .single()
        .expect("valid date")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Turn;
    use keep_providers::ApproxTokenizer;

    fn history() -> History {
        let turn = |s: usize, i: usize, role: Role, text: &str| Turn {
            id: format!("h:{s}:{i}"),
            session_id: format!("s{s}"),
            role,
            text: text.into(),
        };
        let session = |s: usize, day: u32, turns: Vec<Turn>| Session {
            id: format!("s{s}"),
            date_raw: Some(format!("2030/03/{day:02}")),
            date: Utc.with_ymd_and_hms(2030, 3, day, 9, 0, 0).single(),
            turns,
        };
        let mut sessions = vec![session(
            0,
            1,
            vec![
                turn(0, 0, Role::User, "I prefer oat milk in my coffee."),
                turn(0, 1, Role::Assistant, "Noted, oat milk it is."),
            ],
        )];
        for s in 1..40 {
            sessions.push(session(
                s,
                (s % 27 + 2) as u32,
                vec![turn(
                    s,
                    0,
                    Role::User,
                    &format!(
                        "Session {s}: the build server rack number is {s} and {}",
                        "filler words ".repeat(20)
                    ),
                )],
            ));
        }
        sessions.push(session(
            40,
            28,
            vec![turn(
                40,
                0,
                Role::User,
                "The kayak lives in the blue shed by the dock.",
            )],
        ));
        History {
            id: "h".into(),
            sessions,
        }
    }

    fn query(text: &str) -> Query {
        Query {
            id: "q".into(),
            text: text.into(),
            asked_at_raw: None,
            asked_at: None,
        }
    }

    #[tokio::test]
    async fn representations_store_identical_markdown_and_respect_the_budget() {
        let h = history();
        let tok = ApproxTokenizer;
        let mut arm = MemoryArm::default();
        let mut texts = Vec::new();
        for r in [
            Representation::Raw,
            Representation::Structured,
            Representation::StructuredBullets,
        ] {
            let Built::Ready(b) = arm
                .build(r, 1024, &h, &query("where is the kayak?"), &tok)
                .await
            else {
                panic!("memory arm failed for {r:?}")
            };
            assert!(b.tokens <= 1024 && b.tokens == tok.count(&b.text));
            assert!(b.text.starts_with("BEGIN MEMORY CONTEXT"));
            assert!(b.text.contains("oat milk"), "profile is always considered");
            assert!(
                b.text.contains("blue shed"),
                "{r:?}: relevant notes are selected"
            );
            assert!(b.included_turn_ids.contains(&"h:40:0".to_owned()));
            assert!(!b.included_turn_ids.contains(&"h:0:1".to_owned()));
            assert!(b.truncation.is_some(), "most turns cannot fit 1024 tokens");
            let c = &arm.cache[&("h".to_owned(), 41, r)];
            let (_, files) = c.svc.snapshot(TREE).await.unwrap();
            texts.push(
                files
                    .iter()
                    .map(|v| (v.file.meta.path.clone(), v.text()))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(texts[0], texts[1]);
        assert_eq!(texts[1], texts[2]);
        assert!(texts[0].len() > 2, "notes rotate into several files");
    }

    #[tokio::test]
    async fn future_sessions_are_not_curated() {
        let h = history();
        let mut q = query("where is the kayak?");
        q.asked_at = Utc.with_ymd_and_hms(2030, 3, 20, 0, 0, 0).single();
        let Built::Ready(b) = MemoryArm::default()
            .build(Representation::Structured, 1024, &h, &q, &ApproxTokenizer)
            .await
        else {
            panic!("memory arm failed")
        };
        assert!(!b.text.contains("blue shed"));
        assert!(!b.included_turn_ids.iter().any(|t| t == "h:40:0"));
    }
}
