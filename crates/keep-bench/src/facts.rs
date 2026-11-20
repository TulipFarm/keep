//! Model-extracted session facts for the K3 and Mk3 arms.
//!
//! At ingest an extraction model reads each session once
//! ([`keep_knowledge::session_extract`]) and its facts are applied to a
//! separate Knowledge corpus per history, so the ledger supersedes a state
//! fact when a later session gives the same subject and attribute a new
//! value. At question time the arms render the facts known when the
//! question was asked as a dated block, grouped by subject and attribute.
//!
//! Replies are cached on disk keyed by model, prompt version and session
//! content, so reruns, other arms and other histories that contain the same
//! session never pay twice.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use chrono::NaiveDate;
use keep_knowledge::engine::{EpisodeInput, ExternalClaim, SourceInput};
use keep_knowledge::extract::Claim;
use keep_knowledge::model::{FactStatus, TimeBasis};
use keep_knowledge::session_extract::{
    ExtractionCache, FactKind, SessionExtractor, SessionInput, SessionTurn,
};
use keep_knowledge::session_summary::SessionSummarizer;
use keep_knowledge::{Knowledge, ManualClock, SourceKind};
use keep_providers::{ChatModel, ProviderError, Tokenizer, timing};
use serde::{Deserialize, Serialize};

use crate::dataset::{History, Session};
use crate::manifest::ModelSpec;

/// Where extraction replies are cached unless `KEEP_EXTRACT_CACHE` says
/// otherwise.
pub const DEFAULT_CACHE_DIR: &str = ".keep-cache/extractions";

pub const FACTS_HEADER: &str = "Facts extracted from earlier sessions (dates are when each held or happened; \"current\" marks the latest value of something that changed):";
pub const EVIDENCE_HEADER: &str = "Conversation excerpts:";

/// Most lines one subject-and-attribute group shows; the newest are kept.
const GROUP_LINES: usize = 30;

pub fn cache_dir() -> PathBuf {
    std::env::var_os("KEEP_EXTRACT_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_DIR))
}

pub fn cache() -> ExtractionCache {
    ExtractionCache::new(cache_dir())
}

/// A dataset session as the extractor reads it. Empty turns are skipped,
/// as at ingest.
pub fn session_input(s: &Session) -> SessionInput {
    SessionInput {
        id: s.id.clone(),
        date: s.date,
        turns: s
            .turns
            .iter()
            .filter(|t| !t.text.trim().is_empty())
            .map(|t| SessionTurn {
                id: t.id.clone(),
                speaker: t.role.label().into(),
                assistant: t.role == keep_core::knowledge::Role::Assistant,
                text: t.text.clone(),
            })
            .collect(),
    }
}

/// Counts for one extraction pass.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExtractStats {
    /// Distinct sessions (by extraction key) the pass needed.
    pub sessions: usize,
    pub cached: usize,
    pub called: usize,
    /// Not cached and no model to call (cache-only passes).
    pub missing: usize,
    pub failed: usize,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub ai_credits: f64,
    /// The first fatal provider error, which stopped further calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fatal: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
}

/// The cache-keying model id: provider kind, model and revision. The
/// `extract` command, contexts and runs must agree on it to share replies.
pub fn model_id(spec: &ModelSpec) -> String {
    let kind = serde_json::to_value(spec.provider)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default();
    format!("{kind}:{}@{}", spec.model, spec.revision)
}

/// The extractor a run uses: `model` calls on misses; `None` reads the
/// cache only. Offline mock replies are cached under the system temporary
/// directory so plumbing runs leave nothing in the working tree.
pub fn extractor(spec: &ModelSpec, model: Option<Arc<dyn ChatModel>>) -> SessionExtractor {
    let cache = if spec.provider == crate::manifest::ProviderKind::Mock {
        ExtractionCache::new(std::env::temp_dir().join("keep-mock-extractions"))
    } else {
        cache()
    };
    SessionExtractor::new(model, model_id(spec), Some(cache), spec.max_output_tokens)
}

/// Run warnings for an extraction pass that left sessions without facts.
pub fn stats_warnings(stats: &ExtractStats) -> Vec<String> {
    let mut out = Vec::new();
    if stats.missing > 0 {
        out.push(format!(
            "{} of {} sessions have no cached fact extraction and no extraction model to call; k3 and mk3 see no facts from them (run `keep-bench extract` first)",
            stats.missing, stats.sessions
        ));
    }
    if stats.failed > 0 {
        out.push(format!(
            "fact extraction failed for {} of {} sessions; k3 and mk3 see no facts from them: {}",
            stats.failed,
            stats.sessions,
            stats.errors.join("; ")
        ));
    }
    out
}

/// A once-per-session model pass whose replies are cached: fact
/// extraction, or session summaries.
pub trait SessionPass: Send + Sync + 'static {
    /// What the pass is called in errors and progress lines.
    fn what(&self) -> &'static str;
    fn key(&self, session: &SessionInput) -> String;
    fn is_cached(&self, session: &SessionInput) -> bool;
    fn has_model(&self) -> bool;
    /// Read one session: whether it was cached, its usage and credits.
    fn read(
        &self,
        session: &SessionInput,
    ) -> impl std::future::Future<Output = Result<PassReply, ProviderError>> + Send;
}

/// What one [`SessionPass::read`] cost.
pub struct PassReply {
    pub cached: bool,
    pub usage: keep_providers::Usage,
    pub ai_credits: Option<f64>,
}

impl SessionPass for SessionExtractor {
    fn what(&self) -> &'static str {
        "fact extraction"
    }
    fn key(&self, session: &SessionInput) -> String {
        SessionExtractor::key(self, session)
    }
    fn is_cached(&self, session: &SessionInput) -> bool {
        SessionExtractor::is_cached(self, session)
    }
    fn has_model(&self) -> bool {
        SessionExtractor::has_model(self)
    }
    async fn read(&self, session: &SessionInput) -> Result<PassReply, ProviderError> {
        let x = self.extract(session).await?;
        Ok(PassReply {
            cached: x.cached,
            usage: x.usage,
            ai_credits: x.ai_credits,
        })
    }
}

impl SessionPass for SessionSummarizer {
    fn what(&self) -> &'static str {
        "session summary"
    }
    fn key(&self, session: &SessionInput) -> String {
        SessionSummarizer::key(self, session)
    }
    fn is_cached(&self, session: &SessionInput) -> bool {
        SessionSummarizer::is_cached(self, session)
    }
    fn has_model(&self) -> bool {
        SessionSummarizer::has_model(self)
    }
    async fn read(&self, session: &SessionInput) -> Result<PassReply, ProviderError> {
        let x = self.summarize(session).await?;
        Ok(PassReply {
            cached: x.cached,
            usage: x.usage,
            ai_credits: x.ai_credits,
        })
    }
}

/// A summary is at most about 90 words; this leaves room to spare.
pub const SUMMARY_MAX_OUTPUT_TOKENS: u32 = 400;

/// The summarizer a run uses, with the same model, cache and mock rules as
/// [`extractor`].
pub fn summarizer(spec: &ModelSpec, model: Option<Arc<dyn ChatModel>>) -> SessionSummarizer {
    let cache = if spec.provider == crate::manifest::ProviderKind::Mock {
        ExtractionCache::new(std::env::temp_dir().join("keep-mock-extractions"))
    } else {
        cache()
    };
    SessionSummarizer::new(
        model,
        model_id(spec),
        Some(cache),
        spec.max_output_tokens.min(SUMMARY_MAX_OUTPUT_TOKENS),
    )
}

/// Run warnings for a summary pass that left sessions without a summary.
pub fn summary_warnings(stats: &ExtractStats) -> Vec<String> {
    let mut out = Vec::new();
    if stats.missing > 0 {
        out.push(format!(
            "{} of {} sessions have no cached summary and no extraction model to call (run `keep-bench extract` first)",
            stats.missing, stats.sessions
        ));
    }
    if stats.failed > 0 {
        out.push(format!(
            "session summaries failed for {} of {} sessions: {}",
            stats.failed,
            stats.sessions,
            stats.errors.join("; ")
        ));
    }
    out
}

/// Make sure every session has a cached reply, calling the model for the
/// misses `concurrency` at a time. Sessions are deduplicated by their
/// cache key, so a session shared by several histories is read once.
/// A fatal provider error stops further calls.
pub async fn warm_cache<P: SessionPass>(
    extractor: Arc<P>,
    sessions: Vec<SessionInput>,
    concurrency: usize,
    max_attempts: u32,
) -> ExtractStats {
    let what = extractor.what();
    let mut stats = ExtractStats::default();
    let mut seen = BTreeSet::new();
    let mut pending = Vec::new();
    for s in sessions {
        if !seen.insert(extractor.key(&s)) {
            continue;
        }
        stats.sessions += 1;
        if extractor.is_cached(&s) {
            stats.cached += 1;
        } else {
            pending.push(s);
        }
    }
    if !extractor.has_model() {
        stats.missing = pending.len();
        return stats;
    }
    let total = pending.len();
    let gate = Arc::new(tokio::sync::Semaphore::new(concurrency.max(1)));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut tasks = tokio::task::JoinSet::new();
    for s in pending {
        let (extractor, gate, stop) = (extractor.clone(), gate.clone(), stop.clone());
        tasks.spawn(async move {
            let _permit = gate.acquire_owned().await.expect("semaphore open");
            let mut attempts = 0;
            loop {
                if stop.load(std::sync::atomic::Ordering::SeqCst) {
                    return (s.id, Err((false, "stopped after a fatal error".to_owned())));
                }
                attempts += 1;
                match extractor.read(&s).await {
                    Ok(x) => return (s.id, Ok(x)),
                    Err(e) => {
                        if e.fatal() {
                            stop.store(true, std::sync::atomic::Ordering::SeqCst);
                            return (s.id, Err((true, e.to_string())));
                        }
                        if !e.retryable() || attempts >= max_attempts.max(1) {
                            return (s.id, Err((false, e.to_string())));
                        }
                        tokio::time::sleep(e.retry_delay(attempts)).await;
                    }
                }
            }
        });
    }
    let tag = if what == "fact extraction" {
        "extract"
    } else {
        "summarize"
    };
    let mut done = 0;
    while let Some(r) = tasks.join_next().await {
        done += 1;
        match r {
            Ok((_, Ok(x))) => {
                if x.cached {
                    stats.cached += 1;
                } else {
                    stats.called += 1;
                    stats.input_tokens += x.usage.input_tokens;
                    stats.output_tokens += x.usage.output_tokens;
                    stats.ai_credits += x.ai_credits.unwrap_or(0.0);
                }
            }
            Ok((id, Err((fatal, e)))) => {
                stats.failed += 1;
                if fatal && stats.fatal.is_none() {
                    stats.fatal = Some(format!("{what}: {e}"));
                }
                if stats.errors.len() < 10 {
                    stats.errors.push(format!("{id}: {e}"));
                }
            }
            Err(e) => {
                stats.failed += 1;
                if stats.errors.len() < 10 {
                    stats.errors.push(format!("{what} task: {e}"));
                }
            }
        }
        if done % 100 == 0 || done == total {
            eprintln!(
                "[{tag}] {done}/{total} sessions; {} called, {} failed, {:.1} credits",
                stats.called, stats.failed, stats.ai_credits
            );
        }
    }
    stats
}

/// The claims one session's facts become. Events carry their date in the
/// object, so the same kind of event on two dates stays two facts while a
/// restatement of one event only adds evidence.
pub fn claims_for(facts: &[keep_knowledge::session_extract::SessionFact]) -> Vec<ExternalClaim> {
    facts
        .iter()
        .map(|f| {
            let object = match (f.kind, f.valid_from) {
                (FactKind::Event, Some(t)) => format!("{} @ {}", f.value, t.date_naive()),
                _ => f.value.clone(),
            };
            ExternalClaim {
                claim: Claim {
                    subject: f.subject.clone(),
                    subject_kind: "person".into(),
                    predicate: f.attribute.clone(),
                    object,
                    object_kind: None,
                    valid_from: f.valid_from,
                    basis: f.basis,
                    quote: String::new(),
                },
                text: Some(f.statement.clone()),
                single_valued: f.kind == FactKind::State,
                episodes: f.turns.clone(),
            }
        })
        .collect()
}

/// What building one facts corpus found.
#[derive(Clone, Debug, Default)]
pub struct IngestedFacts {
    pub sessions: usize,
    /// Sessions with no cached reply; they contribute no facts.
    pub missing: usize,
    pub claims: usize,
    /// The preference facts, oldest session first, for the Memory profile.
    pub preferences: Vec<Preference>,
    pub source_times: HashMap<String, crate::v5::SourceSnippet>,
}

/// One extracted preference, habit or opinion and where it was said.
#[derive(Clone, Debug, PartialEq)]
pub struct Preference {
    /// The date of the session that stated it.
    pub said: Option<chrono::DateTime<chrono::Utc>>,
    pub attribute: String,
    pub value: String,
    pub statement: String,
    pub turns: Vec<String>,
}

impl Preference {
    /// The Knowledge fact this preference also became, as (predicate,
    /// object), so the facts block can leave it to the profile.
    pub fn fact_key(&self) -> (String, String) {
        (self.attribute.clone(), self.value.clone())
    }
}

/// One facts corpus per history: the same sources and episodes as the
/// history's main corpus, then each session's extracted facts applied with
/// the clock at the session date. No jobs run, so the corpus holds only
/// the extracted facts. Replies come from the cache only; warm it first.
pub async fn ingest_facts(
    k: &Knowledge,
    clock: &ManualClock,
    corpus: &str,
    history: &History,
    extractor: &SessionExtractor,
) -> anyhow::Result<IngestedFacts> {
    let _t = timing::span("bench.facts_corpus").items(1);
    k.delete_corpus(corpus).await?;
    let label = extractor.evidence_label();
    let mut out = IngestedFacts {
        source_times: crate::v5::source_annotations(history),
        ..Default::default()
    };
    let mut sessions: Vec<_> = history.sessions.iter().collect();
    sessions.sort_by_key(|s| s.date.unwrap_or(chrono::DateTime::UNIX_EPOCH));
    for s in sessions {
        let at = s.date.unwrap_or(chrono::DateTime::UNIX_EPOCH);
        clock.set(at);
        let input = session_input(s);
        out.sessions += 1;
        let episodes: Vec<EpisodeInput> = input
            .turns
            .iter()
            .map(|t| EpisodeInput {
                id: t.id.clone(),
                speaker: t.speaker.clone(),
                occurred_at: s.date,
                text: t.text.clone(),
            })
            .collect();
        if episodes.is_empty() {
            continue;
        }
        k.ingest(
            corpus,
            SourceInput {
                id: s.id.clone(),
                partition: crate::knowledge_arm::PARTITION.into(),
                collection: crate::knowledge_arm::COLLECTION.into(),
                title: format!(
                    "Session {} ({})",
                    s.id,
                    s.date_raw.as_deref().unwrap_or("date unknown")
                ),
                kind: SourceKind::Conversation,
            },
            episodes,
        )
        .await?;
        let facts = match extractor.cached(&input) {
            Some(r) => r?.facts,
            None => {
                out.missing += 1;
                continue;
            }
        };
        out.preferences.extend(
            facts
                .iter()
                .filter(|f| f.kind == FactKind::Preference)
                .map(|f| Preference {
                    said: s.date,
                    attribute: f.attribute.clone(),
                    value: f.value.clone(),
                    statement: f.statement.clone(),
                    turns: f.turns.clone(),
                }),
        );
        let claims = claims_for(&facts);
        out.claims += k.apply_claims(corpus, &label, &claims).await?;
    }
    Ok(out)
}

/// One fact as the block shows it: the version known when the question
/// was asked.
#[derive(Clone, Debug, PartialEq)]
pub struct FactLine {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub value: String,
    pub text: String,
    pub valid_from: Option<NaiveDate>,
    pub valid_to: Option<NaiveDate>,
    pub basis: TimeBasis,
    pub status: FactStatus,
    pub turns: Vec<String>,
    /// Days from `valid_from` to the question, shown when set.
    pub age_days: Option<i64>,
}

impl FactLine {
    /// What lexical and dense ranking read.
    pub fn search_text(&self) -> String {
        format!(
            "{} {} {} {}",
            self.subject,
            self.predicate.replace('_', " "),
            self.value,
            self.text
        )
    }

    fn date_label(&self) -> String {
        match self.age_days {
            Some(n) => format!("{}, {}", self.plain_date_label(), age(n)),
            None => self.plain_date_label(),
        }
    }

    fn plain_date_label(&self) -> String {
        let from = self.valid_from.map(|d| d.to_string());
        match (self.status, from, self.valid_to) {
            (FactStatus::Superseded, Some(f), Some(t)) => format!("{f} until {t}, superseded"),
            (FactStatus::Superseded, None, Some(t)) => format!("until {t}, superseded"),
            (FactStatus::Superseded, f, None) => {
                format!("{}superseded", f.map(|f| f + ", ").unwrap_or_default())
            }
            (FactStatus::Disputed, f, _) => {
                format!("{}conflicting", f.map(|f| f + ", ").unwrap_or_default())
            }
            (FactStatus::Active, Some(f), Some(t)) => format!("{f} until {t}"),
            (FactStatus::Active, Some(f), None) => match self.basis {
                TimeBasis::Observed => format!("as of {f}"),
                _ => f,
            },
            (FactStatus::Active, None, _) => "undated".into(),
        }
    }
}

/// The facts block's opening line for a question's time reference.
pub fn window_line(w: &keep_knowledge::time_ref::TimeWindow) -> String {
    format!(
        "Question time reference: \"{}\" = {} to {}.",
        w.phrase, w.from, w.to
    )
}

/// A distance in days from the question date, as the facts block shows it.
pub fn age(days: i64) -> String {
    match days {
        0 => "the question date".into(),
        1 => "1 day before the question".into(),
        n if n > 1 => format!("{n} days before the question"),
        n => format!("{} days after the question", -n),
    }
}

/// Facts that share a subject and attribute, oldest first.
#[derive(Clone, Debug, PartialEq)]
pub struct FactGroup {
    pub subject: String,
    pub predicate: String,
    pub lines: Vec<FactLine>,
}

impl FactGroup {
    /// Whether a later value replaced an earlier one.
    pub fn changed(&self) -> bool {
        self.lines
            .iter()
            .any(|l| l.status == FactStatus::Superseded)
    }

    /// The group as text: one line for a single fact; otherwise a header
    /// (with the current value when it changed, or the count) and one
    /// dated line per fact, newest last.
    pub fn render(&self) -> String {
        self.render_newest(GROUP_LINES)
    }

    /// The group with at most its `limit` newest lines; the header still
    /// counts them all.
    pub fn render_newest(&self, limit: usize) -> String {
        let shown = self.shown(limit);
        let omitted = self.lines.len() - shown.len();
        let line = |l: &FactLine| format!("[{}] {}", l.date_label(), l.text);
        if self.lines.len() == 1 {
            return format!("- {}", line(&self.lines[0]));
        }
        let attribute = self.predicate.replace('_', " ");
        let header = if self.changed() {
            let current: Vec<&FactLine> = self
                .lines
                .iter()
                .filter(|l| l.status != FactStatus::Superseded)
                .collect();
            match current.as_slice() {
                [c] => format!(
                    "- {} / {attribute}: current value {} (since {}); {} earlier value(s):",
                    self.subject,
                    c.value,
                    c.valid_from
                        .map_or_else(|| "unknown".into(), |d| d.to_string()),
                    self.lines.len() - 1
                ),
                _ => format!(
                    "- {} / {attribute}: {} values over time:",
                    self.subject,
                    self.lines.len()
                ),
            }
        } else {
            format!(
                "- {} / {attribute}: {} entries:",
                self.subject,
                self.lines.len()
            )
        };
        let mut out = vec![header];
        if omitted > 0 {
            out.push(format!("  ({omitted} older not shown)"));
        }
        out.extend(shown.iter().map(|l| format!("  {}", line(l))));
        out.join("\n")
    }
}

impl FactGroup {
    fn shown(&self, limit: usize) -> &[FactLine] {
        &self.lines[self.lines.len().saturating_sub(limit.clamp(1, GROUP_LINES))..]
    }
}

/// Group lines by subject and attribute, each group oldest first.
pub fn group_lines(lines: Vec<FactLine>) -> Vec<FactGroup> {
    let mut groups: BTreeMap<(String, String), Vec<FactLine>> = BTreeMap::new();
    for l in lines {
        groups
            .entry((l.subject.clone(), l.predicate.clone()))
            .or_default()
            .push(l);
    }
    groups
        .into_iter()
        .map(|((subject, predicate), mut lines)| {
            lines.sort_by(|a, b| {
                a.valid_from
                    .cmp(&b.valid_from)
                    .then_with(|| a.valid_to.is_none().cmp(&b.valid_to.is_none()))
                    .then_with(|| a.id.cmp(&b.id))
            });
            FactGroup {
                subject,
                predicate,
                lines,
            }
        })
        .collect()
}

/// The rendered facts block.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FactsBlock {
    /// Empty when no group fits or none exist.
    pub text: String,
    pub tokens: usize,
    pub fact_ids: Vec<String>,
    pub turns: BTreeSet<String>,
    pub groups: usize,
}

/// Take groups in rank order while they fit `budget` (header included),
/// then show them in that order. A group too large to fit whole shows its
/// newest lines that fit.
pub fn assemble(ranked: &[FactGroup], budget: usize, tokenizer: &dyn Tokenizer) -> FactsBlock {
    let mut parts: Vec<String> = vec![FACTS_HEADER.to_owned()];
    let mut used = tokenizer.count(FACTS_HEADER);
    let mut block = FactsBlock::default();
    for g in ranked {
        let fit = (1..=g.lines.len().min(GROUP_LINES))
            .rev()
            .find_map(|limit| {
                let text = g.render_newest(limit);
                let cost = tokenizer.count(&text) + 1;
                (used + cost <= budget).then_some((limit, text, cost))
            });
        let Some((limit, text, cost)) = fit else {
            continue;
        };
        used += cost;
        parts.push(text);
        block.groups += 1;
        for l in g.shown(limit) {
            block.fact_ids.push(l.id.clone());
            block.turns.extend(l.turns.iter().cloned());
        }
    }
    if block.groups == 0 {
        return FactsBlock::default();
    }
    block.text = parts.join("\n");
    block.tokens = tokenizer.count(&block.text);
    block
}

/// [`assemble`] with the groups `other` picks set apart under
/// `other_heading`, after the rest. Both sections share `budget` and the
/// rank order decides which groups fit; the heading's tokens are reserved
/// up front.
pub fn assemble_grounded(
    ranked: &[FactGroup],
    other: impl Fn(&FactGroup) -> bool,
    other_heading: &str,
    budget: usize,
    tokenizer: &dyn Tokenizer,
) -> FactsBlock {
    let mut asked_parts: Vec<String> = Vec::new();
    let mut other_parts: Vec<String> = Vec::new();
    let mut used = tokenizer.count(FACTS_HEADER) + tokenizer.count(other_heading) + 1;
    let mut block = FactsBlock::default();
    for g in ranked {
        let fit = (1..=g.lines.len().min(GROUP_LINES))
            .rev()
            .find_map(|limit| {
                let text = g.render_newest(limit);
                let cost = tokenizer.count(&text) + 1;
                (used + cost <= budget).then_some((limit, text, cost))
            });
        let Some((limit, text, cost)) = fit else {
            continue;
        };
        used += cost;
        if other(g) {
            other_parts.push(text);
        } else {
            asked_parts.push(text);
        }
        block.groups += 1;
        for l in g.shown(limit) {
            block.fact_ids.push(l.id.clone());
            block.turns.extend(l.turns.iter().cloned());
        }
    }
    if block.groups == 0 {
        return FactsBlock::default();
    }
    let mut parts = vec![FACTS_HEADER.to_owned()];
    parts.extend(asked_parts);
    if !other_parts.is_empty() {
        parts.push(other_heading.to_owned());
        parts.extend(other_parts);
    }
    block.text = parts.join("\n");
    block.tokens = tokenizer.count(&block.text);
    block
}

/// The facts block's opening line for a question about some of the
/// participants.
pub fn grounding_line(g: &keep_knowledge::subject::Grounding) -> String {
    format!(
        "The question asks about {asked}. Facts and excerpts about {others} are not about {asked}.",
        asked = g.asked_label(),
        others = g.others_label()
    )
}

/// The heading above facts about the other participants.
pub fn other_heading(g: &keep_knowledge::subject::Grounding) -> String {
    format!(
        "Facts about {}, not about {}:",
        g.others_label(),
        g.asked_label()
    )
}

pub const SUMMARIES_HEADER: &str = "Session summaries (most relevant sessions, oldest first):";

/// One session's summary as the summaries block shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct SummaryLine {
    pub session: String,
    pub date: Option<NaiveDate>,
    pub text: String,
}

impl SummaryLine {
    pub fn render(&self) -> String {
        match self.date {
            Some(d) => format!("- [{d}] {}", self.text),
            None => format!("- [date unknown] {}", self.text),
        }
    }
}

/// The rendered summaries block.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SummariesBlock {
    /// Empty when no summary fits.
    pub text: String,
    pub tokens: usize,
    pub sessions: Vec<String>,
}

/// Take summaries in `ranked` order (indexes into `lines`) while they fit
/// `budget` (header included), then show them oldest first.
pub fn assemble_summaries(
    lines: &[SummaryLine],
    ranked: &[usize],
    budget: usize,
    tokenizer: &dyn Tokenizer,
) -> SummariesBlock {
    let mut used = tokenizer.count(SUMMARIES_HEADER);
    let mut taken: Vec<usize> = Vec::new();
    for &i in ranked {
        let cost = tokenizer.count(&lines[i].render()) + 1;
        if used + cost <= budget {
            used += cost;
            taken.push(i);
        }
    }
    if taken.is_empty() {
        return SummariesBlock::default();
    }
    taken.sort_by(|a, b| lines[*a].date.cmp(&lines[*b].date).then_with(|| a.cmp(b)));
    let mut parts = vec![SUMMARIES_HEADER.to_owned()];
    parts.extend(taken.iter().map(|&i| lines[i].render()));
    let text = parts.join("\n");
    SummariesBlock {
        tokens: tokenizer.count(&text),
        text,
        sessions: taken.iter().map(|&i| lines[i].session.clone()).collect(),
    }
}

/// Words that start with a capital letter but name nothing.
const NOT_ENTITIES: &[&str] = &[
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "this",
    "that",
    "these",
    "those",
    "she",
    "he",
    "they",
    "her",
    "his",
    "their",
    "it",
    "its",
    "we",
    "our",
    "you",
    "your",
    "i",
    "my",
    "user",
    "assistant",
    "after",
    "before",
    "during",
    "while",
    "when",
    "what",
    "which",
    "who",
    "also",
    "both",
    "some",
    "many",
    "one",
    "two",
    "three",
    "last",
    "next",
    "first",
    "new",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "today",
    "yesterday",
    "tomorrow",
    "mr",
    "mrs",
    "ms",
    "dr",
];

/// Capitalized words of three or more letters in `text` that are not
/// sentence fillers, dates or one of `people` (the participants, who are
/// in almost every fact). Lowercased.
pub fn named_entities(text: &str, people: &[String]) -> BTreeSet<String> {
    let people: BTreeSet<String> = people
        .iter()
        .flat_map(|p| p.split_whitespace().map(str::to_lowercase))
        .collect();
    text.split(|c: char| !(c.is_alphanumeric() || c == '\''))
        .map(|w| w.trim_end_matches("'s").trim_matches('\''))
        .filter(|w| w.chars().count() >= 3 && w.chars().next().is_some_and(char::is_uppercase))
        .map(str::to_lowercase)
        .filter(|w| !NOT_ENTITIES.contains(&w.as_str()) && !people.contains(w))
        .collect()
}

/// How many top-ranked facts seed the entity expansion.
pub const EXPAND_SEEDS: usize = 5;

/// The expansion list: facts, other than the seeds (the first
/// [`EXPAND_SEEDS`] of `fused`), that name an entity a seed names, or share
/// a seed's subject and start date (the same occasion). In `order`, so the
/// list still prefers relevant facts.
pub fn expansion(
    lines: &[FactLine],
    fused: &[String],
    order: &[String],
    people: &[String],
) -> Vec<String> {
    let by_id: HashMap<&str, &FactLine> = lines.iter().map(|l| (l.id.as_str(), l)).collect();
    let seeds: Vec<&FactLine> = fused
        .iter()
        .take(EXPAND_SEEDS)
        .filter_map(|id| by_id.get(id.as_str()).copied())
        .collect();
    let entities: BTreeSet<String> = seeds
        .iter()
        .flat_map(|l| named_entities(&format!("{} {}", l.value, l.text), people))
        .collect();
    let occasions: BTreeSet<(&str, NaiveDate)> = seeds
        .iter()
        .filter_map(|l| Some((l.subject.as_str(), l.valid_from?)))
        .collect();
    let seed_ids: BTreeSet<&str> = seeds.iter().map(|l| l.id.as_str()).collect();
    order
        .iter()
        .filter(|id| !seed_ids.contains(id.as_str()))
        .filter(|id| {
            by_id.get(id.as_str()).is_some_and(|l| {
                l.valid_from
                    .is_some_and(|d| occasions.contains(&(l.subject.as_str(), d)))
                    || !named_entities(&format!("{} {}", l.value, l.text), people)
                        .is_disjoint(&entities)
            })
        })
        .cloned()
        .collect()
}

/// Rank groups by their best-ranked member; groups with no ranked member
/// keep their place after the ranked ones.
pub fn rank_groups(groups: Vec<FactGroup>, fused: &[String]) -> Vec<FactGroup> {
    let rank: HashMap<&str, usize> = fused
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let mut keyed: Vec<(usize, FactGroup)> = groups
        .into_iter()
        .map(|g| {
            let best = g
                .lines
                .iter()
                .filter_map(|l| rank.get(l.id.as_str()).copied())
                .min()
                .unwrap_or(usize::MAX);
            (best, g)
        })
        .collect();
    keyed.sort_by_key(|(r, _)| *r);
    keyed.into_iter().map(|(_, g)| g).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use keep_providers::ApproxTokenizer;

    fn d(s: &str) -> Option<NaiveDate> {
        Some(s.parse().unwrap())
    }

    fn line(id: &str, pred: &str, value: &str, text: &str) -> FactLine {
        FactLine {
            id: id.into(),
            subject: "user".into(),
            predicate: pred.into(),
            value: value.into(),
            text: text.into(),
            valid_from: None,
            valid_to: None,
            basis: TimeBasis::Explicit,
            status: FactStatus::Active,
            turns: vec![format!("t-{id}")],
            age_days: None,
        }
    }

    #[test]
    fn changed_values_show_the_current_one_and_dated_history() {
        let mut old = line(
            "a",
            "gym_class_time",
            "7 pm",
            "The user's gym class is at 7 pm.",
        );
        old.valid_from = d("2023-03-01");
        old.valid_to = d("2023-05-10");
        old.status = FactStatus::Superseded;
        old.basis = TimeBasis::Observed;
        let mut new = line(
            "b",
            "gym_class_time",
            "6 pm",
            "The user's gym class moved to 6 pm.",
        );
        new.valid_from = d("2023-05-10");
        new.basis = TimeBasis::Observed;
        let groups = group_lines(vec![new, old]);
        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].render(),
            "- user / gym class time: current value 6 pm (since 2023-05-10); 1 earlier value(s):\n  \
             [2023-03-01 until 2023-05-10, superseded] The user's gym class is at 7 pm.\n  \
             [as of 2023-05-10] The user's gym class moved to 6 pm."
        );
    }

    #[test]
    fn temporal_lines_show_their_age_at_the_question() {
        let mut l = line(
            "a",
            "met",
            "Emma @ 2023-04-11",
            "On 2023-04-11 the user met Emma.",
        );
        l.valid_from = d("2023-04-11");
        l.age_days = Some(9);
        assert_eq!(
            group_lines(vec![l])[0].render(),
            "- [2023-04-11, 9 days before the question] On 2023-04-11 the user met Emma."
        );
        assert_eq!(age(1), "1 day before the question");
        assert_eq!(age(0), "the question date");
    }

    #[test]
    fn events_are_counted_and_dated() {
        let mut a = line(
            "a",
            "bought_item",
            "lens @ 2023-05-02",
            "On 2023-05-02 the user bought a lens.",
        );
        a.valid_from = d("2023-05-02");
        let mut b = line(
            "b",
            "bought_item",
            "tripod @ 2023-04-01",
            "On 2023-04-01 the user bought a tripod.",
        );
        b.valid_from = d("2023-04-01");
        let single = line("c", "home_city", "Lisbon", "The user lives in Lisbon.");
        let groups = group_lines(vec![a, b, single]);
        assert_eq!(
            groups[0].render(),
            "- user / bought item: 2 entries:\n  \
             [2023-04-01] On 2023-04-01 the user bought a tripod.\n  \
             [2023-05-02] On 2023-05-02 the user bought a lens."
        );
        assert_eq!(groups[1].render(), "- [undated] The user lives in Lisbon.");
    }

    #[test]
    fn assembly_takes_whole_groups_in_rank_order_within_budget() {
        let groups = group_lines(vec![
            line("a", "pet", "cat", "The user has a cat named Mo."),
            line("b", "job_title", "nurse", "The user works as a nurse."),
        ]);
        let ranked = rank_groups(groups, &["b".into(), "a".into()]);
        assert_eq!(ranked[0].predicate, "job_title");
        let tok = ApproxTokenizer;
        let all = assemble(&ranked, 10_000, &tok);
        assert_eq!(all.groups, 2);
        assert!(all.text.starts_with(FACTS_HEADER));
        assert_eq!(all.fact_ids, vec!["b", "a"]);
        assert!(all.turns.contains("t-a") && all.turns.contains("t-b"));
        assert!(all.tokens <= 10_000);
        // Room for the header and one group only.
        let one = tok.count(FACTS_HEADER) + tok.count(&ranked[0].render()) + 1;
        let cut = assemble(&ranked, one, &tok);
        assert_eq!(cut.fact_ids, vec!["b"]);
        assert!(cut.tokens <= one);
        assert_eq!(assemble(&ranked, 5, &tok), FactsBlock::default());
    }

    #[test]
    fn events_keep_their_date_in_the_object_and_states_supersede() {
        use keep_knowledge::session_extract::SessionFact;
        let at = chrono::DateTime::parse_from_rfc3339("2023-05-02T00:00:00Z")
            .unwrap()
            .to_utc();
        let fact = |kind| SessionFact {
            subject: "user".into(),
            attribute: "bought_item".into(),
            value: "lens".into(),
            kind,
            valid_from: Some(at),
            basis: TimeBasis::Explicit,
            when: Some("today".into()),
            turns: vec!["s:0".into()],
            confidence: 0.9,
            statement: "On 2023-05-02 the user bought a lens.".into(),
        };
        let c = claims_for(&[fact(FactKind::Event), fact(FactKind::State)]);
        assert_eq!(c[0].claim.object, "lens @ 2023-05-02");
        assert!(!c[0].single_valued);
        assert_eq!(c[1].claim.object, "lens");
        assert!(c[1].single_valued);
        assert_eq!(
            c[1].text.as_deref(),
            Some("On 2023-05-02 the user bought a lens.")
        );
        assert_eq!(c[1].episodes, vec!["s:0"]);
    }

    fn person(id: &str, subject: &str, pred: &str, text: &str) -> FactLine {
        FactLine {
            subject: subject.into(),
            ..line(id, pred, text, text)
        }
    }

    #[test]
    fn grounded_facts_set_other_people_apart_under_their_heading() {
        use keep_knowledge::subject::{About, Grounding};
        let people = vec!["Ilse".to_string(), "Bram".to_string()];
        let g = Grounding::of("What does Ilse paint?", &people).unwrap();
        assert_eq!(
            grounding_line(&g),
            "The question asks about Ilse. Facts and excerpts about Bram are not about Ilse."
        );
        let groups = group_lines(vec![
            person("b", "Bram", "paints", "Bram paints sunsets."),
            person("i", "Ilse", "paints", "Ilse paints forests."),
        ]);
        let ranked = rank_groups(groups, &["b".into(), "i".into()]);
        let heading = other_heading(&g);
        let block = assemble_grounded(
            &ranked,
            |grp| {
                grp.lines
                    .iter()
                    .all(|l| g.about(&l.subject) == About::Other)
            },
            &heading,
            1_000,
            &ApproxTokenizer,
        );
        // Bram ranks first but is shown after Ilse, under his own heading.
        let lines: Vec<&str> = block.text.lines().collect();
        assert_eq!(lines[0], FACTS_HEADER);
        assert!(lines[1].contains("Ilse paints forests."), "{lines:?}");
        assert_eq!(lines[2], "Facts about Bram, not about Ilse:");
        assert!(lines[3].contains("Bram paints sunsets."), "{lines:?}");
        assert_eq!(block.groups, 2);

        // No other-person facts: no heading.
        let only_ilse = rank_groups(
            group_lines(vec![person("i", "Ilse", "paints", "Ilse paints forests.")]),
            &["i".into()],
        );
        let block = assemble_grounded(&only_ilse, |_| false, &heading, 1_000, &ApproxTokenizer);
        assert!(!block.text.contains(&heading));
        // The heading's tokens are reserved: a budget that only fits the
        // plain block fits nothing here.
        let plain = assemble(&only_ilse, 1_000, &ApproxTokenizer);
        let tight = assemble_grounded(
            &only_ilse,
            |_| false,
            &heading,
            plain.tokens,
            &ApproxTokenizer,
        );
        assert_eq!(tight, FactsBlock::default());
    }

    #[test]
    fn expansion_adds_facts_sharing_an_entity_or_occasion_with_the_seeds() {
        let people = vec!["Ilse".to_string(), "Bram".to_string()];
        assert_eq!(
            named_entities(
                "On Monday Ilse and Bram's dog Pixel went to Lisbon. The end.",
                &people
            ),
            ["lisbon", "pixel"].into_iter().map(String::from).collect()
        );
        let mut seed = person("s", "Ilse", "trip", "Ilse went to Lisbon.");
        seed.valid_from = d("2023-06-01");
        let same_place = person("p", "Bram", "moved", "Bram once lived in Lisbon.");
        let mut same_day = person("o", "Ilse", "bought", "Ilse bought a hat.");
        same_day.valid_from = d("2023-06-01");
        let mut other_day = person("x", "Ilse", "bought", "Ilse bought shoes.");
        other_day.valid_from = d("2023-06-02");
        let unrelated = person("u", "Bram", "likes", "Bram likes Porto.");
        let lines = vec![seed, same_place, same_day, other_day, unrelated];
        let fused = vec!["s".to_string()];
        let order: Vec<String> = ["u", "x", "o", "s", "p"].map(String::from).to_vec();
        // Kept in `order`; the seed itself is never added again.
        assert_eq!(expansion(&lines, &fused, &order, &people), ["o", "p"]);
        assert!(expansion(&lines, &[], &order, &people).is_empty());
    }

    #[test]
    fn summaries_fill_in_rank_order_and_show_oldest_first() {
        let s = |session: &str, date: &str, text: &str| SummaryLine {
            session: session.into(),
            date: d(date),
            text: text.into(),
        };
        let lines = vec![
            s("a", "2023-01-01", "Ilse started painting."),
            s("b", "2023-03-01", "Bram moved to Lisbon."),
            s("c", "2023-02-01", &"Ilse travelled. ".repeat(80)),
        ];
        let undated = SummaryLine {
            date: None,
            ..lines[0].clone()
        };
        assert_eq!(undated.render(), "- [date unknown] Ilse started painting.");
        // c ranks second but does not fit; a and b fill, oldest first.
        let block = assemble_summaries(&lines, &[1, 2, 0], 60, &ApproxTokenizer);
        assert_eq!(block.sessions, ["a", "b"]);
        assert_eq!(
            block.text,
            format!(
                "{SUMMARIES_HEADER}\n- [2023-01-01] Ilse started painting.\n- [2023-03-01] Bram moved to Lisbon."
            )
        );
        assert!(block.tokens <= 60);
        assert_eq!(
            assemble_summaries(&lines, &[0], 5, &ApproxTokenizer),
            SummariesBlock::default()
        );
    }
}
