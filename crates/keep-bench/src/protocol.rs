//! Scoring protocols: which questions count, which answer and judge prompts
//! are used, and how a judge reply becomes a verdict.
//!
//! `keep` is the harness's own protocol and the default; it leaves every
//! existing manifest and output unchanged. The other two reproduce published
//! evaluation harnesses so that scores can be set beside published numbers:
//!
//! - `mem0-locomo`: Mem0's open `memory-benchmarks` LoCoMo harness. Categories
//!   1-4 only (adversarial category 5 excluded), Mem0's answer prompt over the
//!   retrieved memories, and its CORRECT/WRONG JSON judge ("J score").
//! - `longmemeval-official`: the official LongMemEval evaluation. The
//!   chain-of-thought reader prompt and the per-question-type judge prompts of
//!   `evaluate_qa.py`, including the abstention prompt for `_abs` questions.
//!
//! The upstream prompt texts live under `protocols/`, copied byte-for-byte
//! and hashed into every run record. See `THIRD_PARTY_NOTICES.md`.

use std::collections::BTreeMap;

use chrono::NaiveDate;
use keep_core::ContentHash;
use keep_providers::Message;
use serde::{Deserialize, Serialize};

use crate::dataset::{DatasetKind, History, Query, Question, parse_timestamp};
use crate::prompt;

/// Where a protocol's prompts and scoring rules come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Upstream {
    pub repository: &'static str,
    pub commit: &'static str,
    pub license: &'static str,
    pub files: &'static [&'static str],
}

pub const MEM0_UPSTREAM: Upstream = Upstream {
    repository: "https://github.com/mem0ai/memory-benchmarks",
    commit: "4b61c5d31b9c668a12b4f5e78064248a02c82d2b",
    license: "Apache-2.0",
    files: &["benchmarks/locomo/prompts.py", "benchmarks/locomo/run.py"],
};

pub const LONGMEMEVAL_UPSTREAM: Upstream = Upstream {
    repository: "https://github.com/xiaowu0162/LongMemEval",
    commit: "9e0b455f4ef0e2ab8f2e582289761153549043fc",
    license: "MIT",
    files: &[
        "src/evaluation/evaluate_qa.py",
        "src/evaluation/print_qa_metrics.py",
        "src/generation/run_generation.py",
    ],
};

const MEM0_ANSWER: &str = include_str!("../protocols/mem0-locomo/answer.txt");
const MEM0_JUDGE_SYSTEM: &str = include_str!("../protocols/mem0-locomo/judge-system.txt");
const MEM0_JUDGE: &str = include_str!("../protocols/mem0-locomo/judge.txt");

const LME_READER: &str = include_str!("../protocols/longmemeval-official/reader-cot.txt");
const LME_JUDGE_DEFAULT: &str = include_str!("../protocols/longmemeval-official/judge-default.txt");
const LME_JUDGE_TEMPORAL: &str =
    include_str!("../protocols/longmemeval-official/judge-temporal-reasoning.txt");
const LME_JUDGE_UPDATE: &str =
    include_str!("../protocols/longmemeval-official/judge-knowledge-update.txt");
const LME_JUDGE_PREFERENCE: &str =
    include_str!("../protocols/longmemeval-official/judge-single-session-preference.txt");
const LME_JUDGE_ABSTENTION: &str =
    include_str!("../protocols/longmemeval-official/judge-abstention.txt");

/// Mem0's `ANSWERER_MEMORY_LIMIT`: the answer prompt shows at most this many
/// memories.
pub const MEM0_MEMORY_LIMIT: usize = 200;
/// Mem0's default reference date when a conversation has none.
const MEM0_DEFAULT_REFERENCE_DATE: &str = "2023";
const MEM0_MEMORIES_HEADER: &str =
    "The following memories are presented in chronological order (oldest to newest).";
const MEM0_NO_MEMORIES: &str = "(No relevant memories found)";

/// The six LongMemEval question types in the order `print_qa_metrics.py`
/// reports them.
pub const LONGMEMEVAL_TYPES: [&str; 6] = [
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
];

/// Mem0's `CATEGORY_NAMES` for the LoCoMo categories it scores.
pub fn locomo_category_name(category: &str) -> Option<&'static str> {
    match category {
        "cat-1" => Some("multi-hop"),
        "cat-2" => Some("temporal"),
        "cat-3" => Some("open-domain"),
        "cat-4" => Some("single-hop"),
        "adversarial" | "cat-5" => Some("adversarial"),
        _ => None,
    }
}

/// LoCoMo's numeric category for a normalised label (`cat-2` -> 2,
/// `adversarial` -> 5).
pub fn locomo_category_code(category: &str) -> Option<u8> {
    match category {
        "adversarial" => Some(5),
        c => c.strip_prefix("cat-")?.parse().ok(),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProtocolName {
    /// Keep's own prompts and every question, including unanswerable ones.
    #[default]
    Keep,
    Mem0Locomo,
    LongmemevalOfficial,
}

/// Which questions a parity run may touch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SplitScope {
    /// The development split only; the sealed holdout stays sealed.
    #[default]
    Dev,
    /// Every history, holdout included, as published numbers use. Consumes
    /// the holdout: refused unless the run passes `--consume-holdout`.
    Full,
}

impl SplitScope {
    pub fn as_str(&self) -> &'static str {
        match self {
            SplitScope::Dev => "dev",
            SplitScope::Full => "full",
        }
    }
}

/// The manifest's `[protocol]` table.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolSpec {
    #[serde(default)]
    pub name: ProtocolName,
    #[serde(default)]
    pub split: SplitScope,
}

/// One vendored prompt file and its hash, as recorded in run artifacts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptFile {
    pub file: String,
    pub sha256: ContentHash,
}

/// The protocol as recorded in `run.json` and `summary.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProtocolRecord {
    pub name: ProtocolName,
    pub split: SplitScope,
    pub answer_prompt_version: String,
    pub judge_prompt_version: String,
    pub upstream_repository: String,
    pub upstream_commit: String,
    pub upstream_license: String,
    pub upstream_files: Vec<String>,
    pub prompts: Vec<PromptFile>,
}

/// A judge verdict under a protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    Correct(bool),
    /// No verdict could be read; counted as incorrect and reported.
    Malformed,
}

/// What a judge prompt needs to know about a question.
#[derive(Clone, Debug, Default)]
pub struct JudgeInput<'a> {
    pub question_id: &'a str,
    pub question: &'a str,
    pub category: &'a str,
    pub gold: &'a str,
    pub abstention: bool,
}

impl ProtocolName {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProtocolName::Keep => "keep",
            ProtocolName::Mem0Locomo => "mem0-locomo",
            ProtocolName::LongmemevalOfficial => "longmemeval-official",
        }
    }

    pub fn answer_prompt_version(&self) -> &'static str {
        match self {
            ProtocolName::Keep => prompt::ANSWER_PROMPT_VERSION,
            ProtocolName::Mem0Locomo => "mem0-locomo-answer-v1",
            ProtocolName::LongmemevalOfficial => "longmemeval-official-reader-cot-v1",
        }
    }

    pub fn judge_prompt_version(&self) -> &'static str {
        match self {
            ProtocolName::Keep => prompt::JUDGE_PROMPT_VERSION,
            ProtocolName::Mem0Locomo => "mem0-locomo-judge-v1",
            ProtocolName::LongmemevalOfficial => "longmemeval-official-judge-v1",
        }
    }

    pub fn upstream(&self) -> Option<Upstream> {
        match self {
            ProtocolName::Keep => None,
            ProtocolName::Mem0Locomo => Some(MEM0_UPSTREAM),
            ProtocolName::LongmemevalOfficial => Some(LONGMEMEVAL_UPSTREAM),
        }
    }

    /// Whether the protocol is defined for a dataset.
    pub fn supports(&self, kind: DatasetKind) -> bool {
        match self {
            ProtocolName::Keep => true,
            ProtocolName::Mem0Locomo => kind == DatasetKind::Locomo,
            ProtocolName::LongmemevalOfficial => kind != DatasetKind::Locomo,
        }
    }

    /// Whether a question is scored under this protocol.
    pub fn includes(&self, q: &Question) -> bool {
        match self {
            ProtocolName::Mem0Locomo => {
                matches!(locomo_category_code(&q.category), Some(1..=4))
            }
            ProtocolName::Keep | ProtocolName::LongmemevalOfficial => true,
        }
    }

    /// The vendored prompt files with their hashes; empty for `keep`.
    pub fn prompt_files(&self) -> Vec<PromptFile> {
        let files: &[(&str, &str)] = match self {
            ProtocolName::Keep => &[],
            ProtocolName::Mem0Locomo => &[
                ("mem0-locomo/answer.txt", MEM0_ANSWER),
                ("mem0-locomo/judge-system.txt", MEM0_JUDGE_SYSTEM),
                ("mem0-locomo/judge.txt", MEM0_JUDGE),
            ],
            ProtocolName::LongmemevalOfficial => &[
                ("longmemeval-official/reader-cot.txt", LME_READER),
                ("longmemeval-official/judge-default.txt", LME_JUDGE_DEFAULT),
                (
                    "longmemeval-official/judge-temporal-reasoning.txt",
                    LME_JUDGE_TEMPORAL,
                ),
                (
                    "longmemeval-official/judge-knowledge-update.txt",
                    LME_JUDGE_UPDATE,
                ),
                (
                    "longmemeval-official/judge-single-session-preference.txt",
                    LME_JUDGE_PREFERENCE,
                ),
                (
                    "longmemeval-official/judge-abstention.txt",
                    LME_JUDGE_ABSTENTION,
                ),
            ],
        };
        files
            .iter()
            .map(|(file, text)| PromptFile {
                file: (*file).into(),
                sha256: ContentHash::of_bytes(text.as_bytes()),
            })
            .collect()
    }

    /// Answer-call messages for a rendered context bundle.
    pub fn answer_messages(&self, context: &str, query: &Query, history: &History) -> Vec<Message> {
        match self {
            ProtocolName::Keep => prompt::answer_messages(context, query),
            ProtocolName::Mem0Locomo => {
                let memories = render_mem0_memories(&segment(context));
                vec![Message::user(fill_named(
                    MEM0_ANSWER,
                    &[
                        ("reference_date", &mem0_reference_date(history)),
                        ("memories", &memories),
                        ("question", &query.text),
                    ],
                ))]
            }
            ProtocolName::LongmemevalOfficial => vec![Message::user(fill_positional(
                LME_READER,
                &[
                    context,
                    query.asked_at_raw.as_deref().unwrap_or_default(),
                    &query.text,
                ],
            ))],
        }
    }

    /// The part of the answer reply that is judged and scored. Mem0 keeps
    /// the text after the last `ANSWER:`; the others use the whole reply.
    pub fn extract_answer<'a>(&self, reply: &'a str) -> &'a str {
        match self {
            ProtocolName::Mem0Locomo => match reply.rsplit_once("ANSWER:") {
                Some((_, after)) => after.trim(),
                None => reply,
            },
            _ => reply,
        }
    }

    /// The gold answer as the protocol shows it to the judge. Mem0 keeps
    /// only the part before the first `;` for open-domain (category 3)
    /// questions.
    pub fn judge_gold<'a>(&self, category: &str, gold: &'a str) -> &'a str {
        match self {
            ProtocolName::Mem0Locomo if locomo_category_code(category) == Some(3) => {
                gold.split(';').next().unwrap_or(gold).trim()
            }
            _ => gold,
        }
    }

    pub fn judge_messages(&self, input: &JudgeInput, candidate: &str) -> Vec<Message> {
        let gold = self.judge_gold(input.category, input.gold);
        match self {
            ProtocolName::Keep => {
                prompt::judge_messages(input.question, gold, candidate, input.abstention)
            }
            ProtocolName::Mem0Locomo => vec![
                Message::system(MEM0_JUDGE_SYSTEM),
                Message::user(fill_named(
                    MEM0_JUDGE,
                    &[
                        ("question", input.question),
                        ("answer", gold),
                        ("response", candidate),
                    ],
                )),
            ],
            ProtocolName::LongmemevalOfficial => vec![Message::user(fill_positional(
                longmemeval_judge_template(input.category, input.question_id),
                &[input.question, gold, candidate],
            ))],
        }
    }

    pub fn parse_verdict(&self, reply: &str) -> Verdict {
        let v = match self {
            ProtocolName::Keep => prompt::parse_verdict(reply),
            ProtocolName::Mem0Locomo => parse_mem0_label(reply),
            // `evaluate_qa.py`: label = 'yes' in reply.lower().
            ProtocolName::LongmemevalOfficial => Some(reply.to_lowercase().contains("yes")),
        };
        v.map_or(Verdict::Malformed, Verdict::Correct)
    }
}

impl ProtocolSpec {
    pub fn is_keep(&self) -> bool {
        self.name == ProtocolName::Keep
    }

    /// The record for run artifacts; `None` for the default protocol so its
    /// outputs stay unchanged.
    pub fn record(&self) -> Option<ProtocolRecord> {
        let up = self.name.upstream()?;
        Some(ProtocolRecord {
            name: self.name,
            split: self.split,
            answer_prompt_version: self.name.answer_prompt_version().into(),
            judge_prompt_version: self.name.judge_prompt_version().into(),
            upstream_repository: up.repository.into(),
            upstream_commit: up.commit.into(),
            upstream_license: up.license.into(),
            upstream_files: up.files.iter().map(|f| (*f).into()).collect(),
            prompts: self.name.prompt_files(),
        })
    }
}

/// `get_anscheck_prompt`: the abstention template for any `_abs` question,
/// otherwise the template for the question type.
pub fn longmemeval_judge_template(question_type: &str, question_id: &str) -> &'static str {
    if question_id.contains("_abs") {
        return LME_JUDGE_ABSTENTION;
    }
    match question_type {
        "temporal-reasoning" => LME_JUDGE_TEMPORAL,
        "knowledge-update" => LME_JUDGE_UPDATE,
        "single-session-preference" => LME_JUDGE_PREFERENCE,
        // single-session-user, single-session-assistant, multi-session.
        // Upstream raises on any other type; the default keeps a run going
        // and the summary reports the type as found.
        _ => LME_JUDGE_DEFAULT,
    }
}

/// Mem0's structured judge reply: JSON with a `label` of CORRECT or WRONG,
/// optionally wrapped in a sole `final` key. Upstream asks the API for a
/// JSON object; here the reply may also carry a code fence or text around
/// the object, so the outermost braces are parsed.
fn parse_mem0_label(reply: &str) -> Option<bool> {
    let start = reply.find('{')?;
    let end = reply.rfind('}')?;
    if end < start {
        return None;
    }
    let mut value: serde_json::Value = serde_json::from_str(&reply[start..=end]).ok()?;
    if let Some(obj) = value.as_object()
        && obj.len() == 1
        && let Some(inner) = obj.get("final")
    {
        value = inner.clone();
    }
    let label = value.get("label")?.as_str()?;
    Some(label.to_uppercase() == "CORRECT")
}

/// Mem0's `reference_date`: the raw date of the conversation's last
/// session, or "2023" when there is none.
fn mem0_reference_date(history: &History) -> String {
    history
        .sessions
        .iter()
        .rev()
        .find_map(|s| s.date_raw.clone())
        .unwrap_or_else(|| MEM0_DEFAULT_REFERENCE_DATE.into())
}

/// Fill `{name}` placeholders in one pass, so substituted text is never
/// scanned again (Python `str.format` semantics for these templates).
pub fn fill_named(template: &str, values: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    'scan: while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        for (name, value) in values {
            let token = format!("{{{name}}}");
            if tail.starts_with(&token) {
                out.push_str(value);
                rest = &tail[token.len()..];
                continue 'scan;
            }
        }
        out.push('{');
        rest = &tail[1..];
    }
    out.push_str(rest);
    out
}

/// Fill positional `{}` placeholders in order, in one pass.
pub fn fill_positional(template: &str, values: &[&str]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut parts = template.split("{}");
    out.push_str(parts.next().unwrap_or_default());
    for (i, part) in parts.enumerate() {
        out.push_str(values.get(i).copied().unwrap_or_default());
        out.push_str(part);
    }
    out
}

/// Version of [`segment`]; recorded in exports.
pub const SEGMENTATION_VERSION: &str = "memories-v1";

/// One retrieved memory in the shape Mem0's harness consumes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Memory {
    pub memory: String,
    /// `YYYY-MM-DDTHH:MM:SS`, or `None` when the text carries no date.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

fn parse_date(raw: &str) -> Option<String> {
    let fmt = "%Y-%m-%dT%H:%M:%S";
    parse_timestamp(raw)
        .map(|d| d.format(fmt).to_string())
        .or_else(|| {
            NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d")
                .ok()
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .map(|d| d.format(fmt).to_string())
        })
}

enum Header {
    /// `[Session id | date]`: the following lines form one memory.
    Session(Option<String>),
    /// `[date] text`: a memory starting with `text`.
    Item(String, String),
}

fn header(line: &str) -> Option<Header> {
    let inner = line.strip_prefix('[')?;
    let (inside, after) = inner.split_once(']')?;
    if let Some(session) = inside.strip_prefix("Session ")
        && after.trim().is_empty()
    {
        let date = session.rsplit_once(" | ").and_then(|(_, d)| parse_date(d));
        return Some(Header::Session(date));
    }
    let date = parse_date(inside)?;
    Some(Header::Item(date, after.trim_start().to_owned()))
}

/// Split a rendered context bundle into memories. Each `[date] text` line
/// starts a memory dated `date`; a `[Session id | date]` header starts one
/// memory holding the whole session; other lines continue the current
/// memory, and lines before any header form one undated memory. When there
/// are more than [`MEM0_MEMORY_LIMIT`] memories, neighbours with the same
/// date are merged, then neighbouring pairs, so no text is dropped by the
/// limit.
pub fn segment(text: &str) -> Vec<Memory> {
    let mut out: Vec<Memory> = Vec::new();
    let mut open = false;
    for line in text.lines() {
        match header(line) {
            Some(Header::Session(date)) => {
                out.push(Memory {
                    memory: String::new(),
                    created_at: date,
                });
                open = true;
            }
            Some(Header::Item(date, body)) => {
                out.push(Memory {
                    memory: body,
                    created_at: Some(date),
                });
                open = true;
            }
            None => {
                if line.trim().is_empty() && !open {
                    continue;
                }
                if !open {
                    out.push(Memory {
                        memory: String::new(),
                        created_at: None,
                    });
                    open = true;
                }
                let m = out.last_mut().expect("opened above");
                if !m.memory.is_empty() {
                    m.memory.push('\n');
                }
                m.memory.push_str(line);
            }
        }
    }
    for m in &mut out {
        m.memory = m.memory.trim_end().to_owned();
    }
    out.retain(|m| !m.memory.is_empty());
    while out.len() > MEM0_MEMORY_LIMIT {
        let before = out.len();
        out = merge_neighbours(out, |a, b| a.created_at == b.created_at);
        if out.len() == before {
            out = merge_neighbours(out, |_, _| true);
        }
    }
    out
}

fn merge_neighbours(items: Vec<Memory>, same: impl Fn(&Memory, &Memory) -> bool) -> Vec<Memory> {
    let mut out: Vec<Memory> = Vec::with_capacity(items.len());
    let mut merged_last = false;
    for m in items {
        match out.last_mut() {
            Some(prev) if !merged_last && same(prev, &m) => {
                prev.memory.push('\n');
                prev.memory.push_str(&m.memory);
                merged_last = true;
            }
            _ => {
                out.push(m);
                merged_last = false;
            }
        }
    }
    out
}

/// Mem0's `_to_human_date`, e.g. "Monday, May 08, 2023".
fn human_date(created_at: &str) -> String {
    chrono::NaiveDateTime::parse_from_str(created_at, "%Y-%m-%dT%H:%M:%S")
        .map(|d| d.format("%A, %B %d, %Y").to_string())
        .unwrap_or_else(|_| created_at.chars().take(10).collect())
}

/// The `{memories}` block of Mem0's answer prompt
/// (`get_answer_generation_prompt`): the first 200 memories, stably sorted
/// by `created_at` with undated ones first, one per line.
pub fn render_mem0_memories(memories: &[Memory]) -> String {
    if memories.is_empty() {
        return MEM0_NO_MEMORIES.into();
    }
    let mut top: Vec<&Memory> = memories.iter().take(MEM0_MEMORY_LIMIT).collect();
    top.sort_by(|a, b| {
        a.created_at
            .as_deref()
            .unwrap_or_default()
            .cmp(b.created_at.as_deref().unwrap_or_default())
    });
    let mut lines = vec![MEM0_MEMORIES_HEADER.to_owned(), String::new()];
    for m in top {
        match &m.created_at {
            Some(c) if !c.is_empty() => lines.push(format!("({}) {}", human_date(c), m.memory)),
            _ => lines.push(format!("(unknown date) {}", m.memory)),
        }
    }
    lines.join("\n")
}

/// Accuracy for one group of questions.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Score {
    pub n: usize,
    pub correct: usize,
    pub accuracy: Option<f64>,
}

impl Score {
    fn push(&mut self, correct: bool) {
        self.n += 1;
        self.correct += usize::from(correct);
        self.accuracy = Some(self.correct as f64 / self.n as f64);
    }
}

/// One arm's score under a parity protocol, in the upstream layout.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ArmProtocolScore {
    pub arm: String,
    /// Every scored question: unanswered and unparseable ones count as
    /// wrong, as upstream counts them.
    pub overall: Score,
    /// Mem0 category names, or LongMemEval question types.
    pub by_group: BTreeMap<String, Score>,
    /// LongMemEval only: mean of the per-type accuracies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_averaged: Option<f64>,
    /// LongMemEval only: the `_abs` questions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub abstention: Option<Score>,
    /// Records that did not produce a verdict (failed, unavailable or
    /// malformed); included above as wrong.
    pub unscored: usize,
}

/// Score the records of one arm under a protocol.
pub fn score_arm(
    protocol: ProtocolName,
    arm: &str,
    records: &[&crate::artifacts::QuestionRecord],
) -> ArmProtocolScore {
    use crate::artifacts::{JudgeOutcome, Outcome};
    let mut s = ArmProtocolScore {
        arm: arm.into(),
        ..Default::default()
    };
    if protocol == ProtocolName::LongmemevalOfficial {
        for t in LONGMEMEVAL_TYPES {
            s.by_group.insert(t.into(), Score::default());
        }
    }
    let mut abstention = Score::default();
    for r in records {
        if matches!(
            r.outcome,
            Outcome::Excluded { .. } | Outcome::NotRun { .. } | Outcome::Estimated
        ) {
            continue;
        }
        let correct = r.correct();
        if !matches!(r.judge, JudgeOutcome::Judged { .. }) {
            s.unscored += 1;
        }
        s.overall.push(correct);
        let group = match protocol {
            ProtocolName::Mem0Locomo => locomo_category_name(&r.category)
                .unwrap_or(&r.category)
                .to_owned(),
            _ => r.category.clone(),
        };
        s.by_group.entry(group).or_default().push(correct);
        if protocol == ProtocolName::LongmemevalOfficial && r.question_id.contains("_abs") {
            abstention.push(correct);
        }
    }
    if protocol == ProtocolName::LongmemevalOfficial {
        let accs: Vec<f64> = LONGMEMEVAL_TYPES
            .iter()
            .filter_map(|t| s.by_group.get(*t).and_then(|x| x.accuracy))
            .collect();
        s.task_averaged = (!accs.is_empty()).then(|| accs.iter().sum::<f64>() / accs.len() as f64);
        s.abstention = Some(abstention);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Session, Turn};
    use keep_core::knowledge::Role;

    fn question(id: &str, category: &str, abstention: bool) -> Question {
        Question {
            id: id.into(),
            history_id: "h".into(),
            query: Query {
                id: id.into(),
                text: "Where did Nadia Okafor move?".into(),
                asked_at_raw: Some("2024/03/02 (Sat) 09:00".into()),
                asked_at: None,
            },
            answer: "Lisbon; Porto".into(),
            category: category.into(),
            abstention,
            evidence_turn_ids: vec![],
            evidence_session_ids: vec![],
            distractor: None,
        }
    }

    fn history() -> History {
        History {
            id: "h".into(),
            sessions: vec![
                Session {
                    id: "s1".into(),
                    date_raw: Some("1:56 pm on 8 May, 2023".into()),
                    date: None,
                    turns: vec![Turn {
                        id: "t1".into(),
                        session_id: "s1".into(),
                        role: Role::User,
                        text: "hi".into(),
                    }],
                },
                Session {
                    id: "s2".into(),
                    date_raw: Some("9:00 am on 2 June, 2023".into()),
                    date: None,
                    turns: vec![],
                },
            ],
        }
    }

    #[test]
    fn vendored_prompts_match_their_recorded_hashes() {
        let want = [
            (
                "mem0-locomo/answer.txt",
                "79c9f09bcc8d5e9e8b7e9786af587b02a67d366ab79285fc148b73fd20f6297b",
            ),
            (
                "mem0-locomo/judge-system.txt",
                "36c007917faf1ab84516cdca577fb523711a9b993706fbae8ae37806e6f9adcc",
            ),
            (
                "mem0-locomo/judge.txt",
                "d248e056d993725e28fba8d16ca7081f0b59deae272ef294f3c6b00d48eac02b",
            ),
            (
                "longmemeval-official/reader-cot.txt",
                "9e2b3110622929ab896696dd8937231c7436740ec3b9586f653f97346e19ab2c",
            ),
            (
                "longmemeval-official/judge-default.txt",
                "fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc",
            ),
            (
                "longmemeval-official/judge-temporal-reasoning.txt",
                "8d33a5fdd83afeeb4592454a965eab43d1fcb2dedc042d1d3892f4254be6c273",
            ),
            (
                "longmemeval-official/judge-knowledge-update.txt",
                "183a9b3a6197ec620940f610cdc1207201ec98c1113dd633ea685cfc322fafac",
            ),
            (
                "longmemeval-official/judge-single-session-preference.txt",
                "741ee3bcbea7ff5e8ed359acef61d2f8ded3de021bbcff6ee13de455f2e2aa9b",
            ),
            (
                "longmemeval-official/judge-abstention.txt",
                "5c0b365a1e1d06db36377c735432b56e122ca3c428f89faf61d43a0d5a7e050b",
            ),
        ];
        let got: BTreeMap<String, String> =
            [ProtocolName::Mem0Locomo, ProtocolName::LongmemevalOfficial]
                .iter()
                .flat_map(|p| p.prompt_files())
                .map(|f| (f.file, f.sha256.as_str().to_owned()))
                .collect();
        assert_eq!(got.len(), want.len());
        for (file, hash) in want {
            assert_eq!(got[file], format!("sha256:{hash}"), "{file}");
        }
        assert!(ProtocolName::Keep.prompt_files().is_empty());
    }

    #[test]
    fn mem0_scores_locomo_categories_one_to_four_only() {
        let p = ProtocolName::Mem0Locomo;
        for c in ["cat-1", "cat-2", "cat-3", "cat-4"] {
            assert!(p.includes(&question("q", c, false)), "{c}");
        }
        assert!(!p.includes(&question("q", "adversarial", true)));
        assert!(!p.includes(&question("q", "uncategorised", false)));
        assert!(ProtocolName::Keep.includes(&question("q", "adversarial", true)));
        assert!(p.supports(DatasetKind::Locomo));
        assert!(!p.supports(DatasetKind::LongmemevalS));
        assert!(!ProtocolName::LongmemevalOfficial.supports(DatasetKind::Locomo));
    }

    #[test]
    fn longmemeval_judge_prompt_follows_question_type_and_abstention() {
        let p = ProtocolName::LongmemevalOfficial;
        let text = |id: &str, t: &str| {
            let q = question(id, t, id.contains("_abs"));
            let input = JudgeInput {
                question_id: &q.id,
                question: &q.query.text,
                category: &q.category,
                gold: &q.answer,
                abstention: q.abstention,
            };
            let m = p.judge_messages(&input, "They moved to Lisbon.");
            assert_eq!(m.len(), 1);
            m[0].content.clone()
        };
        for t in [
            "single-session-user",
            "single-session-assistant",
            "multi-session",
        ] {
            assert!(text("q1", t).starts_with(LME_JUDGE_DEFAULT.split("{}").next().unwrap()));
        }
        assert!(text("q1", "temporal-reasoning").contains("off-by-one"));
        assert!(text("q1", "knowledge-update").contains("updated answer"));
        let pref = text("q1", "single-session-preference");
        assert!(pref.contains("\n\nRubric: Lisbon; Porto\n\n"));
        let abs = text("q1_abs", "temporal-reasoning");
        assert!(abs.contains("unanswerable question"));
        assert!(abs.contains("\n\nExplanation: Lisbon; Porto\n\n"));
        assert!(abs.contains("\n\nModel Response: They moved to Lisbon.\n\n"));
    }

    #[test]
    fn longmemeval_verdict_is_yes_substring_and_abstention_is_scored_by_id() {
        let p = ProtocolName::LongmemevalOfficial;
        assert_eq!(p.parse_verdict("Yes."), Verdict::Correct(true));
        assert_eq!(p.parse_verdict("no"), Verdict::Correct(false));
        // Upstream's substring rule, quirks included.
        assert_eq!(p.parse_verdict("eyes"), Verdict::Correct(true));

        let rec = |id: &str, t: &str, correct: bool| crate::artifacts::QuestionRecord {
            run_id: "r".into(),
            non_canonical: None,
            manifest_hash: ContentHash::of_bytes(b"m"),
            arm: "a".into(),
            arm_kind: "bm25".into(),
            diagnostic: false,
            seed: 1,
            question_id: id.into(),
            history_id: id.into(),
            category: t.into(),
            abstention: id.contains("_abs"),
            gold: "g".into(),
            gold_corrected: false,
            fits_full_context: None,
            outcome: crate::artifacts::Outcome::Answered,
            context: None,
            prompt: None,
            answer: None,
            judge: crate::artifacts::JudgeOutcome::Judged { correct },
            judge_call: None,
            judge_consistency: None,
            final_answer: None,
            metrics: None,
        };
        let records = [
            rec("a1", "temporal-reasoning", true),
            rec("a2_abs", "temporal-reasoning", false),
            rec("b1", "knowledge-update", true),
            rec("b2_abs", "knowledge-update", true),
        ];
        let refs: Vec<_> = records.iter().collect();
        let s = score_arm(p, "a", &refs);
        assert_eq!(s.overall.correct, 3);
        assert_eq!(s.overall.n, 4);
        assert_eq!(s.abstention.as_ref().unwrap().n, 2);
        assert_eq!(s.abstention.as_ref().unwrap().correct, 1);
        // Mean of the two types that have questions: (0.5 + 1.0) / 2.
        assert_eq!(s.task_averaged, Some(0.75));
        assert_eq!(s.by_group["single-session-user"].n, 0);
    }

    #[test]
    fn mem0_judge_uses_upstream_template_label_and_category_three_gold() {
        let p = ProtocolName::Mem0Locomo;
        let q = question("q", "cat-3", false);
        let input = JudgeInput {
            question_id: &q.id,
            question: &q.query.text,
            category: &q.category,
            gold: &q.answer,
            abstention: false,
        };
        let m = p.judge_messages(&input, "{response} Lisbon");
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].content, MEM0_JUDGE_SYSTEM);
        assert!(m[1].content.contains(
            "Question: Where did Nadia Okafor move?\nGold answer: Lisbon\nGenerated answer: {response} Lisbon\n"
        ));
        let q4 = question("q", "cat-4", false);
        assert_eq!(p.judge_gold(&q4.category, &q4.answer), "Lisbon; Porto");

        assert_eq!(
            p.parse_verdict(r#"{"reasoning": "same city", "label": "CORRECT"}"#),
            Verdict::Correct(true)
        );
        assert_eq!(
            p.parse_verdict(
                "```json\n{\"final\": {\"reasoning\": \"x\", \"label\": \"wrong\"}}\n```"
            ),
            Verdict::Correct(false)
        );
        assert_eq!(p.parse_verdict("CORRECT"), Verdict::Malformed);
        assert_eq!(
            p.extract_answer("Step 1...\nANSWER: a\nANSWER:  Lisbon "),
            "Lisbon"
        );
        assert_eq!(p.extract_answer("Lisbon"), "Lisbon");
    }

    #[test]
    fn mem0_answer_prompt_renders_memories_like_upstream() {
        let p = ProtocolName::Mem0Locomo;
        let q = question("q", "cat-1", false);
        let context = "[9:00 am on 2 June, 2023] Nadia: Moving to Lisbon.\n[1:56 pm on 8 May, 2023] Nadia: Packing boxes.\n  [shared an image: boxes]";
        let m = p.answer_messages(context, &q.query, &history());
        assert_eq!(m.len(), 1);
        let text = &m[0].content;
        assert!(text.contains("These conversations took place around 9:00 am on 2 June, 2023."));
        assert!(text.contains(
            "The following memories are presented in chronological order (oldest to newest).\n\n\
             (Monday, May 08, 2023) Nadia: Packing boxes.\n  [shared an image: boxes]\n\
             (Friday, June 02, 2023) Nadia: Moving to Lisbon.\n\nQuestion: Where did Nadia Okafor move?\n"
        ));
        let empty = p.answer_messages("", &q.query, &history());
        assert!(empty[0].content.contains("(No relevant memories found)"));
    }

    #[test]
    fn segmentation_handles_sessions_items_and_the_memory_limit() {
        let text = "Profile: likes tea\n[Session conv-1:session_1 | 1:56 pm on 8 May, 2023]\nA: hi\nB: hello\n\n[2023-05-20] user: note\nmore";
        let m = segment(text);
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].created_at, None);
        assert_eq!(m[1].memory, "A: hi\nB: hello");
        assert_eq!(m[1].created_at.as_deref(), Some("2023-05-08T13:56:00"));
        assert_eq!(m[2].memory, "user: note\nmore");
        assert_eq!(m[2].created_at.as_deref(), Some("2023-05-20T00:00:00"));

        let many: String = (0..450)
            .map(|i| format!("[2023-05-{:02}] x{i}\n", 1 + i % 3))
            .collect();
        let m = segment(&many);
        assert!(m.len() <= MEM0_MEMORY_LIMIT);
        let total: usize = m.iter().map(|x| x.memory.lines().count()).sum();
        assert_eq!(total, 450);
    }

    #[test]
    fn templates_fill_in_one_pass() {
        assert_eq!(
            fill_named("{a} and {b} {c}", &[("a", "{b}"), ("b", "2")]),
            "{b} and 2 {c}"
        );
        assert_eq!(fill_positional("x{}y{}z", &["{}", "2"]), "x{}y2z");
    }

    #[test]
    fn longmemeval_reader_uses_question_date() {
        let q = question("q", "multi-session", false);
        let m = ProtocolName::LongmemevalOfficial.answer_messages("[s] hi", &q.query, &history());
        assert!(m[0].content.ends_with(
            "History Chats:\n\n[s] hi\n\nCurrent Date: 2024/03/02 (Sat) 09:00\nQuestion: Where did Nadia Okafor move?\nAnswer (step by step):"
        ));
    }
}
