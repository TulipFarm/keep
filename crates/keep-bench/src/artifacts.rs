//! Per-question records and run directories.
//!
//! One JSON line per (arm, seed, question). Every outcome is an explicit
//! state: a question that was not answered says why.

use std::io::Write;
use std::path::{Path, PathBuf};

use keep_core::ContentHash;
use keep_providers::{Message, ProviderDescriptor, Usage};
use serde::{Deserialize, Serialize};

use crate::context::ContextBundle;
use crate::metrics::AnswerMetrics;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Outcome {
    /// The model returned an answer (it may still be wrong).
    Answered,
    /// The arm cannot serve this question by design, e.g. full context that
    /// exceeds the model's window.
    Unavailable { reason: String },
    /// The arm is declared but not implemented in this build.
    NotRun { reason: String },
    /// Removed by the corrected gold track.
    Excluded { reason: String },
    /// Context building or the answer call failed.
    Failed { error: String },
    /// A dry run: nothing was dispatched.
    Estimated,
}

impl Outcome {
    pub fn key(&self) -> &'static str {
        match self {
            Outcome::Answered => "answered",
            Outcome::Unavailable { .. } => "unavailable",
            Outcome::NotRun { .. } => "not_run",
            Outcome::Excluded { .. } => "excluded",
            Outcome::Failed { .. } => "failed",
            Outcome::Estimated => "estimated",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum JudgeOutcome {
    Judged {
        correct: bool,
    },
    /// The judge replied without a parseable verdict. Counted as incorrect
    /// and reported separately.
    Malformed {
        reply: String,
    },
    Failed {
        error: String,
    },
    Skipped,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptRecord {
    pub version: String,
    pub sha256: ContentHash,
    /// Full messages unless the manifest asks for hashes only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Message>>,
    pub input_tokens: usize,
    pub tokenizer: String,
    pub tokenizer_exact: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CallRecord {
    pub provider: ProviderDescriptor,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    pub usage: Usage,
    pub latency_ms: u64,
    pub attempts: u32,
    pub cost_usd: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
    /// Harness billing (AI credits, hidden overhead tokens) when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub billing: Option<keep_providers::Billing>,
    /// The model the endpoint reported for this call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextRecord {
    pub tokens: usize,
    pub budget_tokens: Option<usize>,
    pub method: Option<String>,
    pub retrieved_ids: Vec<String>,
    pub included_ids: Vec<String>,
    pub visible_sessions: usize,
    pub visible_turns: usize,
    pub undated_sessions: usize,
    pub future_sessions: usize,
    pub truncation: Option<crate::context::Truncation>,
    pub build_ms: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<crate::context::BlockTokens>,
}

impl ContextRecord {
    pub fn from_bundle(b: &ContextBundle, build_ms: u64) -> Self {
        Self {
            tokens: b.tokens,
            budget_tokens: b.budget_tokens,
            method: b.method.clone(),
            retrieved_ids: b.retrieved_ids.clone(),
            included_ids: b.included_ids.clone(),
            visible_sessions: b.visibility.sessions,
            visible_turns: b.visibility.turns,
            undated_sessions: b.visibility.undated_sessions,
            future_sessions: b.visibility.future_sessions,
            truncation: b.truncation.clone(),
            build_ms,
            blocks: b.blocks.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuestionRecord {
    pub run_id: String,
    /// The run's non-canonical banner, repeated on every record so no line
    /// can be quoted without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_canonical: Option<String>,
    pub manifest_hash: ContentHash,
    pub arm: String,
    pub arm_kind: String,
    pub diagnostic: bool,
    pub seed: u64,
    pub question_id: String,
    pub history_id: String,
    pub category: String,
    pub abstention: bool,
    /// Gold answer used for scoring on this run's track.
    pub gold: String,
    pub gold_corrected: bool,
    /// Whether full context fits per the frozen fit mask, when one is pinned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fits_full_context: Option<bool>,
    pub outcome: Outcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<CallRecord>,
    pub judge: JudgeOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_call: Option<CallRecord>,
    /// How the verdict was reached when the manifest asks for judge
    /// consistency: its cache key, whether it was reused, and the votes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_consistency: Option<crate::judge::JudgeConsistencyRecord>,
    /// The scored answer, when the prompt asks for working notes before a
    /// final answer line (answer-v6, answer-v7). `answer.text` keeps the whole reply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_answer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<AnswerMetrics>,
}

impl QuestionRecord {
    /// 1 when judged correct; 0 for every other state that counts.
    pub fn correct(&self) -> bool {
        matches!(self.judge, JudgeOutcome::Judged { correct: true })
    }
}

pub fn write_jsonl<T: Serialize>(path: &Path, items: &[T]) -> anyhow::Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for item in items {
        serde_json::to_writer(&mut f, item)?;
        f.write_all(b"\n")?;
    }
    f.flush()?;
    Ok(())
}

pub fn read_records(path: &Path) -> anyhow::Result<Vec<QuestionRecord>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l)
                .map_err(|e| anyhow::anyhow!("{}:{}: {e}", path.display(), i + 1))
        })
        .collect()
}

/// Create a fresh run directory; refuses to reuse an existing one.
pub fn create_run_dir(root: &Path, run_id: &str) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(root)?;
    let dir = root.join(run_id);
    std::fs::create_dir(&dir)
        .map_err(|e| anyhow::anyhow!("cannot create run directory {}: {e}", dir.display()))?;
    Ok(dir)
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    std::fs::write(path, text)?;
    Ok(())
}
