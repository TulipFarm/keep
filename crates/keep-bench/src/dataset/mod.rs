//! One normalised conversation/question schema for every benchmark.
//!
//! Loaders separate what an agent may see (the [`History`]: sessions and
//! turns, with no labels) from what only the scorer may see (each
//! [`Question`]'s gold answer and evidence ids). Baselines receive histories
//! and a [`Query`], never a `Question`, so gold answers and evidence flags
//! cannot leak into ingestion or retrieval by construction.

pub mod annotations;
pub mod locomo;
pub mod longmemeval;
pub mod registry;

use std::path::Path;

use chrono::{DateTime, NaiveDateTime, Utc};
use keep_core::ContentHash;
use keep_core::knowledge::Role;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DatasetKind {
    Locomo,
    LongmemevalS,
    LongmemevalM,
    LongmemevalOracle,
}

impl DatasetKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            DatasetKind::Locomo => "locomo",
            DatasetKind::LongmemevalS => "longmemeval-s",
            DatasetKind::LongmemevalM => "longmemeval-m",
            DatasetKind::LongmemevalOracle => "longmemeval-oracle",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    /// Deterministic, dataset-scoped id, e.g. `conv-1:D1:3`.
    pub id: String,
    pub session_id: String,
    pub role: Role,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub date_raw: Option<String>,
    pub date: Option<DateTime<Utc>>,
    pub turns: Vec<Turn>,
}

/// An independent history: one LoCoMo conversation, or one LongMemEval
/// question's haystack. It is the unit of reset, isolation and bootstrap
/// resampling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct History {
    pub id: String,
    pub sessions: Vec<Session>,
}

/// What a baseline may see about a question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub id: String,
    pub text: String,
    pub asked_at_raw: Option<String>,
    pub asked_at: Option<DateTime<Utc>>,
}

/// Scorer-only data for one question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Question {
    pub id: String,
    pub history_id: String,
    pub query: Query,
    /// Gold answer as the dataset states it, converted to text.
    pub answer: String,
    /// Normalised category label, e.g. `temporal-reasoning` or `cat-2`.
    pub category: String,
    /// True when the correct behaviour is to say the answer is unknown.
    pub abstention: bool,
    /// Turn ids the dataset marks as evidence. Never shown to baselines
    /// except the explicitly labelled oracle diagnostic.
    pub evidence_turn_ids: Vec<String>,
    pub evidence_session_ids: Vec<String>,
    /// LoCoMo adversarial questions carry a plausible wrong answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distractor: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dataset {
    pub kind: DatasetKind,
    pub source_hash: ContentHash,
    pub histories: Vec<History>,
    pub questions: Vec<Question>,
    /// Non-fatal oddities found while loading, kept for the report.
    pub warnings: Vec<String>,
}

impl Dataset {
    pub fn load(kind: DatasetKind, path: &Path) -> anyhow::Result<Dataset> {
        let bytes = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("cannot read dataset {}: {e}", path.display()))?;
        Self::from_bytes(kind, &bytes)
    }

    pub fn from_bytes(kind: DatasetKind, bytes: &[u8]) -> anyhow::Result<Dataset> {
        let source_hash = ContentHash::of_bytes(bytes);
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        let mut ds = match kind {
            DatasetKind::Locomo => locomo::parse(&value)?,
            DatasetKind::LongmemevalS
            | DatasetKind::LongmemevalM
            | DatasetKind::LongmemevalOracle => longmemeval::parse(&value)?,
        };
        ds.kind = kind;
        ds.source_hash = source_hash;
        ds.check_unique_ids()?;
        Ok(ds)
    }

    fn check_unique_ids(&self) -> anyhow::Result<()> {
        let mut seen = std::collections::HashSet::new();
        for q in &self.questions {
            anyhow::ensure!(seen.insert(&q.id), "duplicate question id {}", q.id);
        }
        let mut turns = std::collections::HashSet::new();
        for h in &self.histories {
            for s in &h.sessions {
                for t in &s.turns {
                    anyhow::ensure!(turns.insert(&t.id), "duplicate turn id {}", t.id);
                }
            }
        }
        Ok(())
    }

    pub fn history(&self, id: &str) -> Option<&History> {
        self.histories.iter().find(|h| h.id == id)
    }
}

impl History {
    /// Sessions known at question time. Sessions dated after the question are
    /// withheld; sessions with unknown dates are kept and counted so the
    /// report can disclose them.
    pub fn visible_sessions(&self, asked_at: Option<DateTime<Utc>>) -> (Vec<&Session>, usize) {
        let mut undated = 0;
        let visible = self
            .sessions
            .iter()
            .filter(|s| match (s.date, asked_at) {
                (Some(d), Some(q)) => d <= q,
                (None, _) => {
                    undated += 1;
                    true
                }
                (Some(_), None) => true,
            })
            .collect();
        (visible, undated)
    }
}

/// Parse a dataset timestamp. Datasets give local wall-clock times without a
/// zone; they are stored as UTC so ordering is preserved.
pub fn parse_timestamp(raw: &str) -> Option<DateTime<Utc>> {
    const FORMATS: &[&str] = &[
        "%Y/%m/%d (%a) %H:%M",   // LongMemEval
        "%I:%M %p on %d %B, %Y", // LoCoMo
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ];
    let trimmed = raw.trim();
    FORMATS
        .iter()
        .find_map(|f| NaiveDateTime::parse_from_str(trimmed, f).ok())
        .map(|n| n.and_utc())
}

/// Convert a JSON answer (string, number or list) to text.
pub fn answer_text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Array(items) => {
            let parts: Vec<String> = items.iter().filter_map(answer_text).collect();
            (!parts.is_empty()).then(|| parts.join("; "))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_both_dataset_timestamp_styles() {
        let lme = parse_timestamp("2023/05/20 (Sat) 02:21").unwrap();
        assert_eq!(lme.to_rfc3339(), "2023-05-20T02:21:00+00:00");
        let locomo = parse_timestamp("1:56 pm on 8 May, 2023").unwrap();
        assert_eq!(locomo.to_rfc3339(), "2023-05-08T13:56:00+00:00");
        assert!(parse_timestamp("sometime last week").is_none());
    }
}
