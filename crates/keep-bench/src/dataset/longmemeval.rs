//! LongMemEval loader (S, M and oracle files share one shape).
//!
//! Each entry is one question with its own haystack of sessions, so each
//! question is its own independent history. Turn-level `has_answer` flags are
//! evidence labels: they are moved onto the question and removed from the
//! visible history. Abstention questions have ids ending in `_abs`.

use keep_core::ContentHash;
use keep_core::knowledge::Role;
use serde_json::Value;

use super::{
    Dataset, DatasetKind, History, Query, Question, Session, Turn, answer_text, parse_timestamp,
};

pub fn parse(value: &Value) -> anyhow::Result<Dataset> {
    let entries = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("LongMemEval file must be a JSON array of questions"))?;
    let mut histories = Vec::with_capacity(entries.len());
    let mut questions = Vec::with_capacity(entries.len());
    let mut warnings = Vec::new();

    for (i, e) in entries.iter().enumerate() {
        let qid = e["question_id"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| anyhow::anyhow!("entry {i}: missing question_id"))?;
        let session_ids: Vec<String> = e["haystack_session_ids"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| v.to_string())
            })
            .collect();
        let dates: Vec<Option<String>> = e["haystack_dates"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|v| v.as_str().map(str::to_owned))
            .collect();
        let raw_sessions = e["haystack_sessions"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("{qid}: missing haystack_sessions"))?;
        anyhow::ensure!(
            session_ids.len() == raw_sessions.len(),
            "{qid}: {} session ids for {} sessions",
            session_ids.len(),
            raw_sessions.len()
        );

        let mut sessions = Vec::with_capacity(raw_sessions.len());
        let mut evidence_turn_ids = Vec::new();
        for (si, turns_json) in raw_sessions.iter().enumerate() {
            // Session ids can repeat inside a haystack; the position keeps
            // ids unique and deterministic.
            let session_id = format!("{qid}:{si}:{}", session_ids[si]);
            let date_raw = dates.get(si).cloned().flatten();
            let date = date_raw.as_deref().and_then(parse_timestamp);
            if date_raw.is_some() && date.is_none() {
                warnings.push(format!("{session_id}: unparsed date {date_raw:?}"));
            }
            let mut turns = Vec::new();
            for (ti, t) in turns_json.as_array().into_iter().flatten().enumerate() {
                let id = format!("{session_id}:{ti}");
                let role = match t["role"].as_str() {
                    Some("user") => Role::User,
                    Some("assistant") => Role::Assistant,
                    other => Role::Speaker(other.unwrap_or("unknown").to_owned()),
                };
                if t["has_answer"].as_bool() == Some(true) {
                    evidence_turn_ids.push(id.clone());
                }
                turns.push(Turn {
                    id,
                    session_id: session_id.clone(),
                    role,
                    text: t["content"].as_str().unwrap_or_default().to_owned(),
                });
            }
            sessions.push(Session {
                id: session_id,
                date_raw,
                date,
                turns,
            });
        }

        let answer_sessions: Vec<String> = e["answer_session_ids"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|sid| {
                sessions
                    .iter()
                    .zip(&session_ids)
                    .filter(move |(_, raw)| raw.as_str() == sid)
                    .map(|(s, _)| s.id.clone())
            })
            .collect();

        let answer =
            answer_text(&e["answer"]).ok_or_else(|| anyhow::anyhow!("{qid}: missing answer"))?;
        let asked_at_raw = e["question_date"].as_str().map(str::to_owned);
        let asked_at = asked_at_raw.as_deref().and_then(parse_timestamp);
        let abstention = qid.ends_with("_abs");
        questions.push(Question {
            id: qid.clone(),
            history_id: qid.clone(),
            query: Query {
                id: qid.clone(),
                text: e["question"].as_str().unwrap_or_default().to_owned(),
                asked_at_raw,
                asked_at,
            },
            answer,
            category: e["question_type"].as_str().unwrap_or("unknown").to_owned(),
            abstention,
            evidence_turn_ids,
            evidence_session_ids: answer_sessions,
            distractor: None,
        });
        histories.push(History { id: qid, sessions });
    }

    Ok(Dataset {
        kind: DatasetKind::LongmemevalS,
        source_hash: ContentHash::of_bytes(b""),
        histories,
        questions,
        warnings,
    })
}
