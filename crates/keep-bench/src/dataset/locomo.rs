//! LoCoMo loader (`locomo10.json` shape).
//!
//! Each sample is one two-speaker conversation with numbered sessions
//! (`session_1`, `session_1_date_time`, ...) and a `qa` list. Category codes
//! are kept as `cat-<n>` because published harnesses disagree on the names
//! for codes 1-4; code 5 is the adversarial, unanswerable category whose
//! entries carry `adversarial_answer` instead of `answer`.

use std::collections::HashSet;

use keep_core::ContentHash;
use keep_core::knowledge::Role;
use serde_json::Value;

use super::{
    Dataset, DatasetKind, History, Query, Question, Session, Turn, answer_text, parse_timestamp,
};

/// Text used as the gold answer for unanswerable questions.
pub const ADVERSARIAL_GOLD: &str = "Not mentioned in the conversation";

pub fn parse(value: &Value) -> anyhow::Result<Dataset> {
    let samples = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("LoCoMo file must be a JSON array of samples"))?;
    let mut histories = Vec::new();
    let mut questions = Vec::new();
    let mut warnings = Vec::new();

    for (si, sample) in samples.iter().enumerate() {
        let sample_id = sample["sample_id"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("sample-{si}"));
        let conv = sample["conversation"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("{sample_id}: missing conversation object"))?;

        let mut numbers: Vec<u32> = conv
            .keys()
            .filter_map(|k| k.strip_prefix("session_"))
            .filter_map(|rest| rest.parse::<u32>().ok())
            .collect();
        numbers.sort_unstable();
        numbers.dedup();

        let mut sessions = Vec::new();
        let mut turn_ids = HashSet::new();
        for n in numbers {
            let session_id = format!("{sample_id}:session_{n}");
            let date_raw = conv
                .get(&format!("session_{n}_date_time"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            let date = date_raw.as_deref().and_then(parse_timestamp);
            if date_raw.is_some() && date.is_none() {
                warnings.push(format!("{session_id}: unparsed date {date_raw:?}"));
            }
            let mut turns = Vec::new();
            for (ti, t) in conv[&format!("session_{n}")]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                let dia = t["dia_id"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("S{n}:{ti}"));
                let speaker = t["speaker"].as_str().unwrap_or("unknown").to_owned();
                let mut text = t["text"].as_str().unwrap_or_default().to_owned();
                if let Some(caption) = t["blip_caption"].as_str() {
                    text.push_str(&format!(" [shared an image: {caption}]"));
                }
                let id = format!("{sample_id}:{dia}");
                turn_ids.insert(id.clone());
                turns.push(Turn {
                    id,
                    session_id: session_id.clone(),
                    role: Role::Speaker(speaker),
                    text,
                });
            }
            sessions.push(Session {
                id: session_id,
                date_raw,
                date,
                turns,
            });
        }

        for (qi, qa) in sample["qa"].as_array().into_iter().flatten().enumerate() {
            let id = format!("{sample_id}:q{qi}");
            let category_code = qa["category"].as_i64();
            let abstention = category_code == Some(5);
            let answer = if abstention {
                ADVERSARIAL_GOLD.to_owned()
            } else {
                match answer_text(&qa["answer"]) {
                    Some(a) => a,
                    None => {
                        warnings.push(format!("{id}: no gold answer; question skipped"));
                        continue;
                    }
                }
            };
            let mut evidence_turn_ids = Vec::new();
            for e in qa["evidence"].as_array().into_iter().flatten() {
                for part in e
                    .as_str()
                    .unwrap_or_default()
                    .split([';', ',', ' '])
                    .filter(|p| !p.is_empty())
                {
                    let tid = format!("{sample_id}:{part}");
                    if turn_ids.contains(&tid) {
                        evidence_turn_ids.push(tid);
                    } else {
                        warnings.push(format!("{id}: evidence {part:?} matches no turn"));
                    }
                }
            }
            let mut evidence_session_ids: Vec<String> = evidence_turn_ids
                .iter()
                .filter_map(|tid| {
                    sessions
                        .iter()
                        .find(|s| s.turns.iter().any(|t| &t.id == tid))
                        .map(|s| s.id.clone())
                })
                .collect();
            evidence_session_ids.dedup();
            questions.push(Question {
                id: id.clone(),
                history_id: sample_id.clone(),
                query: Query {
                    id,
                    text: qa["question"].as_str().unwrap_or_default().to_owned(),
                    asked_at_raw: None,
                    asked_at: None,
                },
                answer,
                category: if abstention {
                    "adversarial".into()
                } else {
                    category_code.map_or_else(|| "uncategorised".into(), |c| format!("cat-{c}"))
                },
                abstention,
                evidence_turn_ids,
                evidence_session_ids,
                distractor: qa["adversarial_answer"].as_str().map(str::to_owned),
            });
        }
        histories.push(History {
            id: sample_id,
            sessions,
        });
    }

    Ok(Dataset {
        kind: DatasetKind::Locomo,
        source_hash: ContentHash::of_bytes(b""),
        histories,
        questions,
        warnings,
    })
}
