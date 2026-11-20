//! Rendering histories and packing context bundles under token budgets.
//!
//! Every baseline produces a [`ContextBundle`]: the text placed in the answer
//! prompt, its token count, which chunks and turns it contains, and an
//! explicit truncation report whenever something visible was left out.

use chrono::{DateTime, Utc};
use keep_providers::Tokenizer;
use serde::{Deserialize, Serialize};

use crate::ablation::Chunking;
use crate::dataset::{History, Session, Turn};

pub const CONTEXT_FORMAT: &str = "ctx-v1";

pub fn session_header(s: &Session) -> String {
    format!(
        "[Session {} | {}]",
        s.id,
        s.date_raw.as_deref().unwrap_or("date unknown")
    )
}

pub fn render_turn(t: &Turn) -> String {
    format!("{}: {}", t.role.label(), t.text)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Visibility {
    pub sessions: usize,
    pub turns: usize,
    /// Sessions without a parseable date, kept visible and disclosed.
    pub undated_sessions: usize,
    /// Sessions dated after the question and therefore withheld.
    pub future_sessions: usize,
}

/// The complete history visible at question time, in order.
pub fn render_full_history(
    history: &History,
    asked_at: Option<DateTime<Utc>>,
) -> (String, Visibility) {
    let (sessions, undated) = history.visible_sessions(asked_at);
    let mut out = String::new();
    let mut turns = 0;
    for s in &sessions {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&session_header(s));
        out.push('\n');
        for t in &s.turns {
            out.push_str(&render_turn(t));
            out.push('\n');
            turns += 1;
        }
    }
    let vis = Visibility {
        sessions: sessions.len(),
        turns,
        undated_sessions: undated,
        future_sessions: history.sessions.len() - sessions.len(),
    };
    (out, vis)
}

/// A retrievable unit cut from visible history.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub id: String,
    /// Position in chronological order, for stable re-ordering.
    pub order: usize,
    pub session_id: String,
    pub turn_ids: Vec<String>,
    /// Text that is indexed and placed in context.
    pub text: String,
}

pub fn chunks(
    history: &History,
    asked_at: Option<DateTime<Utc>>,
    chunking: Chunking,
) -> Vec<Chunk> {
    let (sessions, _) = history.visible_sessions(asked_at);
    let mut out = Vec::new();
    for s in sessions {
        let date = s.date_raw.as_deref().unwrap_or("date unknown");
        match chunking {
            Chunking::Turn => {
                for t in &s.turns {
                    out.push(Chunk {
                        id: t.id.clone(),
                        order: out.len(),
                        session_id: s.id.clone(),
                        turn_ids: vec![t.id.clone()],
                        text: format!("[{date}] {}", render_turn(t)),
                    });
                }
            }
            Chunking::Session => {
                let mut text = session_header(s);
                for t in &s.turns {
                    text.push('\n');
                    text.push_str(&render_turn(t));
                }
                out.push(Chunk {
                    id: s.id.clone(),
                    order: out.len(),
                    session_id: s.id.clone(),
                    turn_ids: s.turns.iter().map(|t| t.id.clone()).collect(),
                    text,
                });
            }
        }
    }
    out
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Truncation {
    pub reason: String,
    pub dropped_chunks: usize,
    pub dropped_turns: usize,
    pub dropped_tokens: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextBundle {
    pub format: String,
    pub text: String,
    pub tokens: usize,
    pub budget_tokens: Option<usize>,
    /// Retrieval method declaration, e.g. the BM25 parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Ids returned by retrieval, in rank order, before budget packing.
    pub retrieved_ids: Vec<String>,
    /// Chunk ids actually placed in context.
    pub included_ids: Vec<String>,
    /// Turn ids actually placed in context (for evidence recall).
    pub included_turn_ids: Vec<String>,
    pub visibility: Visibility,
    /// Present whenever visible material was left out. Retrieval arms always
    /// leave material out by design, so for them this records how much.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
    /// Token cost of each block of a Keep context, in prompt order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<BlockTokens>,
}

/// Which allowance a context block is charged to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Slot {
    /// The standing context injected before every turn (`standing_tokens`).
    Standing,
    /// Evidence retrieved for this question (`evidence_tokens`).
    Evidence,
}

/// Tokens one block of a Keep context costs. Standing blocks together never
/// exceed `standing_tokens`; evidence blocks never exceed `evidence_tokens`
/// plus one separator per hit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockTokens {
    pub name: String,
    pub slot: Slot,
    pub tokens: usize,
    /// The reservation this block was cut to fit.
    pub budget_tokens: usize,
}

impl BlockTokens {
    pub fn new(name: &str, slot: Slot, tokens: usize, budget_tokens: usize) -> Self {
        Self {
            name: name.into(),
            slot,
            tokens,
            budget_tokens,
        }
    }
}

/// Total tokens charged to `slot`, or `None` if no block uses it.
pub fn slot_tokens(blocks: &[BlockTokens], slot: Slot) -> Option<usize> {
    let mut it = blocks.iter().filter(|b| b.slot == slot).peekable();
    it.peek()?;
    Some(it.map(|b| b.tokens).sum())
}

/// Greedily take chunks in rank order while they fit the budget (skipping a
/// chunk that does not fit and trying the next), then present the selected
/// chunks in chronological order.
pub fn pack_ranked(
    ranked: &[&Chunk],
    total_visible: &[Chunk],
    budget: usize,
    tokenizer: &dyn Tokenizer,
    reason: &str,
) -> (String, Vec<String>, Vec<String>, usize, Truncation) {
    let mut used = 0;
    let mut chosen: Vec<&Chunk> = Vec::new();
    for c in ranked {
        // One newline separator per chunk.
        let cost = tokenizer.count(&c.text) + 1;
        if used + cost <= budget {
            used += cost;
            chosen.push(c);
        }
    }
    chosen.sort_by_key(|c| c.order);
    let text = chosen
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let included_ids: Vec<String> = chosen.iter().map(|c| c.id.clone()).collect();
    let included_turns: Vec<String> = chosen
        .iter()
        .flat_map(|c| c.turn_ids.iter().cloned())
        .collect();
    let chosen_ids: std::collections::HashSet<&str> =
        included_ids.iter().map(String::as_str).collect();
    let dropped: Vec<&Chunk> = total_visible
        .iter()
        .filter(|c| !chosen_ids.contains(c.id.as_str()))
        .collect();
    let truncation = Truncation {
        reason: reason.into(),
        dropped_chunks: dropped.len(),
        dropped_turns: dropped.iter().map(|c| c.turn_ids.len()).sum(),
        dropped_tokens: dropped.iter().map(|c| tokenizer.count(&c.text)).sum(),
    };
    let tokens = tokenizer.count(&text);
    (text, included_ids, included_turns, tokens, truncation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{History, Session, Turn, parse_timestamp};
    use keep_core::knowledge::Role;
    use keep_providers::ApproxTokenizer;

    fn history() -> History {
        let mk = |sid: &str, date: &str, texts: &[&str]| Session {
            id: sid.into(),
            date_raw: Some(date.into()),
            date: parse_timestamp(date),
            turns: texts
                .iter()
                .enumerate()
                .map(|(i, t)| Turn {
                    id: format!("{sid}:{i}"),
                    session_id: sid.into(),
                    role: Role::User,
                    text: (*t).into(),
                })
                .collect(),
        };
        History {
            id: "h".into(),
            sessions: vec![
                mk(
                    "s1",
                    "2024/01/01 (Mon) 10:00",
                    &["I adopted a cat named Miso.", "She is grey."],
                ),
                mk(
                    "s2",
                    "2024/03/01 (Fri) 10:00",
                    &["Miso learned to open doors."],
                ),
                mk("s3", "2024/06/01 (Sat) 10:00", &["Future session."]),
            ],
        }
    }

    #[test]
    fn full_history_withholds_sessions_after_the_question() {
        let h = history();
        let (text, vis) = render_full_history(&h, parse_timestamp("2024/04/01 (Mon) 09:00"));
        assert!(text.contains("Miso learned") && !text.contains("Future session"));
        assert_eq!(vis.sessions, 2);
        assert_eq!(vis.turns, 3);
        assert_eq!(vis.future_sessions, 1);
    }

    #[test]
    fn packing_respects_budget_reports_drops_and_restores_chronology() {
        let h = history();
        let all = chunks(&h, None, Chunking::Turn);
        let tok = ApproxTokenizer;
        // Rank the last chunk first.
        let ranked: Vec<&Chunk> = vec![&all[3], &all[0], &all[1], &all[2]];
        let budget = tok.count(&all[3].text) + tok.count(&all[0].text) + 2;
        let (text, ids, turns, tokens, trunc) = pack_ranked(&ranked, &all, budget, &tok, "budget");
        assert_eq!(ids, vec!["s1:0", "s3:0"], "chronological order");
        assert_eq!(turns, ids);
        assert!(tokens <= budget);
        assert!(text.starts_with("[2024/01/01"));
        assert_eq!(trunc.dropped_chunks, 2);
        assert_eq!(trunc.dropped_turns, 2);
        assert!(trunc.dropped_tokens > 0);
    }
}
