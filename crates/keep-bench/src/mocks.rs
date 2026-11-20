//! Offline stand-ins for the answer and judge models.
//!
//! They read only the rendered prompt, exactly as a remote model would, so
//! they exercise the whole pipeline. Their numbers are plumbing checks and are
//! labelled as such in every report; they are never benchmark results.

use std::collections::HashSet;

use keep_providers::ChatRequest;
use keep_providers::mock::FnChat;
use keep_store::lexical;

use crate::metrics;
use crate::prompt;

fn user_text(req: &ChatRequest) -> &str {
    req.messages
        .last()
        .map(|m| m.content.as_str())
        .unwrap_or_default()
}

/// Extractive answerer: returns the history line sharing the most question
/// words, or "I don't know." when nothing overlaps.
pub fn answerer() -> FnChat {
    FnChat::new("mock-extractive-answerer-v1", |req| {
        let text = user_text(req);
        let question = prompt::field(text, "Question:").unwrap_or_default();
        // Crude stemming: compare 4-character prefixes, so "pick" matches
        // "picked".
        let stem = |t: String| t.chars().take(4).collect::<String>();
        let q: HashSet<String> = lexical::tokenize(question)
            .into_iter()
            .filter(|t| t.len() >= 3)
            .map(stem)
            .collect();
        // Keep's prompt, then the parity protocols' prompts.
        let mem0 = text.contains("give your final answer after \"ANSWER:\"");
        let history = [
            ("Conversation history:\n", "\n\nCurrent date:"),
            ("(oldest to newest).\n\n", "\n\nQuestion:"),
            ("History Chats:\n\n", "\n\nCurrent Date:"),
        ]
        .iter()
        .find_map(|(start, end)| {
            text.split_once(start)
                .map(|(_, rest)| rest.split(end).next().unwrap_or_default())
        })
        .unwrap_or_default();
        let best = history
            .lines()
            .filter(|l| !l.starts_with("[Session"))
            .map(|l| {
                let words: HashSet<String> = lexical::tokenize(l).into_iter().map(stem).collect();
                (q.iter().filter(|t| words.contains(*t)).count(), l)
            })
            .filter(|(n, _)| *n > 0)
            // Highest overlap; later lines win ties (recency).
            .max_by_key(|(n, _)| *n);
        let answer = match best {
            Some((_, line)) => {
                let body = line.rsplit_once("] ").map(|(_, b)| b).unwrap_or(line);
                let body = body.split_once(": ").map(|(_, b)| b).unwrap_or(body);
                body.split_whitespace()
                    .take(40)
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            None => "I don't know.".into(),
        };
        Ok(if mem0 {
            format!("Steps: matched the closest memory.\nANSWER: {answer}")
        } else {
            answer
        })
    })
}

/// Lenient lexical judge: CORRECT when token F1 against gold is at least 0.5
/// or the normalised gold appears in the candidate; for unanswerable
/// questions, CORRECT only for an abstention.
pub fn judge() -> FnChat {
    FnChat::new("mock-lexical-judge-v1", |req| {
        let text = user_text(req);
        // Keep's judge prompt, Mem0's (JSON reply) or LongMemEval's (yes/no).
        let mem0 = text.contains("\nGenerated answer:");
        let longmemeval = text.contains("\nModel Response:");
        let (gold, cand, unanswerable) = if mem0 {
            (
                prompt::field(text, "Gold answer:").unwrap_or_default(),
                prompt::field(text, "Generated answer:").unwrap_or_default(),
                false,
            )
        } else if longmemeval {
            (
                ["Correct Answer:", "Rubric:", "Explanation:"]
                    .iter()
                    .find_map(|l| prompt::field(text, l))
                    .unwrap_or_default(),
                prompt::field(text, "Model Response:").unwrap_or_default(),
                text.contains("an unanswerable question"),
            )
        } else {
            (
                prompt::field(text, "Gold answer:").unwrap_or_default(),
                prompt::field(text, "Candidate answer:").unwrap_or_default(),
                prompt::field(text, "Unanswerable:") == Some("yes"),
            )
        };
        let correct = if unanswerable {
            metrics::is_abstention(cand)
        } else {
            let g = metrics::normalize(gold);
            !metrics::is_abstention(cand)
                && (metrics::f1(cand, gold) >= 0.5
                    || (!g.is_empty() && metrics::normalize(cand).contains(&g)))
        };
        Ok(match (mem0, longmemeval, correct) {
            (true, _, c) => format!(
                "{{\"reasoning\": \"mock lexical match\", \"label\": \"{}\"}}",
                if c { "CORRECT" } else { "WRONG" }
            ),
            (_, true, c) => if c { "yes" } else { "no" }.into(),
            (_, _, c) => if c { "CORRECT" } else { "INCORRECT" }.into(),
        })
    })
}

/// Session fact extractor: one event per non-assistant turn, its value the
/// turn's first words, dated by the session; turns saying "I like", "I
/// love" or "I prefer" become preferences. Asked for a session summary
/// ([`keep_knowledge::session_summary`]), it joins the first words of the
/// first three non-assistant turns. Deterministic, for plumbing.
pub fn extractor() -> FnChat {
    FnChat::new("mock-turn-extractor-v1", |req| {
        let text = user_text(req);
        if req.messages.first().map(|m| m.content.as_str())
            == Some(keep_knowledge::session_summary::SUMMARY_PROMPT)
        {
            let said: Vec<String> = text
                .lines()
                .filter_map(|l| {
                    let (_, rest) = l.strip_prefix("[t")?.split_once("] ")?;
                    let (speaker, said) = rest.split_once(": ")?;
                    (speaker != "assistant" && !said.trim().is_empty()).then(|| {
                        let words: Vec<&str> = said.split_whitespace().take(8).collect();
                        format!("{speaker} said {}.", words.join(" "))
                    })
                })
                .take(3)
                .collect();
            return Ok(if said.is_empty() {
                "Nothing about the people.".into()
            } else {
                said.join(" ")
            });
        }
        let facts: Vec<serde_json::Value> = text
            .lines()
            .filter_map(|l| {
                let rest = l.strip_prefix("[t")?;
                let (n, rest) = rest.split_once("] ")?;
                let (speaker, said) = rest.split_once(": ")?;
                let n: u32 = n.parse().ok()?;
                if speaker == "assistant" || said.trim().is_empty() {
                    return None;
                }
                let value: Vec<&str> = said.split_whitespace().take(8).collect();
                let lower = said.to_lowercase();
                let preference = ["i like", "i love", "i prefer"]
                    .iter()
                    .any(|c| lower.contains(c));
                Some(serde_json::json!({
                    "subject": speaker,
                    "attribute": if preference { "likes" } else { "mentioned" },
                    "value": value.join(" "),
                    "kind": if preference { "preference" } else { "event" },
                    "when": null,
                    "date": null,
                    "turns": [n],
                    "confidence": 0.5,
                    "statement": format!("{speaker} said: {}", value.join(" ")),
                }))
            })
            .take(40)
            .collect();
        Ok(serde_json::json!({ "facts": facts }).to_string())
    })
}
