//! Local, model-free answer metrics.
//!
//! Exact match and token F1 use SQuAD-style normalisation (lower-case, strip
//! punctuation and English articles, collapse whitespace). They are cheap,
//! deterministic diagnostics that sit next to the judge verdict. They are not
//! the official evaluators of any benchmark: LoCoMo's own script, for
//! example, also stems tokens, so its F1 numbers differ from these.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub const METRICS_VERSION: &str = "local-f1-squad-v1";

pub fn normalize(text: &str) -> String {
    let lowered = text.to_lowercase();
    let no_punct: String = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    no_punct
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the"))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn exact_match(prediction: &str, gold: &str) -> bool {
    normalize(prediction) == normalize(gold)
}

pub fn f1(prediction: &str, gold: &str) -> f64 {
    let p = normalize(prediction);
    let g = normalize(gold);
    let pt: Vec<&str> = p.split_whitespace().collect();
    let gt: Vec<&str> = g.split_whitespace().collect();
    if pt.is_empty() || gt.is_empty() {
        return if pt.is_empty() && gt.is_empty() {
            1.0
        } else {
            0.0
        };
    }
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for t in &gt {
        *counts.entry(t).or_default() += 1;
    }
    let mut common = 0i64;
    for t in &pt {
        if let Some(c) = counts.get_mut(t)
            && *c > 0
        {
            *c -= 1;
            common += 1;
        }
    }
    if common == 0 {
        return 0.0;
    }
    let precision = common as f64 / pt.len() as f64;
    let recall = common as f64 / gt.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

const ABSTAIN_PHRASES: &[&str] = &[
    "i don t know",
    "i do not know",
    "not mentioned",
    "no information",
    "not enough information",
    "cannot be determined",
    "can t be determined",
    "unknown",
    "not specified",
    "does not mention",
    "doesn t mention",
    "not stated",
    "unanswerable",
];

/// Whether an answer declines to answer.
pub fn is_abstention(answer: &str) -> bool {
    let n = format!(" {} ", normalize(answer));
    ABSTAIN_PHRASES
        .iter()
        .any(|p| n.contains(&format!(" {p} ")))
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnswerMetrics {
    pub exact_match: bool,
    pub f1: f64,
    pub abstained: bool,
    /// Share of gold evidence turns placed in context; `None` when the
    /// dataset gives no evidence for this question.
    pub evidence_recall: Option<f64>,
}

pub fn score(
    prediction: &str,
    gold: &str,
    abstention_question: bool,
    evidence: &[String],
    included: &[String],
) -> AnswerMetrics {
    let abstained = is_abstention(prediction);
    let (exact_match, f1) = if abstention_question {
        // Gold for unanswerable questions is a refusal; lexical overlap with
        // it is meaningless, so abstaining counts as the match.
        (abstained, if abstained { 1.0 } else { 0.0 })
    } else {
        (exact_match(prediction, gold), f1(prediction, gold))
    };
    let evidence_recall = (!evidence.is_empty()).then(|| {
        let set: std::collections::HashSet<&str> = included.iter().map(String::as_str).collect();
        evidence.iter().filter(|e| set.contains(e.as_str())).count() as f64 / evidence.len() as f64
    });
    AnswerMetrics {
        exact_match,
        f1,
        abstained,
        evidence_recall,
    }
}

/// Metrics for a context that was built but not answered: only evidence
/// recall is meaningful, and reports leave the answer metrics out.
pub fn evidence_only(evidence: &[String], included: &[String]) -> AnswerMetrics {
    AnswerMetrics {
        evidence_recall: score("", "", false, evidence, included).evidence_recall,
        ..AnswerMetrics::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalisation_matches_squad_conventions() {
        assert_eq!(normalize("The  Golden-Gate, Bridge!"), "golden gate bridge");
        assert!(exact_match("an apple", "Apple."));
        assert!(!exact_match("apples", "apple"));
    }

    #[test]
    fn f1_counts_overlapping_tokens_with_multiplicity() {
        assert!((f1("red red car", "red car") - 0.8).abs() < 1e-9);
        assert_eq!(f1("blue", "red car"), 0.0);
        assert_eq!(f1("", ""), 1.0);
        assert_eq!(f1("the", "car"), 0.0);
        assert!((f1("7 May 2023", "May 7, 2023") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn abstention_questions_reward_refusals_only() {
        let refused = score("I don't know.", "Not mentioned", true, &[], &[]);
        assert!(refused.exact_match && refused.abstained);
        let guessed = score("Paris", "Not mentioned", true, &[], &[]);
        assert!(!guessed.exact_match && guessed.f1 == 0.0);
        assert!(!is_abstention("She knows the unknowns of physics well"));
    }

    #[test]
    fn evidence_recall_is_partial_and_absent_without_labels() {
        let ev = vec!["a".to_string(), "b".to_string()];
        let m = score("x", "y", false, &ev, &["b".into(), "c".into()]);
        assert_eq!(m.evidence_recall, Some(0.5));
        assert_eq!(score("x", "y", false, &[], &[]).evidence_recall, None);
    }
}
