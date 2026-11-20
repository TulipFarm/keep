//! Length-only full-context fit mask.
//!
//! Whether a question's complete history fits the answer model is decided
//! once, before any answers exist, from token counts alone: the rendered
//! history, the question text, the fixed prompt template and the reserved
//! output. The computation only receives [`LengthView`]s, which carry no gold
//! answers or evidence, so the mask cannot depend on them. The frozen mask is
//! pinned in the manifest by hash.

use keep_core::ContentHash;
use keep_providers::Tokenizer;
use serde::{Deserialize, Serialize};

use crate::context::render_full_history;
use crate::dataset::{Dataset, History, Query};
use crate::prompt;

/// The parts of a question that length fitting may look at.
pub struct LengthView<'a> {
    pub history: &'a History,
    pub query: &'a Query,
}

pub fn length_views(dataset: &Dataset) -> Vec<LengthView<'_>> {
    dataset
        .questions
        .iter()
        .filter_map(|q| {
            dataset.history(&q.history_id).map(|history| LengthView {
                history,
                query: &q.query,
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitEntry {
    pub question_id: String,
    pub input_tokens: usize,
    pub fits: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitMask {
    pub schema: String,
    pub dataset_hash: ContentHash,
    pub tokenizer: String,
    pub tokenizer_exact: bool,
    pub prompt_version: String,
    pub context_limit: usize,
    pub output_reserve: usize,
    pub entries: Vec<FitEntry>,
    pub fit_fraction: f64,
    /// Whether enough questions fit for full context to serve as a gate
    /// comparator under the manifest's minimum fit fraction.
    pub gate_eligible: bool,
    pub min_fit_fraction: f64,
}

impl FitMask {
    pub fn compute(
        dataset_hash: &ContentHash,
        views: &[LengthView<'_>],
        tokenizer: &dyn Tokenizer,
        context_limit: usize,
        output_reserve: usize,
        min_fit_fraction: f64,
    ) -> FitMask {
        let entries: Vec<FitEntry> = views
            .iter()
            .map(|v| {
                let (history_text, _) = render_full_history(v.history, v.query.asked_at);
                let messages = prompt::answer_messages(&history_text, v.query);
                let input_tokens: usize =
                    messages.iter().map(|m| tokenizer.count(&m.content)).sum();
                FitEntry {
                    question_id: v.query.id.clone(),
                    input_tokens,
                    fits: input_tokens + output_reserve <= context_limit,
                }
            })
            .collect();
        let fit = entries.iter().filter(|e| e.fits).count();
        let fit_fraction = if entries.is_empty() {
            0.0
        } else {
            fit as f64 / entries.len() as f64
        };
        FitMask {
            schema: "keep-fit-mask-v1".into(),
            dataset_hash: dataset_hash.clone(),
            tokenizer: tokenizer.id().into(),
            tokenizer_exact: tokenizer.exact(),
            prompt_version: prompt::ANSWER_PROMPT_VERSION.into(),
            context_limit,
            output_reserve,
            entries,
            fit_fraction,
            gate_eligible: fit_fraction >= min_fit_fraction,
            min_fit_fraction,
        }
    }

    pub fn fits(&self, question_id: &str) -> Option<bool> {
        self.entries
            .iter()
            .find(|e| e.question_id == question_id)
            .map(|e| e.fits)
    }
}

/// One full-context answer call, as the run sent it.
#[derive(Clone, Debug, PartialEq)]
pub struct SentPrompt {
    pub question_id: String,
    /// The harness's own count of the prompt it sent.
    pub estimated_tokens: usize,
    /// The provider's `prompt_tokens` for that call, when it reported one.
    pub reported_tokens: Option<u64>,
    /// The call failed, so there is no usage to check.
    pub failed: bool,
}

/// The fit mask's decisions checked after a run against the token counts
/// the answer model's provider reported, for models whose tokenizer is not
/// published. Every full-context prompt the run sent is checked against its
/// exact count. Decisions for questions the run did not send (the other
/// split, or masked-out questions) are re-derived from their mask count
/// scaled by the lowest and highest provider-to-estimate ratio seen; any
/// decision that would flip inside that range counts as unverified.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FitVerification {
    pub method: String,
    pub calls_checked: usize,
    /// Successful calls without a reported prompt count. Any blocks
    /// verification.
    pub calls_unreported: usize,
    /// Failed calls: no usage to check; their outcome is counted by the
    /// completion criterion instead.
    pub calls_failed: usize,
    /// Questions whose reported count contradicts the mask's decision.
    pub disagreements: Vec<String>,
    pub min_ratio: f64,
    pub max_ratio: f64,
    pub max_reported_prompt_tokens: u64,
    /// Mask entries the run did not send whose decision could flip within
    /// the observed ratio range.
    pub uncertain_unsent: usize,
    pub verified: bool,
}

pub const REPORTED_FIT_METHOD: &str = "provider-reported-prompt-tokens";

impl FitMask {
    pub fn verify_reported(&self, sent: &[SentPrompt]) -> FitVerification {
        let mut v = FitVerification {
            method: REPORTED_FIT_METHOD.into(),
            min_ratio: f64::INFINITY,
            max_ratio: 0.0,
            ..FitVerification::default()
        };
        let mut seen = std::collections::BTreeSet::new();
        let mut disagree = std::collections::BTreeSet::new();
        for s in sent {
            seen.insert(s.question_id.as_str());
            if s.failed {
                v.calls_failed += 1;
                continue;
            }
            let Some(reported) = s.reported_tokens else {
                v.calls_unreported += 1;
                continue;
            };
            v.calls_checked += 1;
            v.max_reported_prompt_tokens = v.max_reported_prompt_tokens.max(reported);
            if s.estimated_tokens > 0 {
                let r = reported as f64 / s.estimated_tokens as f64;
                v.min_ratio = v.min_ratio.min(r);
                v.max_ratio = v.max_ratio.max(r);
            }
            let fits = reported as usize + self.output_reserve <= self.context_limit;
            if self.fits(&s.question_id).is_some_and(|m| m != fits) {
                disagree.insert(s.question_id.clone());
            }
        }
        v.disagreements = disagree.into_iter().collect();
        if v.calls_checked == 0 || !v.min_ratio.is_finite() {
            v.min_ratio = 0.0;
            v.verified = false;
            return v;
        }
        let decide = |tokens: f64| tokens + self.output_reserve as f64 <= self.context_limit as f64;
        v.uncertain_unsent = self
            .entries
            .iter()
            .filter(|e| !seen.contains(e.question_id.as_str()))
            .filter(|e| {
                let t = e.input_tokens as f64;
                decide(t * v.min_ratio) != e.fits || decide(t * v.max_ratio) != e.fits
            })
            .count();
        v.verified =
            v.calls_unreported == 0 && v.disagreements.is_empty() && v.uncertain_unsent == 0;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(entries: &[(&str, usize)], limit: usize) -> FitMask {
        let entries: Vec<FitEntry> = entries
            .iter()
            .map(|(id, n)| FitEntry {
                question_id: (*id).into(),
                input_tokens: *n,
                fits: n + 100 <= limit,
            })
            .collect();
        FitMask {
            schema: "keep-fit-mask-v1".into(),
            dataset_hash: ContentHash::of_bytes(b"d"),
            tokenizer: "tiktoken-o200k-base".into(),
            tokenizer_exact: false,
            prompt_version: "answer-v1".into(),
            context_limit: limit,
            output_reserve: 100,
            fit_fraction: 1.0,
            gate_eligible: true,
            min_fit_fraction: 0.8,
            entries,
        }
    }

    fn sent(id: &str, est: usize, reported: Option<u64>) -> SentPrompt {
        SentPrompt {
            question_id: id.into(),
            estimated_tokens: est,
            reported_tokens: reported,
            failed: false,
        }
    }

    #[test]
    fn decisions_with_wide_headroom_verify_from_reported_counts() {
        let m = mask(&[("a", 1000), ("b", 2000), ("held", 3000)], 100_000);
        let v = m.verify_reported(&[sent("a", 1000, Some(1100)), sent("b", 2000, Some(2050))]);
        assert!(v.verified, "{v:?}");
        assert_eq!((v.calls_checked, v.uncertain_unsent), (2, 0));
        assert!((v.max_ratio - 1.1).abs() < 1e-9);
    }

    #[test]
    fn a_reported_count_that_breaks_the_decision_blocks_verification() {
        let m = mask(&[("a", 900)], 1000);
        let v = m.verify_reported(&[sent("a", 900, Some(950))]);
        assert_eq!(v.disagreements, vec!["a".to_string()]);
        assert!(!v.verified);
    }

    #[test]
    fn unsent_decisions_near_the_limit_and_missing_usage_stay_unverified() {
        // "edge" fits by 50 tokens on the estimate; a 10% ratio could flip it.
        let m = mask(&[("a", 1000), ("edge", 9850)], 10_000);
        let v = m.verify_reported(&[sent("a", 1000, Some(1100))]);
        assert_eq!(v.uncertain_unsent, 1);
        assert!(!v.verified);

        let m = mask(&[("a", 1000)], 100_000);
        let v = m.verify_reported(&[sent("a", 1000, None)]);
        assert_eq!(v.calls_unreported, 1);
        assert!(!v.verified);
        assert!(!m.verify_reported(&[]).verified);
    }
}
