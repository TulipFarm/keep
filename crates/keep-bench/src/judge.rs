//! Judge consistency: one verdict per distinct answer, optionally by vote.
//!
//! The judge model is not deterministic. Without this, two arms (or two
//! runs) that give the same answer to the same question can be scored
//! differently, which adds noise to every paired comparison. With a
//! `[judge_consistency]` section in the manifest:
//!
//! - every verdict is keyed on the judge configuration, the question, its
//!   gold answer and the answer text after light normalisation, and stored
//!   in a cache shared by every arm and run, so an answer that was judged
//!   once is never judged again;
//! - optionally, a borderline answer (or every answer) is judged by
//!   majority of three: a second vote, and a third only when the first two
//!   disagree.
//!
//! Manifests without the section judge exactly as before and hash as before.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use keep_core::{ContentHash, canonical_json};
use serde::{Deserialize, Serialize};

use crate::artifacts::{CallRecord, JudgeOutcome};

pub const DEFAULT_CACHE_DIR: &str = ".keep-cache/judgments";

/// The `[judge_consistency]` manifest section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeConsistencySpec {
    /// Reuse one verdict for every identical (question, gold, answer).
    #[serde(default = "yes")]
    pub cache: bool,
    /// 1 judges once; 3 judges by majority of up to three calls.
    #[serde(default = "one")]
    pub votes: u32,
    /// Which answers get the extra votes when `votes` is 3.
    #[serde(default)]
    pub vote_on: VoteOn,
}

fn yes() -> bool {
    true
}

fn one() -> u32 {
    1
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteOn {
    /// Answers that neither abstain nor clearly miss or match the gold
    /// answer: token F1 strictly between 0 and 0.6. Identical answers with
    /// conflicting single verdicts concentrate here.
    #[default]
    Borderline,
    All,
}

/// Token F1 at or above this is a near-verbatim match, not a borderline one.
pub const BORDERLINE_F1_MAX: f64 = 0.6;

impl JudgeConsistencySpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.votes != 1 && self.votes != 3 {
            return Err(format!("votes must be 1 or 3, got {}", self.votes));
        }
        Ok(())
    }

    /// Whether an answer with these metrics gets the extra votes.
    pub fn votes_for(&self, abstained: bool, f1: f64) -> bool {
        self.votes > 1
            && match self.vote_on {
                VoteOn::All => true,
                VoteOn::Borderline => !abstained && f1 > 0.0 && f1 < BORDERLINE_F1_MAX,
            }
    }
}

/// Lowercase, collapse whitespace, and drop surrounding quotes, emphasis and
/// trailing punctuation: changes that never alter what an answer says.
pub fn normalize_answer(answer: &str) -> String {
    let collapsed = answer.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '*' | '`' | '_'))
        .trim_end_matches(['.', '!', ';', ',', ' '])
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '*' | '`' | '_'));
    trimmed.to_lowercase()
}

/// What a verdict depends on.
#[derive(Serialize)]
pub struct KeyInput<'a> {
    pub judge_provider: serde_json::Value,
    pub judge_model: &'a str,
    pub judge_revision: &'a str,
    pub judge_prompt: &'a str,
    pub protocol: &'a str,
    pub votes: u32,
    pub vote_on: VoteOn,
    pub question_id: &'a str,
    pub question: &'a str,
    pub category: &'a str,
    pub gold: &'a str,
    pub abstention: bool,
    pub answer: String,
}

impl KeyInput<'_> {
    pub fn key(&self) -> String {
        let v = serde_json::to_value(self).expect("key input serialises");
        ContentHash::of_bytes(canonical_json(&v).as_bytes())
            .hex()
            .to_string()
    }
}

/// One stored verdict.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CachedVerdict {
    pub correct: bool,
    /// Each vote cast: `None` for a malformed or failed one.
    pub votes: Vec<Option<bool>>,
}

/// Where a record's verdict came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictSource {
    /// Judged by this record's own calls.
    Judged,
    /// Reused from an identical answer judged earlier, in this run or before.
    Cache,
}

/// How a record's verdict was reached, stored on the record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JudgeConsistencyRecord {
    pub key: String,
    pub source: VerdictSource,
    pub votes: Vec<Option<bool>>,
    /// Calls beyond the first (`judge_call`), billed like it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_calls: Vec<CallRecord>,
}

/// The outcome of judging one answer, possibly by vote.
#[derive(Clone, Debug)]
pub struct Judgement {
    pub judge: JudgeOutcome,
    pub first_call: Option<CallRecord>,
    pub extra_calls: Vec<CallRecord>,
    pub votes: Vec<Option<bool>>,
    pub fatal: Option<String>,
}

/// Combine votes: the side with more votes wins; a tie is no verdict.
pub fn majority(votes: &[Option<bool>]) -> Option<bool> {
    let yes = votes.iter().filter(|v| **v == Some(true)).count();
    let no = votes.iter().filter(|v| **v == Some(false)).count();
    match yes.cmp(&no) {
        std::cmp::Ordering::Greater => Some(true),
        std::cmp::Ordering::Less => Some(false),
        std::cmp::Ordering::Equal => None,
    }
}

/// Whether more votes are needed after `votes`, aiming for `target` total.
pub fn needs_another_vote(votes: &[Option<bool>], target: u32) -> bool {
    let n = votes.len() as u32;
    if n >= target {
        return false;
    }
    // Two matching valid votes already settle a majority of three.
    !(n >= 2 && votes[0].is_some() && votes[0] == votes[1])
}

type Slot = Arc<tokio::sync::OnceCell<Option<CachedVerdict>>>;

/// The verdict cache: on disk across runs, and single-flight within a run
/// so concurrent identical answers make one set of calls.
pub struct VerdictCache {
    dir: Option<PathBuf>,
    slots: std::sync::Mutex<HashMap<String, Slot>>,
}

impl VerdictCache {
    /// `dir` of `None` keeps verdicts for this run only.
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            slots: Default::default(),
        }
    }

    /// The cache directory: `KEEP_JUDGE_CACHE`, or `.keep-cache/judgments`.
    pub fn default_dir() -> PathBuf {
        std::env::var_os("KEEP_JUDGE_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_DIR))
    }

    fn path(&self, key: &str) -> Option<PathBuf> {
        self.dir
            .as_ref()
            .map(|d| d.join(&key[..2]).join(format!("{key}.json")))
    }

    fn load(&self, key: &str) -> Option<CachedVerdict> {
        let body = std::fs::read(self.path(key)?).ok()?;
        serde_json::from_slice(&body).ok()
    }

    fn store(&self, key: &str, v: &CachedVerdict) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let Some(path) = self.path(key) else { return };
        let dir = path.parent().expect("cache path has a parent");
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }
        let nonce = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = dir.join(format!(".{key}.{}.{nonce}.tmp", std::process::id()));
        if std::fs::write(&tmp, serde_json::to_vec(v).expect("verdict serialises")).is_ok() {
            let _ = std::fs::rename(tmp, path);
        }
    }

    fn slot(&self, key: &str) -> Slot {
        self.slots
            .lock()
            .expect("verdict cache lock")
            .entry(key.to_string())
            .or_default()
            .clone()
    }

    /// Judge through the cache. `judge` runs at most once per key across
    /// concurrent callers; a caller that waited on another's verdict, or
    /// found it on disk, gets `VerdictSource::Cache` and no calls. A
    /// verdict-less judgement (failed or malformed) is not cached, so a
    /// waiting caller then judges on its own.
    pub async fn judge<F, Fut>(&self, key: &str, judge: F) -> (Judgement, VerdictSource)
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Judgement>,
    {
        let slot = self.slot(key);
        let mut own: Option<Judgement> = None;
        let cached = slot
            .get_or_init(|| async {
                if let Some(v) = self.load(key) {
                    return Some(v);
                }
                let j = judge().await;
                let verdict = match &j.judge {
                    JudgeOutcome::Judged { correct } => Some(CachedVerdict {
                        correct: *correct,
                        votes: j.votes.clone(),
                    }),
                    _ => None,
                };
                if let Some(v) = &verdict {
                    self.store(key, v);
                }
                own = Some(j);
                verdict
            })
            .await
            .clone();
        if let Some(j) = own {
            return (j, VerdictSource::Judged);
        }
        match cached {
            Some(v) => (
                Judgement {
                    judge: JudgeOutcome::Judged { correct: v.correct },
                    first_call: None,
                    extra_calls: Vec::new(),
                    votes: v.votes,
                    fatal: None,
                },
                VerdictSource::Cache,
            ),
            None => (judge().await, VerdictSource::Judged),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalisation_ignores_case_space_and_trailing_punctuation() {
        assert_eq!(normalize_answer("  Late June\n 2023. "), "late june 2023");
        assert_eq!(normalize_answer("\"Paris\"!"), "paris");
        assert_eq!(normalize_answer("**Yes**"), "yes");
        assert_ne!(normalize_answer("In late June 2023"), "late june 2023");
    }

    #[test]
    fn borderline_means_partial_overlap_without_abstaining() {
        let spec = JudgeConsistencySpec {
            cache: true,
            votes: 3,
            vote_on: VoteOn::Borderline,
        };
        assert!(spec.votes_for(false, 0.3));
        assert!(!spec.votes_for(false, 0.0));
        assert!(!spec.votes_for(false, 0.6));
        assert!(!spec.votes_for(true, 0.3));
        let single = JudgeConsistencySpec { votes: 1, ..spec };
        assert!(!single.votes_for(false, 0.3));
        assert!(single.validate().is_ok());
        assert!(
            JudgeConsistencySpec { votes: 2, ..spec }
                .validate()
                .is_err()
        );
    }

    #[test]
    fn majority_of_three_stops_early_on_agreement() {
        assert!(needs_another_vote(&[], 3));
        assert!(needs_another_vote(&[Some(true)], 3));
        assert!(!needs_another_vote(&[Some(true), Some(true)], 3));
        assert!(needs_another_vote(&[Some(true), Some(false)], 3));
        assert!(needs_another_vote(&[None, None], 3));
        assert!(!needs_another_vote(&[Some(true), Some(false), None], 3));
        assert_eq!(majority(&[Some(true), Some(false), Some(true)]), Some(true));
        assert_eq!(majority(&[Some(true), Some(false), None]), None);
        assert_eq!(majority(&[None, Some(false), None]), Some(false));
    }

    fn judged(correct: bool) -> Judgement {
        Judgement {
            judge: JudgeOutcome::Judged { correct },
            first_call: None,
            extra_calls: Vec::new(),
            votes: vec![Some(correct)],
            fatal: None,
        }
    }

    #[tokio::test]
    async fn identical_keys_share_one_verdict_and_persist() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let judge = || async {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            tokio::task::yield_now().await;
            judged(true)
        };
        let cache = VerdictCache::new(Some(dir.clone()));
        let (a, b) = tokio::join!(cache.judge("ab12", judge), cache.judge("ab12", judge));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        let sources = [a.1, b.1];
        assert!(sources.contains(&VerdictSource::Judged));
        assert!(sources.contains(&VerdictSource::Cache));
        assert_eq!(a.0.judge, b.0.judge);

        // A new run reads the verdict from disk without calling the judge.
        let again = VerdictCache::new(Some(dir.clone()));
        let (j, src) = again.judge("ab12", || async { judged(false) }).await;
        assert_eq!(src, VerdictSource::Cache);
        assert_eq!(j.judge, JudgeOutcome::Judged { correct: true });
    }

    #[tokio::test]
    async fn failed_verdicts_are_not_cached() {
        let cache = VerdictCache::new(None);
        let failed = || async {
            Judgement {
                judge: JudgeOutcome::Malformed { reply: "?".into() },
                first_call: None,
                extra_calls: Vec::new(),
                votes: vec![None],
                fatal: None,
            }
        };
        let (j, _) = cache.judge("cd34", failed).await;
        assert!(matches!(j.judge, JudgeOutcome::Malformed { .. }));
        let (j, src) = cache.judge("cd34", || async { judged(false) }).await;
        assert_eq!(src, VerdictSource::Judged);
        assert_eq!(j.judge, JudgeOutcome::Judged { correct: false });
    }
}
