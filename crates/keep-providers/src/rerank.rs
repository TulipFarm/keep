//! Reranking contract: a model that scores how well each passage answers a
//! query, read jointly (a cross-encoder), plus an offline stand-in.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::ProviderDescriptor;
use crate::chat::ProviderError;

#[async_trait]
pub trait Reranker: Send + Sync {
    fn descriptor(&self) -> &ProviderDescriptor;
    /// One relevance score per text, in order; higher is more relevant.
    /// A text's score depends only on the query, the text and the model.
    async fn score(&self, query: &str, texts: &[String]) -> Result<Vec<f32>, ProviderError>;
}

/// Offline stand-in: the share of distinct query words (two or more
/// characters, case-folded) that occur in the text. Deterministic and free;
/// it exercises reranking plumbing but is not a relevance model.
#[derive(Clone, Debug)]
pub struct OverlapReranker {
    descriptor: ProviderDescriptor,
}

impl Default for OverlapReranker {
    fn default() -> Self {
        Self {
            descriptor: ProviderDescriptor {
                provider: "mock".into(),
                model: "overlap-reranker-v1".into(),
                revision: "v1".into(),
                mock: true,
            },
        }
    }
}

fn words(text: &str) -> std::collections::BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() > 1)
        .map(str::to_lowercase)
        .collect()
}

#[async_trait]
impl Reranker for OverlapReranker {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    async fn score(&self, query: &str, texts: &[String]) -> Result<Vec<f32>, ProviderError> {
        let q = words(query);
        Ok(texts
            .iter()
            .map(|t| {
                if q.is_empty() {
                    return 0.0;
                }
                let w = words(t);
                q.iter().filter(|x| w.contains(*x)).count() as f32 / q.len() as f32
            })
            .collect())
    }
}

/// Remembers scores in memory, so the same query and passage are scored by
/// the model once per process. Bounded: the memo is cleared when it reaches
/// `capacity` entries.
pub struct MemoReranker {
    inner: Arc<dyn Reranker>,
    capacity: usize,
    memo: Mutex<HashMap<[u8; 32], f32>>,
}

impl MemoReranker {
    pub fn new(inner: Arc<dyn Reranker>, capacity: usize) -> Self {
        Self {
            inner,
            capacity: capacity.max(1),
            memo: Mutex::new(HashMap::new()),
        }
    }

    fn key(query: &str, text: &str) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update((query.len() as u64).to_le_bytes());
        h.update(query.as_bytes());
        h.update(text.as_bytes());
        h.finalize().into()
    }
}

#[async_trait]
impl Reranker for MemoReranker {
    fn descriptor(&self) -> &ProviderDescriptor {
        self.inner.descriptor()
    }

    async fn score(&self, query: &str, texts: &[String]) -> Result<Vec<f32>, ProviderError> {
        let keys: Vec<[u8; 32]> = texts.iter().map(|t| Self::key(query, t)).collect();
        let mut out: Vec<Option<f32>> = {
            let memo = self.memo.lock().expect("rerank memo lock");
            keys.iter().map(|k| memo.get(k).copied()).collect()
        };
        let missing: Vec<usize> = (0..texts.len()).filter(|&i| out[i].is_none()).collect();
        if !missing.is_empty() {
            let todo: Vec<String> = missing.iter().map(|&i| texts[i].clone()).collect();
            let scores = self.inner.score(query, &todo).await?;
            if scores.len() != todo.len() {
                return Err(ProviderError::Malformed {
                    message: format!(
                        "reranker returned {} scores for {} texts",
                        scores.len(),
                        todo.len()
                    ),
                });
            }
            let mut memo = self.memo.lock().expect("rerank memo lock");
            if memo.len() + missing.len() > self.capacity {
                memo.clear();
            }
            for (&i, s) in missing.iter().zip(scores) {
                memo.insert(keys[i], s);
                out[i] = Some(s);
            }
        }
        Ok(out.into_iter().map(|s| s.unwrap_or_default()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counting(OverlapReranker, AtomicUsize);

    #[async_trait]
    impl Reranker for Counting {
        fn descriptor(&self) -> &ProviderDescriptor {
            self.0.descriptor()
        }
        async fn score(&self, q: &str, texts: &[String]) -> Result<Vec<f32>, ProviderError> {
            self.1.fetch_add(texts.len(), Ordering::Relaxed);
            self.0.score(q, texts).await
        }
    }

    #[tokio::test]
    async fn overlap_scores_query_word_coverage() {
        let r = OverlapReranker::default();
        let s = r
            .score(
                "Where did Mira move?",
                &["Mira moved to Lisbon".into(), "Lunch at noon".into()],
            )
            .await
            .unwrap();
        assert!(s[0] > s[1]);
        assert_eq!(s[1], 0.0);
    }

    #[tokio::test]
    async fn memo_scores_each_pair_once_and_keeps_order() {
        let inner = Arc::new(Counting(OverlapReranker::default(), AtomicUsize::new(0)));
        let memo = MemoReranker::new(inner.clone(), 100);
        let texts: Vec<String> = vec!["a b".into(), "b c".into()];
        let first = memo.score("b", &texts).await.unwrap();
        let again = memo
            .score("b", &[texts[1].clone(), texts[0].clone()])
            .await
            .unwrap();
        assert_eq!(first, vec![again[1], again[0]]);
        assert_eq!(inner.1.load(Ordering::Relaxed), 2);
    }
}
