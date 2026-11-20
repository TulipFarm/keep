//! Plain BM25 over an in-memory corpus.

use std::collections::{BTreeMap, HashMap};

use crate::vector::ScoredId;

pub trait LexicalIndex {
    /// Tokenizer and scoring declaration recorded in artifacts.
    fn method(&self) -> String;
    fn search(&self, query: &str, k: usize) -> Vec<ScoredId>;
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Lower-cased alphanumeric terms; no stemming and no stop-word list, so the
/// baseline stays plain and reproducible.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect()
}

pub struct Bm25Index {
    params: Bm25Params,
    ids: Vec<String>,
    doc_terms: Vec<HashMap<String, u32>>,
    doc_len: Vec<f32>,
    avg_len: f32,
    doc_freq: BTreeMap<String, u32>,
}

impl Bm25Index {
    pub fn build(params: Bm25Params, docs: impl IntoIterator<Item = (String, String)>) -> Self {
        let mut ids = Vec::new();
        let mut doc_terms = Vec::new();
        let mut doc_len = Vec::new();
        let mut doc_freq: BTreeMap<String, u32> = BTreeMap::new();
        for (id, text) in docs {
            let terms = tokenize(&text);
            let mut tf: HashMap<String, u32> = HashMap::new();
            for t in &terms {
                *tf.entry(t.clone()).or_default() += 1;
            }
            for t in tf.keys() {
                *doc_freq.entry(t.clone()).or_default() += 1;
            }
            ids.push(id);
            doc_len.push(terms.len() as f32);
            doc_terms.push(tf);
        }
        let avg_len = if doc_len.is_empty() {
            0.0
        } else {
            doc_len.iter().sum::<f32>() / doc_len.len() as f32
        };
        Self {
            params,
            ids,
            doc_terms,
            doc_len,
            avg_len,
            doc_freq,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn idf(&self, term: &str) -> f32 {
        let n = self.ids.len() as f32;
        let df = *self.doc_freq.get(term).unwrap_or(&0) as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }
}

impl LexicalIndex for Bm25Index {
    fn method(&self) -> String {
        format!(
            "bm25(k1={},b={});tokenizer=lower-alnum-v1;stopwords=none",
            self.params.k1, self.params.b
        )
    }

    fn search(&self, query: &str, k: usize) -> Vec<ScoredId> {
        let mut terms = tokenize(query);
        terms.sort();
        terms.dedup();
        let Bm25Params { k1, b } = self.params;
        let hits = self
            .doc_terms
            .iter()
            .enumerate()
            .filter_map(|(i, tf)| {
                let score: f32 = terms
                    .iter()
                    .filter_map(|t| tf.get(t).map(|f| (t, *f as f32)))
                    .map(|(t, f)| {
                        let norm = 1.0 - b + b * self.doc_len[i] / self.avg_len.max(1e-6);
                        self.idf(t) * f * (k1 + 1.0) / (f + k1 * norm)
                    })
                    .sum();
                (score > 0.0).then(|| ScoredId {
                    id: self.ids[i].clone(),
                    score,
                })
            })
            .collect();
        crate::rank(hits, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_prefers_rare_matching_terms_and_breaks_ties_by_id() {
        let idx = Bm25Index::build(
            Bm25Params::default(),
            [
                ("d1".to_string(), "we talked about the weather".to_string()),
                (
                    "d2".to_string(),
                    "my greyhound Comet loves the beach".to_string(),
                ),
                ("d3".to_string(), "the the the weather".to_string()),
            ],
        );
        let hits = idx.search("What is the greyhound called?", 3);
        assert_eq!(hits[0].id, "d2");
        let tie = Bm25Index::build(
            Bm25Params::default(),
            [
                ("b".to_string(), "x".to_string()),
                ("a".to_string(), "x".to_string()),
            ],
        );
        let ids: Vec<_> = tie.search("x", 2).into_iter().map(|h| h.id).collect();
        assert_eq!(ids, ["a", "b"]);
        assert!(idx.search("zebra", 3).is_empty());
    }
}
