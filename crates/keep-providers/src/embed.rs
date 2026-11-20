//! Embedding contract and the offline hashing embedder.

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::ProviderDescriptor;
use crate::chat::{ProviderError, Usage};

#[derive(Clone, Debug, PartialEq)]
pub struct EmbedResponse {
    pub vectors: Vec<Vec<f32>>,
    pub usage: Usage,
}

#[async_trait]
pub trait Embedder: Send + Sync {
    fn descriptor(&self) -> &ProviderDescriptor;
    fn dimensions(&self) -> usize;
    async fn embed(&self, texts: &[String]) -> Result<EmbedResponse, ProviderError>;
}

/// Feature-hashing bag-of-words embedder. Deterministic, offline and free; it
/// exercises dense-retrieval plumbing but is not a semantic model, so it must
/// never stand in for a pinned embedding model in a reported result.
#[derive(Clone, Debug)]
pub struct HashEmbedder {
    descriptor: ProviderDescriptor,
    dimensions: usize,
}

impl HashEmbedder {
    pub fn new(dimensions: usize) -> Self {
        assert!(dimensions > 0, "embedding dimensions must be positive");
        Self {
            descriptor: ProviderDescriptor {
                provider: "mock".into(),
                model: "hash-embedder-v1".into(),
                revision: format!("dims-{dimensions}"),
                mock: true,
            },
            dimensions,
        }
    }

    pub fn embed_one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dimensions];
        for word in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
        {
            let digest = Sha256::digest(word.to_lowercase().as_bytes());
            let bucket = u64::from_le_bytes(digest[..8].try_into().expect("8 bytes"));
            let sign = if digest[8] & 1 == 0 { 1.0 } else { -1.0 };
            v[(bucket % self.dimensions as u64) as usize] += sign;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbedResponse, ProviderError> {
        Ok(EmbedResponse {
            vectors: texts.iter().map(|t| self.embed_one(t)).collect(),
            usage: Usage::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn hash_embedder_is_deterministic_normalised_and_overlap_sensitive() {
        let e = HashEmbedder::new(128);
        let a = e.embed_one("Priya adopted a greyhound named Comet");
        assert_eq!(a, e.embed_one("Priya adopted a greyhound named Comet"));
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5);
        let near = e.embed_one("What is the name of Priya's greyhound?");
        let far = e.embed_one("The quarterly budget review moved to Friday");
        assert!(cosine(&a, &near) > cosine(&a, &far));
        assert!(e.embed_one("").iter().all(|x| *x == 0.0));
    }
}
