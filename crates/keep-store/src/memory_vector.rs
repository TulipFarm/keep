use std::collections::BTreeMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::vector::{Distance, ScoredId, StoreError, VectorCapabilities, VectorIndex, VectorPoint};

/// A stored vector and its tags.
type Entry = (Vec<f32>, Vec<String>);

#[derive(Default)]
struct Collection {
    dimensions: usize,
    namespaces: BTreeMap<String, BTreeMap<String, Entry>>,
}

/// Exact brute-force cosine search held in process memory.
#[derive(Default)]
pub struct InMemoryVectorIndex {
    collections: Mutex<BTreeMap<String, Collection>>,
}

impl InMemoryVectorIndex {
    pub fn new() -> Self {
        Self::default()
    }

    fn scan(
        &self,
        collection: &str,
        namespace: &str,
        query: &[f32],
        k: usize,
        tags: Option<&[String]>,
    ) -> Result<Vec<ScoredId>, StoreError> {
        let cols = self.collections.lock().expect("lock");
        let col = cols
            .get(collection)
            .ok_or_else(|| StoreError::UnknownCollection(collection.into()))?;
        if query.len() != col.dimensions {
            return Err(StoreError::DimensionMismatch {
                expected: col.dimensions,
                got: query.len(),
            });
        }
        let hits = col
            .namespaces
            .get(namespace)
            .map(|ns| {
                ns.iter()
                    .filter(|(_, (_, t))| {
                        tags.is_none_or(|want| t.iter().any(|x| want.contains(x)))
                    })
                    .map(|(id, (v, _))| ScoredId {
                        id: id.clone(),
                        score: cosine(query, v),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(crate::rank(hits, k))
    }
}

pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[async_trait]
impl VectorIndex for InMemoryVectorIndex {
    fn capabilities(&self) -> VectorCapabilities {
        VectorCapabilities {
            backend: "in-memory".into(),
            distance: Distance::Cosine,
            max_dimensions: 65_536,
            read_your_writes: true,
            namespace_filter: true,
            exact_search: true,
        }
    }

    async fn ensure_collection(
        &self,
        collection: &str,
        dimensions: usize,
    ) -> Result<(), StoreError> {
        let mut cols = self.collections.lock().expect("lock");
        let col = cols
            .entry(collection.to_owned())
            .or_insert_with(|| Collection {
                dimensions,
                ..Default::default()
            });
        if col.dimensions != dimensions {
            return Err(StoreError::DimensionMismatch {
                expected: col.dimensions,
                got: dimensions,
            });
        }
        Ok(())
    }

    async fn upsert(
        &self,
        collection: &str,
        namespace: &str,
        points: &[VectorPoint],
    ) -> Result<(), StoreError> {
        let mut cols = self.collections.lock().expect("lock");
        let col = cols
            .get_mut(collection)
            .ok_or_else(|| StoreError::UnknownCollection(collection.into()))?;
        for p in points {
            if p.vector.len() != col.dimensions {
                return Err(StoreError::DimensionMismatch {
                    expected: col.dimensions,
                    got: p.vector.len(),
                });
            }
        }
        let ns = col.namespaces.entry(namespace.to_owned()).or_default();
        for p in points {
            ns.insert(p.id.clone(), (p.vector.clone(), p.tags.clone()));
        }
        Ok(())
    }

    async fn search(
        &self,
        collection: &str,
        namespace: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<ScoredId>, StoreError> {
        self.scan(collection, namespace, query, k, None)
    }

    async fn search_tagged(
        &self,
        collection: &str,
        namespace: &str,
        query: &[f32],
        k: usize,
        tags: &[String],
    ) -> Result<Vec<ScoredId>, StoreError> {
        self.scan(collection, namespace, query, k, Some(tags))
    }

    async fn delete_namespace(&self, collection: &str, namespace: &str) -> Result<(), StoreError> {
        if let Some(col) = self.collections.lock().expect("lock").get_mut(collection) {
            col.namespaces.remove(namespace);
        }
        Ok(())
    }

    async fn delete_points(
        &self,
        collection: &str,
        namespace: &str,
        ids: &[String],
    ) -> Result<(), StoreError> {
        if let Some(ns) = self
            .collections
            .lock()
            .expect("lock")
            .get_mut(collection)
            .and_then(|c| c.namespaces.get_mut(namespace))
        {
            for id in ids {
                ns.remove(id);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn namespaces_are_isolated_and_dimensions_enforced() {
        let idx = InMemoryVectorIndex::new();
        idx.ensure_collection("c", 2).await.unwrap();
        idx.upsert(
            "c",
            "a",
            &[VectorPoint {
                id: "x".into(),
                vector: vec![1.0, 0.0],
                tags: vec![],
            }],
        )
        .await
        .unwrap();
        idx.upsert(
            "c",
            "b",
            &[VectorPoint {
                id: "y".into(),
                vector: vec![1.0, 0.0],
                tags: vec![],
            }],
        )
        .await
        .unwrap();
        let hits = idx.search("c", "a", &[1.0, 0.0], 10).await.unwrap();
        assert_eq!(
            hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(),
            ["x"]
        );
        assert!(matches!(
            idx.upsert(
                "c",
                "a",
                &[VectorPoint {
                    id: "z".into(),
                    vector: vec![1.0],
                    tags: vec![],
                }]
            )
            .await,
            Err(StoreError::DimensionMismatch { .. })
        ));
        idx.delete_points("c", "b", &["y".into(), "missing".into()])
            .await
            .unwrap();
        assert!(
            idx.search("c", "b", &[1.0, 0.0], 10)
                .await
                .unwrap()
                .is_empty()
        );
        idx.delete_namespace("c", "a").await.unwrap();
        assert!(
            idx.search("c", "a", &[1.0, 0.0], 10)
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn tagged_search_filters_before_the_limit() {
        let idx = InMemoryVectorIndex::new();
        idx.ensure_collection("c", 2).await.unwrap();
        let point = |id: &str, v: Vec<f32>, tags: &[&str]| VectorPoint {
            id: id.into(),
            vector: v,
            tags: tags.iter().map(|t| t.to_string()).collect(),
        };
        idx.upsert(
            "c",
            "a",
            &[
                point("near", vec![1.0, 0.0], &["s1"]),
                point("mid", vec![0.9, 0.1], &["s1", "s2"]),
                point("far", vec![0.0, 1.0], &["s3"]),
            ],
        )
        .await
        .unwrap();
        let ids = |hits: Vec<ScoredId>| hits.into_iter().map(|h| h.id).collect::<Vec<_>>();
        let only_s3 = idx
            .search_tagged("c", "a", &[1.0, 0.0], 1, &["s3".into()])
            .await
            .unwrap();
        assert_eq!(ids(only_s3), ["far"]);
        let s2 = idx
            .search_tagged("c", "a", &[1.0, 0.0], 5, &["s2".into()])
            .await
            .unwrap();
        assert_eq!(ids(s2), ["mid"]);
        assert!(
            idx.search_tagged("c", "a", &[1.0, 0.0], 5, &[])
                .await
                .unwrap()
                .is_empty()
        );
    }
}
