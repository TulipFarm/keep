use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distance {
    Cosine,
}

/// What a backend can do. Callers check these instead of assuming every
/// database behaves alike.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorCapabilities {
    pub backend: String,
    pub distance: Distance,
    pub max_dimensions: usize,
    /// A search issued after `upsert` returns sees the upserted points.
    pub read_your_writes: bool,
    /// Namespace filtering happens inside the backend.
    pub namespace_filter: bool,
    /// Results are exact nearest neighbours rather than approximate.
    pub exact_search: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorPoint {
    /// Caller-stable id; the backend may derive its own key from it.
    pub id: String,
    pub vector: Vec<f32>,
    /// Opaque labels a search can be restricted to with
    /// [`VectorIndex::search_tagged`], such as digests of the point's sources.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoredId {
    pub id: String,
    pub score: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("unsupported capability: {0}")]
    Capability(String),
    #[error("vector has {got} dimensions, collection expects {expected}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("unknown collection {0}")]
    UnknownCollection(String),
    #[error("backend error: {0}")]
    Backend(String),
}

/// A derived vector projection. Namespaces isolate independent histories
/// (one conversation or user) so a search can never see another's points.
#[async_trait]
pub trait VectorIndex: Send + Sync {
    fn capabilities(&self) -> VectorCapabilities;
    async fn ensure_collection(
        &self,
        collection: &str,
        dimensions: usize,
    ) -> Result<(), StoreError>;
    async fn upsert(
        &self,
        collection: &str,
        namespace: &str,
        points: &[VectorPoint],
    ) -> Result<(), StoreError>;
    async fn search(
        &self,
        collection: &str,
        namespace: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<ScoredId>, StoreError>;
    /// [`VectorIndex::search`] over only the points carrying at least one of
    /// `tags`, filtered inside the backend before the top `k` are chosen.
    /// No tags matches nothing.
    async fn search_tagged(
        &self,
        collection: &str,
        namespace: &str,
        query: &[f32],
        k: usize,
        tags: &[String],
    ) -> Result<Vec<ScoredId>, StoreError> {
        let _ = (collection, namespace, query, k, tags);
        Err(StoreError::Capability("search_tagged".into()))
    }
    async fn delete_namespace(&self, collection: &str, namespace: &str) -> Result<(), StoreError>;
    /// Remove individual points. Deleting ids that do not exist is not an
    /// error, so a purge can be replayed safely.
    async fn delete_points(
        &self,
        collection: &str,
        namespace: &str,
        ids: &[String],
    ) -> Result<(), StoreError> {
        let _ = (collection, namespace, ids);
        Err(StoreError::Capability("delete_points".into()))
    }
}
