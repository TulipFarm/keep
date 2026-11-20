//! Qdrant adapter over the REST API.
//!
//! Qdrant holds only derived vectors and the minimum payload needed to filter
//! by namespace and map hits back to caller ids. It is never the authority for
//! records, permissions or deletion; that stays in the Postgres ledger.
//! Searches request exact (non-HNSW) scoring so baseline results are
//! reproducible.

use async_trait::async_trait;
use keep_store::{Distance, ScoredId, StoreError, VectorCapabilities, VectorIndex, VectorPoint};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

pub struct QdrantIndex {
    base_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
    /// Vector sizes of collections seen to exist, so upserts and searches
    /// skip a lookup per call. Forgotten when a request on it fails.
    known_dims: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    /// Collections whose payload indexes this process has ensured.
    indexed: std::sync::Mutex<std::collections::HashSet<String>>,
}

fn backend(e: impl std::fmt::Display) -> StoreError {
    StoreError::Backend(e.to_string())
}

/// Deterministic point id: a UUID-shaped digest of namespace and caller id,
/// so re-upserting the same point overwrites rather than duplicates it.
pub fn point_id(namespace: &str, id: &str) -> String {
    let mut h = Sha256::new();
    h.update(namespace.as_bytes());
    h.update([0u8]);
    h.update(id.as_bytes());
    let d = hex::encode(&h.finalize()[..16]);
    format!(
        "{}-{}-{}-{}-{}",
        &d[..8],
        &d[8..12],
        &d[12..16],
        &d[16..20],
        &d[20..32]
    )
}

fn namespace_filter(namespace: &str) -> Value {
    json!({"must": [{"key": "ns", "match": {"value": namespace}}]})
}

/// Points in `namespace` carrying at least one of `tags`.
fn tagged_filter(namespace: &str, tags: &[String]) -> Value {
    json!({"must": [
        {"key": "ns", "match": {"value": namespace}},
        {"key": "tags", "match": {"any": tags}},
    ]})
}

impl QdrantIndex {
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        Self::connect(base_url, api_key, None).expect("a client without extra roots always builds")
    }

    /// A client for `base_url`, also trusting the PEM certificates in
    /// `ca_pem` (for a Qdrant behind a private CA). Every request has a
    /// deadline, so an unreachable Qdrant fails instead of hanging.
    pub fn connect(
        base_url: &str,
        api_key: Option<String>,
        ca_pem: Option<&[u8]>,
    ) -> Result<Self, StoreError> {
        let mut builder = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT);
        if let Some(pem) = ca_pem {
            let certs = reqwest::Certificate::from_pem_bundle(pem)
                .map_err(|e| StoreError::Backend(format!("Qdrant CA certificate: {e}")))?;
            if certs.is_empty() {
                return Err(StoreError::Backend(
                    "Qdrant CA file holds no certificates".into(),
                ));
            }
            for c in certs {
                builder = builder.add_root_certificate(c);
            }
        }
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            api_key,
            client: builder.build().map_err(backend)?,
            known_dims: Default::default(),
            indexed: Default::default(),
        })
    }

    async fn call(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<(u16, Value), StoreError> {
        let mut req = self
            .client
            .request(method, format!("{}{}", self.base_url, path));
        if let Some(key) = &self.api_key {
            req = req.header("api-key", key);
        }
        if let Some(body) = body {
            req = req.json(&body);
        }
        let resp = req.send().await.map_err(backend)?;
        let status = resp.status().as_u16();
        let value = resp.json::<Value>().await.unwrap_or(Value::Null);
        Ok((status, value))
    }

    async fn ok(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value, StoreError> {
        let (status, value) = self.call(method, path, body).await?;
        if (200..300).contains(&status) {
            Ok(value)
        } else {
            Err(StoreError::Backend(format!(
                "qdrant {path} returned {status}: {value}"
            )))
        }
    }

    async fn collection_dims(&self, collection: &str) -> Result<Option<usize>, StoreError> {
        let (status, value) = self
            .call(
                reqwest::Method::GET,
                &format!("/collections/{collection}"),
                None,
            )
            .await?;
        match status {
            404 => {
                self.forget(collection);
                Ok(None)
            }
            200 => {
                let dims = value["result"]["config"]["params"]["vectors"]["size"]
                    .as_u64()
                    .ok_or_else(|| backend("collection has no single unnamed vector config"))?
                    as usize;
                self.known().insert(collection.to_owned(), dims);
                Ok(Some(dims))
            }
            _ => Err(StoreError::Backend(format!(
                "qdrant returned {status}: {value}"
            ))),
        }
    }

    /// Drop a whole collection; a missing one is not an error.
    pub async fn delete_collection(&self, collection: &str) -> Result<(), StoreError> {
        check_name(collection)?;
        self.forget(collection);
        let (status, value) = self
            .call(
                reqwest::Method::DELETE,
                &format!("/collections/{collection}"),
                None,
            )
            .await?;
        match status {
            200..=299 | 404 => Ok(()),
            _ => Err(StoreError::Backend(format!(
                "qdrant returned {status}: {value}"
            ))),
        }
    }

    fn known(&self) -> std::sync::MutexGuard<'_, std::collections::HashMap<String, usize>> {
        self.known_dims.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn indexed_lock(&self) -> std::sync::MutexGuard<'_, std::collections::HashSet<String>> {
        self.indexed.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn forget(&self, collection: &str) {
        self.known().remove(collection);
        self.indexed_lock().remove(collection);
    }

    /// Exact nearest neighbours among the points matching `filter`.
    async fn query(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        filter: Value,
    ) -> Result<Vec<ScoredId>, StoreError> {
        check_name(collection)?;
        self.require_dims(collection, query.len()).await?;
        let value = self
            .ok(
                reqwest::Method::POST,
                &format!("/collections/{collection}/points/query"),
                Some(json!({
                    "query": query,
                    "filter": filter,
                    "limit": k,
                    "with_payload": ["key"],
                    "params": {"exact": true},
                })),
            )
            .await
            .inspect_err(|_| self.forget(collection))?;
        let points = value["result"]["points"]
            .as_array()
            .ok_or_else(|| backend("query response has no result.points"))?;
        let hits = points
            .iter()
            .map(|p| {
                Ok(ScoredId {
                    id: p["payload"]["key"]
                        .as_str()
                        .ok_or_else(|| backend("point payload has no key"))?
                        .to_owned(),
                    score: p["score"].as_f64().unwrap_or(0.0) as f32,
                })
            })
            .collect::<Result<Vec<_>, StoreError>>()?;
        Ok(keep_store::rank(hits, k))
    }

    async fn require_dims(&self, collection: &str, got: usize) -> Result<(), StoreError> {
        let cached = self.known().get(collection).copied();
        let dims = match cached {
            Some(d) => Some(d),
            None => self.collection_dims(collection).await?,
        };
        match dims {
            None => Err(StoreError::UnknownCollection(collection.into())),
            Some(expected) if expected != got => {
                Err(StoreError::DimensionMismatch { expected, got })
            }
            Some(_) => Ok(()),
        }
    }
}

fn check_name(collection: &str) -> Result<(), StoreError> {
    let ok = !collection.is_empty()
        && collection.len() <= 64
        && collection
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if ok {
        Ok(())
    } else {
        Err(StoreError::Capability(format!(
            "collection name {collection:?} must be 1-64 chars of [A-Za-z0-9_-]"
        )))
    }
}

#[async_trait]
impl VectorIndex for QdrantIndex {
    fn capabilities(&self) -> VectorCapabilities {
        VectorCapabilities {
            backend: "qdrant".into(),
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
        check_name(collection)?;
        match self.collection_dims(collection).await? {
            Some(existing) if existing != dimensions => {
                return Err(StoreError::DimensionMismatch {
                    expected: existing,
                    got: dimensions,
                });
            }
            Some(_) => {}
            None => {
                self.ok(
                    reqwest::Method::PUT,
                    &format!("/collections/{collection}"),
                    Some(json!({"vectors": {"size": dimensions, "distance": "Cosine"}})),
                )
                .await?;
            }
        }
        if self.indexed_lock().contains(collection) {
            return Ok(());
        }
        // Creating an existing payload index is a no-op, so collections made
        // before tags existed gain the index too.
        for field in ["ns", "tags"] {
            self.ok(
                reqwest::Method::PUT,
                &format!("/collections/{collection}/index?wait=true"),
                Some(json!({"field_name": field, "field_schema": "keyword"})),
            )
            .await?;
        }
        self.indexed_lock().insert(collection.to_owned());
        Ok(())
    }

    async fn upsert(
        &self,
        collection: &str,
        namespace: &str,
        points: &[VectorPoint],
    ) -> Result<(), StoreError> {
        check_name(collection)?;
        let Some(first) = points.first() else {
            return Ok(());
        };
        self.require_dims(collection, first.vector.len()).await?;
        if let Some(p) = points.iter().find(|p| p.vector.len() != first.vector.len()) {
            return Err(StoreError::DimensionMismatch {
                expected: first.vector.len(),
                got: p.vector.len(),
            });
        }
        for batch in points.chunks(256) {
            let body: Vec<Value> = batch
                .iter()
                .map(|p| {
                    json!({
                        "id": point_id(namespace, &p.id),
                        "vector": p.vector,
                        "payload": {"ns": namespace, "key": p.id, "tags": p.tags},
                    })
                })
                .collect();
            self.ok(
                reqwest::Method::PUT,
                &format!("/collections/{collection}/points?wait=true"),
                Some(json!({"points": body})),
            )
            .await
            .inspect_err(|_| self.forget(collection))?;
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
        self.query(collection, query, k, namespace_filter(namespace))
            .await
    }

    async fn search_tagged(
        &self,
        collection: &str,
        namespace: &str,
        query: &[f32],
        k: usize,
        tags: &[String],
    ) -> Result<Vec<ScoredId>, StoreError> {
        if tags.is_empty() {
            return Ok(Vec::new());
        }
        self.query(collection, query, k, tagged_filter(namespace, tags))
            .await
    }

    async fn delete_namespace(&self, collection: &str, namespace: &str) -> Result<(), StoreError> {
        check_name(collection)?;
        if self.collection_dims(collection).await?.is_none() {
            return Ok(());
        }
        self.ok(
            reqwest::Method::POST,
            &format!("/collections/{collection}/points/delete?wait=true"),
            Some(json!({"filter": namespace_filter(namespace)})),
        )
        .await?;
        Ok(())
    }

    async fn delete_points(
        &self,
        collection: &str,
        namespace: &str,
        ids: &[String],
    ) -> Result<(), StoreError> {
        check_name(collection)?;
        if ids.is_empty() || self.collection_dims(collection).await?.is_none() {
            return Ok(());
        }
        for batch in ids.chunks(256) {
            let points: Vec<String> = batch.iter().map(|id| point_id(namespace, id)).collect();
            self.ok(
                reqwest::Method::POST,
                &format!("/collections/{collection}/points/delete?wait=true"),
                Some(json!({"points": points})),
            )
            .await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_ids_are_deterministic_uuids_scoped_by_namespace() {
        let a = point_id("conv-1", "turn-1");
        assert_eq!(a, point_id("conv-1", "turn-1"));
        assert_ne!(a, point_id("conv-2", "turn-1"));
        // Separator prevents ("ab","c") colliding with ("a","bc").
        assert_ne!(point_id("ab", "c"), point_id("a", "bc"));
        assert_eq!(a.len(), 36);
        assert_eq!(a.matches('-').count(), 4);
    }
}
