//! Postgres adapter.
//!
//! In both storage profiles Postgres is the authoritative ledger. In the
//! `postgres-vector` profile it also stores the derived vector projection with
//! pgvector. Only that projection exists so far: the research baselines need
//! dense retrieval, while the ledger tables arrive with the engine prototypes.
//!
//! [`connect`] opens every Keep Postgres connection, with TLS for remote
//! databases; see [`connect`](mod@connect) for the policy.

use async_trait::async_trait;
use keep_store::{Distance, ScoredId, StoreError, VectorCapabilities, VectorIndex, VectorPoint};
use pgvector::Vector;
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio_postgres::Client;

pub mod connect;
pub use connect::{ConnectError, connect};

pub struct PgVectorIndex {
    url: String,
    /// Shared by concurrent calls; replaced when the server or network
    /// closed it, so the index recovers after a database outage.
    client: RwLock<Arc<Client>>,
}

fn backend(e: impl std::fmt::Display) -> StoreError {
    StoreError::Backend(e.to_string())
}

/// Collection names become table-name suffixes, so they are restricted.
fn table_for(collection: &str) -> Result<String, StoreError> {
    let ok = !collection.is_empty()
        && collection.len() <= 48
        && collection
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
    if ok {
        Ok(format!("keep_vectors_{collection}"))
    } else {
        Err(StoreError::Capability(format!(
            "collection name {collection:?} must be 1-48 chars of [a-z0-9_]"
        )))
    }
}

/// Run idempotent `IF NOT EXISTS` DDL. Postgres does not make it safe
/// against a concurrent run of the same statement on a fresh database: the
/// loser gets a unique violation, which means the object now exists.
async fn ddl(client: &Client, sql: &str) -> Result<(), tokio_postgres::Error> {
    match client.batch_execute(sql).await {
        Err(e) if e.code() == Some(&tokio_postgres::error::SqlState::UNIQUE_VIOLATION) => Ok(()),
        other => other,
    }
}

/// Install pgvector if it is missing, tolerating a concurrent install.
pub async fn ensure_vector_extension(client: &Client) -> Result<(), tokio_postgres::Error> {
    ddl(client, "CREATE EXTENSION IF NOT EXISTS vector").await
}

async fn open(url: &str) -> Result<Client, StoreError> {
    let client = connect(url).await.map_err(backend)?;
    ensure_vector_extension(&client).await.map_err(backend)?;
    ddl(
        &client,
        "CREATE TABLE IF NOT EXISTS keep_vector_collections (
             name text PRIMARY KEY,
             dimensions integer NOT NULL CHECK (dimensions > 0)
         )",
    )
    .await
    .map_err(backend)?;
    Ok(client)
}

impl PgVectorIndex {
    pub async fn connect(url: &str) -> Result<Self, StoreError> {
        Ok(Self {
            url: url.into(),
            client: RwLock::new(Arc::new(open(url).await?)),
        })
    }

    /// The live connection, reopened first if it was closed.
    async fn client(&self) -> Result<Arc<Client>, StoreError> {
        let current = self.client.read().await.clone();
        if !current.is_closed() {
            return Ok(current);
        }
        let mut slot = self.client.write().await;
        if slot.is_closed() {
            *slot = Arc::new(open(&self.url).await?);
        }
        Ok(slot.clone())
    }

    async fn dimensions(&self, collection: &str) -> Result<usize, StoreError> {
        let row = self
            .client()
            .await?
            .query_opt(
                "SELECT dimensions FROM keep_vector_collections WHERE name = $1",
                &[&collection],
            )
            .await
            .map_err(backend)?
            .ok_or_else(|| StoreError::UnknownCollection(collection.into()))?;
        Ok(row.get::<_, i32>(0) as usize)
    }
}

#[async_trait]
impl VectorIndex for PgVectorIndex {
    fn capabilities(&self) -> VectorCapabilities {
        VectorCapabilities {
            backend: "postgres-pgvector".into(),
            distance: Distance::Cosine,
            max_dimensions: 16_000,
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
        let table = table_for(collection)?;
        let dims = i32::try_from(dimensions).map_err(backend)?;
        self.client()
            .await?
            .execute(
                "INSERT INTO keep_vector_collections (name, dimensions) VALUES ($1, $2)
                 ON CONFLICT (name) DO NOTHING",
                &[&collection, &dims],
            )
            .await
            .map_err(backend)?;
        let existing = self.dimensions(collection).await?;
        if existing != dimensions {
            return Err(StoreError::DimensionMismatch {
                expected: existing,
                got: dimensions,
            });
        }
        ddl(
            &*self.client().await?,
            &format!(
                "CREATE TABLE IF NOT EXISTS {table} (
                     namespace text NOT NULL,
                     id text NOT NULL,
                     embedding vector({dimensions}) NOT NULL,
                     PRIMARY KEY (namespace, id)
                 )"
            ),
        )
        .await
        .map_err(backend)
    }

    async fn upsert(
        &self,
        collection: &str,
        namespace: &str,
        points: &[VectorPoint],
    ) -> Result<(), StoreError> {
        let table = table_for(collection)?;
        let dims = self.dimensions(collection).await?;
        if let Some(p) = points.iter().find(|p| p.vector.len() != dims) {
            return Err(StoreError::DimensionMismatch {
                expected: dims,
                got: p.vector.len(),
            });
        }
        let client = self.client().await?;
        let stmt = client
            .prepare(&format!(
                "INSERT INTO {table} (namespace, id, embedding) VALUES ($1, $2, $3)
                 ON CONFLICT (namespace, id) DO UPDATE SET embedding = EXCLUDED.embedding"
            ))
            .await
            .map_err(backend)?;
        for p in points {
            let v = Vector::from(p.vector.clone());
            client
                .execute(&stmt, &[&namespace, &p.id, &v])
                .await
                .map_err(backend)?;
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
        let table = table_for(collection)?;
        let dims = self.dimensions(collection).await?;
        if query.len() != dims {
            return Err(StoreError::DimensionMismatch {
                expected: dims,
                got: query.len(),
            });
        }
        let q = Vector::from(query.to_vec());
        let limit = i64::try_from(k).map_err(backend)?;
        let rows = self
            .client()
            .await?
            .query(
                &format!(
                    "SELECT id, (1 - (embedding <=> $1))::real AS score FROM {table}
                     WHERE namespace = $2
                     ORDER BY embedding <=> $1, id
                     LIMIT $3"
                ),
                &[&q, &namespace, &limit],
            )
            .await
            .map_err(backend)?;
        let hits = rows
            .iter()
            .map(|r| ScoredId {
                id: r.get(0),
                score: r.get::<_, Option<f32>>(1).unwrap_or(0.0),
            })
            .collect();
        Ok(keep_store::rank(hits, k))
    }

    async fn delete_namespace(&self, collection: &str, namespace: &str) -> Result<(), StoreError> {
        let table = table_for(collection)?;
        self.client()
            .await?
            .execute(
                &format!("DELETE FROM {table} WHERE namespace = $1"),
                &[&namespace],
            )
            .await
            .map_err(backend)?;
        Ok(())
    }

    async fn delete_points(
        &self,
        collection: &str,
        namespace: &str,
        ids: &[String],
    ) -> Result<(), StoreError> {
        let table = table_for(collection)?;
        if ids.is_empty() || self.dimensions(collection).await.is_err() {
            return Ok(());
        }
        self.client()
            .await?
            .execute(
                &format!("DELETE FROM {table} WHERE namespace = $1 AND id = ANY($2)"),
                &[&namespace, &ids],
            )
            .await
            .map_err(backend)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_names_cannot_inject_sql() {
        assert_eq!(table_for("bench_dev").unwrap(), "keep_vectors_bench_dev");
        assert!(table_for("x; DROP TABLE y").is_err());
        assert!(table_for("Upper").is_err());
        assert!(table_for("").is_err());
    }
}
