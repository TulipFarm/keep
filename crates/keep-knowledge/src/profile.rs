//! Ready-made storage profiles.
//!
//! - `in-memory`: everything in process; for tests and dry runs.
//! - `postgres-vector`: Postgres ledger plus a pgvector projection in the
//!   same database.
//! - `postgres-qdrant`: Postgres ledger plus a Qdrant projection.
//!
//! In every profile the ledger is the only authority. Vector points can be
//! dropped and rebuilt with [`Knowledge::rebuild_projection`].

use std::sync::Arc;

use keep_postgres::PgVectorIndex;
use keep_qdrant::QdrantIndex;
use keep_store::InMemoryVectorIndex;

use crate::engine::Knowledge;
use crate::ledger::Result;
use crate::memory_ledger::MemoryLedger;
use crate::pg_ledger::PgLedger;

pub fn in_memory() -> Knowledge {
    Knowledge::new(
        Arc::new(MemoryLedger::new()),
        Some((Arc::new(InMemoryVectorIndex::new()), "memory".into())),
    )
}

pub async fn postgres_vector(postgres_url: &str) -> Result<Knowledge> {
    let ledger = PgLedger::connect(postgres_url).await?;
    let index = PgVectorIndex::connect(postgres_url).await?;
    Ok(Knowledge::new(
        Arc::new(ledger),
        Some((Arc::new(index), "pgvector".into())),
    ))
}

pub async fn postgres_qdrant(
    postgres_url: &str,
    qdrant_url: &str,
    qdrant_api_key: Option<String>,
) -> Result<Knowledge> {
    postgres_qdrant_with_ca(postgres_url, qdrant_url, qdrant_api_key, None).await
}

/// [`postgres_qdrant`], also trusting the PEM certificates in `qdrant_ca`
/// for an `https://` Qdrant behind a private CA.
pub async fn postgres_qdrant_with_ca(
    postgres_url: &str,
    qdrant_url: &str,
    qdrant_api_key: Option<String>,
    qdrant_ca: Option<&[u8]>,
) -> Result<Knowledge> {
    let index = QdrantIndex::connect(qdrant_url, qdrant_api_key, qdrant_ca)?;
    let ledger = PgLedger::connect(postgres_url).await?;
    Ok(Knowledge::new(
        Arc::new(ledger),
        Some((Arc::new(index), "qdrant".into())),
    ))
}
