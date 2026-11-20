//! Storage contracts.
//!
//! Keep keeps all authoritative state in a Postgres ledger. Vector and lexical
//! indexes are derived, rebuildable projections that never make authority or
//! permission decisions. This crate defines the index traits with explicit
//! capability declarations plus in-memory reference implementations used by
//! tests and offline dry runs. Backend adapters live in `keep-postgres`
//! (pgvector) and `keep-qdrant`.

pub mod lexical;
pub mod memory_vector;
pub mod vector;

pub use lexical::{Bm25Index, Bm25Params, LexicalIndex};
pub use memory_vector::InMemoryVectorIndex;
pub use vector::{Distance, ScoredId, StoreError, VectorCapabilities, VectorIndex, VectorPoint};

/// Sort hits by descending score, breaking ties by id so results are
/// deterministic across backends.
pub fn rank(mut hits: Vec<ScoredId>, k: usize) -> Vec<ScoredId> {
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    hits.truncate(k);
    hits
}
