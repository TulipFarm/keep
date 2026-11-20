//! Provider contracts for chat models, embedders and tokenizers.
//!
//! Every path that can reach a remote model needs a [`dispatch::DispatchPermit`].
//! A permit is only granted when the run manifest *and* the operator both opt
//! in, and every remote call is charged against a [`budget::BudgetLedger`]
//! before it is sent. The [`mock`] module provides deterministic offline
//! stand-ins for tests and dry runs; their output is never a benchmark result.

pub mod budget;
pub mod chat;
pub mod copilot;
pub mod credits;
pub mod dispatch;
pub mod embed;
pub mod embed_cache;
#[cfg(feature = "local-embeddings")]
pub mod local_embed;
#[cfg(feature = "local-embeddings")]
pub mod local_rerank;
pub mod mock;
pub mod openai;
pub mod ratelimit;
pub mod rerank;
pub mod timing;
pub mod tokenizer;

pub use budget::{BudgetCaps, BudgetExceeded, BudgetLedger, Price};
pub use chat::{
    Billing, ChatModel, ChatRequest, ChatResponse, Message, MessageRole, ProviderError, Usage,
};
pub use credits::{CreditLedger, CreditUsage};
pub use dispatch::{DispatchPermit, DispatchPolicy, DispatchRefused};
pub use embed::{EmbedResponse, Embedder, HashEmbedder};
pub use embed_cache::CachedEmbedder;
pub use rerank::{MemoReranker, OverlapReranker, Reranker};
pub use tokenizer::{
    ApproxTokenizer, TiktokenEncoding, TiktokenTokenizer, Tokenizer, tokenizer_for,
};

/// Identifies the implementation behind a provider so artifacts can say
/// exactly what produced a number.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProviderDescriptor {
    pub provider: String,
    pub model: String,
    pub revision: String,
    /// True for offline stand-ins. Results from mock providers are plumbing
    /// checks, not measurements.
    pub mock: bool,
}
