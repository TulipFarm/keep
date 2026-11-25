//! Choosing and loading Knowledge's embedder.

use std::path::Path;
use std::sync::Arc;

use keep_providers::{Embedder, HashEmbedder};

use crate::config::{EmbedderKind, ServiceArgs};
use crate::settings::EmbedderRecord;

/// Dimensions of the hash embedder, as earlier servers used.
pub const HASH_DIMENSIONS: usize = 256;

/// What `kind` would record, without loading a model.
pub fn record(kind: EmbedderKind) -> anyhow::Result<EmbedderRecord> {
    match kind {
        EmbedderKind::Hash => Ok(EmbedderRecord::of(&HashEmbedder::new(HASH_DIMENSIONS))),
        EmbedderKind::Local => local_record(),
    }
}

/// Build the embedder. `local` downloads its pinned model into the model
/// cache on first use; nothing else touches the network.
pub async fn load(args: &ServiceArgs) -> anyhow::Result<Arc<dyn Embedder>> {
    match args.embedder_kind() {
        EmbedderKind::Hash => Ok(Arc::new(HashEmbedder::new(HASH_DIMENSIONS))),
        EmbedderKind::Local => load_local(&args.model_cache()?).await,
    }
}

#[cfg(feature = "local-embeddings")]
fn local_record() -> anyhow::Result<EmbedderRecord> {
    use keep_providers::local_embed::*;
    Ok(EmbedderRecord {
        provider: "local-onnx".into(),
        model: BGE_SMALL_EN_V1_5_REPO.into(),
        revision: BGE_SMALL_EN_V1_5_REVISION.into(),
        dimensions: BGE_SMALL_EN_V1_5_DIMENSIONS,
    })
}

/// Bounded in-memory embedding cache size for `keep serve` (~30 MiB for 384-d vectors).
#[cfg(feature = "local-embeddings")]
const LOCAL_EMBED_CACHE_MAX_ENTRIES: usize = 16_384;

#[cfg(feature = "local-embeddings")]
async fn load_local(model_cache: &Path) -> anyhow::Result<Arc<dyn Embedder>> {
    use keep_providers::CachedEmbedder;
    use keep_providers::local_embed::{LocalEmbedConfig, LocalEmbedder};
    let config =
        LocalEmbedConfig::bge_small_en_v1_5(model_cache.to_owned()).with_env_overrides()?;
    if !config.model_dir().exists() {
        eprintln!(
            "keep: downloading the local embedding model (about 130 MB) into {}",
            model_cache.display()
        );
    }
    let cache_extra = format!(
        "pool={:?};max_len={};device={:?};sha256={}",
        config.pooling,
        config.max_length,
        config.device,
        config.sha256.join(",")
    );
    let e = LocalEmbedder::load(config)
        .await
        .map_err(|e| anyhow::anyhow!("loading the local embedding model: {e}"))?;
    let cached = CachedEmbedder::new(Arc::new(e), &cache_extra, None)
        .map_err(|e| anyhow::anyhow!("initializing local embedding cache: {e}"))?
        .with_max_entries(LOCAL_EMBED_CACHE_MAX_ENTRIES);
    Ok(Arc::new(cached))
}

/// Download (or verify) the pinned local model under `model_cache`, and
/// return the directory holding it.
#[cfg(feature = "local-embeddings")]
pub async fn fetch_local(model_cache: &Path) -> anyhow::Result<std::path::PathBuf> {
    use keep_providers::local_embed::{LocalEmbedConfig, ensure_files};
    let config =
        LocalEmbedConfig::bge_small_en_v1_5(model_cache.to_owned()).with_env_overrides()?;
    ensure_files(&config)
        .await
        .map_err(|e| anyhow::anyhow!("fetching the local embedding model: {e}"))?;
    Ok(config.model_dir())
}

#[cfg(not(feature = "local-embeddings"))]
pub async fn fetch_local(_: &Path) -> anyhow::Result<std::path::PathBuf> {
    Err(not_compiled())
}

#[cfg(not(feature = "local-embeddings"))]
fn local_record() -> anyhow::Result<EmbedderRecord> {
    Err(not_compiled())
}

#[cfg(not(feature = "local-embeddings"))]
async fn load_local(_: &Path) -> anyhow::Result<Arc<dyn Embedder>> {
    Err(not_compiled())
}

#[cfg(not(feature = "local-embeddings"))]
fn not_compiled() -> anyhow::Error {
    anyhow::anyhow!(
        "--embedder local needs a keep binary built with the local-embeddings feature \
         (cargo build -p keep-server --features local-embeddings); use --embedder hash otherwise"
    )
}
