//! Local ONNX cross-encoder reranking through `fastembed`, so Knowledge
//! search can reorder its fused candidates without a paid API.
//!
//! Model files are fetched and verified exactly like the local embedding
//! model ([`crate::local_embed`]): a pinned commit, pinned SHA-256 digests
//! and a disk cache. Nothing is downloaded unless a reranker is loaded.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fastembed::{
    AutoDevicePolicy, RerankInitOptionsUserDefined, TextRerank, TokenizerFiles,
    UserDefinedRerankingModel,
};

use crate::ProviderDescriptor;
use crate::chat::ProviderError;
use crate::local_embed::{LocalDevice, config_err, default_parallelism, ensure_pinned};
use crate::rerank::Reranker;

#[derive(Clone, Debug)]
pub struct LocalRerankConfig {
    /// Hugging Face repository, e.g. `cross-encoder/ms-marco-MiniLM-L6-v2`.
    pub repo: String,
    /// Full commit hash; branch names are rejected.
    pub revision: String,
    /// `sha256:<hex>` for each entry of [`crate::local_embed::MODEL_FILES`].
    pub sha256: Vec<String>,
    /// Query and passage tokens per pair; longer pairs are truncated.
    pub max_length: usize,
    pub cache_dir: PathBuf,
    pub device: LocalDevice,
    pub intra_threads: Option<usize>,
    pub sessions: Option<usize>,
}

/// The default local cross-encoder: 22M parameters, trained on MS MARCO
/// passage ranking.
pub const MS_MARCO_MINILM_L6_V2_REPO: &str = "cross-encoder/ms-marco-MiniLM-L6-v2";
pub const MS_MARCO_MINILM_L6_V2_REVISION: &str = "233902d25c440f23af6f7d6e94d2946bac0bee0a";
const MS_MARCO_MINILM_L6_V2_SHA256: [&str; 5] = [
    "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a",
    "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc",
    "3c3507f36dff57bce437223db3b3081d1e2b52ec3e56ee55438193ecb2c94dd6",
    "a5c2e5a7b1a29a0702cd28c08a399b5ecc110c263009d17f7e3b415f25905fd8",
];

impl LocalRerankConfig {
    /// `cross-encoder/ms-marco-MiniLM-L6-v2` at a pinned commit with pinned
    /// file digests, cached under `cache_dir`.
    pub fn ms_marco_minilm_l6_v2(cache_dir: PathBuf) -> Self {
        Self {
            repo: MS_MARCO_MINILM_L6_V2_REPO.into(),
            revision: MS_MARCO_MINILM_L6_V2_REVISION.into(),
            sha256: MS_MARCO_MINILM_L6_V2_SHA256
                .iter()
                .map(|h| format!("sha256:{h}"))
                .collect(),
            max_length: 512,
            cache_dir,
            device: LocalDevice::Cpu,
            intra_threads: None,
            sessions: None,
        }
    }

    /// Apply `KEEP_ONNX_DEVICE`, `KEEP_ONNX_THREADS` and
    /// `KEEP_ONNX_SESSIONS`, as for local embeddings. None of them changes
    /// which model runs.
    pub fn with_env_overrides(mut self) -> Result<Self, ProviderError> {
        let e = crate::local_embed::LocalEmbedConfig {
            device: self.device,
            intra_threads: self.intra_threads,
            sessions: self.sessions,
            ..crate::local_embed::LocalEmbedConfig::bge_small_en_v1_5(self.cache_dir.clone())
        }
        .with_env_overrides()?;
        self.device = e.device;
        self.intra_threads = e.intra_threads;
        self.sessions = e.sessions;
        Ok(self)
    }

    pub fn parallelism(&self) -> (usize, usize) {
        let (sessions, threads) = default_parallelism();
        (
            self.sessions.unwrap_or(sessions),
            self.intra_threads.unwrap_or(threads),
        )
    }

    pub fn model_dir(&self) -> PathBuf {
        self.cache_dir
            .join(self.repo.replace('/', "--"))
            .join(&self.revision)
    }
}

/// Scores each query and passage pair in its own model run, so a score
/// depends only on the pair and the model, never on which other pairs
/// shared the call. Parallelism comes from several sessions taking pairs
/// from a shared queue.
pub struct LocalReranker {
    sessions: Vec<Arc<Mutex<TextRerank>>>,
    next_session: AtomicUsize,
    descriptor: ProviderDescriptor,
}

impl LocalReranker {
    /// Fetch (if needed), verify and load the pinned model.
    pub async fn load(config: LocalRerankConfig) -> Result<Self, ProviderError> {
        let mut files = ensure_pinned(
            &config.repo,
            &config.revision,
            &config.sha256,
            &config.model_dir(),
        )
        .await?;
        let tokenizer_config_file = files.pop().expect("five files");
        let special_tokens_map_file = files.pop().expect("five files");
        let config_file = files.pop().expect("five files");
        let tokenizer_file = files.pop().expect("five files");
        let onnx = files.pop().expect("five files");
        let model = UserDefinedRerankingModel::new(
            onnx,
            TokenizerFiles {
                tokenizer_file,
                config_file,
                special_tokens_map_file,
                tokenizer_config_file,
            },
        );
        let (sessions, threads) = config.parallelism();
        let mut options = RerankInitOptionsUserDefined::new()
            .with_max_length(config.max_length)
            .with_intra_threads(threads);
        if config.device == LocalDevice::Cpu {
            options = options.with_auto_device(AutoDevicePolicy::PreferCPU);
        }
        let loaded = tokio::task::spawn_blocking(move || {
            (0..sessions)
                .map(|_| {
                    TextRerank::try_new_from_user_defined(model.clone(), options.clone())
                        .map(|m| Arc::new(Mutex::new(m)))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .await
        .map_err(|e| config_err(format!("loading local reranker panicked: {e}")))?
        .map_err(|e| config_err(format!("cannot load local reranker: {e}")))?;
        Ok(Self {
            sessions: loaded,
            next_session: AtomicUsize::new(0),
            descriptor: ProviderDescriptor {
                provider: "local-onnx".into(),
                model: config.repo.clone(),
                revision: config.revision.clone(),
                mock: false,
            },
        })
    }
}

#[async_trait]
impl Reranker for LocalReranker {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    async fn score(&self, query: &str, texts: &[String]) -> Result<Vec<f32>, ProviderError> {
        let _t = crate::timing::span("onnx.rerank").items(texts.len());
        let texts: Arc<[String]> = texts.into();
        let query: Arc<str> = query.into();
        let next = Arc::new(AtomicUsize::new(0));
        let workers = self.sessions.len().min(texts.len());
        let first = self.next_session.fetch_add(1, Ordering::Relaxed);
        let mut tasks = Vec::with_capacity(workers);
        for w in 0..workers {
            let session = self.sessions[(first + w) % self.sessions.len()].clone();
            let (texts, next, query) = (texts.clone(), next.clone(), query.clone());
            tasks.push(tokio::task::spawn_blocking(move || {
                let mut m = session.lock().expect("reranker lock");
                let mut out = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= texts.len() {
                        return Ok(out);
                    }
                    let r = m.rerank(query.as_ref(), [texts[i].as_str()], false, Some(1))?;
                    out.push((i, r.first().map_or(f32::NEG_INFINITY, |r| r.score)));
                }
            }));
        }
        let mut scores = vec![0f32; texts.len()];
        for task in tasks {
            let done: Result<Vec<(usize, f32)>, fastembed::Error> =
                task.await.map_err(|e| ProviderError::Transport {
                    message: format!("rerank task panicked: {e}"),
                })?;
            for (i, s) in done.map_err(|e| ProviderError::Malformed {
                message: format!("local reranking failed: {e}"),
            })? {
                scores[i] = s;
            }
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_embed::MODEL_FILES;

    #[test]
    fn the_preset_pins_every_file() {
        let c = LocalRerankConfig::ms_marco_minilm_l6_v2(PathBuf::from("cache"));
        assert_eq!(c.sha256.len(), MODEL_FILES.len());
        assert!(c.sha256.iter().all(|h| h.len() == "sha256:".len() + 64));
        assert_eq!(
            c.model_dir(),
            PathBuf::from("cache")
                .join("cross-encoder--ms-marco-MiniLM-L6-v2")
                .join(MS_MARCO_MINILM_L6_V2_REVISION)
        );
    }

    #[tokio::test]
    async fn rejects_branch_revisions_before_any_download() {
        let dir = tempfile::tempdir().unwrap();
        let c = LocalRerankConfig {
            revision: "main".into(),
            ..LocalRerankConfig::ms_marco_minilm_l6_v2(dir.path().to_owned())
        };
        assert!(matches!(
            LocalReranker::load(c).await,
            Err(ProviderError::Config { .. })
        ));
    }

    /// Run with `KEEP_MODEL_CACHE=<dir holding the model> cargo test -p
    /// keep-providers --features local-embeddings`; never downloads.
    #[tokio::test]
    async fn scores_relevant_passages_higher_and_independently_of_grouping() {
        let Some(cache) = std::env::var_os("KEEP_MODEL_CACHE").map(PathBuf::from) else {
            eprintln!("skipped: KEEP_MODEL_CACHE is not set");
            return;
        };
        let c = LocalRerankConfig::ms_marco_minilm_l6_v2(cache);
        if !MODEL_FILES.iter().all(|f| c.model_dir().join(f).exists()) {
            eprintln!("skipped: KEEP_MODEL_CACHE does not hold the pinned reranker");
            return;
        }
        let r = LocalReranker::load(c).await.unwrap();
        let texts: Vec<String> = vec![
            "Lunch is at noon on Fridays.".into(),
            "Mira moved to Lisbon in March to start a new job.".into(),
            "The weather was rainy all week.".into(),
        ];
        let all = r.score("Where did Mira move?", &texts).await.unwrap();
        assert!(all[1] > all[0] && all[1] > all[2], "{all:?}");
        let one = r.score("Where did Mira move?", &texts[1..2]).await.unwrap();
        assert_eq!(one[0], all[1]);
    }
}
