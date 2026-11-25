//! Local ONNX sentence embeddings through `fastembed`, so dense retrieval and
//! the Knowledge arms can run without a paid embedding API.
//!
//! Model files are fetched once from the Hugging Face Hub at a pinned commit,
//! checked against pinned SHA-256 digests, and cached on disk. Nothing is
//! downloaded unless an embedder is constructed, and only builds with the
//! `local-embeddings` feature contain this module.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use fastembed::{
    AutoDevicePolicy, InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use sha2::{Digest, Sha256};

use crate::ProviderDescriptor;
use crate::chat::{ProviderError, Usage};
use crate::embed::{EmbedResponse, Embedder};

/// The files a sentence-transformer export needs, in the order fastembed
/// takes them.
pub const MODEL_FILES: [&str; 5] = [
    "onnx/model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalPooling {
    Cls,
    Mean,
}

/// Where ONNX Runtime runs the model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LocalDevice {
    /// The CPU execution provider only. The default: it is the fastest
    /// measured choice for small models on Apple silicon, and it computes in
    /// full `f32`.
    #[default]
    Cpu,
    /// ONNX Runtime's automatic choice, which registers CoreML on macOS.
    Auto,
}

impl std::str::FromStr for LocalDevice {
    type Err = ProviderError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "auto" => Ok(Self::Auto),
            other => Err(config_err(format!(
                "unknown ONNX device {other:?}; use \"cpu\" or \"auto\""
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LocalEmbedConfig {
    /// Hugging Face repository, e.g. `BAAI/bge-small-en-v1.5`.
    pub repo: String,
    /// Full commit hash; branch names are rejected.
    pub revision: String,
    /// `sha256:<hex>` for each entry of [`MODEL_FILES`], in order.
    pub sha256: Vec<String>,
    pub dimensions: usize,
    pub pooling: LocalPooling,
    pub max_length: usize,
    pub cache_dir: PathBuf,
    pub device: LocalDevice,
    /// ONNX Runtime intra-op threads per session; `None` picks
    /// [`default_parallelism`].
    pub intra_threads: Option<usize>,
    /// Model sessions that embed texts in parallel; `None` picks
    /// [`default_parallelism`].
    pub sessions: Option<usize>,
}

/// Sessions and threads per session when none are configured: up to three
/// threads per session (clamped to the host's available cores so small
/// machines are not oversubscribed) and one session per group of threads.
pub fn default_parallelism() -> (usize, usize) {
    let cores = std::thread::available_parallelism().map_or(3, |n| n.get());
    parallelism_for_cores(cores)
}

fn parallelism_for_cores(cores: usize) -> (usize, usize) {
    let cores = cores.max(1);
    let threads = cores.min(3);
    ((cores / threads).max(1), threads)
}

/// Hugging Face repository of the default local model.
pub const BGE_SMALL_EN_V1_5_REPO: &str = "BAAI/bge-small-en-v1.5";
/// The pinned commit of [`BGE_SMALL_EN_V1_5_REPO`].
pub const BGE_SMALL_EN_V1_5_REVISION: &str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a";
/// Its output size.
pub const BGE_SMALL_EN_V1_5_DIMENSIONS: usize = 384;
const BGE_SMALL_EN_V1_5_SHA256: [&str; 5] = [
    "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
    "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
];

impl LocalEmbedConfig {
    /// `BAAI/bge-small-en-v1.5` (384 dimensions, CLS pooling) at a pinned
    /// commit with pinned file digests, cached under `cache_dir`.
    pub fn bge_small_en_v1_5(cache_dir: PathBuf) -> Self {
        Self {
            repo: BGE_SMALL_EN_V1_5_REPO.into(),
            revision: BGE_SMALL_EN_V1_5_REVISION.into(),
            sha256: BGE_SMALL_EN_V1_5_SHA256
                .iter()
                .map(|h| format!("sha256:{h}"))
                .collect(),
            dimensions: BGE_SMALL_EN_V1_5_DIMENSIONS,
            pooling: LocalPooling::Cls,
            max_length: 512,
            cache_dir,
            device: LocalDevice::Cpu,
            intra_threads: None,
            sessions: None,
        }
    }

    /// Apply `KEEP_ONNX_DEVICE` (`cpu` or `auto`), `KEEP_ONNX_THREADS` and
    /// `KEEP_ONNX_SESSIONS` when they are set and non-empty. None of them
    /// changes which model runs.
    pub fn with_env_overrides(mut self) -> Result<Self, ProviderError> {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        if let Some(d) = var("KEEP_ONNX_DEVICE") {
            self.device = d.parse()?;
        }
        let count = |name: &str| -> Result<Option<usize>, ProviderError> {
            var(name)
                .map(|t| {
                    t.parse().ok().filter(|n| *n > 0).ok_or_else(|| {
                        config_err(format!("{name} must be a positive integer, got {t:?}"))
                    })
                })
                .transpose()
        };
        if let Some(n) = count("KEEP_ONNX_THREADS")? {
            self.intra_threads = Some(n);
        }
        if let Some(n) = count("KEEP_ONNX_SESSIONS")? {
            self.sessions = Some(n);
        }
        Ok(self)
    }

    /// `(sessions, threads per session)` after defaults.
    pub fn parallelism(&self) -> (usize, usize) {
        let (sessions, threads) = default_parallelism();
        (
            self.sessions.unwrap_or(sessions),
            self.intra_threads.unwrap_or(threads),
        )
    }

    /// Where the pinned files live: `<cache>/<repo>/<revision>/<file>`.
    pub fn model_dir(&self) -> PathBuf {
        self.cache_dir
            .join(self.repo.replace('/', "--"))
            .join(&self.revision)
    }
}

fn sha256_label(bytes: &[u8]) -> String {
    format!("sha256:{}", hex_lower(&Sha256::digest(bytes)))
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn config_err(message: String) -> ProviderError {
    ProviderError::Config { message }
}

fn read_verified(path: &Path, want: &str) -> Result<Vec<u8>, ProviderError> {
    let bytes = std::fs::read(path)
        .map_err(|e| config_err(format!("cannot read {}: {e}", path.display())))?;
    let got = sha256_label(&bytes);
    if got != want {
        return Err(config_err(format!(
            "{} has {got}, manifest pins {want}",
            path.display()
        )));
    }
    Ok(bytes)
}

/// Download any missing pinned files into the cache and verify every digest.
pub async fn ensure_files(config: &LocalEmbedConfig) -> Result<Vec<Vec<u8>>, ProviderError> {
    ensure_pinned(
        &config.repo,
        &config.revision,
        &config.sha256,
        &config.model_dir(),
    )
    .await
}

/// Download any missing [`MODEL_FILES`] of `repo` at `revision` into `dir`
/// and verify each against its pinned `sha256:<hex>` digest.
pub(crate) async fn ensure_pinned(
    repo: &str,
    revision: &str,
    sha256: &[String],
    dir: &Path,
) -> Result<Vec<Vec<u8>>, ProviderError> {
    if revision.len() != 40 || !revision.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(config_err(format!(
            "local model revision must be a full commit hash, got {revision:?}"
        )));
    }
    if sha256.len() != MODEL_FILES.len() {
        return Err(config_err(format!(
            "local models need {} pinned file digests ({})",
            MODEL_FILES.len(),
            MODEL_FILES.join(", ")
        )));
    }
    let client = reqwest::Client::new();
    let mut out = Vec::new();
    for (file, want) in MODEL_FILES.iter().zip(sha256) {
        let path = dir.join(file);
        if !path.exists() {
            let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{file}");
            let resp = client
                .get(&url)
                .send()
                .await
                .and_then(|r| r.error_for_status())
                .map_err(|e| ProviderError::Transport {
                    message: format!("downloading {url}: {e}"),
                })?;
            let bytes = resp.bytes().await.map_err(|e| ProviderError::Transport {
                message: format!("downloading {url}: {e}"),
            })?;
            let got = sha256_label(&bytes);
            if &got != want {
                return Err(config_err(format!("{url} has {got}, manifest pins {want}")));
            }
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| config_err(format!("cannot create {}: {e}", parent.display())))?;
            }
            let tmp = path.with_extension("partial");
            std::fs::write(&tmp, &bytes)
                .and_then(|_| std::fs::rename(&tmp, &path))
                .map_err(|e| config_err(format!("cannot write {}: {e}", path.display())))?;
        }
        out.push(read_verified(&path, want)?);
    }
    Ok(out)
}

/// Embeds each text in its own model run, so a text's vector depends only
/// on the text and the model, never on which other texts shared the call
/// (padding a batch to its longest text changes the floating-point
/// reductions). Parallelism comes from several sessions instead, each
/// taking the next text from a shared queue.
pub struct LocalEmbedder {
    sessions: Vec<Arc<Mutex<TextEmbedding>>>,
    next_session: AtomicUsize,
    descriptor: ProviderDescriptor,
    dimensions: usize,
}

impl LocalEmbedder {
    /// Fetch (if needed), verify and load the pinned model.
    pub async fn load(config: LocalEmbedConfig) -> Result<Self, ProviderError> {
        let mut files = ensure_files(&config).await?;
        let tokenizer_config_file = files.pop().expect("five files");
        let special_tokens_map_file = files.pop().expect("five files");
        let config_file = files.pop().expect("five files");
        let tokenizer_file = files.pop().expect("five files");
        let onnx = files.pop().expect("five files");
        let model = UserDefinedEmbeddingModel::new(
            onnx,
            TokenizerFiles {
                tokenizer_file,
                config_file,
                special_tokens_map_file,
                tokenizer_config_file,
            },
        )
        .with_pooling(match config.pooling {
            LocalPooling::Cls => Pooling::Cls,
            LocalPooling::Mean => Pooling::Mean,
        });
        let (sessions, threads) = config.parallelism();
        let mut options = InitOptionsUserDefined::new()
            .with_max_length(config.max_length)
            .with_intra_threads(threads);
        if config.device == LocalDevice::Cpu {
            options = options.with_auto_device(AutoDevicePolicy::PreferCPU);
        }
        let loaded = tokio::task::spawn_blocking(move || {
            (0..sessions)
                .map(|_| {
                    TextEmbedding::try_new_from_user_defined(model.clone(), options.clone())
                        .map(|m| Arc::new(Mutex::new(m)))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .await
        .map_err(|e| config_err(format!("loading local embedding model panicked: {e}")))?
        .map_err(|e| config_err(format!("cannot load local embedding model: {e}")))?;
        Ok(Self {
            sessions: loaded,
            next_session: AtomicUsize::new(0),
            descriptor: ProviderDescriptor {
                provider: "local-onnx".into(),
                model: config.repo.clone(),
                revision: config.revision.clone(),
                mock: false,
            },
            dimensions: config.dimensions,
        })
    }
}

#[async_trait]
impl Embedder for LocalEmbedder {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbedResponse, ProviderError> {
        let _t = crate::timing::span("onnx.embed").items(texts.len());
        let texts: Arc<[String]> = texts.into();
        let next = Arc::new(AtomicUsize::new(0));
        let workers = self.sessions.len().min(texts.len());
        // Rotate the first session so concurrent callers spread out.
        let first = self.next_session.fetch_add(1, Ordering::Relaxed);
        let mut tasks = Vec::with_capacity(workers);
        for w in 0..workers {
            let session = self.sessions[(first + w) % self.sessions.len()].clone();
            let (texts, next) = (texts.clone(), next.clone());
            tasks.push(tokio::task::spawn_blocking(move || {
                let mut m = session.lock().expect("embedding model lock");
                let mut out = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= texts.len() {
                        return Ok(out);
                    }
                    let mut v = m.embed(&texts[i..=i], Some(1))?;
                    out.push((i, v.pop().unwrap_or_default()));
                }
            }));
        }
        let mut vectors = vec![Vec::new(); texts.len()];
        for task in tasks {
            let done: Result<Vec<(usize, Vec<f32>)>, fastembed::Error> =
                task.await.map_err(|e| ProviderError::Transport {
                    message: format!("embedding task panicked: {e}"),
                })?;
            for (i, v) in done.map_err(|e| ProviderError::Malformed {
                message: format!("local embedding failed: {e}"),
            })? {
                vectors[i] = v;
            }
        }
        let dims = self.dimensions;
        if let Some(v) = vectors.iter().find(|v| v.len() != dims) {
            return Err(ProviderError::Malformed {
                message: format!(
                    "model returned {} dimensions, manifest pins {dims}",
                    v.len()
                ),
            });
        }
        Ok(EmbedResponse {
            vectors,
            usage: Usage::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(dir: &Path) -> LocalEmbedConfig {
        LocalEmbedConfig {
            sha256: vec![],
            ..LocalEmbedConfig::bge_small_en_v1_5(dir.to_owned())
        }
    }

    #[tokio::test]
    async fn rejects_branch_revisions_and_missing_digests_before_any_download() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = config(dir.path());
        c.revision = "main".into();
        assert!(matches!(
            ensure_files(&c).await,
            Err(ProviderError::Config { .. })
        ));
        let c = config(dir.path());
        assert!(matches!(
            ensure_files(&c).await,
            Err(ProviderError::Config { .. })
        ));
    }

    #[test]
    fn cached_files_are_verified_against_their_pins() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f");
        std::fs::write(&p, b"abc").unwrap();
        let good = sha256_label(b"abc");
        assert!(read_verified(&p, &good).is_ok());
        assert!(read_verified(&p, "sha256:00").is_err());
        assert_eq!(
            config(dir.path()).model_dir(),
            dir.path()
                .join("BAAI--bge-small-en-v1.5")
                .join("5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
        );
    }

    #[test]
    fn the_preset_pins_every_file() {
        let c = LocalEmbedConfig::bge_small_en_v1_5(PathBuf::from("cache"));
        assert_eq!(c.sha256.len(), MODEL_FILES.len());
        assert!(c.sha256.iter().all(|h| h.len() == "sha256:".len() + 64));
    }

    #[test]
    fn default_parallelism_never_oversubscribes_small_hosts() {
        assert_eq!(parallelism_for_cores(1), (1, 1));
        assert_eq!(parallelism_for_cores(2), (1, 2));
        assert_eq!(parallelism_for_cores(3), (1, 3));
        assert_eq!(parallelism_for_cores(6), (2, 3));
        assert_eq!(parallelism_for_cores(10), (3, 3));
    }

    /// Downloads about 130 MB. Run with
    /// `KEEP_TEST_DOWNLOAD_MODELS=1 cargo test -p keep-providers --features local-embeddings`.
    #[tokio::test]
    async fn downloads_and_embeds_with_the_pinned_model() {
        let Some(cache) = std::env::var_os("KEEP_TEST_DOWNLOAD_MODELS").map(|_| {
            std::env::var_os("KEEP_MODEL_CACHE")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("../../.keep-models"))
        }) else {
            eprintln!("skipped: set KEEP_TEST_DOWNLOAD_MODELS=1 to download the model");
            return;
        };
        let c = LocalEmbedConfig::bge_small_en_v1_5(cache);
        let e = LocalEmbedder::load(c).await.unwrap();
        let r = e
            .embed(&[
                "The meeting moved to Tuesday.".into(),
                "Lunch is at noon.".into(),
            ])
            .await
            .unwrap();
        assert_eq!(r.vectors.len(), 2);
        assert_eq!(r.vectors[0].len(), 384);
        let norm: f32 = r.vectors[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3);
    }

    /// The model already in `KEEP_MODEL_CACHE`, or `None` to skip; never
    /// downloads.
    fn cached_model() -> Option<LocalEmbedConfig> {
        let cache = PathBuf::from(std::env::var_os("KEEP_MODEL_CACHE")?);
        let c = LocalEmbedConfig::bge_small_en_v1_5(cache);
        let dir = c.model_dir();
        MODEL_FILES
            .iter()
            .all(|f| dir.join(f).exists())
            .then_some(c)
    }

    fn passages() -> Vec<String> {
        (0..24)
            .map(|i| {
                let words = [
                    "meeting", "Tuesday", "budget", "flight", "garden", "invoice",
                ];
                (0..(3 + i * 7))
                    .map(|j| words[(i + j) % words.len()])
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    /// Run with `KEEP_MODEL_CACHE=<dir holding the model> cargo test -p
    /// keep-providers --features local-embeddings`.
    #[tokio::test]
    async fn vectors_do_not_depend_on_sessions_threads_or_call_grouping() {
        let Some(base) = cached_model() else {
            eprintln!("skipped: KEEP_MODEL_CACHE does not hold the pinned model");
            return;
        };
        let texts = passages();
        let mut runs = Vec::new();
        for (sessions, threads) in [(1, 1), (3, 3), (2, 4)] {
            let e = LocalEmbedder::load(LocalEmbedConfig {
                sessions: Some(sessions),
                intra_threads: Some(threads),
                ..base.clone()
            })
            .await
            .unwrap();
            let all = e.embed(&texts).await.unwrap().vectors;
            let mut piecewise = Vec::new();
            for chunk in texts.chunks(5) {
                piecewise.extend(e.embed(chunk).await.unwrap().vectors);
            }
            assert_eq!(all, piecewise, "grouping changed vectors");
            runs.push(all);
        }
        for r in &runs[1..] {
            assert_eq!(r, &runs[0], "sessions or threads changed vectors");
        }
        // fastembed's own batching pads to the longest text; the vectors
        // agree with one-text runs to float tolerance.
        let mut padded = TextEmbedding::try_new_from_user_defined(
            UserDefinedEmbeddingModel::new(
                std::fs::read(base.model_dir().join(MODEL_FILES[0])).unwrap(),
                TokenizerFiles {
                    tokenizer_file: std::fs::read(base.model_dir().join(MODEL_FILES[1])).unwrap(),
                    config_file: std::fs::read(base.model_dir().join(MODEL_FILES[2])).unwrap(),
                    special_tokens_map_file: std::fs::read(base.model_dir().join(MODEL_FILES[3]))
                        .unwrap(),
                    tokenizer_config_file: std::fs::read(base.model_dir().join(MODEL_FILES[4]))
                        .unwrap(),
                },
            )
            .with_pooling(Pooling::Cls),
            InitOptionsUserDefined::new()
                .with_max_length(base.max_length)
                .with_auto_device(AutoDevicePolicy::PreferCPU),
        )
        .unwrap();
        let batched = padded.embed(&texts, Some(4)).unwrap();
        let max_diff = batched
            .iter()
            .zip(&runs[0])
            .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
            .fold(0f32, f32::max);
        eprintln!("max |padded batch - single| = {max_diff:e}");
        assert!(max_diff < 1e-5, "{max_diff}");
    }
}
