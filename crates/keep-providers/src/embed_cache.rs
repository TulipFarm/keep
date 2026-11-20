//! A content-addressed cache in front of a deterministic local embedder.
//!
//! Each text's vector is keyed by a SHA-256 over a namespace and the exact
//! text. The namespace names the provider, model, revision, dimensions and
//! anything else that changes the output (pooling, maximum length, model file
//! digest, execution device), so a different model can never hit. Misses are
//! embedded together in one batched call. Vectors live in memory and, when a
//! directory is given, in an append-only file per namespace, so later runs
//! and other benchmark arms reuse them.
//!
//! Only wrap free, local embedders: a hit reports no usage, which would hide
//! the cost of a paid provider.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::ProviderDescriptor;
use crate::chat::{ProviderError, Usage};
use crate::embed::{EmbedResponse, Embedder};

type Key = [u8; 32];

const RECORD_CHECK: usize = 8;

pub struct CachedEmbedder {
    inner: Arc<dyn Embedder>,
    namespace: Key,
    dims: usize,
    vectors: Mutex<HashMap<Key, Arc<[f32]>>>,
    file: Option<Mutex<File>>,
}

impl CachedEmbedder {
    /// `extra` must name every setting outside the descriptor that changes
    /// the vectors, such as pooling and maximum length.
    pub fn new(
        inner: Arc<dyn Embedder>,
        extra: &str,
        dir: Option<&Path>,
    ) -> Result<Self, ProviderError> {
        let d = inner.descriptor();
        let dims = inner.dimensions();
        let mut h = Sha256::new();
        for part in [
            "keep-embed-cache-v1",
            &d.provider,
            &d.model,
            &d.revision,
            &dims.to_string(),
            extra,
        ] {
            h.update((part.len() as u64).to_le_bytes());
            h.update(part.as_bytes());
        }
        let namespace: Key = h.finalize().into();
        let mut vectors = HashMap::new();
        let file = match dir {
            None => None,
            Some(dir) => {
                let path = cache_path(dir, &namespace);
                std::fs::create_dir_all(dir).map_err(|e| io_err(dir, e))?;
                load(&path, dims, &mut vectors)?;
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|e| io_err(&path, e))?;
                Some(Mutex::new(f))
            }
        };
        Ok(Self {
            inner,
            namespace,
            dims,
            vectors: Mutex::new(vectors),
            file,
        })
    }

    fn key(&self, text: &str) -> Key {
        let mut h = Sha256::new();
        h.update(self.namespace);
        h.update(text.as_bytes());
        h.finalize().into()
    }

    /// Vectors held in memory, including those loaded from disk.
    pub fn len(&self) -> usize {
        self.vectors.lock().expect("embedding cache lock").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn cache_path(dir: &Path, namespace: &Key) -> PathBuf {
    dir.join(format!("{}.vec", hex(&namespace[..16])))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn io_err(path: &Path, e: std::io::Error) -> ProviderError {
    ProviderError::Config {
        message: format!("embedding cache {}: {e}", path.display()),
    }
}

fn check(key: &Key, body: &[u8]) -> [u8; RECORD_CHECK] {
    let mut h = Sha256::new();
    h.update(key);
    h.update(body);
    h.finalize()[..RECORD_CHECK].try_into().expect("8 bytes")
}

fn record(key: &Key, v: &[f32]) -> Vec<u8> {
    let body: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut out = Vec::with_capacity(32 + body.len() + RECORD_CHECK);
    out.extend_from_slice(key);
    out.extend_from_slice(&body);
    out.extend_from_slice(&check(key, &body));
    out
}

/// Read every whole, intact record. A torn tail from an interrupted write,
/// or a corrupt record, is skipped; its text is simply embedded again.
fn load(
    path: &Path,
    dims: usize,
    into: &mut HashMap<Key, Arc<[f32]>>,
) -> Result<(), ProviderError> {
    let mut bytes = Vec::new();
    match File::open(path) {
        Ok(mut f) => f.read_to_end(&mut bytes).map_err(|e| io_err(path, e))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_err(path, e)),
    };
    let size = 32 + dims * 4 + RECORD_CHECK;
    for rec in bytes.chunks_exact(size) {
        let key: Key = rec[..32].try_into().expect("32 bytes");
        let body = &rec[32..32 + dims * 4];
        if rec[32 + dims * 4..] != check(&key, body) {
            continue;
        }
        let v: Arc<[f32]> = body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        into.entry(key).or_insert(v);
    }
    Ok(())
}

#[async_trait]
impl Embedder for CachedEmbedder {
    fn descriptor(&self) -> &ProviderDescriptor {
        self.inner.descriptor()
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    async fn embed(&self, texts: &[String]) -> Result<EmbedResponse, ProviderError> {
        let keys: Vec<Key> = texts.iter().map(|t| self.key(t)).collect();
        let mut out: Vec<Option<Arc<[f32]>>> = {
            let cache = self.vectors.lock().expect("embedding cache lock");
            keys.iter().map(|k| cache.get(k).cloned()).collect()
        };
        // Embed each distinct missing text once, in input order.
        let mut miss_keys = Vec::new();
        let mut miss_texts = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for (i, k) in keys.iter().enumerate() {
            if out[i].is_none() && seen.insert(*k) {
                miss_keys.push(*k);
                miss_texts.push(texts[i].clone());
            }
        }
        crate::timing::record(
            "embed_cache.hit",
            Default::default(),
            (texts.len() - miss_texts.len()) as u64,
        );
        let mut usage = Usage::default();
        if !miss_texts.is_empty() {
            let resp = self.inner.embed(&miss_texts).await?;
            if resp.vectors.len() != miss_texts.len() {
                return Err(ProviderError::Malformed {
                    message: format!(
                        "embedder returned {} vectors for {} texts",
                        resp.vectors.len(),
                        miss_texts.len()
                    ),
                });
            }
            usage = resp.usage;
            let mut bytes = Vec::new();
            {
                let mut cache = self.vectors.lock().expect("embedding cache lock");
                for (k, v) in miss_keys.iter().zip(resp.vectors) {
                    if v.len() != self.dims {
                        return Err(ProviderError::Malformed {
                            message: format!(
                                "embedder returned {} dimensions, expected {}",
                                v.len(),
                                self.dims
                            ),
                        });
                    }
                    if self.file.is_some() {
                        bytes.extend_from_slice(&record(k, &v));
                    }
                    cache.entry(*k).or_insert_with(|| v.into());
                }
                for (i, k) in keys.iter().enumerate() {
                    if out[i].is_none() {
                        out[i] = cache.get(k).cloned();
                    }
                }
            }
            if let Some(f) = &self.file {
                // A failed write only loses reuse in a later run.
                let mut f = f.lock().expect("embedding cache file lock");
                let _ = f.write_all(&bytes);
            }
        }
        Ok(EmbedResponse {
            vectors: out
                .into_iter()
                .map(|v| v.expect("every text has a vector").to_vec())
                .collect(),
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HashEmbedder;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counting {
        inner: HashEmbedder,
        texts: AtomicUsize,
    }

    #[async_trait]
    impl Embedder for Counting {
        fn descriptor(&self) -> &ProviderDescriptor {
            self.inner.descriptor()
        }
        fn dimensions(&self) -> usize {
            self.inner.dimensions()
        }
        async fn embed(&self, texts: &[String]) -> Result<EmbedResponse, ProviderError> {
            self.texts.fetch_add(texts.len(), Ordering::SeqCst);
            self.inner.embed(texts).await
        }
    }

    fn counting(dims: usize) -> Arc<Counting> {
        Arc::new(Counting {
            inner: HashEmbedder::new(dims),
            texts: AtomicUsize::new(0),
        })
    }

    fn texts(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[tokio::test]
    async fn hits_return_the_same_vectors_and_skip_the_model() {
        let inner = counting(16);
        let c = CachedEmbedder::new(inner.clone(), "x", None).unwrap();
        let direct = inner
            .inner
            .embed(&texts(&["a b", "c d", "a b"]))
            .await
            .unwrap();
        let first = c.embed(&texts(&["a b", "c d", "a b"])).await.unwrap();
        assert_eq!(first.vectors, direct.vectors);
        assert_eq!(
            inner.texts.load(Ordering::SeqCst),
            2,
            "duplicates embed once"
        );
        let second = c.embed(&texts(&["c d", "e f"])).await.unwrap();
        assert_eq!(second.vectors[0], direct.vectors[1]);
        assert_eq!(inner.texts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn the_disk_cache_survives_restarts_and_never_crosses_namespaces() {
        let dir = tempfile::tempdir().unwrap();
        let t = texts(&["the meeting moved", "lunch at noon"]);
        let first = counting(8);
        let a = CachedEmbedder::new(first.clone(), "pool=cls", Some(dir.path())).unwrap();
        let v = a.embed(&t).await.unwrap().vectors;
        drop(a);

        let again = counting(8);
        let b = CachedEmbedder::new(again.clone(), "pool=cls", Some(dir.path())).unwrap();
        assert_eq!(b.embed(&t).await.unwrap().vectors, v);
        assert_eq!(again.texts.load(Ordering::SeqCst), 0);

        let other = counting(8);
        let c = CachedEmbedder::new(other.clone(), "pool=mean", Some(dir.path())).unwrap();
        c.embed(&t).await.unwrap();
        assert_eq!(other.texts.load(Ordering::SeqCst), 2);
        let wider = counting(16);
        let d = CachedEmbedder::new(wider.clone(), "pool=cls", Some(dir.path())).unwrap();
        assert_eq!(d.embed(&t).await.unwrap().vectors[0].len(), 16);
        assert_eq!(wider.texts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_torn_or_corrupt_record_is_embedded_again() {
        let dir = tempfile::tempdir().unwrap();
        let t = texts(&["one", "two"]);
        let a = CachedEmbedder::new(counting(4), "", Some(dir.path())).unwrap();
        let v = a.embed(&t).await.unwrap().vectors;
        drop(a);
        let path = std::fs::read_dir(dir.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[40] ^= 0xff;
        bytes.extend_from_slice(&[1, 2, 3]);
        std::fs::write(&path, bytes).unwrap();
        let inner = counting(4);
        let b = CachedEmbedder::new(inner.clone(), "", Some(dir.path())).unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b.embed(&t).await.unwrap().vectors, v);
        assert_eq!(inner.texts.load(Ordering::SeqCst), 1);
    }
}
