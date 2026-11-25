//! Server settings persisted next to a Postgres deployment's data: the
//! embedder each vector collection was built with, and the receipt key.

use std::sync::Arc;

use anyhow::Context as _;
use keep_providers::Embedder;
use keep_store::{StoreError, VectorIndex};
use serde::{Deserialize, Serialize};
use tokio_postgres::Client;

/// A small key-value table, `keep_server_settings`.
pub struct Settings {
    client: Client,
}

impl Settings {
    /// Connect and create the table if needed. `url` already selects the
    /// schema through its search path, if any.
    pub async fn connect(url: &str, schema: Option<&str>) -> anyhow::Result<Self> {
        let client = keep_postgres::connect(url)
            .await
            .context("connecting for server settings")?;
        if let Some(s) = schema {
            client
                .batch_execute(&format!(
                    "CREATE SCHEMA IF NOT EXISTS \"{}\"",
                    s.replace('"', "\"\"")
                ))
                .await
                .context("creating the settings schema")?;
        }
        client
            .batch_execute(
                "CREATE TABLE IF NOT EXISTS keep_server_settings (
                     key text PRIMARY KEY,
                     value text NOT NULL
                 )",
            )
            .await
            .context("creating keep_server_settings")?;
        Ok(Self { client })
    }

    pub async fn get(&self, key: &str) -> anyhow::Result<Option<String>> {
        Ok(self
            .client
            .query_opt(
                "SELECT value FROM keep_server_settings WHERE key = $1",
                &[&key],
            )
            .await?
            .map(|r| r.get(0)))
    }

    /// Store `value` unless the key exists, and return what is stored, so
    /// concurrent first starts agree.
    pub async fn get_or_insert(&self, key: &str, value: &str) -> anyhow::Result<String> {
        self.client
            .execute(
                "INSERT INTO keep_server_settings (key, value) VALUES ($1, $2)
                 ON CONFLICT (key) DO NOTHING",
                &[&key, &value],
            )
            .await?;
        self.get(key)
            .await?
            .context("settings row vanished after insert")
    }
}

/// What built a collection's vectors.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderRecord {
    pub provider: String,
    pub model: String,
    pub revision: String,
    pub dimensions: usize,
}

impl EmbedderRecord {
    pub fn of(e: &dyn Embedder) -> Self {
        let d = e.descriptor();
        Self {
            provider: d.provider.clone(),
            model: d.model.clone(),
            revision: d.revision.clone(),
            dimensions: e.dimensions(),
        }
    }

    pub fn label(&self) -> String {
        format!(
            "{} ({}, {} dimensions)",
            self.model, self.revision, self.dimensions
        )
    }
}

fn rebuild_hint(collection: &str) -> String {
    format!(
        "Either start with the embedder that built it (--embedder), or re-embed: start with a \
         new, empty --collection (the old one \"{collection}\" is left untouched) and call \
         knowledge_rebuild_projection for each user's scope, which re-embeds every chunk and \
         fact from the ledger."
    )
}

/// Refuse to serve a collection built with another embedder. The first
/// start records `ours`; later starts must match it. A collection that
/// predates the record is checked by its vector size.
pub async fn check_embedder(
    settings: &Settings,
    index: Option<&Arc<dyn VectorIndex>>,
    collection: &str,
    ours: &EmbedderRecord,
) -> anyhow::Result<()> {
    let key = format!("embedder:{collection}");
    let mine = serde_json::to_string(ours)?;
    let stored = match settings.get(&key).await? {
        Some(v) => v,
        None => {
            if let Some(index) = index {
                match index.ensure_collection(collection, ours.dimensions).await {
                    Ok(()) => {}
                    Err(StoreError::DimensionMismatch { expected, .. }) => anyhow::bail!(
                        "vector collection \"{collection}\" holds {expected}-dimensional \
                         vectors, but the {} embedder makes {}. {}",
                        ours.model,
                        ours.dimensions,
                        rebuild_hint(collection)
                    ),
                    Err(e) => {
                        return Err(anyhow::anyhow!(e))
                            .context(format!("checking vector collection \"{collection}\""));
                    }
                }
            }
            settings.get_or_insert(&key, &mine).await?
        }
    };
    let stored: EmbedderRecord = serde_json::from_str(&stored)
        .with_context(|| format!("reading the stored embedder for \"{collection}\""))?;
    if &stored != ours {
        anyhow::bail!(
            "vector collection \"{collection}\" was built with {}, but this server is set to \
             use {}. {}",
            stored.label(),
            ours.label(),
            rebuild_hint(collection)
        );
    }
    Ok(())
}

/// Bytes in a generated receipt key.
const RECEIPT_KEY_BYTES: usize = 32;

fn new_receipt_key() -> String {
    use rand::TryRng;
    let mut bytes = [0u8; RECEIPT_KEY_BYTES];
    rand::rngs::SysRng
        .try_fill_bytes(&mut bytes)
        .expect("the OS random source is available");
    hex::encode(bytes)
}

fn decode_receipt_key(text: &str, from: &str) -> anyhow::Result<Vec<u8>> {
    let key = hex::decode(text.trim()).with_context(|| format!("{from} is not hex"))?;
    anyhow::ensure!(
        key.len() >= 16,
        "{from} is too short: {} bytes, want at least 16",
        key.len()
    );
    Ok(key)
}

/// The deployment's receipt key, stored in the settings table so every
/// server on the same database signs receipts alike.
pub async fn database_receipt_key(settings: &Settings) -> anyhow::Result<Vec<u8>> {
    let stored = settings
        .get_or_insert("receipt_key", &new_receipt_key())
        .await?;
    decode_receipt_key(&stored, "the stored receipt key")
}

/// The receipt key in `path`, created on first use with owner-only
/// permissions. Concurrent first starts agree on one key.
pub fn file_receipt_key(path: &std::path::Path) -> anyhow::Result<Vec<u8>> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            warn_if_shared(path);
            return decode_receipt_key(&text, &path.display().to_string());
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    }
    let dir = path.parent().context("receipt key path has no directory")?;
    create_private_dir(dir)?;
    let tmp = dir.join(format!(".receipt.key.{}", std::process::id()));
    write_private(&tmp, new_receipt_key().as_bytes())?;
    // A hard link fails if another process created the key first.
    let linked = std::fs::hard_link(&tmp, path);
    let _ = std::fs::remove_file(&tmp);
    match linked {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e).with_context(|| format!("creating {}", path.display())),
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    decode_receipt_key(&text, &path.display().to_string())
}

#[cfg(unix)]
fn create_private_dir(dir: &std::path::Path) -> anyhow::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
        .with_context(|| format!("creating {}", dir.display()))
}

#[cfg(not(unix))]
fn create_private_dir(dir: &std::path::Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))
}

#[cfg(unix)]
fn write_private(path: &std::path::Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .with_context(|| format!("creating {}", path.display()))?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn write_private(path: &std::path::Path, bytes: &[u8]) -> anyhow::Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("creating {}", path.display()))
}

#[cfg(unix)]
fn warn_if_shared(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(m) = std::fs::metadata(path)
        && m.permissions().mode() & 0o077 != 0
    {
        eprintln!(
            "keep: warning: {} is readable by other users; chmod 600 it",
            path.display()
        );
    }
}

#[cfg(not(unix))]
fn warn_if_shared(_: &std::path::Path) {}
