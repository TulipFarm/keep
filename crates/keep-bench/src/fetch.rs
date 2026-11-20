//! Download public datasets into the git-ignored data directory.

use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::dataset::DatasetKind;
use crate::dataset::registry::{DatasetSource, SOURCES, source};

#[derive(Debug)]
pub struct Fetched {
    pub kind: DatasetKind,
    pub path: PathBuf,
    pub sha256: String,
    pub verified: bool,
    pub skipped: bool,
}

pub fn file_sha256(path: &Path) -> anyhow::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(hex::encode(h.finalize()))
}

/// Datasets to fetch: those named, or every default one.
pub fn selection(kinds: &[DatasetKind]) -> Vec<&'static DatasetSource> {
    if kinds.is_empty() {
        SOURCES.iter().filter(|s| s.default).collect()
    } else {
        kinds.iter().map(|k| source(*k)).collect()
    }
}

pub async fn fetch(src: &DatasetSource, data_dir: &Path, force: bool) -> anyhow::Result<Fetched> {
    let path = data_dir.join(src.file);
    if path.exists() && !force {
        let sha = file_sha256(&path)?;
        if let Some(expected) = src.sha256 {
            anyhow::ensure!(
                sha == expected,
                "{} exists but has sha256 {sha}, expected {expected}; delete it or pass --force",
                path.display()
            );
        }
        return Ok(Fetched {
            kind: src.kind,
            path,
            verified: src.sha256.is_some(),
            sha256: sha,
            skipped: true,
        });
    }
    std::fs::create_dir_all(path.parent().expect("file has a parent"))?;
    let partial = path.with_extension("partial");
    let mut resp = reqwest::Client::new()
        .get(src.url)
        .send()
        .await?
        .error_for_status()?;
    let mut file = std::io::BufWriter::new(std::fs::File::create(&partial)?);
    let mut hasher = Sha256::new();
    while let Some(chunk) = resp.chunk().await? {
        hasher.update(&chunk);
        file.write_all(&chunk)?;
    }
    file.flush()?;
    drop(file);
    let sha = hex::encode(hasher.finalize());
    if let Some(expected) = src.sha256
        && sha != expected
    {
        std::fs::remove_file(&partial)?;
        anyhow::bail!(
            "{} downloaded with sha256 {sha}, expected {expected}; file discarded",
            src.url
        );
    }
    std::fs::rename(&partial, &path)?;
    let notice = PathBuf::from(format!("{}.source.txt", path.display()));
    std::fs::write(
        notice,
        format!(
            "Downloaded by keep-bench fetch.\nSource: {}\nLicense: {}\nCitation: {}\n\
             This data is not part of Keep and is not redistributed by it.\n",
            src.url, src.license, src.citation
        ),
    )?;
    Ok(Fetched {
        kind: src.kind,
        path,
        verified: src.sha256.is_some(),
        sha256: sha,
        skipped: false,
    })
}
