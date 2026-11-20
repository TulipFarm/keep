#![allow(dead_code)]

use std::path::{Path, PathBuf};

use keep_bench::manifest::LoadedManifest;

pub fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .canonicalize()
        .unwrap()
}

pub fn fixture_manifest(name: &str) -> String {
    std::fs::read_to_string(fixtures().join("manifests").join(name)).unwrap()
}

/// Write a (possibly edited) fixture manifest into `dir`, rewriting its
/// relative file paths so they still point at the committed fixtures.
pub fn write_manifest(dir: &Path, toml_text: &str) -> LoadedManifest {
    let abs = format!("\"{}/", fixtures().display());
    let text = toml_text.replace("\"../", &abs);
    let path = dir.join("manifest.toml");
    std::fs::write(&path, text).unwrap();
    LoadedManifest::load(&path).unwrap()
}

pub fn load(name: &str) -> LoadedManifest {
    LoadedManifest::load(&fixtures().join("manifests").join(name)).unwrap()
}

/// Replace one `key = value` line (first match) in a TOML string.
pub fn set(toml_text: &str, key: &str, value: &str) -> String {
    let mut done = false;
    toml_text
        .lines()
        .map(|l| {
            if !done && l.trim_start().starts_with(&format!("{key} =")) {
                done = true;
                format!("{key} = {value}")
            } else {
                l.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
