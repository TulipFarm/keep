//! Where the public datasets come from, pinned to exact revisions.
//!
//! Files are downloaded into the git-ignored `data/` directory by
//! `keep-bench fetch` and verified against the checksums below. Datasets are
//! never committed and never downloaded by tests.

use super::DatasetKind;

#[derive(Clone, Copy, Debug)]
pub struct DatasetSource {
    pub kind: DatasetKind,
    /// Path under the data directory.
    pub file: &'static str,
    pub url: &'static str,
    /// Expected SHA-256. `None` means the file is too large to have been
    /// verified when the pin was made; the first download's hash is printed
    /// and must be recorded in the run manifest.
    pub sha256: Option<&'static str>,
    pub approx_bytes: u64,
    pub license: &'static str,
    pub citation: &'static str,
    /// Fetched when no dataset is named explicitly.
    pub default: bool,
}

pub const SOURCES: &[DatasetSource] = &[
    DatasetSource {
        kind: DatasetKind::Locomo,
        file: "locomo/locomo10.json",
        url: const_format_url::LOCOMO,
        sha256: Some("79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"),
        approx_bytes: 2_805_274,
        license: "CC BY-NC 4.0 (non-commercial use only)",
        citation: "Maharana et al., Evaluating Very Long-Term Conversational Memory of LLM Agents, ACL 2024",
        default: true,
    },
    DatasetSource {
        kind: DatasetKind::LongmemevalS,
        file: "longmemeval/longmemeval_s_cleaned.json",
        url: const_format_url::LME_S,
        sha256: Some("d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"),
        approx_bytes: 277_383_467,
        license: "MIT",
        citation: "Wu et al., LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory, ICLR 2025",
        default: true,
    },
    DatasetSource {
        kind: DatasetKind::LongmemevalOracle,
        file: "longmemeval/longmemeval_oracle.json",
        url: const_format_url::LME_ORACLE,
        sha256: Some("821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"),
        approx_bytes: 15_388_478,
        license: "MIT",
        citation: "Wu et al., LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory, ICLR 2025",
        default: true,
    },
    DatasetSource {
        kind: DatasetKind::LongmemevalM,
        file: "longmemeval/longmemeval_m_cleaned.json",
        url: const_format_url::LME_M,
        sha256: None,
        approx_bytes: 2_737_100_077,
        license: "MIT",
        citation: "Wu et al., LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory, ICLR 2025",
        default: false,
    },
];

/// URLs are spelled out in full so the pinned revisions are visible in one
/// place and greppable.
mod const_format_url {
    pub const LOCOMO: &str = "https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json";
    pub const LME_S: &str = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json";
    pub const LME_ORACLE: &str = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json";
    pub const LME_M: &str = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_m_cleaned.json";
}

pub fn source(kind: DatasetKind) -> &'static DatasetSource {
    SOURCES
        .iter()
        .find(|s| s.kind == kind)
        .expect("every kind has a source")
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCOMO_REV: &str = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376";
    const LME_REV: &str = "98d7416c24c778c2fee6e6f3006e7a073259d48f";

    #[test]
    fn urls_use_the_pinned_revisions() {
        for s in SOURCES {
            let rev = if s.kind == DatasetKind::Locomo {
                LOCOMO_REV
            } else {
                LME_REV
            };
            assert!(s.url.contains(rev), "{} is not pinned", s.url);
            assert!(s.url.ends_with(s.file.rsplit('/').next().unwrap()));
        }
    }
}
