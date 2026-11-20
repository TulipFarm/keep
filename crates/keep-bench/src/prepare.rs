//! Load and check everything a manifest pins before any arm runs.

use std::collections::HashSet;

use keep_core::ContentHash;

use crate::dataset::Dataset;
use crate::dataset::annotations::AnnotationFile;
use crate::fitmask::FitMask;
use crate::holdout::{self, Split};
use crate::manifest::{LoadedManifest, has_errors};

#[derive(Debug)]
pub struct Prepared {
    pub dataset: Dataset,
    pub annotations: AnnotationFile,
    pub split: Split,
    pub fit_mask: Option<FitMask>,
    pub subset_size: Option<usize>,
}

/// Load the full dataset and verify its pinned hash (no split applied).
pub fn load_dataset(m: &LoadedManifest) -> anyhow::Result<Dataset> {
    let spec = &m.manifest.dataset;
    let bytes = m.read_pinned(&crate::manifest::PinnedFile {
        path: spec.path.clone(),
        sha256: spec.sha256.clone(),
    })?;
    Dataset::from_bytes(spec.kind, &bytes)
}

pub fn prepare(m: &LoadedManifest, unseal: bool) -> anyhow::Result<Prepared> {
    prepare_scoped(m, unseal, false)
}

/// [`prepare`] for a run that may consume the holdout when the manifest's
/// protocol asks for the full set. Also applies the protocol's question
/// filter, after the split and subset.
pub fn prepare_scoped(
    m: &LoadedManifest,
    unseal: bool,
    consume_holdout: bool,
) -> anyhow::Result<Prepared> {
    let issues = m.manifest.validate();
    if has_errors(&issues) {
        let lines: Vec<String> = issues
            .iter()
            .filter(|i| i.error)
            .map(|i| format!("  - {}: {}", i.field, i.message))
            .collect();
        anyhow::bail!(
            "manifest {} is invalid:\n{}",
            m.path.display(),
            lines.join("\n")
        );
    }
    let protocol = m.manifest.protocol.name;
    for (name, model, expected) in [
        (
            "answer",
            &m.manifest.models.answer,
            protocol.answer_prompt_version(),
        ),
        (
            "judge",
            &m.manifest.models.judge,
            protocol.judge_prompt_version(),
        ),
    ] {
        anyhow::ensure!(
            model.prompt_version == expected,
            "manifest pins {name} prompt {} but this build implements {expected}",
            model.prompt_version
        );
    }
    let mut dataset = load_dataset(m)?;

    // The split is defined over the whole dataset, so it is applied before
    // any subset narrows the question list.
    let split = holdout::apply_scoped(&m.manifest, &mut dataset, unseal, consume_holdout)?;

    let mut subset_size = None;
    if let Some(subset) = &m.manifest.dataset.subset {
        let text = String::from_utf8(m.read_pinned(subset)?)?;
        let ids: HashSet<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        subset_size = Some(ids.len());
        dataset.questions.retain(|q| ids.contains(q.id.as_str()));
        let used: HashSet<&str> = dataset
            .questions
            .iter()
            .map(|q| q.history_id.as_str())
            .collect();
        let used: HashSet<String> = used.into_iter().map(str::to_owned).collect();
        dataset.histories.retain(|h| used.contains(&h.id));
        anyhow::ensure!(
            !dataset.questions.is_empty(),
            "subset selects no questions from this split"
        );
    }

    if !m.manifest.protocol.is_keep() {
        dataset.questions.retain(|q| protocol.includes(q));
        let used: HashSet<String> = dataset
            .questions
            .iter()
            .map(|q| q.history_id.clone())
            .collect();
        dataset.histories.retain(|h| used.contains(&h.id));
        anyhow::ensure!(
            !dataset.questions.is_empty(),
            "protocol {} scores no questions in this split",
            protocol.as_str()
        );
    }

    let annotations = match &m.manifest.dataset.annotations {
        Some(p) => AnnotationFile::parse(&String::from_utf8(m.read_pinned(p)?)?)?,
        None => AnnotationFile::default(),
    };

    let fit_mask = match &m.manifest.fit_mask {
        Some(p) => {
            let mask: FitMask = serde_json::from_slice(&m.read_pinned(p)?)?;
            anyhow::ensure!(
                mask.dataset_hash == m.manifest.dataset.sha256,
                "fit mask was computed for dataset {}, manifest pins {}",
                mask.dataset_hash,
                m.manifest.dataset.sha256
            );
            Some(mask)
        }
        None => None,
    };

    Ok(Prepared {
        dataset,
        annotations,
        split,
        fit_mask,
        subset_size,
    })
}

/// The commitment string for a manifest's holdout definition over its
/// pinned dataset.
pub fn seal(m: &LoadedManifest) -> anyhow::Result<(ContentHash, usize, usize)> {
    let ds = load_dataset(m)?;
    let h = &m.manifest.holdout;
    let ids = holdout::holdout_ids(&ds, &h.salt, h.fraction);
    Ok((
        holdout::commitment(&ds, &h.salt, h.fraction),
        ids.len(),
        ds.histories.len(),
    ))
}

/// Structural validation plus every check that needs the pinned files:
/// dataset hash, holdout commitment, subset, annotations, fit mask and
/// prompt versions. Does not unseal anything.
pub fn check(m: &LoadedManifest) -> Vec<crate::manifest::Issue> {
    use crate::manifest::Issue;
    let mut issues = m.manifest.validate();
    let mut err = |field: &str, message: String| {
        issues.push(Issue {
            error: true,
            field: field.into(),
            message,
        })
    };
    let protocol = m.manifest.protocol.name;
    for (name, model, expected) in [
        (
            "models.answer",
            &m.manifest.models.answer,
            protocol.answer_prompt_version(),
        ),
        (
            "models.judge",
            &m.manifest.models.judge,
            protocol.judge_prompt_version(),
        ),
    ] {
        if model.prompt_version != expected {
            err(
                &format!("{name}.prompt_version"),
                format!(
                    "{} is not implemented; this build has {expected}",
                    model.prompt_version
                ),
            );
        }
    }
    match load_dataset(m) {
        Err(e) => err("dataset", e.to_string()),
        Ok(ds) => {
            let h = &m.manifest.holdout;
            if let Some(expected) = &h.commitment {
                let actual = holdout::commitment(&ds, &h.salt, h.fraction);
                if &actual != expected {
                    err(
                        "holdout.commitment",
                        format!(
                            "does not match the dataset and salt (recomputed {actual}); the holdout definition changed"
                        ),
                    );
                }
            }
        }
    }
    if let Some(p) = &m.manifest.dataset.subset
        && let Err(e) = m.read_pinned(p)
    {
        err("dataset.subset", e.to_string());
    }
    if let Some(p) = &m.manifest.dataset.annotations {
        match m
            .read_pinned(p)
            .and_then(|b| AnnotationFile::parse(&String::from_utf8(b)?))
        {
            Ok(_) => {}
            Err(e) => err("dataset.annotations", e.to_string()),
        }
    }
    if let Some(p) = &m.manifest.fit_mask {
        match m
            .read_pinned(p)
            .and_then(|b| Ok(serde_json::from_slice::<FitMask>(&b)?))
        {
            Ok(mask) if mask.dataset_hash != m.manifest.dataset.sha256 => {
                err("fit_mask", "computed for a different dataset file".into())
            }
            Ok(_) => {}
            Err(e) => err("fit_mask", e.to_string()),
        }
    }
    issues
}

/// Length-only full-context fit mask over the whole pinned dataset
/// (development and holdout), computed from prompt lengths alone so it can
/// be frozen before any answer is seen.
pub fn compute_fit_mask(m: &LoadedManifest) -> anyhow::Result<FitMask> {
    let full = load_dataset(m)?;
    let a = &m.manifest.models.answer;
    Ok(FitMask::compute(
        &full.source_hash,
        &crate::fitmask::length_views(&full),
        m.manifest.answer_tokenizer().0.as_ref(),
        a.context_limit,
        a.max_output_tokens as usize,
        m.manifest.statistics.min_fit_fraction,
    ))
}
