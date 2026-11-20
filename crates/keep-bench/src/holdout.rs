//! Sealed holdout.
//!
//! Each independent history is assigned to the development or holdout split
//! by hashing a preregistered salt with its id, so membership is fixed before
//! anyone looks at results. The manifest records a commitment (the hash of
//! the sorted holdout ids); validation recomputes it, so a changed dataset,
//! salt or fraction is caught. Development runs drop holdout histories and
//! their questions before anything else happens. Confirmation runs need an
//! explicit unseal flag and a named frozen candidate.

use keep_core::{ContentHash, sha256_hex};

use crate::dataset::Dataset;
use crate::manifest::{Manifest, Phase};

/// True when a history belongs to the sealed holdout.
pub fn in_holdout(salt: &str, fraction: f64, history_id: &str) -> bool {
    let digest = sha256_hex(format!("{salt}\u{0}{history_id}").as_bytes());
    let bucket = u64::from_str_radix(&digest[..16], 16).expect("hex digest");
    (bucket as f64 / u64::MAX as f64) < fraction
}

pub fn holdout_ids(dataset: &Dataset, salt: &str, fraction: f64) -> Vec<String> {
    let mut ids: Vec<String> = dataset
        .histories
        .iter()
        .filter(|h| in_holdout(salt, fraction, &h.id))
        .map(|h| h.id.clone())
        .collect();
    ids.sort();
    ids
}

pub fn commitment(dataset: &Dataset, salt: &str, fraction: f64) -> ContentHash {
    let ids = holdout_ids(dataset, salt, fraction);
    let body = format!(
        "keep-holdout-v1\ndataset={}\nsalt={salt}\nfraction={fraction}\n{}",
        dataset.source_hash,
        ids.join("\n")
    );
    ContentHash::of_bytes(body.as_bytes())
}

#[derive(Debug, thiserror::Error)]
pub enum SealError {
    #[error("holdout is not sealed: the manifest has no holdout.commitment")]
    Unsealed,
    #[error("holdout commitment mismatch: manifest has {expected}, dataset gives {actual}")]
    Mismatch {
        expected: ContentHash,
        actual: ContentHash,
    },
    #[error("confirmation runs read the sealed holdout; pass --unseal to confirm this is intended")]
    NotUnsealed,
    #[error("--unseal is only valid for confirmation-phase manifests")]
    UnsealInDevelopment,
    #[error(
        "this manifest's protocol uses split = \"full\", which scores the sealed holdout; pass --consume-holdout only with explicit approval to spend it"
    )]
    FullNotConsumed,
    #[error("holdout split leaves no {0} histories")]
    Empty(&'static str),
}

/// Which histories a run may touch.
#[derive(Debug)]
pub struct Split {
    pub phase: Phase,
    pub history_ids: Vec<String>,
    pub holdout_count: usize,
    pub development_count: usize,
    /// A full-set parity run: holdout histories are included.
    pub consumes_holdout: bool,
}

/// Apply the sealed split to a dataset in place, keeping only the histories
/// and questions this phase may use.
pub fn apply(manifest: &Manifest, dataset: &mut Dataset, unseal: bool) -> Result<Split, SealError> {
    apply_scoped(manifest, dataset, unseal, false)
}

/// [`apply`], honouring the protocol's split scope. A `full` split keeps
/// every history, holdout included, and is refused unless
/// `consume_holdout` is set.
pub fn apply_scoped(
    manifest: &Manifest,
    dataset: &mut Dataset,
    unseal: bool,
    consume_holdout: bool,
) -> Result<Split, SealError> {
    let h = &manifest.holdout;
    let expected = h.commitment.clone().ok_or(SealError::Unsealed)?;
    let actual = commitment(dataset, &h.salt, h.fraction);
    if expected != actual {
        return Err(SealError::Mismatch { expected, actual });
    }
    let holdout: std::collections::HashSet<String> = holdout_ids(dataset, &h.salt, h.fraction)
        .into_iter()
        .collect();
    let total = dataset.histories.len();
    if manifest.protocol.split == crate::protocol::SplitScope::Full {
        if unseal {
            return Err(SealError::UnsealInDevelopment);
        }
        if !consume_holdout {
            return Err(SealError::FullNotConsumed);
        }
        return Ok(Split {
            phase: manifest.phase,
            history_ids: dataset.histories.iter().map(|x| x.id.clone()).collect(),
            holdout_count: holdout.len(),
            development_count: total - holdout.len(),
            consumes_holdout: true,
        });
    }
    let keep_holdout = match manifest.phase {
        Phase::Development if unseal => return Err(SealError::UnsealInDevelopment),
        Phase::Development => false,
        Phase::Confirmation => {
            if !unseal {
                return Err(SealError::NotUnsealed);
            }
            true
        }
    };
    dataset
        .histories
        .retain(|x| holdout.contains(&x.id) == keep_holdout);
    dataset
        .questions
        .retain(|q| holdout.contains(&q.history_id) == keep_holdout);
    if dataset.histories.is_empty() {
        return Err(SealError::Empty(if keep_holdout {
            "holdout"
        } else {
            "development"
        }));
    }
    Ok(Split {
        phase: manifest.phase,
        history_ids: dataset.histories.iter().map(|x| x.id.clone()).collect(),
        holdout_count: holdout.len(),
        development_count: total - holdout.len(),
        consumes_holdout: false,
    })
}
