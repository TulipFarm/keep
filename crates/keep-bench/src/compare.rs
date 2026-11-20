//! Compare a candidate arm with comparator arms from saved runs and evaluate
//! the superiority gate.
//!
//! Comparisons are paired by question id. Against the full-context arm, only
//! questions that fit full context are compared: the frozen fit mask decides
//! when one is pinned; otherwise the questions where the full-context arm
//! was available. Records are grouped by storage profile and the gate needs
//! every required profile to pass on its own.
//!
//! Non-canonical runs (Copilot CLI or other development-track runs) are
//! refused unless the caller opts in with [`CompareOptions::allow_non_canonical`].
//! An opted-in report is stamped [`INTERNAL_NOTICE`] in every output and
//! never passes the gate.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::artifacts::{Outcome, QuestionRecord, read_records};
use crate::manifest::{Manifest, ModelSpec, Phase, StatisticsSpec, StorageProfile};
use crate::runner::RunMeta;
use crate::stats::{Check, Comparison, Paired, compare_family, superiority_checks};

pub struct LoadedRun {
    pub dir: PathBuf,
    pub meta: RunMeta,
    pub manifest: Manifest,
    pub records: Vec<QuestionRecord>,
}

pub fn load_run(dir: &Path) -> anyhow::Result<LoadedRun> {
    let meta: RunMeta = serde_json::from_str(&std::fs::read_to_string(dir.join("run.json"))?)?;
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    anyhow::ensure!(
        manifest.hash() == meta.manifest_hash,
        "{}: manifest.json does not match the hash recorded in run.json",
        dir.display()
    );
    let records = read_records(&dir.join("records.jsonl"))?;
    anyhow::ensure!(
        records
            .iter()
            .all(|r| r.manifest_hash == meta.manifest_hash),
        "{}: records carry a different manifest hash",
        dir.display()
    );
    Ok(LoadedRun {
        dir: dir.to_owned(),
        meta,
        manifest,
        records,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileResult {
    pub storage_profile: StorageProfile,
    pub runs: Vec<String>,
    pub comparisons: Vec<Comparison>,
    pub unpaired: BTreeMap<String, usize>,
    pub checks: Vec<Check>,
}

/// Heads every report built from non-canonical runs.
pub const INTERNAL_NOTICE: &str = "INTERNAL / NON-CANONICAL: internal comparison of non-canonical runs; not comparable to published numbers and never a gate pass";

/// Options for [`evaluate_with`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CompareOptions {
    /// Let non-canonical runs into the comparison. Every run must then be
    /// non-canonical, and the report is stamped [`INTERNAL_NOTICE`].
    pub allow_non_canonical: bool,
}

/// The stamp on a report built from non-canonical runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InternalStamp {
    pub notice: String,
    /// The distinct non-canonical labels carried by the runs.
    pub labels: Vec<String>,
    pub answer_model: String,
    pub judge_model: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateReport {
    /// Present only for internal comparisons of non-canonical runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub internal_non_canonical: Option<InternalStamp>,
    pub candidate: String,
    pub comparators: Vec<String>,
    pub statistics: StatisticsSpec,
    pub profiles: Vec<ProfileResult>,
    pub checks: Vec<Check>,
    pub passed: bool,
}

fn seed_mean(records: &[&QuestionRecord]) -> HashMap<String, (String, f64, usize)> {
    let mut m: HashMap<String, (String, f64, usize)> = HashMap::new();
    for r in records {
        let e = m
            .entry(r.question_id.clone())
            .or_insert((r.history_id.clone(), 0.0, 0));
        e.1 += f64::from(u8::from(r.correct()));
        e.2 += 1;
    }
    m
}

fn usable(r: &QuestionRecord) -> bool {
    !matches!(
        r.outcome,
        Outcome::Excluded { .. } | Outcome::Unavailable { .. } | Outcome::NotRun { .. }
    )
}

/// Pair the candidate with one comparator. Returns the pairs and how many
/// candidate questions had no comparator partner.
pub fn pair(
    records: &[&QuestionRecord],
    candidate: &str,
    comparator: &str,
) -> anyhow::Result<(Vec<Paired>, usize)> {
    let cand: Vec<&QuestionRecord> = records
        .iter()
        .copied()
        .filter(|r| r.arm == candidate)
        .collect();
    anyhow::ensure!(!cand.is_empty(), "no records for candidate arm {candidate}");
    anyhow::ensure!(
        !cand
            .iter()
            .any(|r| matches!(r.outcome, Outcome::NotRun { .. } | Outcome::Estimated)),
        "candidate arm {candidate} was not executed"
    );
    let comp_all: Vec<&QuestionRecord> = records
        .iter()
        .copied()
        .filter(|r| r.arm == comparator)
        .collect();
    anyhow::ensure!(
        !comp_all.is_empty(),
        "no records for comparator arm {comparator}"
    );
    let full_context = comp_all[0].arm_kind == "full-context";
    let comp: Vec<&QuestionRecord> = comp_all
        .iter()
        .copied()
        .filter(|r| {
            if full_context {
                match r.fits_full_context {
                    Some(fits) => fits && !matches!(r.outcome, Outcome::Excluded { .. }),
                    None => usable(r),
                }
            } else {
                usable(r)
            }
        })
        .collect();
    // Candidate failures and unavailability count as incorrect; exclusions
    // drop the question from both sides.
    let cand: Vec<&QuestionRecord> = cand
        .into_iter()
        .filter(|r| !matches!(r.outcome, Outcome::Excluded { .. }))
        .collect();
    let c = seed_mean(&cand);
    let b = seed_mean(&comp);
    let mut pairs = Vec::new();
    let mut unpaired = 0;
    let mut ids: Vec<&String> = c.keys().collect();
    ids.sort();
    for id in ids {
        let (cluster, cs, cn) = &c[id];
        match b.get(id) {
            Some((_, bs, bn)) => pairs.push(Paired {
                cluster: cluster.clone(),
                candidate: cs / *cn as f64,
                baseline: bs / *bn as f64,
            }),
            None => unpaired += 1,
        }
    }
    Ok((pairs, unpaired))
}

fn completion(records: &[&QuestionRecord], arm: &str) -> Option<f64> {
    let rs: Vec<_> = records
        .iter()
        .filter(|r| r.arm == arm && usable(r))
        .collect();
    (!rs.is_empty()).then(|| {
        rs.iter().filter(|r| r.outcome == Outcome::Answered).count() as f64 / rs.len() as f64
    })
}

/// Provider, model, revision, reasoning effort and prompt version: what
/// must match for two runs' accuracies to be comparable.
pub fn model_identity(m: &ModelSpec) -> String {
    let provider = serde_json::to_value(m.provider)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default();
    let effort = m
        .copilot
        .as_ref()
        .and_then(|c| c.reasoning_effort.as_deref())
        .or(m.reasoning_effort.as_deref())
        .map(|e| format!(", reasoning effort {e}"))
        .unwrap_or_default();
    format!(
        "{provider} {} ({}{effort}, prompt {})",
        m.model, m.revision, m.prompt_version
    )
}

fn non_canonical_label(r: &LoadedRun) -> Option<&str> {
    r.meta
        .non_canonical
        .as_deref()
        .or(r.manifest.non_canonical_label())
}

/// [`evaluate_with`] with default options: non-canonical runs are refused.
pub fn evaluate(
    runs: &[LoadedRun],
    candidate: &str,
    comparators: &[String],
    required_profiles: &[StorageProfile],
) -> anyhow::Result<GateReport> {
    evaluate_with(
        runs,
        candidate,
        comparators,
        required_profiles,
        CompareOptions::default(),
    )
}

pub fn evaluate_with(
    runs: &[LoadedRun],
    candidate: &str,
    comparators: &[String],
    required_profiles: &[StorageProfile],
    options: CompareOptions,
) -> anyhow::Result<GateReport> {
    anyhow::ensure!(!runs.is_empty(), "no runs given");
    anyhow::ensure!(!comparators.is_empty(), "name at least one comparator arm");
    let stats = runs[0].manifest.statistics.clone();
    let dataset_hash = &runs[0].meta.dataset_hash;
    let answer_model = model_identity(&runs[0].manifest.models.answer);
    let judge_model = model_identity(&runs[0].manifest.models.judge);
    let mut labels: Vec<String> = Vec::new();
    for r in runs {
        if let Some(reason) = &r.meta.aborted {
            anyhow::bail!(
                "{} was aborted and cannot enter a comparison or gate ({reason})",
                r.dir.display()
            );
        }
        match non_canonical_label(r) {
            Some(label) if !options.allow_non_canonical => anyhow::bail!(
                "{} is a non-canonical run and cannot enter a comparison or gate ({label}); \
                 pass --allow-non-canonical for an internal, non-publishable comparison",
                r.dir.display()
            ),
            Some(label) => {
                if !labels.iter().any(|l| l == label) {
                    labels.push(label.to_owned());
                }
            }
            None if options.allow_non_canonical => anyhow::bail!(
                "{} is a canonical run; an internal comparison takes non-canonical runs only, \
                 so canonical and non-canonical numbers are never mixed",
                r.dir.display()
            ),
            None => {}
        }
        for (role, expected, actual) in [
            (
                "answer",
                &answer_model,
                model_identity(&r.manifest.models.answer),
            ),
            (
                "judge",
                &judge_model,
                model_identity(&r.manifest.models.judge),
            ),
        ] {
            anyhow::ensure!(
                *expected == actual,
                "{} uses a different {role} model ({actual}, expected {expected}); runs are not comparable",
                r.dir.display()
            );
        }
        anyhow::ensure!(
            &r.meta.dataset_hash == dataset_hash,
            "{} uses a different dataset file; runs are not comparable",
            r.dir.display()
        );
        anyhow::ensure!(
            r.manifest.protocol == runs[0].manifest.protocol,
            "{} is scored under a different protocol; runs are not comparable",
            r.dir.display()
        );
        anyhow::ensure!(
            r.manifest.statistics == stats,
            "{} preregistered different statistics settings",
            r.dir.display()
        );
    }
    let alpha_conf = stats.confidence;
    let mut profiles: BTreeMap<String, Vec<&LoadedRun>> = BTreeMap::new();
    for r in runs {
        let key = serde_json::to_value(r.meta.storage_profile)?
            .as_str()
            .unwrap_or_default()
            .to_owned();
        profiles.entry(key).or_default().push(r);
    }

    let mut results = Vec::new();
    let mut global = Vec::new();
    // Every comparator in every storage profile is one superiority test, and
    // all of them form a single Holm family (G2: 2 comparators x 2 profiles
    // = 4 tests). A single-profile comparison is a family of its comparators.
    let mut families = Vec::new();
    let mut unpaired_by_profile = Vec::new();
    for (key, group) in &profiles {
        let records: Vec<&QuestionRecord> = group.iter().flat_map(|r| r.records.iter()).collect();
        let mut comps = Vec::new();
        let mut unpaired = BTreeMap::new();
        for c in comparators {
            let (pairs, missing) = pair(&records, candidate, c)?;
            unpaired.insert(c.clone(), missing);
            comps.push((c.clone(), pairs));
        }
        let label = if profiles.len() > 1 {
            key.clone()
        } else {
            String::new()
        };
        families.push((label, comps));
        unpaired_by_profile.push(unpaired);
    }
    let adjusted = compare_family(
        candidate,
        families,
        stats.bootstrap_resamples,
        stats.bootstrap_seed,
        alpha_conf,
    )?;
    for ((group, comparisons), unpaired) in profiles.values().zip(adjusted).zip(unpaired_by_profile)
    {
        let records: Vec<&QuestionRecord> = group.iter().flat_map(|r| r.records.iter()).collect();
        let mut checks = superiority_checks(&comparisons, stats.min_gain_pp);
        let comp = completion(&records, candidate);
        checks.push(Check {
            name: "answer completion".into(),
            passed: comp.is_some_and(|c| c >= stats.min_completion),
            detail: format!(
                "{} answered (need >= {:.1}%)",
                comp.map(|c| format!("{:.2}%", c * 100.0))
                    .unwrap_or_else(|| "none".into()),
                stats.min_completion * 100.0
            ),
        });
        let seeds: BTreeSet<u64> = records
            .iter()
            .filter(|r| r.arm == candidate)
            .map(|r| r.seed)
            .collect();
        checks.push(Check {
            name: "seeds".into(),
            passed: seeds.len() >= 3,
            detail: format!("{} distinct seeds (need >= 3)", seeds.len()),
        });
        results.push(ProfileResult {
            storage_profile: group[0].meta.storage_profile,
            runs: group.iter().map(|r| r.meta.run_id.clone()).collect(),
            comparisons,
            unpaired,
            checks,
        });
    }

    for p in required_profiles {
        let found = results.iter().find(|r| r.storage_profile == *p);
        global.push(Check {
            name: format!(
                "storage profile {}",
                serde_json::to_value(p)?.as_str().unwrap_or_default()
            ),
            passed: found.is_some_and(|r| r.checks.iter().all(|c| c.passed)),
            detail: match found {
                None => "no runs on this profile".into(),
                Some(r) if r.checks.iter().all(|c| c.passed) => "all criteria pass".into(),
                Some(_) => "some criteria fail".into(),
            },
        });
    }
    let fit: Vec<Option<f64>> = runs
        .iter()
        .map(|r| r.meta.fit_mask.as_ref().map(|f| f.fit_fraction))
        .collect();
    let min_fit = fit
        .iter()
        .copied()
        .map(|f| f.unwrap_or(0.0))
        .fold(f64::INFINITY, f64::min);
    let all_exact = runs
        .iter()
        .all(|r| r.meta.fit_mask.as_ref().is_some_and(|f| f.exact()));
    let by_tokenizer = runs
        .iter()
        .all(|r| r.meta.fit_mask.as_ref().is_some_and(|f| f.tokenizer_exact));
    global.push(Check {
        name: "full-context fit".into(),
        passed: fit.iter().all(Option::is_some) && min_fit >= stats.min_fit_fraction && all_exact,
        detail: if fit.iter().any(Option::is_none) {
            "a run has no frozen fit mask".into()
        } else {
            format!(
                "lowest fit fraction {:.1}% (need >= {:.0}%), tokenizer {}",
                min_fit * 100.0,
                stats.min_fit_fraction * 100.0,
                if by_tokenizer {
                    "exact"
                } else if all_exact {
                    "estimate, every decision verified against provider-reported prompt tokens"
                } else {
                    "approximate and not verified against provider-reported counts (not gate-eligible)"
                }
            )
        },
    });
    global.push(Check {
        name: "real providers".into(),
        passed: runs.iter().all(|r| !r.meta.mock),
        detail: if runs.iter().any(|r| r.meta.mock) {
            "mock providers were used; results are plumbing checks".into()
        } else {
            "no mock providers".into()
        },
    });
    global.push(Check {
        name: "confirmation phase".into(),
        passed: runs.iter().all(|r| r.meta.phase == Phase::Confirmation),
        detail: "superiority is claimed only from sealed-holdout confirmation runs".into(),
    });
    let internal = options.allow_non_canonical.then(|| InternalStamp {
        notice: INTERNAL_NOTICE.into(),
        labels: labels.clone(),
        answer_model,
        judge_model,
    });
    if internal.is_some() {
        global.push(Check {
            name: "canonical runs".into(),
            passed: false,
            detail: format!(
                "internal comparison of non-canonical runs ({}); never a gate pass",
                labels.join("; ")
            ),
        });
    }
    let passed = global.iter().all(|c| c.passed);
    Ok(GateReport {
        internal_non_canonical: internal,
        candidate: candidate.into(),
        comparators: comparators.to_vec(),
        statistics: stats,
        profiles: results,
        checks: global,
        passed,
    })
}

pub fn markdown(g: &GateReport) -> String {
    let mut out = String::new();
    if let Some(s) = &g.internal_non_canonical {
        out.push_str(&format!(
            "> **{}**\n>\n> Runs: {}\n>\n> Answer model: {}; judge model: {}\n\n",
            s.notice,
            s.labels.join("; "),
            s.answer_model,
            s.judge_model
        ));
    }
    out.push_str(&format!(
        "# {}Comparison: {} vs {}\n\nMethod: {}, {} resamples (seed {}), {:.0}% confidence, one Holm family of {} tests across every comparator and storage profile.\n\n",
        if g.internal_non_canonical.is_some() {
            "INTERNAL / NON-CANONICAL "
        } else {
            ""
        },
        g.candidate,
        g.comparators.join(", "),
        g.statistics.method,
        g.statistics.bootstrap_resamples,
        g.statistics.bootstrap_seed,
        g.statistics.confidence * 100.0,
        g.profiles.iter().map(|p| p.comparisons.len()).sum::<usize>()
    ));
    for p in &g.profiles {
        out.push_str(&format!(
            "## Storage profile {}\n\n",
            p.storage_profile.as_str()
        ));
        out.push_str("| Comparator | Questions | Histories | Candidate | Comparator | Delta pp | CI pp | p (one-sided) | Holm level | Adjusted LB pp | Unpaired |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
        for c in &p.comparisons {
            let b = &c.bootstrap;
            out.push_str(&format!(
                "| {} | {} | {} | {:.1}% | {:.1}% | {:+.2} | [{:+.2}, {:+.2}] | {:.4} | {:.4} | {:+.2} | {} |\n",
                c.comparator,
                b.questions,
                b.clusters,
                b.candidate_mean * 100.0,
                b.baseline_mean * 100.0,
                b.delta * 100.0,
                b.ci_low * 100.0,
                b.ci_high * 100.0,
                b.p_one_sided,
                c.holm.level,
                c.adjusted_lower_bound * 100.0,
                p.unpaired.get(&c.comparator).copied().unwrap_or(0)
            ));
        }
        out.push('\n');
        for c in &p.checks {
            out.push_str(&format!(
                "- [{}] {}: {}\n",
                if c.passed { "x" } else { " " },
                c.name,
                c.detail
            ));
        }
        out.push('\n');
    }
    out.push_str("## Gate\n\n");
    for c in &g.checks {
        out.push_str(&format!(
            "- [{}] {}: {}\n",
            if c.passed { "x" } else { " " },
            c.name,
            c.detail
        ));
    }
    out.push_str(&format!(
        "\n**Result: {}**\n",
        if g.passed { "PASS" } else { "NOT PASSED" }
    ));
    if let Some(s) = &g.internal_non_canonical {
        out.push_str(&format!("\n> **{}**\n", s.notice));
    }
    out
}
