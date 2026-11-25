//! Paired statistics for comparing arms on the same questions.
//!
//! Questions that share a history are correlated, so the bootstrap resamples
//! whole histories (clusters) with replacement and keeps each question's two
//! arms paired. Seeds are averaged inside each question first, so repeated
//! seeds narrow per-question noise without being treated as independent
//! questions. Multiple comparisons are controlled with Holm's step-down
//! procedure.

use std::collections::BTreeMap;

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

/// One question's paired scores, each already averaged over seeds.
#[derive(Clone, Debug, PartialEq)]
pub struct Paired {
    pub cluster: String,
    pub candidate: f64,
    pub baseline: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bootstrap {
    pub questions: usize,
    pub clusters: usize,
    pub candidate_mean: f64,
    pub baseline_mean: f64,
    /// Candidate minus baseline, as a proportion (0.02 = 2 points).
    pub delta: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub confidence: f64,
    /// One-sided p-value for "delta <= 0", with the +1 correction.
    pub p_one_sided: f64,
    pub resamples: usize,
    pub seed: u64,
    #[serde(skip)]
    pub replicates: Vec<f64>,
}

impl Bootstrap {
    /// One-sided lower bound at significance `alpha` (percentile method).
    pub fn lower_bound(&self, alpha: f64) -> f64 {
        quantile(&self.replicates, alpha)
    }
}

/// Linear-interpolated quantile of sorted data.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    assert!(!sorted.is_empty(), "quantile of empty data");
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

pub fn paired_cluster_bootstrap(
    pairs: &[Paired],
    resamples: usize,
    seed: u64,
    confidence: f64,
) -> anyhow::Result<Bootstrap> {
    anyhow::ensure!(!pairs.is_empty(), "no paired questions to compare");
    anyhow::ensure!(resamples > 0, "resamples must be positive");
    // Per cluster: (sum of candidate - baseline, count).
    let mut by_cluster: BTreeMap<&str, (f64, f64, usize)> = BTreeMap::new();
    for p in pairs {
        let e = by_cluster.entry(p.cluster.as_str()).or_default();
        e.0 += p.candidate;
        e.1 += p.baseline;
        e.2 += 1;
    }
    let clusters: Vec<(f64, f64, usize)> = by_cluster.into_values().collect();
    let n = pairs.len() as f64;
    let candidate_mean = pairs.iter().map(|p| p.candidate).sum::<f64>() / n;
    let baseline_mean = pairs.iter().map(|p| p.baseline).sum::<f64>() / n;
    let delta = candidate_mean - baseline_mean;

    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut replicates = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let (mut d, mut count) = (0.0, 0usize);
        for _ in 0..clusters.len() {
            let (c, b, k) = clusters[rng.random_range(0..clusters.len())];
            d += c - b;
            count += k;
        }
        replicates.push(d / count as f64);
    }
    replicates.sort_by(f64::total_cmp);
    let alpha = 1.0 - confidence;
    let at_or_below = replicates.iter().take_while(|x| **x <= 0.0).count();
    Ok(Bootstrap {
        questions: pairs.len(),
        clusters: clusters.len(),
        candidate_mean,
        baseline_mean,
        delta,
        ci_low: quantile(&replicates, alpha / 2.0),
        ci_high: quantile(&replicates, 1.0 - alpha / 2.0),
        confidence,
        p_one_sided: (at_or_below + 1) as f64 / (resamples + 1) as f64,
        resamples,
        seed,
        replicates,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HolmDecision {
    pub name: String,
    pub p: f64,
    /// Position in ascending p order (0-based).
    pub rank: usize,
    /// Significance level Holm assigns at this rank: alpha / (m - rank).
    pub level: f64,
    pub rejected: bool,
}

/// Holm step-down: sort by p; reject while p <= alpha / (m - i); once one
/// comparison fails, all later ones fail too.
pub fn holm(pvalues: &[(String, f64)], alpha: f64) -> Vec<HolmDecision> {
    let m = pvalues.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|a, b| {
        pvalues[*a]
            .1
            .total_cmp(&pvalues[*b].1)
            .then(pvalues[*a].0.cmp(&pvalues[*b].0))
    });
    let mut still = true;
    let mut out = vec![None; m];
    for (rank, &i) in order.iter().enumerate() {
        let level = alpha / (m - rank) as f64;
        let rejected = still && pvalues[i].1 <= level;
        still = rejected;
        out[i] = Some(HolmDecision {
            name: pvalues[i].0.clone(),
            p: pvalues[i].1,
            rank,
            level,
            rejected,
        });
    }
    out.into_iter()
        .map(|d| d.expect("every index assigned"))
        .collect()
}

/// A candidate-versus-comparator result with its Holm-adjusted lower bound.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub candidate: String,
    pub comparator: String,
    pub bootstrap: Bootstrap,
    pub holm: HolmDecision,
    /// One-sided lower bound at the Holm level for this comparison.
    pub adjusted_lower_bound: f64,
}

/// Run every comparison and attach Holm-adjusted lower bounds, with one Holm
/// family for these comparisons alone.
pub fn compare_all(
    candidate: &str,
    comparisons: Vec<(String, Vec<Paired>)>,
    resamples: usize,
    seed: u64,
    confidence: f64,
) -> anyhow::Result<Vec<Comparison>> {
    let mut groups = compare_family(
        candidate,
        vec![(String::new(), comparisons)],
        resamples,
        seed,
        confidence,
    )?;
    Ok(groups.pop().unwrap_or_default())
}

/// Named paired outcomes for each comparator, grouped by storage profile.
pub type ComparisonGroups = Vec<(String, Vec<(String, Vec<Paired>)>)>;
/// Bootstrapped comparisons, grouped by storage profile.
pub type BootstrapGroups = Vec<(String, Vec<(String, Bootstrap)>)>;

/// Run every comparison of every group (one group per storage profile) and
/// adjust them all as **one** Holm family, as G2 preregisters: with two
/// comparators in two profiles the family has four tests. Returns the
/// comparisons per group, in input order.
pub fn compare_family(
    candidate: &str,
    groups: ComparisonGroups,
    resamples: usize,
    seed: u64,
    confidence: f64,
) -> anyhow::Result<Vec<Vec<Comparison>>> {
    let boots = groups
        .into_iter()
        .map(|(group, comparisons)| {
            let boots = comparisons
                .into_iter()
                .map(|(name, pairs)| {
                    Ok((
                        name,
                        paired_cluster_bootstrap(&pairs, resamples, seed, confidence)?,
                    ))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok((group, boots))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(adjust_family(candidate, boots, confidence))
}

/// Holm-adjust already bootstrapped comparisons as one family across all
/// groups. A test is named `group/comparator` (just `comparator` when the
/// group name is empty).
pub fn adjust_family(
    candidate: &str,
    groups: BootstrapGroups,
    confidence: f64,
) -> Vec<Vec<Comparison>> {
    let ps: Vec<(String, f64)> = groups
        .iter()
        .flat_map(|(group, boots)| {
            boots.iter().map(move |(name, b)| {
                let label = if group.is_empty() {
                    name.clone()
                } else {
                    format!("{group}/{name}")
                };
                (label, b.p_one_sided)
            })
        })
        .collect();
    let mut decisions = holm(&ps, 1.0 - confidence).into_iter();
    groups
        .into_iter()
        .map(|(_, boots)| {
            boots
                .into_iter()
                .map(|(name, b)| {
                    let h = decisions.next().expect("one decision per test");
                    Comparison {
                        candidate: candidate.into(),
                        comparator: name,
                        adjusted_lower_bound: b.lower_bound(h.level),
                        bootstrap: b,
                        holm: h,
                    }
                })
                .collect()
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Check {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

/// Superiority criteria for one storage profile's comparisons.
pub fn superiority_checks(comparisons: &[Comparison], min_gain_pp: f64) -> Vec<Check> {
    comparisons
        .iter()
        .map(|c| {
            let gain_pp = c.bootstrap.delta * 100.0;
            // Holm is step-down: once a smaller p fails, every later test fails.
            let passed =
                gain_pp >= min_gain_pp && c.holm.rejected && c.adjusted_lower_bound > 0.0;
            Check {
                name: format!("{} beats {}", c.candidate, c.comparator),
                passed,
                detail: format!(
                    "gain {gain_pp:+.2} pp (need >= {min_gain_pp}), Holm-adjusted lower bound {:+.2} pp at level {:.4} \
                     (need > 0), Holm {} (p {:.4}, rank {}), {} questions in {} histories",
                    c.adjusted_lower_bound * 100.0,
                    c.holm.level,
                    if c.holm.rejected { "rejects" } else { "does not reject" },
                    c.holm.p,
                    c.holm.rank,
                    c.bootstrap.questions,
                    c.bootstrap.clusters
                ),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(n_clusters: usize, per: usize, f: impl Fn(usize, usize) -> (f64, f64)) -> Vec<Paired> {
        (0..n_clusters)
            .flat_map(|c| {
                let f = &f;
                (0..per).map(move |i| {
                    let (a, b) = f(c, i);
                    Paired {
                        cluster: format!("h{c}"),
                        candidate: a,
                        baseline: b,
                    }
                })
            })
            .collect()
    }

    #[test]
    fn identical_arms_give_zero_delta_and_no_evidence() {
        let p = pairs(30, 5, |c, i| {
            let v = f64::from(u8::from((c + i) % 3 == 0));
            (v, v)
        });
        let b = paired_cluster_bootstrap(&p, 2000, 7, 0.95).unwrap();
        assert_eq!(b.delta, 0.0);
        assert_eq!((b.ci_low, b.ci_high), (0.0, 0.0));
        assert!(b.p_one_sided > 0.99);
    }

    #[test]
    fn consistent_gain_is_detected_and_ci_brackets_the_estimate() {
        // Candidate wins one extra question in most histories.
        let p = pairs(40, 5, |c, i| {
            let base = f64::from(u8::from(i < 2));
            let cand = if i == 2 && c % 5 != 0 { 1.0 } else { base };
            (cand, base)
        });
        let b = paired_cluster_bootstrap(&p, 4000, 11, 0.95).unwrap();
        assert!((b.delta - 0.16).abs() < 1e-9);
        assert!(b.ci_low > 0.0 && b.ci_low < b.delta && b.delta < b.ci_high);
        assert!(b.p_one_sided < 0.001);
        assert_eq!(b.clusters, 40);
        // Deterministic for a fixed seed.
        let again = paired_cluster_bootstrap(&p, 4000, 11, 0.95).unwrap();
        assert_eq!(b.ci_low, again.ci_low);
    }

    #[test]
    fn clustering_widens_intervals_for_correlated_questions() {
        // Whole histories either all win or all tie: few effective samples.
        let correlated = pairs(10, 20, |c, _| if c < 3 { (1.0, 0.0) } else { (0.0, 0.0) });
        let b = paired_cluster_bootstrap(&correlated, 4000, 3, 0.95).unwrap();
        // An iid bootstrap over 200 questions would put the lower bound near
        // 0.24; resampling 10 histories must be far wider.
        assert!(b.ci_low < 0.1, "ci_low {}", b.ci_low);
    }

    /// A bootstrap whose replicates put `share_at_or_below_zero` of the
    /// mass at or below zero, with a 3-point mean gain.
    fn boot(share_at_or_below_zero: f64) -> Bootstrap {
        let n = 10_000;
        let low = (share_at_or_below_zero * n as f64).round() as usize;
        let replicates: Vec<f64> = (0..n).map(|i| if i < low { -0.01 } else { 0.03 }).collect();
        Bootstrap {
            questions: 300,
            clusters: 30,
            candidate_mean: 0.6,
            baseline_mean: 0.57,
            delta: 0.03,
            ci_low: 0.0,
            ci_high: 0.03,
            confidence: 0.95,
            p_one_sided: (low + 1) as f64 / (n + 1) as f64,
            resamples: n,
            seed: 1,
            replicates,
        }
    }

    #[test]
    fn one_holm_family_spans_every_comparator_in_every_profile() {
        // p = 0.02 for each of 2 comparators in 2 profiles.
        let group = |name: &str| {
            (
                name.to_string(),
                vec![
                    ("full".to_string(), boot(0.02)),
                    ("dense".to_string(), boot(0.02)),
                ],
            )
        };
        // Adjusted per profile (families of 2), every test passes ...
        for g in [group("a"), group("b")] {
            let per_profile = adjust_family("cand", vec![g], 0.95);
            let first = per_profile[0].iter().find(|c| c.holm.rank == 0).unwrap();
            assert!((first.holm.level - 0.05 / 2.0).abs() < 1e-12);
            assert!(
                superiority_checks(&per_profile[0], 2.0)
                    .iter()
                    .all(|c| c.passed)
            );
        }
        // ... but the preregistered family of four fails them all.
        let family = adjust_family("cand", vec![group("a"), group("b")], 0.95);
        let decisions: Vec<&HolmDecision> = family.iter().flatten().map(|c| &c.holm).collect();
        assert_eq!(decisions.len(), 4);
        let mut ranks: Vec<usize> = decisions.iter().map(|d| d.rank).collect();
        ranks.sort();
        assert_eq!(ranks, [0, 1, 2, 3]);
        let first = decisions.iter().find(|d| d.rank == 0).unwrap();
        assert!((first.level - 0.05 / 4.0).abs() < 1e-12, "{}", first.level);
        assert!(decisions.iter().any(|d| d.name == "b/dense"));
        for comparisons in &family {
            assert!(
                superiority_checks(comparisons, 2.0)
                    .iter()
                    .all(|c| !c.passed)
            );
        }
    }

    #[test]
    fn holm_steps_down_and_stops_at_first_failure() {
        let ps = vec![
            ("a".to_string(), 0.01),
            ("b".to_string(), 0.04),
            ("c".to_string(), 0.03),
        ];
        let d = holm(&ps, 0.05);
        // Sorted: a 0.01 <= 0.0167 yes; c 0.03 <= 0.025 no; b stops.
        assert!(d[0].rejected);
        assert!(!d[2].rejected && (d[2].level - 0.025).abs() < 1e-12);
        assert!(!d[1].rejected && (d[1].level - 0.05).abs() < 1e-12);
        let all = holm(&[("x".into(), 0.001), ("y".into(), 0.02)], 0.05);
        assert!(all.iter().all(|d| d.rejected));
    }

    #[test]
    fn superiority_needs_both_gain_and_adjusted_bound() {
        let strong = pairs(40, 5, |c, i| {
            (
                f64::from(u8::from(i < 3 || c % 2 == 0)),
                f64::from(u8::from(i < 2)),
            )
        });
        let tiny = pairs(40, 50, |c, i| {
            (
                f64::from(u8::from(i < 25 || (i == 25 && c % 4 == 0))),
                f64::from(u8::from(i < 25)),
            )
        });
        let cmp = compare_all(
            "cand",
            vec![("strong".into(), strong), ("tiny".into(), tiny)],
            4000,
            5,
            0.95,
        )
        .unwrap();
        let checks = superiority_checks(&cmp, 2.0);
        assert!(checks[0].passed, "{}", checks[0].detail);
        // Gain of 0.5 pp is real but below the 2 pp minimum.
        assert!(!checks[1].passed, "{}", checks[1].detail);
        assert!(cmp[1].adjusted_lower_bound > 0.0);
    }
}
