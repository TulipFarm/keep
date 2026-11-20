//! Reciprocal rank fusion.
//!
//! Lexical and vector scores live on different scales, so instead of mixing
//! scores we mix ranks: each list contributes `1 / (k + rank)` for every id
//! it contains (rank starting at 1). Ids found by both lists rise to the top.

/// The constant from the original RRF paper; large enough that the first
/// few ranks of each list are weighted similarly.
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Fuse ranked id lists. Duplicates within one list count once, at their
/// best rank. Ties break by id so output is deterministic.
pub fn rrf(lists: &[Vec<String>], k: f32) -> Vec<(String, f32)> {
    let mut scores: std::collections::BTreeMap<String, f32> = Default::default();
    for list in lists {
        let mut seen = std::collections::HashSet::new();
        for (i, id) in list.iter().enumerate() {
            if seen.insert(id) {
                *scores.entry(id.clone()).or_default() += 1.0 / (k + (i as f32) + 1.0);
            }
        }
    }
    let mut out: Vec<(String, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn agreement_beats_a_single_first_place() {
        let fused = rrf(&[ids(&["a", "b", "c"]), ids(&["d", "b", "a"])], 60.0);
        let order: Vec<&str> = fused.iter().map(|x| x.0.as_str()).collect();
        // a: 1/61 + 1/63, b: 2/62, d: 1/61, c: 1/63
        assert_eq!(order, ["a", "b", "d", "c"]);
        assert!((fused[0].1 - (1.0 / 61.0 + 1.0 / 63.0)).abs() < 1e-6);
    }

    #[test]
    fn single_list_keeps_its_order_and_duplicates_count_once() {
        let fused = rrf(&[ids(&["x", "y", "x"]), vec![]], 60.0);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, "x");
        assert!((fused[0].1 - 1.0 / 61.0).abs() < 1e-6);
    }

    #[test]
    fn ties_break_by_id() {
        let fused = rrf(&[ids(&["b"]), ids(&["a"])], 60.0);
        assert_eq!(fused[0].0, "a");
    }
}
