//! Bi-temporal reads.
//!
//! Two clocks are kept apart. *World time* (`valid_from`/`valid_to`) says when
//! a statement was true. *System time* (`recorded_at`/`invalidated_at`) says
//! when the ledger believed it. "What did we know on 1 March?" fixes system
//! time; "where did Mira live on 1 March?" fixes world time. Both bounds are
//! half-open: `from <= t < to`.

use serde::{Deserialize, Serialize};

use crate::model::{Chunk, Evidence, Fact, FactVersion, Time, TimeBasis};

/// A point-in-time read. `None` means "now" for `known_at` and "any time"
/// for `valid_at`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsOf {
    /// System time: only what the ledger had recorded by then.
    pub known_at: Option<Time>,
    /// World time: only statements true at that instant.
    pub valid_at: Option<Time>,
}

impl AsOf {
    pub fn now() -> Self {
        Self::default()
    }
    pub fn known_at(t: Time) -> Self {
        Self {
            known_at: Some(t),
            valid_at: None,
        }
    }
    pub fn valid_at(t: Time) -> Self {
        Self {
            known_at: None,
            valid_at: Some(t),
        }
    }
    pub fn both(known: Time, valid: Time) -> Self {
        Self {
            known_at: Some(known),
            valid_at: Some(valid),
        }
    }
}

/// The version of a fact the ledger held at `known_at`.
pub fn version_known_at(fact: &Fact, known_at: Option<Time>) -> Option<&FactVersion> {
    match known_at {
        None => fact.current(),
        Some(t) => fact
            .versions
            .iter()
            .rev()
            .find(|v| v.recorded_at <= t && v.invalidated_at.is_none_or(|end| t < end)),
    }
}

/// Whether a version's world-time window contains `t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidMatch {
    Inside,
    /// Not closed before `t`, but the start is unknown or only observed
    /// later than `t`. Returned with a label rather than hidden.
    StartUnknown,
    Outside,
}

pub fn valid_match(v: &FactVersion, t: Time) -> ValidMatch {
    if v.valid_to.is_some_and(|end| end <= t) {
        return ValidMatch::Outside;
    }
    match (v.valid_from, v.valid_from_basis) {
        (None, _) | (_, TimeBasis::Unknown) => ValidMatch::StartUnknown,
        (Some(from), _) if from <= t => ValidMatch::Inside,
        // Explicit start after t: definitely not yet true.
        (Some(_), TimeBasis::Explicit) => ValidMatch::Outside,
        // Observed later than t: we only know it was true by then.
        (Some(_), _) => ValidMatch::Outside,
    }
}

/// A fact as seen by one read.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FactAsOf {
    pub version: FactVersion,
    pub valid: Option<ValidMatch>,
    /// Evidence the ledger had recorded by `known_at`.
    pub evidence: Vec<Evidence>,
}

pub fn fact_as_of(fact: &Fact, as_of: AsOf) -> Option<FactAsOf> {
    let version = version_known_at(fact, as_of.known_at)?.clone();
    let valid = match as_of.valid_at {
        None => None,
        Some(t) => match valid_match(&version, t) {
            ValidMatch::Outside => return None,
            m => Some(m),
        },
    };
    let evidence: Vec<Evidence> = fact
        .evidence
        .iter()
        .filter(|e| as_of.known_at.is_none_or(|t| e.recorded_at <= t))
        .cloned()
        .collect();
    if evidence.is_empty() {
        return None;
    }
    Some(FactAsOf {
        version,
        valid,
        evidence,
    })
}

/// Whether a raw chunk is visible to a read. Chunks are observations, so
/// world time filters out ones that happened after `valid_at`.
pub fn chunk_visible(chunk: &Chunk, as_of: AsOf) -> bool {
    as_of.known_at.is_none_or(|t| chunk.recorded_at <= t)
        && as_of
            .valid_at
            .is_none_or(|t| chunk.occurred_at.is_none_or(|o| o <= t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FactStatus;
    use chrono::TimeZone;

    fn t(day: u32) -> Time {
        chrono::Utc.with_ymd_and_hms(2024, 1, day, 0, 0, 0).unwrap()
    }

    fn version(n: i32, from: u32, to: Option<u32>, rec: u32, inv: Option<u32>) -> FactVersion {
        FactVersion {
            version: n,
            valid_from: Some(t(from)),
            valid_to: to.map(t),
            valid_from_basis: TimeBasis::Observed,
            status: if to.is_some() {
                FactStatus::Superseded
            } else {
                FactStatus::Active
            },
            recorded_at: t(rec),
            invalidated_at: inv.map(t),
            cause: None,
        }
    }

    fn fact(versions: Vec<FactVersion>) -> Fact {
        Fact {
            corpus: "c".into(),
            id: "fact_1".into(),
            partition: "p".into(),
            subject: "e".into(),
            predicate: "lives_in".into(),
            object: "Lisbon".into(),
            object_entity: None,
            text: "x lives in Lisbon".into(),
            versions,
            evidence: vec![Evidence {
                corpus: "c".into(),
                id: "ev".into(),
                fact_id: "fact_1".into(),
                episode_id: "ep".into(),
                source_id: "s".into(),
                partition: "p".into(),
                char_start: 0,
                char_end: 1,
                quote: "x".into(),
                extractor: "t".into(),
                recorded_at: t(2),
            }],
        }
    }

    #[test]
    fn system_time_selects_the_version_believed_then() {
        // Recorded day 2 as open; on day 10 we learned it ended on day 8.
        let f = fact(vec![
            version(1, 2, None, 2, Some(10)),
            version(2, 2, Some(8), 10, None),
        ]);
        assert_eq!(version_known_at(&f, Some(t(1))), None);
        assert_eq!(version_known_at(&f, Some(t(5))).unwrap().version, 1);
        assert_eq!(version_known_at(&f, Some(t(10))).unwrap().version, 2);
        assert_eq!(version_known_at(&f, None).unwrap().version, 2);

        // On day 9 we still believed it held on day 9; today we know it did not.
        assert!(fact_as_of(&f, AsOf::both(t(9), t(9))).is_some());
        assert!(fact_as_of(&f, AsOf::both(t(11), t(9))).is_none());
        assert!(fact_as_of(&f, AsOf::valid_at(t(5))).is_some());
        // Before the observation the fact is not asserted.
        assert!(fact_as_of(&f, AsOf::valid_at(t(1))).is_none());
    }

    #[test]
    fn unknown_start_is_labelled_not_hidden() {
        let mut v = version(1, 2, None, 2, None);
        v.valid_from = None;
        v.valid_from_basis = TimeBasis::Unknown;
        let got = fact_as_of(&fact(vec![v]), AsOf::valid_at(t(1))).unwrap();
        assert_eq!(got.valid, Some(ValidMatch::StartUnknown));
    }

    #[test]
    fn evidence_recorded_later_is_invisible_to_earlier_reads() {
        let f = fact(vec![version(1, 1, None, 1, None)]);
        // Version recorded day 1 but its only evidence day 2.
        assert!(fact_as_of(&f, AsOf::known_at(t(1))).is_none());
        assert!(fact_as_of(&f, AsOf::known_at(t(2))).is_some());
    }
}
