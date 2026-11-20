//! Contradiction handling for single-valued predicates.
//!
//! When a new value arrives for a predicate like `lives_in`, nothing is
//! deleted. Instead a new immutable version closes the older value's world
//! time window, pointing at the fact whose evidence caused the change:
//!
//! - new value starts later: the old value is superseded at that instant;
//! - new value starts earlier: the new value is the historical one and is
//!   closed where the old one starts;
//! - same start, or either start unknown: neither can win, so both are
//!   marked disputed and remain visible.

use crate::model::{FactStatus, FactVersion, Time, TimeBasis};

/// The current version of an existing competing fact.
#[derive(Clone, Debug)]
pub struct Rival {
    pub fact_id: String,
    pub current: FactVersion,
}

/// The incoming value.
#[derive(Clone, Debug)]
pub struct Incoming {
    pub fact_id: String,
    pub valid_from: Option<Time>,
    pub basis: TimeBasis,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// First version of the incoming fact (version 1).
    pub incoming: FactVersion,
    /// New versions for rivals: (fact id, expected current version, new version).
    pub rival_versions: Vec<(String, i32, FactVersion)>,
}

fn known_start(from: Option<Time>, basis: TimeBasis) -> Option<Time> {
    from.filter(|_| basis != TimeBasis::Unknown)
}

pub fn plan(incoming: &Incoming, rivals: &[Rival], now: Time) -> Plan {
    let mut first = FactVersion {
        version: 1,
        valid_from: incoming.valid_from,
        valid_to: None,
        valid_from_basis: incoming.basis,
        status: FactStatus::Active,
        recorded_at: now,
        invalidated_at: None,
        cause: None,
    };
    let mut rival_versions = Vec::new();
    let new_start = known_start(incoming.valid_from, incoming.basis);
    for r in rivals {
        let cur = &r.current;
        // Windows already closed before the incoming value starts are history.
        if let (Some(end), Some(ns)) = (cur.valid_to, new_start)
            && end <= ns
        {
            continue;
        }
        let old_start = known_start(cur.valid_from, cur.valid_from_basis);
        let next = |status: FactStatus, valid_to: Option<Time>| FactVersion {
            version: cur.version + 1,
            valid_from: cur.valid_from,
            valid_to,
            valid_from_basis: cur.valid_from_basis,
            status,
            recorded_at: now,
            invalidated_at: None,
            cause: Some(incoming.fact_id.clone()),
        };
        match (old_start, new_start) {
            (Some(o), Some(n)) if n > o => {
                if cur.valid_to.is_some_and(|end| end <= n) {
                    continue;
                }
                rival_versions.push((
                    r.fact_id.clone(),
                    cur.version,
                    next(FactStatus::Superseded, Some(n)),
                ));
            }
            (Some(o), Some(n)) if n < o => {
                let end = first.valid_to.map_or(o, |e| e.min(o));
                first.valid_to = Some(end);
                first.status = FactStatus::Superseded;
                first.cause = Some(r.fact_id.clone());
            }
            _ => {
                if cur.status != FactStatus::Disputed {
                    rival_versions.push((
                        r.fact_id.clone(),
                        cur.version,
                        next(FactStatus::Disputed, cur.valid_to),
                    ));
                }
                if first.status == FactStatus::Active {
                    first.status = FactStatus::Disputed;
                    first.cause = Some(r.fact_id.clone());
                }
            }
        }
    }
    Plan {
        incoming: first,
        rival_versions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn t(d: u32) -> Time {
        chrono::Utc.with_ymd_and_hms(2024, 3, d, 0, 0, 0).unwrap()
    }

    fn rival(from: Option<u32>, basis: TimeBasis) -> Rival {
        Rival {
            fact_id: "old".into(),
            current: FactVersion {
                version: 1,
                valid_from: from.map(t),
                valid_to: None,
                valid_from_basis: basis,
                status: FactStatus::Active,
                recorded_at: t(1),
                invalidated_at: None,
                cause: None,
            },
        }
    }

    fn incoming(from: Option<u32>, basis: TimeBasis) -> Incoming {
        Incoming {
            fact_id: "new".into(),
            valid_from: from.map(t),
            basis,
        }
    }

    #[test]
    fn later_value_supersedes_with_a_pointer_to_its_cause() {
        let p = plan(
            &incoming(Some(10), TimeBasis::Observed),
            &[rival(Some(2), TimeBasis::Observed)],
            t(20),
        );
        assert_eq!(p.incoming.status, FactStatus::Active);
        let (id, expected, v) = &p.rival_versions[0];
        assert_eq!((id.as_str(), *expected), ("old", 1));
        assert_eq!(v.version, 2);
        assert_eq!(v.valid_to, Some(t(10)));
        assert_eq!(v.status, FactStatus::Superseded);
        assert_eq!(v.cause.as_deref(), Some("new"));
        assert_eq!(v.recorded_at, t(20));
    }

    #[test]
    fn earlier_value_arriving_late_becomes_history() {
        let p = plan(
            &incoming(Some(1), TimeBasis::Explicit),
            &[rival(Some(5), TimeBasis::Observed)],
            t(20),
        );
        assert!(p.rival_versions.is_empty());
        assert_eq!(p.incoming.valid_to, Some(t(5)));
        assert_eq!(p.incoming.status, FactStatus::Superseded);
    }

    #[test]
    fn unordered_values_are_disputed_not_dropped() {
        for (a, b) in [
            (
                incoming(Some(5), TimeBasis::Observed),
                rival(Some(5), TimeBasis::Observed),
            ),
            (
                incoming(None, TimeBasis::Unknown),
                rival(Some(5), TimeBasis::Observed),
            ),
        ] {
            let p = plan(&a, &[b], t(20));
            assert_eq!(p.incoming.status, FactStatus::Disputed);
            assert_eq!(p.rival_versions[0].2.status, FactStatus::Disputed);
            assert_eq!(p.rival_versions[0].2.valid_to, None);
        }
    }
}
