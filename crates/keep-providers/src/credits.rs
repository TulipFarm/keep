//! A run-level cap on harness-metered AI credits.
//!
//! Works like [`crate::budget::BudgetLedger`]: each call reserves a
//! worst-case credit estimate before it is dispatched, and settles to the
//! credits the harness reports afterwards. A reservation that would cross the
//! cap is refused, so a run stops dispatching instead of overshooting. When a
//! call fails after it may have reached the model and no charge was reported,
//! the reservation stays charged, because the harness may still bill for it.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::budget::BudgetExceeded;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CreditUsage {
    pub cap: f64,
    /// Credits reported by the harness for settled calls.
    pub reported: f64,
    /// Reservations kept because a call failed without a reported charge.
    pub unsettled: f64,
    /// Worst-case credits held by calls still in flight.
    pub in_flight: f64,
    pub calls: u64,
    /// Calls refused because their reservation would cross the cap.
    pub refused: u64,
    /// The largest charge reported for one call so far. Later reservations
    /// never go below it, so rates set too low stop mattering after the
    /// first settled call.
    #[serde(default)]
    pub largest_call: f64,
}

impl CreditUsage {
    /// Everything counted against the cap.
    pub fn charged(&self) -> f64 {
        self.reported + self.unsettled + self.in_flight
    }
}

#[derive(Debug)]
pub struct CreditLedger {
    used: Mutex<CreditUsage>,
}

#[derive(Debug)]
#[must_use = "settle or keep a credit reservation"]
pub struct CreditReservation {
    credits: f64,
}

impl CreditReservation {
    pub fn credits(&self) -> f64 {
        self.credits
    }
}

impl CreditLedger {
    pub fn new(cap: f64) -> Self {
        Self {
            used: Mutex::new(CreditUsage {
                cap,
                ..Default::default()
            }),
        }
    }

    pub fn usage(&self) -> CreditUsage {
        *self.used.lock().expect("credit lock")
    }

    pub fn reserve(&self, credits: f64) -> Result<CreditReservation, BudgetExceeded> {
        let mut u = self.used.lock().expect("credit lock");
        let credits = credits.max(u.largest_call).max(0.0);
        let current = u.charged();
        // The tolerance absorbs float noise from summing many small charges.
        if u.cap.is_nan() || u.cap <= 0.0 || current + credits > u.cap + 1e-9 {
            u.refused += 1;
            return Err(BudgetExceeded {
                cap: "AI credits",
                used: current,
                requested: credits,
                limit: u.cap,
            });
        }
        u.in_flight += credits;
        u.calls += 1;
        Ok(CreditReservation { credits })
    }

    /// Replace the reservation with the harness-reported charge. Reported
    /// credits are recorded even when they exceed the reservation.
    pub fn settle(&self, r: CreditReservation, reported: f64) {
        let mut u = self.used.lock().expect("credit lock");
        u.in_flight = (u.in_flight - r.credits).max(0.0);
        u.reported += reported.max(0.0);
        u.largest_call = u.largest_call.max(reported);
    }

    /// The call may have been billed but reported nothing: keep the worst case.
    pub fn keep(&self, r: CreditReservation) {
        let mut u = self.used.lock().expect("credit lock");
        u.in_flight = (u.in_flight - r.credits).max(0.0);
        u.unsettled += r.credits;
    }

    /// The call never reached a model (for example the CLI rejected its
    /// arguments): nothing could have been billed.
    pub fn release(&self, r: CreditReservation) {
        let mut u = self.used.lock().expect("credit lock");
        u.in_flight = (u.in_flight - r.credits).max(0.0);
        u.calls = u.calls.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_reservations_that_would_cross_the_cap_and_settles_to_reported() {
        let l = CreditLedger::new(10.0);
        let a = l.reserve(4.0).unwrap();
        let b = l.reserve(4.0).unwrap();
        // 8 in flight: a third worst case of 4 would cross 10.
        let err = l.reserve(4.0).unwrap_err();
        assert_eq!(err.cap, "AI credits");
        l.settle(a, 1.5);
        l.keep(b);
        let u = l.usage();
        assert_eq!((u.reported, u.unsettled, u.in_flight), (1.5, 4.0, 0.0));
        assert_eq!((u.calls, u.refused), (2, 1));
        // 5.5 charged; 4 more still fits, 5 does not.
        l.release(l.reserve(4.0).unwrap());
        assert!(l.reserve(5.0).is_err());
        assert_eq!(l.usage().calls, 2);
    }

    #[test]
    fn reservations_never_fall_below_the_largest_reported_charge() {
        let l = CreditLedger::new(1.0);
        // The rates guessed 0.01, but the call cost 0.4.
        l.settle(l.reserve(0.01).unwrap(), 0.4);
        let r = l.reserve(0.01).unwrap();
        assert_eq!(r.credits(), 0.4);
        // 0.4 reported + 0.4 in flight: another 0.4 would cross 1.0.
        assert!(l.reserve(0.01).is_err());
        l.settle(r, 0.4);
        assert!(l.usage().charged() <= 1.0);
    }

    #[test]
    fn a_zero_cap_refuses_everything() {
        assert!(CreditLedger::new(0.0).reserve(0.0).is_err());
    }
}
