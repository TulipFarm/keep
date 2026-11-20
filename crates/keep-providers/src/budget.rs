//! Hard caps on remote calls, tokens and spend.
//!
//! A call reserves its worst case (estimated input plus the full output
//! allowance) before dispatch and settles to the provider-reported usage
//! afterwards. A reservation that would cross any cap is refused, so a run
//! stops at its budget instead of overshooting it.

use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Price {
    pub input_per_mtok_usd: f64,
    pub output_per_mtok_usd: f64,
}

impl Price {
    pub fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 * self.input_per_mtok_usd
            + output_tokens as f64 * self.output_per_mtok_usd)
            / 1_000_000.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BudgetCaps {
    pub max_calls: u64,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub max_cost_usd: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BudgetUsage {
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("budget cap reached: {cap} (used {used}, requested {requested}, cap {limit})")]
pub struct BudgetExceeded {
    pub cap: &'static str,
    pub used: f64,
    pub requested: f64,
    pub limit: f64,
}

#[derive(Debug)]
pub struct Reservation {
    input_tokens: u64,
    output_tokens: u64,
    cost_usd: f64,
}

#[derive(Debug)]
pub struct BudgetLedger {
    caps: BudgetCaps,
    used: Mutex<BudgetUsage>,
}

impl Reservation {
    /// The input tokens reserved for the call.
    pub fn input_tokens(&self) -> u64 {
        self.input_tokens
    }
}

impl BudgetLedger {
    pub fn new(caps: BudgetCaps) -> Self {
        Self {
            caps,
            used: Mutex::new(BudgetUsage::default()),
        }
    }

    pub fn caps(&self) -> BudgetCaps {
        self.caps
    }

    pub fn usage(&self) -> BudgetUsage {
        *self.used.lock().expect("budget lock")
    }

    pub fn reserve(
        &self,
        input_tokens: u64,
        max_output_tokens: u64,
        price: &Price,
    ) -> Result<Reservation, BudgetExceeded> {
        let cost = price.cost(input_tokens, max_output_tokens);
        let mut used = self.used.lock().expect("budget lock");
        let checks: [(&'static str, f64, f64, f64); 4] = [
            ("calls", used.calls as f64, 1.0, self.caps.max_calls as f64),
            (
                "input tokens",
                used.input_tokens as f64,
                input_tokens as f64,
                self.caps.max_input_tokens as f64,
            ),
            (
                "output tokens",
                used.output_tokens as f64,
                max_output_tokens as f64,
                self.caps.max_output_tokens as f64,
            ),
            ("cost (USD)", used.cost_usd, cost, self.caps.max_cost_usd),
        ];
        for (cap, current, requested, limit) in checks {
            if current + requested > limit {
                return Err(BudgetExceeded {
                    cap,
                    used: current,
                    requested,
                    limit,
                });
            }
        }
        used.calls += 1;
        used.input_tokens += input_tokens;
        used.output_tokens += max_output_tokens;
        used.cost_usd += cost;
        Ok(Reservation {
            input_tokens,
            output_tokens: max_output_tokens,
            cost_usd: cost,
        })
    }

    /// Replace a reservation's worst case with actual usage. Actual usage is
    /// recorded even if it exceeds the reservation, so the ledger never
    /// under-reports spend.
    pub fn settle(
        &self,
        reservation: Reservation,
        input_tokens: u64,
        output_tokens: u64,
        price: &Price,
    ) {
        let mut used = self.used.lock().expect("budget lock");
        used.input_tokens = used.input_tokens - reservation.input_tokens + input_tokens;
        used.output_tokens = used.output_tokens - reservation.output_tokens + output_tokens;
        used.cost_usd =
            used.cost_usd - reservation.cost_usd + price.cost(input_tokens, output_tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_refuses_before_crossing_any_cap_and_settles_to_actuals() {
        let price = Price {
            input_per_mtok_usd: 1.0,
            output_per_mtok_usd: 2.0,
        };
        let ledger = BudgetLedger::new(BudgetCaps {
            max_calls: 3,
            max_input_tokens: 1_000,
            max_output_tokens: 300,
            max_cost_usd: 1.0,
        });
        let r = ledger.reserve(400, 100, &price).unwrap();
        ledger.settle(r, 380, 20, &price);
        let u = ledger.usage();
        assert_eq!((u.calls, u.input_tokens, u.output_tokens), (1, 380, 20));
        assert!((u.cost_usd - (380.0 + 40.0) / 1e6).abs() < 1e-12);

        let err = ledger.reserve(700, 100, &price).unwrap_err();
        assert_eq!(err.cap, "input tokens");

        ledger.reserve(100, 100, &price).unwrap();
        ledger.reserve(100, 100, &price).unwrap();
        assert_eq!(ledger.reserve(1, 1, &price).unwrap_err().cap, "calls");
    }

    #[test]
    fn zero_caps_refuse_everything() {
        let ledger = BudgetLedger::new(BudgetCaps::default());
        assert!(ledger.reserve(1, 1, &Price::default()).is_err());
    }
}
