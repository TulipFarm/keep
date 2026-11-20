//! An optional cap on Knowledge evidence per caller over a rolling window.
//!
//! Each `knowledge_search` and `knowledge_fetch` reserves its allowance
//! before it runs and is charged what it returned. A call's allowance is cut
//! to what is left; with nothing left it fails with `budget_exceeded`.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::json;

use crate::error::{ApiError, ErrorCode};

/// More callers than this triggers a sweep of idle ones.
const SWEEP_AT: usize = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum BudgetPer {
    /// One budget per bearer token (or per process without HTTP tokens).
    Token,
    /// One budget per session: per `keep mcp` process, or per value of the
    /// `x-keep-session` header for HTTP clients.
    Session,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetConfig {
    pub tokens: usize,
    pub window: Duration,
    pub per: BudgetPer,
}

pub struct EvidenceBudget {
    config: BudgetConfig,
    spent: Mutex<HashMap<String, VecDeque<Charge>>>,
    next_id: Mutex<u64>,
}

struct Charge {
    at: Instant,
    id: u64,
    tokens: usize,
}

/// Tokens set aside for one call; settle it with what the call used.
#[must_use]
pub struct Reservation {
    caller: String,
    id: u64,
    pub tokens: usize,
}

impl EvidenceBudget {
    pub fn new(config: BudgetConfig) -> Self {
        Self {
            config,
            spent: Mutex::new(HashMap::new()),
            next_id: Mutex::new(0),
        }
    }

    pub fn config(&self) -> BudgetConfig {
        self.config
    }

    /// Set aside up to `want` tokens for `caller`.
    pub fn reserve(&self, caller: &str, want: usize) -> Result<Reservation, ApiError> {
        self.reserve_at(caller, want, Instant::now())
    }

    fn reserve_at(&self, caller: &str, want: usize, now: Instant) -> Result<Reservation, ApiError> {
        let id = {
            let mut n = self.next_id.lock().expect("budget lock");
            *n += 1;
            *n
        };
        let mut spent = self.spent.lock().expect("budget lock");
        if spent.len() >= SWEEP_AT {
            spent.retain(|_, q| {
                expire(q, now, self.config.window);
                !q.is_empty()
            });
        }
        let q = spent.entry(caller.to_owned()).or_default();
        expire(q, now, self.config.window);
        let used: usize = q.iter().map(|c| c.tokens).sum();
        let left = self.config.tokens.saturating_sub(used);
        if left == 0 {
            let retry = q
                .front()
                .map(|c| (c.at + self.config.window).saturating_duration_since(now))
                .unwrap_or_default();
            return Err(ApiError::new(
                ErrorCode::BudgetExceeded,
                format!(
                    "the evidence budget of {} tokens per {} seconds is used up",
                    self.config.tokens,
                    self.config.window.as_secs()
                ),
            )
            .with_reason("EVIDENCE_BUDGET_EXHAUSTED")
            .with_details(json!({
                "limit": self.config.tokens,
                "windowSeconds": self.config.window.as_secs(),
                "retryAfterSeconds": retry.as_secs().max(1),
            })));
        }
        let tokens = want.min(left);
        q.push_back(Charge {
            at: now,
            id,
            tokens,
        });
        Ok(Reservation {
            caller: caller.to_owned(),
            id,
            tokens,
        })
    }

    /// Charge what the call actually used, at most what it reserved.
    pub fn settle(&self, r: Reservation, used: usize) {
        let mut spent = self.spent.lock().expect("budget lock");
        if let Some(q) = spent.get_mut(&r.caller)
            && let Some(c) = q.iter_mut().find(|c| c.id == r.id)
        {
            c.tokens = used.min(r.tokens);
        }
    }
}

fn expire(q: &mut VecDeque<Charge>, now: Instant, window: Duration) {
    while q
        .front()
        .is_some_and(|c| now.saturating_duration_since(c.at) >= window)
    {
        q.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> EvidenceBudget {
        EvidenceBudget::new(BudgetConfig {
            tokens: 100,
            window: Duration::from_secs(60),
            per: BudgetPer::Token,
        })
    }

    #[test]
    fn caps_reservations_and_refuses_when_spent() {
        let b = budget();
        let t0 = Instant::now();
        let r = b.reserve_at("a", 70, t0).unwrap();
        assert_eq!(r.tokens, 70);
        b.settle(r, 60);
        let r = b.reserve_at("a", 70, t0).unwrap();
        assert_eq!(r.tokens, 40, "cut to what is left");
        b.settle(r, 40);
        let e = b.reserve_at("a", 1, t0).err().unwrap();
        assert_eq!(e.code, ErrorCode::BudgetExceeded);
        // Other callers have their own budget.
        assert_eq!(b.reserve_at("b", 70, t0).unwrap().tokens, 70);
        // The window rolls.
        let later = t0 + Duration::from_secs(61);
        assert_eq!(b.reserve_at("a", 70, later).unwrap().tokens, 70);
    }

    #[test]
    fn unsettled_reservations_count_in_full() {
        let b = budget();
        let t0 = Instant::now();
        let _held = b.reserve_at("a", 100, t0).unwrap();
        assert!(b.reserve_at("a", 1, t0).is_err());
    }
}
