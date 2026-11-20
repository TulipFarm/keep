//! Retained-evidence allowance.
//!
//! Everything a caller keeps from `search` and `fetch` while answering one
//! question is charged against a single token allowance, so an agent cannot
//! grow its context by calling tools repeatedly.

use serde::{Deserialize, Serialize};

pub const DEFAULT_EVIDENCE_TOKENS: usize = 2_000;
pub const MAX_EVIDENCE_TOKENS: usize = 8_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceAllowance {
    limit: usize,
    used: usize,
}

impl Default for EvidenceAllowance {
    fn default() -> Self {
        Self::new(DEFAULT_EVIDENCE_TOKENS)
    }
}

impl EvidenceAllowance {
    /// Requests above the hard maximum are clamped, never honoured.
    pub fn new(limit: usize) -> Self {
        Self {
            limit: limit.min(MAX_EVIDENCE_TOKENS),
            used: 0,
        }
    }
    pub fn limit(&self) -> usize {
        self.limit
    }
    pub fn used(&self) -> usize {
        self.used
    }
    pub fn remaining(&self) -> usize {
        self.limit - self.used
    }
    /// Charge `tokens` if they fit; otherwise charge nothing.
    pub fn try_charge(&mut self, tokens: usize) -> bool {
        if tokens <= self.remaining() {
            self.used += tokens;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charges_until_exhausted_and_clamps_the_limit() {
        let mut a = EvidenceAllowance::new(100);
        assert!(a.try_charge(60));
        assert!(!a.try_charge(41));
        assert!(a.try_charge(40));
        assert_eq!(a.remaining(), 0);
        assert_eq!(
            EvidenceAllowance::new(1_000_000).limit(),
            MAX_EVIDENCE_TOKENS
        );
        assert_eq!(EvidenceAllowance::default().limit(), 2_000);
    }
}
