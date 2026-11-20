//! Keep's research harness.
//!
//! Everything here runs offline by default. Remote model calls need an
//! explicit manifest switch, the `--allow-model-calls` flag, pinned models
//! and non-zero budget caps; otherwise they are refused before any request
//! is built.

pub mod ablation;
pub mod artifacts;
pub mod baselines;
pub mod calibration;
pub mod compare;
pub mod context;
pub mod dataset;
pub mod dryrun;
pub mod export;
pub mod facts;
pub mod fetch;
pub mod fitmask;
pub mod holdout;
pub mod isolation;
pub mod judge;
pub mod knowledge_arm;
pub mod manifest;
pub mod memory_arm;
pub mod metrics;
pub mod mocks;
pub mod prepare;
pub mod prompt;
pub mod protocol;
pub mod report;
pub mod runner;
pub mod spend;
pub mod stats;
pub mod v5;
pub mod v8;
