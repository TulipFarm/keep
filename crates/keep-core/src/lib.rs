//! Shared contracts for Keep.
//!
//! This crate holds identifiers, content hashing and the plain record types for
//! the two primitives: MEMORY (a small curated notebook of Markdown files backed
//! by structured revisions) and KNOWLEDGE (episodes and temporal facts with
//! evidence). It has no I/O, no SQL and no HTTP. Only the parts the research
//! harness needs exist today; the full engine is later work.

pub mod hash;
pub mod knowledge;
pub mod memory;

pub use hash::{ContentHash, canonical_json, sha256_hex};
