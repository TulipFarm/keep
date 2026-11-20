//! Experiment arms: outside baselines plus Keep's path and representation
//! ablations.
//!
//! Every Keep path runs: Memory (M) on `keep-memory` (see
//! [`crate::memory_arm`]), Knowledge (K0, K1, K2, K3) on `keep-knowledge` and
//! the combined Mk0, Mk1, Mk2 and Mk3 on both (see [`crate::knowledge_arm`]).

use serde::{Deserialize, Serialize};

/// Which Keep primitives an arm gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeepPath {
    /// Curated Memory files injected under a fixed budget; no Knowledge.
    M,
    /// Knowledge search/fetch only; no table of contents.
    K0,
    /// Knowledge search/fetch plus the bounded table of contents.
    K1,
    /// Memory files plus Knowledge search, no table of contents.
    Mk0,
    /// Memory files plus Knowledge search and table of contents.
    Mk1,
    /// Hybrid Knowledge search (visibility-first BM25 and dense, fused with
    /// RRF, optionally reranked), filled to the evidence allowance.
    K2,
    /// Memory entries and hybrid Knowledge results ranked together, deduplicated
    /// and filled to the combined budget, preferring raw source turns.
    Mk2,
    /// K2 preceded by a dated block of model-extracted session facts
    /// (current values with their superseded history, grouped by subject
    /// and attribute); the ranked fill takes what the block leaves.
    K3,
    /// Mk2 preceded by the same extracted-facts block, within Mk2's
    /// combined budget.
    Mk3,
}

impl KeepPath {
    pub fn uses_memory(self) -> bool {
        matches!(
            self,
            KeepPath::M | KeepPath::Mk0 | KeepPath::Mk1 | KeepPath::Mk2 | KeepPath::Mk3
        )
    }
    pub fn uses_knowledge(self) -> bool {
        !matches!(self, KeepPath::M)
    }
    pub fn uses_toc(self) -> bool {
        matches!(self, KeepPath::K1 | KeepPath::Mk1)
    }
    /// The hybrid, budget-filling paths.
    pub fn is_ranked(self) -> bool {
        matches!(
            self,
            KeepPath::K2 | KeepPath::Mk2 | KeepPath::K3 | KeepPath::Mk3
        )
    }
    /// The paths that put model-extracted session facts first.
    pub fn uses_extracted_facts(self) -> bool {
        matches!(self, KeepPath::K3 | KeepPath::Mk3)
    }
}

/// How Memory files are stored underneath the same visible Markdown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Representation {
    /// Whole-file Markdown text with file-level metadata.
    Raw,
    /// Lossless structured sections with optional annotations.
    #[default]
    Structured,
    /// Structured sections plus stable bullet entries in opted-in lists.
    StructuredBullets,
}

/// How chunks are cut for retrieval baselines.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Chunking {
    #[default]
    Turn,
    Session,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ArmKind {
    /// No history at all: measures what the model guesses unaided.
    NoMemory,
    /// The complete visible history, never truncated. Questions whose
    /// history does not fit are reported as unavailable.
    FullContext,
    /// The most recent turns that fit the budget, labelled as truncation.
    RecentWindow { budget_tokens: usize },
    /// Plain BM25 over fixed chunks.
    Bm25 { budget_tokens: usize, top_k: usize },
    /// Plain dense retrieval over fixed chunks through the storage profile.
    Dense { budget_tokens: usize, top_k: usize },
    /// Sessions written as Markdown notes and searched with deterministic
    /// keyword grep. A model-driven filesystem agent is a separate arm.
    NotesGrep { budget_tokens: usize },
    /// A model-driven read/grep agent over the same notes. Needs model tool
    /// loops, so it only runs with dispatch enabled in a later stage.
    FilesystemAgent {
        budget_tokens: usize,
        max_steps: usize,
    },
    /// Gold evidence turns: an upper-bound diagnostic, not a competitor.
    OracleEvidence,
    /// A Keep prototype arm.
    Keep {
        path: KeepPath,
        #[serde(default)]
        representation: Representation,
    },
}

impl ArmKind {
    pub fn label(&self) -> &'static str {
        match self {
            ArmKind::NoMemory => "no-memory",
            ArmKind::FullContext => "full-context",
            ArmKind::RecentWindow { .. } => "recent-window",
            ArmKind::Bm25 { .. } => "bm25",
            ArmKind::Dense { .. } => "dense",
            ArmKind::NotesGrep { .. } => "notes-grep",
            ArmKind::FilesystemAgent { .. } => "filesystem-agent",
            ArmKind::OracleEvidence => "oracle-evidence",
            ArmKind::Keep { .. } => "keep",
        }
    }

    /// Why this arm cannot run in the current build, if it cannot.
    pub fn unavailable_reason(&self) -> Option<String> {
        match self {
            ArmKind::FilesystemAgent { .. } => {
                Some("the model-driven filesystem agent is not implemented yet".into())
            }
            _ => None,
        }
    }

    pub fn is_diagnostic(&self) -> bool {
        matches!(self, ArmKind::OracleEvidence)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArmSpec {
    pub name: String,
    #[serde(flatten)]
    pub kind: ArmKind,
    /// K3 and Mk3: read the question's time reference and favour facts and
    /// evidence from that window; show each fact's age at question time.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub temporal: bool,
    /// Mk3: put the extracted preferences, habits and opinions in an
    /// always-included Memory profile instead of the facts block.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub preferences: bool,
    /// K3 and Mk3: answer questions about the assistant's own earlier
    /// replies from the ranked fill alone, without the facts block.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub assistant_route: bool,
    /// K3 and Mk3: find the participants the question names and set facts
    /// about other participants apart in the context.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub subjects: bool,
    /// K3 and Mk3: add the most relevant per-session summaries, within
    /// `extraction.summary_tokens` of the same budget.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub summaries: bool,
    /// K3 and Mk3: rank facts that share a named entity, or a subject and
    /// date, with the top-ranked facts as one more fused list.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub expand: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub provenance: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub relative_dates: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub decompose: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub fact_selection: bool,
    /// K3 and Mk3: rank conversation excerpts by what the question asks
    /// rather than by the participant it names, and favour the asked
    /// participant's own turns. See `crate::v8`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub focus: bool,
    /// K3 and Mk3: leave out of the facts block the facts whose source
    /// turns the conversation excerpts already show whole, and give the
    /// excerpts that room.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub verbatim_first: bool,
    /// The answer prompt version this arm uses, when it is not the one
    /// pinned in `[models.answer]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer_prompt: Option<String>,
}

/// What a fact arm adds on top of the extracted facts block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FactOptions {
    pub temporal: bool,
    pub preferences: bool,
    pub assistant_route: bool,
    pub subjects: bool,
    pub summaries: bool,
    pub expand: bool,
    pub provenance: bool,
    pub relative_dates: bool,
    pub decompose: bool,
    pub fact_selection: bool,
    pub focus: bool,
    pub verbatim_first: bool,
}

impl ArmSpec {
    pub fn new(name: impl Into<String>, kind: ArmKind) -> Self {
        Self {
            name: name.into(),
            kind,
            temporal: false,
            preferences: false,
            assistant_route: false,
            subjects: false,
            summaries: false,
            expand: false,
            provenance: false,
            relative_dates: false,
            decompose: false,
            fact_selection: false,
            focus: false,
            verbatim_first: false,
            answer_prompt: None,
        }
    }

    pub fn fact_options(&self) -> FactOptions {
        FactOptions {
            temporal: self.temporal,
            preferences: self.preferences,
            assistant_route: self.assistant_route,
            subjects: self.subjects,
            summaries: self.summaries,
            expand: self.expand,
            provenance: self.provenance,
            relative_dates: self.relative_dates,
            decompose: self.decompose,
            fact_selection: self.fact_selection,
            focus: self.focus,
            verbatim_first: self.verbatim_first,
        }
    }

    /// The answer prompt version this arm's records use.
    pub fn answer_prompt<'a>(&'a self, default: &'a str) -> &'a str {
        self.answer_prompt.as_deref().unwrap_or(default)
    }
}

/// The standard Keep ablation grid: every path with the default
/// representation, plus the representation variants on the M path.
pub fn keep_ablation_grid() -> Vec<ArmSpec> {
    let mut arms: Vec<ArmSpec> = [
        KeepPath::M,
        KeepPath::K0,
        KeepPath::K1,
        KeepPath::Mk0,
        KeepPath::Mk1,
    ]
    .into_iter()
    .map(|path| {
        ArmSpec::new(
            format!("keep-{}", format!("{path:?}").to_lowercase()),
            ArmKind::Keep {
                path,
                representation: Representation::Structured,
            },
        )
    })
    .collect();
    for representation in [Representation::Raw, Representation::StructuredBullets] {
        arms.push(ArmSpec::new(
            format!(
                "keep-m-{}",
                serde_json::to_value(representation)
                    .unwrap()
                    .as_str()
                    .unwrap()
            ),
            ArmKind::Keep {
                path: KeepPath::M,
                representation,
            },
        ));
    }
    arms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arms_round_trip_through_toml_and_every_keep_path_runs() {
        #[derive(Serialize, Deserialize)]
        struct Wrap {
            arms: Vec<ArmSpec>,
        }
        let text = r#"
[[arms]]
name = "bm25-2k"
kind = "bm25"
budget_tokens = 2000
top_k = 20

[[arms]]
name = "keep-mk1"
kind = "keep"
path = "mk1"
representation = "structured-bullets"
"#;
        let w: Wrap = toml::from_str(text).unwrap();
        assert_eq!(
            w.arms[0].kind,
            ArmKind::Bm25 {
                budget_tokens: 2000,
                top_k: 20
            }
        );
        let ArmKind::Keep {
            path,
            representation,
        } = w.arms[1].kind
        else {
            panic!("expected keep arm")
        };
        assert!(path.uses_memory() && path.uses_knowledge() && path.uses_toc());
        assert_eq!(representation, Representation::StructuredBullets);
        assert!(w.arms.iter().all(|a| a.kind.unavailable_reason().is_none()));
    }

    #[test]
    fn ablation_grid_covers_paths_and_representations() {
        let grid = keep_ablation_grid();
        let names: Vec<_> = grid.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "keep-m",
                "keep-k0",
                "keep-k1",
                "keep-mk0",
                "keep-mk1",
                "keep-m-raw",
                "keep-m-structured-bullets"
            ]
        );
    }
}
