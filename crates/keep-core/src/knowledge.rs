//! KNOWLEDGE record types.
//!
//! Knowledge holds source episodes (conversation turns, documents, events)
//! and temporal facts extracted from them with evidence. The research harness
//! indexes episodes directly for its baselines; facts are declared here so
//! later prototypes share one shape.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Who produced an episode's text.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    /// A named participant in a multi-party conversation.
    Speaker(String),
    System,
}

impl Role {
    pub fn label(&self) -> &str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Speaker(name) => name,
            Role::System => "system",
        }
    }
}

/// One unit of source evidence, such as a single conversation turn.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    /// Deterministic, dataset-scoped identifier.
    pub id: String,
    /// Groups episodes from one session or document.
    pub source_id: String,
    pub role: Role,
    /// Source timestamp as written by the dataset.
    pub occurred_at_raw: Option<String>,
    /// Parsed source timestamp, when the raw value could be parsed.
    pub occurred_at: Option<DateTime<Utc>>,
    pub text: String,
}

/// A time interval in which a fact holds. Unknown bounds stay `None`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ValidTime {
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_to: Option<DateTime<Utc>>,
}

/// A conservative, evidence-backed statement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub valid: ValidTime,
    /// When the system learned it (server time), distinct from validity.
    pub recorded_at: DateTime<Utc>,
    /// Episode ids that support the fact. Never empty for an accepted fact.
    pub evidence: Vec<String>,
}

impl Fact {
    pub fn validate(&self) -> Result<(), String> {
        if self.evidence.is_empty() {
            return Err(format!("fact {} has no evidence", self.id));
        }
        if let (Some(from), Some(to)) = (self.valid.valid_from, self.valid.valid_to)
            && to < from
        {
            return Err(format!("fact {} ends before it starts", self.id));
        }
        Ok(())
    }
}
