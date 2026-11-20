//! Structured errors with stable codes, shared by MCP and HTTP.

use keep_knowledge::KnowledgeError;
use keep_memory::MemoryError;
use serde::Serialize;
use serde_json::{Value, json};

/// Stable, lowercase error codes. Clients may match on these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The request is malformed or names something impossible.
    Invalid,
    /// Missing or wrong bearer token.
    Unauthorized,
    PermissionDenied,
    NotFound,
    /// The state changed underneath the caller; re-read and retry.
    Conflict,
    PayloadTooLarge,
    /// A size or count limit of the Memory tree would be broken.
    LimitExceeded,
    /// A token budget is above its cap, or required context does not fit.
    BudgetExceeded,
    /// Storage or a vector backend failed; the request may be retried.
    Unavailable,
    Internal,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::Invalid => "invalid",
            ErrorCode::Unauthorized => "unauthorized",
            ErrorCode::PermissionDenied => "permission_denied",
            ErrorCode::NotFound => "not_found",
            ErrorCode::Conflict => "conflict",
            ErrorCode::PayloadTooLarge => "payload_too_large",
            ErrorCode::LimitExceeded => "limit_exceeded",
            ErrorCode::BudgetExceeded => "budget_exceeded",
            ErrorCode::Unavailable => "unavailable",
            ErrorCode::Internal => "internal",
        }
    }

    pub const ALL: [ErrorCode; 10] = [
        ErrorCode::Invalid,
        ErrorCode::Unauthorized,
        ErrorCode::PermissionDenied,
        ErrorCode::NotFound,
        ErrorCode::Conflict,
        ErrorCode::PayloadTooLarge,
        ErrorCode::LimitExceeded,
        ErrorCode::BudgetExceeded,
        ErrorCode::Unavailable,
        ErrorCode::Internal,
    ];

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.as_str() == s)
    }

    pub fn http_status(self) -> u16 {
        match self {
            ErrorCode::Invalid => 400,
            ErrorCode::Unauthorized => 401,
            ErrorCode::PermissionDenied => 403,
            ErrorCode::NotFound => 404,
            ErrorCode::Conflict => 409,
            ErrorCode::PayloadTooLarge => 413,
            ErrorCode::LimitExceeded | ErrorCode::BudgetExceeded => 422,
            ErrorCode::Unavailable => 503,
            ErrorCode::Internal => 500,
        }
    }

    pub fn retryable(self) -> bool {
        matches!(self, ErrorCode::Unavailable)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApiError {
    pub code: ErrorCode,
    pub message: String,
    /// The engine's own, finer-grained code, such as `VERSION_CONFLICT`.
    pub reason: Option<String>,
    /// Machine-readable context, such as the current head on a conflict.
    pub details: Option<Value>,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for ApiError {}

impl ApiError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            reason: None,
            details: None,
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Invalid, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, message)
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Conflict, message)
    }

    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    /// The JSON body both transports return.
    pub fn to_json(&self) -> Value {
        let mut e = json!({
            "code": self.code.as_str(),
            "message": self.message,
            "retryable": self.code.retryable(),
        });
        if let Some(r) = &self.reason {
            e["reason"] = json!(r);
        }
        if let Some(d) = &self.details {
            e["details"] = d.clone();
        }
        json!({ "error": e })
    }

    /// Read back a body made by [`ApiError::to_json`].
    pub fn from_json(v: &Value) -> Option<Self> {
        let e = v.get("error")?;
        let code = ErrorCode::parse(e.get("code")?.as_str()?)?;
        Some(Self {
            code,
            message: e
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            reason: e.get("reason").and_then(Value::as_str).map(str::to_owned),
            details: e.get("details").cloned(),
        })
    }
}

impl From<MemoryError> for ApiError {
    fn from(e: MemoryError) -> Self {
        use MemoryError as M;
        let code = match &e {
            M::NotFound { .. } | M::RevisionNotRetained(_) | M::TrashExpired => ErrorCode::NotFound,
            M::VersionConflict { .. }
            | M::TreeConflict { .. }
            | M::PathExists(_)
            | M::IdempotencyConflict
            | M::IdempotencyKeyRetired
            | M::InTrash
            | M::NotInTrash
            | M::MergeConflict { .. } => ErrorCode::Conflict,
            M::InvalidPath(_)
            | M::InvalidText(_)
            | M::InvalidRequest(_)
            | M::NoMatch
            | M::MultipleMatches(_)
            | M::NothingToUndo => ErrorCode::Invalid,
            M::Limit { .. } => ErrorCode::LimitExceeded,
            M::PermissionDenied(_) | M::InstructionFileRefused { .. } => {
                ErrorCode::PermissionDenied
            }
            M::RequiredContextOverBudget(_) | M::Budget(_) => ErrorCode::BudgetExceeded,
            M::Storage(_) => ErrorCode::Unavailable,
        };
        let details = match &e {
            M::VersionConflict { current } => Some(json!({
                "current": { "revisionId": current.revision_id, "version": current.version }
            })),
            M::TreeConflict { current } => Some(json!({ "currentTreeRevision": current })),
            M::MergeConflict { entries, .. } => Some(json!({ "entries": entries })),
            M::Limit {
                kind,
                limit,
                actual,
            } => Some(json!({ "kind": kind, "limit": limit, "actual": actual })),
            _ => None,
        };
        let mut out = ApiError::new(code, e.to_string()).with_reason(e.code());
        out.details = details;
        out
    }
}

impl From<KnowledgeError> for ApiError {
    fn from(e: KnowledgeError) -> Self {
        use KnowledgeError as K;
        let (code, reason) = match &e {
            K::NotFound => (ErrorCode::NotFound, "NOT_FOUND"),
            K::Conflict(_) => (ErrorCode::Conflict, "CONFLICT"),
            K::BudgetExhausted => (ErrorCode::BudgetExceeded, "BUDGET_EXHAUSTED"),
            K::Invalid(_) => (ErrorCode::Invalid, "INVALID"),
            K::Storage(_) => (ErrorCode::Unavailable, "STORAGE"),
            K::Vector(_) => (ErrorCode::Unavailable, "VECTOR"),
            K::Provider(_) => (ErrorCode::Unavailable, "PROVIDER"),
        };
        ApiError::new(code, e.to_string()).with_reason(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trips() {
        for code in ErrorCode::ALL {
            let e = ApiError::new(code, "m")
                .with_reason("R")
                .with_details(json!({ "k": 1 }));
            assert_eq!(ApiError::from_json(&e.to_json()), Some(e));
        }
        assert_eq!(
            ApiError::from_json(&json!({ "error": { "code": "nope" } })),
            None
        );
        let bare = ApiError::new(ErrorCode::NotFound, "gone");
        assert_eq!(ApiError::from_json(&bare.to_json()), Some(bare));
    }
}
