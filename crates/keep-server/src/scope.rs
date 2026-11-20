//! Scopes: who a call is for and which project it belongs to.
//!
//! A scope is a user plus an optional project. It maps onto the engines'
//! own boundaries:
//!
//! - **Memory**: one tree per scope. A user scope and each of that user's
//!   project scopes are separate trees, so projects never see each other's
//!   files.
//! - **Knowledge**: one corpus per user (the hard boundary), with one
//!   visibility partition per scope. A project scope writes to its own
//!   partition and reads its own plus the user's personal partition; a user
//!   scope reads and writes only the personal partition.
//!
//! Agents that pass the same scope share the same Memory and Knowledge.

use keep_knowledge::Access;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::ApiError;

/// Long enough for composite host ids such as `ws-<uuid>-m-<uuid>`.
pub const MAX_ID_LEN: usize = 128;
const USER_PARTITION: &str = "user";

/// Who the call is for. Omitted fields fall back to the server's defaults.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScopeArg {
    /// User id, such as `alex`. Letters, digits, `.`, `_` and `-`.
    #[serde(default)]
    pub user: Option<String>,
    /// Project id, such as `billing-api`. Omit (and set no server default)
    /// for the user's personal scope. Pass an empty string to force the
    /// personal scope when the server has a default project.
    #[serde(default)]
    pub project: Option<String>,
}

/// A validated scope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Scope {
    pub user: String,
    pub project: Option<String>,
}

impl Scope {
    pub fn new(user: impl Into<String>, project: Option<String>) -> Result<Self, ApiError> {
        let user = user.into();
        validate_id("user", &user)?;
        if let Some(p) = &project {
            validate_id("project", p)?;
        }
        Ok(Self { user, project })
    }

    /// Apply a call's scope argument over the server defaults.
    pub fn resolve(arg: Option<&ScopeArg>, defaults: &Scope) -> Result<Self, ApiError> {
        let arg = arg.cloned().unwrap_or_default();
        let user = arg.user.unwrap_or_else(|| defaults.user.clone());
        let project = match arg.project {
            Some(p) if p.is_empty() => None,
            Some(p) => Some(p),
            None => defaults.project.clone(),
        };
        Self::new(user, project)
    }

    pub fn memory_tree(&self) -> String {
        match &self.project {
            Some(p) => format!("user/{}/project/{p}", self.user),
            None => format!("user/{}", self.user),
        }
    }

    /// The owning-scope label stored on the Memory tree.
    pub fn memory_scope(&self) -> String {
        match &self.project {
            Some(p) => format!("user:{}/project:{p}", self.user),
            None => format!("user:{}", self.user),
        }
    }

    pub fn corpus(&self) -> String {
        format!("user:{}", self.user)
    }

    /// The partition this scope writes Knowledge into.
    pub fn partition(&self) -> String {
        match &self.project {
            Some(p) => format!("project:{p}"),
            None => USER_PARTITION.into(),
        }
    }

    /// The partitions this scope may read.
    pub fn access(&self) -> Access {
        match &self.project {
            Some(_) => Access::to([self.partition(), USER_PARTITION.into()]),
            None => Access::to([USER_PARTITION]),
        }
    }

    pub fn label(&self) -> String {
        self.memory_scope()
    }
}

fn validate_id(what: &str, v: &str) -> Result<(), ApiError> {
    let ok = !v.is_empty()
        && v.len() <= MAX_ID_LEN
        && v.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && v.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if ok {
        Ok(())
    } else {
        Err(ApiError::invalid(format!(
            "{what} id must be 1-{MAX_ID_LEN} characters of letters, digits, '.', '_' or '-', \
             starting with a letter or digit"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_ids_with_uuids_are_accepted_up_to_the_limit() {
        let ws = "ws-3f2b9c1e-8d4a-4f6b-9e2d-1a7c5b3e9f00";
        let member = format!("{ws}-m-0b8e6d4c-2a1f-4e3d-8c5b-7a9f1e2d3c4b");
        assert_eq!(member.len(), 78);
        let s = Scope::new(ws, Some(member.clone())).unwrap();
        assert_eq!(s.partition(), format!("project:{member}"));
        Scope::new(member, None).unwrap();
        Scope::new("a".repeat(MAX_ID_LEN), None).unwrap();
        assert!(Scope::new("a".repeat(MAX_ID_LEN + 1), None).is_err());
        assert!(Scope::new("ws/../x", None).is_err());
    }

    #[test]
    fn project_scopes_share_the_user_corpus_but_not_partitions_or_trees() {
        let defaults = Scope::new("alex", None).unwrap();
        let a = Scope::resolve(
            Some(&ScopeArg {
                user: None,
                project: Some("api".into()),
            }),
            &defaults,
        )
        .unwrap();
        let b = Scope::new("alex", Some("web".into())).unwrap();
        assert_eq!(a.corpus(), b.corpus());
        assert_ne!(a.memory_tree(), b.memory_tree());
        assert!(!a.access().allows(&b.partition()));
        assert!(a.access().allows("user") && b.access().allows("user"));
        assert!(!defaults.access().allows(&a.partition()));
    }

    #[test]
    fn rejects_ids_that_could_break_out_of_their_namespace() {
        for bad in ["", "a/b", "a:b", "../x", "-x", &"x".repeat(MAX_ID_LEN + 1)] {
            assert!(Scope::new(bad, None).is_err(), "{bad:?}");
            assert!(Scope::new("ok", Some(bad.to_string())).is_err(), "{bad:?}");
        }
        let defaults = Scope::new("alex", Some("api".into())).unwrap();
        let personal = Scope::resolve(
            Some(&ScopeArg {
                user: None,
                project: Some(String::new()),
            }),
            &defaults,
        )
        .unwrap();
        assert_eq!(personal.project, None);
    }
}
