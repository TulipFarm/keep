//! Bearer tokens and what each may touch.
//!
//! Two sources of tokens, both optional:
//!
//! - `KEEP_API_TOKEN`: one administrator token with full access.
//! - A tokens file (`--tokens-file`), TOML, mapping each token's SHA-256
//!   hash to the users and projects it may use and its permissions:
//!
//! ```toml
//! [[token]]
//! name = "laptop-agents"
//! sha256 = "<64 hex characters from `keep token hash`>"
//! users = ["alex"]          # or ["*"]
//! projects = ["api", "web"] # or ["*"]; [] for none
//! personal = true           # may name the user's personal scope
//! permissions = ["read", "write"] # and "admin" for administrative tools
//! ```
//!
//! Only hashes are stored. A presented token is hashed and compared with
//! every entry in constant time.

use std::path::Path;

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use subtle::{ConditionallySelectable, ConstantTimeEq};

use crate::api::ToolSpec;
use crate::error::{ApiError, ErrorCode};
use crate::scope::{Scope, ScopeArg};

/// Tools that change the whole corpus and need the `admin` permission.
pub const ADMIN_TOOLS: &[&str] = &["knowledge_rebuild_projection"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Permission {
    Read,
    Write,
    Admin,
}

impl Permission {
    pub fn as_str(self) -> &'static str {
        match self {
            Permission::Read => "read",
            Permission::Write => "write",
            Permission::Admin => "admin",
        }
    }

    pub fn for_tool(spec: &ToolSpec) -> Self {
        if ADMIN_TOOLS.contains(&spec.name) {
            Permission::Admin
        } else if spec.read_only {
            Permission::Read
        } else {
            Permission::Write
        }
    }
}

/// A list of ids, or `*` for any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ids {
    Any,
    Only(Vec<String>),
}

impl Ids {
    fn parse(field: &str, list: Vec<String>) -> Result<Self, String> {
        if list.iter().any(|s| s == "*") {
            if list.len() > 1 {
                return Err(format!("{field}: \"*\" must be the only entry"));
            }
            return Ok(Ids::Any);
        }
        for id in &list {
            if id.trim().is_empty() {
                return Err(format!("{field}: empty id"));
            }
        }
        Ok(Ids::Only(list))
    }

    fn allows(&self, id: &str) -> bool {
        match self {
            Ids::Any => true,
            Ids::Only(v) => v.iter().any(|x| x == id),
        }
    }

    fn single(&self) -> Option<&str> {
        match self {
            Ids::Only(v) if v.len() == 1 => Some(&v[0]),
            _ => None,
        }
    }

    fn is_empty(&self) -> bool {
        matches!(self, Ids::Only(v) if v.is_empty())
    }
}

/// What one token may do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grant {
    pub name: String,
    pub users: Ids,
    pub projects: Ids,
    pub personal: bool,
    pub permissions: Vec<Permission>,
}

impl Grant {
    fn allows_scope(&self, scope: &Scope) -> bool {
        self.users.allows(&scope.user)
            && match &scope.project {
                Some(p) => self.projects.allows(p),
                None => self.personal,
            }
    }

    /// The user to use when a call names none, if the grant implies one.
    fn default_user(&self) -> Option<&str> {
        self.users.single()
    }

    /// The project to use when a call names none: `Some(None)` for the
    /// personal scope, if that is the only scope the grant allows.
    fn default_project(&self) -> Option<Option<&str>> {
        match (self.projects.single(), self.personal) {
            (Some(p), false) => Some(Some(p)),
            (None, true) if self.projects.is_empty() => Some(None),
            _ => None,
        }
    }
}

/// Who made a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Principal {
    /// No authentication is configured (loopback only).
    Open,
    /// The `KEEP_API_TOKEN` administrator.
    Admin,
    Token(Grant),
}

impl Principal {
    /// A stable, non-secret label for budgets and logs.
    pub fn label(&self) -> &str {
        match self {
            Principal::Open => "open",
            Principal::Admin => "admin",
            Principal::Token(g) => &g.name,
        }
    }

    /// Check `spec` may run with `args`, and pin the call's scope: a
    /// missing user or project is filled from the grant when it allows only
    /// one, otherwise from the server's defaults, and the result is written
    /// back so the service sees exactly the scope that was checked.
    pub fn authorize(
        &self,
        spec: &ToolSpec,
        mut args: Value,
        defaults: &Scope,
    ) -> Result<Value, ApiError> {
        let Principal::Token(grant) = self else {
            return Ok(args);
        };
        let needed = Permission::for_tool(spec);
        if !grant.permissions.contains(&needed) {
            return Err(denied(format!(
                "this token lacks the {} permission needed for {}",
                needed.as_str(),
                spec.name
            )));
        }
        if args.is_null() {
            args = json!({});
        }
        let Some(obj) = args.as_object_mut() else {
            return Err(ApiError::invalid("request body must be a JSON object"));
        };
        let arg: ScopeArg = match obj.get("scope") {
            None | Some(Value::Null) => ScopeArg::default(),
            Some(v) => serde_json::from_value(v.clone()).map_err(|e| {
                ApiError::invalid(format!("invalid arguments: scope: {e}")).with_reason("ARGUMENTS")
            })?,
        };
        let user = arg
            .user
            .or_else(|| grant.default_user().map(str::to_owned))
            .unwrap_or_else(|| defaults.user.clone());
        let project = match arg.project {
            Some(p) if p.is_empty() => None,
            Some(p) => Some(p),
            None => match grant.default_project() {
                Some(p) => p.map(str::to_owned),
                None => defaults.project.clone(),
            },
        };
        let scope = Scope::new(user, project)?;
        if !grant.allows_scope(&scope) {
            return Err(denied(format!(
                "this token may not use scope {}",
                scope.label()
            )));
        }
        obj.insert(
            "scope".into(),
            json!({ "user": scope.user, "project": scope.project.clone().unwrap_or_default() }),
        );
        Ok(args)
    }
}

fn denied(message: String) -> ApiError {
    ApiError::new(ErrorCode::PermissionDenied, message).with_reason("SCOPE_NOT_ALLOWED")
}

/// Every configured token.
#[derive(Clone, Debug, Default)]
pub struct Auth {
    admin: Option<[u8; 32]>,
    tokens: Vec<([u8; 32], Grant)>,
}

impl From<Option<String>> for Auth {
    /// Only an administrator token, or none.
    fn from(admin: Option<String>) -> Self {
        Self::new(admin.as_deref(), Vec::new())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokensFile {
    #[serde(default)]
    token: Vec<TokenEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenEntry {
    name: String,
    sha256: String,
    users: Vec<String>,
    #[serde(default)]
    projects: Vec<String>,
    #[serde(default)]
    personal: bool,
    permissions: Vec<Permission>,
}

impl Auth {
    pub fn new(admin: Option<&str>, tokens: Vec<([u8; 32], Grant)>) -> Self {
        Self {
            admin: admin.map(digest),
            tokens,
        }
    }

    /// No token is needed: neither an administrator token nor a tokens
    /// file is configured.
    pub fn is_open(&self) -> bool {
        self.admin.is_none() && self.tokens.is_empty()
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Load a tokens file, with `admin` as the administrator token.
    pub fn load(admin: Option<&str>, path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading tokens file {}: {e}", path.display()))?;
        let tokens = Self::parse(&text)
            .map_err(|e| anyhow::anyhow!("tokens file {}: {e}", path.display()))?;
        if tokens.is_empty() {
            anyhow::bail!("tokens file {} lists no [[token]] entries", path.display());
        }
        Ok(Self::new(admin, tokens))
    }

    pub fn parse(text: &str) -> Result<Vec<([u8; 32], Grant)>, String> {
        let file: TokensFile = toml::from_str(text).map_err(|e| e.to_string())?;
        let mut out: Vec<([u8; 32], Grant)> = Vec::new();
        for e in file.token {
            let at = |m: String| format!("token {:?}: {m}", e.name);
            if e.name.trim().is_empty() {
                return Err("a token has an empty name".into());
            }
            if out.iter().any(|(_, g)| g.name == e.name) {
                return Err(at("the name is used twice".into()));
            }
            let hash = hex::decode(e.sha256.trim())
                .ok()
                .and_then(|b| <[u8; 32]>::try_from(b).ok())
                .ok_or_else(|| at("sha256 must be 64 hex characters".into()))?;
            if out.iter().any(|(h, _)| *h == hash) {
                return Err(at("the same token hash is listed twice".into()));
            }
            let users = Ids::parse("users", e.users).map_err(at)?;
            if users.is_empty() {
                return Err(at("users is empty".into()));
            }
            let projects = Ids::parse("projects", e.projects).map_err(at)?;
            if projects.is_empty() && !e.personal {
                return Err(at("allows no project and not the personal scope".into()));
            }
            if e.permissions.is_empty() {
                return Err(at("permissions is empty".into()));
            }
            out.push((
                hash,
                Grant {
                    name: e.name,
                    users,
                    projects,
                    personal: e.personal,
                    permissions: e.permissions,
                },
            ));
        }
        Ok(out)
    }

    /// Who presented `token`, or `None` if it matches nothing. Every entry
    /// is compared, in constant time, whatever matches.
    pub fn identify(&self, token: Option<&str>) -> Option<Principal> {
        if self.is_open() {
            return Some(Principal::Open);
        }
        let token = token?;
        let given = digest(token);
        let zero = [0u8; 32];
        let admin = given.ct_eq(self.admin.as_ref().unwrap_or(&zero))
            & subtle::Choice::from(u8::from(self.admin.is_some()));
        let mut found = subtle::Choice::from(0);
        let mut index = 0u64;
        for (i, (hash, _)) in self.tokens.iter().enumerate() {
            let hit = given.ct_eq(hash);
            index.conditional_assign(&(i as u64), hit);
            found |= hit;
        }
        if bool::from(admin) {
            Some(Principal::Admin)
        } else if bool::from(found) {
            let (_, grant) = &self.tokens[usize::try_from(index).ok()?];
            Some(Principal::Token(grant.clone()))
        } else {
            None
        }
    }
}

pub fn digest(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

/// The hex SHA-256 of a token, as written in a tokens file.
pub fn hash_hex(token: &str) -> String {
    hex::encode(digest(token))
}

/// A new random token: 32 bytes from the OS, hex encoded.
pub fn new_token() -> String {
    use rand::TryRng;
    let mut bytes = [0u8; 32];
    rand::rngs::SysRng
        .try_fill_bytes(&mut bytes)
        .expect("the OS random source is available");
    format!("keep_{}", hex::encode(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::tool;

    fn auth() -> Auth {
        let file = format!(
            r#"
[[token]]
name = "api-agent"
sha256 = "{}"
users = ["alex"]
projects = ["api"]
permissions = ["read", "write"]

[[token]]
name = "reader"
sha256 = "{}"
users = ["*"]
projects = ["*"]
personal = true
permissions = ["read"]

[[token]]
name = "mine"
sha256 = "{}"
users = ["alex"]
personal = true
permissions = ["read", "write", "admin"]
"#,
            hash_hex("t-api"),
            hash_hex("t-reader"),
            hash_hex("t-mine")
        );
        Auth::new(Some("t-admin"), Auth::parse(&file).unwrap())
    }

    fn run(token: &str, name: &str, args: Value) -> Result<Value, ApiError> {
        let p = auth().identify(Some(token)).expect("known token");
        let defaults = Scope::new("local", None).unwrap();
        p.authorize(tool(name).unwrap(), args, &defaults)
    }

    fn code(r: Result<Value, ApiError>) -> &'static str {
        match r {
            Ok(_) => "ok",
            Err(e) => e.code.as_str(),
        }
    }

    #[test]
    fn identifies_tokens_by_hash() {
        let a = auth();
        assert_eq!(a.identify(Some("t-admin")), Some(Principal::Admin));
        assert_eq!(a.identify(Some("t-api")).unwrap().label(), "api-agent");
        assert_eq!(a.identify(Some("t-mine")).unwrap().label(), "mine");
        assert_eq!(a.identify(Some("nope")), None);
        assert_eq!(a.identify(None), None);
        assert_eq!(Auth::default().identify(None), Some(Principal::Open));
        let only_file = Auth::new(None, Auth::parse("").unwrap());
        assert!(only_file.is_open());
    }

    #[test]
    fn scope_and_permission_matrix() {
        let api = json!({ "user": "alex", "project": "api" });
        let web = json!({ "user": "alex", "project": "web" });
        let personal = json!({ "user": "alex", "project": "" });
        let other = json!({ "user": "sam", "project": "api" });
        let cases = [
            ("t-api", "memory_read", api.clone(), "ok"),
            ("t-api", "memory_write", api.clone(), "ok"),
            ("t-api", "memory_read", web.clone(), "permission_denied"),
            (
                "t-api",
                "memory_read",
                personal.clone(),
                "permission_denied",
            ),
            ("t-api", "memory_read", other.clone(), "permission_denied"),
            (
                "t-api",
                "knowledge_rebuild_projection",
                api.clone(),
                "permission_denied",
            ),
            ("t-reader", "memory_read", other.clone(), "ok"),
            ("t-reader", "memory_read", personal.clone(), "ok"),
            ("t-reader", "memory_write", api.clone(), "permission_denied"),
            (
                "t-reader",
                "knowledge_forget_source",
                api.clone(),
                "permission_denied",
            ),
            ("t-mine", "memory_write", personal.clone(), "ok"),
            ("t-mine", "memory_write", api.clone(), "permission_denied"),
            (
                "t-mine",
                "knowledge_rebuild_projection",
                personal.clone(),
                "ok",
            ),
            (
                "t-admin",
                "knowledge_rebuild_projection",
                other.clone(),
                "ok",
            ),
        ];
        for (token, tool, scope, want) in cases {
            let got = code(run(token, tool, json!({ "scope": scope })));
            assert_eq!(got, want, "{token} {tool} {scope}");
        }
    }

    #[test]
    fn missing_scope_defaults_to_the_only_allowed_one() {
        let out = run("t-api", "memory_read", json!({})).unwrap();
        assert_eq!(out["scope"], json!({ "user": "alex", "project": "api" }));
        let out = run(
            "t-api",
            "memory_read",
            json!({ "scope": { "project": "api" } }),
        )
        .unwrap();
        assert_eq!(out["scope"]["user"], "alex");
        let out = run("t-mine", "memory_read", Value::Null).unwrap();
        assert_eq!(out["scope"], json!({ "user": "alex", "project": "" }));
        // Ambiguous: falls back to the server default user `local`, which
        // the reader may use, in the personal scope.
        let out = run("t-reader", "memory_read", json!({})).unwrap();
        assert_eq!(out["scope"], json!({ "user": "local", "project": "" }));
    }

    #[test]
    fn bad_files_are_refused() {
        let h = hash_hex("x");
        for (file, msg) in [
            (
                "[[token]]\nname='a'\nsha256='zz'\nusers=['u']\nprojects=['p']\npermissions=['read']",
                "64 hex",
            ),
            (
                &format!(
                    "[[token]]\nname='a'\nsha256='{h}'\nusers=[]\nprojects=['p']\npermissions=['read']"
                ),
                "users is empty",
            ),
            (
                &format!("[[token]]\nname='a'\nsha256='{h}'\nusers=['u']\npermissions=['read']"),
                "no project",
            ),
            (
                &format!(
                    "[[token]]\nname='a'\nsha256='{h}'\nusers=['u', '*']\nprojects=['p']\npermissions=['read']"
                ),
                "only entry",
            ),
            (
                &format!(
                    "[[token]]\nname='a'\nsha256='{h}'\nusers=['u']\nprojects=['p']\npermissions=['root']"
                ),
                "unknown variant",
            ),
        ] {
            let e = Auth::parse(file).unwrap_err();
            assert!(e.contains(msg), "{e}");
        }
    }

    #[test]
    fn new_tokens_are_random_and_hash_stably() {
        let (a, b) = (new_token(), new_token());
        assert_ne!(a, b);
        assert_eq!(hash_hex(&a), hash_hex(&a));
        assert_eq!(hash_hex(&a).len(), 64);
    }
}
