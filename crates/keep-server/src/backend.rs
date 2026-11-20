//! Where tool calls run: in this process ([`KeepService`]) or on a shared
//! `keep serve` daemon ([`HttpBackend`]). The MCP front end talks to either
//! through [`Backend`], so both expose the same tools and error codes.

use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::StatusCode;
use serde_json::{Value, json};

use crate::api::{tool, unknown_tool};
use crate::error::{ApiError, ErrorCode};
use crate::service::KeepService;

/// Header carrying a random id per client process, so a daemon can tell
/// sessions apart (for example for per-session evidence budgets).
pub const SESSION_HEADER: &str = "x-keep-session";

#[async_trait]
pub trait Backend: Send + Sync {
    /// Run one tool. `args` is the tool's JSON arguments, or `null`.
    async fn call(&self, tool: &str, args: Value) -> Result<Value, ApiError>;
}

#[async_trait]
impl Backend for KeepService {
    async fn call(&self, tool: &str, args: Value) -> Result<Value, ApiError> {
        KeepService::call(self, tool, args).await
    }
}

/// Default scope fields a client fills in when a call leaves them out.
/// Unset fields fall through to the daemon's own defaults.
#[derive(Clone, Debug, Default)]
pub struct ScopeDefaults {
    pub user: Option<String>,
    pub project: Option<String>,
}

impl ScopeDefaults {
    fn apply(&self, args: &mut Value) {
        if self.user.is_none() && self.project.is_none() {
            return;
        }
        let Some(obj) = args.as_object_mut() else {
            return;
        };
        let scope = obj.entry("scope").or_insert_with(|| json!({}));
        let Some(scope) = scope.as_object_mut() else {
            return;
        };
        if let Some(u) = &self.user {
            scope.entry("user").or_insert_with(|| json!(u));
        }
        if let Some(p) = &self.project {
            scope.entry("project").or_insert_with(|| json!(p));
        }
    }
}

/// Forwards every call to `POST {base}/v1/{area}/{op}` on a `keep serve`
/// daemon, and turns its error bodies back into [`ApiError`]s.
pub struct HttpBackend {
    client: reqwest::Client,
    base: String,
    token: Option<String>,
    session: String,
    defaults: ScopeDefaults,
}

impl HttpBackend {
    /// `base` is the daemon's URL, such as `http://127.0.0.1:7357`.
    /// `ca_pem` adds a trusted root, for a daemon with a private
    /// certificate.
    pub fn new(
        base: &str,
        token: Option<String>,
        ca_pem: Option<&Path>,
        defaults: ScopeDefaults,
    ) -> anyhow::Result<Self> {
        let base = base.trim_end_matches('/').to_owned();
        let url = reqwest::Url::parse(&base)
            .map_err(|e| anyhow::anyhow!("invalid server URL {base:?}: {e}"))?;
        if !matches!(url.scheme(), "http" | "https") {
            anyhow::bail!("server URL must start with http:// or https://");
        }
        let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(600));
        if let Some(path) = ca_pem {
            let pem = std::fs::read(path)
                .map_err(|e| anyhow::anyhow!("reading CA certificate {}: {e}", path.display()))?;
            for cert in reqwest::Certificate::from_pem_bundle(&pem)? {
                builder = builder.add_root_certificate(cert);
            }
        }
        Ok(Self {
            client: builder.build()?,
            base,
            token,
            session: uuid::Uuid::new_v4().simple().to_string(),
            defaults,
        })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let mut req = self
            .client
            .request(method, format!("{}{path}", self.base))
            .header(SESSION_HEADER, &self.session);
        if let Some(t) = &self.token {
            req = req.bearer_auth(t);
        }
        req
    }

    /// `GET /v1/health`: the daemon's storage, scope defaults and version.
    pub async fn health(&self) -> Result<Value, ApiError> {
        let res = self
            .request(reqwest::Method::GET, "/v1/health")
            .send()
            .await
            .map_err(|e| self.unreachable(&e))?;
        decode(res).await
    }

    fn unreachable(&self, e: &reqwest::Error) -> ApiError {
        let what = if e.is_timeout() {
            "timed out"
        } else if e.is_connect() {
            "could not connect"
        } else {
            "request failed"
        };
        ApiError::new(
            ErrorCode::Unavailable,
            format!("keep server at {}: {what}", self.base),
        )
        .with_reason("SERVER_UNREACHABLE")
    }
}

#[async_trait]
impl Backend for HttpBackend {
    async fn call(&self, name: &str, mut args: Value) -> Result<Value, ApiError> {
        let Some(spec) = tool(name) else {
            return Err(unknown_tool(name));
        };
        if args.is_null() {
            args = json!({});
        }
        self.defaults.apply(&mut args);
        let res = self
            .request(reqwest::Method::POST, &spec.http_path())
            .json(&args)
            .send()
            .await
            .map_err(|e| self.unreachable(&e))?;
        decode(res).await
    }
}

async fn decode(res: reqwest::Response) -> Result<Value, ApiError> {
    let status = res.status();
    let body = res.bytes().await.map_err(|_| {
        ApiError::new(
            ErrorCode::Unavailable,
            "the keep server's response was cut off",
        )
    })?;
    let parsed: Option<Value> = serde_json::from_slice(&body).ok();
    if status.is_success() {
        return parsed.ok_or_else(|| {
            ApiError::new(ErrorCode::Internal, "the keep server returned invalid JSON")
        });
    }
    if let Some(e) = parsed.as_ref().and_then(ApiError::from_json) {
        return Err(e);
    }
    let code = match status {
        StatusCode::UNAUTHORIZED => ErrorCode::Unauthorized,
        StatusCode::FORBIDDEN => ErrorCode::PermissionDenied,
        StatusCode::NOT_FOUND => ErrorCode::NotFound,
        StatusCode::PAYLOAD_TOO_LARGE => ErrorCode::PayloadTooLarge,
        s if s.is_server_error() => ErrorCode::Unavailable,
        _ => ErrorCode::Internal,
    };
    Err(ApiError::new(
        code,
        format!("the keep server answered HTTP {}", status.as_u16()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_fill_only_missing_fields() {
        let d = ScopeDefaults {
            user: Some("alex".into()),
            project: Some("api".into()),
        };
        let mut a = json!({});
        d.apply(&mut a);
        assert_eq!(a["scope"], json!({ "user": "alex", "project": "api" }));
        let mut a = json!({ "scope": { "project": "" } });
        d.apply(&mut a);
        assert_eq!(a["scope"], json!({ "user": "alex", "project": "" }));
        let mut a = json!({ "scope": { "user": "sam", "project": "web" } });
        d.apply(&mut a);
        assert_eq!(a["scope"], json!({ "user": "sam", "project": "web" }));
        let mut a = json!({});
        ScopeDefaults::default().apply(&mut a);
        assert_eq!(a, json!({}));
    }
}
