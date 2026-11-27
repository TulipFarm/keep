//! The JSON API over HTTP: a thin adapter from routes to
//! [`KeepService::call`].
//!
//! - `GET  /v1/health`: liveness and configuration, never needs a token.
//! - `GET  /v1/health/ready`: 200 when storage answers, else 503; no token.
//! - `GET  /v1/tools`: every operation with its route and input schema.
//! - `POST /v1/{area}/{op}`: run tool `{area}_{op}` with the JSON body as
//!   its arguments, such as `POST /v1/memory/write`.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};

use crate::api::{TOOLS, tool};
use crate::auth::{Auth, Principal};
use crate::backend::SESSION_HEADER;
use crate::config::check_bind;
use crate::error::{ApiError, ErrorCode};
use crate::service::KeepService;
use crate::tls::TlsListener;

pub const DEFAULT_MAX_BODY_BYTES: usize = 1 << 20;

struct AppState {
    service: Arc<KeepService>,
    auth: Auth,
}

/// The API's routes. `auth` is an [`Auth`], or an `Option<String>` for a
/// single administrator token.
pub fn router(service: Arc<KeepService>, auth: impl Into<Auth>, max_body_bytes: usize) -> Router {
    let state = Arc::new(AppState {
        service,
        auth: auth.into(),
    });
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/health/live", get(health))
        .route("/v1/health/ready", get(ready))
        .route("/v1/tools", get(tools))
        .route("/v1/{area}/{op}", post(call))
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

/// How `keep serve` listens.
pub struct ServeConfig {
    pub bind: SocketAddr,
    pub auth: Auth,
    pub max_body_bytes: usize,
    /// Serve HTTPS with this configuration, from [`crate::tls::server_config`].
    pub tls: Option<Arc<rustls::ServerConfig>>,
    /// Allow plain HTTP on a non-loopback address.
    pub insecure_http: bool,
    /// Also serve the graph view under `/dev`; see [`crate::dev`].
    pub dev: bool,
}

/// Listen until Ctrl-C. Refuses non-loopback addresses without a token, and
/// without TLS unless `insecure_http` is set.
pub async fn serve(service: Arc<KeepService>, config: ServeConfig) -> anyhow::Result<()> {
    let ServeConfig {
        bind,
        auth,
        max_body_bytes,
        tls,
        insecure_http,
        dev,
    } = config;
    if let Some(warning) = check_bind(&bind, !auth.is_open(), tls.is_some(), insecure_http)
        .map_err(anyhow::Error::msg)?
    {
        eprintln!("keep: warning: {warning}");
    }
    if dev {
        crate::config::check_dev(&bind, auth.is_open()).map_err(anyhow::Error::msg)?;
    }
    let tcp = tokio::net::TcpListener::bind(bind).await?;
    eprintln!(
        "keep: listening on {}://{} (token {})",
        if tls.is_some() { "https" } else { "http" },
        tcp.local_addr()?,
        if auth.is_open() {
            "not required"
        } else {
            "required"
        }
    );
    let mut app = router(service.clone(), auth, max_body_bytes);
    if dev {
        app = app.merge(crate::dev::routes(service));
        eprintln!(
            "keep: graph view on {}://{}/dev/graph",
            if tls.is_some() { "https" } else { "http" },
            tcp.local_addr()?
        );
    }
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    match tls {
        Some(config) => {
            axum::serve(TlsListener::new(tcp, config)?, app)
                .with_graceful_shutdown(shutdown)
                .await?
        }
        None => {
            axum::serve(tcp, app)
                .with_graceful_shutdown(shutdown)
                .await?
        }
    }
    Ok(())
}

struct ErrorResponse(ApiError);

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.0.code.http_status())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut res = (status, axum::Json(self.0.to_json())).into_response();
        if self.0.code == ErrorCode::Unauthorized {
            res.headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        res
    }
}

impl From<ApiError> for ErrorResponse {
    fn from(e: ApiError) -> Self {
        Self(e)
    }
}

async fn health(State(state): State<Arc<AppState>>) -> axum::Json<Value> {
    let mut info = state.service.info();
    info["status"] = json!("ok");
    axum::Json(info)
}

/// 200 when the ledger answers, 503 when it does not. No authentication,
/// like `/v1/health`, and the same content-free detail.
async fn ready(State(state): State<Arc<AppState>>) -> Response {
    let (ok, body) = state.service.readiness().await;
    let status = if ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, axum::Json(body)).into_response()
}

async fn tools(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<axum::Json<Value>, ErrorResponse> {
    guard_open(&state, &headers)?;
    authorize(&state, &headers)?;
    let list: Vec<Value> = TOOLS
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "method": "POST",
                "path": t.http_path(),
                "description": t.description,
                "readOnly": t.read_only,
                "destructive": t.destructive,
                "inputSchema": (t.input_schema)(),
            })
        })
        .collect();
    Ok(axum::Json(json!({ "tools": list })))
}

async fn call(
    State(state): State<Arc<AppState>>,
    Path((area, op)): Path<(String, String)>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<axum::Json<Value>, ErrorResponse> {
    guard_open(&state, &headers)?;
    let principal = authorize(&state, &headers)?;
    let name = format!("{area}_{op}");
    let Some(spec) = tool(&name) else {
        return Err(
            ApiError::not_found(format!("no operation at /v1/{area}/{op}"))
                .with_reason("UNKNOWN_ROUTE")
                .into(),
        );
    };
    let body = body.map_err(|e| {
        if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::new(ErrorCode::PayloadTooLarge, "request body is too large")
        } else {
            ApiError::invalid(format!("could not read the request body: {e}"))
        }
    })?;
    let args: Value = if body.iter().all(u8::is_ascii_whitespace) {
        Value::Null
    } else {
        require_json(&headers)?;
        serde_json::from_slice(&body)
            .map_err(|e| ApiError::invalid(format!("request body is not valid JSON: {e}")))?
    };
    if !(args.is_object() || args.is_null()) {
        return Err(ApiError::invalid("request body must be a JSON object").into());
    }
    let args = principal.authorize(spec, args, state.service.defaults())?;
    let session = headers
        .get(SESSION_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.len() <= 128);
    Ok(axum::Json(
        state
            .service
            .call_as(spec.name, args, principal.label(), session)
            .await?,
    ))
}

async fn fallback() -> ErrorResponse {
    ApiError::not_found("no such route")
        .with_reason("UNKNOWN_ROUTE")
        .into()
}

/// A server without a token trusts every caller that can reach it, so it
/// refuses what only a web browser would send: a cross-site page posting to
/// it (`Origin`), or a DNS-rebound name that resolves to this machine
/// (`Host`). Requests without a `Host` header come from in-process callers.
fn guard_open(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    if !state.auth.is_open() {
        return Ok(());
    }
    if headers.contains_key(header::ORIGIN) {
        return Err(ApiError::new(
            ErrorCode::PermissionDenied,
            "browser requests are refused when the server has no token",
        )
        .with_reason("BROWSER_ORIGIN"));
    }
    match headers.get(header::HOST).map(|h| h.to_str()) {
        None => Ok(()),
        Some(Ok(host)) if is_loopback_host(host) => Ok(()),
        Some(_) => Err(ApiError::new(
            ErrorCode::PermissionDenied,
            "the Host header must name this machine when the server has no token",
        )
        .with_reason("NON_LOOPBACK_HOST")),
    }
}

/// `localhost`, a loopback IP literal or a bracketed IPv6 one, with an
/// optional port.
pub(crate) fn is_loopback_host(host: &str) -> bool {
    let name = if let Some(rest) = host.strip_prefix('[') {
        match rest.split_once(']') {
            Some((ip, port)) if port.is_empty() || port.starts_with(':') => ip,
            _ => return false,
        }
    } else {
        host.rsplit_once(':').map_or(host, |(name, _)| name)
    };
    name.eq_ignore_ascii_case("localhost")
        || name
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// A body must be declared as JSON. Browsers cannot send that content type
/// to another site without a CORS preflight, which this server never grants.
fn require_json(headers: &HeaderMap) -> Result<(), ApiError> {
    let json = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("application/json"));
    if json {
        Ok(())
    } else {
        Err(
            ApiError::invalid("the request body must be sent as Content-Type: application/json")
                .with_reason("NOT_JSON"),
        )
    }
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Result<Principal, ApiError> {
    let given = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    state
        .auth
        .identify(given)
        .ok_or_else(|| ApiError::new(ErrorCode::Unauthorized, "missing or wrong bearer token"))
}
