//! `keep serve --dev`: a page at `/dev/graph` that draws one scope's Memory
//! and Knowledge as a graph, for people developing with Keep.
//!
//! - `GET /dev/graph`: the page. `?user=` and `?project=` pick the scope.
//! - `GET /dev/graph.js`, `GET /dev/graph.css`: the page's script and
//!   styles, built from `dev-ui/`.
//! - `GET /dev/graph.json`: the graph, in the format of [`crate::graph`].
//!
//! Dev mode runs only on a loopback address with no token (see
//! [`crate::config::check_dev`]), so these routes carry no authentication.
//! Like the open API they refuse a `Host` that is not this machine and a
//! page from another origin.

use std::sync::Arc;

use axum::Router;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde::Deserialize;
use serde_json::json;

use crate::scope::ScopeArg;
use crate::service::KeepService;

pub const DEFAULT_MAX_FACTS: usize = 5_000;
pub const MAX_FACTS: usize = 50_000;

const PAGE: &str = include_str!("../dev-ui/graph.html");
const SCRIPT: &str = include_str!("../dev-ui/dist/graph.js");
const STYLE: &str = include_str!("../dev-ui/graph.css");
const CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; \
                   connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'";

pub fn routes(service: Arc<KeepService>) -> Router {
    Router::new()
        .route("/dev/graph", get(page))
        .route("/dev/graph.js", get(script))
        .route("/dev/graph.css", get(style))
        .route("/dev/graph.json", get(data))
        .with_state(service)
}

#[derive(Debug, Default, Deserialize)]
struct GraphQuery {
    user: Option<String>,
    project: Option<String>,
    max_facts: Option<usize>,
}

async fn page(headers: HeaderMap) -> Response {
    if let Some(r) = guard(&headers) {
        return r;
    }
    with_headers(PAGE.into_response(), "text/html; charset=utf-8", Some(CSP))
}

async fn script(headers: HeaderMap) -> Response {
    if let Some(r) = guard(&headers) {
        return r;
    }
    with_headers(
        SCRIPT.into_response(),
        "text/javascript; charset=utf-8",
        None,
    )
}

async fn style(headers: HeaderMap) -> Response {
    if let Some(r) = guard(&headers) {
        return r;
    }
    with_headers(STYLE.into_response(), "text/css; charset=utf-8", None)
}

async fn data(
    State(service): State<Arc<KeepService>>,
    headers: HeaderMap,
    Query(q): Query<GraphQuery>,
) -> Response {
    if let Some(r) = guard(&headers) {
        return r;
    }
    let max_facts = q.max_facts.unwrap_or(DEFAULT_MAX_FACTS);
    if !(1..=MAX_FACTS).contains(&max_facts) {
        return refuse(
            StatusCode::BAD_REQUEST,
            &format!("max_facts must be between 1 and {MAX_FACTS}"),
        );
    }
    let scope = (q.user.is_some() || q.project.is_some()).then_some(ScopeArg {
        user: q.user,
        project: q.project,
    });
    match service.graph(scope.as_ref(), max_facts).await {
        Ok(g) => with_headers(axum::Json(g).into_response(), "application/json", None),
        Err(e) => {
            let status = StatusCode::from_u16(e.code.http_status())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            (status, axum::Json(e.to_json())).into_response()
        }
    }
}

/// Returns a refusal when the request does not come from this machine.
fn guard(headers: &HeaderMap) -> Option<Response> {
    let host = headers.get(header::HOST).and_then(|h| h.to_str().ok());
    let Some(host) = host.filter(|h| crate::http::is_loopback_host(h)) else {
        return Some(refuse(
            StatusCode::FORBIDDEN,
            "the Host header must name this machine",
        ));
    };
    if let Some(origin) = headers.get(header::ORIGIN) {
        let same = origin
            .to_str()
            .ok()
            .and_then(|o| o.split_once("://"))
            .is_some_and(|(_, rest)| rest == host);
        if !same {
            return Some(refuse(
                StatusCode::FORBIDDEN,
                "requests from another site are refused",
            ));
        }
    }
    None
}

fn refuse(status: StatusCode, message: &str) -> Response {
    (status, axum::Json(json!({ "error": message }))).into_response()
}

fn with_headers(
    mut res: Response,
    content_type: &'static str,
    csp: Option<&'static str>,
) -> Response {
    let h = res.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    h.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    if let Some(csp) = csp {
        h.insert(
            header::CONTENT_SECURITY_POLICY,
            HeaderValue::from_static(csp),
        );
    }
    res
}
