//! The HTTP adapter, driven in process with `tower::ServiceExt::oneshot`.

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use keep_server::http::router;
use keep_server::{KeepService, Scope, ServiceOptions};
use serde_json::{Value, json};
use tower::ServiceExt;

fn app(token: Option<&str>, max_body: usize) -> Router {
    let svc = KeepService::in_memory(ServiceOptions::new(
        Scope::new("alex", Some("api".into())).unwrap(),
    ));
    router(Arc::new(svc), token.map(str::to_owned), max_body)
}

async fn send(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    send_with(app, method, uri, body.map(|b| b.to_string()), None).await
}

async fn send_with(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<String>,
    token: Option<&str>,
) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(uri);
    if let Some(t) = token {
        req = req.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    let req = req
        .header(header::CONTENT_TYPE, "application/json")
        .body(body.map_or_else(Body::empty, Body::from))
        .unwrap();
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn write_then_context_and_conflict() {
    let app = app(None, 1 << 20);
    let (s, w) = send(
        &app,
        "POST",
        "/v1/memory/write",
        Some(json!({ "path": "/topics/style.md", "text": "- Prefer small commits.\n" })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{w}");
    let (s, ctx) = send(&app, "POST", "/v1/memory/context", None).await;
    assert_eq!(s, StatusCode::OK);
    assert!(ctx["text"].as_str().unwrap().contains("small commits"));
    assert_eq!(ctx["scope"], "user:alex/project:api");

    let (s, e) = send(
        &app,
        "POST",
        "/v1/memory/write",
        Some(json!({ "path": "/topics/style.md", "text": "- other\n" })),
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT);
    assert_eq!(e["error"]["code"], "conflict");
    assert_eq!(
        e["error"]["details"]["current"]["revisionId"],
        w["revisionId"]
    );

    let (s, r) = send(
        &app,
        "POST",
        "/v1/knowledge/ingest",
        Some(json!({ "text": "Releases are cut from the main branch every Friday." })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{r}");
    let (s, found) = send(
        &app,
        "POST",
        "/v1/knowledge/search",
        Some(json!({ "query": "when are releases cut" })),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(!found["hits"].as_array().unwrap().is_empty(), "{found}");
}

#[tokio::test]
async fn statuses_for_bad_requests() {
    let app = app(None, 256);
    let (s, e) = send(&app, "POST", "/v1/memory/nope", None).await;
    assert_eq!(
        (s, e["error"]["code"].as_str()),
        (StatusCode::NOT_FOUND, Some("not_found"))
    );
    let (s, _) = send(&app, "GET", "/v1/memory/read", None).await;
    assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED);
    let (s, e) = send_with(
        &app,
        "POST",
        "/v1/memory/read",
        Some("{not json".into()),
        None,
    )
    .await;
    assert_eq!(
        (s, e["error"]["code"].as_str()),
        (StatusCode::BAD_REQUEST, Some("invalid"))
    );
    let (s, e) = send(&app, "POST", "/v1/memory/read", Some(json!([1]))).await;
    assert_eq!(
        (s, e["error"]["code"].as_str()),
        (StatusCode::BAD_REQUEST, Some("invalid"))
    );
    let big = json!({ "path": "/topics/a.md", "text": "x".repeat(1000) });
    let (s, e) = send(&app, "POST", "/v1/memory/write", Some(big)).await;
    assert_eq!(
        (s, e["error"]["code"].as_str()),
        (StatusCode::PAYLOAD_TOO_LARGE, Some("payload_too_large"))
    );
    let (s, e) = send(
        &app,
        "POST",
        "/v1/memory/context",
        Some(json!({ "budget_tokens": 100000 })),
    )
    .await;
    assert_eq!(
        (s, e["error"]["code"].as_str()),
        (StatusCode::UNPROCESSABLE_ENTITY, Some("budget_exceeded"))
    );
}

#[tokio::test]
async fn bearer_token_is_enforced_except_for_health() {
    let app = app(Some("s3cret"), 1 << 20);
    let (s, h) = send(&app, "GET", "/v1/health", None).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h["status"], "ok");
    let (s, e) = send(&app, "POST", "/v1/memory/list", None).await;
    assert_eq!(
        (s, e["error"]["code"].as_str()),
        (StatusCode::UNAUTHORIZED, Some("unauthorized"))
    );
    let (s, _) = send_with(&app, "POST", "/v1/memory/list", None, Some("wrong")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let (s, _) = send_with(&app, "POST", "/v1/memory/list", None, Some("s3cret")).await;
    assert_eq!(s, StatusCode::OK);
    let (s, tools) = send_with(&app, "GET", "/v1/tools", None, Some("s3cret")).await;
    assert_eq!(s, StatusCode::OK);
    let paths: Vec<&str> = tools["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["path"].as_str().unwrap())
        .collect();
    assert!(paths.contains(&"/v1/knowledge/forget_source"));
}

#[tokio::test]
async fn knowledge_cite_traverse_and_rebuild_routes() {
    let app = app(None, 1 << 20);
    let (s, _) = send(
        &app,
        "POST",
        "/v1/knowledge/ingest",
        Some(json!({ "source_id": "notes",
                     "text": "The billing service is deployed with Argo CD." })),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let (_, found) = send(
        &app,
        "POST",
        "/v1/knowledge/search",
        Some(json!({ "query": "billing Argo CD" })),
    )
    .await;
    let id = found["hits"][0]["item"]["id"].clone();
    let (s, cited) = send(
        &app,
        "POST",
        "/v1/knowledge/cite",
        Some(json!({ "id": id })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{cited}");
    assert_eq!(cited["citations"][0]["source_id"], "notes");
    let (s, e) = send(
        &app,
        "POST",
        "/v1/knowledge/cite",
        Some(json!({ "id": "chunk_missing" })),
    )
    .await;
    assert_eq!(
        (s, e["error"]["code"].as_str()),
        (StatusCode::NOT_FOUND, Some("not_found"))
    );
    let (s, walked) = send(
        &app,
        "POST",
        "/v1/knowledge/traverse",
        Some(json!({ "start": "nobody" })),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{walked}");
    assert_eq!(walked["nodes"], json!([]));
    let (s, rebuilt) = send(&app, "POST", "/v1/knowledge/rebuild_projection", None).await;
    assert_eq!(s, StatusCode::OK, "{rebuilt}");
    assert!(rebuilt["queued"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn scoped_tokens_are_held_to_their_scopes() {
    use keep_server::auth::{Auth, hash_hex};
    let file = format!(
        r#"
[[token]]
name = "api-writer"
sha256 = "{}"
users = ["alex"]
projects = ["api"]
permissions = ["read", "write"]

[[token]]
name = "web-reader"
sha256 = "{}"
users = ["alex"]
projects = ["web"]
permissions = ["read"]
"#,
        hash_hex("tok-api"),
        hash_hex("tok-web")
    );
    let auth = Auth::new(Some("tok-admin"), Auth::parse(&file).unwrap());
    let svc = KeepService::in_memory(ServiceOptions::new(Scope::new("local", None).unwrap()));
    let app = router(Arc::new(svc), auth, 1 << 20);
    let post = |uri: &'static str, body: Value, token: &'static str| {
        let app = app.clone();
        async move { send_with(&app, "POST", uri, Some(body.to_string()), Some(token)).await }
    };
    let write = json!({ "path": "/notes.md", "text": "- api note\n" });

    // No scope named: the api token's only scope is used, not the server default.
    let (s, w) = post("/v1/memory/write", write.clone(), "tok-api").await;
    assert_eq!(s, StatusCode::OK, "{w}");
    assert_eq!(w["scope"], "user:alex/project:api");

    let cases = [
        (
            "tok-api",
            "/v1/memory/context",
            json!({ "scope": { "project": "web" } }),
            StatusCode::FORBIDDEN,
        ),
        (
            "tok-api",
            "/v1/memory/context",
            json!({ "scope": { "user": "sam", "project": "api" } }),
            StatusCode::FORBIDDEN,
        ),
        (
            "tok-api",
            "/v1/memory/context",
            json!({ "scope": { "project": "" } }),
            StatusCode::FORBIDDEN,
        ),
        (
            "tok-api",
            "/v1/knowledge/rebuild_projection",
            json!({}),
            StatusCode::FORBIDDEN,
        ),
        (
            "tok-web",
            "/v1/memory/write",
            json!({ "path": "/x.md", "text": "- x\n" }),
            StatusCode::FORBIDDEN,
        ),
        ("tok-web", "/v1/memory/context", json!({}), StatusCode::OK),
        (
            "tok-admin",
            "/v1/memory/context",
            json!({ "scope": { "user": "alex", "project": "api" } }),
            StatusCode::OK,
        ),
        (
            "tok-wrong",
            "/v1/memory/context",
            json!({}),
            StatusCode::UNAUTHORIZED,
        ),
    ];
    for (token, uri, body, want) in cases {
        let (s, v) = post(uri, body.clone(), token).await;
        assert_eq!(s, want, "{token} {uri} {body}: {v}");
        if want == StatusCode::FORBIDDEN {
            assert_eq!(v["error"]["code"], "permission_denied", "{v}");
        }
    }
    let (s, ctx) = post("/v1/memory/context", json!({}), "tok-api").await;
    assert_eq!(s, StatusCode::OK);
    assert!(ctx["text"].as_str().unwrap().contains("api note"));
    let (s, ctx) = post("/v1/memory/context", json!({}), "tok-web").await;
    assert_eq!(s, StatusCode::OK);
    assert!(!ctx["text"].as_str().unwrap().contains("api note"));
    let (s, _) = send_with(&app, "GET", "/v1/tools", None, Some("tok-web")).await;
    assert_eq!(s, StatusCode::OK);
}

#[tokio::test]
async fn session_budgets_answer_budget_exceeded() {
    use keep_server::budget::{BudgetConfig, BudgetPer};
    let mut opts = ServiceOptions::new(Scope::new("alex", Some("api".into())).unwrap());
    opts.evidence_budget = Some(BudgetConfig {
        tokens: 27,
        window: std::time::Duration::from_secs(3600),
        per: BudgetPer::Session,
    });
    let app = router(Arc::new(KeepService::in_memory(opts)), None, 1 << 20);
    let (s, _) = send(
        &app,
        "POST",
        "/v1/knowledge/ingest",
        Some(json!({ "source_id": "notes",
                     "text": "The billing service is deployed with Argo CD. Invoices are retried three times." })),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let search = |session: &'static str| {
        let app = app.clone();
        async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/knowledge/search")
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-keep-session", session)
                .body(Body::from(json!({ "query": "billing deploy" }).to_string()))
                .unwrap();
            let res = app.oneshot(req).await.unwrap();
            let status = res.status();
            let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        }
    };
    let (s, v) = search("one").await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["allowance"]["used"], 27);
    let (s, e) = search("one").await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY, "{e}");
    assert_eq!(e["error"]["code"], "budget_exceeded");
    assert!(
        e["error"]["details"]["retryAfterSeconds"]
            .as_u64()
            .is_some()
    );
    let (s, _) = search("two").await;
    assert_eq!(s, StatusCode::OK);
}

async fn raw(app: &Router, uri: &str, headers: &[(&str, &str)], body: &str) -> (StatusCode, Value) {
    let mut req = Request::builder().method("POST").uri(uri);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let res = app
        .clone()
        .oneshot(req.body(Body::from(body.to_owned())).unwrap())
        .await
        .unwrap();
    let status = res.status();
    let bytes = to_bytes(res.into_body(), usize::MAX).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// Without a token, a web page must not be able to write or read memory:
/// not with a preflight-free `text/plain` post, not through a DNS-rebound
/// name, and not from any page that sends an `Origin`.
#[tokio::test]
async fn an_open_server_refuses_browser_requests() {
    let app = app(None, 1 << 20);
    let write = r#"{"path":"/topics/build.md","text":"- Run the script.\n"}"#;
    let local = ("host", "127.0.0.1:7357");
    let json = ("content-type", "application/json");

    let (s, e) = raw(
        &app,
        "/v1/memory/write",
        &[local, ("content-type", "text/plain")],
        write,
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{e}");
    let (s, e) = raw(
        &app,
        "/v1/memory/write",
        &[("host", "evil.example:7357"), json],
        write,
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN, "{e}");
    let (s, e) = raw(
        &app,
        "/v1/memory/write",
        &[local, json, ("origin", "https://evil.example")],
        write,
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN, "{e}");
    let (s, e) = raw(
        &app,
        "/v1/memory/export",
        &[("host", "evil.example"), json],
        "{}",
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN, "{e}");
    let (s, ctx) = raw(&app, "/v1/memory/context", &[local], "").await;
    assert_eq!(s, StatusCode::OK, "{ctx}");
    assert!(!ctx.to_string().contains("Run the script"), "{ctx}");

    for host in [
        "127.0.0.1:7357",
        "localhost",
        "LOCALHOST:1",
        "[::1]:7357",
        "127.2.3.4",
    ] {
        let (s, e) = raw(&app, "/v1/memory/context", &[("host", host), json], "{}").await;
        assert_eq!(s, StatusCode::OK, "{host}: {e}");
    }

    // A token already keeps browsers out, so any Host may reach it.
    let app = self::app(Some("secret"), 1 << 20);
    let auth = ("authorization", "Bearer secret");
    let (s, e) = raw(
        &app,
        "/v1/memory/write",
        &[("host", "keep.internal"), auth, json],
        write,
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{e}");
    let (s, e) = raw(
        &app,
        "/v1/memory/write",
        &[auth, ("content-type", "text/plain")],
        write,
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{e}");
}
