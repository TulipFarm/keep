//! Proxy mode: `keep mcp --server URL` forwards every call to one shared
//! `keep serve`, so separate agent processes share its Memory.

mod common;

use std::sync::Arc;

use common::Session;
use keep_server::http::router;
use keep_server::{KeepService, Scope, ServiceOptions};
use serde_json::json;

const TOKEN: &str = "proxy-test-token";

/// A daemon on a free loopback port, serving until the runtime is dropped.
fn daemon(rt: &tokio::runtime::Runtime) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let addr = listener.local_addr().unwrap();
    let svc = KeepService::in_memory(ServiceOptions::new(
        Scope::new("alex", Some("api".into())).unwrap(),
    ));
    let app = router(Arc::new(svc), Some(TOKEN.into()), 1 << 20);
    rt.spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

fn proxy(url: &str, extra: &[&str], token: Option<&str>) -> Session {
    let mut args = extra.to_vec();
    args.extend(["mcp", "--server", url]);
    let envs: Vec<(&str, &str)> = token.map(|t| ("KEEP_API_TOKEN", t)).into_iter().collect();
    let mut s = Session::spawn(&args, &envs);
    s.initialize();
    s
}

#[test]
fn two_agents_share_one_daemon() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let url = daemon(&rt);

    // Agent A names its scope; agent B relies on the daemon's defaults.
    let mut a = proxy(&url, &["--user", "alex", "--project", "api"], Some(TOKEN));
    let mut b = proxy(&url, &[], Some(TOKEN));
    let w = a.call(
        "memory_write",
        json!({ "path": "/topics/release.md", "text": "- Releases are cut on Thursdays.\n" }),
    );
    assert_eq!(w["created"], true);
    assert_eq!(w["scope"], "user:alex/project:api");
    let ctx = b.call("memory_context", json!({}));
    assert!(ctx["text"].as_str().unwrap().contains("Thursdays"), "{ctx}");

    // The daemon's compare-and-swap errors come back unchanged.
    let e = b.call_err(
        "memory_write",
        json!({ "path": "/topics/release.md", "text": "- clobber\n" }),
    );
    assert_eq!(e["code"], "conflict");
    assert_eq!(e["reason"], "EXPECTED_REVISION_REQUIRED");
    assert_eq!(e["details"]["current"]["revisionId"], w["revisionId"]);

    // A different explicit scope on A is isolated from B's default scope.
    let mut c = proxy(&url, &["--project", "web"], Some(TOKEN));
    let other = c.call("memory_context", json!({ "include_personal": false }));
    assert!(!other["text"].as_str().unwrap().contains("Thursdays"));
    assert_eq!(other["scope"], "user:alex/project:web");
}

#[test]
fn tools_and_errors_match_local_mode() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let url = daemon(&rt);
    let mut remote = proxy(&url, &[], Some(TOKEN));
    let mut local = Session::spawn(
        &[
            "--storage",
            "in-memory",
            "--user",
            "alex",
            "--project",
            "api",
            "mcp",
        ],
        &[],
    );
    local.initialize();
    assert_eq!(
        remote.request("tools/list", json!({})),
        local.request("tools/list", json!({}))
    );
    for (tool, args) in [
        ("memory_nope", json!({})),
        ("memory_read", json!({ "path": "/missing.md" })),
        ("memory_read", json!({ "bogus": true })),
        ("memory_context", json!({ "budget_tokens": 9000 })),
        ("knowledge_search", json!({ "query": " " })),
        ("memory_list", json!({ "scope": { "user": "Not Valid" } })),
    ] {
        assert_eq!(
            remote.call_err(tool, args.clone()),
            local.call_err(tool, args),
            "{tool}"
        );
    }
}

#[test]
fn proxy_reports_auth_and_connection_failures() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let url = daemon(&rt);
    let mut anonymous = proxy(&url, &[], None);
    let e = anonymous.call_err("memory_context", json!({}));
    assert_eq!(e["code"], "unauthorized");

    // Nothing listens on a port we just released.
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let mut lost = proxy(&format!("http://127.0.0.1:{port}"), &[], None);
    let e = lost.call_err("memory_context", json!({}));
    assert_eq!(e["code"], "unavailable");
    assert_eq!(e["retryable"], true);
    assert_eq!(e["reason"], "SERVER_UNREACHABLE");
}
