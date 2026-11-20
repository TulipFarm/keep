//! An MCP round trip over stdio against the real `keep` binary.

mod common;

use common::Session;
use serde_json::json;

#[test]
fn stdio_round_trip() {
    let mut s = Session::spawn(
        &[
            "--storage",
            "in-memory",
            "--user",
            "alex",
            "--project",
            "demo",
            "mcp",
        ],
        &[],
    );
    let init = s.initialize();
    assert!(init["capabilities"]["tools"].is_object(), "{init}");
    let names = s.tool_names();
    assert_eq!(names.len(), keep_server::api::TOOLS.len());
    for want in [
        "memory_context",
        "memory_write",
        "knowledge_ingest",
        "knowledge_search",
        "knowledge_cite",
        "knowledge_traverse",
        "knowledge_rebuild_projection",
        "memory_update_metadata",
    ] {
        assert!(names.iter().any(|n| n == want), "missing {want}: {names:?}");
    }

    let w = s.call(
        "memory_write",
        json!({ "path": "/topics/testing.md", "text": "- Run cargo test before committing.\n" }),
    );
    assert_eq!(w["created"], true);
    let ctx = s.call("memory_context", json!({}));
    assert!(
        ctx["text"].as_str().unwrap().contains("cargo test"),
        "{ctx}"
    );
    assert_eq!(ctx["scope"], "user:alex/project:demo");

    let stale = s.request(
        "tools/call",
        json!({ "name": "memory_write",
                "arguments": { "path": "/topics/testing.md", "text": "- other\n" } }),
    );
    assert_eq!(stale["isError"], true);
    assert_eq!(stale["structuredContent"]["error"]["code"], "conflict");

    s.call(
        "knowledge_ingest",
        json!({ "text": "The staging database is refreshed from production every Sunday night." }),
    );
    let found = s.call(
        "knowledge_search",
        json!({ "query": "when is staging refreshed" }),
    );
    assert!(!found["hits"].as_array().unwrap().is_empty(), "{found}");
}
