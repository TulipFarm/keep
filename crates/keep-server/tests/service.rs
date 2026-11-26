//! The service layer, end to end: in memory always, and against Postgres
//! (pgvector) and Postgres + Qdrant when `KEEP_TEST_POSTGRES_URL` and
//! `KEEP_TEST_QDRANT_URL` are set.

use keep_server::budget::BudgetPer;
use keep_server::config::{EmbedderKind, ServiceArgs, StorageProfile};
use keep_server::{ErrorCode, KeepService, Scope, ServiceOptions};
use serde_json::{Value, json};

fn in_memory() -> KeepService {
    KeepService::in_memory(ServiceOptions::new(Scope::new("alex", None).unwrap()))
}

fn api() -> Value {
    json!({ "user": "alex", "project": "api" })
}

fn web() -> Value {
    json!({ "user": "alex", "project": "web" })
}

fn personal() -> Value {
    json!({ "user": "alex" })
}

async fn ok(svc: &KeepService, tool: &str, args: Value) -> Value {
    match svc.call(tool, args).await {
        Ok(v) => v,
        Err(e) => panic!("{tool} failed: {e:?}"),
    }
}

async fn err(svc: &KeepService, tool: &str, args: Value) -> keep_server::ApiError {
    match svc.call(tool, args).await {
        Ok(v) => panic!("{tool} should fail, got {v}"),
        Err(e) => e,
    }
}

/// Two agents sharing a project scope, a second project, and the user's
/// personal scope, through every tool.
async fn shared_memory_scenario(svc: &KeepService) {
    // Agent A creates a file; agent B, in the same scope, reads it.
    let created = ok(
        svc,
        "memory_write",
        json!({ "scope": api(), "path": "/topics/deploys.md",
                "text": "# Deploys\n\n- Deploys go out on Tuesdays.\n",
                "description": "How we ship" }),
    )
    .await;
    assert_eq!(created["created"], true);
    let read = ok(
        svc,
        "memory_read",
        json!({ "scope": api(), "path": "/topics/deploys.md", "include_entries": true }),
    )
    .await;
    assert!(read["text"].as_str().unwrap().contains("Tuesdays"));
    let rev1 = read["revisionId"].as_str().unwrap().to_owned();
    assert_eq!(rev1, created["revisionId"]);
    assert!(!read["entries"].as_array().unwrap().is_empty());

    // Overwriting without a revision is a conflict that carries the head.
    let e = err(
        svc,
        "memory_write",
        json!({ "scope": api(), "path": "/topics/deploys.md", "text": "clobber" }),
    )
    .await;
    assert_eq!(e.code, ErrorCode::Conflict);
    assert_eq!(e.reason.as_deref(), Some("EXPECTED_REVISION_REQUIRED"));
    assert_eq!(e.details.as_ref().unwrap()["current"]["revisionId"], rev1);

    // Compare-and-swap: the first writer wins, the stale one conflicts.
    let second = ok(
        svc,
        "memory_write",
        json!({ "scope": api(), "path": "/topics/deploys.md", "expected_revision_id": rev1,
                "edit": { "command": "str_replace", "old": "Tuesdays", "new": "Wednesdays" } }),
    )
    .await;
    let rev2 = second["revisionId"].as_str().unwrap().to_owned();
    let e = err(
        svc,
        "memory_write",
        json!({ "scope": api(), "path": "/topics/deploys.md", "expected_revision_id": rev1,
                "text": "# Deploys\n\n- Stale.\n" }),
    )
    .await;
    assert_eq!(e.code, ErrorCode::Conflict);
    assert_eq!(e.reason.as_deref(), Some("VERSION_CONFLICT"));
    assert_eq!(e.details.as_ref().unwrap()["current"]["revisionId"], rev2);

    // History, then undo back to the first text.
    let history = ok(
        svc,
        "memory_history",
        json!({ "scope": api(), "path": "/topics/deploys.md" }),
    )
    .await;
    assert_eq!(history["revisions"].as_array().unwrap().len(), 2);
    ok(
        svc,
        "memory_undo",
        json!({ "scope": api(), "path": "/topics/deploys.md", "expected_revision_id": rev2 }),
    )
    .await;
    let read = ok(
        svc,
        "memory_read",
        json!({ "scope": api(), "path": "/topics/deploys.md" }),
    )
    .await;
    assert!(read["text"].as_str().unwrap().contains("Tuesdays"));

    // The project's context shows the file; another project's does not.
    let ctx = ok(svc, "memory_context", json!({ "scope": api() })).await;
    assert!(ctx["text"].as_str().unwrap().contains("Tuesdays"));
    assert!(ctx["tokens"].as_u64().unwrap() <= 4096);
    let other = ok(svc, "memory_context", json!({ "scope": web() })).await;
    assert!(!other["text"].as_str().unwrap().contains("Tuesdays"));
    let list = ok(svc, "memory_list", json!({ "scope": web() })).await;
    assert_eq!(list["total"], 0);
    let list = ok(svc, "memory_list", json!({ "scope": api() })).await;
    assert_eq!(list["total"], 1);
    assert!(
        list["tree"]
            .as_str()
            .unwrap()
            .contains("/topics/deploys.md")
    );

    // Knowledge: each project writes its own partition; the personal
    // partition is visible from every project of the user.
    let billing = ok(
        svc,
        "knowledge_ingest",
        json!({ "scope": api(), "source_id": "billing-runbook", "collection": "runbooks",
                "title": "Billing runbook",
                "text": "The billing service is deployed with Argo CD. Invoices are retried three times.",
                "occurred_at": "2025-01-10" }),
    )
    .await;
    assert_eq!(billing["partition"], "project:api");
    assert!(billing["report"]["chunks_added"].as_u64().unwrap() > 0);
    assert_eq!(billing["captured"], true);
    assert_eq!(billing["enrichment"]["state"], "complete", "{billing}");
    // A write skips finished jobs, so it reports no `succeeded` count.
    assert!(
        billing["enrichment"]["jobs"].get("succeeded").is_none(),
        "{billing}"
    );
    ok(
        svc,
        "knowledge_ingest",
        json!({ "scope": web(), "source_id": "web-notes",
                "text": "The marketing site is built with Vite and deployed to a CDN." }),
    )
    .await;
    ok(
        svc,
        "knowledge_ingest",
        json!({ "scope": personal(),
                "episodes": [
                    { "speaker": "alex", "text": "I prefer tabs over spaces in every repository.",
                      "occurred_at": "2025-02-01T09:00:00Z" }
                ] }),
    )
    .await;

    let hits = |v: &Value| -> Vec<String> {
        v["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["source_id"].as_str().unwrap().to_owned())
            .collect()
    };
    let found = ok(
        svc,
        "knowledge_search",
        json!({ "scope": api(), "query": "How is billing deployed? Argo CD", "budget_tokens": 1000 }),
    )
    .await;
    assert!(
        hits(&found).contains(&"billing-runbook".to_owned()),
        "{found}"
    );
    assert!(found["allowance"]["used"].as_u64().unwrap() <= 1000);
    let vite = ok(
        svc,
        "knowledge_search",
        json!({ "scope": api(), "query": "Vite marketing site CDN" }),
    )
    .await;
    assert!(!hits(&vite).contains(&"web-notes".to_owned()), "{vite}");
    let tabs = ok(
        svc,
        "knowledge_search",
        json!({ "scope": api(), "query": "tabs or spaces" }),
    )
    .await;
    assert!(
        hits(&tabs).iter().any(|s| s.starts_with("src_")),
        "personal knowledge is visible from a project: {tabs}"
    );

    // Fetch a hit's evidence; the other project cannot.
    let id = found["hits"][0]["item"]["id"].as_str().unwrap().to_owned();
    let fetched = ok(svc, "knowledge_fetch", json!({ "scope": api(), "id": id })).await;
    assert!(!fetched["text"].as_str().unwrap().is_empty());
    let e = err(svc, "knowledge_fetch", json!({ "scope": web(), "id": id })).await;
    assert_eq!(e.code, ErrorCode::NotFound);

    // Citations resolve a hit to its quoted span, within the scope only.
    let cited = ok(svc, "knowledge_cite", json!({ "scope": api(), "id": id })).await;
    let citation = &cited["citations"][0];
    assert_eq!(citation["source_id"], "billing-runbook", "{cited}");
    assert!(!citation["quote"].as_str().unwrap().is_empty());
    let e = err(svc, "knowledge_cite", json!({ "scope": web(), "id": id })).await;
    assert_eq!(e.code, ErrorCode::NotFound);

    // The graph around the user, from the personal fact.
    let walked = ok(
        svc,
        "knowledge_traverse",
        json!({ "scope": api(), "start": "alex", "depth": 2 }),
    )
    .await;
    assert_eq!(walked["nodes"][0]["name"], "alex", "{walked}");
    assert_eq!(walked["edges"].as_array().unwrap().len(), 1, "{walked}");
    let e = err(
        svc,
        "knowledge_traverse",
        json!({ "scope": api(), "start": "billing", "depth": 3 }),
    )
    .await;
    assert_eq!(e.code, ErrorCode::Invalid);

    // Rebuilding the projection re-indexes everything and search still works.
    let rebuilt = ok(
        svc,
        "knowledge_rebuild_projection",
        json!({ "scope": api() }),
    )
    .await;
    assert!(rebuilt["queued"].as_u64().unwrap() > 0, "{rebuilt}");
    assert_eq!(rebuilt["jobs"]["failed"], 0, "{rebuilt}");
    let again = ok(
        svc,
        "knowledge_search",
        json!({ "scope": api(), "query": "How is billing deployed? Argo CD" }),
    )
    .await;
    assert!(hits(&again).contains(&"billing-runbook".to_owned()));

    // The index lists this project's and the personal collections.
    let index = ok(svc, "knowledge_index", json!({ "scope": api() })).await;
    assert_eq!(index["coverage"]["sources"], 2, "{index}");
    assert!(index["text"].as_str().unwrap().contains("runbooks"));
    let ctx = ok(svc, "memory_context", json!({ "scope": api() })).await;
    assert_eq!(ctx["knowledgeIncluded"], true, "{ctx}");

    // Enrichment status counts only the scope's own jobs.
    let api_status = ok(svc, "knowledge_status", json!({ "scope": api() })).await;
    assert_eq!(
        api_status["enrichment"]["state"], "complete",
        "{api_status}"
    );
    let web_status = ok(svc, "knowledge_status", json!({ "scope": web() })).await;
    let done = |v: &Value| v["enrichment"]["jobs"]["succeeded"].as_i64().unwrap();
    assert!(done(&web_status) > 0 && done(&api_status) > 0);
    let other = ok(
        svc,
        "knowledge_status",
        json!({ "scope": { "user": "alex", "project": "unused" } }),
    )
    .await;
    assert_eq!(done(&other), 0, "{other}");

    // Export returns the scope's live files with a manifest.
    let export = ok(svc, "memory_export", json!({ "scope": api() })).await;
    assert_eq!(export["files"].as_array().unwrap().len(), 1, "{export}");
    assert!(
        export["files"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Tuesdays")
    );
    assert_eq!(
        export["manifest"]["files"][0]["path"], "/topics/deploys.md",
        "{export}"
    );
    let export = ok(svc, "memory_export", json!({ "scope": web() })).await;
    assert!(export["files"].as_array().unwrap().is_empty(), "{export}");

    // Only the owning scope may erase a source.
    let e = err(
        svc,
        "knowledge_forget_source",
        json!({ "scope": api(), "source_id": "web-notes" }),
    )
    .await;
    assert_eq!(e.code, ErrorCode::NotFound);
    let e = err(
        svc,
        "knowledge_ingest",
        json!({ "scope": api(), "source_id": "web-notes", "text": "hijack" }),
    )
    .await;
    assert_eq!(e.code, ErrorCode::Conflict);
    let forgot = ok(
        svc,
        "knowledge_forget_source",
        json!({ "scope": web(), "source_id": "web-notes" }),
    )
    .await;
    assert!(forgot["receipt"]["chunks"].as_i64().unwrap() > 0);
    // A rebuilt projection does not bring the forgotten source back.
    ok(
        svc,
        "knowledge_rebuild_projection",
        json!({ "scope": web() }),
    )
    .await;
    let vite = ok(
        svc,
        "knowledge_search",
        json!({ "scope": web(), "query": "Vite marketing site CDN" }),
    )
    .await;
    assert!(!hits(&vite).contains(&"web-notes".to_owned()), "{vite}");
    let index = ok(svc, "knowledge_index", json!({ "scope": web() })).await;
    assert!(
        !index["text"].as_str().unwrap().contains("web-notes"),
        "{index}"
    );
    assert_eq!(index["coverage"]["sources"], 1, "personal only: {index}");

    // Forgetting a memory file erases it everywhere and returns a receipt.
    let receipt = ok(
        svc,
        "memory_forget",
        json!({ "scope": api(), "path": "/topics/deploys.md" }),
    )
    .await;
    assert_eq!(
        receipt["receipt"]["filesPurged"].as_array().unwrap().len(),
        1
    );
    let e = err(
        svc,
        "memory_read",
        json!({ "scope": api(), "path": "/topics/deploys.md" }),
    )
    .await;
    assert_eq!(e.code, ErrorCode::NotFound);
    // Nothing of it survives in history, context or export.
    let e = err(
        svc,
        "memory_history",
        json!({ "scope": api(), "path": "/topics/deploys.md" }),
    )
    .await;
    assert_eq!(e.code, ErrorCode::NotFound);
    let ctx = ok(svc, "memory_context", json!({ "scope": api() })).await;
    assert!(!ctx["text"].as_str().unwrap().contains("Tuesdays"), "{ctx}");
    let export = ok(svc, "memory_export", json!({ "scope": api() })).await;
    assert!(export["files"].as_array().unwrap().is_empty(), "{export}");
}

/// A host application: long composite scope ids, per-read source
/// allowlists, source metadata and replacing a source with a new revision.
async fn host_scenario(svc: &KeepService) {
    let ws = "ws-3f2b9c1e-8d4a-4f6b-9e2d-1a7c5b3e9f00";
    let member = format!("{ws}-m-0b8e6d4c-2a1f-4e3d-8c5b-7a9f1e2d3c4b");
    let scope = json!({ "user": ws, "project": member });
    let ingest = |id: &str, collection: &str, text: &str, extra: Value| {
        let mut a = json!({ "scope": scope, "source_id": id, "collection": collection,
                            "title": id, "text": text });
        for (k, v) in extra.as_object().unwrap() {
            a[k] = v.clone();
        }
        a
    };
    let doc_a = ok(
        svc,
        "knowledge_ingest",
        ingest(
            "doc-a",
            "handbook",
            "The quarterly offsite is held in Lisbon. Parking passes come from the front desk.",
            json!({ "metadata": { "url": "https://docs.example/a", "doc": "a" } }),
        ),
    )
    .await;
    assert_eq!(doc_a["replaced"], false, "{doc_a}");
    ok(
        svc,
        "knowledge_ingest",
        ingest(
            "doc-b",
            "policies",
            "The quarterly offsite budget is approved by finance. Lisbon hotels are booked early.",
            json!({}),
        ),
    )
    .await;
    let sources = |v: &Value| -> Vec<String> {
        let mut s: Vec<String> = v["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["source_id"].as_str().unwrap().to_owned())
            .collect();
        s.sort();
        s.dedup();
        s
    };
    let query = "quarterly offsite Lisbon";
    let all = ok(
        svc,
        "knowledge_search",
        json!({ "scope": scope, "query": query }),
    )
    .await;
    assert_eq!(sources(&all), ["doc-a", "doc-b"], "{all}");

    // The allowlist narrows retrieval; metadata rides on the citation.
    let only_a = ok(
        svc,
        "knowledge_search",
        json!({ "scope": scope, "query": query, "source_ids": ["doc-a"], "limit": 1 }),
    )
    .await;
    assert_eq!(sources(&only_a), ["doc-a"], "{only_a}");
    assert_eq!(
        only_a["hits"][0]["citation"]["source_metadata"]["url"],
        "https://docs.example/a"
    );
    let by_collection = ok(
        svc,
        "knowledge_search",
        json!({ "scope": scope, "query": query, "collections": ["policies"] }),
    )
    .await;
    assert_eq!(sources(&by_collection), ["doc-b"], "{by_collection}");
    let none = ok(
        svc,
        "knowledge_search",
        json!({ "scope": scope, "query": query, "source_ids": [] }),
    )
    .await;
    assert!(none["hits"].as_array().unwrap().is_empty(), "{none}");

    // A hit outside the allowlist reads exactly like a missing id.
    let b_hit = all["hits"]
        .as_array()
        .unwrap()
        .iter()
        .find(|h| h["source_id"] == "doc-b")
        .unwrap()["item"]["id"]
        .as_str()
        .unwrap()
        .to_owned();
    for tool in ["knowledge_fetch", "knowledge_cite"] {
        let hidden = err(
            svc,
            tool,
            json!({ "scope": scope, "id": b_hit, "source_ids": ["doc-a"] }),
        )
        .await;
        let missing = err(
            svc,
            tool,
            json!({ "scope": scope, "id": "chunk_missing", "source_ids": ["doc-a"] }),
        )
        .await;
        assert_eq!(hidden.code, ErrorCode::NotFound, "{tool}");
        assert_eq!(
            hidden.message.replace(&b_hit, "ID"),
            missing.message.replace("chunk_missing", "ID"),
            "{tool}"
        );
        ok(
            svc,
            tool,
            json!({ "scope": scope, "id": b_hit, "source_ids": ["doc-b"] }),
        )
        .await;
    }
    let index = ok(
        svc,
        "knowledge_index",
        json!({ "scope": scope, "level": 1, "source_ids": ["doc-a"] }),
    )
    .await;
    assert_eq!(index["coverage"]["sources"], 1, "{index}");
    ok(
        svc,
        "knowledge_traverse",
        json!({ "scope": scope, "start": "Lisbon", "source_ids": ["doc-a"] }),
    )
    .await;
    let e = err(
        svc,
        "knowledge_search",
        json!({ "scope": scope, "query": query,
                "source_ids": vec!["x"; 1001] }),
    )
    .await;
    assert_eq!(e.code, ErrorCode::Invalid);

    // A new revision of the same source replaces the old text.
    let changed = "The quarterly offsite moved to Porto this year.";
    let e = err(
        svc,
        "knowledge_ingest",
        ingest("doc-a", "handbook", changed, json!({})),
    )
    .await;
    assert_eq!(e.code, ErrorCode::Conflict, "{e:?}");
    let revised = ok(
        svc,
        "knowledge_ingest",
        ingest(
            "doc-a",
            "handbook",
            changed,
            json!({ "replace": true, "metadata": { "url": "https://docs.example/a?v=2" } }),
        ),
    )
    .await;
    assert_eq!(revised["replaced"], true, "{revised}");
    let after = ok(
        svc,
        "knowledge_search",
        json!({ "scope": scope, "query": "offsite Lisbon Porto", "source_ids": ["doc-a"] }),
    )
    .await;
    let texts: Vec<&str> = after["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["text"].as_str().unwrap())
        .collect();
    assert!(texts.iter().any(|t| t.contains("Porto")), "{after}");
    assert!(!texts.iter().any(|t| t.contains("Lisbon")), "{after}");
    assert_eq!(
        after["hits"][0]["citation"]["source_metadata"]["url"],
        "https://docs.example/a?v=2"
    );
    let same = ok(
        svc,
        "knowledge_ingest",
        ingest(
            "doc-a",
            "handbook",
            changed,
            json!({ "replace": true, "metadata": { "url": "https://docs.example/a?v=2" } }),
        ),
    )
    .await;
    assert_eq!(
        same["replaced"], false,
        "an unchanged revision is a no-op: {same}"
    );

    let e = err(
        svc,
        "knowledge_ingest",
        ingest(
            "doc-c",
            "handbook",
            "text",
            json!({ "metadata": { "k": "v".repeat(2000) } }),
        ),
    )
    .await;
    assert_eq!(e.code, ErrorCode::Invalid);
}

#[tokio::test]
async fn in_memory_host_scenario() {
    host_scenario(&in_memory()).await;
}

#[tokio::test]
async fn in_memory_scopes_share_and_isolate() {
    shared_memory_scenario(&in_memory()).await;
}

#[tokio::test]
async fn trash_and_restore_by_path() {
    let svc = in_memory();
    let w = ok(
        &svc,
        "memory_write",
        json!({ "path": "/people/sam.md", "text": "# Sam\n\n- Reviews Rust PRs.\n" }),
    )
    .await;
    ok(
        &svc,
        "memory_delete",
        json!({ "path": "/people/sam.md", "expected_revision_id": w["revisionId"] }),
    )
    .await;
    let list = ok(&svc, "memory_list", json!({ "include_trashed": true })).await;
    assert!(list["tree"].as_str().unwrap().contains("(trashed)"));
    ok(&svc, "memory_restore", json!({ "path": "/people/sam.md" })).await;
    let read = ok(&svc, "memory_read", json!({ "path": "/people/sam.md" })).await;
    assert!(read["text"].as_str().unwrap().contains("Reviews"));
}

#[tokio::test]
async fn description_update_is_a_revision_and_undoable() {
    let svc = in_memory();
    let w = ok(
        &svc,
        "memory_write",
        json!({ "path": "/topics/ci.md", "text": "- CI runs on every push.\n",
                "description": "Continuous integration" }),
    )
    .await;
    let e = err(
        &svc,
        "memory_update_metadata",
        json!({ "path": "/topics/ci.md", "expected_revision_id": w["revisionId"] }),
    )
    .await;
    assert_eq!(e.code, ErrorCode::Invalid);
    let u = ok(
        &svc,
        "memory_update_metadata",
        json!({ "path": "/topics/ci.md", "expected_revision_id": w["revisionId"],
                "description": "How CI is set up" }),
    )
    .await;
    assert_eq!(u["description"], "How CI is set up");
    assert_ne!(u["revisionId"], w["revisionId"]);
    // A stale revision conflicts and carries the current head.
    let e = err(
        &svc,
        "memory_update_metadata",
        json!({ "path": "/topics/ci.md", "expected_revision_id": w["revisionId"],
                "description": "stale" }),
    )
    .await;
    assert_eq!(e.code, ErrorCode::Conflict);
    assert_eq!(e.details.unwrap()["current"]["revisionId"], u["revisionId"]);
    let read = ok(&svc, "memory_read", json!({ "path": "/topics/ci.md" })).await;
    assert_eq!(read["description"], "How CI is set up");
    assert_eq!(read["text"], "- CI runs on every push.\n");
    let history = ok(&svc, "memory_history", json!({ "path": "/topics/ci.md" })).await;
    assert_eq!(history["revisions"][0]["change"], "metadata");
    ok(
        &svc,
        "memory_undo",
        json!({ "path": "/topics/ci.md", "expected_revision_id": u["revisionId"] }),
    )
    .await;
    let read = ok(&svc, "memory_read", json!({ "path": "/topics/ci.md" })).await;
    assert_eq!(read["description"], "Continuous integration");
}

#[tokio::test]
async fn project_context_includes_personal_memory() {
    let svc = in_memory();
    for (scope, path, text) in [
        (api(), "/topics/style.md", "- Project style: tabs.\n"),
        (
            personal(),
            "/topics/style.md",
            "- Personal style: spaces.\n",
        ),
        (personal(), "/topics/editor.md", "- Uses a modal editor.\n"),
        (web(), "/topics/web.md", "- The web project uses Vite.\n"),
    ] {
        ok(
            &svc,
            "memory_write",
            json!({ "scope": scope, "path": path, "text": text }),
        )
        .await;
    }
    let ctx = ok(&svc, "memory_context", json!({ "scope": api() })).await;
    let text = ctx["text"].as_str().unwrap();
    assert!(
        text.contains("tabs") && text.contains("modal editor"),
        "{text}"
    );
    assert!(!text.contains("spaces"), "project wins on a path conflict");
    assert!(!text.contains("Vite"), "other projects stay out");
    assert!(ctx["tokens"].as_u64().unwrap() <= 4096);
    let origin = |path: &str| {
        ctx["included"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["path"] == path)
            .map(|f| f["origin"].clone())
            .unwrap()
    };
    assert_eq!(origin("/topics/style.md"), "project");
    assert_eq!(origin("/topics/editor.md"), "personal");

    let only = ok(
        &svc,
        "memory_context",
        json!({ "scope": api(), "include_personal": false }),
    )
    .await;
    assert!(!only["text"].as_str().unwrap().contains("modal editor"));
    assert!(only["included"][0].get("origin").is_none());
    let mine = ok(&svc, "memory_context", json!({ "scope": personal() })).await;
    assert!(mine["text"].as_str().unwrap().contains("spaces"));
    assert!(mine["included"][0].get("origin").is_none());
}

#[tokio::test]
async fn idempotent_create_replays() {
    let svc = in_memory();
    let args = json!({ "path": "/topics/x.md", "text": "- one\n", "idempotency_key": "k1" });
    let first = ok(&svc, "memory_write", args.clone()).await;
    let again = ok(&svc, "memory_write", args).await;
    assert_eq!(again["replayed"], true);
    assert_eq!(again["revisionId"], first["revisionId"]);
}

#[tokio::test]
async fn errors_have_stable_codes() {
    let svc = in_memory();
    let cases = [
        (
            "memory_context",
            json!({ "budget_tokens": 9000 }),
            ErrorCode::BudgetExceeded,
        ),
        (
            "knowledge_search",
            json!({ "query": "x", "budget_tokens": 9000 }),
            ErrorCode::BudgetExceeded,
        ),
        (
            "memory_read",
            json!({ "path": "/nope.md" }),
            ErrorCode::NotFound,
        ),
        ("nope", json!({}), ErrorCode::NotFound),
        ("memory_read", json!({ "bogus": 1 }), ErrorCode::Invalid),
        (
            "memory_list",
            json!({ "scope": { "user": "../etc" } }),
            ErrorCode::Invalid,
        ),
        (
            "memory_write",
            json!({ "path": "/topics/a.md", "text": "x", "edit": { "command": "insert", "line": 0, "text": "y" } }),
            ErrorCode::Invalid,
        ),
        (
            "memory_write",
            json!({ "path": "/topics/gone.md", "text": "x", "expected_revision_id": "rev_x" }),
            ErrorCode::Conflict,
        ),
        (
            "knowledge_ingest",
            json!({ "text": "   " }),
            ErrorCode::Invalid,
        ),
        (
            "knowledge_search",
            json!({ "query": "x", "as_of": "last week" }),
            ErrorCode::Invalid,
        ),
    ];
    for (tool, args, code) in cases {
        let e = err(&svc, tool, args.clone()).await;
        assert_eq!(e.code, code, "{tool} {args}: {e:?}");
        assert_eq!(e.to_json()["error"]["code"], code.as_str());
    }
}

// ---------------------------------------------------------------- databases

/// Tests in one binary share a process id and may read the same clock tick.
fn unique(prefix: &str) -> String {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    format!("{prefix}_{}_{nanos}_{n}", std::process::id())
}

fn args(storage: StorageProfile, pg: &str, qdrant: Option<&str>, name: &str) -> ServiceArgs {
    ServiceArgs {
        storage,
        postgres_url: Some(pg.to_owned()),
        postgres_schema: Some(name.to_owned()),
        qdrant_url: qdrant.map(str::to_owned),
        qdrant_api_key: std::env::var("KEEP_TEST_QDRANT_API_KEY").ok(),
        qdrant_ca: std::env::var_os("KEEP_TEST_QDRANT_CA").map(Into::into),
        postgres_ca: std::env::var_os("KEEP_TEST_POSTGRES_CA").map(Into::into),
        collection: name.to_owned(),
        user: "alex".into(),
        project: None,
        agent: Some("test".into()),
        tokenizer: "approx-v1".into(),
        no_profile_writes: false,
        embedder: Some(EmbedderKind::Hash),
        data_dir: None,
        evidence_budget_tokens: None,
        evidence_budget_window_secs: 3600,
        evidence_budget_per: BudgetPer::Token,
        ingest_job_budget_ms: 5000,
        job_pause_ms: 0,
        onnx_sessions: None,
        onnx_threads: None,
        model: Default::default(),
    }
}

async fn drop_schema(pg: &str, schema: &str) {
    let client = keep_postgres::connect(&test_pg_url(pg)).await.unwrap();
    client
        .batch_execute(&format!("DROP SCHEMA IF EXISTS {schema} CASCADE"))
        .await
        .unwrap();
}

/// The test Postgres URL with `KEEP_TEST_POSTGRES_CA`, if set, as its root.
fn test_pg_url(pg: &str) -> String {
    match std::env::var("KEEP_TEST_POSTGRES_CA") {
        Ok(ca) if !pg.contains("sslrootcert") => {
            let sep = if pg.contains('?') { '&' } else { '?' };
            format!("{pg}{sep}sslrootcert={ca}")
        }
        _ => pg.to_owned(),
    }
}

async fn drop_collection(qdrant: &str, name: &str) {
    let ca = std::env::var_os("KEEP_TEST_QDRANT_CA").map(|p| std::fs::read(p).unwrap());
    let index = keep_qdrant::QdrantIndex::connect(
        qdrant,
        std::env::var("KEEP_TEST_QDRANT_API_KEY").ok(),
        ca.as_deref(),
    )
    .unwrap();
    let _ = index.delete_collection(name).await;
}

async fn run_db(storage: StorageProfile, pg: &str, qdrant: Option<&str>) {
    let name = unique("keep_server_test");
    let result = {
        let name = name.clone();
        let (pg, qdrant) = (pg.to_owned(), qdrant.map(str::to_owned));
        tokio::spawn(async move {
            let a = args(storage, &pg, qdrant.as_deref(), &name);
            let svc = KeepService::open(&a).await.unwrap();
            shared_memory_scenario(&svc).await;
            host_scenario(&svc).await;
            drop(svc);
            // A second process sees what the first one wrote.
            let first = KeepService::open(&a).await.unwrap();
            ok(
                &first,
                "memory_write",
                json!({ "path": "/topics/shared.md", "text": "- from agent one\n" }),
            )
            .await;
            let second = KeepService::open(&a).await.unwrap();
            let read = ok(
                &second,
                "memory_read",
                json!({ "path": "/topics/shared.md" }),
            )
            .await;
            assert!(read["text"].as_str().unwrap().contains("agent one"));
        })
        .await
    };
    drop_schema(pg, &name).await;
    if let Some(q) = qdrant {
        drop_collection(q, &name).await;
    }
    if let Err(e) = result {
        std::panic::resume_unwind(e.into_panic());
    }
}

#[tokio::test]
async fn postgres_vector_profile() {
    let Ok(pg) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    run_db(StorageProfile::PostgresVector, &pg, None).await;
}

#[tokio::test]
async fn postgres_qdrant_profile() {
    let (Ok(pg), Ok(q)) = (
        std::env::var("KEEP_TEST_POSTGRES_URL"),
        std::env::var("KEEP_TEST_QDRANT_URL"),
    ) else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL and KEEP_TEST_QDRANT_URL are not both set");
        return;
    };
    run_db(StorageProfile::PostgresQdrant, &pg, Some(&q)).await;
}

/// A TCP forwarder a test can cut, to simulate a dependency outage without
/// touching the shared database containers.
struct Cuttable {
    addr: std::net::SocketAddr,
    up: std::sync::Arc<std::sync::atomic::AtomicBool>,
    live: std::sync::Arc<std::sync::Mutex<Vec<tokio::task::AbortHandle>>>,
    accept: tokio::task::JoinHandle<()>,
}

impl Cuttable {
    async fn to(target: &str) -> Self {
        let target = target.to_owned();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let up = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let live = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let (u, l) = (up.clone(), live.clone());
        let accept = tokio::spawn(async move {
            loop {
                let Ok((mut inbound, _)) = listener.accept().await else {
                    return;
                };
                if !u.load(std::sync::atomic::Ordering::SeqCst) {
                    continue; // dropped: the client sees a reset
                }
                let target = target.clone();
                let task = tokio::spawn(async move {
                    if let Ok(mut out) = tokio::net::TcpStream::connect(&target).await {
                        let _ = tokio::io::copy_bidirectional(&mut inbound, &mut out).await;
                    }
                });
                l.lock().unwrap().push(task.abort_handle());
            }
        });
        Self {
            addr,
            up,
            live,
            accept,
        }
    }

    fn cut(&self) {
        self.up.store(false, std::sync::atomic::Ordering::SeqCst);
        for t in self.live.lock().unwrap().drain(..) {
            t.abort();
        }
    }

    fn restore(&self) {
        self.up.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for Cuttable {
    fn drop(&mut self) {
        self.accept.abort();
        self.cut();
    }
}

/// `host:port` of a `scheme://[user@]host:port/...` URL.
fn host_port(url: &str) -> String {
    let rest = url.split("://").nth(1).unwrap();
    let rest = rest.rsplit_once('@').map_or(rest, |(_, r)| r);
    rest.split(['/', '?']).next().unwrap().to_owned()
}

/// Outages are reported, never answered with success, and recovered from:
/// a cut ledger makes calls fail as `unavailable` and the service unready;
/// a cut vector store leaves ingest captured with enrichment `pending`,
/// search lexical, and the background worker finishes the work afterwards.
async fn outage_scenario(storage: StorageProfile, pg: &str, qdrant: Option<&str>) {
    let local = |u: &str| {
        let h = host_port(u);
        h.starts_with("127.0.0.1:") || h.starts_with("localhost:")
    };
    let tls = pg.contains("sslmode=verify") || qdrant.is_some_and(|q| q.starts_with("https:"));
    if !local(pg) || !qdrant.is_none_or(local) || tls {
        eprintln!("skipping: outage tests cut a local plaintext forwarder");
        return;
    }
    let name = unique("keep_server_outage");
    let result = {
        let (pg, qdrant, name) = (pg.to_owned(), qdrant.map(str::to_owned), name.clone());
        tokio::spawn(async move {
            let pg_proxy = Cuttable::to(&host_port(&pg)).await;
            let pg_via = pg.replacen(&host_port(&pg), &pg_proxy.addr.to_string(), 1);
            let q_proxy = match &qdrant {
                Some(q) => Some(Cuttable::to(&host_port(q)).await),
                None => None,
            };
            let q_via = qdrant
                .as_deref()
                .zip(q_proxy.as_ref())
                .map(|(q, p)| q.replacen(&host_port(q), &p.addr.to_string(), 1));
            let svc = KeepService::open(&args(storage, &pg_via, q_via.as_deref(), &name))
                .await
                .unwrap();
            assert!(svc.readiness().await.0);

            if let Some(q) = &q_proxy {
                q.cut();
                let (ready, info) = svc.readiness().await;
                assert!(ready, "vectors only degrade: {info}");
                assert_eq!(info["checks"]["vectors"], "degraded", "{info}");
                let ingested = ok(
                    &svc,
                    "knowledge_ingest",
                    json!({ "source_id": "outage", "text": "The pager rota rotates every Monday." }),
                )
                .await;
                assert_eq!(ingested["captured"], true);
                assert_eq!(ingested["enrichment"]["state"], "pending", "{ingested}");
                let found = ok(
                    &svc,
                    "knowledge_search",
                    json!({ "query": "pager rota Monday" }),
                )
                .await;
                assert!(
                    found["degraded"].to_string().contains("vector_unavailable"),
                    "{found}"
                );
                assert!(!found["hits"].as_array().unwrap().is_empty(), "{found}");
                q.restore();
                let mut state = Value::Null;
                for _ in 0..20 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    svc.run_due_jobs().await.unwrap();
                    let status = ok(&svc, "knowledge_status", json!({})).await;
                    state = status["enrichment"]["state"].clone();
                    if state == "complete" {
                        break;
                    }
                }
                assert_eq!(state, "complete");
            }

            pg_proxy.cut();
            let (ready, info) = svc.readiness().await;
            assert!(!ready, "{info}");
            assert_eq!(info["checks"]["ledger"], "unavailable");
            assert!(!info.to_string().contains(&pg_proxy.addr.port().to_string()));
            let e = err(
                &svc,
                "memory_write",
                json!({ "path": "/topics/outage.md", "text": "- during the outage\n" }),
            )
            .await;
            assert_eq!(e.code, ErrorCode::Unavailable, "{e:?}");
            let e = err(&svc, "knowledge_status", json!({})).await;
            assert_eq!(e.code, ErrorCode::Unavailable, "{e:?}");
            pg_proxy.restore();
            assert!(svc.readiness().await.0, "reconnects after the outage");
            ok(
                &svc,
                "memory_write",
                json!({ "path": "/topics/outage.md", "text": "- after the outage\n" }),
            )
            .await;
            let after = ok(
                &svc,
                "knowledge_ingest",
                json!({ "source_id": "after", "text": "Backups run nightly at two." }),
            )
            .await;
            assert_eq!(after["enrichment"]["state"], "complete", "{after}");
        })
        .await
    };
    drop_schema(pg, &name).await;
    if let Some(q) = qdrant {
        drop_collection(q, &name).await;
    }
    if let Err(e) = result {
        std::panic::resume_unwind(e.into_panic());
    }
}

#[tokio::test]
async fn postgres_vector_outages_are_reported_and_recovered() {
    let Ok(pg) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    outage_scenario(StorageProfile::PostgresVector, &pg, None).await;
}

#[tokio::test]
async fn postgres_qdrant_outages_are_reported_and_recovered() {
    let (Ok(pg), Ok(q)) = (
        std::env::var("KEEP_TEST_POSTGRES_URL"),
        std::env::var("KEEP_TEST_QDRANT_URL"),
    ) else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL and KEEP_TEST_QDRANT_URL are not both set");
        return;
    };
    outage_scenario(StorageProfile::PostgresQdrant, &pg, Some(&q)).await;
}

#[tokio::test]
async fn embedder_mismatch_is_refused() {
    use keep_server::service::in_schema;
    use keep_server::settings::{EmbedderRecord, Settings, check_embedder};
    let Ok(pg) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    let name = unique("keep_server_embedder");
    let result = {
        let (pg, name) = (pg.clone(), name.clone());
        tokio::spawn(async move {
            let a = args(StorageProfile::PostgresVector, &pg, None, &name);
            // The first start records the hash embedder; a restart agrees.
            KeepService::open(&a).await.unwrap();
            KeepService::open(&a).await.unwrap();

            let url = in_schema(&pg, &name);
            let settings = Settings::connect(&url, Some(&name)).await.unwrap();
            let knowledge = keep_knowledge::profile::postgres_vector(&url)
                .await
                .unwrap();
            let other = EmbedderRecord {
                provider: "local-onnx".into(),
                model: "example/other-model".into(),
                revision: "0".repeat(40),
                dimensions: 384,
            };
            let e = check_embedder(&settings, knowledge.vectors(), &name, &other)
                .await
                .unwrap_err()
                .to_string();
            assert!(e.contains("built with hash-embedder-v1"), "{e}");
            assert!(e.contains("knowledge_rebuild_projection"), "{e}");
            // Same size, different model: still refused.
            let same_size = EmbedderRecord {
                dimensions: 256,
                ..other.clone()
            };
            assert!(
                check_embedder(&settings, knowledge.vectors(), &name, &same_size)
                    .await
                    .is_err()
            );

            // A collection from before the record existed is checked by size.
            let client = keep_postgres::connect(&url).await.unwrap();
            client
                .execute("DELETE FROM keep_server_settings", &[])
                .await
                .unwrap();
            let e = check_embedder(&settings, knowledge.vectors(), &name, &other)
                .await
                .unwrap_err()
                .to_string();
            assert!(e.contains("256-dimensional"), "{e}");
            assert!(e.contains("rebuild"), "{e}");
            // A new collection takes the new embedder.
            let fresh = format!("{name}_v2");
            check_embedder(&settings, knowledge.vectors(), &fresh, &other)
                .await
                .unwrap();
            check_embedder(&settings, knowledge.vectors(), &fresh, &other)
                .await
                .unwrap();
        })
        .await
    };
    drop_schema(&pg, &name).await;
    if let Err(e) = result {
        std::panic::resume_unwind(e.into_panic());
    }
}

#[cfg(not(feature = "local-embeddings"))]
#[tokio::test]
async fn local_embedder_needs_the_build_feature() {
    let mut a = args(StorageProfile::InMemory, "unused", None, "knowledge");
    a.embedder = Some(EmbedderKind::Local);
    let Err(e) = KeepService::open(&a).await else {
        panic!("local embedder without the feature must fail");
    };
    assert!(e.to_string().contains("local-embeddings"), "{e}");
}

#[tokio::test]
async fn evidence_budget_is_per_caller_and_runs_out() {
    use keep_server::budget::BudgetConfig;
    let mut opts = ServiceOptions::new(Scope::new("alex", None).unwrap());
    opts.evidence_budget = Some(BudgetConfig {
        // Each search below returns one 27-token hit.
        tokens: 54,
        window: std::time::Duration::from_secs(3600),
        per: BudgetPer::Token,
    });
    let svc = KeepService::in_memory(opts);
    ok(
        &svc,
        "knowledge_ingest",
        json!({ "scope": api(), "source_id": "notes",
                "text": "The billing service is deployed with Argo CD. Invoices are retried three times." }),
    )
    .await;
    let search = json!({ "scope": api(), "query": "billing deploy", "budget_tokens": 1000 });
    let mut spent = 0;
    let mut exhausted = None;
    for _ in 0..20 {
        match svc
            .call_as("knowledge_search", search.clone(), "a", None)
            .await
        {
            Ok(v) => {
                let limit = v["allowance"]["limit"].as_u64().unwrap();
                assert!(limit <= 54, "the allowance is cut to the budget: {v}");
                spent += v["allowance"]["used"].as_u64().unwrap();
            }
            Err(e) => {
                exhausted = Some(e);
                break;
            }
        }
    }
    let e = exhausted.expect("the budget runs out");
    assert_eq!(e.code, ErrorCode::BudgetExceeded);
    assert_eq!(spent, 54);
    // Another caller has its own budget.
    svc.call_as("knowledge_search", search, "b", None)
        .await
        .unwrap();
}

#[test]
fn receipt_key_file_is_created_once_and_private() {
    use keep_server::settings::file_receipt_key;
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(unique("receipt"));
    let path = dir.join("nested").join("receipt.key");
    let first = file_receipt_key(&path).unwrap();
    let again = file_receipt_key(&path).unwrap();
    assert_eq!(first, again);
    assert_eq!(first.len(), 32);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        let dmode = std::fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(dmode & 0o077, 0);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn in_memory_receipt_key_lives_in_the_data_dir() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(unique("data"));
    let mut a = args(StorageProfile::InMemory, "unused", None, "knowledge");
    a.data_dir = Some(dir.clone());
    KeepService::open(&a).await.unwrap();
    let key = std::fs::read_to_string(dir.join("receipt.key")).unwrap();
    KeepService::open(&a).await.unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.join("receipt.key")).unwrap(),
        key
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test]
async fn idempotent_retries_replay_after_a_restart() {
    let Ok(pg) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    let name = unique("keep_server_receipt");
    let result = {
        let (pg, name) = (pg.clone(), name.clone());
        tokio::spawn(async move {
            let a = args(StorageProfile::PostgresVector, &pg, None, &name);
            let write = json!({ "path": "/topics/r.md", "text": "- once\n", "idempotency_key": "restart-1" });
            let svc = KeepService::open(&a).await.unwrap();
            let first = ok(&svc, "memory_write", write.clone()).await;
            drop(svc);
            let svc = KeepService::open(&a).await.unwrap();
            let again = ok(&svc, "memory_write", write).await;
            assert_eq!(again["replayed"], true);
            assert_eq!(again["revisionId"], first["revisionId"]);
        })
        .await
    };
    drop_schema(&pg, &name).await;
    if let Err(e) = result {
        std::panic::resume_unwind(e.into_panic());
    }
}

/// The hash embedder, `delay` slower per call, as a busy local model is.
struct SlowEmbedder {
    inner: keep_providers::HashEmbedder,
    delay: std::time::Duration,
}

#[async_trait::async_trait]
impl keep_providers::Embedder for SlowEmbedder {
    fn descriptor(&self) -> &keep_providers::ProviderDescriptor {
        keep_providers::Embedder::descriptor(&self.inner)
    }
    fn dimensions(&self) -> usize {
        keep_providers::Embedder::dimensions(&self.inner)
    }
    async fn embed(
        &self,
        texts: &[String],
    ) -> Result<keep_providers::EmbedResponse, keep_providers::ProviderError> {
        tokio::time::sleep(self.delay).await;
        keep_providers::Embedder::embed(&self.inner, texts).await
    }
}

fn slow_service(budget: std::time::Duration) -> KeepService {
    use std::sync::Arc;
    let mut opts = ServiceOptions::new(Scope::new("alex", None).unwrap());
    opts.ingest_job_budget = budget;
    let memory = keep_memory::MemoryService::new(
        Arc::new(keep_memory::InMemoryLedger::new()),
        Arc::new(keep_memory::InMemoryJournal::new()),
    );
    let knowledge = keep_knowledge::profile::in_memory().with_embedder(Arc::new(SlowEmbedder {
        inner: keep_providers::HashEmbedder::new(256),
        delay: std::time::Duration::from_millis(300),
    }));
    KeepService::new(memory, knowledge, "in-memory", opts)
}

/// A long document on a slow embedder: the ingest answers soon after its
/// job budget, says enrichment is pending, and the worker finishes it.
#[tokio::test]
async fn ingest_answers_within_its_job_budget() {
    let text = (0..400)
        .map(|i| format!("Line {i} of the runbook covers failover drills."))
        .collect::<Vec<_>>()
        .join("\n\n");
    for budget in [200, 0] {
        let svc = slow_service(std::time::Duration::from_millis(budget));
        let started = std::time::Instant::now();
        let ingested = ok(
            &svc,
            "knowledge_ingest",
            json!({ "source_id": "runbook", "text": text }),
        )
        .await;
        let took = started.elapsed();
        assert!(took < std::time::Duration::from_secs(2), "{took:?}");
        assert_eq!(ingested["captured"], true, "{ingested}");
        assert_eq!(ingested["enrichment"]["state"], "pending", "{ingested}");
        assert_eq!(ingested["jobs"]["failed"], 0, "{ingested}");
        if budget == 0 {
            assert_eq!(ingested["jobs"]["succeeded"], 0, "none run inline");
        }
        let mut state = Value::Null;
        for _ in 0..20 {
            svc.run_due_jobs().await.unwrap();
            let status = ok(&svc, "knowledge_status", json!({})).await;
            state = status["enrichment"]["state"].clone();
            if state == "complete" {
                break;
            }
        }
        assert_eq!(state, "complete");
    }
}

/// A client sends a document in batches, times out, and resumes: a stored
/// batch sent again changes nothing, and a retried first batch with
/// `replace` drops the later batches until they are appended again.
#[tokio::test]
async fn batched_ingest_resumes_and_retries() {
    let svc = in_memory();
    let batch = |from: usize, replace: bool| {
        let episodes: Vec<Value> = (from..from + 2)
            .map(|i| json!({ "id": format!("p{i}"), "text": format!("Section {i} of the escalation policy.") }))
            .collect();
        json!({ "source_id": "policy", "kind": "document", "title": "Escalation policy",
                "metadata": { "url": "https://example.com/policy" },
                "episodes": episodes, "replace": replace })
    };
    let first = ok(&svc, "knowledge_ingest", batch(0, true)).await;
    assert_eq!(first["report"]["episodes_added"], 2, "{first}");
    let second = ok(&svc, "knowledge_ingest", batch(2, false)).await;
    assert_eq!(second["report"]["episodes_added"], 2, "{second}");
    assert_eq!(second["replaced"], false, "{second}");
    let again = ok(&svc, "knowledge_ingest", batch(2, false)).await;
    assert_eq!(again["report"]["episodes_added"], 0, "{again}");
    assert_eq!(again["report"]["jobs_enqueued"], 0, "{again}");
    assert_eq!(again["jobs"]["succeeded"], 0, "{again}");

    let retried = ok(&svc, "knowledge_ingest", batch(0, true)).await;
    assert_eq!(retried["replaced"], true, "{retried}");
    assert_eq!(retried["report"]["episodes_unchanged"], 2, "{retried}");
    assert_eq!(retried["report"]["episodes_removed"], 2, "{retried}");
    let resumed = ok(&svc, "knowledge_ingest", batch(2, false)).await;
    assert_eq!(resumed["report"]["episodes_added"], 2, "{resumed}");
    assert_eq!(resumed["enrichment"]["state"], "complete", "{resumed}");
    let found = ok(
        &svc,
        "knowledge_search",
        json!({ "query": "Section 3 escalation policy" }),
    )
    .await;
    assert!(!found["hits"].as_array().unwrap().is_empty(), "{found}");
}
