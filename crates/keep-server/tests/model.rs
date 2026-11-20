//! Model-backed extraction and `knowledge_summarize`, with offline models:
//! an in-process mock, and a local OpenAI-compatible HTTP stub reached
//! through the server's own model configuration.

use std::sync::{Arc, Mutex};

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::routing::post;
use keep_providers::ProviderError;
use keep_providers::mock::FnChat;
use keep_server::model::{ModelArgs, ModelSetup};
use keep_server::{ErrorCode, KeepService, Scope, ServiceOptions};
use serde_json::{Value, json};

const FACTS: &str = r#"{"facts":[{"subject":"Billing team","attribute":"deploy_tool",
"value":"Argo CD","kind":"state","turns":[1],"confidence":0.9,
"statement":"The billing team deploys with Argo CD."}]}"#;
const SUMMARY: &str = "The runbook says the billing team ships with Argo CD on Tuesdays.";

fn with_model(model: Option<ModelSetup>) -> KeepService {
    let mut opts = ServiceOptions::new(Scope::new("alex", None).unwrap());
    opts.model = model;
    KeepService::in_memory(opts)
}

/// Answers a fact-extraction prompt with [`FACTS`] and anything else with
/// [`SUMMARY`].
fn reply_for(system: &str) -> String {
    if system.contains("summary") {
        SUMMARY.into()
    } else {
        FACTS.into()
    }
}

fn mock() -> ModelSetup {
    ModelSetup::new(
        Arc::new(FnChat::new("mock-model", |req| {
            Ok(reply_for(&req.messages[0].content))
        })),
        2,
    )
}

fn team() -> Value {
    json!({ "user": "alex", "project": "team" })
}

async fn ingest_runbook(svc: &KeepService) -> Value {
    svc.call(
        "knowledge_ingest",
        json!({ "scope": team(), "source_id": "runbook", "collection": "docs",
                "text": "Our team ships billing through Argo CD every Tuesday." }),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn a_configured_model_extracts_facts_and_summarizes() {
    let svc = with_model(Some(mock()));
    assert_eq!(svc.info()["model"]["model"], "mock/mock-model");
    let ingested = ingest_runbook(&svc).await;
    assert_eq!(ingested["enrichment"]["state"], "complete", "{ingested}");

    let found = svc
        .call(
            "knowledge_search",
            json!({ "scope": team(), "query": "which deploy tool does billing use" }),
        )
        .await
        .unwrap();
    let texts: Vec<&str> = found["hits"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|h| h["text"].as_str())
        .collect();
    assert!(
        texts
            .iter()
            .any(|t| t.contains("The billing team deploys with Argo CD.")),
        "{texts:?}"
    );

    let s = svc
        .call(
            "knowledge_summarize",
            json!({ "scope": team(), "source_id": "runbook" }),
        )
        .await
        .unwrap();
    assert_eq!(s["summary"], SUMMARY);
    assert_eq!(s["model"], "mock/mock-model");
    let s = svc
        .call(
            "knowledge_summarize",
            json!({ "text": "Some notes about deploys." }),
        )
        .await
        .unwrap();
    assert_eq!(s["summary"], SUMMARY);
    let s = svc
        .call(
            "knowledge_summarize",
            json!({ "episodes": [{ "speaker": "Ana", "text": "We ship on Tuesdays." }] }),
        )
        .await
        .unwrap();
    assert_eq!(s["summary"], SUMMARY);

    // A source the scope cannot read looks exactly like a missing one.
    let other = json!({ "user": "alex", "project": "other" });
    for id in ["runbook", "missing"] {
        let e = svc
            .call(
                "knowledge_summarize",
                json!({ "scope": other, "source_id": id }),
            )
            .await
            .unwrap_err();
        assert_eq!(e.code, ErrorCode::NotFound);
        assert_eq!(e.message, format!("source {id:?}"));
    }
    for bad in [
        json!({}),
        json!({ "text": "a", "source_id": "runbook" }),
        json!({ "episodes": [] }),
        json!({ "text": "   " }),
    ] {
        let e = svc.call("knowledge_summarize", bad).await.unwrap_err();
        assert_eq!(e.code, ErrorCode::Invalid);
    }
    let e = svc
        .call(
            "knowledge_summarize",
            json!({ "text": "word ".repeat(50_000) }),
        )
        .await
        .unwrap_err();
    assert_eq!(e.code, ErrorCode::LimitExceeded);
}

#[tokio::test]
async fn without_a_model_extraction_uses_rules_and_summaries_are_unavailable() {
    let svc = with_model(None);
    assert!(svc.info()["model"].is_null());
    let ingested = ingest_runbook(&svc).await;
    assert_eq!(ingested["enrichment"]["state"], "complete");
    let e = svc
        .call("knowledge_summarize", json!({ "text": "Some notes." }))
        .await
        .unwrap_err();
    assert_eq!(e.code, ErrorCode::Unavailable);
    assert_eq!(e.reason.as_deref(), Some("MODEL_NOT_CONFIGURED"));
}

#[tokio::test]
async fn model_failures_are_explicit_enrichment_states() {
    let broken = ModelSetup::new(
        Arc::new(FnChat::new("broken", |_| {
            Err(ProviderError::Malformed {
                message: "not json".into(),
            })
        })),
        1,
    );
    let svc = with_model(Some(broken));
    let ingested = ingest_runbook(&svc).await;
    assert_eq!(ingested["enrichment"]["state"], "failed", "{ingested}");
    // The text itself stays searchable.
    let found = svc
        .call(
            "knowledge_search",
            json!({ "scope": team(), "query": "Argo CD Tuesday" }),
        )
        .await
        .unwrap();
    assert!(!found["hits"].as_array().unwrap().is_empty());

    let flaky = ModelSetup::new(
        Arc::new(FnChat::new("flaky", |_| Err(ProviderError::Timeout))),
        1,
    );
    let svc = with_model(Some(flaky));
    let ingested = ingest_runbook(&svc).await;
    assert_eq!(ingested["enrichment"]["state"], "pending", "{ingested}");
    let e = svc
        .call("knowledge_summarize", json!({ "text": "Some notes." }))
        .await
        .unwrap_err();
    assert_eq!(e.code, ErrorCode::Unavailable);
    assert_eq!(e.reason.as_deref(), Some("PROVIDER"));
}

#[derive(Clone, Default)]
struct Seen {
    auth: Arc<Mutex<Vec<String>>>,
    models: Arc<Mutex<Vec<String>>>,
}

async fn chat_completions(
    State(seen): State<Seen>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Json<Value> {
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    seen.auth.lock().unwrap().push(auth.to_owned());
    seen.models
        .lock()
        .unwrap()
        .push(body["model"].as_str().unwrap_or_default().to_owned());
    let system = body["messages"][0]["content"].as_str().unwrap_or_default();
    Json(json!({
        "model": "stub-model",
        "choices": [{
            "message": { "role": "assistant", "content": reply_for(system) },
            "finish_reason": "stop",
        }],
        "usage": { "prompt_tokens": 40, "completion_tokens": 20 },
    }))
}

#[tokio::test]
async fn the_server_reaches_an_openai_compatible_endpoint_with_a_budget() {
    let seen = Seen::default();
    let app = axum::Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await });

    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("model.key");
    std::fs::write(&key, "test-key\n").unwrap();
    let args = ModelArgs {
        model_endpoint: Some(format!("http://{addr}/v1")),
        model_api_key_file: Some(key),
        model_name: Some("stub-model".into()),
        model_served_name: Some("stub-model".into()),
        model_timeout_secs: 5,
        model_concurrency: 2,
        model_reply_tokens: 256,
        // One extraction and one summary, then the budget is spent.
        model_budget_calls: 2,
        model_budget_input_tokens: 100_000,
        model_budget_output_tokens: 100_000,
        ..ModelArgs::default()
    };
    let setup = args.build().unwrap().unwrap();
    assert_eq!(setup.id, format!("stub-model@{}", addr.ip()));
    let svc = with_model(Some(setup));
    let ingested = ingest_runbook(&svc).await;
    assert_eq!(ingested["enrichment"]["state"], "complete", "{ingested}");
    let s = svc
        .call(
            "knowledge_summarize",
            json!({ "scope": team(), "source_id": "runbook" }),
        )
        .await
        .unwrap();
    assert_eq!(s["summary"], SUMMARY);
    assert_eq!(s["usage"]["inputTokens"], 40);
    assert_eq!(svc.info()["model"]["usage"]["calls"], 2);
    assert!(!svc.info().to_string().contains("test-key"));

    let e = svc
        .call("knowledge_summarize", json!({ "text": "More notes." }))
        .await
        .unwrap_err();
    assert_eq!(e.code, ErrorCode::BudgetExceeded, "{e:?}");
    assert_eq!(e.reason.as_deref(), Some("MODEL_BUDGET_EXHAUSTED"));

    assert_eq!(*seen.auth.lock().unwrap(), vec!["Bearer test-key"; 2]);
    assert_eq!(*seen.models.lock().unwrap(), vec!["stub-model"; 2]);
    server.abort();
}
