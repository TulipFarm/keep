//! Model-backed extraction through the job queue, with an offline mock
//! chat model.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use keep_knowledge::engine::{EpisodeInput, SourceInput, day};
use keep_knowledge::session_extract::{MAX_TURN_CHARS, SessionExtractor, episode_sessions};
use keep_knowledge::{Access, AsOf, EvidenceAllowance, SearchRequest, SourceKind, profile};
use keep_providers::ProviderError;
use keep_providers::mock::FnChat;

fn doc(id: &str) -> SourceInput {
    SourceInput {
        id: id.into(),
        partition: "team".into(),
        collection: "docs".into(),
        title: id.into(),
        kind: SourceKind::Document,
        metadata: Default::default(),
    }
}

fn body(source: &str, text: &str) -> Vec<EpisodeInput> {
    vec![EpisodeInput {
        id: format!("{source}#body"),
        speaker: String::new(),
        occurred_at: Some(day(2025, 3, 1)),
        text: text.into(),
    }]
}

const REPLY: &str = r#"{"facts":[{"subject":"Billing team","attribute":"deploy_tool",
"value":"Argo CD","kind":"state","turns":[1],"confidence":0.9,
"statement":"The billing team deploys with Argo CD."}]}"#;

#[tokio::test]
async fn model_facts_are_extracted_by_jobs_and_cite_the_episode() {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let model = FnChat::new("mock-extractor", move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
        Ok(REPLY.into())
    });
    let sx = SessionExtractor::new(Some(Arc::new(model)), "mock-extractor", None, 1024);
    let k = profile::in_memory().with_session_extractor(Arc::new(sx), 4);
    k.ingest(
        "c",
        doc("runbook"),
        body(
            "runbook",
            "Our team ships billing through Argo CD every Tuesday.",
        ),
    )
    .await
    .unwrap();
    let run = k.run_jobs("c", 10).await.unwrap();
    assert_eq!(run.failed, 0, "{run:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let res = k
        .search(
            &SearchRequest {
                corpus: "c".into(),
                access: Access::to(["team"]),
                query: "billing deploy tool Argo CD".into(),
                as_of: AsOf::default(),
                limit: 10,
            },
            &mut EvidenceAllowance::new(2000),
        )
        .await
        .unwrap();
    let fact = res
        .hits
        .iter()
        .find(|h| h.fact.is_some())
        .expect("a fact hit");
    assert!(
        fact.text.contains("The billing team deploys with Argo CD."),
        "{}",
        fact.text
    );
    assert_eq!(fact.citation.source_id, "runbook");
    assert!(fact.citation.quote.contains("every Tuesday"));
}

#[tokio::test]
async fn model_failures_are_job_states_not_silence() {
    let model = FnChat::new("broken", |req| {
        if req.messages[1].content.contains("timeout") {
            Err(ProviderError::Timeout)
        } else {
            Ok("this is not JSON".into())
        }
    });
    let sx = SessionExtractor::new(Some(Arc::new(model)), "broken", None, 1024);
    let k = profile::in_memory().with_session_extractor(Arc::new(sx), 1);
    k.ingest("c", doc("a"), body("a", "A reply that is malformed."))
        .await
        .unwrap();
    k.ingest("c", doc("b"), body("b", "A call that hits a timeout."))
        .await
        .unwrap();
    let run = k.run_jobs("c", 10).await.unwrap();
    assert_eq!((run.failed, run.retried), (1, 1), "{run:?}");
    let counts = k.job_counts("c", "team").await.unwrap();
    assert_eq!(counts.failed, 1, "{counts:?}");
    assert_eq!(counts.retry_scheduled, 1, "{counts:?}");
}

#[test]
fn long_episodes_are_read_whole_in_windows() {
    let word = "lorem ";
    let text = word.repeat(MAX_TURN_CHARS * 10 / word.len());
    let ep = keep_knowledge::model::Episode {
        corpus: "c".into(),
        id: "src#body".into(),
        source_id: "src".into(),
        partition: "team".into(),
        ordinal: 0,
        speaker: String::new(),
        occurred_at: None,
        recorded_at: day(2025, 1, 1),
        text: text.clone(),
        content_hash: String::new(),
    };
    let sessions = episode_sessions(&ep);
    assert_eq!(sessions.len(), 2);
    let turns: Vec<_> = sessions.iter().flat_map(|s| &s.turns).collect();
    assert!(
        turns
            .iter()
            .all(|t| t.text.chars().count() <= MAX_TURN_CHARS)
    );
    assert!(turns.iter().all(|t| t.id == "src#body"));
    let joined: Vec<&str> = turns.iter().map(|t| t.text.as_str()).collect();
    assert_eq!(joined.join(" "), text.trim());
}
