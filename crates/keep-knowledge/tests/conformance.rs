//! One behavioural suite, run against every storage profile.
//!
//! `in-memory` always runs. The Postgres profiles run when
//! KEEP_TEST_POSTGRES_URL (and, for Qdrant, KEEP_TEST_QDRANT_URL) are set;
//! each run gets its own Postgres schema and Qdrant collection, and removes
//! them afterwards.

use std::sync::Arc;

use async_trait::async_trait;
use keep_knowledge::engine::{EpisodeInput, ExternalClaim, SourceInput, day, namespace};
use keep_knowledge::extract::{Claim, FactExtractor, RuleExtractor};
use keep_knowledge::model::{Episode, FactStatus, JobState, TimeBasis};
use keep_knowledge::temporal::ValidMatch;
use keep_knowledge::{
    Access, AsOf, EvidenceAllowance, Fetched, IndexRequest, ItemRef, Knowledge, KnowledgeConfig,
    KnowledgeError, LexicalMode, ManualClock, SearchRequest, SourceKind, profile,
};
use keep_providers::{ProviderError, Reranker};
use keep_store::Bm25Params;

struct Env {
    k: Knowledge,
    clock: Arc<ManualClock>,
}

impl Env {
    fn new(k: Knowledge) -> Self {
        let clock = Arc::new(ManualClock::new(day(2024, 1, 1)));
        Self {
            k: k.with_clock(clock.clone()),
            clock,
        }
    }
}

fn source(id: &str, partition: &str, collection: &str, title: &str) -> SourceInput {
    SourceInput {
        id: id.into(),
        partition: partition.into(),
        collection: collection.into(),
        title: title.into(),
        kind: SourceKind::Conversation,
    }
}

fn turn(id: &str, at: chrono::DateTime<chrono::Utc>, text: &str) -> EpisodeInput {
    EpisodeInput {
        id: id.into(),
        speaker: "user".into(),
        occurred_at: Some(at),
        text: text.into(),
    }
}

fn everyone() -> Access {
    Access::to(["team", "hr"])
}

fn search_req(corpus: &str, access: Access, query: &str, as_of: AsOf) -> SearchRequest {
    SearchRequest {
        corpus: corpus.into(),
        access,
        query: query.into(),
        as_of,
        limit: 10,
    }
}

/// Facts found by a search, as (object, status, valid_to, valid match).
async fn facts_found(
    k: &Knowledge,
    corpus: &str,
    query: &str,
    as_of: AsOf,
) -> Vec<(
    String,
    FactStatus,
    Option<chrono::DateTime<chrono::Utc>>,
    Option<ValidMatch>,
)> {
    let res = k
        .search(
            &search_req(corpus, everyone(), query, as_of),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    let mut out = Vec::new();
    for h in res.hits {
        if let ItemRef::Fact(id) = &h.item {
            let f = k
                .fetch(
                    corpus,
                    &everyone(),
                    id,
                    as_of,
                    &mut EvidenceAllowance::default(),
                )
                .await
                .unwrap();
            let Fetched::Fact { fact, .. } = f.item else {
                panic!("fact id fetched a chunk")
            };
            let s = h.fact.unwrap();
            out.push((fact.object, s.status, s.valid_to, s.valid_match));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

async fn suite(label: &str, env: &Env) {
    eprintln!("[{label}] idempotent ingest");
    idempotent_ingest(env).await;
    eprintln!("[{label}] contradiction and as-of");
    contradiction_and_as_of(env).await;
    eprintln!("[{label}] external claims");
    external_claims(env).await;
    eprintln!("[{label}] hybrid search");
    hybrid_search(env).await;
    eprintln!("[{label}] permissions");
    permissions(env).await;
    eprintln!("[{label}] forget");
    forget(env).await;
    eprintln!("[{label}] evidence budget");
    evidence_budget(env).await;
    eprintln!("[{label}] traverse");
    traverse(env).await;
    eprintln!("[{label}] table of contents");
    table_of_contents(env).await;
    eprintln!("[{label}] projection rebuild");
    projection_rebuild(env).await;
}

async fn idempotent_ingest(env: &Env) {
    let k = &env.k;
    let c = "idem";
    let eps = vec![
        turn("t1", day(2024, 1, 2), "Mira Sato works at Northwind Labs."),
        turn("t2", day(2024, 1, 2), "Tomas Reyes leads Project Heron."),
    ];
    let first = k
        .ingest(c, source("s1", "team", "chats", "Kickoff"), eps.clone())
        .await
        .unwrap();
    assert!(first.source_created);
    assert_eq!(first.episodes_added, 2);
    assert!(first.jobs_enqueued > 0);

    let again = k
        .ingest(c, source("s1", "team", "chats", "Kickoff"), eps)
        .await
        .unwrap();
    assert!(!again.source_created);
    assert_eq!(again.episodes_added, 0);
    assert_eq!(again.episodes_unchanged, 2);
    assert_eq!(again.jobs_enqueued, 0);

    let changed = k
        .ingest(
            c,
            source("s1", "team", "chats", "Kickoff"),
            vec![turn("t1", day(2024, 1, 2), "Something else entirely.")],
        )
        .await;
    assert!(matches!(changed, Err(KnowledgeError::Conflict(_))));

    // A source cannot change partition by re-ingesting.
    let moved = k
        .ingest(
            c,
            source("s1", "hr", "chats", "Kickoff"),
            vec![turn("t3", day(2024, 1, 3), "Hello.")],
        )
        .await;
    assert!(moved.is_err());

    k.run_jobs(c, 10).await.unwrap();
    let run = k.run_jobs(c, 10).await.unwrap();
    assert_eq!(run.succeeded + run.retried + run.failed, 0);
    let found = facts_found(k, c, "Mira Sato Northwind", AsOf::now()).await;
    let objects: Vec<&str> = found.iter().map(|f| f.0.as_str()).collect();
    assert_eq!(
        objects,
        ["Northwind Labs", "Project Heron"],
        "one fact each"
    );
    k.delete_corpus(c).await.unwrap();
}

async fn contradiction_and_as_of(env: &Env) {
    let k = &env.k;
    let c = "temporal";
    env.clock.set(day(2024, 1, 10));
    k.ingest(
        c,
        source("jan", "team", "chats", "January catch-up"),
        vec![turn("j1", day(2024, 1, 10), "Mira Sato lives in Lisbon.")],
    )
    .await
    .unwrap();
    k.run_jobs(c, 10).await.unwrap();

    env.clock.set(day(2024, 6, 1));
    k.ingest(
        c,
        source("jun", "team", "chats", "June catch-up"),
        vec![turn("u1", day(2024, 6, 1), "Mira Sato moved to Porto.")],
    )
    .await
    .unwrap();
    k.run_jobs(c, 10).await.unwrap();

    // Today: Porto is current; Lisbon is kept but closed at the move.
    let now = facts_found(k, c, "Mira Sato lives", AsOf::now()).await;
    assert_eq!(now.len(), 2, "{now:?}");
    assert_eq!(now[0].0, "Lisbon");
    assert_eq!(now[0].1, FactStatus::Superseded);
    assert_eq!(now[0].2, Some(day(2024, 6, 1)));
    assert_eq!(now[1].0, "Porto");
    assert_eq!(now[1].1, FactStatus::Active);
    assert_eq!(now[1].2, None);

    // World time in March: only Lisbon was true.
    let march = facts_found(k, c, "Mira Sato lives", AsOf::valid_at(day(2024, 3, 1))).await;
    assert_eq!(march.len(), 1, "{march:?}");
    assert_eq!(march[0].0, "Lisbon");
    assert_eq!(march[0].3, Some(ValidMatch::Inside));

    // What we knew in March: Lisbon, still open, and nothing about Porto.
    let known = facts_found(k, c, "Mira Sato lives", AsOf::known_at(day(2024, 3, 1))).await;
    assert_eq!(known.len(), 1, "{known:?}");
    assert_eq!(known[0].0, "Lisbon");
    assert_eq!(known[0].1, FactStatus::Active);
    assert_eq!(known[0].2, None);

    // Before anything was recorded there is nothing at all.
    let res = k
        .search(
            &search_req(c, everyone(), "Lisbon", AsOf::known_at(day(2023, 1, 1))),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    assert!(res.hits.is_empty());

    // An older statement arriving late does not reopen the past.
    env.clock.set(day(2024, 7, 1));
    k.ingest(
        c,
        source("old", "team", "chats", "Old notes"),
        vec![turn("o1", day(2023, 5, 1), "Mira Sato lives in Faro.")],
    )
    .await
    .unwrap();
    k.run_jobs(c, 10).await.unwrap();
    let now = facts_found(k, c, "Mira Sato lives", AsOf::now()).await;
    let faro = now.iter().find(|f| f.0 == "Faro").expect("faro fact");
    assert_eq!(faro.1, FactStatus::Superseded);
    assert_eq!(faro.2, Some(day(2024, 1, 10)));
    let porto = now.iter().find(|f| f.0 == "Porto").unwrap();
    assert_eq!(porto.1, FactStatus::Active);
    k.delete_corpus(c).await.unwrap();
}

fn external(
    predicate: &str,
    object: &str,
    from: chrono::DateTime<chrono::Utc>,
    single_valued: bool,
    episodes: &[&str],
) -> ExternalClaim {
    ExternalClaim {
        claim: Claim {
            subject: "user".into(),
            subject_kind: "person".into(),
            predicate: predicate.into(),
            object: object.into(),
            object_kind: None,
            valid_from: Some(from),
            basis: TimeBasis::Observed,
            quote: String::new(),
        },
        text: Some(format!("The user's {predicate} is {object}.")),
        single_valued,
        episodes: episodes.iter().map(|e| e.to_string()).collect(),
    }
}

async fn external_claims(env: &Env) {
    let k = &env.k;
    let c = "external";
    let label = "session:test";
    env.clock.set(day(2024, 1, 10));
    k.ingest(
        c,
        source("s1", "team", "chats", "January"),
        vec![
            turn("s1:1", day(2024, 1, 10), "I work as an engineer."),
            turn("s1:2", day(2024, 1, 10), "I bought a zoom lens."),
        ],
    )
    .await
    .unwrap();
    k.run_jobs(c, 10).await.unwrap();
    let n = k
        .apply_claims(
            c,
            label,
            &[
                external("job_title", "engineer", day(2024, 1, 10), true, &["s1:1"]),
                external(
                    "bought_item",
                    "zoom lens",
                    day(2024, 1, 10),
                    false,
                    &["s1:2"],
                ),
                external("pet", "cat", day(2024, 1, 10), true, &["missing"]),
            ],
        )
        .await
        .unwrap();
    assert_eq!(n, 2, "a claim citing no known turn is skipped");

    env.clock.set(day(2024, 6, 1));
    k.ingest(
        c,
        source("s2", "team", "chats", "June"),
        vec![
            turn("s2:1", day(2024, 6, 1), "I'm a senior engineer now."),
            turn(
                "s2:2",
                day(2024, 6, 1),
                "Got a tripod, and I still love that lens.",
            ),
        ],
    )
    .await
    .unwrap();
    k.run_jobs(c, 10).await.unwrap();
    k.apply_claims(
        c,
        label,
        &[
            external(
                "job_title",
                "senior engineer",
                day(2024, 6, 1),
                true,
                &["s2:1"],
            ),
            external("bought_item", "tripod", day(2024, 6, 1), false, &["s2:2"]),
            external(
                "bought_item",
                "zoom lens",
                day(2024, 6, 1),
                false,
                &["s2:2", "s1:2"],
            ),
        ],
    )
    .await
    .unwrap();

    let user = k
        .ledger()
        .find_entity(c, "team", "user")
        .await
        .unwrap()
        .expect("user entity");
    let mut facts = k
        .ledger()
        .facts_about(c, &["team".to_string()], &user.id, 50)
        .await
        .unwrap();
    facts.sort_by(|a, b| a.object.cmp(&b.object));
    let summary: Vec<(
        String,
        FactStatus,
        Option<chrono::DateTime<chrono::Utc>>,
        usize,
    )> = facts
        .iter()
        .map(|f| {
            let v = f.current().unwrap();
            (f.object.clone(), v.status, v.valid_to, f.evidence.len())
        })
        .collect();
    assert_eq!(
        summary,
        vec![
            // The old job title is closed where the new one starts.
            (
                "engineer".into(),
                FactStatus::Superseded,
                Some(day(2024, 6, 1)),
                1
            ),
            ("senior engineer".into(), FactStatus::Active, None, 1),
            // Events accumulate; a restatement adds evidence, not a fact.
            ("tripod".into(), FactStatus::Active, None, 1),
            ("zoom lens".into(), FactStatus::Active, None, 2),
        ]
    );
    let lens = facts.iter().find(|f| f.object == "zoom lens").unwrap();
    assert_eq!(lens.text, "The user's bought_item is zoom lens.");
    assert!(lens.evidence.iter().all(|e| e.extractor == label));
    let whole = lens
        .evidence
        .iter()
        .find(|e| e.episode_id == "s1:2")
        .unwrap();
    assert_eq!(
        (whole.char_start, whole.quote.as_str()),
        (0, "I bought a zoom lens.")
    );
    assert_eq!(
        whole.char_end as usize,
        "I bought a zoom lens.".chars().count()
    );
    k.delete_corpus(c).await.unwrap();
}

async fn hybrid_search(env: &Env) {
    let k = &env.k;
    let c = "hybrid";
    env.clock.set(day(2024, 2, 1));
    k.ingest(
        c,
        source("s", "team", "chats", "Planning"),
        vec![
            turn("a", day(2024, 2, 1), "Tomas Reyes leads Project Heron."),
            turn(
                "b",
                day(2024, 2, 1),
                "The quarterly offsite is booked for the coast.",
            ),
            turn(
                "c",
                day(2024, 2, 1),
                "Lunch options were discussed at length.",
            ),
        ],
    )
    .await
    .unwrap();
    k.run_jobs(c, 10).await.unwrap();
    let res = k
        .search(
            &search_req(c, everyone(), "who leads Project Heron", AsOf::now()),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    assert!(res.method.contains("rrf"), "{}", res.method);
    assert!(res.degraded.is_empty(), "{:?}", res.degraded);
    assert!(!res.hits.is_empty());
    assert!(res.hits.iter().any(|h| matches!(h.item, ItemRef::Fact(_))));
    assert!(res.hits.iter().any(|h| matches!(h.item, ItemRef::Chunk(_))));
    assert!(
        res.hits.iter().take(2).any(|h| h.text.contains("Heron")),
        "{:?}",
        res.hits
    );
    for (i, h) in res.hits.iter().enumerate() {
        assert_eq!(h.rank, i + 1);
        assert!(h.citation.reference.starts_with("keep:"));
        assert!(h.citation.quote.len() <= h.text.len() + 200);
    }
    // Citations are stable and point at exact spans.
    let id = res.hits[0].item.point_id();
    let a = k.cite(c, &everyone(), &id).await.unwrap();
    let b = k.cite(c, &everyone(), &id).await.unwrap();
    assert_eq!(a, b);
    assert!(!a.is_empty());
    let ep = k
        .ledger()
        .episode(c, &a[0].episode_id)
        .await
        .unwrap()
        .unwrap();
    let span: String = ep
        .text
        .chars()
        .skip(a[0].char_start as usize)
        .take((a[0].char_end - a[0].char_start) as usize)
        .collect();
    assert_eq!(span, a[0].quote);
    k.delete_corpus(c).await.unwrap();
}

async fn permissions(env: &Env) {
    let k = &env.k;
    let c = "perm";
    env.clock.set(day(2024, 3, 1));
    k.ingest(
        c,
        source("pub", "team", "chats", "Team standup"),
        vec![turn(
            "p1",
            day(2024, 3, 1),
            "Mira Sato works on Project Heron.",
        )],
    )
    .await
    .unwrap();
    k.ingest(
        c,
        source("sec", "hr", "reviews", "Confidential Salary Review"),
        vec![turn(
            "h1",
            day(2024, 3, 1),
            "Mira Sato prefers remote work. Ingrid Falk works at Bluefin Partners.",
        )],
    )
    .await
    .unwrap();
    k.run_jobs(c, 10).await.unwrap();
    let team = Access::to(["team"]);
    let secret_words = [
        "Confidential",
        "Salary",
        "reviews",
        "Ingrid",
        "Bluefin",
        "remote",
    ];

    let res = k
        .search(
            &search_req(
                c,
                team.clone(),
                "Mira Sato remote work Bluefin",
                AsOf::now(),
            ),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    for h in &res.hits {
        for w in secret_words {
            assert!(!h.text.contains(w), "leaked {w} in {}", h.text);
            assert!(!h.citation.source_title.contains(w));
        }
    }
    let all = k
        .search(
            &search_req(c, everyone(), "Bluefin Partners", AsOf::now()),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    let secret = all
        .hits
        .iter()
        .find(|h| h.text.contains("Bluefin"))
        .expect("authorised caller sees it")
        .item
        .point_id();

    // No access at all sees nothing.
    let none = k
        .search(
            &search_req(c, Access::default(), "Mira", AsOf::now()),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    assert!(none.hits.is_empty());

    // Fetch and cite look exactly like a missing id.
    let f = k
        .fetch(
            c,
            &team,
            &secret,
            AsOf::now(),
            &mut EvidenceAllowance::default(),
        )
        .await;
    assert!(matches!(f, Err(KnowledgeError::NotFound)), "{f:?}");
    let missing = k
        .fetch(
            c,
            &team,
            "fact_000000000000000000000000",
            AsOf::now(),
            &mut EvidenceAllowance::default(),
        )
        .await;
    assert!(matches!(missing, Err(KnowledgeError::NotFound)));
    assert!(matches!(
        k.cite(c, &team, &secret).await,
        Err(KnowledgeError::NotFound)
    ));

    // Traverse stays inside readable partitions.
    let n = k
        .traverse(c, &team, "Mira Sato", 2, 10, AsOf::now())
        .await
        .unwrap();
    assert!(!n.nodes.is_empty());
    for e in &n.edges {
        assert_ne!(e.predicate, "prefers");
        assert!(!e.object.contains("Bluefin"));
    }
    for node in &n.nodes {
        assert_eq!(node.partition, "team");
    }

    // The index never shows restricted titles, collections, names or counts.
    for level in 0..=2u8 {
        let view = k
            .index(
                c,
                &team,
                &IndexRequest {
                    level,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let dump = format!("{} {:?}", view.text, view.entries);
        for w in secret_words {
            assert!(!dump.contains(w), "level {level} leaked {w}: {dump}");
        }
        assert_eq!(view.coverage.sources, 1, "level {level}");
        assert_eq!(view.coverage.episodes, 1);
        let full = k
            .index(
                c,
                &everyone(),
                &IndexRequest {
                    level,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(full.coverage.sources, 2);
        assert_ne!(full.generation, view.generation);
    }
    let full = k
        .index(
            c,
            &everyone(),
            &IndexRequest {
                level: 2,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(full.text.contains("Ingrid Falk"), "{}", full.text);

    // A caller without access learns nothing from the index either.
    let empty = k
        .index(c, &Access::default(), &IndexRequest::default())
        .await
        .unwrap();
    assert_eq!(empty.coverage.sources, 0);
    assert!(empty.entries.is_empty());
    k.delete_corpus(c).await.unwrap();
}

async fn forget(env: &Env) {
    let k = &env.k;
    let c = "forget";
    env.clock.set(day(2024, 4, 1));
    k.ingest(
        c,
        source("keep", "team", "chats", "Weekly sync"),
        vec![turn(
            "k1",
            day(2024, 4, 1),
            "Tomas Reyes works at Northwind Labs.",
        )],
    )
    .await
    .unwrap();
    k.ingest(
        c,
        source("drop", "team", "notes", "Garden Party Plans"),
        vec![
            turn(
                "d1",
                day(2024, 4, 2),
                "Tomas Reyes works at Northwind Labs.",
            ),
            turn("d2", day(2024, 4, 2), "Lena Park leads Project Juniper."),
        ],
    )
    .await
    .unwrap();
    k.run_jobs(c, 10).await.unwrap();

    let before = k
        .index(
            c,
            &everyone(),
            &IndexRequest {
                level: 2,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(before.text.contains("Juniper"), "{}", before.text);
    let toc_before = k
        .index(c, &everyone(), &IndexRequest::default())
        .await
        .unwrap();
    assert!(toc_before.text.contains("Garden Party Plans") || toc_before.text.contains("notes"));

    let juniper = k
        .search(
            &search_req(c, everyone(), "Lena Park Project Juniper", AsOf::now()),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    let juniper_ids: Vec<String> = juniper
        .hits
        .iter()
        .filter(|h| h.text.contains("Juniper"))
        .map(|h| h.item.point_id())
        .collect();
    assert!(!juniper_ids.is_empty());
    let point_ids_before = k.ledger().point_ids(c).await.unwrap();

    let report = k
        .forget_source(c, "drop")
        .await
        .unwrap()
        .expect("source existed");
    assert_eq!(report.episodes, 2);
    assert!(report.chunks >= 2);
    assert_eq!(report.facts_deleted, 1, "{report:?}");
    assert_eq!(report.facts_kept, 1, "{report:?}");
    assert_eq!(report.vectors_deferred, 0);
    assert!(report.vectors_deleted >= 3);
    assert!(k.forget_source(c, "drop").await.unwrap().is_none());

    // Ledger, vectors and every read path have let go.
    let point_ids_after = k.ledger().point_ids(c).await.unwrap();
    assert!(point_ids_after.len() < point_ids_before.len());
    let vectors = k.vectors().unwrap();
    let ns = namespace(c, "team");
    let probe = keep_providers::HashEmbedder::new(256);
    let q =
        keep_providers::Embedder::embed(&probe, &["Lena Park leads Project Juniper.".to_string()])
            .await
            .unwrap()
            .vectors
            .remove(0);
    let hits = vectors
        .search(&k.config().collection, &ns, &q, 100)
        .await
        .unwrap();
    for id in &juniper_ids {
        assert!(!hits.iter().any(|h| &h.id == id), "vector {id} survived");
        assert!(!point_ids_after.contains(id));
        let f = k
            .fetch(
                c,
                &everyone(),
                id,
                AsOf::now(),
                &mut EvidenceAllowance::default(),
            )
            .await;
        assert!(matches!(f, Err(KnowledgeError::NotFound)));
    }
    assert_eq!(hits.len(), point_ids_after.len());
    let res = k
        .search(
            &search_req(c, everyone(), "Juniper", AsOf::now()),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    assert!(res.hits.iter().all(|h| !h.text.contains("Juniper")));

    // The fact both sources stated survives on the remaining evidence.
    let kept = facts_found(k, c, "Tomas Reyes Northwind", AsOf::now()).await;
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].1, FactStatus::Active);

    // The index is rebuilt: deleted titles and names are gone.
    for level in 0..=2u8 {
        let after = k
            .index(
                c,
                &everyone(),
                &IndexRequest {
                    level,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let dump = format!("{} {:?}", after.text, after.entries);
        for w in ["Garden Party", "Juniper", "Lena Park"] {
            assert!(!dump.contains(w), "level {level} still shows {w}");
        }
        assert_eq!(after.coverage.sources, 1);
        assert_ne!(after.generation, toc_before.generation);
    }

    // The journal says something was erased, but not what.
    let journal = k.ledger().erasure_journal().await.unwrap();
    let entry = journal.last().expect("journal entry");
    assert_eq!(entry.target_kind, "source");
    assert_eq!(entry.episodes, 2);
    let dump = format!("{journal:?}");
    for w in ["drop", "Garden", "Juniper", "Lena", "forget"] {
        assert!(!dump.contains(w), "journal holds {w}");
    }
    k.delete_corpus(c).await.unwrap();
}

async fn evidence_budget(env: &Env) {
    let k = &env.k;
    let c = "budget";
    env.clock.set(day(2024, 5, 1));
    let eps: Vec<EpisodeInput> = (0..30)
        .map(|i| {
            turn(
                &format!("e{i}"),
                day(2024, 5, 1),
                &format!(
                    "Status note {i}: the Heron migration moved forward and the rollout checklist grew longer again."
                ),
            )
        })
        .collect();
    k.ingest(c, source("s", "team", "chats", "Status"), eps)
        .await
        .unwrap();
    k.run_jobs(c, 10).await.unwrap();
    let mut allowance = EvidenceAllowance::new(60);
    let mut req = search_req(c, everyone(), "Heron migration rollout", AsOf::now());
    req.limit = 30;
    let res = k.search(&req, &mut allowance).await.unwrap();
    assert!(res.used_tokens <= 60);
    assert!(res.omitted > 0);
    assert!(!res.hits.is_empty());
    assert_eq!(allowance.used(), res.used_tokens);
    // Default allowance is 2,000 tokens per answer.
    assert_eq!(EvidenceAllowance::default().limit(), 2000);
    // Spent allowance carries over to a follow-up fetch.
    let id = res.hits[0].item.point_id();
    let mut spent = EvidenceAllowance::new(1);
    let f = k.fetch(c, &everyone(), &id, AsOf::now(), &mut spent).await;
    assert!(
        matches!(f, Err(KnowledgeError::BudgetExhausted) | Ok(_)),
        "{f:?}"
    );
    if let Ok(f) = f {
        assert!(f.truncated);
    }
    assert!(spent.used() <= 1);
    k.delete_corpus(c).await.unwrap();
}

async fn traverse(env: &Env) {
    let k = &env.k;
    let c = "graph";
    env.clock.set(day(2024, 5, 1));
    k.ingest(
        c,
        source("s", "team", "chats", "Org chart"),
        vec![
            turn("a", day(2024, 5, 1), "Mira Sato works on Project Heron."),
            turn("b", day(2024, 5, 1), "Tomas Reyes works on Project Heron."),
            turn("c", day(2024, 5, 1), "Lena Park works on Project Heron."),
            turn("d", day(2024, 5, 1), "Tomas Reyes works at Northwind Labs."),
        ],
    )
    .await
    .unwrap();
    k.run_jobs(c, 10).await.unwrap();
    let one = k
        .traverse(c, &everyone(), "Mira Sato", 1, 10, AsOf::now())
        .await
        .unwrap();
    let names: Vec<&str> = one.nodes.iter().map(|n| n.name.as_str()).collect();
    assert!(names.contains(&"Project Heron"), "{names:?}");
    assert!(!names.contains(&"Northwind Labs"));
    let two = k
        .traverse(c, &everyone(), "Mira Sato", 2, 10, AsOf::now())
        .await
        .unwrap();
    let names: Vec<&str> = two.nodes.iter().map(|n| n.name.as_str()).collect();
    assert!(names.contains(&"Tomas Reyes"), "{names:?}");
    let narrow = k
        .traverse(c, &everyone(), "Project Heron", 1, 1, AsOf::now())
        .await
        .unwrap();
    assert_eq!(narrow.edges.len(), 1);
    assert!(narrow.truncated);
    k.delete_corpus(c).await.unwrap();
}

async fn table_of_contents(env: &Env) {
    let k = &env.k;
    let c = "toc";
    env.clock.set(day(2024, 6, 1));
    for i in 0..40 {
        k.ingest(
            c,
            source(
                &format!("s{i}"),
                "team",
                &format!("collection-{i:02}"),
                &format!("Meeting notes number {i} about the long running Heron programme"),
            ),
            vec![turn(
                &format!("t{i}"),
                day(2024, 1 + (i % 12) as u32, 1),
                &format!("Person{i} Example works on Project Heron."),
            )],
        )
        .await
        .unwrap();
    }
    k.run_jobs(c, 20).await.unwrap();
    let view = k
        .index(c, &everyone(), &IndexRequest::default())
        .await
        .unwrap();
    assert_eq!(view.level, 0);
    assert!(view.tokens <= 640, "{}", view.tokens);
    assert!(view.truncated);
    assert!(view.text.contains("more"), "{}", view.text);
    assert_eq!(view.coverage.sources, 40);
    // Same input, same output.
    let again = k
        .index(c, &everyone(), &IndexRequest::default())
        .await
        .unwrap();
    assert_eq!(view.text, again.text);
    assert_eq!(view.generation, again.generation);
    // Paging continues where the slot ran out.
    let next = k
        .index(
            c,
            &everyone(),
            &IndexRequest {
                level: 1,
                cursor: view.next_cursor,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(next.tokens <= 1000);
    // A smaller explicit budget is honoured.
    let tiny = k
        .index(
            c,
            &everyone(),
            &IndexRequest {
                budget_tokens: Some(120),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(tiny.tokens <= 120, "{}", tiny.tokens);
    // Level 2 finds an entity by prefix with pointers to its facts.
    let entities = k
        .index(
            c,
            &everyone(),
            &IndexRequest {
                level: 2,
                prefix: Some("person1".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert!(!entities.entries.is_empty());
    assert!(
        entities
            .entries
            .iter()
            .all(|e| e.label.to_lowercase().starts_with("person1"))
    );
    assert!(entities.entries.iter().all(|e| !e.pointers.is_empty()));
    k.delete_corpus(c).await.unwrap();
}

async fn projection_rebuild(env: &Env) {
    let k = &env.k;
    let c = "rebuild";
    env.clock.set(day(2024, 7, 1));
    k.ingest(
        c,
        source("s", "team", "chats", "Notes"),
        vec![turn("a", day(2024, 7, 1), "Mira Sato leads Project Heron.")],
    )
    .await
    .unwrap();
    k.run_jobs(c, 10).await.unwrap();
    let ns = namespace(c, "team");
    let vectors = k.vectors().unwrap();
    vectors
        .delete_namespace(&k.config().collection, &ns)
        .await
        .unwrap();
    let rebuilt = k.rebuild_projection(c).await.unwrap();
    assert_eq!(rebuilt, k.ledger().point_ids(c).await.unwrap().len());
    let res = k
        .search(
            &search_req(c, everyone(), "Heron", AsOf::now()),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    assert!(!res.hits.is_empty());
    k.delete_corpus(c).await.unwrap();
}

/// An extractor whose provider is always timing out.
struct Flaky;

#[async_trait]
impl FactExtractor for Flaky {
    fn id(&self) -> &str {
        "flaky"
    }
    async fn extract(&self, _: &Episode) -> Result<Vec<Claim>, ProviderError> {
        Err(ProviderError::Timeout)
    }
}

async fn job_retries(env: Env) {
    let k = env.k.with_extractor(Arc::new(Flaky));
    let c = "jobs";
    env.clock.set(day(2024, 8, 1));
    k.ingest(
        c,
        source("s", "team", "chats", "Notes"),
        vec![turn("a", day(2024, 8, 1), "Mira Sato leads Project Heron.")],
    )
    .await
    .unwrap();
    let queued = k.job_counts(c, "team").await.unwrap();
    assert!(queued.pending >= 2, "extract and project: {queued:?}");
    // Another partition of the same corpus sees none of them.
    assert_eq!(
        k.job_counts(c, "private").await.unwrap().outstanding(),
        0,
        "job counts are scoped to the partition"
    );
    let due = k.ledger().due_corpora(k.now(), 100).await.unwrap();
    assert!(due.iter().any(|d| d == c), "{due:?}");
    let first = k.run_jobs(c, 10).await.unwrap();
    assert_eq!(first.retried, 1, "{first:?}");
    let waiting = k.job_counts(c, "team").await.unwrap();
    assert_eq!((waiting.retry_scheduled, waiting.outstanding()), (1, 1));
    assert!(waiting.succeeded >= 1, "projection finished: {waiting:?}");
    // Nothing is due until the back-off passes.
    assert!(
        !k.ledger()
            .due_corpora(k.now(), 100)
            .await
            .unwrap()
            .contains(&c.to_string())
    );
    assert_eq!(k.run_jobs(c, 10).await.unwrap().retried, 0);
    env.clock.advance(chrono::Duration::minutes(5));
    // A background worker finds the corpus without being told about it.
    assert_eq!(k.run_due_jobs(100, 10).await.unwrap().retried, 1);
    env.clock.advance(chrono::Duration::minutes(5));
    let last = k.run_jobs(c, 10).await.unwrap();
    assert_eq!(last.failed, 1, "{last:?}");
    let done = k.job_counts(c, "team").await.unwrap();
    assert_eq!((done.failed, done.outstanding()), (1, 0), "{done:?}");
    let jobs = k.ledger().jobs(c).await.unwrap();
    let extract = jobs
        .iter()
        .find(|j| j.state == JobState::Failed)
        .expect("failed job kept for inspection");
    assert_eq!(extract.attempts, 3);
    assert!(
        extract
            .last_error
            .as_deref()
            .unwrap_or("")
            .contains("timed out")
    );
    // The episode is still searchable as raw text.
    let res = k
        .search(
            &search_req(c, everyone(), "Heron", AsOf::now()),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    assert!(!res.hits.is_empty());
    assert!(res.hits.iter().all(|h| matches!(h.item, ItemRef::Chunk(_))));
    k.delete_corpus(c).await.unwrap();
}

#[test]
fn rule_extractor_is_the_default() {
    assert_eq!(RuleExtractor.id(), RuleExtractor::ID);
}

#[tokio::test]
async fn in_memory_profile() {
    suite("in-memory", &Env::new(profile::in_memory())).await;
    job_retries(Env::new(profile::in_memory())).await;
}

fn visible_first(k: Knowledge) -> Knowledge {
    let config = KnowledgeConfig {
        lexical: LexicalMode::Bm25(Bm25Params::default()),
        ..k.config().clone()
    };
    k.with_config(config)
}

#[tokio::test]
async fn in_memory_profile_with_visible_first_bm25() {
    suite(
        "in-memory bm25",
        &Env::new(visible_first(profile::in_memory())),
    )
    .await;
}

/// Many later turns mention the query; the one visible turn must still be
/// found although it ranks below all of them before time filtering.
#[tokio::test]
async fn visible_first_search_keeps_early_items_that_later_ones_would_crowd_out() {
    let ingest = |k: Knowledge| async move {
        let env = Env::new(k);
        env.clock.set(day(2024, 1, 1));
        env.k
            .ingest(
                "crowd",
                source("early", "team", "chats", "Early"),
                vec![turn(
                    "e",
                    day(2024, 1, 1),
                    "We once talked about the heron, briefly, among many other topics that day.",
                )],
            )
            .await
            .unwrap();
        env.clock.set(day(2024, 6, 1));
        let later: Vec<_> = (0..80)
            .map(|i| turn(&format!("l{i}"), day(2024, 6, 1), "Heron heron heron."))
            .collect();
        env.k
            .ingest("crowd", source("late", "team", "chats", "Late"), later)
            .await
            .unwrap();
        env.k.run_jobs("crowd", 200).await.unwrap();
        let mut req = search_req(
            "crowd",
            everyone(),
            "heron",
            AsOf::known_at(day(2024, 2, 1)),
        );
        req.limit = 5;
        env.k
            .search(&req, &mut EvidenceAllowance::default())
            .await
            .unwrap()
            .hits
            .iter()
            .map(|h| h.citation.episode_id.clone())
            .collect::<Vec<_>>()
    };
    let classic = ingest(profile::in_memory()).await;
    assert!(classic.is_empty(), "the pool cut drops it: {classic:?}");
    let first = ingest(visible_first(profile::in_memory())).await;
    assert_eq!(first, ["e"]);
}

/// Visible-first BM25 indexes a chunk as a hit shows it, so a question that
/// names the speaker matches turns that never repeat their own name.
#[tokio::test]
async fn visible_first_bm25_matches_the_speaker() {
    let speaking = |speaker: &str, id: &str, text: &str| EpisodeInput {
        speaker: speaker.into(),
        ..turn(id, day(2024, 1, 1), text)
    };
    let top = |k: Knowledge| async move {
        let env = Env::new(k);
        env.k
            .ingest(
                "who",
                source("s", "team", "chats", "Notes"),
                vec![
                    speaking("Caroline", "c", "I went hiking last weekend."),
                    speaking("Melanie", "m", "I went hiking last weekend too."),
                ],
            )
            .await
            .unwrap();
        env.k.run_jobs("who", 10).await.unwrap();
        let req = search_req("who", everyone(), "Melanie hiking", AsOf::now());
        env.k
            .search(&req, &mut EvidenceAllowance::default())
            .await
            .unwrap()
            .hits
            .iter()
            .filter(|h| matches!(h.item, ItemRef::Chunk(_)))
            .map(|h| h.citation.episode_id.clone())
            .collect::<Vec<_>>()
    };
    let first = top(visible_first(Knowledge::new(
        Arc::new(keep_knowledge::MemoryLedger::new()),
        None,
    )))
    .await;
    assert_eq!(first.first().map(String::as_str), Some("m"), "{first:?}");
}

#[tokio::test]
async fn a_reranker_reorders_the_fused_head_only() {
    let k = visible_first(profile::in_memory());
    let env = Env::new(k);
    env.k
        .ingest(
            "rr",
            source("s", "team", "chats", "Notes"),
            vec![
                turn("a", day(2024, 1, 1), "Heron heron heron planning notes."),
                turn(
                    "b",
                    day(2024, 1, 1),
                    "Heron budget for the coastal office move.",
                ),
                turn("c", day(2024, 1, 1), "Heron kickoff."),
            ],
        )
        .await
        .unwrap();
    env.k.run_jobs("rr", 10).await.unwrap();
    let (plain, _) = chunk_order(&env.k).await;
    let reranked = env
        .k
        .sharing_storage(env.clock.clone())
        .with_reranker(Arc::new(Prefers::new("kickoff")), 50);
    let (ids, method) = chunk_order(&reranked).await;
    assert_eq!(ids[0], "c", "{plain:?} -> {ids:?}");
    assert_ne!(plain[0], "c", "{plain:?}");
    assert!(method.contains("rerank=mock:prefers"), "{method}");
    // Equal scores keep their fused order.
    let rest: Vec<String> = plain.iter().filter(|i| *i != "c").cloned().collect();
    assert_eq!(ids[1..], rest[..]);
    // A head of one leaves the fused order alone.
    let head_one = env
        .k
        .sharing_storage(env.clock.clone())
        .with_reranker(Arc::new(Prefers::new("kickoff")), 1);
    assert_eq!(chunk_order(&head_one).await.0, plain);
}

/// Scores 1 for passages containing a word and 0 otherwise.
struct Prefers(&'static str, keep_providers::ProviderDescriptor);

impl Prefers {
    fn new(word: &'static str) -> Self {
        Self(
            word,
            keep_providers::ProviderDescriptor {
                provider: "mock".into(),
                model: "prefers".into(),
                revision: "v1".into(),
                mock: true,
            },
        )
    }
}

#[async_trait]
impl Reranker for Prefers {
    fn descriptor(&self) -> &keep_providers::ProviderDescriptor {
        &self.1
    }
    async fn score(&self, _: &str, texts: &[String]) -> Result<Vec<f32>, ProviderError> {
        Ok(texts
            .iter()
            .map(|t| f32::from(u8::from(t.contains(self.0))))
            .collect())
    }
}

async fn chunk_order(k: &Knowledge) -> (Vec<String>, String) {
    let req = search_req("rr", everyone(), "heron coastal office budget", AsOf::now());
    let res = k
        .search(&req, &mut EvidenceAllowance::default())
        .await
        .unwrap();
    let ids = res
        .hits
        .iter()
        .filter(|h| matches!(h.item, ItemRef::Chunk(_)))
        .map(|h| h.citation.episode_id.clone())
        .collect();
    (ids, res.method)
}

#[test]
fn the_default_lexical_mode_is_left_out_of_serialized_configs() {
    let json = serde_json::to_value(KnowledgeConfig::default()).unwrap();
    assert!(json.get("lexical").is_none(), "{json}");
    let back: KnowledgeConfig = serde_json::from_value(json).unwrap();
    assert_eq!(back, KnowledgeConfig::default());
}

#[tokio::test]
async fn ledger_only_search_degrades_without_vectors() {
    let k = Knowledge::new(Arc::new(keep_knowledge::MemoryLedger::new()), None);
    let env = Env::new(k);
    env.k
        .ingest(
            "lex",
            source("s", "team", "chats", "Notes"),
            vec![turn("a", day(2024, 1, 1), "Mira Sato leads Project Heron.")],
        )
        .await
        .unwrap();
    env.k.run_jobs("lex", 10).await.unwrap();
    let res = env
        .k
        .search(
            &search_req("lex", everyone(), "Heron", AsOf::now()),
            &mut EvidenceAllowance::default(),
        )
        .await
        .unwrap();
    assert!(!res.hits.is_empty());
    assert_eq!(res.degraded, vec!["vector_disabled".to_string()]);
}

// ----- Postgres profiles ---------------------------------------------------

struct PgSchema {
    admin: String,
    schema: String,
}

impl PgSchema {
    async fn create(admin: &str) -> (Self, String) {
        // Parallel tests share a process id and can read the same clock tick.
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let schema = format!(
            "keep_knowledge_test_{}_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                % 1_000_000_000
        );
        let client = keep_postgres::connect(admin).await.unwrap();
        keep_postgres::ensure_vector_extension(&client)
            .await
            .unwrap();
        client
            .batch_execute(&format!("CREATE SCHEMA {schema}"))
            .await
            .unwrap();
        let sep = if admin.contains('?') { '&' } else { '?' };
        let url = format!("{admin}{sep}options=-csearch_path%3D{schema},public");
        (
            Self {
                admin: admin.into(),
                schema,
            },
            url,
        )
    }

    async fn drop(self) {
        let client = keep_postgres::connect(&self.admin).await.unwrap();
        client
            .batch_execute(&format!("DROP SCHEMA {} CASCADE", self.schema))
            .await
            .unwrap();
    }
}

fn test_collection() -> String {
    format!("knowledge_test_{}", std::process::id())
}

fn with_collection(k: Knowledge, collection: &str) -> Knowledge {
    k.with_config(KnowledgeConfig {
        collection: collection.into(),
        ..KnowledgeConfig::default()
    })
}

/// Run `body` and always clean up afterwards, even if it panics.
async fn guarded<F>(body: F, cleanup: impl std::future::Future<Output = ()>)
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let outcome = tokio::spawn(body).await;
    cleanup.await;
    if let Err(e) = outcome {
        std::panic::resume_unwind(e.into_panic());
    }
}

#[tokio::test]
async fn postgres_vector_profile() {
    let Ok(admin) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    let (schema, url) = PgSchema::create(&admin).await;
    let col = format!("{}_pg", test_collection());
    guarded(
        async move {
            let k = with_collection(profile::postgres_vector(&url).await.unwrap(), &col);
            suite("postgres-vector", &Env::new(k)).await;
            let k = with_collection(profile::postgres_vector(&url).await.unwrap(), &col);
            job_retries(Env::new(k)).await;
        },
        schema.drop(),
    )
    .await;
}

/// `KEEP_TEST_QDRANT_CA`'s PEM, for a Qdrant behind TLS with a private CA.
fn test_qdrant_ca() -> Option<Vec<u8>> {
    std::env::var_os("KEEP_TEST_QDRANT_CA").map(|p| std::fs::read(p).unwrap())
}

async fn qdrant_profile(url: &str, qdrant: &str) -> Knowledge {
    profile::postgres_qdrant_with_ca(
        url,
        qdrant,
        std::env::var("KEEP_TEST_QDRANT_API_KEY").ok(),
        test_qdrant_ca().as_deref(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn postgres_qdrant_profile() {
    let (Ok(admin), Ok(qdrant)) = (
        std::env::var("KEEP_TEST_POSTGRES_URL"),
        std::env::var("KEEP_TEST_QDRANT_URL"),
    ) else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL or KEEP_TEST_QDRANT_URL is not set");
        return;
    };
    let (schema, url) = PgSchema::create(&admin).await;
    let col = format!("{}_qd", test_collection());
    let (q, c) = (qdrant.clone(), col.clone());
    guarded(
        async move {
            let k = with_collection(qdrant_profile(&url, &q).await, &c);
            suite("postgres-qdrant", &Env::new(k)).await;
            let k = with_collection(qdrant_profile(&url, &q).await, &c);
            suite("postgres-qdrant bm25", &Env::new(visible_first(k))).await;
            let k = with_collection(qdrant_profile(&url, &q).await, &c);
            job_retries(Env::new(k)).await;
        },
        async move {
            let index = keep_qdrant::QdrantIndex::connect(
                &qdrant,
                std::env::var("KEEP_TEST_QDRANT_API_KEY").ok(),
                test_qdrant_ca().as_deref(),
            )
            .unwrap();
            let _ = index.delete_collection(&col).await;
            schema.drop().await;
        },
    )
    .await;
}

#[tokio::test]
async fn postgres_ledger_reconnects_after_the_server_closes_its_connection() {
    use keep_knowledge::ledger::Ledger;
    use keep_knowledge::pg_ledger::PgLedger;
    let Ok(admin) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    let (schema, url) = PgSchema::create(&admin).await;
    let app = schema.schema.clone();
    let admin_url = admin.clone();
    guarded(
        async move {
            let ledger = PgLedger::connect(&format!("{url}&application_name={app}"))
                .await
                .unwrap();
            assert!(ledger.point_ids("c").await.unwrap().is_empty());
            let client = keep_postgres::connect(&admin_url).await.unwrap();
            let killed = client
                .query_one(
                    "SELECT count(*) FROM (SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
                     WHERE application_name = $1) t",
                    &[&app],
                )
                .await
                .unwrap()
                .get::<_, i64>(0);
            assert_eq!(killed, 1);
            // The first use after the kill may see the dying connection.
            let mut last = ledger.point_ids("c").await;
            for _ in 0..20 {
                if last.is_ok() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                last = ledger.point_ids("c").await;
            }
            assert!(last.unwrap().is_empty());
        },
        schema.drop(),
    )
    .await;
}

/// Everything an ingest leaves behind that a benchmark reads: jobs, chunks,
/// facts, and the top vector hits for a few queries.
async fn ingest_snapshot(
    ledger: Arc<dyn keep_knowledge::ledger::Ledger>,
) -> (
    Vec<(String, String, JobState)>,
    Vec<keep_knowledge::model::Chunk>,
    Vec<keep_knowledge::model::Fact>,
    Vec<Vec<String>>,
) {
    use keep_providers::{Embedder, HashEmbedder};
    use keep_store::{InMemoryVectorIndex, VectorIndex};
    let index = Arc::new(InMemoryVectorIndex::new());
    let env = Env::new(Knowledge::new(
        ledger.clone(),
        Some((index.clone(), "memory".into())),
    ));
    let c = "equiv";
    let long = "Mira Sato works at Northwind Labs. ".repeat(80);
    let sessions = [
        vec![
            turn("a1", day(2024, 1, 2), "Tomas Reyes leads Project Heron."),
            turn("a2", day(2024, 1, 2), &long),
        ],
        vec![
            turn("b1", day(2024, 2, 3), "Mira Sato works at Contoso."),
            turn("b2", day(2024, 2, 3), "Lunch moved to Tuesday at noon."),
            turn("b3", day(2024, 2, 3), "Tomas Reyes lives in Lisbon."),
        ],
    ];
    for (i, eps) in sessions.into_iter().enumerate() {
        env.clock.set(day(2024, 1 + i as u32, 2));
        env.k
            .ingest(c, source(&format!("s{i}"), "team", "chats", "Notes"), eps)
            .await
            .unwrap();
        env.k.run_jobs(c, 50).await.unwrap();
    }
    let jobs = ledger
        .jobs(c)
        .await
        .unwrap()
        .into_iter()
        .map(|j| (j.kind.as_str().to_owned(), j.target, j.state))
        .collect();
    let ids = ledger.point_ids(c).await.unwrap();
    let chunks = ledger.chunks(c, &ids).await.unwrap();
    let facts = ledger.facts(c, &ids).await.unwrap();
    let embedder = HashEmbedder::new(256);
    let mut hits = Vec::new();
    for q in ["Where does Mira Sato work?", "Project Heron", "lunch"] {
        let v = embedder.embed(&[q.into()]).await.unwrap().vectors.remove(0);
        let ns = namespace(c, "team");
        let col = KnowledgeConfig::default().collection;
        hits.push(
            index
                .search(&col, &ns, &v, 10)
                .await
                .unwrap()
                .into_iter()
                .map(|h| h.id)
                .collect(),
        );
    }
    (jobs, chunks, facts, hits)
}

/// The Postgres ledger writes rows, jobs and completions in batches; the
/// in-memory ledger keeps the one-at-a-time defaults. Both must leave the
/// same chunks, facts, job queue and vector hits.
#[tokio::test]
async fn batched_postgres_writes_match_one_at_a_time_writes() {
    use keep_knowledge::pg_ledger::PgLedger;
    let Ok(admin) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    let (schema, url) = PgSchema::create(&admin).await;
    guarded(
        async move {
            let expected = ingest_snapshot(Arc::new(keep_knowledge::MemoryLedger::new())).await;
            let pg = Arc::new(PgLedger::connect(&url).await.unwrap());
            let got = ingest_snapshot(pg.clone()).await;
            assert!(!expected.1.is_empty() && !expected.2.is_empty());
            assert!(expected.0.iter().all(|j| j.2 == JobState::Succeeded));
            assert_eq!(got.0, expected.0, "jobs");
            assert_eq!(got.1, expected.1, "chunks");
            assert_eq!(got.2, expected.2, "facts");
            assert_eq!(got.3, expected.3, "vector hits");

            use keep_knowledge::ledger::Ledger;
            let mut ids = pg.point_ids("equiv").await.unwrap();
            ids.push("missing".into());
            let mut one_by_one = Vec::new();
            for id in &ids {
                one_by_one.push(pg.projection("equiv", id).await.unwrap());
            }
            assert_eq!(pg.projections("equiv", &ids).await.unwrap(), one_by_one);
        },
        schema.drop(),
    )
    .await;
}
