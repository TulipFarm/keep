//! The Knowledge arms end to end: dry run, mock run, and (opt-in) the two
//! Postgres storage profiles through the binary.

mod common;

use std::process::Command;

use keep_bench::artifacts::{Outcome, QuestionRecord};
use keep_bench::runner::{Mode, RunOptions, run};
use keep_bench::{dryrun, report};

const KM: &str = "knowledge-mock.toml";

fn opts(mode: Mode, root: Option<&std::path::Path>) -> RunOptions {
    RunOptions {
        mode,
        unseal: false,
        allow_model_calls: false,
        results_root: root.map(ToOwned::to_owned),
        spend: None,
    }
}

#[tokio::test]
async fn dry_run_builds_knowledge_contexts() {
    let lm = common::load(KM);
    let r = dryrun::dry_run(&lm).await.unwrap();
    for name in ["keep-k0", "keep-k1"] {
        let a = r.arms.iter().find(|a| a.arm == name).unwrap();
        assert_eq!((a.not_run, a.failed, a.unavailable), (0, 0, 0), "{name}");
        assert_eq!(a.answer_calls, 24);
        assert!(a.context_tokens_max > 0);
    }
    let mk = r.arms.iter().find(|a| a.arm == "keep-mk1").unwrap();
    assert_eq!((mk.not_run, mk.answer_calls), (0, 24));
}

fn check_records(records: &[QuestionRecord], evidence_tokens: usize) {
    let mut toc_seen = false;
    for r in records.iter().filter(|r| r.arm.starts_with("keep-k")) {
        assert!(
            matches!(r.outcome, Outcome::Answered),
            "{} {}: {:?}",
            r.arm,
            r.question_id,
            r.outcome
        );
        let c = r.context.as_ref().unwrap();
        let method = c.method.as_deref().unwrap();
        assert!(method.starts_with("keep-knowledge;"), "{method}");
        assert!(c.tokens <= c.budget_tokens.unwrap());
        // Evidence never exceeds the retained allowance; K1 may add the
        // table of contents on top, within its own slot.
        if r.arm == "keep-k0" {
            assert!(c.tokens <= evidence_tokens + c.included_ids.len());
        } else {
            toc_seen |= c.budget_tokens.unwrap() > evidence_tokens + c.included_ids.len();
        }
    }
    assert!(toc_seen, "K1 never used its table-of-contents slot");
}

#[tokio::test]
async fn mock_run_answers_with_knowledge_arms() {
    let dir = tempfile::tempdir().unwrap();
    let lm = common::load(KM);
    let out = run(&lm, &opts(Mode::Execute, Some(dir.path())))
        .await
        .unwrap();
    assert_eq!(out.records.len(), 5 * 8 * 3);
    check_records(&out.records, lm.manifest.retrieval.evidence_tokens);
    let k0: Vec<_> = out.records.iter().filter(|r| r.arm == "keep-k0").collect();
    // Same questions and seeds, so the arms are paired.
    assert_eq!(k0.len(), 24);
    let summary = report::summarize(&out.meta, &lm.manifest, &out.records);
    for name in ["keep-k0", "keep-k1"] {
        let a = summary.arms.iter().find(|a| a.arm == name).unwrap();
        assert!(a.accuracy_scheduled.accuracy.is_some(), "{name}");
    }
    // Same manifest, same answers.
    let again = run(&lm, &opts(Mode::Execute, None)).await.unwrap();
    let answers = |rs: &[QuestionRecord]| -> Vec<Option<String>> {
        rs.iter()
            .filter(|r| r.arm.starts_with("keep-k"))
            .map(|r| r.answer.as_ref().and_then(|a| a.text.clone()))
            .collect()
    };
    assert_eq!(answers(&out.records), answers(&again.records));
}

/// Runs the manifest through the binary with `profile` against a private
/// Postgres schema (and Qdrant collection), then removes them.
async fn profile_run(profile: &str, admin: &str, qdrant: Option<&str>) {
    let schema = format!(
        "keep_knowledge_test_bench_{}_{}",
        profile.replace('-', "_"),
        std::process::id()
    );
    let (client, conn) = tokio_postgres::connect(admin, tokio_postgres::NoTls)
        .await
        .unwrap();
    tokio::spawn(conn);
    client
        .batch_execute(&format!(
            "CREATE EXTENSION IF NOT EXISTS vector; DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema};"
        ))
        .await
        .unwrap();
    let sep = if admin.contains('?') { '&' } else { '?' };
    let url = format!("{admin}{sep}options=-csearch_path%3D{schema},public");

    let dir = tempfile::tempdir().unwrap();
    let text = common::fixture_manifest(KM).replace(
        "[storage]\nprofile = \"in-memory\"",
        &format!(
            "[storage]\nprofile = \"{profile}\"\npostgres_url_env = \"KEEP_KNOWLEDGE_BENCH_PG\"\nqdrant_url_env = \"KEEP_KNOWLEDGE_BENCH_QDRANT\""
        ),
    );
    let lm = common::write_manifest(dir.path(), &text);
    let results = dir.path().join("results");
    let out = Command::new(env!("CARGO_BIN_EXE_keep-bench"))
        .args(["run", lm.path.to_str().unwrap(), "--results"])
        .arg(&results)
        .env("KEEP_KNOWLEDGE_BENCH_PG", &url)
        .env("KEEP_KNOWLEDGE_BENCH_QDRANT", qdrant.unwrap_or(""))
        .output()
        .unwrap();

    // The run drops its own dense and Knowledge collections.
    let mut kept_collections = Vec::new();
    if let Some(q) = qdrant {
        let base = q.trim_end_matches('/');
        let list: serde_json::Value = reqwest::get(format!("{base}/collections"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        for c in list["result"]["collections"].as_array().unwrap() {
            let name = c["name"].as_str().unwrap();
            if name.starts_with("keepbench_") && name.ends_with(&format!("z_{}", lm.hash.short())) {
                kept_collections.push(name.to_owned());
                let _ = reqwest::Client::new()
                    .delete(format!("{base}/collections/{name}"))
                    .send()
                    .await;
            }
        }
    }
    let leftover: i64 = client
        .query_one(
            &format!("SELECT count(*) FROM {schema}.knowledge_sources"),
            &[],
        )
        .await
        .map(|r| r.get(0))
        .unwrap_or(-1);
    client
        .batch_execute(&format!("DROP SCHEMA {schema} CASCADE"))
        .await
        .unwrap();

    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(leftover, 0, "the run removes its corpora");
    assert!(kept_collections.is_empty(), "{kept_collections:?}");
    let run_dir = std::fs::read_dir(&results)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let loaded = keep_bench::compare::load_run(&run_dir).unwrap();
    assert_eq!(
        format!("{:?}", loaded.meta.storage_profile).to_lowercase(),
        profile.replace('-', "")
    );
    check_records(&loaded.records, lm.manifest.retrieval.evidence_tokens);
}

#[tokio::test]
async fn postgres_vector_profile_runs_knowledge_arms() {
    let Ok(admin) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    profile_run("postgres-vector", &admin, None).await;
}

#[tokio::test]
async fn postgres_qdrant_profile_runs_knowledge_arms() {
    let (Ok(admin), Ok(qdrant)) = (
        std::env::var("KEEP_TEST_POSTGRES_URL"),
        std::env::var("KEEP_TEST_QDRANT_URL"),
    ) else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL or KEEP_TEST_QDRANT_URL is not set");
        return;
    };
    profile_run("postgres-qdrant", &admin, Some(&qdrant)).await;
}

/// Ingesting every history up front, several at a time, leaves the stores
/// one-at-a-time ingestion leaves: every dense and Keep context is the same.
#[tokio::test]
async fn concurrent_prepare_builds_the_same_contexts_as_lazy_ingestion() {
    use std::sync::Arc;

    use keep_bench::ablation::{ArmKind, KeepPath};
    use keep_bench::baselines::{Baselines, Built, DenseBackend};
    use keep_bench::knowledge_arm::{KnowledgeArms, KnowledgeStorage};
    use keep_providers::{Embedder, HashEmbedder};
    use keep_store::InMemoryVectorIndex;

    let lm = common::load(KM);
    let m = &lm.manifest;
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let (tokenizer, _) = m.answer_tokenizer();
    let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(m.embedding.dimensions));
    let setup = || async {
        let dense = DenseBackend::new(
            embedder.clone(),
            Arc::new(InMemoryVectorIndex::new()),
            "dense".into(),
            m.embedding.request_batch(),
        );
        let base = Baselines::new(
            tokenizer.clone(),
            m.models.answer.clone(),
            m.retrieval.clone(),
            Some(dense),
        );
        let k = KnowledgeArms::new(
            KnowledgeStorage::InMemory,
            embedder.clone(),
            tokenizer.clone(),
            m.retrieval.clone(),
            "prepare-test",
            "unused".into(),
            None,
        )
        .await
        .unwrap();
        (base, k)
    };
    let (mut lazy_base, mut lazy_k) = setup().await;
    let (mut eager_base, mut eager_k) = setup().await;
    let histories: Vec<_> = data
        .questions
        .iter()
        .map(|q| data.history(&q.history_id).unwrap())
        .collect();
    assert!(histories.len() > 1);
    let dense = eager_base.dense.as_mut().unwrap();
    assert!(
        dense
            .prepare(&histories, m.retrieval.chunking, 4)
            .await
            .is_empty()
    );
    assert!(eager_k.prepare(&histories, 4).await.is_empty());

    let show = |b: Built| match b {
        Built::Ready(c) => format!("{c:?}"),
        other => panic!("{other:?}"),
    };
    let arm = ArmKind::Dense {
        budget_tokens: 500,
        top_k: 5,
    };
    // Lazy ingestion in reverse order, so the stores also differ in the
    // order their histories arrived.
    for q in data.questions.iter().rev() {
        let h = data.history(&q.history_id).unwrap();
        let lazy = show(lazy_base.build(&arm, h, &q.query, None).await);
        let eager = show(eager_base.build(&arm, h, &q.query, None).await);
        assert_eq!(lazy, eager, "dense {}", q.id);
        for path in [KeepPath::K0, KeepPath::K1] {
            let lazy = show(lazy_k.build(path, h, &q.query).await);
            let eager = show(eager_k.build(path, h, &q.query).await);
            assert_eq!(lazy, eager, "{path:?} {}", q.id);
        }
    }
}

/// Given a plain database URL, a run works in its own schema and drops it.
#[tokio::test]
async fn postgres_runs_work_in_their_own_schema() {
    let Ok(admin) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    let (client, conn) = tokio_postgres::connect(&admin, tokio_postgres::NoTls)
        .await
        .unwrap();
    tokio::spawn(conn);
    client
        .batch_execute("CREATE EXTENSION IF NOT EXISTS vector")
        .await
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let text = common::fixture_manifest(KM).replace(
        "[storage]\nprofile = \"in-memory\"",
        "[storage]\nprofile = \"postgres-vector\"\npostgres_url_env = \"KEEP_KNOWLEDGE_BENCH_OWN_PG\"",
    );
    let lm = common::write_manifest(dir.path(), &text);
    let pattern = format!("keepbench\\_%z\\_{}", lm.hash.short());
    let schemas = || async {
        client
            .query(
                "SELECT nspname FROM pg_namespace WHERE nspname LIKE $1",
                &[&pattern],
            )
            .await
            .unwrap()
            .len()
    };
    assert_eq!(schemas().await, 0);
    let out = Command::new(env!("CARGO_BIN_EXE_keep-bench"))
        .args(["run", lm.path.to_str().unwrap(), "--results"])
        .arg(dir.path().join("results"))
        .env("KEEP_KNOWLEDGE_BENCH_OWN_PG", &admin)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    let schema = stderr
        .split_whitespace()
        .skip_while(|w| *w != "schema")
        .nth(1)
        .unwrap_or_else(|| panic!("no schema reported: {stderr}"));
    assert!(schema.starts_with("keepbench_"), "{schema}");
    assert_eq!(schemas().await, 0, "the run drops its schema");
    let in_public: i64 = client
        .query_one(
            "SELECT count(*) FROM information_schema.tables
             WHERE table_schema = 'public' AND table_name = 'knowledge_sources'",
            &[],
        )
        .await
        .unwrap()
        .get(0);
    if in_public > 0 {
        let rows: i64 = client
            .query_one(
                "SELECT count(*) FROM public.knowledge_sources WHERE corpus LIKE $1",
                &[&format!("bench-{}-%", lm.hash.short())],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(rows, 0, "nothing lands in the public schema");
    }
}

/// The existing Keep arms (k0, k1, mk0, mk1) build byte-identical contexts
/// to the release that produced the first G0 results, so those runs stay
/// reproducible. The digest was taken from that release's code.
#[tokio::test]
async fn existing_keep_arms_build_the_same_contexts_as_before() {
    use std::sync::Arc;

    use keep_bench::ablation::{KeepPath, Representation};
    use keep_bench::baselines::Built;
    use keep_bench::knowledge_arm::{KnowledgeArms, KnowledgeStorage};
    use keep_bench::memory_arm::MemoryArm;
    use keep_providers::{Embedder, HashEmbedder};
    use sha2::{Digest, Sha256};

    let lm = common::load(KM);
    let m = &lm.manifest;
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let (tokenizer, _) = m.answer_tokenizer();
    let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(m.embedding.dimensions));
    let mut k = KnowledgeArms::new(
        KnowledgeStorage::InMemory,
        embedder,
        tokenizer,
        m.retrieval.clone(),
        "golden",
        "unused".into(),
        None,
    )
    .await
    .unwrap();
    let mut memory = MemoryArm::default();
    let mut digest = Sha256::new();
    for q in &data.questions {
        let h = data.history(&q.history_id).unwrap();
        for path in [KeepPath::K0, KeepPath::K1, KeepPath::Mk0, KeepPath::Mk1] {
            let built = if path.uses_memory() {
                k.build_with_memory(path, Representation::Structured, &mut memory, h, &q.query)
                    .await
            } else {
                k.build(path, h, &q.query).await
            };
            let Built::Ready(c) = built else {
                panic!("{path:?} {}: {built:?}", q.id)
            };
            digest.update(format!("{}|{path:?}|{c:?}\n", q.id));
        }
    }
    let got = hex::encode(digest.finalize());
    assert_eq!(
        got, GOLDEN_KEEP_CONTEXTS,
        "existing Keep arm contexts changed"
    );
}

const GOLDEN_KEEP_CONTEXTS: &str =
    "3bc65092531d491015a9ea2b392c0614bf5d1b8064a1e8c5a17bba1d680c73da";

const RANKED: &str = "ranked-mock.toml";

/// Fresh in-memory ranked arms over the fixture, with an optional reranker.
async fn ranked_arms(
    lm: &keep_bench::manifest::LoadedManifest,
    retrieval: keep_bench::manifest::RetrievalSpec,
    reranker: Option<(std::sync::Arc<dyn keep_providers::Reranker>, usize)>,
) -> keep_bench::knowledge_arm::KnowledgeArms {
    use keep_bench::knowledge_arm::{KnowledgeArms, KnowledgeStorage};
    let m = &lm.manifest;
    let (tokenizer, _) = m.answer_tokenizer();
    KnowledgeArms::new(
        KnowledgeStorage::InMemory,
        std::sync::Arc::new(keep_providers::HashEmbedder::new(m.embedding.dimensions)),
        tokenizer,
        retrieval,
        "ranked",
        "unused".into(),
        reranker,
    )
    .await
    .unwrap()
}

async fn build_ranked(
    k: &mut keep_bench::knowledge_arm::KnowledgeArms,
    memory: &mut keep_bench::memory_arm::MemoryArm,
    path: keep_bench::ablation::KeepPath,
    h: &keep_bench::dataset::History,
    q: &keep_bench::dataset::Query,
) -> keep_bench::context::ContextBundle {
    use keep_bench::ablation::{KeepPath, Representation};
    let mem = (path == KeepPath::Mk2).then_some(memory);
    match k
        .build_ranked(
            path,
            Representation::Structured,
            mem,
            h,
            q,
            Default::default(),
        )
        .await
    {
        keep_bench::baselines::Built::Ready(c) => c,
        other => panic!("{path:?} {}: {other:?}", q.id),
    }
}

#[tokio::test]
async fn mock_run_answers_with_ranked_arms() {
    let dir = tempfile::tempdir().unwrap();
    let lm = common::load(RANKED);
    let out = run(&lm, &opts(Mode::Execute, Some(dir.path())))
        .await
        .unwrap();
    let r = &lm.manifest.retrieval;
    for (arm, budget) in [
        ("keep-k2", r.evidence_tokens),
        ("keep-mk2", r.standing_tokens + r.evidence_tokens),
    ] {
        let records: Vec<_> = out.records.iter().filter(|x| x.arm == arm).collect();
        assert_eq!(records.len(), 24, "{arm}");
        for x in records {
            assert!(
                matches!(x.outcome, Outcome::Answered),
                "{arm}: {:?}",
                x.outcome
            );
            let c = x.context.as_ref().unwrap();
            assert_eq!(c.budget_tokens, Some(budget), "{arm}");
            assert!(c.tokens <= budget, "{arm}: {} > {budget}", c.tokens);
            let method = c.method.as_deref().unwrap();
            assert!(method.starts_with("keep-ranked;hybrid("), "{method}");
            assert!(method.contains("visible-first"), "{method}");
        }
    }
}

/// The ranked arms show each source turn at most once, stay within their
/// budget and fill it whenever the visible history is larger than it.
#[tokio::test]
async fn ranked_arms_fill_their_budget_without_repeating_a_turn() {
    use keep_bench::ablation::KeepPath;
    use keep_bench::memory_arm::MemoryArm;

    let lm = common::load(RANKED);
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let (tokenizer, _) = lm.manifest.answer_tokenizer();
    let mut retrieval = lm.manifest.retrieval.clone();
    // Small enough that every history overflows it.
    retrieval.standing_tokens = 96;
    retrieval.evidence_tokens = 64;
    let mut k = ranked_arms(&lm, retrieval.clone(), None).await;
    let mut memory = MemoryArm::default();
    for q in &data.questions {
        let h = data.history(&q.history_id).unwrap();
        // Each visible turn, rendered as the context shows it whole.
        let visible: Vec<(&str, String)> = h
            .visible_sessions(q.query.asked_at)
            .0
            .into_iter()
            .flat_map(|s| {
                let date = s
                    .date
                    .map_or_else(|| "undated".into(), |d| d.date_naive().to_string());
                s.turns
                    .iter()
                    .filter(|t| !t.text.trim().is_empty())
                    .map(move |t| {
                        let line = format!("[{date}] {}: {}", t.role.label(), t.text.trim());
                        (t.id.as_str(), line)
                    })
            })
            .collect();
        let largest = visible
            .iter()
            .map(|(_, line)| tokenizer.count(line) + 1)
            .max()
            .unwrap();
        for (path, budget) in [
            (KeepPath::K2, retrieval.evidence_tokens),
            (
                KeepPath::Mk2,
                retrieval.standing_tokens + retrieval.evidence_tokens,
            ),
        ] {
            let c = build_ranked(&mut k, &mut memory, path, h, &q.query).await;
            assert!(c.tokens <= budget, "{path:?} {}", q.id);
            assert_eq!(c.blocks.len(), 1);
            assert!(
                c.tokens + largest.min(budget) >= budget,
                "{path:?} {}: {} tokens left unused",
                q.id,
                budget - c.tokens
            );
            let mut ids = c.included_ids.clone();
            ids.sort();
            ids.dedup();
            assert_eq!(ids.len(), c.included_ids.len(), "{path:?} repeats an item");
            for (id, line) in &visible {
                assert!(
                    c.text.matches(line.as_str()).count() <= 1,
                    "{path:?} {} shows turn {id} twice",
                    q.id
                );
            }
        }
    }
}

/// Ranking is deterministic: two fresh runs build identical contexts.
#[tokio::test]
async fn ranked_arms_are_deterministic() {
    use keep_bench::ablation::KeepPath;
    use keep_bench::memory_arm::MemoryArm;

    let lm = common::load(RANKED);
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let mut contexts = Vec::new();
    for _ in 0..2 {
        let mut k = ranked_arms(&lm, lm.manifest.retrieval.clone(), None).await;
        let mut memory = MemoryArm::default();
        let mut all = Vec::new();
        for q in &data.questions {
            let h = data.history(&q.history_id).unwrap();
            for path in [KeepPath::K2, KeepPath::Mk2] {
                let c = build_ranked(&mut k, &mut memory, path, h, &q.query).await;
                all.push(format!("{c:?}"));
            }
        }
        contexts.push(all);
    }
    assert_eq!(contexts[0], contexts[1]);
}

/// Scores 1 for passages containing `marker`, 0 otherwise.
struct Prefers(String, keep_providers::ProviderDescriptor);

#[async_trait::async_trait]
impl keep_providers::Reranker for Prefers {
    fn descriptor(&self) -> &keep_providers::ProviderDescriptor {
        &self.1
    }
    async fn score(
        &self,
        _query: &str,
        texts: &[String],
    ) -> Result<Vec<f32>, keep_providers::ProviderError> {
        Ok(texts
            .iter()
            .map(|t| f32::from(u8::from(t.contains(&self.0))))
            .collect())
    }
}

/// With a reranker, both ranked arms show the passage it prefers first,
/// even one fusion alone would have left out of a small budget.
#[tokio::test]
async fn the_reranker_decides_what_fills_a_small_budget() {
    use std::sync::Arc;

    use keep_bench::ablation::KeepPath;
    use keep_bench::memory_arm::MemoryArm;

    let lm = common::load(RANKED);
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let (tokenizer, _) = lm.manifest.answer_tokenizer();
    let mut retrieval = lm.manifest.retrieval.clone();
    retrieval.standing_tokens = 0;
    retrieval.evidence_tokens = 64;
    let q = &data.questions[0];
    let h = data.history(&q.history_id).unwrap();
    let mut memory = MemoryArm::default();
    for path in [KeepPath::K2, KeepPath::Mk2] {
        let mut plain = ranked_arms(&lm, retrieval.clone(), None).await;
        let before = build_ranked(&mut plain, &mut memory, path, h, &q.query).await;
        // A user turn (so Memory also remembers it) that fits but was left out.
        let target = h
            .visible_sessions(q.query.asked_at)
            .0
            .into_iter()
            .flat_map(|s| s.turns.iter())
            .find(|t| {
                t.role == keep_core::knowledge::Role::User
                    && !before.included_turn_ids.contains(&t.id)
                    && (4..48).contains(&tokenizer.count(t.text.trim()))
            })
            .expect("a left-out user turn that fits");
        let marker = t_marker(&target.text);
        let reranker = Prefers(
            marker,
            keep_providers::ProviderDescriptor {
                provider: "test".into(),
                model: "prefers".into(),
                revision: "v1".into(),
                mock: true,
            },
        );
        let mut k = ranked_arms(&lm, retrieval.clone(), Some((Arc::new(reranker), 1000))).await;
        let after = build_ranked(&mut k, &mut memory, path, h, &q.query).await;
        assert!(
            after.included_turn_ids.contains(&target.id),
            "{path:?}: {:?}",
            after.included_turn_ids
        );
        assert!(
            after
                .method
                .as_deref()
                .unwrap()
                .contains("rerank=test:prefers@v1"),
            "{:?}",
            after.method
        );
    }
}

/// A distinctive slice of a turn's text.
fn t_marker(text: &str) -> String {
    text.trim().chars().take(24).collect()
}

/// Puts the question and dated source turns containing `marker` on one axis
/// and everything else on another, so only Mk2's dense Memory ranking (which
/// embeds source turns as `[date] Role: text`) can tell the target apart.
struct Points {
    marker: String,
    query: String,
    descriptor: keep_providers::ProviderDescriptor,
}

#[async_trait::async_trait]
impl keep_providers::Embedder for Points {
    fn descriptor(&self) -> &keep_providers::ProviderDescriptor {
        &self.descriptor
    }
    fn dimensions(&self) -> usize {
        2
    }
    async fn embed(
        &self,
        texts: &[String],
    ) -> Result<keep_providers::EmbedResponse, keep_providers::ProviderError> {
        let vectors = texts
            .iter()
            .map(|t| {
                if *t == self.query || (t.starts_with('[') && t.contains(&self.marker)) {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                }
            })
            .collect();
        Ok(keep_providers::EmbedResponse {
            vectors,
            usage: Default::default(),
        })
    }
}

/// Mk2 ranks Memory entries by the dense similarity of their source turns.
/// Knowledge vectors and both BM25 lists are the same for both embedders
/// here, so any change in the joint order comes from that Memory ranking.
#[tokio::test]
async fn mk2_ranks_memory_by_the_similarity_of_source_turns() {
    use std::sync::Arc;

    use keep_bench::ablation::{KeepPath, Representation};
    use keep_bench::knowledge_arm::{KnowledgeArms, KnowledgeStorage};
    use keep_bench::memory_arm::MemoryArm;

    let lm = common::load(RANKED);
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let retrieval = lm.manifest.retrieval.clone();
    let q = &data.questions[0];
    let h = data.history(&q.history_id).unwrap();
    // No word of the question occurs in the history, so neither BM25 list
    // can find the target.
    let query = keep_bench::dataset::Query {
        text: "zqxv".into(),
        ..q.query.clone()
    };
    let query = &query;
    let lm = &lm;
    let build = |marker: String| {
        let retrieval = retrieval.clone();
        async move {
            let (tokenizer, _) = lm.manifest.answer_tokenizer();
            let embedder = Points {
                marker,
                query: query.text.clone(),
                descriptor: keep_providers::ProviderDescriptor {
                    provider: "test".into(),
                    model: "points".into(),
                    revision: "v1".into(),
                    mock: true,
                },
            };
            let mut k = KnowledgeArms::new(
                KnowledgeStorage::InMemory,
                Arc::new(embedder),
                tokenizer,
                retrieval,
                "points",
                "unused".into(),
                None,
            )
            .await
            .unwrap();
            let mut memory = MemoryArm::default();
            match k
                .build_ranked(
                    KeepPath::Mk2,
                    Representation::Structured,
                    Some(&mut memory),
                    h,
                    query,
                    Default::default(),
                )
                .await
            {
                keep_bench::baselines::Built::Ready(c) => c,
                other => panic!("{other:?}"),
            }
        }
    };
    let control = build("no turn says this".into()).await;
    let entries = MemoryArm::default()
        .entries(Representation::Structured, h, query)
        .await
        .unwrap();
    assert!(entries.len() > 1, "{entries:?}");
    let mut orders = std::collections::BTreeSet::new();
    for e in &entries {
        let turn = h
            .sessions
            .iter()
            .flat_map(|s| &s.turns)
            .find(|t| t.id == e.turn_id)
            .unwrap();
        let c = build(t_marker(&turn.text)).await;
        let method = c.method.as_deref().unwrap();
        assert!(method.contains("+dense(test:points"), "{method}");
        orders.insert(c.retrieved_ids);
    }
    // Pointing the question at different remembered turns changes the joint
    // ranking.
    orders.insert(control.retrieved_ids);
    assert!(orders.len() > 1, "{orders:?}");
}

const FACTS: &str = "facts-mock.toml";

/// Ranked arms over the fixture with mock-extracted facts enabled.
async fn fact_arms(
    lm: &keep_bench::manifest::LoadedManifest,
    retrieval: keep_bench::manifest::RetrievalSpec,
) -> keep_bench::knowledge_arm::KnowledgeArms {
    use std::sync::Arc;
    let m = &lm.manifest;
    let data = keep_bench::prepare::prepare(lm, false).unwrap().dataset;
    let spec = m.extract_model().unwrap();
    let extractor = Arc::new(keep_bench::facts::extractor(
        spec,
        Some(Arc::new(keep_bench::mocks::extractor())),
    ));
    let sessions = data
        .histories
        .iter()
        .flat_map(|h| h.sessions.iter().map(keep_bench::facts::session_input))
        .collect();
    let stats = keep_bench::facts::warm_cache(extractor.clone(), sessions, 4, 1).await;
    assert_eq!((stats.failed, stats.missing), (0, 0), "{stats:?}");
    let mut k = ranked_arms(lm, retrieval, None).await;
    let x = m.extraction.as_ref().unwrap();
    k.enable_facts(extractor, x.facts_tokens, x.profile_tokens);
    k
}

/// K3 and Mk3 put a dated facts block ahead of the ranked fill, stay within
/// K2's and Mk2's budgets, and cite only turns visible when the question
/// was asked.
#[tokio::test]
async fn fact_arms_lead_with_dated_facts_within_budget() {
    use keep_bench::ablation::{KeepPath, Representation};
    use keep_bench::baselines::Built;
    use keep_bench::memory_arm::MemoryArm;

    let lm = common::load(FACTS);
    let r = lm.manifest.retrieval.clone();
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let mut k = fact_arms(&lm, r.clone()).await;
    let mut memory = MemoryArm::default();
    let mut with_facts = 0;
    for q in &data.questions {
        let h = data.history(&q.history_id).unwrap();
        let visible: std::collections::BTreeSet<&str> = h
            .visible_sessions(q.query.asked_at)
            .0
            .into_iter()
            .flat_map(|s| s.turns.iter().map(|t| t.id.as_str()))
            .collect();
        for (path, budget) in [
            (KeepPath::K3, r.evidence_tokens),
            (KeepPath::Mk3, r.standing_tokens + r.evidence_tokens),
        ] {
            let mem = (path == KeepPath::Mk3).then_some(&mut memory);
            let Built::Ready(c) = k
                .build_ranked(
                    path,
                    Representation::Structured,
                    mem,
                    h,
                    &q.query,
                    Default::default(),
                )
                .await
            else {
                panic!("{path:?} {}", q.id)
            };
            assert!(c.tokens <= budget, "{path:?} {}: {}", q.id, c.tokens);
            assert_eq!(c.budget_tokens, Some(budget));
            let method = c.method.as_deref().unwrap();
            assert!(method.starts_with("keep-facts;"), "{method}");
            assert_eq!(c.blocks.len(), 2);
            if c.text.starts_with(keep_bench::facts::FACTS_HEADER) {
                with_facts += 1;
                assert!(c.text.contains(keep_bench::facts::EVIDENCE_HEADER));
                assert!(c.included_ids[0].starts_with("fact_"));
            }
            for t in &c.included_turn_ids {
                assert!(visible.contains(t.as_str()), "{path:?} {} cites {t}", q.id);
            }
        }
    }
    assert!(with_facts > 0, "no context showed extracted facts");
}

/// Mk3 with temporal and preference options opens with the dated Memory
/// preference profile, shows the question's time window and each fact's age,
/// and still fits Mk2's budget with only visible turns cited.
#[tokio::test]
async fn mk3_options_add_profile_and_question_window() {
    use keep_bench::ablation::{FactOptions, KeepPath, Representation};
    use keep_bench::baselines::Built;
    use keep_bench::memory_arm::MemoryArm;

    let lm = common::load(FACTS);
    let r = lm.manifest.retrieval.clone();
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let mut k = fact_arms(&lm, r.clone()).await;
    let mut memory = MemoryArm::default();
    let opts = FactOptions {
        temporal: true,
        preferences: true,
        assistant_route: true,
        ..Default::default()
    };
    let q = &data.questions[0];
    let h = data.history(&q.history_id).unwrap();
    let mut query = q.query.clone();
    query.text = "What hobby did I pick up in the past 3 months?".into();
    let c = match k
        .build_ranked(
            KeepPath::Mk3,
            Representation::Structured,
            Some(&mut memory),
            h,
            &query,
            opts,
        )
        .await
    {
        Built::Ready(c) => c,
        other => panic!("mk3 {}: {other:?}", q.id),
    };
    let budget = r.standing_tokens + r.evidence_tokens;
    assert!(c.tokens <= budget, "{}", c.tokens);
    assert!(
        c.text
            .starts_with("# User preferences, habits and opinions"),
        "{}",
        c.text
    );
    assert!(c.text.contains("I love it."), "{}", c.text);
    assert!(
        c.text
            .contains("Question time reference: \"past 3 months\" = 2023-03-01 to 2023-06-01."),
        "{}",
        c.text
    );
    assert!(c.text.contains("days before the question]"), "{}", c.text);
    let names: Vec<&str> = c.blocks.iter().map(|b| b.name.as_str()).collect();
    assert_eq!(names, ["keep-profile", "keep-facts", "keep-ranked"]);
    let method = c.method.as_deref().unwrap();
    assert!(
        method.contains("profile=memory(extracted-preferences-v1"),
        "{method}"
    );
    assert!(
        method.contains(";window=2023-03-01..2023-06-01("),
        "{method}"
    );
    let visible: std::collections::BTreeSet<&str> = h
        .visible_sessions(query.asked_at)
        .0
        .into_iter()
        .flat_map(|s| s.turns.iter().map(|t| t.id.as_str()))
        .collect();
    for t in &c.included_turn_ids {
        assert!(visible.contains(t.as_str()), "cites {t}");
    }
}

/// With `assistant_route`, a question about the assistant's own earlier
/// reply gets Mk2's ranked context unchanged, without facts or profile.
#[tokio::test]
async fn mk3_routes_questions_about_earlier_replies_to_the_ranked_fill() {
    use keep_bench::ablation::{FactOptions, KeepPath, Representation};
    use keep_bench::baselines::Built;
    use keep_bench::memory_arm::MemoryArm;

    let lm = common::load(FACTS);
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let mut k = fact_arms(&lm, lm.manifest.retrieval.clone()).await;
    let mut memory = MemoryArm::default();
    let q = &data.questions[0];
    let h = data.history(&q.history_id).unwrap();
    let mut query = q.query.clone();
    query.text = "Can you remind me which trail you suggested for the weekend?".into();
    let mut build = async |path, opts| match k
        .build_ranked(
            path,
            Representation::Structured,
            Some(&mut memory),
            h,
            &query,
            opts,
        )
        .await
    {
        Built::Ready(c) => c,
        other => panic!("{path:?}: {other:?}"),
    };
    let routed = build(
        KeepPath::Mk3,
        FactOptions {
            temporal: true,
            preferences: true,
            assistant_route: true,
            ..Default::default()
        },
    )
    .await;
    let plain = build(KeepPath::Mk2, FactOptions::default()).await;
    let unrouted = build(
        KeepPath::Mk3,
        FactOptions {
            temporal: true,
            preferences: true,
            assistant_route: false,
            ..Default::default()
        },
    )
    .await;
    assert_eq!(routed.text, plain.text);
    assert_eq!(routed.included_ids, plain.included_ids);
    let method = routed.method.as_deref().unwrap();
    assert!(
        method.starts_with("keep-facts;route=assistant-reply(ranked only);"),
        "{method}"
    );
    assert_ne!(unrouted.text, plain.text);
}

/// K2 and Mk2 build byte-identical contexts to the release before the fact
/// arms, so v2 runs stay reproducible. The digest was taken from that
/// release's code.
#[tokio::test]
async fn ranked_arms_build_the_same_contexts_as_before_fact_arms() {
    use keep_bench::ablation::KeepPath;
    use keep_bench::memory_arm::MemoryArm;
    use sha2::{Digest, Sha256};

    let lm = common::load(RANKED);
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let mut k = ranked_arms(&lm, lm.manifest.retrieval.clone(), None).await;
    let mut memory = MemoryArm::default();
    let mut digest = Sha256::new();
    for q in &data.questions {
        let h = data.history(&q.history_id).unwrap();
        for path in [KeepPath::K2, KeepPath::Mk2] {
            let c = build_ranked(&mut k, &mut memory, path, h, &q.query).await;
            digest.update(format!("{}|{path:?}|{c:?}\n", q.id));
        }
    }
    assert_eq!(
        hex::encode(digest.finalize()),
        "b3b51738f39f552a7beec819c2de6c9d5f9b1ce2d6f3552f4048f3984d254fd6",
        "k2 or mk2 contexts changed"
    );
}

/// K3 and Mk3 with the v3 options build byte-identical contexts to the
/// release before the v4 options, so v3 runs stay reproducible.
#[tokio::test]
async fn fact_arms_build_the_same_contexts_as_before_v4_options() {
    use keep_bench::ablation::{FactOptions, KeepPath, Representation};
    use keep_bench::memory_arm::MemoryArm;
    use sha2::{Digest, Sha256};

    let lm = common::load(FACTS);
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let mut k = fact_arms(&lm, lm.manifest.retrieval.clone()).await;
    let mut memory = MemoryArm::default();
    let mut digest = Sha256::new();
    for q in &data.questions {
        let h = data.history(&q.history_id).unwrap();
        for (path, opts) in [
            (KeepPath::K3, FactOptions::default()),
            (
                KeepPath::K3,
                FactOptions {
                    temporal: true,
                    assistant_route: true,
                    ..Default::default()
                },
            ),
            (
                KeepPath::Mk3,
                FactOptions {
                    temporal: true,
                    preferences: true,
                    assistant_route: true,
                    ..Default::default()
                },
            ),
        ] {
            let mem = (path == KeepPath::Mk3).then_some(&mut memory);
            let c = k
                .build_ranked(path, Representation::Structured, mem, h, &q.query, opts)
                .await;
            // Keep the original snapshot's option layout, independently
            // of new opt-in fields added to FactOptions.
            let legacy = format!(
                "FactOptions {{ temporal: {}, preferences: {}, assistant_route: {}, subjects: {}, summaries: {}, expand: {} }}",
                opts.temporal,
                opts.preferences,
                opts.assistant_route,
                opts.subjects,
                opts.summaries,
                opts.expand
            );
            digest.update(format!("{}|{path:?}|{legacy}|{c:?}\n", q.id));
        }
    }
    assert_eq!(
        hex::encode(digest.finalize()),
        "909cc485b88e63c61cff3d139addd292577cadca89aab3a10d9230b24572bf72",
        "k3 or mk3 contexts changed"
    );
}

/// The fact manifest over the LoCoMo-shaped fixture, whose conversations
/// are between two named people, with session summaries enabled.
fn locomo_facts_manifest(dir: &std::path::Path) -> keep_bench::manifest::LoadedManifest {
    let facts = common::fixture_manifest(FACTS);
    let locomo = common::fixture_manifest("locomo-mock.toml");
    let sections = |t: &str| {
        let i = t.find("[dataset]").unwrap();
        let j = t.find("[models.answer]").unwrap();
        t[i..j].to_owned()
    };
    let text = facts
        .replacen(&sections(&facts), &sections(&locomo), 1)
        .replacen(
            "expected_output_tokens = 400\n",
            "expected_output_tokens = 400\nsummary_prompt_version = \"session-summary-v1\"\nsummary_tokens = 96\n",
            1,
        )
        + "\n[[arms]]\nname = \"keep-mk4\"\nkind = \"keep\"\npath = \"mk3\"\n\
           representation = \"structured\"\nsubjects = true\nsummaries = true\nexpand = true\n";
    common::write_manifest(dir, &text)
}

#[tokio::test]
async fn v5_source_facts_and_decomposed_context_fit_the_same_budget() {
    use keep_bench::ablation::{FactOptions, KeepPath, Representation};
    use keep_bench::baselines::Built;
    use keep_bench::memory_arm::MemoryArm;
    let dir = tempfile::tempdir().unwrap();
    let mut lm = locomo_facts_manifest(dir.path());
    lm.manifest.extraction.as_mut().unwrap().facts_tokens = 384;
    lm.manifest.retrieval.evidence_tokens = 512;
    let r = lm.manifest.retrieval.clone();
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let mut k = fact_arms(&lm, r.clone()).await;
    let mut memory = MemoryArm::default();
    let q = data
        .questions
        .iter()
        .find(|q| q.query.text == "What pet did Tomas adopt?")
        .unwrap();
    let h = data.history(&q.history_id).unwrap();
    let mut query = q.query.clone();
    query.text = "What pet did Tomas adopt and what instrument does he play?".into();
    let opts = FactOptions {
        subjects: true,
        provenance: true,
        relative_dates: true,
        decompose: true,
        fact_selection: true,
        ..Default::default()
    };
    for path in [KeepPath::K3, KeepPath::Mk3] {
        let Built::Ready(c) = k
            .build_ranked(
                path,
                Representation::Structured,
                Some(&mut memory),
                h,
                &query,
                opts,
            )
            .await
        else {
            panic!("v5 context failed")
        };
        assert!(c.tokens <= c.budget_tokens.unwrap());
        assert!(c.text.contains("Source ["), "{}", c.text);
        assert!(c.method.as_ref().unwrap().contains("decompose=facets-v1"));
        assert!(c.text.contains("Tomas"));
    }
}

/// Mk3 with the v4 options names the person the question asks about, keeps
/// facts about the other participant apart, adds session summaries and an
/// expansion list, and still fits Mk3's budget.
#[tokio::test]
async fn v4_options_ground_the_question_and_add_summaries_within_budget() {
    use keep_bench::ablation::{FactOptions, KeepPath, Representation};
    use keep_bench::baselines::Built;
    use keep_bench::memory_arm::MemoryArm;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let lm = locomo_facts_manifest(dir.path());
    let m = &lm.manifest;
    let r = m.retrieval.clone();
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let mut k = fact_arms(&lm, r.clone()).await;
    let summarizer = Arc::new(keep_bench::facts::summarizer(
        m.extract_model().unwrap(),
        Some(Arc::new(keep_bench::mocks::extractor())),
    ));
    let sessions = data
        .histories
        .iter()
        .flat_map(|h| h.sessions.iter().map(keep_bench::facts::session_input))
        .collect();
    let stats = keep_bench::facts::warm_cache(summarizer.clone(), sessions, 4, 1).await;
    assert_eq!((stats.failed, stats.missing), (0, 0), "{stats:?}");
    k.enable_summaries(summarizer, 96);
    let mut memory = MemoryArm::default();
    let q = data
        .questions
        .iter()
        .find(|q| q.query.text == "What pet did Tomas adopt?")
        .unwrap();
    let h = data.history(&q.history_id).unwrap();
    let opts = FactOptions {
        temporal: true,
        preferences: true,
        assistant_route: true,
        subjects: true,
        summaries: true,
        expand: true,
        ..Default::default()
    };
    let Built::Ready(c) = k
        .build_ranked(
            KeepPath::Mk3,
            Representation::Structured,
            Some(&mut memory),
            h,
            &q.query,
            opts,
        )
        .await
    else {
        panic!("mk4 {}", q.id)
    };
    assert!(
        c.tokens <= r.standing_tokens + r.evidence_tokens,
        "{}",
        c.tokens
    );
    assert!(
        c.text.contains(
            "The question asks about Tomas. Facts and excerpts about Mira are not about Tomas."
        ),
        "{}",
        c.text
    );
    assert!(
        c.text.contains(keep_bench::facts::SUMMARIES_HEADER),
        "{}",
        c.text
    );
    let names: Vec<&str> = c.blocks.iter().map(|b| b.name.as_str()).collect();
    assert!(names.contains(&"keep-summaries"), "{names:?}");
    let method = c.method.as_deref().unwrap();
    for part in [
        " subjects(asked=Tomas;others=Mira)",
        " expand(seeds=5,added=",
        ";summaries=summarize(",
    ] {
        assert!(method.contains(part), "{part}: {method}");
    }

    // A question naming both people has no grounding.
    let mut both = q.query.clone();
    both.text = "Which instrument does Tomas play, and where did Mira move?".into();
    let Built::Ready(c) = k
        .build_ranked(
            KeepPath::Mk3,
            Representation::Structured,
            Some(&mut memory),
            h,
            &both,
            opts,
        )
        .await
    else {
        panic!("mk4 both")
    };
    assert!(!c.text.contains("The question asks about"), "{}", c.text);
    assert!(c.method.as_deref().unwrap().contains(" subjects(none)"));
}

/// Mk3 with the v8 options ranks excerpts by what the question asks about
/// the named participant, leaves out facts the excerpts already show word
/// for word, and fits the same budget as without them.
#[tokio::test]
async fn v8_options_focus_the_excerpts_and_drop_facts_they_already_show() {
    use keep_bench::ablation::{FactOptions, KeepPath, Representation};
    use keep_bench::baselines::Built;
    use keep_bench::memory_arm::MemoryArm;

    let dir = tempfile::tempdir().unwrap();
    let lm = locomo_facts_manifest(dir.path());
    let r = lm.manifest.retrieval.clone();
    let budget = r.standing_tokens + r.evidence_tokens;
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let mut k = fact_arms(&lm, r.clone()).await;
    let mut memory = MemoryArm::default();
    let base = FactOptions {
        temporal: true,
        assistant_route: true,
        subjects: true,
        ..Default::default()
    };
    let v8 = FactOptions {
        focus: true,
        verbatim_first: true,
        ..base
    };
    let mut pruned = 0;
    for q in &data.questions {
        let h = data.history(&q.history_id).unwrap();
        let mut build = async |opts| match k
            .build_ranked(
                KeepPath::Mk3,
                Representation::Structured,
                Some(&mut memory),
                h,
                &q.query,
                opts,
            )
            .await
        {
            Built::Ready(c) => c,
            other => panic!("{}: {other:?}", q.id),
        };
        let before = build(base).await;
        let after = build(v8).await;
        assert!(after.tokens <= budget, "{}: {}", q.id, after.tokens);
        let method = after.method.as_deref().unwrap();
        let named = ["Mira", "Tomas", "Ada", "Kofi", "Lena", "Ravi"]
            .iter()
            .chain(&["Nora", "Ilya", "Sana", "Joel", "Pia", "Bruno"])
            .filter(|n| q.query.text.contains(*n))
            .count();
        // A named participant is searched without the name; when the
        // question asks about exactly one, their own turns are fused in.
        match named {
            0 => assert!(method.contains(";focus=v8(query=as-asked,own=0)")),
            1 => {
                assert!(method.contains(";focus=v8(query=unnamed,own="));
                assert!(!method.contains(",own=0)"), "{}: {method}", q.id);
            }
            _ => assert!(method.contains(";focus=v8(query=unnamed,own=0)")),
        }
        let facts = |c: &keep_bench::context::ContextBundle| {
            c.included_ids
                .iter()
                .filter(|i| i.starts_with("fact_"))
                .count()
        };
        assert!(facts(&after) <= facts(&before), "{}", q.id);
        if let Some(n) = method
            .split("verbatim_first(pruned=")
            .nth(1)
            .and_then(|s| s.split(')').next())
        {
            pruned += n.parse::<usize>().unwrap();
        }
        // The v7 options build the same context as before the v8 options.
        assert!(!before.method.as_deref().unwrap().contains("focus="));
        assert!(!before.method.as_deref().unwrap().contains("verbatim_first"));
    }
    assert!(pruned > 0, "no fact was left out for an excerpt");
}

/// A mock run answers with K3 and Mk3, records the extraction, and the dry
/// run prices one extraction call per distinct uncached session.
#[tokio::test]
async fn mock_run_answers_with_fact_arms() {
    let dir = tempfile::tempdir().unwrap();
    let lm = common::load(FACTS);
    let out = run(&lm, &opts(Mode::Execute, Some(dir.path())))
        .await
        .unwrap();
    let x = out.meta.extraction.as_ref().unwrap();
    assert!(x.sessions > 0 && x.failed == 0 && x.missing == 0, "{x:?}");
    assert_eq!(x.cached + x.called, x.sessions);
    for (arm, prompt) in [
        ("keep-k3", "answer-v1"),
        ("keep-mk3", "answer-v1"),
        ("keep-mk3-full", "answer-v3"),
        ("keep-mk2", "answer-v1"),
    ] {
        let records: Vec<_> = out.records.iter().filter(|r| r.arm == arm).collect();
        assert_eq!(records.len(), 24, "{arm}");
        assert!(
            records
                .iter()
                .all(|r| matches!(r.outcome, Outcome::Answered)),
            "{arm}"
        );
        assert!(
            records
                .iter()
                .all(|r| r.prompt.as_ref().unwrap().version == prompt),
            "{arm}"
        );
    }
    let r = dryrun::dry_run(&lm).await.unwrap();
    let e = r.extraction.unwrap();
    assert_eq!(e.sessions, x.sessions);
    assert_eq!(e.cached + e.calls as usize, e.sessions);
    assert_eq!(e.output_tokens_max, e.calls * 1024);
}

/// Every file under `dir`, by relative path, with its bytes.
fn cache_files(dir: &std::path::Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![dir.to_owned()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                let rel = p.strip_prefix(dir).unwrap().to_string_lossy().into_owned();
                out.insert(rel, std::fs::read(&p).unwrap());
            }
        }
    }
    out
}

/// Fact extraction and summaries fill the same cache, byte for byte,
/// whether sessions are read one at a time or eight at a time.
#[tokio::test]
async fn extraction_cache_does_not_depend_on_concurrency() {
    use keep_knowledge::session_extract::{ExtractionCache, SessionExtractor};
    use keep_knowledge::session_summary::SessionSummarizer;
    use std::sync::Arc;

    let lm = common::load(FACTS);
    let data = keep_bench::prepare::prepare(&lm, false).unwrap().dataset;
    let sessions: Vec<_> = data
        .histories
        .iter()
        .flat_map(|h| h.sessions.iter().map(keep_bench::facts::session_input))
        .collect();
    let mut caches = Vec::new();
    for concurrency in [1, 8] {
        let dir = tempfile::tempdir().unwrap();
        let model: Arc<dyn keep_providers::ChatModel> = Arc::new(keep_bench::mocks::extractor());
        let extractor = Arc::new(SessionExtractor::new(
            Some(model.clone()),
            "mock:extractor@1",
            Some(ExtractionCache::new(dir.path().join("facts"))),
            1024,
        ));
        let facts =
            keep_bench::facts::warm_cache(extractor, sessions.clone(), concurrency, 1).await;
        let summarizer = Arc::new(SessionSummarizer::new(
            Some(model),
            "mock:extractor@1",
            Some(ExtractionCache::new(dir.path().join("summaries"))),
            400,
        ));
        let summaries =
            keep_bench::facts::warm_cache(summarizer, sessions.clone(), concurrency, 1).await;
        for stats in [&facts, &summaries] {
            assert_eq!((stats.failed, stats.missing, stats.cached), (0, 0, 0));
            assert!(stats.called > 1, "{stats:?}");
        }
        caches.push((facts, summaries, cache_files(dir.path())));
    }
    assert!(!caches[0].2.is_empty());
    assert_eq!(caches[0], caches[1]);
}
