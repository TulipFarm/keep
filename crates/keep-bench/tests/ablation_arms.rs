//! The full Keep ablation (M, K0, K1, Mk0, Mk1) end to end: dry run, mock
//! run with per-block token accounting, and (opt-in) the two Postgres
//! storage profiles through the binary.

mod common;

use std::collections::HashMap;
use std::process::Command;

use keep_bench::artifacts::{ContextRecord, Outcome, QuestionRecord};
use keep_bench::context::{Slot, slot_tokens};
use keep_bench::runner::{Mode, RunOptions, run};
use keep_bench::{dryrun, report};
use keep_memory::ContextBudget;

const AM: &str = "ablation-mock.toml";
const KEEP_ARMS: [&str; 5] = ["keep-m", "keep-k0", "keep-k1", "keep-mk0", "keep-mk1"];

fn opts(mode: Mode, root: Option<&std::path::Path>) -> RunOptions {
    RunOptions {
        mode,
        unseal: false,
        allow_model_calls: false,
        results_root: root.map(ToOwned::to_owned),
        spend: None,
    }
}

fn names(c: &ContextRecord) -> Vec<&str> {
    c.blocks.iter().map(|b| b.name.as_str()).collect()
}

fn block(c: &ContextRecord, name: &str) -> usize {
    c.blocks.iter().find(|b| b.name == name).unwrap().tokens
}

/// Every Keep record answered, with standing and evidence blocks inside
/// their allowances and the block layout each path promises.
fn check_records(records: &[QuestionRecord], standing: usize, evidence: usize) {
    let toc_slot = ContextBudget::for_total(standing).unwrap().knowledge;
    let mut toc_used = 0;
    for r in records
        .iter()
        .filter(|r| KEEP_ARMS.contains(&r.arm.as_str()))
    {
        assert!(
            matches!(r.outcome, Outcome::Answered),
            "{} {}: {:?}",
            r.arm,
            r.question_id,
            r.outcome
        );
        let c = r.context.as_ref().unwrap();
        let s = slot_tokens(&c.blocks, Slot::Standing).unwrap_or(0);
        let e = slot_tokens(&c.blocks, Slot::Evidence).unwrap_or(0);
        assert!(s <= standing, "{}: standing {s} > {standing}", r.arm);
        let hits = c
            .included_ids
            .iter()
            .filter(|i| !i.starts_with('/'))
            .count();
        assert!(e <= evidence + hits, "{}: evidence {e} > {evidence}", r.arm);
        assert!(c.tokens <= c.budget_tokens.unwrap(), "{}", r.arm);
        for b in &c.blocks {
            assert!(b.tokens <= b.budget_tokens, "{} {}", r.arm, b.name);
        }
        let expected: &[&str] = match r.arm.as_str() {
            "keep-m" => &["memory"],
            "keep-k0" => &["knowledge-evidence"],
            "keep-k1" => &["knowledge-toc", "knowledge-evidence"],
            "keep-mk0" => &["memory", "knowledge-evidence"],
            _ => &["memory", "knowledge-toc", "knowledge-evidence"],
        };
        assert_eq!(names(c), expected, "{}", r.arm);
        if r.arm == "keep-mk1" {
            let toc = block(c, "knowledge-toc");
            assert!(toc <= toc_slot);
            toc_used += usize::from(toc > 0);
            assert!(c.method.as_deref().unwrap().starts_with("keep-mk;"));
        }
    }
    assert!(toc_used > 0, "Mk1 never filled the Knowledge slot");
}

#[tokio::test]
async fn dry_run_builds_every_keep_arm() {
    let lm = common::load(AM);
    let r = dryrun::dry_run(&lm).await.unwrap();
    let limit = lm.manifest.retrieval.standing_tokens + lm.manifest.retrieval.evidence_tokens;
    for name in KEEP_ARMS {
        let a = r.arms.iter().find(|a| a.arm == name).unwrap();
        assert_eq!(
            (a.answer_calls, a.not_run, a.failed, a.unavailable),
            (24, 0, 0, 0),
            "{name}"
        );
        assert!(a.context_tokens_max > 0);
        // Plus at most one separator per evidence hit.
        assert!(a.context_tokens_max <= limit + 20, "{name}");
    }
}

#[tokio::test]
async fn mock_run_combines_memory_and_knowledge_within_budgets() {
    let dir = tempfile::tempdir().unwrap();
    let lm = common::load(AM);
    let rt = &lm.manifest.retrieval;
    let out = run(&lm, &opts(Mode::Execute, Some(dir.path())))
        .await
        .unwrap();
    check_records(&out.records, rt.standing_tokens, rt.evidence_tokens);

    // The combined arms reuse each half unchanged: Mk0's Memory block is
    // M's context and its evidence is K0's, question by question.
    let by = |arm: &str| -> HashMap<(String, u64), &ContextRecord> {
        out.records
            .iter()
            .filter(|r| r.arm == arm)
            .map(|r| ((r.question_id.clone(), r.seed), r.context.as_ref().unwrap()))
            .collect()
    };
    let (m, k0, mk0, mk1) = (by("keep-m"), by("keep-k0"), by("keep-mk0"), by("keep-mk1"));
    assert_eq!(mk0.len(), 24);
    for (key, c) in &mk0 {
        assert_eq!(block(c, "memory"), block(m[key], "memory"));
        assert_eq!(
            block(c, "knowledge-evidence"),
            block(k0[key], "knowledge-evidence")
        );
        // The table of contents takes room from the Memory files, never
        // from the evidence allowance.
        let mk1 = mk1[key];
        assert_eq!(
            block(mk1, "knowledge-evidence"),
            block(c, "knowledge-evidence")
        );
        assert!(
            block(mk1, "memory") + block(mk1, "knowledge-toc") <= rt.standing_tokens,
            "{key:?}"
        );
    }
    let prompt = |arm: &str| -> String {
        out.records
            .iter()
            .find(|r| r.arm == arm)
            .and_then(|r| r.prompt.as_ref()?.messages.clone())
            .unwrap()
            .iter()
            .map(|m| m.content.clone())
            .collect()
    };
    assert!(prompt("keep-mk1").contains("KNOWLEDGE CONTENTS"));
    assert!(!prompt("keep-mk0").contains("KNOWLEDGE CONTENTS"));
    assert!(prompt("keep-mk0").contains("BEGIN MEMORY CONTEXT"));

    let summary = report::summarize(&out.meta, &lm.manifest, &out.records);
    for name in KEEP_ARMS {
        let a = summary.arms.iter().find(|a| a.arm == name).unwrap();
        assert_eq!(a.status, "ok", "{name}");
        assert!(a.accuracy_scheduled.accuracy.is_some(), "{name}");
        let t = a.tokens.as_ref().unwrap();
        assert!(t.standing_max.unwrap_or(0) <= rt.standing_tokens);
    }
    let full = summary
        .arms
        .iter()
        .find(|a| a.arm == "full-context")
        .unwrap();
    assert!(full.tokens.is_none(), "baselines have no Keep blocks");
    assert!(report::markdown(&summary).contains("## Keep context tokens"));
}

#[tokio::test]
async fn a_small_standing_budget_scales_the_knowledge_slot() {
    let dir = tempfile::tempdir().unwrap();
    let text = common::set(&common::fixture_manifest(AM), "standing_tokens", "800");
    let lm = common::write_manifest(dir.path(), &text);
    let out = run(&lm, &opts(Mode::Execute, None)).await.unwrap();
    check_records(&out.records, 800, lm.manifest.retrieval.evidence_tokens);
    let slot = ContextBudget::for_total(800).unwrap().knowledge;
    assert!(
        out.records.iter().filter(|r| r.arm == "keep-mk1").all(|r| r
            .context
            .as_ref()
            .unwrap()
            .blocks[1]
            .budget_tokens
            == slot)
    );
}

/// Runs the manifest through the binary with `profile` against a private
/// Postgres schema (and Qdrant collection), then removes them.
async fn profile_run(profile: &str, admin: &str, qdrant: Option<&str>) {
    let schema = format!(
        "keep_ablation_test_{}_{}",
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
    let text = common::fixture_manifest(AM).replace(
        "[storage]\nprofile = \"in-memory\"",
        &format!(
            "[storage]\nprofile = \"{profile}\"\npostgres_url_env = \"KEEP_ABLATION_BENCH_PG\"\nqdrant_url_env = \"KEEP_ABLATION_BENCH_QDRANT\""
        ),
    );
    let lm = common::write_manifest(dir.path(), &text);
    let results = dir.path().join("results");
    let out = Command::new(env!("CARGO_BIN_EXE_keep-bench"))
        .args(["run", lm.path.to_str().unwrap(), "--results"])
        .arg(&results)
        .env("KEEP_ABLATION_BENCH_PG", &url)
        .env("KEEP_ABLATION_BENCH_QDRANT", qdrant.unwrap_or(""))
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
    let rt = &lm.manifest.retrieval;
    check_records(&loaded.records, rt.standing_tokens, rt.evidence_tokens);
}

#[tokio::test]
async fn postgres_vector_profile_runs_every_keep_arm() {
    let Ok(admin) = std::env::var("KEEP_TEST_POSTGRES_URL") else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL is not set");
        return;
    };
    profile_run("postgres-vector", &admin, None).await;
}

#[tokio::test]
async fn postgres_qdrant_profile_runs_every_keep_arm() {
    let (Ok(admin), Ok(qdrant)) = (
        std::env::var("KEEP_TEST_POSTGRES_URL"),
        std::env::var("KEEP_TEST_QDRANT_URL"),
    ) else {
        eprintln!("skipping: KEEP_TEST_POSTGRES_URL or KEEP_TEST_QDRANT_URL is not set");
        return;
    };
    profile_run("postgres-qdrant", &admin, Some(&qdrant)).await;
}
