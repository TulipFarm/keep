mod common;

use std::collections::BTreeSet;

use common::{fixture_manifest, write_manifest};
use keep_bench::artifacts::{JudgeOutcome, Outcome};
use keep_bench::manifest::StorageProfile;
use keep_bench::runner::{Mode, RunOptions, run};
use keep_bench::{compare, dryrun, report};
use keep_providers::{ApproxTokenizer, Tokenizer};

const LME: &str = "longmemeval-mock.toml";

#[tokio::test]
async fn non_adversarial_summary_keeps_failures_and_only_removes_that_category() {
    let lm = common::load(LME);
    let out = run(&lm, &opts(Mode::Estimate, None, false)).await.unwrap();
    let mut records = out.records[..3].to_vec();
    for r in &mut records {
        r.outcome = Outcome::Answered;
        r.judge = JudgeOutcome::Judged { correct: true };
    }
    records[0].category = "adversarial".into();
    records[1].category = "cat-4".into();
    records[2].category = "cat-1".into();
    records[2].outcome = Outcome::Unavailable {
        reason: "does not fit".into(),
    };
    records[2].judge = JudgeOutcome::Skipped;
    let refs = records.iter().collect::<Vec<_>>();
    let summary = report::summarize_arm("sample", &refs);
    assert_eq!(summary.accuracy_non_adversarial.n, 2);
    assert_eq!(summary.accuracy_non_adversarial.correct, 1);
    assert_eq!(summary.accuracy_non_adversarial.accuracy, Some(0.5));
    assert_eq!(summary.accuracy_scheduled.n, 3);
}

fn opts(mode: Mode, root: Option<&std::path::Path>, allow: bool) -> RunOptions {
    RunOptions {
        mode,
        unseal: false,
        allow_model_calls: allow,
        results_root: root.map(ToOwned::to_owned),
        spend: None,
    }
}

/// The fixture switched to a fully pinned remote endpoint with positive caps:
/// only the two dispatch switches stand between it and the network.
fn remote_manifest(allow_in_manifest: bool) -> String {
    let remote = |model: &str| {
        format!(
            "provider = \"openai-compatible\"\nmodel = \"{model}\"\nrevision = \"r1\"\ntokenizer = \"{model}-tok\"\n\
             base_url_env = \"KEEP_TEST_UNUSED_URL\""
        )
    };
    fixture_manifest(LME)
        .replacen("provider = \"mock\"\nmodel = \"mock-extractive\"\nrevision = \"v1\"\ntokenizer = \"approx-v1\"", &remote("a"), 1)
        .replacen("provider = \"mock\"\nmodel = \"mock-f1-judge\"\nrevision = \"v1\"\ntokenizer = \"approx-v1\"", &remote("j"), 1)
        .replace("max_model_calls = 0", "max_model_calls = 10")
        .replace("max_input_tokens = 0", "max_input_tokens = 10")
        .replace("max_output_tokens = 0\nmax_cost", "max_output_tokens = 10\nmax_cost")
        .replace("max_cost_usd = 0.0", "max_cost_usd = 0.01")
        + &format!("\n[dispatch]\nallow_model_calls = {allow_in_manifest}\n")
}

#[tokio::test]
async fn remote_dispatch_is_refused_unless_manifest_and_flag_both_allow_it() {
    let dir = tempfile::tempdir().unwrap();
    let results = dir.path().join("results");
    for (manifest_allows, flag, expect) in [
        (
            false,
            false,
            vec!["dispatch.allow_model_calls", "--allow-model-calls flag"],
        ),
        (true, false, vec!["--allow-model-calls flag"]),
        (false, true, vec!["dispatch.allow_model_calls"]),
    ] {
        let lm = write_manifest(dir.path(), &remote_manifest(manifest_allows));
        assert!(lm.manifest.uses_remote_models());
        assert!(
            lm.manifest.dispatch_problems().is_empty(),
            "{:?}",
            lm.manifest.dispatch_problems()
        );
        let err = match run(&lm, &opts(Mode::Execute, Some(&results), flag)).await {
            Ok(_) => panic!("dispatch must be refused"),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("model calls are disabled"), "{err}");
        for needle in expect {
            assert!(err.contains(needle), "{needle} missing from {err}");
        }
        // Refusal happens before anything is written.
        assert!(!results.exists());
    }
}

#[tokio::test]
async fn dry_run_estimates_without_dispatch_or_artifacts() {
    let lm = common::load(LME);
    let r = dryrun::dry_run(&lm).await.unwrap();
    assert_eq!(r.questions, 8);
    assert_eq!(r.seeds, 3);
    assert!(!r.remote_models);
    assert!(r.dispatch_blockers.is_empty());
    // Every manifest arm appears, including ones that cannot run yet.
    let arms: Vec<&str> = r.arms.iter().map(|a| a.arm.as_str()).collect();
    assert_eq!(arms.len(), lm.manifest.arms.len());
    let agent = r.arms.iter().find(|a| a.arm == "filesystem-agent").unwrap();
    assert_eq!((agent.not_run, agent.answer_calls), (24, 0));
    let keep = r.arms.iter().find(|a| a.arm == "keep-mk1").unwrap();
    assert_eq!((keep.not_run, keep.answer_calls), (0, 24));
    let full = r.arms.iter().find(|a| a.arm == "full-context").unwrap();
    assert_eq!(
        full.unavailable, 3,
        "the long synthetic history does not fit 4096 tokens"
    );
    assert!(full.answer_input_tokens > 0);
    // Worst-case output assumes every call uses its full output allowance.
    let bm25 = r.arms.iter().find(|a| a.arm == "bm25").unwrap();
    assert_eq!(bm25.answer_output_tokens_max, 24 * 64);
    assert!((r.full_context_fit_fraction - 11.0 / 12.0).abs() < 1e-9);

    // Estimate mode is the same code path as `run`, with no providers.
    let out = run(&lm, &opts(Mode::Estimate, None, false)).await.unwrap();
    assert!(out.dir.is_none());
    assert!(
        out.records
            .iter()
            .all(|r| r.answer.is_none() && r.judge_call.is_none())
    );
    assert!(
        out.records
            .iter()
            .any(|r| matches!(r.outcome, Outcome::Estimated))
    );
}

#[tokio::test]
async fn mock_run_writes_complete_artifacts_and_accounts_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let lm = common::load(LME);
    let out = run(&lm, &opts(Mode::Execute, Some(dir.path()), false))
        .await
        .unwrap();
    let run_dir = out.dir.clone().unwrap();
    for f in [
        "manifest.toml",
        "manifest.json",
        "run.json",
        "records.jsonl",
        "summary.json",
        "summary.md",
    ] {
        assert!(run_dir.join(f).is_file(), "{f} missing");
    }
    assert_eq!(
        std::fs::read_to_string(run_dir.join("manifest.toml")).unwrap(),
        lm.raw
    );
    assert!(out.meta.mock);
    // 9 arms x 8 questions x 3 seeds: one record per cell, none dropped.
    assert_eq!(out.records.len(), 9 * 8 * 3);

    let tok = ApproxTokenizer;
    for r in &out.records {
        assert_eq!(r.manifest_hash, lm.hash);
        match &r.outcome {
            Outcome::Answered => {
                let p = r.prompt.as_ref().unwrap();
                let counted: usize = p
                    .messages
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|m| tok.count(&m.content))
                    .sum();
                assert_eq!(p.input_tokens, counted, "{} {}", r.arm, r.question_id);
                let a = r.answer.as_ref().unwrap();
                assert_eq!(a.usage.input_tokens as usize, counted);
                assert!(matches!(r.judge, JudgeOutcome::Judged { .. }));
                let c = r.context.as_ref().unwrap();
                if let Some(budget) = c.budget_tokens {
                    assert!(c.tokens <= budget, "{} over budget", r.arm);
                }
                assert!(r.metrics.is_some());
            }
            Outcome::Unavailable { reason } => {
                assert!(r.answer.is_none());
                assert!(!reason.is_empty());
            }
            Outcome::NotRun { .. } => assert_eq!(r.arm, "filesystem-agent"),
            other => panic!("unexpected outcome {other:?}"),
        }
    }

    // Retrieval arms report when ranked evidence had to be dropped.
    let truncated = out
        .records
        .iter()
        .filter(|r| r.arm == "recent-window")
        .filter_map(|r| r.context.as_ref()?.truncation.as_ref())
        .count();
    assert!(truncated > 0);
    // Full context is never truncated: it is either complete or unavailable.
    assert!(
        out.records
            .iter()
            .filter(|r| r.arm == "full-context")
            .all(|r| r.context.as_ref().is_none_or(|c| c.truncation.is_none()))
    );

    let summary = report::summarize(&out.meta, &lm.manifest, &out.records);
    let names: Vec<&str> = summary.arms.iter().map(|a| a.arm.as_str()).collect();
    assert!(names.contains(&"full-context"));
    let full = summary
        .arms
        .iter()
        .find(|a| a.arm == "full-context")
        .unwrap();
    assert_eq!(full.status, "partially-unavailable");
    let agent = summary
        .arms
        .iter()
        .find(|a| a.arm == "filesystem-agent")
        .unwrap();
    assert_eq!(
        (agent.status.as_str(), agent.accuracy_scheduled.accuracy),
        ("not-run", None)
    );
    let md = report::markdown(&summary);
    assert!(md.contains("Mock providers"));
    assert!(md.contains(&lm.hash.to_string()));

    // The directory reloads, and tampering with the stored manifest is caught.
    let loaded = compare::load_run(&run_dir).unwrap();
    assert_eq!(loaded.records.len(), out.records.len());
    let mj = run_dir.join("manifest.json");
    let edited = std::fs::read_to_string(&mj)
        .unwrap()
        .replace("\"seeds\": [\n    1,", "\"seeds\": [\n    9,");
    std::fs::write(&mj, edited).unwrap();
    assert!(compare::load_run(&run_dir).is_err());
}

#[tokio::test]
async fn runs_are_deterministic_for_mocks() {
    let lm = common::load(LME);
    let a = run(&lm, &opts(Mode::Execute, None, false)).await.unwrap();
    let b = run(&lm, &opts(Mode::Execute, None, false)).await.unwrap();
    let key = |r: &keep_bench::artifacts::QuestionRecord| {
        (r.arm.clone(), r.question_id.clone(), r.seed, r.correct())
    };
    assert_eq!(
        a.records.iter().map(key).collect::<Vec<_>>(),
        b.records.iter().map(key).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn mock_runs_never_pass_the_gate() {
    let dir = tempfile::tempdir().unwrap();
    let lm = common::load("locomo-mock.toml");
    let out = run(&lm, &opts(Mode::Execute, Some(dir.path()), false))
        .await
        .unwrap();
    let loaded = vec![compare::load_run(&out.dir.unwrap()).unwrap()];
    let g = compare::evaluate(
        &loaded,
        "notes-grep",
        &["full-context".into(), "bm25".into(), "dense".into()],
        &[
            StorageProfile::PostgresVector,
            StorageProfile::PostgresQdrant,
        ],
    )
    .unwrap();
    assert!(!g.passed);
    let failed: BTreeSet<&str> = g
        .checks
        .iter()
        .filter(|c| !c.passed)
        .map(|c| c.name.as_str())
        .collect();
    for name in [
        "real providers",
        "confirmation phase",
        "full-context fit",
        "storage profile postgres-vector",
    ] {
        assert!(failed.contains(name), "{name} should fail: {failed:?}");
    }
    let p = &g.profiles[0];
    assert_eq!(p.comparisons.len(), 3);
    // Each comparison pairs the same questions across arms.
    for c in &p.comparisons {
        let b = &c.bootstrap;
        assert_eq!((b.questions, b.clusters), (26, 5));
        assert!(b.ci_low <= b.delta && b.delta <= b.ci_high);
    }
}

#[tokio::test]
async fn gate_adjusts_every_profile_and_comparator_as_one_holm_family() {
    let dir = tempfile::tempdir().unwrap();
    let lm = common::load("locomo-mock.toml");
    let out = run(&lm, &opts(Mode::Execute, Some(dir.path()), false))
        .await
        .unwrap();
    let run_dir = out.dir.unwrap();
    let qdrant = compare::load_run(&run_dir).unwrap();
    let mut vector = compare::load_run(&run_dir).unwrap();
    vector.meta.storage_profile = StorageProfile::PostgresVector;
    let profiles = [
        StorageProfile::PostgresVector,
        StorageProfile::PostgresQdrant,
    ];
    let g = compare::evaluate(
        &[qdrant, vector],
        "notes-grep",
        &["full-context".into(), "dense".into()],
        &profiles,
    )
    .unwrap();
    // 2 comparators x 2 storage profiles = one family of 4 tests.
    let decisions: Vec<_> = g
        .profiles
        .iter()
        .flat_map(|p| p.comparisons.iter().map(|c| &c.holm))
        .collect();
    assert_eq!(decisions.len(), 4);
    let mut ranks: Vec<usize> = decisions.iter().map(|d| d.rank).collect();
    ranks.sort();
    assert_eq!(ranks, [0, 1, 2, 3]);
    let alpha = 1.0 - g.statistics.confidence;
    let first = decisions.iter().find(|d| d.rank == 0).unwrap();
    assert!((first.level - alpha / 4.0).abs() < 1e-12, "{}", first.level);
    assert!(decisions.iter().any(|d| d.name == "postgres-vector/dense"));
    assert!(compare::markdown(&g).contains("one Holm family of 4 tests"));
}

/// Run the binary on a mock manifest with judge consistency, in `cwd` so
/// the verdict cache lands in `cwd/.keep-cache/judgments`.
fn consistent_run(cwd: &std::path::Path) -> Vec<keep_bench::artifacts::QuestionRecord> {
    let text = fixture_manifest(LME)
        + "\n[judge_consistency]\ncache = true\nvotes = 3\nvote_on = \"all\"\n";
    let lm = write_manifest(cwd, &text);
    let results = cwd.join("results");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_keep-bench"))
        .args([
            "run",
            lm.path.to_str().unwrap(),
            "--arms",
            "full-context,bm25",
        ])
        .arg("--results")
        .arg(&results)
        .current_dir(cwd)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut runs: Vec<_> = std::fs::read_dir(&results)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    runs.sort();
    keep_bench::artifacts::read_records(&runs.last().unwrap().join("records.jsonl")).unwrap()
}

#[test]
fn judge_consistency_reuses_verdicts_for_identical_answers_across_runs() {
    use keep_bench::judge::VerdictSource;
    let dir = tempfile::tempdir().unwrap();
    let first = consistent_run(dir.path());
    let answered: Vec<_> = first
        .iter()
        .filter(|r| r.outcome == Outcome::Answered)
        .collect();
    assert!(!answered.is_empty());
    let mut verdicts = std::collections::HashMap::new();
    let mut judged = 0;
    for r in &answered {
        let jc = r.judge_consistency.as_ref().unwrap();
        // One verdict per key, whichever record judged it.
        assert_eq!(
            *verdicts.entry(jc.key.clone()).or_insert(r.correct()),
            r.correct()
        );
        match jc.source {
            VerdictSource::Judged => {
                judged += 1;
                assert!(r.judge_call.is_some());
                // The deterministic mock agrees with itself: two votes settle it.
                assert_eq!(jc.votes.len(), 2);
                assert_eq!(jc.extra_calls.len(), 1);
            }
            VerdictSource::Cache => {
                assert!(r.judge_call.is_none() && jc.extra_calls.is_empty());
            }
        }
    }
    // Three seeds give the mock's identical answers three times: only the
    // first is judged.
    assert!(judged < answered.len(), "{judged} of {}", answered.len());
    assert_eq!(judged, verdicts.len());

    // A second run judges nothing and scores every answer the same way.
    let second = consistent_run(dir.path());
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(
            (&a.question_id, a.seed, a.correct()),
            (&b.question_id, b.seed, b.correct())
        );
        if let Some(jc) = &b.judge_consistency {
            assert_eq!(jc.source, VerdictSource::Cache);
            assert!(b.judge_call.is_none());
        }
    }
}
