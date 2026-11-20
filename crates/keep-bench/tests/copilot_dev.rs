//! Development-track runs through the Copilot CLI provider, driven by a fake
//! CLI script so nothing leaves the machine.
#![cfg(unix)]

mod common;

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::Command;

use common::{fixture_manifest, set, write_manifest};
use keep_bench::artifacts::{JudgeOutcome, Outcome};
use keep_bench::manifest::LoadedManifest;
use keep_bench::manifest::{DEVELOPMENT_TRACK_BANNER, StorageProfile};
use keep_bench::runner::{Mode, RunOptions, run};
use keep_bench::{compare, dryrun, report};
use keep_providers::copilot::NON_CANONICAL_BANNER;

const LME: &str = "longmemeval-mock.toml";

fn opts(root: &Path) -> RunOptions {
    RunOptions {
        mode: Mode::Execute,
        unseal: false,
        allow_model_calls: false,
        results_root: Some(root.to_owned()),
        spend: None,
    }
}

/// Every call replies `CORRECT` and reports 0.1 AI credits and 100 hidden
/// system-prompt tokens.
fn fake_cli(dir: &Path) -> std::path::PathBuf {
    let line = |s: &str| format!("printf '%s\\n' '{s}'\n");
    let script = format!(
        "#!/bin/sh\ncat > /dev/null\necho call >> \"{}/calls.txt\"\n",
        dir.display()
    ) + &line(
        r#"{"type":"assistant.message","data":{"model":"fake-model","content":"CORRECT"}}"#,
    ) + &line(
        r#"{"type":"session.usage_checkpoint","data":{"totalNanoAiu":100000000,"promptCacheBreakState":[{"models":{"fake-model":{"tool_tokens":0,"system_segments":[{"tokens":100}]}}}],"accountingSnapshot":{"modelMetrics":{"fake-model":{"usage":{"inputTokens":400,"outputTokens":2}}}}}}"#,
    ) + &line(r#"{"type":"result","exitCode":0}"#);
    let path = dir.join("copilot");
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn copilot_model(model: &str) -> String {
    format!(
        "provider = \"copilot-cli\"\nmodel = \"{model}\"\nrevision = \"unpinned-upstream\"\n\
         tokenizer = \"tiktoken-o200k-base\"\ncontext_limit = 4096\nmax_output_tokens = 8\n\
         prompt_version = \"PV\"\n\n[models.ROLE.copilot]\ncredits_per_1k_input = 0.001\n\
         credits_per_1k_output = 0.001\noverhead_tokens = 0\n"
    )
}

fn copilot_manifest(cap: f64) -> String {
    let text = fixture_manifest(LME)
        .replacen(
            "provider = \"mock\"\nmodel = \"mock-extractive\"\nrevision = \"v1\"\ntokenizer = \"approx-v1\"\ncontext_limit = 4096\nmax_output_tokens = 64\nprompt_version = \"answer-v1\"\n",
            &copilot_model("fake-answer")
                .replace("ROLE", "answer")
                .replace("PV", "answer-v1"),
            1,
        )
        .replacen(
            "provider = \"mock\"\nmodel = \"mock-f1-judge\"\nrevision = \"v1\"\ntokenizer = \"approx-v1\"\ncontext_limit = 4096\nmax_output_tokens = 8\nprompt_version = \"judge-v1\"\n",
            &copilot_model("fake-judge")
                .replace("ROLE", "judge")
                .replace("PV", "judge-v1"),
            1,
        )
        .replace("max_attempts = 1", &format!("max_attempts = 1\nmax_ai_credits = {cap:?}"));
    let text = set(&text, "seeds", "[1]");
    let text = text.replacen(
        "phase = \"development\"",
        "phase = \"development\"\ntrack = \"development\"",
        1,
    );
    text + "\n[dispatch]\nallow_model_calls = true\nconcurrency = 2\n"
}

fn development_mock_manifest() -> String {
    fixture_manifest(LME).replacen(
        "phase = \"development\"",
        "phase = \"development\"\ntrack = \"development\"",
        1,
    )
}

#[tokio::test]
async fn copilot_runs_respect_the_credit_cap_and_are_stamped_non_canonical() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(dir.path());
    let lm = write_manifest(dir.path(), &copilot_manifest(2.0));
    assert!(
        lm.manifest.dispatch_problems().is_empty(),
        "{:?}",
        lm.manifest.dispatch_problems()
    );

    // The dry run projects credits and carries the banner.
    let dry = dryrun::dry_run(&lm).await.unwrap();
    assert_eq!(dry.non_canonical.as_deref(), Some(NON_CANONICAL_BANNER));
    assert!(dry.caps.iter().any(|c| c.cap.contains("AI credits")));
    assert!(dryrun::markdown(&dry).starts_with(&format!("> **{NON_CANONICAL_BANNER}**")));

    // The binary runs with the fake CLI; without the operator flag it never
    // spawns it.
    let results = dir.path().join("results");
    let keep_bench = |flag: bool| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_keep-bench"));
        c.arg("run").arg(dir.path().join("manifest.toml"));
        c.arg("--results").arg(&results);
        if flag {
            c.arg("--allow-model-calls");
        }
        c.env("KEEP_COPILOT_BIN", &cli)
            .env("KEEP_COPILOT_WORKDIR", dir.path())
            .current_dir(dir.path())
            .output()
            .unwrap()
    };
    let refused = keep_bench(false);
    assert!(!refused.status.success());
    assert!(!dir.path().join("calls.txt").exists());
    assert!(!results.exists());

    let done = keep_bench(true);
    assert!(
        done.status.success(),
        "{}",
        String::from_utf8_lossy(&done.stderr)
    );
    let run_dir = std::fs::read_dir(&results)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let out = compare::load_run(&run_dir).unwrap();
    assert_eq!(
        out.meta.non_canonical.as_deref(),
        Some(NON_CANONICAL_BANNER)
    );
    assert!(
        out.records
            .iter()
            .all(|r| r.non_canonical.as_deref() == Some(NON_CANONICAL_BANNER))
    );
    let raw = std::fs::read_to_string(run_dir.join("records.jsonl")).unwrap();
    assert!(raw.lines().all(|l| l.contains(NON_CANONICAL_BANNER)));

    // 0.1 credits per call against a cap of 2.0: at most 20 calls are
    // charged, the rest are refused and reported as such, never as answers.
    let usage = out.meta.credit_usage.unwrap();
    assert!(usage.charged() <= 2.0 + 1e-9, "{usage:?}");
    assert!(usage.calls <= 20 && usage.calls >= 19, "{usage:?}");
    assert!(usage.refused > 0, "{usage:?}");
    let refused_answers = out
        .records
        .iter()
        .filter(|r| matches!(&r.outcome, Outcome::Failed { error } if error.starts_with("budget exhausted")))
        .count();
    let refused_judges = out
        .records
        .iter()
        .filter(|r| matches!(&r.judge, JudgeOutcome::Failed { error } if error.starts_with("budget exhausted")))
        .count();
    assert!(refused_answers > 0);
    assert_eq!(refused_answers + refused_judges, usage.refused as usize);
    let answered: Vec<_> = out
        .records
        .iter()
        .filter(|r| r.outcome == Outcome::Answered)
        .collect();
    assert!(!answered.is_empty());
    for r in &answered {
        let b = r.answer.as_ref().unwrap().billing.as_ref().unwrap();
        assert!((b.ai_credits - 0.1).abs() < 1e-9);
        assert_eq!(b.harness_overhead_tokens, 100);
    }

    // The summary keeps Keep's prompt tokens and the hidden overhead apart.
    let summary = report::summarize(&out.meta, &lm.manifest, &out.records);
    assert_eq!(summary.non_canonical.as_deref(), Some(NON_CANONICAL_BANNER));
    let total: f64 = summary.arms.iter().map(|a| a.ai_credits).sum();
    assert!(
        (total - usage.reported).abs() < 1e-6,
        "{total} vs {usage:?}"
    );
    assert_eq!(
        summary.arms.iter().map(|a| a.budget_refused).sum::<usize>(),
        usage.refused as usize
    );
    let md = report::markdown(&summary);
    assert!(md.contains(NON_CANONICAL_BANNER));
    assert!(md.contains("Hidden overhead per answer call"));

    assert_eq!(
        std::fs::read_to_string(run_dir.join("summary.md")).unwrap(),
        md
    );

    // Comparison and the gate refuse the run outright.
    let loaded = vec![out];
    let err = compare::evaluate(
        &loaded,
        "bm25",
        &["full-context".into()],
        &[StorageProfile::InMemory],
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("non-canonical"), "{err}");
    assert!(err.contains("--allow-non-canonical"), "{err}");

    // With the internal opt-in the binary compares the run, stamps stdout and
    // both written reports, and never reports a gate pass.
    let report_dir = dir.path().join("comparison");
    let compared = Command::new(env!("CARGO_BIN_EXE_keep-bench"))
        .arg("compare")
        .arg("--run")
        .arg(&run_dir)
        .args(["--candidate", "bm25", "--against", "full-context"])
        .arg("--allow-non-canonical")
        .arg("--out")
        .arg(&report_dir)
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert_eq!(
        compared.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&compared.stderr)
    );
    let stdout = String::from_utf8_lossy(&compared.stdout);
    let written = std::fs::read_to_string(report_dir.join("comparison.md")).unwrap();
    assert_eq!(stdout, written);
    for text in [&stdout.to_string(), &written] {
        assert!(text.starts_with(&format!("> **{}**", compare::INTERNAL_NOTICE)));
        assert!(text.contains(NON_CANONICAL_BANNER));
        assert!(text.contains("# INTERNAL / NON-CANONICAL Comparison"));
        assert!(text.contains("**Result: NOT PASSED**"));
    }
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(report_dir.join("comparison.json")).unwrap())
            .unwrap();
    let stamp = &json["internal_non_canonical"];
    assert_eq!(stamp["notice"], compare::INTERNAL_NOTICE);
    assert_eq!(stamp["labels"][0], NON_CANONICAL_BANNER);
    assert!(
        stamp["answer_model"]
            .as_str()
            .unwrap()
            .contains("fake-answer")
    );
    assert_eq!(json["passed"], false);
}

#[tokio::test]
async fn development_track_runs_without_copilot_are_still_refused_by_compare() {
    let dir = tempfile::tempdir().unwrap();
    let lm = write_manifest(dir.path(), &development_mock_manifest());
    let out = run(&lm, &opts(dir.path())).await.unwrap();
    assert!(out.records.iter().all(|r| r.non_canonical.is_some()));
    let loaded = vec![compare::load_run(out.dir.as_ref().unwrap()).unwrap()];
    let err = compare::evaluate(
        &loaded,
        "bm25",
        &["full-context".into()],
        &[StorageProfile::InMemory],
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("non-canonical"), "{err}");
}

#[test]
fn copilot_requires_the_development_track_and_a_credit_cap() {
    let dir = tempfile::tempdir().unwrap();
    let canonical = copilot_manifest(5.0).replace("track = \"development\"\n", "");
    let path = dir.path().join("manifest.toml");
    std::fs::write(
        &path,
        canonical.replace("\"../", &format!("\"{}/", common::fixtures().display())),
    )
    .unwrap();
    let lm = LoadedManifest::load(&path).unwrap();
    let errors: Vec<String> = keep_bench::prepare::check(&lm)
        .into_iter()
        .filter(|i| i.error)
        .map(|i| format!("{}: {}", i.field, i.message))
        .collect();
    assert!(
        errors.iter().any(|e| e.contains("development")),
        "{errors:?}"
    );
    assert!(
        lm.manifest
            .dispatch_problems()
            .iter()
            .any(|p| p.contains("track = \"development\""))
    );

    let uncapped = copilot_manifest(5.0).replace("max_ai_credits = 5.0\n", "");
    let lm = write_manifest(dir.path(), &uncapped);
    assert!(
        lm.manifest
            .dispatch_problems()
            .iter()
            .any(|p| p.contains("max_ai_credits")),
        "{:?}",
        lm.manifest.dispatch_problems()
    );
}

async fn mock_run(dir: &Path, text: &str) -> compare::LoadedRun {
    std::fs::create_dir_all(dir).unwrap();
    let lm = write_manifest(dir, text);
    let out = run(&lm, &opts(dir)).await.unwrap();
    compare::load_run(out.dir.as_ref().unwrap()).unwrap()
}

fn internal(runs: &[compare::LoadedRun]) -> anyhow::Result<compare::GateReport> {
    compare::evaluate_with(
        runs,
        "bm25",
        &["full-context".into()],
        &[StorageProfile::InMemory],
        compare::CompareOptions {
            allow_non_canonical: true,
        },
    )
}

#[tokio::test]
async fn internal_comparisons_take_non_canonical_runs_only_and_are_stamped() {
    let dir = tempfile::tempdir().unwrap();
    let dev = mock_run(&dir.path().join("dev"), &development_mock_manifest()).await;
    let canonical = mock_run(&dir.path().join("canonical"), &fixture_manifest(LME)).await;

    let g = internal(std::slice::from_ref(&dev)).unwrap();
    let stamp = g.internal_non_canonical.as_ref().unwrap();
    assert_eq!(stamp.labels, [DEVELOPMENT_TRACK_BANNER]);
    assert!(!g.passed);
    assert!(
        g.checks
            .iter()
            .any(|c| c.name == "canonical runs" && !c.passed)
    );
    let md = compare::markdown(&g);
    assert!(md.contains(DEVELOPMENT_TRACK_BANNER));
    assert!(
        md.trim_end()
            .ends_with(&format!("> **{}**", compare::INTERNAL_NOTICE))
    );

    // Canonical numbers are never mixed into, or relabelled as, an internal
    // comparison; without the opt-in nothing changes for canonical runs.
    let canonical_dir = canonical.dir.clone();
    let mixed = internal(&[dev, canonical]).unwrap_err().to_string();
    assert!(mixed.contains("canonical run"), "{mixed}");
    let canonical = vec![compare::load_run(&canonical_dir).unwrap()];
    assert!(internal(&canonical).is_err());
    let g = compare::evaluate(
        &canonical,
        "bm25",
        &["full-context".into()],
        &[StorageProfile::InMemory],
    )
    .unwrap();
    assert!(g.internal_non_canonical.is_none());
    assert!(!compare::markdown(&g).contains("NON-CANONICAL"));
}

#[tokio::test]
async fn runs_with_different_models_are_never_compared() {
    let dir = tempfile::tempdir().unwrap();
    let a = mock_run(&dir.path().join("a"), &development_mock_manifest()).await;
    let mut b = mock_run(&dir.path().join("b"), &development_mock_manifest()).await;
    b.manifest.models.judge.model = "another-judge".into();
    let err = internal(&[a, b]).unwrap_err().to_string();
    assert!(err.contains("different judge model"), "{err}");

    let a = mock_run(&dir.path().join("c"), &fixture_manifest(LME)).await;
    let mut b = mock_run(&dir.path().join("d"), &fixture_manifest(LME)).await;
    b.manifest.models.answer.prompt_version = "answer-v2".into();
    let err = compare::evaluate(
        &[a, b],
        "bm25",
        &["full-context".into()],
        &[StorageProfile::InMemory],
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("different answer model"), "{err}");
}

#[tokio::test]
async fn contexts_runs_score_evidence_recall_without_calling_any_model() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(dir.path());
    // The manifest authorises Copilot calls; the contexts command still makes none.
    write_manifest(dir.path(), &copilot_manifest(2.0));
    let out = dir.path().join("contexts");
    let done = Command::new(env!("CARGO_BIN_EXE_keep-bench"))
        .arg("contexts")
        .arg(dir.path().join("manifest.toml"))
        .args(["--arms", "bm25,keep-mk1"])
        .arg("--out")
        .arg(&out)
        .env("KEEP_COPILOT_BIN", &cli)
        .env("KEEP_COPILOT_WORKDIR", dir.path())
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        done.status.success(),
        "{}",
        String::from_utf8_lossy(&done.stderr)
    );
    assert!(!dir.path().join("calls.txt").exists(), "a model was called");
    let stdout = String::from_utf8_lossy(&done.stdout);
    assert!(stdout.contains("Contexts only"), "{stdout}");

    let run = std::fs::read_dir(&out)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    assert!(run.to_string_lossy().ends_with("-contexts"));
    let records: Vec<keep_bench::artifacts::QuestionRecord> =
        std::fs::read_to_string(run.join("records.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
    let arms: std::collections::BTreeSet<&str> = records.iter().map(|r| r.arm.as_str()).collect();
    assert_eq!(arms, ["bm25", "keep-mk1"].into());
    assert!(!records.is_empty());
    for r in &records {
        assert_eq!(r.outcome, Outcome::Estimated);
        assert!(r.answer.is_none() && r.judge_call.is_none());
        assert!(r.context.is_some());
    }
    assert!(records.iter().any(|r| {
        r.metrics
            .as_ref()
            .is_some_and(|m| m.evidence_recall.is_some())
    }));
    // One record per question and arm, not per seed.
    let ids: std::collections::BTreeSet<(&str, &str)> = records
        .iter()
        .map(|r| (r.arm.as_str(), r.question_id.as_str()))
        .collect();
    assert_eq!(ids.len(), records.len());
}
