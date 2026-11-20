mod common;

#[test]
fn v5_manifests_preserve_old_arms_models_and_development_split() {
    use keep_bench::manifest::LoadedManifest;
    let root = common::fixtures().join("../manifests");
    for (folder, pattern) in [("g0", "-copilot"), ("dev", "-dev")] {
        for dataset in ["locomo", "longmemeval-s"] {
            let old = LoadedManifest::load(
                &root
                    .join(folder)
                    .join(format!("{dataset}{pattern}-v4.toml")),
            )
            .unwrap()
            .manifest;
            let new = LoadedManifest::load(
                &root
                    .join(folder)
                    .join(format!("{dataset}{pattern}-v5.toml")),
            )
            .unwrap()
            .manifest;
            assert!(
                new.validate().iter().all(|i| !i.error),
                "{:?}",
                new.validate()
            );
            for arm in old.arms {
                assert_eq!(new.arms.iter().find(|a| a.name == arm.name), Some(&arm));
            }
            assert_eq!(
                serde_json::to_value(old.models).unwrap(),
                serde_json::to_value(new.models).unwrap()
            );
            assert_eq!(
                serde_json::to_value(old.holdout).unwrap(),
                serde_json::to_value(new.holdout).unwrap()
            );
            assert_eq!(new.phase, old.phase);
            assert_eq!(new.budget.max_ai_credits, old.budget.max_ai_credits);
            assert!(
                new.arms
                    .iter()
                    .any(|a| a.name == "keep-mk5" && a.provenance && a.decompose)
            );
        }
    }
}

#[test]
fn v6_manifests_preserve_old_arms_and_differ_only_in_the_answer_reservation() {
    use keep_bench::judge::VoteOn;
    use keep_bench::manifest::LoadedManifest;
    let root = common::fixtures().join("../manifests");
    for (folder, pattern) in [("g0", "-copilot"), ("dev", "-dev")] {
        for dataset in ["locomo", "longmemeval-s"] {
            let load = |v: &str| {
                LoadedManifest::load(
                    &root
                        .join(folder)
                        .join(format!("{dataset}{pattern}-{v}.toml")),
                )
                .unwrap()
                .manifest
            };
            let (old, new) = (load("v5"), load("v6"));
            assert!(
                new.validate().iter().all(|i| !i.error),
                "{:?}",
                new.validate()
            );
            // G0 keeps every v5 arm; the dev manifests keep the v4/v5 Keep arms.
            for arm in old.arms.iter().filter(|a| {
                folder == "g0"
                    || ["keep-k4", "keep-mk4", "keep-k5", "keep-mk5"].contains(&a.name.as_str())
            }) {
                assert_eq!(new.arms.iter().find(|a| a.name == arm.name), Some(arm));
            }
            // Only the answer's output reservation changes.
            let mut models = serde_json::to_value(&old.models).unwrap();
            models["answer"]["max_output_tokens"] = 1024.into();
            assert_eq!(models, serde_json::to_value(&new.models).unwrap());
            assert_eq!(
                serde_json::to_value(&old.holdout).unwrap(),
                serde_json::to_value(&new.holdout).unwrap()
            );
            assert_eq!(new.phase, old.phase);
            assert_eq!(new.budget.max_ai_credits, old.budget.max_ai_credits);
            let jc = new.judge_consistency.expect("v6 judges consistently");
            // G0 takes one vote so its worst case fits the credit cap.
            let votes = if folder == "g0" { 1 } else { 3 };
            assert!(jc.cache && jc.votes == votes && jc.vote_on == VoteOn::Borderline);
            assert!(old.judge_consistency.is_none());
            for name in ["keep-k6", "keep-mk6", "dense-6k-v6", "full-context-v6"] {
                let arm = new.arms.iter().find(|a| a.name == name).unwrap();
                assert_eq!(arm.answer_prompt.as_deref(), Some("answer-v6"), "{name}");
            }
        }
    }
}

#[test]
fn v7_manifests_keep_v6_and_only_add_answer_v7_arms() {
    use keep_bench::manifest::LoadedManifest;
    let root = common::fixtures().join("../manifests");
    for (folder, pattern) in [("g0", "-copilot"), ("dev", "-dev")] {
        for dataset in ["locomo", "longmemeval-s"] {
            let load = |v: &str| {
                LoadedManifest::load(
                    &root
                        .join(folder)
                        .join(format!("{dataset}{pattern}-{v}.toml")),
                )
                .unwrap()
                .manifest
            };
            let (old, new) = (load("v6"), load("v7"));
            assert!(
                new.validate().iter().all(|i| !i.error),
                "{:?}",
                new.validate()
            );
            // G0 keeps every v6 arm; dev keeps the paired v6 controls.
            for arm in old.arms.iter().filter(|a| {
                folder == "g0" || ["keep-mk6", "full-context-v6"].contains(&a.name.as_str())
            }) {
                assert_eq!(new.arms.iter().find(|a| a.name == arm.name), Some(arm));
            }
            assert_eq!(
                serde_json::to_value(&old.models).unwrap(),
                serde_json::to_value(&new.models).unwrap()
            );
            assert_eq!(
                serde_json::to_value(&old.holdout).unwrap(),
                serde_json::to_value(&new.holdout).unwrap()
            );
            assert_eq!(
                serde_json::to_value(&old.judge_consistency).unwrap(),
                serde_json::to_value(&new.judge_consistency).unwrap()
            );
            assert_eq!(new.phase, old.phase);
            // keep-mk7 differs from keep-mk6 only in its prompt.
            let mk6 = new.arms.iter().find(|a| a.name == "keep-mk6").unwrap();
            let mut mk7 = new
                .arms
                .iter()
                .find(|a| a.name == "keep-mk7")
                .unwrap()
                .clone();
            assert_eq!(mk7.answer_prompt.as_deref(), Some("answer-v7"));
            mk7.name = mk6.name.clone();
            mk7.answer_prompt = mk6.answer_prompt.clone();
            assert_eq!(&mk7, mk6);
            for arm in new.arms.iter().filter(|a| a.name.ends_with("-v7")) {
                assert_eq!(
                    arm.answer_prompt.as_deref(),
                    Some("answer-v7"),
                    "{}",
                    arm.name
                );
            }
        }
    }
}

#[test]
fn failure_review_excludes_incomplete_and_duplicate_comparisons() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("records.jsonl");
    let row = |q: &str, arm: &str, correct: bool| {
        serde_json::json!({
            "question_id": q, "seed": 1, "arm": arm, "category": "cat-1", "gold": "invented",
            "outcome": {"state":"answered"}, "judge": {"state":"judged", "correct":correct},
            "answer": {"text":"other"}
        })
    };
    let rows = [
        row("all-wrong", "a", false),
        row("all-wrong", "b", false),
        row("one-right", "a", true),
        row("one-right", "b", false),
        row("incomplete", "a", false),
        row("duplicate", "a", false),
        row("duplicate", "a", false),
        row("duplicate", "b", false),
    ];
    std::fs::write(
        &path,
        rows.iter()
            .map(|r| r.to_string() + "\n")
            .collect::<String>(),
    )
    .unwrap();
    let out = std::process::Command::new("python3")
        .arg(common::fixtures().join("../scripts/analyze-failures.py"))
        .arg(path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["all_arms_wrong_count"], 1);
    assert_eq!(report["incomplete_pairs_excluded"], 2);
    assert_eq!(report["all_arms_wrong"][0]["question_id"], "all-wrong");
    assert_eq!(
        report["identical_answer_judge_disagreements"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

use std::process::Command;

fn keep_bench(args: &[&str], cwd: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_keep-bench"))
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap()
}

#[test]
fn offline_dry_run_and_validation_through_the_binary() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = common::fixtures().join("manifests/longmemeval-mock.toml");
    let m = manifest.to_str().unwrap();

    let out = keep_bench(&["validate-manifest", m], dir.path());
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).ends_with("valid\n"));

    let out = keep_bench(&["dry-run", m, "--json"], dir.path());
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["questions"], 8);
    // A dry run writes nothing.
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn invalid_manifest_exits_non_zero() {
    let dir = tempfile::tempdir().unwrap();
    let text = common::fixture_manifest("longmemeval-mock.toml")
        .replace("fraction = 0.25", "fraction = 0.3");
    let lm = common::write_manifest(dir.path(), &text);
    let out = keep_bench(
        &["validate-manifest", lm.path.to_str().unwrap()],
        dir.path(),
    );
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("holdout.commitment"));
}

#[test]
fn template_manifests_refuse_to_dispatch() {
    // The shipped templates point at real datasets that tests never fetch,
    // so only check that they parse and that dispatch is blocked.
    for name in ["locomo-dev.toml", "longmemeval-s-dev.toml"] {
        let path = common::fixtures().join("../manifests").join(name);
        let lm = keep_bench::manifest::LoadedManifest::load(&path).unwrap();
        assert!(!lm.manifest.dispatch.allow_model_calls);
        assert!(lm.manifest.uses_remote_models());
        assert!(!lm.manifest.dispatch_problems().is_empty(), "{name}");
        assert!(lm.manifest.validate().iter().all(|i| !i.error), "{name}");
    }
}

#[test]
fn g0_manifests_pin_azure_models_and_compare_at_equal_context() {
    use keep_bench::ablation::ArmKind;
    for name in ["locomo.toml", "longmemeval-s.toml"] {
        let path = common::fixtures().join("../manifests/g0").join(name);
        let m = keep_bench::manifest::LoadedManifest::load(&path)
            .unwrap()
            .manifest;
        assert!(m.validate().iter().all(|i| !i.error), "{name}");
        assert!(
            m.dispatch_problems().is_empty(),
            "{name}: {:?}",
            m.dispatch_problems()
        );
        assert!(
            m.dispatch.allow_model_calls && m.fit_mask.is_some(),
            "{name}"
        );
        let pins = [
            (
                &m.models.answer,
                "gpt-4o-mini-2024-07-18",
                "KEEP_AZURE_ANSWER_DEPLOYMENT",
            ),
            (
                &m.models.judge,
                "gpt-4o-2024-08-06",
                "KEEP_AZURE_JUDGE_DEPLOYMENT",
            ),
        ];
        for (model, served, deployment) in pins {
            assert_eq!(model.served_model.as_deref(), Some(served), "{name}");
            assert_eq!(model.deployment_env.as_deref(), Some(deployment), "{name}");
            assert_eq!(
                model.base_url_env.as_deref(),
                Some("AZURE_FOUNDRY_ENDPOINT")
            );
            assert_eq!(model.api_key_env.as_deref(), Some("AZURE_FOUNDRY_API_KEY"));
        }
        assert_eq!(m.has_equal_context_baseline(), Some(true), "{name}");
        for arm in ["bm25-6k", "dense-6k"] {
            let a = m.arms.iter().find(|a| a.name == arm).unwrap();
            let (ArmKind::Bm25 { budget_tokens, .. } | ArmKind::Dense { budget_tokens, .. }) =
                a.kind
            else {
                panic!("{arm} is not a retrieval arm")
            };
            assert_eq!(budget_tokens, m.keep_context_tokens(), "{name} {arm}");
        }
    }
}

#[test]
fn copilot_g0_manifests_match_the_pinned_g0_except_models() {
    use keep_bench::manifest::{LoadedManifest, ProviderKind, Track};
    let dir = common::fixtures().join("../manifests");
    let dev = LoadedManifest::load(&dir.join("dev/longmemeval-s-copilot-dev.toml"))
        .unwrap()
        .manifest;
    for name in ["locomo", "longmemeval-s"] {
        let load = |file: String| LoadedManifest::load(&dir.join(file)).unwrap().manifest;
        let g0 = load(format!("g0/{name}.toml"));
        let m = load(format!("g0/{name}-copilot.toml"));
        assert!(m.validate().iter().all(|i| !i.error), "{name}");
        assert!(
            m.dispatch_problems().is_empty(),
            "{name}: {:?}",
            m.dispatch_problems()
        );
        assert_eq!(m.track, Track::Development, "{name}");
        assert!(m.non_canonical_label().is_some(), "{name}");
        assert_eq!(m.dispatch_concurrency_with(|_| None).unwrap(), 16, "{name}");
        assert!(m.budget.max_ai_credits.is_some_and(|c| c > 0.0), "{name}");
        assert!(m.fit_mask.is_some() && m.dataset.subset.is_none(), "{name}");
        assert_eq!(m.dataset, g0.dataset, "{name}");
        assert_eq!(m.holdout, g0.holdout, "{name}");
        assert_eq!(m.embedding, g0.embedding, "{name}");
        assert_eq!(m.storage, g0.storage, "{name}");
        assert_eq!(m.retrieval, g0.retrieval, "{name}");
        assert_eq!(m.seeds, g0.seeds, "{name}");
        assert_eq!(m.arms, g0.arms, "{name}");
        for (model, dev_model) in [
            (&m.models.answer, &dev.models.answer),
            (&m.models.judge, &dev.models.judge),
        ] {
            assert_eq!(model.provider, ProviderKind::CopilotCli, "{name}");
            assert_eq!(model.model, dev_model.model, "{name}");
            assert_eq!(model.prompt_version, dev_model.prompt_version, "{name}");
            assert_eq!(model.copilot, dev_model.copilot, "{name}");
            // Full context must fit the longest LongMemEval-S history.
            assert!(model.context_limit >= 110_000, "{name}");
        }
    }
}

#[test]
fn copilot_g0_v2_manifests_only_add_the_ranked_arms_and_their_reranker() {
    use keep_bench::ablation::{ArmKind, KeepPath};
    use keep_bench::manifest::LoadedManifest;
    let dir = common::fixtures().join("../manifests");
    for name in ["locomo", "longmemeval-s"] {
        let load = |file: String| LoadedManifest::load(&dir.join(file)).unwrap().manifest;
        let v1 = load(format!("g0/{name}-copilot.toml"));
        let v2 = load(format!("g0/{name}-copilot-v2.toml"));
        assert!(v2.validate().iter().all(|i| !i.error), "{name}");
        assert!(v2.dispatch_problems().is_empty(), "{name}");
        assert_ne!(v2.id, v1.id, "{name}");
        assert!(v1.reranker.is_none(), "{name}");
        let r = v2.reranker.as_ref().expect("v2 pins a reranker");
        assert_eq!(r.top_n, 50, "{name}");
        let (old, added) = v2.arms.split_at(v1.arms.len());
        assert_eq!(old, v1.arms.as_slice(), "{name}");
        let paths: Vec<KeepPath> = added
            .iter()
            .map(|a| match a.kind {
                ArmKind::Keep { path, .. } => path,
                _ => panic!("{name}: v2 adds only Keep arms"),
            })
            .collect();
        assert_eq!(paths, [KeepPath::K2, KeepPath::Mk2], "{name}");
        assert_eq!(v2.has_equal_context_baseline(), Some(true), "{name}");
        for (a, b) in [
            (format!("{:?}", v1.models), format!("{:?}", v2.models)),
            (format!("{:?}", v1.dataset), format!("{:?}", v2.dataset)),
            (format!("{:?}", v1.holdout), format!("{:?}", v2.holdout)),
            (format!("{:?}", v1.embedding), format!("{:?}", v2.embedding)),
            (format!("{:?}", v1.storage), format!("{:?}", v2.storage)),
            (format!("{:?}", v1.retrieval), format!("{:?}", v2.retrieval)),
            (format!("{:?}", v1.dispatch), format!("{:?}", v2.dispatch)),
            (format!("{:?}", v1.fit_mask), format!("{:?}", v2.fit_mask)),
            (format!("{:?}", v1.seeds), format!("{:?}", v2.seeds)),
        ] {
            assert_eq!(a, b, "{name}");
        }
    }
}

#[test]
fn judge_calibration_runs_offline_on_labels_and_probes() {
    let dir = tempfile::tempdir().unwrap();
    let f = common::fixtures();
    let out = keep_bench(
        &[
            "calibrate-judge",
            f.join("manifests/locomo-mock.toml").to_str().unwrap(),
            "--labels",
            f.join("judge-calibration.jsonl").to_str().unwrap(),
            "--probes-per-type",
            "3",
            "--json",
        ],
        dir.path(),
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let r: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(r["mock"], true);
    assert_eq!(r["by_probe"]["human"]["n"], 6);
    // The lexical mock judge accepts a shifted year: calibration must show it.
    assert!(
        r["by_probe"]["date-shift"]["false_accepts"]
            .as_u64()
            .unwrap()
            > 0
    );
}

#[test]
fn copilot_g0_v4_manifests_add_v4_arms_and_leave_v3_unchanged() {
    use keep_bench::manifest::LoadedManifest;
    let dir = common::fixtures().join("../manifests");
    for name in ["locomo", "longmemeval-s"] {
        let load = |file: String| LoadedManifest::load(&dir.join(file)).unwrap().manifest;
        let v3 = load(format!("g0/{name}-copilot-v3.toml"));
        let v4 = load(format!("g0/{name}-copilot-v4.toml"));
        assert!(v4.validate().iter().all(|i| !i.error), "{name}");
        assert!(v4.dispatch_problems().is_empty(), "{name}");
        assert_ne!(v4.id, v3.id, "{name}");
        let (old, added) = v4.arms.split_at(v3.arms.len());
        assert_eq!(old, v3.arms.as_slice(), "{name}");
        let arm = |n: &str| v3.arms.iter().find(|a| a.name == n).unwrap();
        for a in added {
            assert_eq!(
                a.answer_prompt("answer-v1"),
                "answer-v4",
                "{name} {}",
                a.name
            );
        }
        for (new, base) in [
            ("dense-6k-v4", "dense-6k"),
            ("full-context-v4", "full-context"),
            ("keep-k4", "keep-k3"),
            ("keep-mk4", "keep-mk3"),
        ] {
            let a = added.iter().find(|a| a.name == new).unwrap();
            let b = arm(base);
            assert_eq!(a.kind, b.kind, "{name} {new}");
            let control = b.fact_options() == a.fact_options();
            assert_eq!(control, new.ends_with("-v4"), "{name} {new}");
        }
        for (a, b) in [
            (format!("{:?}", v3.models), format!("{:?}", v4.models)),
            (format!("{:?}", v3.dataset), format!("{:?}", v4.dataset)),
            (format!("{:?}", v3.holdout), format!("{:?}", v4.holdout)),
            (format!("{:?}", v3.retrieval), format!("{:?}", v4.retrieval)),
            (format!("{:?}", v3.fit_mask), format!("{:?}", v4.fit_mask)),
        ] {
            assert_eq!(a, b, "{name}");
        }
    }
}

#[test]
fn g2_confirmation_manifests_share_every_frozen_setting_with_development() {
    use keep_bench::manifest::{LoadedManifest, Phase, StorageProfile};
    let root = common::fixtures().join("../manifests");
    let load = |path: &str| LoadedManifest::load(&root.join(path)).unwrap().manifest;
    for dataset in ["locomo", "longmemeval-s"] {
        let dev = load(&format!("g2/{dataset}.toml"));
        let conf = load(&format!("g2/{dataset}-confirmation.toml"));
        for m in [&dev, &conf] {
            assert!(m.validate().iter().all(|i| !i.error), "{:?}", m.validate());
            assert!(
                m.dispatch_problems().is_empty(),
                "{:?}",
                m.dispatch_problems()
            );
            assert!(m.track.is_canonical() && m.seeds == [1, 2, 3]);
            assert!(m.dispatch.tokens_per_minute.is_some());
            // Answer, judge and extraction all use the one pinned deployment.
            let models = [
                &m.models.answer,
                &m.models.judge,
                m.extract_model().unwrap(),
            ];
            for model in models {
                assert_eq!(model.served_model.as_deref(), Some("gpt-6-luna-2026-09-22"));
                assert_eq!(
                    model.deployment_env.as_deref(),
                    Some("KEEP_AZURE_OPENAI_DEPLOYMENT")
                );
            }
        }
        assert_eq!(
            (dev.phase, conf.phase),
            (Phase::Development, Phase::Confirmation)
        );
        let candidate = &conf.confirmation.as_ref().unwrap().frozen_candidate;
        // Exactly the candidate, the two gate comparators and the reported
        // keep-mk6, each unchanged from G0 v7.
        assert_eq!(candidate, "keep-mk7");
        let names: Vec<_> = dev.arms.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            names,
            ["full-context-v7", "dense-6k-v7", "keep-mk6", "keep-mk7"]
        );
        let v7 = load(&format!("g0/{dataset}-copilot-v7.toml"));
        let arm = |m: &keep_bench::manifest::Manifest, name: &str| {
            m.arms.iter().find(|a| a.name == name).cloned()
        };
        for name in [
            candidate.as_str(),
            "full-context-v7",
            "dense-6k-v7",
            "keep-mk6",
        ] {
            assert!(
                arm(&dev, name).is_some() && arm(&dev, name) == arm(&v7, name),
                "{name}"
            );
        }
        let frozen = |m: &keep_bench::manifest::Manifest| {
            let mut v = serde_json::to_value(m).unwrap();
            for key in ["id", "description", "phase", "confirmation"] {
                v.as_object_mut().unwrap().remove(key);
            }
            v
        };
        assert_eq!(frozen(&dev), frozen(&conf), "{dataset}");
        // The gate needs both storage profiles: each manifest has a
        // postgres-vector twin that differs only in storage.
        for (name, m) in [
            (dataset.to_string(), &dev),
            (format!("{dataset}-confirmation"), &conf),
        ] {
            let twin = load(&format!("g2/{name}-pgvector.toml"));
            assert_eq!(
                (m.storage.profile, twin.storage.profile),
                (
                    StorageProfile::PostgresQdrant,
                    StorageProfile::PostgresVector
                )
            );
            let without_storage = |m: &keep_bench::manifest::Manifest| {
                let mut v = frozen(m);
                v.as_object_mut().unwrap().remove("storage");
                v
            };
            assert_eq!(without_storage(m), without_storage(&twin), "{name}");
            assert_eq!(m.confirmation, twin.confirmation, "{name}");
        }
    }
}

/// `KEEP_DISPATCH_CONCURRENCY` changes how many calls are in flight, not
/// the manifest hash or any record, and `run.json` records the value used.
#[test]
fn concurrency_override_changes_neither_records_nor_hash() {
    use keep_bench::manifest::CONCURRENCY_ENV;
    let dir = tempfile::tempdir().unwrap();
    let manifest = common::fixtures().join("manifests/facts-mock.toml");
    let m = manifest.to_str().unwrap();
    let run_with = |concurrency: &str| {
        let out = Command::new(env!("CARGO_BIN_EXE_keep-bench"))
            .args(["run", m, "--results", concurrency])
            .env(CONCURRENCY_ENV, concurrency)
            .current_dir(dir.path())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let run_dir = stdout
            .lines()
            .filter_map(|l| l.strip_prefix("artifacts: "))
            .next_back()
            .map(|d| dir.path().join(d))
            .unwrap();
        let run: serde_json::Value =
            serde_json::from_slice(&std::fs::read(run_dir.join("run.json")).unwrap()).unwrap();
        // Ids and timings differ between any two runs; nothing else may.
        fn strip(v: &mut serde_json::Value) {
            match v {
                serde_json::Value::Object(o) => {
                    for k in ["run_id", "latency_ms", "build_ms"] {
                        o.remove(k);
                    }
                    o.values_mut().for_each(strip);
                }
                serde_json::Value::Array(a) => a.iter_mut().for_each(strip),
                _ => {}
            }
        }
        let records: Vec<serde_json::Value> =
            std::fs::read_to_string(run_dir.join("records.jsonl"))
                .unwrap()
                .lines()
                .map(|l| {
                    let mut v = serde_json::from_str(l).unwrap();
                    strip(&mut v);
                    v
                })
                .collect();
        (run, records)
    };
    let (one, records_one) = run_with("1");
    let (eight, records_eight) = run_with("8");
    assert_eq!(
        (one["concurrency"].as_u64(), eight["concurrency"].as_u64()),
        (Some(1), Some(8))
    );
    assert_eq!(one["manifest_hash"], eight["manifest_hash"]);
    assert!(records_one.len() > 1);
    assert_eq!(records_one, records_eight);

    let out = Command::new(env!("CARGO_BIN_EXE_keep-bench"))
        .args(["run", m, "--results", "zero"])
        .env(CONCURRENCY_ENV, "0")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains(CONCURRENCY_ENV));
}

#[test]
fn g2_v2_manifests_swap_only_the_candidate() {
    use keep_bench::manifest::{LoadedManifest, Phase};
    let root = common::fixtures().join("../manifests");
    let load = |path: &str| LoadedManifest::load(&root.join(path)).unwrap().manifest;
    let without = |m: &keep_bench::manifest::Manifest, keys: &[&str]| {
        let mut v = serde_json::to_value(m).unwrap();
        for key in keys {
            v.as_object_mut().unwrap().remove(*key);
        }
        v
    };
    for dataset in ["locomo", "longmemeval-s"] {
        for name in [
            dataset.to_string(),
            format!("{dataset}-pgvector"),
            format!("{dataset}-confirmation"),
            format!("{dataset}-confirmation-pgvector"),
        ] {
            let v1 = load(&format!("g2/{name}.toml"));
            let v2 = load(&format!("g2/{name}-v2.toml"));
            assert!(
                v2.validate().iter().all(|i| !i.error),
                "{:?}",
                v2.validate()
            );
            assert!(v2.dispatch_problems().is_empty(), "{name}");
            assert_eq!(v2.id, format!("{}-v2", v1.id));
            assert_eq!(v2.has_equal_context_baseline(), Some(true), "{name}");
            // Everything else, including the holdout, seeds, models,
            // budget and statistics, is version 1's.
            let keys = ["id", "description", "arms", "confirmation"];
            assert_eq!(without(&v1, &keys), without(&v2, &keys), "{name}");
            let names: Vec<_> = v2.arms.iter().map(|a| a.name.as_str()).collect();
            assert_eq!(
                names,
                ["full-context-v7", "dense-6k-v7", "keep-mk7", "keep-mk8"]
            );
            let arm = |m: &keep_bench::manifest::Manifest, n: &str| {
                m.arms.iter().find(|a| a.name == n).cloned().unwrap()
            };
            for n in ["full-context-v7", "dense-6k-v7", "keep-mk7"] {
                assert_eq!(arm(&v1, n), arm(&v2, n), "{name} {n}");
            }
            // keep-mk8 is keep-mk7 with the two v8 options.
            let mut mk8 = arm(&v2, "keep-mk8");
            assert!(mk8.focus && mk8.verbatim_first, "{name}");
            mk8.name = "keep-mk7".into();
            mk8.focus = false;
            mk8.verbatim_first = false;
            assert_eq!(mk8, arm(&v1, "keep-mk7"), "{name}");
            match v2.phase {
                Phase::Development => assert!(v2.confirmation.is_none()),
                Phase::Confirmation => assert_eq!(
                    v2.confirmation.as_ref().unwrap().frozen_candidate,
                    "keep-mk8"
                ),
            }
        }
        let dev = load(&format!("g2/{dataset}-v2.toml"));
        let conf = load(&format!("g2/{dataset}-confirmation-v2.toml"));
        let keys = ["id", "description", "phase", "confirmation"];
        assert_eq!(without(&dev, &keys), without(&conf, &keys), "{dataset}");
    }
}
