//! Scoring protocols and exports for external harnesses, end to end on
//! the offline fixtures.

mod common;

use common::{fixture_manifest, write_manifest};
use keep_bench::artifacts::Outcome;
use keep_bench::export::{self, ExportRequest, IdMapEntry};
use keep_bench::manifest::LoadedManifest;
use keep_bench::protocol::ProtocolName;
use keep_bench::runner::{Mode, RunOptions, Selection, run, run_selected};
use keep_bench::{dryrun, report};
use serde_json::Value;

const LOCOMO: &str = "locomo-mock.toml";
const LME: &str = "longmemeval-mock.toml";

fn opts(mode: Mode, root: Option<&std::path::Path>) -> RunOptions {
    RunOptions {
        mode,
        unseal: false,
        allow_model_calls: false,
        results_root: root.map(ToOwned::to_owned),
        spend: None,
    }
}

/// A fixture manifest under a protocol, with that protocol's prompt
/// versions and a context window large enough for its prompts.
fn with_protocol(fixture: &str, name: ProtocolName, split: &str) -> String {
    fixture_manifest(fixture)
        .replace(
            "prompt_version = \"answer-v1\"",
            &format!("prompt_version = \"{}\"", name.answer_prompt_version()),
        )
        .replace(
            "prompt_version = \"judge-v1\"",
            &format!("prompt_version = \"{}\"", name.judge_prompt_version()),
        )
        .replace("context_limit = 4096", "context_limit = 16384")
        .replace("max_output_tokens = 8\n", "max_output_tokens = 256\n")
        + &format!(
            "\n[protocol]\nname = \"{}\"\nsplit = \"{split}\"\n",
            name.as_str()
        )
}

fn load_text(dir: &std::path::Path, text: &str) -> LoadedManifest {
    std::fs::create_dir_all(dir).unwrap();
    write_manifest(dir, text)
}

#[tokio::test]
async fn mem0_locomo_scores_categories_one_to_four_only() {
    let tmp = tempfile::tempdir().unwrap();
    let lm = load_text(
        tmp.path(),
        &with_protocol(LOCOMO, ProtocolName::Mem0Locomo, "dev"),
    );
    let root = tmp.path().join("results");
    let out = run(&lm, &opts(Mode::Execute, Some(&root))).await.unwrap();
    assert!(!out.records.is_empty());
    assert!(
        out.records
            .iter()
            .all(|r| ["cat-1", "cat-2", "cat-3", "cat-4"].contains(&r.category.as_str())),
        "adversarial questions must be filtered out"
    );
    let rec = out.meta.protocol.as_ref().expect("protocol recorded");
    assert_eq!(rec.name, ProtocolName::Mem0Locomo);
    assert_eq!(rec.prompts.len(), 3);
    assert!(rec.upstream_commit.len() == 40);

    // The mock answers in Mem0's format and the judge returns its JSON
    // label, so every answered question gets a verdict.
    let summary = report::summarize(&out.meta, &lm.manifest, &out.records);
    let p = summary.protocol.as_ref().expect("protocol section");
    let full = p.arms.iter().find(|a| a.arm == "oracle-evidence").unwrap();
    assert!(full.overall.n > 0);
    assert!(full.overall.accuracy.unwrap() > 0.5, "{full:?}");
    // Only questions the arm could not answer (dangling fixture evidence)
    // go unscored, and they count as wrong.
    let unanswered = out
        .records
        .iter()
        .filter(|r| r.arm == "oracle-evidence" && r.answer.is_none())
        .count();
    assert_eq!(full.unscored, unanswered);
    assert!(
        full.by_group
            .keys()
            .all(|k| ["single-hop", "multi-hop", "temporal", "open-domain"].contains(&k.as_str())),
        "{:?}",
        full.by_group.keys()
    );
    let run_json: Value = serde_json::from_str(
        &std::fs::read_to_string(out.dir.as_ref().unwrap().join("run.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(run_json["protocol"]["name"], "mem0-locomo");
    let md = std::fs::read_to_string(out.dir.unwrap().join("summary.md")).unwrap();
    assert!(md.contains("mem0-locomo"), "{md}");
}

#[tokio::test]
async fn longmemeval_official_reports_types_abstention_and_task_average() {
    let tmp = tempfile::tempdir().unwrap();
    let lm = load_text(
        tmp.path(),
        &with_protocol(LME, ProtocolName::LongmemevalOfficial, "dev"),
    );
    let out = run(&lm, &opts(Mode::Execute, None)).await.unwrap();
    let summary = report::summarize(&out.meta, &lm.manifest, &out.records);
    let p = summary.protocol.as_ref().unwrap();
    let oracle = p.arms.iter().find(|a| a.arm == "oracle-evidence").unwrap();
    assert!(oracle.task_averaged.is_some());
    assert!(oracle.overall.n > 0);
    let has_abs = out.records.iter().any(|r| r.abstention);
    assert_eq!(oracle.abstention.is_some(), has_abs);
    for k in oracle.by_group.keys() {
        assert!(
            keep_bench::protocol::LONGMEMEVAL_TYPES.contains(&k.as_str()),
            "{k}"
        );
    }
    // Judge prompts are the official per-type templates.
    let judged = out
        .records
        .iter()
        .find(|r| r.arm == "oracle-evidence" && r.judge_call.is_some())
        .unwrap();
    assert!(judged.judge_call.as_ref().unwrap().text.is_some());
}

#[tokio::test]
async fn default_protocol_leaves_manifests_and_outputs_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let plain = load_text(&tmp.path().join("plain"), &fixture_manifest(LME));
    let explicit = load_text(
        &tmp.path().join("explicit"),
        &(fixture_manifest(LME) + "\n[protocol]\nname = \"keep\"\n"),
    );
    assert_eq!(plain.hash, explicit.hash);
    assert_eq!(plain.manifest.hash(), explicit.manifest.hash());

    let root = tmp.path().join("results");
    let out = run(&plain, &opts(Mode::Execute, Some(&root)))
        .await
        .unwrap();
    assert!(out.meta.protocol.is_none());
    let dir = out.dir.unwrap();
    for f in ["run.json", "summary.json", "manifest.json"] {
        let v: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join(f)).unwrap()).unwrap();
        assert!(v.get("protocol").is_none(), "{f} gained a protocol key");
    }
    let md = std::fs::read_to_string(dir.join("summary.md")).unwrap();
    assert!(!md.contains("Scoring protocol"));
}

#[tokio::test]
async fn full_split_needs_explicit_holdout_consumption() {
    let tmp = tempfile::tempdir().unwrap();
    let lm = load_text(
        tmp.path(),
        &with_protocol(LME, ProtocolName::LongmemevalOfficial, "full"),
    );
    let Err(err) = run(&lm, &opts(Mode::Execute, None)).await else {
        panic!("the full split must not run without consume_holdout");
    };
    assert!(err.to_string().contains("holdout"), "{err}");

    let all = run_selected(
        &lm,
        &opts(Mode::Execute, None),
        &Selection {
            consume_holdout: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let dev_dir = tmp.path().join("dev");
    let dev = run(
        &load_text(
            &dev_dir,
            &with_protocol(LME, ProtocolName::LongmemevalOfficial, "dev"),
        ),
        &opts(Mode::Execute, None),
    )
    .await;
    let dev = dev.unwrap();
    let ids = |o: &keep_bench::runner::RunOutput| {
        o.records
            .iter()
            .map(|r| r.question_id.clone())
            .collect::<std::collections::BTreeSet<_>>()
    };
    assert!(ids(&all).len() > ids(&dev).len());
    assert!(ids(&dev).is_subset(&ids(&all)));

    // A dry run calls no model and may size the full set.
    let est = dryrun::dry_run(&lm).await.unwrap();
    assert!(est.consumes_holdout);
}

#[tokio::test]
async fn protocol_validation_rejects_mismatches() {
    let tmp = tempfile::tempdir().unwrap();
    let errors = |text: String| {
        load_text(tmp.path(), &text)
            .manifest
            .validate()
            .into_iter()
            .filter(|i| i.error)
            .map(|i| format!("{}: {}", i.field, i.message))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let e = errors(with_protocol(LME, ProtocolName::Mem0Locomo, "dev"));
    assert!(e.contains("mem0-locomo"), "{e}");
    let e = errors(fixture_manifest(LME) + "\n[protocol]\nname = \"keep\"\nsplit = \"full\"\n");
    assert!(e.contains("protocol.split"), "{e}");
    assert!(errors(with_protocol(LOCOMO, ProtocolName::Mem0Locomo, "dev")).is_empty());

    // The manifest must name the protocol's prompt versions.
    let lm = load_text(
        tmp.path(),
        &with_protocol(LOCOMO, ProtocolName::Mem0Locomo, "dev").replace(
            "prompt_version = \"mem0-locomo-answer-v1\"",
            "prompt_version = \"answer-v1\"",
        ),
    );
    let Err(e) = run(&lm, &opts(Mode::Execute, None)).await else {
        panic!("a mismatched prompt version must be refused");
    };
    assert!(e.to_string().contains("prompt"), "{e}");
}

#[tokio::test]
async fn locomo_export_matches_the_mem0_harness_layout() {
    let tmp = tempfile::tempdir().unwrap();
    let lm = load_text(
        tmp.path(),
        &with_protocol(LOCOMO, ProtocolName::Mem0Locomo, "dev"),
    );
    let out = tmp.path().join("export");
    let arms = ["bm25", "full-context"];
    let report = export::export(
        &lm,
        ExportRequest {
            out: out.clone(),
            project: "keep_test".into(),
            single_memory: false,
            selection: Selection {
                arms: Some(arms.iter().map(|s| s.to_string()).collect()),
                ..Default::default()
            },
        },
    )
    .await
    .unwrap();
    assert_eq!(report.arms.len(), 2);

    let dataset: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("dataset.json")).unwrap()).unwrap();
    let ids: Vec<IdMapEntry> =
        serde_json::from_str(&std::fs::read_to_string(out.join("id_map.json")).unwrap()).unwrap();
    assert_eq!(ids.len(), report.questions);
    let original: Value = serde_json::from_str(
        &std::fs::read_to_string(common::fixtures().join("locomo-mini.json")).unwrap(),
    )
    .unwrap();

    // Re-derive the harness's expected items exactly as its evaluate-only
    // mode does: conv index in the file, qa index in that conversation's
    // list, categories 1-4.
    let mut expected = Vec::new();
    for (ci, conv) in dataset.as_array().unwrap().iter().enumerate() {
        for (qi, qa) in conv["qa"].as_array().unwrap().iter().enumerate() {
            if (1..=4).contains(&qa["category"].as_u64().unwrap()) {
                expected.push((format!("conv{ci}_q{qi}"), qa.clone()));
            }
        }
    }
    assert_eq!(expected.len(), ids.len(), "every exported qa is scored");
    for e in &ids {
        let (ci, qi) = (
            e.original_conversation_idx.unwrap(),
            e.original_qa_idx.unwrap(),
        );
        let (_, qa) = expected
            .iter()
            .find(|(h, _)| h == &e.harness_question_id)
            .unwrap();
        assert_eq!(qa, &original[ci]["qa"][qi], "{e:?}");
        assert!(e.keep_question_id.ends_with(&format!(":q{qi}")));
    }

    for a in &report.arms {
        let dir = std::path::Path::new(&a.output_dir).join("predicted_keep_test");
        for (hid, qa) in &expected {
            let path = dir.join(format!("{hid}.json"));
            if !path.exists() {
                assert!(a.arm == "full-context" && !a.missing.is_empty());
                continue;
            }
            let f: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            assert_eq!(&f["question_id"], hid);
            assert_eq!(f["question"], qa["question"]);
            assert_eq!(f["category"], qa["category"]);
            let results = f["retrieval"]["search_results"].as_array().unwrap();
            assert!(!results.is_empty());
            for r in results {
                assert!(r["memory"].is_string() && r["id"].is_string() && r["score"].is_f64());
            }
            let scores: Vec<f64> = results
                .iter()
                .map(|r| r["score"].as_f64().unwrap())
                .collect();
            assert!(scores.windows(2).all(|w| w[0] >= w[1]));
        }
        assert_eq!(a.written + a.missing.len(), expected.len());
    }
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("export.json")).unwrap()).unwrap();
    assert_eq!(meta["protocol"]["name"], "mem0-locomo");
    assert_eq!(meta["commands"].as_array().unwrap().len(), 2);
    // Exports never overwrite.
    assert!(
        export::export(
            &lm,
            ExportRequest {
                out,
                project: "keep_test".into(),
                single_memory: false,
                selection: Selection::default(),
            },
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn longmemeval_export_and_hypotheses_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let lm = load_text(
        tmp.path(),
        &with_protocol(LME, ProtocolName::LongmemevalOfficial, "dev"),
    );
    let out = tmp.path().join("export");
    let report = export::export(
        &lm,
        ExportRequest {
            out: out.clone(),
            project: "p".into(),
            single_memory: true,
            selection: Selection {
                arms: Some(["dense".to_string()].into()),
                ..Default::default()
            },
        },
    )
    .await
    .unwrap();
    let a = &report.arms[0];
    assert_eq!(a.written, report.questions);
    let reference: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("dataset.json")).unwrap()).unwrap();
    for e in reference.as_array().unwrap() {
        assert!(e.get("haystack_sessions").is_none());
        let id = e["question_id"].as_str().unwrap();
        let f: Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::path::Path::new(&a.output_dir).join(format!("predicted_p/{id}.json")),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(f["is_abstention"], id.ends_with("_abs"));
        assert_eq!(f["question_type"], e["question_type"]);
        assert_eq!(
            f["retrieval"]["search_results"].as_array().unwrap().len(),
            1
        );
    }

    let run = run(&lm, &opts(Mode::Execute, None)).await.unwrap();
    let (hyps, unanswered) = export::hypotheses(&run.records, "dense", None).unwrap();
    let expected = run
        .records
        .iter()
        .filter(|r| r.arm == "dense" && r.seed == run.records[0].seed)
        .count();
    assert_eq!(hyps.len(), expected);
    assert_eq!(
        unanswered,
        run.records
            .iter()
            .filter(|r| r.arm == "dense"
                && r.seed == run.records[0].seed
                && !matches!(r.outcome, Outcome::Answered))
            .count()
    );
    let path = tmp.path().join("hyp.jsonl");
    export::write_hypotheses(&path, &hyps).unwrap();
    assert_eq!(export::read_hypotheses(&path).unwrap(), hyps);
    for line in std::fs::read_to_string(&path).unwrap().lines() {
        let v: Value = serde_json::from_str(line).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 2);
    }
}
