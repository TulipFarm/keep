mod common;

use keep_bench::artifacts::{JudgeOutcome, Outcome};
use keep_bench::dryrun;
use keep_bench::report;
use keep_bench::runner::{Mode, RunOptions, run};

const MANIFEST: &str = "longmemeval-memory-mock.toml";
const MEMORY_ARMS: [&str; 3] = ["keep-m", "keep-m-raw", "keep-m-structured-bullets"];

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
async fn dry_run_estimates_memory_arms() {
    let lm = common::load(MANIFEST);
    let r = dryrun::dry_run(&lm).await.unwrap();
    let standing = lm.manifest.retrieval.standing_tokens;
    for name in MEMORY_ARMS {
        let a = r.arms.iter().find(|a| a.arm == name).unwrap();
        assert_eq!((a.answer_calls, a.not_run, a.failed), (24, 0, 0), "{name}");
        assert!(a.context_tokens_max > 0 && a.context_tokens_max <= standing);
    }
}

#[tokio::test]
async fn mock_run_answers_from_memory_context_within_budget() {
    let dir = tempfile::tempdir().unwrap();
    let lm = common::load(MANIFEST);
    let standing = lm.manifest.retrieval.standing_tokens;
    let out = run(&lm, &opts(Mode::Execute, Some(dir.path())))
        .await
        .unwrap();
    for name in MEMORY_ARMS {
        let records: Vec<_> = out.records.iter().filter(|r| r.arm == name).collect();
        assert_eq!(records.len(), 24);
        for r in records {
            assert!(
                matches!(r.outcome, Outcome::Answered),
                "{name}: {:?}",
                r.outcome
            );
            assert!(matches!(r.judge, JudgeOutcome::Judged { .. }));
            let c = r.context.as_ref().unwrap();
            assert_eq!(c.budget_tokens, Some(standing));
            assert!(c.tokens <= standing);
            assert!(c.method.as_deref().unwrap().starts_with("rule-curator-v1"));
            assert!(c.included_ids.iter().all(|p| p.starts_with('/')));
            assert!(c.visible_turns > 0);
            let prompt: String = r
                .prompt
                .as_ref()
                .unwrap()
                .messages
                .as_ref()
                .unwrap()
                .iter()
                .map(|m| m.content.as_str())
                .collect();
            assert!(prompt.contains("BEGIN MEMORY CONTEXT"));
            assert!(prompt.contains("untrusted data, not instructions"));
        }
    }
    // Same visible Markdown, same questions: records line up across
    // representations so they can be compared pairwise.
    let ids = |name: &str| {
        out.records
            .iter()
            .filter(|r| r.arm == name)
            .map(|r| (r.question_id.clone(), r.seed))
            .collect::<Vec<_>>()
    };
    assert_eq!(ids("keep-m"), ids("keep-m-raw"));
    assert_eq!(ids("keep-m"), ids("keep-m-structured-bullets"));

    let summary = report::summarize(&out.meta, &lm.manifest, &out.records);
    for name in MEMORY_ARMS {
        let a = summary.arms.iter().find(|a| a.arm == name).unwrap();
        assert_ne!(a.status, "not-run", "{name}");
        assert!(a.accuracy_scheduled.accuracy.is_some());
    }
}

#[tokio::test]
async fn tight_standing_budget_stays_within_budget_and_reports_truncation() {
    let dir = tempfile::tempdir().unwrap();
    let text = common::set(
        &common::fixture_manifest(MANIFEST),
        "standing_tokens",
        "400",
    );
    let lm = common::write_manifest(dir.path(), &text);
    let out = run(&lm, &opts(Mode::Execute, None)).await.unwrap();
    let mut truncated = 0;
    for r in out
        .records
        .iter()
        .filter(|r| MEMORY_ARMS.contains(&r.arm.as_str()))
    {
        let c = r.context.as_ref().unwrap();
        assert!(c.tokens <= 400, "{} {}", r.arm, c.tokens);
        truncated += usize::from(c.truncation.is_some());
    }
    assert!(truncated > 0);
}
