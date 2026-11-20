//! The Copilot CLI provider against a fake `copilot` executable: no real
//! model is called.
#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use keep_providers::copilot::{
    CopilotCliChat, CopilotConfig, CreditRates, OVERHEAD_REPORTED, UNPINNED_REVISION, parse_jsonl,
};
use keep_providers::{
    ChatModel, ChatRequest, CreditLedger, DispatchPolicy, Message, ProviderError,
};
use tokio::sync::Semaphore;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/copilot-cli-sample.jsonl")
}

/// A fake CLI that records its stdin, arguments, working directory and call
/// count, then runs `body`.
fn fake_cli(dir: &Path, body: &str) -> PathBuf {
    let d = dir.display();
    let script = format!(
        "#!/bin/sh\n\
         cat > \"{d}/stdin.txt\"\n\
         for a in \"$@\"; do printf '%s\\n' \"$a\"; done > \"{d}/args.txt\"\n\
         pwd > \"{d}/cwd.txt\"\n\
         ls -A > \"{d}/cwd-listing.txt\"\n\
         echo call >> \"{d}/calls.txt\"\n\
         {body}\n"
    );
    let path = dir.join("copilot");
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn calls(dir: &Path) -> usize {
    std::fs::read_to_string(dir.join("calls.txt"))
        .map(|s| s.lines().count())
        .unwrap_or(0)
}

fn chat(dir: &Path, program: PathBuf, cap: f64) -> CopilotCliChat {
    let mut c = CopilotConfig::new("claude-haiku-4.5");
    c.program = program;
    c.workdir_root = dir.join("work");
    c.timeout = Duration::from_secs(20);
    c.rates = CreditRates {
        input_per_1k: 0.1,
        output_per_1k: 0.5,
        overhead_tokens: 2000,
    };
    let permit = DispatchPolicy {
        manifest_allows: true,
        operator_allows: true,
    }
    .permit("test", vec![])
    .unwrap();
    CopilotCliChat::new(
        c,
        &permit,
        Arc::new(CreditLedger::new(cap)),
        Arc::new(Semaphore::new(4)),
    )
    .unwrap()
}

fn request(system: &str, user: &str) -> ChatRequest {
    ChatRequest {
        messages: vec![Message::system(system), Message::user(user)],
        max_output_tokens: 200,
        temperature: 0.0,
        seed: Some(1),
        estimated_input_tokens: 1000,
    }
}

#[test]
fn parses_answer_usage_credits_and_hidden_overhead_from_sample_output() {
    let p = parse_jsonl(&std::fs::read_to_string(fixture()).unwrap());
    assert_eq!(
        p.text.as_deref(),
        Some("The meeting moved to \"Tuesday\".\nIt was 3 o'clock.")
    );
    assert_eq!(p.model.as_deref(), Some("claude-haiku-4.5"));
    assert_eq!((p.input_tokens, p.output_tokens), (Some(2202), Some(94)));
    assert_eq!(p.reasoning_tokens, 87);
    assert!((p.credits().unwrap() - 0.2672).abs() < 1e-9);
    assert_eq!(p.premium_requests, Some(0.33));
    assert_eq!(p.overhead_tokens, Some(2000));
    assert!(p.model_called);
    assert_eq!(p.exit_code, Some(0));

    let garbage =
        parse_jsonl("not json\n{\"type\":\"session.error\",\"data\":{\"message\":\"boom\"}}\n");
    assert_eq!(garbage.unparsed_lines, 1);
    assert_eq!(garbage.errors, vec!["boom".to_owned()]);
    assert!(garbage.text.is_none());
}

#[tokio::test]
async fn prompts_with_shell_metacharacters_and_large_size_arrive_verbatim_on_stdin() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(dir.path(), &format!("cat \"{}\"", fixture().display()));
    let model = chat(dir.path(), cli, 100.0);
    let nasty = "He said \"it's $HOME\" and `rm -rf /`; echo $(whoami) | tee x\n\n\\n\t'single'";
    let big = format!("{nasty}\n{}", "filler line with words\n".repeat(20_000));
    let r = model
        .complete(&request("System rules.", &big))
        .await
        .unwrap();

    let stdin = std::fs::read_to_string(dir.path().join("stdin.txt")).unwrap();
    assert_eq!(stdin, format!("System rules.\n\n{big}"));
    let args = std::fs::read_to_string(dir.path().join("args.txt")).unwrap();
    assert!(!args.contains("HOME") && !args.contains("filler"));
    assert!(args.lines().any(|a| a == "--available-tools=none"));
    // Ran in an empty directory under the configured root, removed afterwards.
    let cwd = std::fs::read_to_string(dir.path().join("cwd.txt")).unwrap();
    let cwd = PathBuf::from(cwd.trim());
    assert!(cwd.starts_with(dir.path().join("work").canonicalize().unwrap()));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("cwd-listing.txt")).unwrap(),
        ""
    );
    assert!(!cwd.exists());

    assert_eq!(model.descriptor().revision, UNPINNED_REVISION);
    assert_eq!(r.usage.input_tokens, 2202);
    let b = r.billing.unwrap();
    assert!((b.ai_credits - 0.2672).abs() < 1e-9);
    assert_eq!(b.harness_overhead_tokens, 2000);
    assert_eq!(b.overhead_source, OVERHEAD_REPORTED);
}

#[tokio::test]
async fn stops_dispatching_once_the_credit_cap_would_be_crossed() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(dir.path(), &format!("cat \"{}\"", fixture().display()));
    // Worst case per call: (1000 + 2000) * 0.1/1k + 200 * 0.5/1k = 0.4.
    let model = chat(dir.path(), cli, 0.9);
    let req = request("s", "u");
    model.complete(&req).await.unwrap();
    model.complete(&req).await.unwrap();
    // 2 * 0.2672 charged; another 0.4 would cross 0.9.
    let err = model.complete(&req).await.unwrap_err();
    assert!(
        matches!(err, ProviderError::BudgetExhausted { .. }),
        "{err}"
    );
    assert!(!err.retryable());
    assert_eq!(
        calls(dir.path()),
        2,
        "the refused call must not spawn the CLI"
    );
    let u = model.ledger().usage();
    assert!((u.reported - 0.5344).abs() < 1e-9);
    assert_eq!((u.calls, u.refused), (2, 1));
}

#[tokio::test]
async fn systematic_failures_stop_the_provider_without_further_spawns() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(
        dir.path(),
        "echo 'Error: You are not authenticated. Please log in.' >&2; exit 1",
    );
    let model = chat(dir.path(), cli, 10.0);
    let err = model.complete(&request("s", "u")).await.unwrap_err();
    assert!(
        matches!(&err, ProviderError::Config { message } if message.contains("not authenticated"))
    );
    let again = model.complete(&request("s", "u")).await.unwrap_err();
    assert!(matches!(again, ProviderError::DispatchRefused { .. }));
    assert_eq!(calls(dir.path()), 1);
    // The CLI never reached a model, so nothing stays charged.
    assert_eq!(model.ledger().usage().charged(), 0.0);
}

#[tokio::test]
async fn an_empty_answer_is_an_explicit_error_and_a_hung_cli_times_out() {
    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(
        dir.path(),
        r#"echo '{"type":"model.call_start","data":{}}'; echo '{"type":"result","exitCode":0}'"#,
    );
    let model = chat(dir.path(), cli, 10.0);
    let err = model.complete(&request("s", "u")).await.unwrap_err();
    assert!(matches!(err, ProviderError::Malformed { .. }), "{err}");
    // The model was called but reported no charge: the worst case is kept.
    assert!((model.ledger().usage().unsettled - 0.4).abs() < 1e-9);

    let dir = tempfile::tempdir().unwrap();
    let cli = fake_cli(dir.path(), "sleep 30");
    let mut c = CopilotConfig::new("m");
    c.program = cli;
    c.workdir_root = dir.path().join("work");
    c.timeout = Duration::from_millis(300);
    c.rates = CreditRates {
        input_per_1k: 0.1,
        output_per_1k: 0.5,
        overhead_tokens: 0,
    };
    let permit = DispatchPolicy {
        manifest_allows: true,
        operator_allows: true,
    }
    .permit("t", vec![])
    .unwrap();
    let model = CopilotCliChat::new(
        c,
        &permit,
        Arc::new(CreditLedger::new(10.0)),
        Arc::new(Semaphore::new(1)),
    )
    .unwrap();
    let err = model.complete(&request("s", "u")).await.unwrap_err();
    assert_eq!(err, ProviderError::Timeout);
    assert!(err.retryable());
}
