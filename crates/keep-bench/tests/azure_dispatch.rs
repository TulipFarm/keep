//! Azure-style dispatch end to end through the binary, against a loopback
//! mock of the Azure OpenAI v1 endpoint. No external endpoint is contacted.

mod common;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Output};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const SECRET: &str = "azure-test-key-must-never-be-written";
const ANSWER: &str = "gpt-4o-mini-2024-07-18";
const JUDGE: &str = "gpt-4o-2024-08-06";

/// Serve chat completions forever on loopback. Each reply reports the
/// `model` chosen by `served` from the requested deployment.
fn mock_azure(served: fn(&str) -> &'static str) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let count = Arc::new(AtomicUsize::new(0));
    let seen = count.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut raw = Vec::new();
            let mut buf = [0u8; 65536];
            loop {
                let n = s.read(&mut buf).unwrap_or(0);
                raw.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&raw);
                if let Some(end) = text.find("\r\n\r\n") {
                    let len = text[..end]
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if raw.len() >= end + 4 + len {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            seen.fetch_add(1, Ordering::SeqCst);
            let text = String::from_utf8_lossy(&raw).to_string();
            assert!(text.starts_with("POST /openai/v1/chat/completions "));
            assert!(
                text.to_ascii_lowercase()
                    .contains(&format!("api-key: {SECRET}"))
            );
            let deployment = if text.contains(r#""model":"judge-deploy""#) {
                "judge-deploy"
            } else {
                "answer-deploy"
            };
            let content = if deployment == "judge-deploy" {
                "CORRECT"
            } else {
                "unknown"
            };
            let body = format!(
                r#"{{"model":"{}","system_fingerprint":"fp_mock","choices":[{{"message":{{"content":"{content}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":100,"completion_tokens":2}}}}"#,
                served(deployment)
            );
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    (origin, count)
}

fn azure_model(model: &str, served: &str, deployment_env: &str) -> String {
    format!(
        "provider = \"openai-compatible\"\nmodel = \"{model}\"\nrevision = \"{}\"\n\
         tokenizer = \"tiktoken-o200k-base\"\nbase_url_env = \"KEEP_TEST_AZ_ENDPOINT\"\n\
         api_key_env = \"KEEP_TEST_AZ_KEY\"\napi_style = \"azure-v1\"\nauth_header = \"api-key\"\n\
         deployment_env = \"{deployment_env}\"\nserved_model = \"{served}\"\n\
         input_usd_per_mtok = 1.0\noutput_usd_per_mtok = 2.0",
        &served[served.len() - 10..]
    )
}

/// The mock LongMemEval fixture, one seed, three arms, pointed at Azure.
fn manifest(dir: &Path) -> std::path::PathBuf {
    let text = common::fixture_manifest("longmemeval-mock.toml");
    let head = &text[..text.find("[[arms]]").unwrap()];
    let text = head
        .replacen(
            "provider = \"mock\"\nmodel = \"mock-extractive\"\nrevision = \"v1\"\ntokenizer = \"approx-v1\"",
            &azure_model("gpt-4o-mini", ANSWER, "KEEP_TEST_AZ_ANSWER"),
            1,
        )
        .replacen(
            "provider = \"mock\"\nmodel = \"mock-f1-judge\"\nrevision = \"v1\"\ntokenizer = \"approx-v1\"",
            &azure_model("gpt-4o", JUDGE, "KEEP_TEST_AZ_JUDGE"),
            1,
        )
        .replace("seeds = [1, 2, 3]", "seeds = [1]")
        .replace("max_model_calls = 0", "max_model_calls = 1000")
        .replace("max_input_tokens = 0", "max_input_tokens = 10000000")
        .replace("max_output_tokens = 0\nmax_cost", "max_output_tokens = 1000000\nmax_cost")
        .replace("max_cost_usd = 0.0", "max_cost_usd = 5.0")
        + "[dispatch]\nallow_model_calls = true\n\n\
           [[arms]]\nname = \"no-memory\"\nkind = \"no-memory\"\n\n\
           [[arms]]\nname = \"full-context\"\nkind = \"full-context\"\n\n\
           [[arms]]\nname = \"bm25\"\nkind = \"bm25\"\nbudget_tokens = 256\ntop_k = 8\n";
    let lm = common::write_manifest(dir, &text);
    assert!(
        lm.manifest.dispatch_problems().is_empty(),
        "{:?}",
        lm.manifest.dispatch_problems()
    );
    lm.path
}

fn run(dir: &Path, origin: &str, extra: &[&str]) -> Output {
    let m = manifest(dir);
    let mut args = vec![
        "run",
        m.to_str().unwrap(),
        "--allow-model-calls",
        "--results",
        "results",
        "--spend-ledger",
        "spend.jsonl",
    ];
    args.extend_from_slice(extra);
    Command::new(env!("CARGO_BIN_EXE_keep-bench"))
        .args(&args)
        .current_dir(dir)
        .env("KEEP_TEST_AZ_ENDPOINT", format!("{origin}/openai/v1/"))
        .env("KEEP_TEST_AZ_KEY", SECRET)
        .env("KEEP_TEST_AZ_ANSWER", "answer-deploy")
        .env("KEEP_TEST_AZ_JUDGE", "judge-deploy")
        .output()
        .unwrap()
}

fn run_json(dir: &Path) -> serde_json::Value {
    let runs: Vec<_> = std::fs::read_dir(dir.join("results"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(runs.len(), 1);
    serde_json::from_slice(&std::fs::read(runs[0].join("run.json")).unwrap()).unwrap()
}

/// Every file under `dir`, recursively, must be free of the key.
fn assert_key_absent(dir: &Path) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_dir() {
            assert_key_absent(&p);
        } else {
            let bytes = std::fs::read(&p).unwrap();
            assert!(
                !String::from_utf8_lossy(&bytes).contains(SECRET),
                "key leaked into {}",
                p.display()
            );
        }
    }
}

fn spend_lines(dir: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(dir.join("spend.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn pinned_azure_run_records_endpoint_and_settles_spend() {
    let dir = tempfile::tempdir().unwrap();
    let (origin, count) = mock_azure(|d| if d == "judge-deploy" { JUDGE } else { ANSWER });
    let out = run(dir.path(), &origin, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{stderr}");
    assert!(count.load(Ordering::SeqCst) > 0);
    assert!(!stderr.contains(SECRET) && !stdout.contains(SECRET));

    let meta = run_json(dir.path());
    assert!(meta.get("aborted").is_none());
    let a = &meta["answer_endpoint"];
    assert_eq!(a["api_style"], "azure-v1");
    assert_eq!(a["host"], "127.0.0.1");
    assert_eq!(a["deployment"], "answer-deploy");
    assert_eq!(a["served_model"], ANSWER);
    assert_eq!(a["served_models"], serde_json::json!([ANSWER]));
    assert_eq!(a["system_fingerprints"], serde_json::json!(["fp_mock"]));
    assert_eq!(
        meta["judge_endpoint"]["served_models"],
        serde_json::json!([JUDGE])
    );
    assert_key_absent(dir.path());

    let lines = spend_lines(dir.path());
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert_eq!(lines[0]["event"], "reserve");
    assert_eq!(lines[0]["reserved_usd"], 5.0);
    assert_eq!(lines[1]["event"], "settle");
    let spent = lines[1]["spent_usd"].as_f64().unwrap();
    let calls = count.load(Ordering::SeqCst) as f64;
    assert!(
        (spent - calls * 104.0 / 1e6).abs() < 1e-9,
        "{spent} for {calls} calls"
    );
    assert!(stdout.contains("settled"), "{stdout}");
}

#[test]
fn a_wrong_served_model_aborts_the_run_after_one_call() {
    let dir = tempfile::tempdir().unwrap();
    // The judge deployment serves a different gpt-4o version.
    let (origin, count) = mock_azure(|d| {
        if d == "judge-deploy" {
            "gpt-4o-2024-11-20"
        } else {
            ANSWER
        }
    });
    let out = run(dir.path(), &origin, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("run aborted"), "{stderr}");
    assert!(stderr.contains("gpt-4o-2024-11-20"), "{stderr}");
    // One answer and one judge call, then nothing more is sent.
    assert_eq!(count.load(Ordering::SeqCst), 2);

    let meta = run_json(dir.path());
    assert!(meta["aborted"].as_str().unwrap().contains("judge model"));
    let runs: Vec<_> = std::fs::read_dir(dir.path().join("results"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    let records = std::fs::read_to_string(runs[0].join("records.jsonl")).unwrap();
    assert!(records.contains("run aborted before this call"));
    assert_key_absent(dir.path());
    // The spend reservation is settled even though the run failed.
    let lines = spend_lines(dir.path());
    assert_eq!(lines.last().unwrap()["event"], "settle");

    // An aborted run cannot be compared.
    let cmp = Command::new(env!("CARGO_BIN_EXE_keep-bench"))
        .args([
            "compare",
            "--run",
            runs[0].to_str().unwrap(),
            "--candidate",
            "bm25",
            "--against",
            "no-memory",
        ])
        .output()
        .unwrap();
    assert!(!cmp.status.success());
    assert!(String::from_utf8_lossy(&cmp.stderr).contains("aborted"));
}

#[test]
fn the_spend_cap_refuses_a_run_before_any_call() {
    let dir = tempfile::tempdir().unwrap();
    let (origin, count) = mock_azure(|_| ANSWER);
    let out = run(dir.path(), &origin, &["--spend-cap-usd", "4.99"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(stderr.contains("spend cap refused"), "{stderr}");
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(!dir.path().join("results").exists());
}

#[test]
fn max_run_usd_lowers_the_enforced_cap_and_the_reservation() {
    // A smoke run below the manifest's $5 budget fits a smaller total cap.
    let dir = tempfile::tempdir().unwrap();
    let (origin, _) = mock_azure(|d| if d == "judge-deploy" { JUDGE } else { ANSWER });
    let out = run(
        dir.path(),
        &origin,
        &["--spend-cap-usd", "1", "--max-run-usd", "1"],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(spend_lines(dir.path())[0]["reserved_usd"], 1.0);

    // The run's own budget stops calls at the lowered cap: each mock call
    // reserves $0.000104, so $0.0003 admits two.
    let dir = tempfile::tempdir().unwrap();
    let (origin, count) = mock_azure(|d| if d == "judge-deploy" { JUDGE } else { ANSWER });
    run(dir.path(), &origin, &["--max-run-usd", "0.0003"]);
    assert_eq!(count.load(Ordering::SeqCst), 2);
    let lines = spend_lines(dir.path());
    assert_eq!(lines[0]["reserved_usd"], 0.0003);
    assert!(lines[1]["spent_usd"].as_f64().unwrap() <= 0.0003);

    // It never raises the manifest's cap.
    let dir = tempfile::tempdir().unwrap();
    let (origin, _) = mock_azure(|d| if d == "judge-deploy" { JUDGE } else { ANSWER });
    run(dir.path(), &origin, &["--max-run-usd", "50"]);
    assert_eq!(spend_lines(dir.path())[0]["reserved_usd"], 5.0);
}
