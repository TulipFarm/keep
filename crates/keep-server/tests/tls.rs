//! HTTPS: a self-signed certificate, a client that trusts it, one that
//! does not, and the MCP proxy's backend over TLS.

use std::path::PathBuf;
use std::sync::Arc;

use keep_server::http::router;
use keep_server::tls::{TlsListener, server_config};
use keep_server::{Backend, HttpBackend, KeepService, Scope, ScopeDefaults, ServiceOptions};
use serde_json::json;

struct Pems {
    dir: PathBuf,
    cert: PathBuf,
    key: PathBuf,
    other_key: PathBuf,
}

impl Drop for Pems {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn pems() -> Pems {
    let dir =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("tls-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ck =
        rcgen::generate_simple_self_signed(vec!["localhost".into(), "127.0.0.1".into()]).unwrap();
    let other = rcgen::KeyPair::generate().unwrap();
    let p = Pems {
        cert: dir.join("cert.pem"),
        key: dir.join("key.pem"),
        other_key: dir.join("other.pem"),
        dir,
    };
    std::fs::write(&p.cert, ck.cert.pem()).unwrap();
    std::fs::write(&p.key, ck.signing_key.serialize_pem()).unwrap();
    std::fs::write(&p.other_key, other.serialize_pem()).unwrap();
    p
}

#[tokio::test]
async fn https_handshake_with_a_self_signed_certificate() {
    let p = pems();
    assert!(
        server_config(&p.cert, &p.other_key).is_err(),
        "mismatched key"
    );
    let config = server_config(&p.cert, &p.key).unwrap();
    let tcp = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = tcp.local_addr().unwrap();
    let svc = KeepService::in_memory(ServiceOptions::new(
        Scope::new("alex", Some("api".into())).unwrap(),
    ));
    let app = router(Arc::new(svc), Some("secret".to_owned()), 1 << 20);
    let server = tokio::spawn(async move {
        axum::serve(TlsListener::new(tcp, config).unwrap(), app)
            .await
            .unwrap();
    });

    let url = format!("https://localhost:{}/v1/health", addr.port());
    let root = reqwest::Certificate::from_pem(&std::fs::read(&p.cert).unwrap()).unwrap();
    let trusting = reqwest::Client::builder()
        .add_root_certificate(root)
        .resolve("localhost", addr)
        .build()
        .unwrap();
    let h: serde_json::Value = trusting
        .get(&url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(h["status"], "ok");

    // Not trusted: the handshake fails.
    let plain = reqwest::Client::builder()
        .resolve("localhost", addr)
        .build()
        .unwrap();
    assert!(plain.get(&url).send().await.is_err());
    // Plain HTTP to the TLS port gets nothing usable back.
    let http = format!("http://127.0.0.1:{}/v1/health", addr.port());
    assert!(plain.get(&http).send().await.is_err());
    // A failed handshake does not stop the listener.
    let h: serde_json::Value = trusting
        .get(&url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(h["status"], "ok");

    // The MCP proxy's backend trusts the same file through --server-ca.
    let backend = HttpBackend::new(
        &format!("https://127.0.0.1:{}", addr.port()),
        Some("secret".into()),
        Some(&p.cert),
        ScopeDefaults::default(),
    )
    .unwrap();
    backend
        .call(
            "memory_write",
            json!({ "path": "/tls.md", "text": "- over https\n" }),
        )
        .await
        .unwrap();
    let ctx = backend.call("memory_context", json!({})).await.unwrap();
    assert!(ctx["text"].as_str().unwrap().contains("over https"));
    server.abort();
}

#[test]
fn public_binds_need_tls_or_an_explicit_opt_out() {
    let keep = env!("CARGO_BIN_EXE_keep");
    let run = |extra: &[&str]| {
        let mut cmd = std::process::Command::new(keep);
        for (k, _) in std::env::vars() {
            if k.starts_with("KEEP_") {
                cmd.env_remove(k);
            }
        }
        cmd.env("KEEP_API_TOKEN", "secret")
            .env("KEEP_EMBEDDER", "hash")
            .env(
                "KEEP_DATA_DIR",
                concat!(env!("CARGO_TARGET_TMPDIR"), "/keep-data"),
            )
            .args(["serve", "--bind", "0.0.0.0:0"])
            .args(extra)
            .stdin(std::process::Stdio::null())
            .output()
            .unwrap()
    };
    let out = run(&[]);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--insecure-http"), "{err}");
    let out = run(&["--tls-cert", "missing.pem"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--tls-key"));
}
