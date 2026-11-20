//! Helpers shared by the integration tests.
#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::Duration;

use serde_json::{Value, json};

/// A `keep mcp` child process driven over stdin/stdout.
pub struct Session {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    next_id: u64,
}

impl Session {
    /// Run `keep` with `args` and extra environment, with every `KEEP_*`
    /// variable of the test's own environment removed.
    pub fn spawn(args: &[&str], envs: &[(&str, &str)]) -> Self {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_keep"));
        for (k, _) in std::env::vars() {
            if k.starts_with("KEEP_") {
                cmd.env_remove(k);
            }
        }
        // Never download a model in tests.
        cmd.env("KEEP_EMBEDDER", "hash");
        cmd.env(
            "KEEP_DATA_DIR",
            concat!(env!("CARGO_TARGET_TMPDIR"), "/keep-data"),
        );
        let mut child = cmd
            .args(args)
            .envs(envs.iter().copied())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn keep");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, lines) = channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Self {
            child,
            stdin,
            lines,
            next_id: 1,
        }
    }

    pub fn send(&mut self, msg: &Value) {
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
    }

    pub fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        loop {
            let line = self
                .lines
                .recv_timeout(Duration::from_secs(30))
                .unwrap_or_else(|_| panic!("no response to {method}"));
            let msg: Value = serde_json::from_str(&line).expect("stdout carries only JSON-RPC");
            if msg["id"] == json!(id) {
                assert!(msg.get("error").is_none(), "{method} failed: {msg}");
                return msg["result"].clone();
            }
        }
    }

    pub fn call(&mut self, tool: &str, args: Value) -> Value {
        let result = self.call_raw(tool, args);
        assert_ne!(result["isError"], json!(true), "{tool} failed: {result}");
        result["structuredContent"].clone()
    }

    /// A tool call that should fail; returns its `error` object.
    pub fn call_err(&mut self, tool: &str, args: Value) -> Value {
        let result = self.call_raw(tool, args);
        assert_eq!(
            result["isError"],
            json!(true),
            "{tool} should fail: {result}"
        );
        result["structuredContent"]["error"].clone()
    }

    pub fn call_raw(&mut self, tool: &str, args: Value) -> Value {
        self.request("tools/call", json!({ "name": tool, "arguments": args }))
    }

    /// The MCP handshake.
    pub fn initialize(&mut self) -> Value {
        let init = self.request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "test", "version": "0" }
            }),
        );
        self.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
        init
    }

    pub fn tool_names(&mut self) -> Vec<String> {
        let tools = self.request("tools/list", json!({}));
        tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_owned())
            .collect()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
