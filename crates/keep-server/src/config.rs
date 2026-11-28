//! Command-line and environment configuration.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use keep_knowledge::engine::JobRetention;

use crate::budget::{BudgetConfig, BudgetPer};

#[derive(Debug, Parser)]
#[command(
    name = "keep",
    version,
    about = "Serve Keep's Memory and Knowledge to coding agents over MCP (stdio) or HTTP"
)]
pub struct Cli {
    #[command(flatten)]
    pub service: ServiceArgs,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Speak MCP over stdin/stdout. This is how coding agents connect.
    Mcp(McpArgs),
    /// Serve the JSON API over HTTP.
    Serve(HttpArgs),
    /// Make and hash bearer tokens for a tokens file.
    #[command(subcommand)]
    Token(TokenCommand),
    /// Exit 0 if a running `keep serve` is ready, 1 if not. For container
    /// health checks.
    Healthcheck(HealthcheckArgs),
    /// Download the pinned local embedding model into the model cache and
    /// verify its SHA-256 digests, so a later `--embedder local` start
    /// never touches the network. For baking the model into an image.
    FetchModel(FetchModelArgs),
}

#[derive(Debug, Args)]
pub struct FetchModelArgs {
    /// The model cache to fill.
    #[arg(long, env = "KEEP_MODEL_CACHE")]
    pub dir: PathBuf,
}

#[derive(Debug, Args)]
pub struct HealthcheckArgs {
    /// The server's readiness endpoint.
    #[arg(long, default_value = "http://127.0.0.1:7357/v1/health/ready")]
    pub url: String,
}

#[derive(Debug, Subcommand)]
pub enum TokenCommand {
    /// Read a token from stdin and print its SHA-256, for a tokens file.
    Hash,
    /// Print a new random token on the first line and its SHA-256 on the
    /// second.
    New,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum StorageProfile {
    /// Everything in this process; lost on exit. For tests and trying it out.
    #[default]
    InMemory,
    /// Postgres for records, pgvector for vectors.
    PostgresVector,
    /// Postgres for records, Qdrant as a rebuildable vector projection.
    PostgresQdrant,
}

impl StorageProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            StorageProfile::InMemory => "in-memory",
            StorageProfile::PostgresVector => "postgres-vector",
            StorageProfile::PostgresQdrant => "postgres-qdrant",
        }
    }
}

#[derive(Clone, Debug, Args)]
pub struct ServiceArgs {
    /// Where Memory and Knowledge are stored.
    #[arg(long, env = "KEEP_STORAGE", value_enum, default_value_t, global = true)]
    pub storage: StorageProfile,
    /// Postgres connection URL for the postgres-* profiles.
    #[arg(long, env = "KEEP_POSTGRES_URL", hide_env_values = true, global = true)]
    pub postgres_url: Option<String>,
    /// Postgres schema for every table (created if missing). Defaults to the
    /// connection's search path.
    #[arg(long, env = "KEEP_POSTGRES_SCHEMA", global = true)]
    pub postgres_schema: Option<String>,
    /// Extra CA certificate (PEM) to trust for Postgres TLS, for a database
    /// behind a private CA. Same as `sslrootcert` in the URL.
    #[arg(long, env = "KEEP_POSTGRES_CA", global = true)]
    pub postgres_ca: Option<PathBuf>,
    /// Qdrant URL for the postgres-qdrant profile.
    #[arg(long, env = "KEEP_QDRANT_URL", global = true)]
    pub qdrant_url: Option<String>,
    #[arg(
        long,
        env = "KEEP_QDRANT_API_KEY",
        hide_env_values = true,
        global = true
    )]
    pub qdrant_api_key: Option<String>,
    /// Extra CA certificate (PEM) to trust for an `https://` Qdrant URL.
    #[arg(long, env = "KEEP_QDRANT_CA", global = true)]
    pub qdrant_ca: Option<PathBuf>,
    /// Vector collection that holds Knowledge's points.
    #[arg(
        long,
        env = "KEEP_COLLECTION",
        default_value = "knowledge",
        global = true
    )]
    pub collection: String,
    /// Default user for calls that do not name one.
    #[arg(long, env = "KEEP_USER", default_value = "local", global = true)]
    pub user: String,
    /// Default project for calls that do not name one.
    #[arg(long, env = "KEEP_PROJECT", global = true)]
    pub project: Option<String>,
    /// Name recorded as the harness on every Memory revision, such as
    /// `claude-code` or `codex`.
    #[arg(long, env = "KEEP_AGENT", global = true)]
    pub agent: Option<String>,
    /// Tokenizer used to count budgets: tiktoken-o200k-base,
    /// tiktoken-cl100k-base or approx-v1.
    #[arg(
        long,
        env = "KEEP_TOKENIZER",
        default_value = "tiktoken-o200k-base",
        global = true
    )]
    pub tokenizer: String,
    /// Refuse placing files in the always-included You category
    /// (`/profile.md`, `/you/<slug>.md`).
    #[arg(long, env = "KEEP_NO_PROFILE_WRITES", global = true)]
    pub no_profile_writes: bool,
    /// Embedder for Knowledge's vectors: `local` (an on-device ONNX model,
    /// the default when compiled in) or `hash` (offline word hashing, the
    /// default otherwise). A persisted collection remembers its embedder.
    #[arg(long, env = "KEEP_EMBEDDER", value_enum, global = true)]
    pub embedder: Option<EmbedderKind>,
    /// Directory for this server's own files, such as the receipt key and
    /// the model cache. Defaults to the platform's data directory.
    #[arg(long, env = "KEEP_DATA_DIR", global = true)]
    pub data_dir: Option<PathBuf>,
    /// Cap the Knowledge evidence (tokens returned by knowledge_search and
    /// knowledge_fetch) each caller may draw per window. Off by default.
    #[arg(long, env = "KEEP_EVIDENCE_BUDGET_TOKENS", global = true)]
    pub evidence_budget_tokens: Option<usize>,
    /// Length of the evidence budget's rolling window, in seconds.
    #[arg(
        long,
        env = "KEEP_EVIDENCE_BUDGET_WINDOW_SECS",
        default_value_t = 3600,
        global = true
    )]
    pub evidence_budget_window_secs: u64,
    /// Whom the evidence budget applies to: each bearer token, or each
    /// session (each `keep mcp` process).
    #[arg(
        long,
        env = "KEEP_EVIDENCE_BUDGET_PER",
        value_enum,
        default_value = "token",
        global = true
    )]
    pub evidence_budget_per: BudgetPer,
    /// Milliseconds an ingest spends on its own source's extraction and
    /// projection jobs before answering; the background worker finishes the
    /// rest. 0 leaves all of them to the worker.
    #[arg(
        long,
        env = "KEEP_INGEST_JOB_BUDGET_MS",
        default_value_t = 5000,
        global = true
    )]
    pub ingest_job_budget_ms: u64,
    /// Milliseconds the background worker sleeps between job batches, so
    /// backlog work yields CPU to requests. 0 runs batches back to back.
    #[arg(long, env = "KEEP_JOB_PAUSE_MS", default_value_t = 200, global = true)]
    pub job_pause_ms: u64,
    /// Days a succeeded background job is kept before the worker deletes
    /// it. 0 keeps them forever.
    #[arg(
        long,
        env = "KEEP_SUCCEEDED_JOB_RETENTION_DAYS",
        default_value_t = 7,
        global = true
    )]
    pub succeeded_job_retention_days: u32,
    /// Days a failed background job, and its last error, is kept before the
    /// worker deletes it. 0 keeps them forever.
    #[arg(
        long,
        env = "KEEP_FAILED_JOB_RETENTION_DAYS",
        default_value_t = 30,
        global = true
    )]
    pub failed_job_retention_days: u32,
    /// Model sessions embedding at once with `--embedder local`. Sessions
    /// times threads is the most cores embedding uses. Defaults to fit half
    /// the available cores.
    #[arg(long, env = "KEEP_ONNX_SESSIONS", global = true, value_parser = clap::value_parser!(u32).range(1..))]
    pub onnx_sessions: Option<u32>,
    /// ONNX Runtime threads per session with `--embedder local`.
    #[arg(long, env = "KEEP_ONNX_THREADS", global = true, value_parser = clap::value_parser!(u32).range(1..))]
    pub onnx_threads: Option<u32>,
    #[command(flatten)]
    pub model: crate::model::ModelArgs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EmbedderKind {
    /// Deterministic word hashing, 256 dimensions. No model, no download.
    Hash,
    /// BAAI/bge-small-en-v1.5 on device, 384 dimensions. Needs the
    /// `local-embeddings` build feature.
    Local,
}

impl EmbedderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EmbedderKind::Hash => "hash",
            EmbedderKind::Local => "local",
        }
    }
}

impl ServiceArgs {
    /// The embedder asked for, or the build's default.
    pub fn embedder_kind(&self) -> EmbedderKind {
        self.embedder
            .unwrap_or(if cfg!(feature = "local-embeddings") {
                EmbedderKind::Local
            } else {
                EmbedderKind::Hash
            })
    }

    /// How long finished background jobs are kept.
    pub fn job_retention(&self) -> JobRetention {
        let days = |d: u32| (d > 0).then(|| chrono::Duration::days(d.into()));
        JobRetention {
            succeeded: days(self.succeeded_job_retention_days),
            failed: days(self.failed_job_retention_days),
        }
    }

    /// The evidence budget, if one is set.
    pub fn evidence_budget(&self) -> anyhow::Result<Option<BudgetConfig>> {
        let Some(tokens) = self.evidence_budget_tokens else {
            return Ok(None);
        };
        anyhow::ensure!(tokens > 0, "--evidence-budget-tokens must be at least 1");
        anyhow::ensure!(
            self.evidence_budget_window_secs > 0,
            "--evidence-budget-window-secs must be at least 1"
        );
        Ok(Some(BudgetConfig {
            tokens,
            window: std::time::Duration::from_secs(self.evidence_budget_window_secs),
            per: self.evidence_budget_per,
        }))
    }

    /// `--data-dir`, or the platform default.
    pub fn data_dir(&self) -> anyhow::Result<PathBuf> {
        match &self.data_dir {
            Some(d) => Ok(d.clone()),
            None => default_data_dir().ok_or_else(|| {
                anyhow::anyhow!("cannot find a data directory: set KEEP_DATA_DIR or --data-dir")
            }),
        }
    }

    /// `KEEP_MODEL_CACHE`, or `models` in the data directory.
    pub fn model_cache(&self) -> anyhow::Result<PathBuf> {
        match std::env::var_os("KEEP_MODEL_CACHE").filter(|v| !v.is_empty()) {
            Some(d) => Ok(PathBuf::from(d)),
            None => Ok(self.data_dir()?.join("models")),
        }
    }
}

/// `$XDG_DATA_HOME/keep`, `~/Library/Application Support/keep` on macOS,
/// `%APPDATA%\keep` on Windows, else `~/.local/share/keep`.
pub fn default_data_dir() -> Option<PathBuf> {
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if let Some(x) = env("XDG_DATA_HOME") {
        return Some(x.join("keep"));
    }
    if cfg!(windows) {
        return env("APPDATA").map(|a| a.join("keep"));
    }
    let home = env("HOME")?;
    Some(if cfg!(target_os = "macos") {
        home.join("Library")
            .join("Application Support")
            .join("keep")
    } else {
        home.join(".local").join("share").join("keep")
    })
}

#[derive(Clone, Debug, Default, Args)]
pub struct McpArgs {
    /// Forward every tool call to a running `keep serve` at this URL, such
    /// as `http://127.0.0.1:7357`, instead of opening storage here. Agents
    /// connected to one daemon share its Memory and Knowledge. The bearer
    /// token comes from KEEP_API_TOKEN.
    #[arg(long, env = "KEEP_SERVER_URL")]
    pub server: Option<String>,
    /// PEM file of a CA certificate to trust for an `https://` server with
    /// a private certificate.
    #[arg(long, env = "KEEP_SERVER_CA", requires = "server")]
    pub server_ca: Option<std::path::PathBuf>,
}

#[derive(Clone, Debug, Args)]
pub struct HttpArgs {
    /// Address to listen on. Anything but loopback needs KEEP_API_TOKEN or
    /// a tokens file.
    #[arg(long, env = "KEEP_BIND", default_value = "127.0.0.1:7357")]
    pub bind: SocketAddr,
    /// TOML file of hashed bearer tokens, each limited to some users,
    /// projects and permissions. KEEP_API_TOKEN stays a full-access
    /// administrator token.
    #[arg(long, env = "KEEP_TOKENS_FILE")]
    pub tokens_file: Option<std::path::PathBuf>,
    /// PEM certificate chain for HTTPS. Needs --tls-key.
    #[arg(long, env = "KEEP_TLS_CERT", requires = "tls_key")]
    pub tls_cert: Option<std::path::PathBuf>,
    /// PEM private key for HTTPS. Needs --tls-cert.
    #[arg(long, env = "KEEP_TLS_KEY", requires = "tls_cert")]
    pub tls_key: Option<std::path::PathBuf>,
    /// Allow plain HTTP on a non-loopback address, such as behind a
    /// TLS-terminating reverse proxy on a private network. Tokens then
    /// cross the network in the clear.
    #[arg(long, env = "KEEP_INSECURE_HTTP")]
    pub insecure_http: bool,
    /// Largest accepted request body, in bytes.
    #[arg(long, env = "KEEP_MAX_BODY_BYTES", default_value_t = 1 << 20)]
    pub max_body_bytes: usize,
}

/// The bearer token comes from the environment only, so it never shows in
/// process listings.
pub const TOKEN_ENV: &str = "KEEP_API_TOKEN";

pub fn api_token() -> Option<String> {
    std::env::var(TOKEN_ENV)
        .ok()
        .filter(|t| !t.trim().is_empty())
}

/// Refuse to expose the API beyond this machine without tokens, or
/// without TLS unless plain HTTP is explicitly allowed. Returns a warning
/// to print when plain HTTP leaves the machine.
pub fn check_bind(
    bind: &SocketAddr,
    authenticated: bool,
    tls: bool,
    insecure_http: bool,
) -> Result<Option<String>, String> {
    if bind.ip().is_loopback() {
        return Ok(None);
    }
    if !authenticated {
        return Err(format!(
            "refusing to listen on {bind} without bearer tokens: set {TOKEN_ENV} \
             or --tokens-file, or bind to 127.0.0.1"
        ));
    }
    if tls {
        return Ok(None);
    }
    if insecure_http {
        return Ok(Some(format!(
            "serving plain HTTP on {bind}: bearer tokens and memory cross the network \
             unencrypted. Use --tls-cert and --tls-key, or a TLS-terminating reverse proxy."
        )));
    }
    Err(format!(
        "refusing plain HTTP on {bind}: set --tls-cert and --tls-key, or pass \
         --insecure-http (for example behind a TLS-terminating reverse proxy)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_loopback_binds_need_a_token() {
        let lo: SocketAddr = "127.0.0.1:7357".parse().unwrap();
        let lo6: SocketAddr = "[::1]:7357".parse().unwrap();
        let any: SocketAddr = "0.0.0.0:7357".parse().unwrap();
        assert_eq!(check_bind(&lo, false, false, false), Ok(None));
        assert_eq!(check_bind(&lo6, false, false, false), Ok(None));
        assert!(check_bind(&any, false, true, true).is_err());
        assert!(check_bind(&any, true, false, false).is_err());
        assert_eq!(check_bind(&any, true, true, false), Ok(None));
        assert!(check_bind(&any, true, false, true).unwrap().is_some());
    }

    #[test]
    fn flags_parse_with_defaults() {
        let cli = Cli::try_parse_from(["keep", "serve", "--project", "api"]).unwrap();
        assert_eq!(cli.service.project.as_deref(), Some("api"));
        assert!(matches!(cli.command, Command::Serve(_)));
        let cli = Cli::try_parse_from(["keep", "--storage", "postgres-qdrant", "mcp"]).unwrap();
        assert_eq!(cli.service.storage, StorageProfile::PostgresQdrant);
        let cli = Cli::try_parse_from(["keep", "mcp", "--server", "http://127.0.0.1:1"]).unwrap();
        let Command::Mcp(m) = cli.command else {
            panic!("mcp")
        };
        assert_eq!(m.server.as_deref(), Some("http://127.0.0.1:1"));
    }

    #[test]
    fn job_retention_is_in_days_and_zero_keeps_forever() {
        let cli = Cli::try_parse_from(["keep", "serve"]).unwrap();
        assert_eq!(cli.service.job_retention(), JobRetention::default());
        let cli = Cli::try_parse_from([
            "keep",
            "serve",
            "--succeeded-job-retention-days",
            "0",
            "--failed-job-retention-days",
            "2",
        ])
        .unwrap();
        assert_eq!(
            cli.service.job_retention(),
            JobRetention {
                succeeded: None,
                failed: Some(chrono::Duration::days(2)),
            }
        );
    }
}
