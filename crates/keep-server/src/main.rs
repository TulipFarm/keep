use std::sync::Arc;

use clap::parser::ValueSource;
use clap::{CommandFactory, FromArgMatches};
use keep_server::auth::{self, Auth};
use keep_server::config::{Cli, Command, StorageProfile, TokenCommand, api_token, check_bind};
use keep_server::http::ServeConfig;
use keep_server::mcp::KeepMcp;
use keep_server::service::WORKER_INTERVAL;
use keep_server::{HttpBackend, KeepService, ScopeDefaults, http, tls};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches)?;
    // Logs go to stderr only: in MCP mode stdout carries the protocol.
    match cli.command {
        Command::Mcp(args) if args.server.is_some() => {
            let url = args.server.as_deref().unwrap_or_default();
            // Only scope flags given explicitly override the daemon's defaults.
            let explicit = |id: &str| {
                matches!(
                    matches.value_source(id),
                    Some(ValueSource::CommandLine | ValueSource::EnvVariable)
                )
            };
            let defaults = ScopeDefaults {
                user: explicit("user").then(|| cli.service.user.clone()),
                project: explicit("project")
                    .then(|| cli.service.project.clone())
                    .flatten(),
            };
            let backend = HttpBackend::new(url, api_token(), args.server_ca.as_deref(), defaults)?;
            match backend.health().await {
                Ok(h) => eprintln!(
                    "keep: forwarding to {} (storage={}, version={})",
                    backend.base(),
                    h["storage"].as_str().unwrap_or("?"),
                    h["version"].as_str().unwrap_or("?"),
                ),
                Err(e) => eprintln!(
                    "keep: forwarding to {}, which is not answering yet ({})",
                    backend.base(),
                    e.code.as_str()
                ),
            }
            KeepMcp::new(Arc::new(backend)).serve_stdio().await
        }
        Command::Mcp(_) => {
            let service = Arc::new(KeepService::open(&cli.service).await?);
            announce(&cli.service.storage, &service);
            service.spawn_worker(WORKER_INTERVAL);
            if cli.service.storage == StorageProfile::InMemory {
                eprintln!(
                    "keep: note: in-memory storage lives in this process only. Memory is \
                     not shared with other agents and is lost on exit. To share it, run one \
                     `keep serve` and connect each agent with `keep mcp --server URL`, or \
                     use a postgres storage profile."
                );
            }
            KeepMcp::new(service).serve_stdio().await
        }
        Command::Serve(args) => {
            let admin = api_token();
            let auth = match &args.tokens_file {
                Some(path) => Auth::load(admin.as_deref(), path)?,
                None => Auth::from(admin),
            };
            let tls = match (&args.tls_cert, &args.tls_key) {
                (Some(cert), Some(key)) => Some(tls::server_config(cert, key)?),
                _ => None,
            };
            // Fail before opening storage.
            check_bind(
                &args.bind,
                !auth.is_open(),
                tls.is_some(),
                args.insecure_http,
            )
            .map_err(anyhow::Error::msg)?;
            if args.dev {
                keep_server::config::check_dev(&args.bind, auth.is_open())
                    .map_err(anyhow::Error::msg)?;
            }
            if auth.token_count() > 0 {
                eprintln!("keep: {} scoped tokens loaded", auth.token_count());
            }
            let service = Arc::new(KeepService::open(&cli.service).await?);
            announce(&cli.service.storage, &service);
            service.spawn_worker(WORKER_INTERVAL);
            let config = ServeConfig {
                bind: args.bind,
                auth,
                max_body_bytes: args.max_body_bytes,
                tls,
                insecure_http: args.insecure_http,
                dev: args.dev,
            };
            http::serve(service, config).await
        }
        Command::Token(TokenCommand::Hash) => {
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)?;
            let token = line.trim_end_matches(['\r', '\n']);
            anyhow::ensure!(!token.is_empty(), "no token on stdin");
            println!("{}", auth::hash_hex(token));
            Ok(())
        }
        Command::Healthcheck(args) => {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()?;
            let status = client.get(&args.url).send().await?.status();
            anyhow::ensure!(status.is_success(), "not ready: HTTP {status}");
            Ok(())
        }
        Command::FetchModel(args) => {
            let dir = keep_server::embedder::fetch_local(&args.dir).await?;
            eprintln!("keep: the local embedding model is in {}", dir.display());
            Ok(())
        }
        Command::Token(TokenCommand::New) => {
            let token = auth::new_token();
            println!("{token}\n{}", auth::hash_hex(&token));
            Ok(())
        }
    }
}

fn announce(storage: &StorageProfile, service: &KeepService) {
    eprintln!(
        "keep: storage={} default scope={}",
        storage.as_str(),
        service.defaults().label()
    );
}
