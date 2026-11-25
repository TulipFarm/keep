# Agent and contributor guide

## Commands

```sh
cargo fmt --all                                         # format
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy -p keep-server --all-targets --locked --features local-embeddings -- -D warnings
cargo test --workspace --locked                         # offline
cargo run -p keep-bench -- dry-run fixtures/manifests/longmemeval-mock.toml
cargo run -p keep-server -- --project demo mcp          # MCP server, in-memory
cargo run -p keep-server -- --project demo serve        # HTTP on 127.0.0.1:7357
cargo run -p keep-server -- --project demo mcp --server http://127.0.0.1:7357
```

Server tests force `KEEP_EMBEDDER=hash` and a data directory under `target/`,
so they never download a model or touch your real data directory.

Database integration tests are opt-in:

```sh
docker compose --profile postgres-qdrant up -d
KEEP_TEST_POSTGRES_URL='postgres://keep:keep-local-only@127.0.0.1:15433/keep' \
KEEP_TEST_QDRANT_URL='http://127.0.0.1:16333' \
cargo test -p keep-postgres -p keep-qdrant -p keep-ledger -p keep-knowledge -p keep-bench \
  -p keep-server
```

Without those variables the tests skip. Knowledge tests create their own
`keep_knowledge_test_*` schemas and `knowledge_test_*` Qdrant collections and
drop them afterwards; server tests use `keep_server_test_*` for both.

Both storage profiles run the same suites. Server tests also cut Postgres
and Qdrant through a local forwarder to check that outages are reported
honestly and recovered; they skip for remote or TLS endpoints.

To run the suites against TLS-only databases, or real remote ones, set
`KEEP_TEST_POSTGRES_URL` with `sslmode=verify-full` (and `sslrootcert=` or
`KEEP_TEST_POSTGRES_CA` for a private CA), an `https://`
`KEEP_TEST_QDRANT_URL`, `KEEP_TEST_QDRANT_API_KEY` and `KEEP_TEST_QDRANT_CA`.
`scripts/remote-profile-test.sh` starts throwaway TLS databases on ports
25433 and 26333 and does this for you. `scripts/compose-smoke.sh` builds the
image and checks both `deploy/compose.yaml` profiles end to end.

Release with `scripts/release.sh [auto|patch|minor|major|X.Y.Z]`; see
"Releases" in `README.md`. Don't tag or edit `CHANGELOG.md` by hand.

## Layout

- `crates/keep-core` — domain types; no I/O.
- `crates/keep-providers` — model, embedding and tokenizer traits, mocks,
  OpenAI-compatible client, budgets.
- `crates/keep-store` — index contracts and in-memory implementations.
- `crates/keep-postgres`, `crates/keep-qdrant` — storage adapters.
- `crates/keep-memory` — the Memory primitive: codec, operations, in-memory
  ledger, context assembly, export/import and the ledger conformance suite.
- `crates/keep-ledger` — Postgres Memory ledger; migrations are prefixed
  `memory_`.
- `crates/keep-knowledge` — the Knowledge primitive: model, ingestion,
  retrieval, book index, forgetting, in-memory and Postgres ledgers
  (migrations prefixed `knowledge_`). See `docs/knowledge.md`.
- `crates/keep-bench` — harness library (`src/`) and CLI (`src/main.rs`).
- `crates/keep-server` — the `keep` binary: a `Backend` trait (`backend.rs`)
  implemented by `KeepService` (local storage) and `HttpBackend` (forwarding
  to `keep serve`), behind an MCP stdio adapter (`mcp.rs`) and an HTTP
  adapter (`http.rs`). Scoped tokens are in `auth.rs`, TLS in `tls.rs`,
  embedder choice in `embedder.rs`, persisted settings (embedder record,
  receipt key) in `settings.rs` and evidence budgets in `budget.rs`. See
  `docs/server.md`.
- `fixtures/` — synthetic datasets and manifests used by tests.
- `manifests/` — templates for real-data runs.
- `docs/benchmarks/` — methodology; `docs/memory.md` — the Memory model;
  `docs/knowledge.md` — Knowledge design; `docs/server.md` — the server.

## Rules

- Tests never download data or call real models. Use `fixtures/` and the mock
  providers.
- Never weaken the dispatch guard: remote calls need the manifest field, the
  CLI flag, pinned models and non-zero caps.
- Errors and unavailable answers are explicit states in artifacts. Do not
  drop, retry silently or impute them.
- Never commit `data/` or `results/`, real conversations or credentials.
- Changing a manifest after seeing results requires a new manifest version.
- Fixtures use invented people and projects only.
- Memory text must round-trip byte for byte, and Markdown edits must never
  drop protected lineage. Any new ledger must pass
  `keep_memory::conformance::run`.
