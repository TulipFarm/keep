# Keep server

`keep` serves Keep's Memory and Knowledge to coding agents. Claude Code,
Codex, Copilot CLI and any other MCP client can share one memory per project.

[Website](https://keep.tulipfarm.ai) · [GitHub](https://github.com/TulipFarm/keep) · [TulipFarm](https://tulipfarm.ai)

It has two transports over one service layer, so both behave the same way:

- **MCP over stdio** (`keep mcp`). This is how coding agents connect.
- **HTTP/JSON** (`keep serve`). Use it for scripts, dashboards and anything
  that is not an MCP client, and as a daemon that several agents share.

`keep mcp --server URL` is a third front end: an MCP stdio server that
forwards every tool call to a running `keep serve`. It lists the same tools
and returns the same errors as local mode.

## Install and run

```sh
cargo install --path crates/keep-server     # installs the `keep` binary
keep --project billing-api mcp              # MCP over stdio, in-memory storage
keep --project billing-api serve            # HTTP on 127.0.0.1:7357
```

From a checkout without installing, run
`cargo run -p keep-server -- --project billing-api serve`. Add
`--features local-embeddings` to build with the local embedding model (see
[Embeddings](#embeddings)).

### Recommended setup: one daemon, many agents

Each MCP client starts its own `keep mcp` process. To let agents share
Memory and Knowledge, run one `keep serve` daemon and connect every agent to
it with `keep mcp --server`:

```sh
export KEEP_API_TOKEN="$(keep token new | head -1)"   # admin token
keep --user alex serve &                              # http://127.0.0.1:7357
keep --project billing-api mcp --server http://127.0.0.1:7357
```

The daemon may use any storage profile, including `in-memory`: every agent
talks to the same process. Use a Postgres profile when Memory must survive a
restart. `--user` and `--project` given to `keep mcp --server` become that
agent's default scope; scope fields it leaves out fall back to the daemon's
defaults. The token comes from `KEEP_API_TOKEN`, or give each agent a scoped
token (see [Tokens and scopes](#tokens-and-scopes)). For an `https://`
daemon with a private certificate, pass `--server-ca ca.pem` or set
`KEEP_SERVER_CA`.

`keep mcp` without `--server` on in-memory storage prints a note on stderr
that its Memory is neither shared nor kept.

### Storage profiles

| `--storage` / `KEEP_STORAGE` | Records | Vectors | Shared between processes |
| --- | --- | --- | --- |
| `in-memory` (default) | this process | this process | no; lost on exit |
| `postgres-vector` | Postgres | pgvector | yes |
| `postgres-qdrant` | Postgres | Qdrant (a rebuildable projection) | yes |

Without a daemon, agents share memory only through a Postgres profile, and
each `keep mcp` opens storage itself:

```sh
docker compose --profile postgres-qdrant up -d
export KEEP_STORAGE=postgres-qdrant
export KEEP_POSTGRES_URL='postgres://keep:keep-local-only@127.0.0.1:15433/keep'
export KEEP_QDRANT_URL='http://127.0.0.1:16333'
keep --project billing-api mcp
```

For a quick start that runs Keep itself in Compose with generated secrets,
see [Run with Docker Compose](#run-with-docker-compose).

#### Remote databases and TLS

Keep connects to Postgres with TLS according to `sslmode` in the URL:

| `sslmode` | Behaviour |
| --- | --- |
| not set, loopback host | TLS if the server offers it, otherwise plaintext |
| not set, any other host | TLS required, certificate verified |
| `require`, `verify-ca`, `verify-full` | TLS required, certificate and host name verified |
| `prefer` | TLS if offered, otherwise plaintext |
| `disable` | Plaintext |

Certificates are checked against the Mozilla root store plus any CA from
`sslrootcert=` or `--postgres-ca`, so a managed database works with its
provider's CA file. Qdrant over `https://` is verified the same way; add
`--qdrant-ca` for a private CA. Keep warns when it would send a Qdrant API key
over plain HTTP to a non-loopback host. Qdrant calls time out after 5 seconds
to connect and 60 seconds per request.

Tables are created on first start. Memory tables are prefixed `memory_`,
Knowledge tables `knowledge_` and server settings live in
`keep_server_settings`, so all share one schema.

### Configuration

Every flag can also be set through its environment variable. Flags go before
the subcommand.

| Flag | Environment | Default | Meaning |
| --- | --- | --- | --- |
| `--storage` | `KEEP_STORAGE` | `in-memory` | Storage profile, as in the table above |
| `--postgres-url` | `KEEP_POSTGRES_URL` | | Postgres URL for the `postgres-*` profiles |
| `--postgres-schema` | `KEEP_POSTGRES_SCHEMA` | search path | Schema for all tables; created if missing |
| `--qdrant-url` | `KEEP_QDRANT_URL` | | Qdrant URL for `postgres-qdrant` |
| `--postgres-ca` | `KEEP_POSTGRES_CA` | none | Extra CA certificate (PEM) for Postgres TLS; same as `sslrootcert` in the URL |
| `--qdrant-api-key` | `KEEP_QDRANT_API_KEY` | | Qdrant API key |
| `--qdrant-ca` | `KEEP_QDRANT_CA` | none | Extra CA certificate (PEM) for an `https://` Qdrant |
| `--collection` | `KEEP_COLLECTION` | `knowledge` | Vector collection for Knowledge |
| `--user` | `KEEP_USER` | `local` | Default user |
| `--project` | `KEEP_PROJECT` | none | Default project |
| `--agent` | `KEEP_AGENT` | none | Recorded as the harness on each Memory revision, such as `codex` |
| `--tokenizer` | `KEEP_TOKENIZER` | `tiktoken-o200k-base` | Budget tokenizer: `tiktoken-o200k-base`, `tiktoken-cl100k-base` or `approx-v1` |
| `--no-profile-writes` | `KEEP_NO_PROFILE_WRITES` | off | Refuse writes to `/profile.md` |
| `--embedder` | `KEEP_EMBEDDER` | `local` if built with `local-embeddings`, else `hash` | Knowledge embedder; see [Embeddings](#embeddings) |
| `--data-dir` | `KEEP_DATA_DIR` | platform data dir | Receipt key and model cache; see [Receipts](#receipts-and-the-data-directory) |
| | `KEEP_MODEL_CACHE` | `<data-dir>/models` | Where the local model is downloaded; `/opt/keep/models` (baked in) in the image |
| `--evidence-budget-tokens` | `KEEP_EVIDENCE_BUDGET_TOKENS` | off | Evidence each caller may draw per window; see [Evidence budget](#evidence-budget) |
| `--evidence-budget-window-secs` | `KEEP_EVIDENCE_BUDGET_WINDOW_SECS` | 3600 | Length of the rolling window |
| `--evidence-budget-per` | `KEEP_EVIDENCE_BUDGET_PER` | `token` | `token` or `session` |
| `mcp --server` | `KEEP_SERVER_URL` | none | Forward tool calls to this `keep serve` |
| `mcp --server-ca` | `KEEP_SERVER_CA` | none | Extra CA certificate (PEM) to trust for the daemon |
| `serve --bind` | `KEEP_BIND` | `127.0.0.1:7357` | HTTP listen address |
| `serve --tokens-file` | `KEEP_TOKENS_FILE` | none | Scoped tokens (TOML) |
| `serve --tls-cert`, `--tls-key` | `KEEP_TLS_CERT`, `KEEP_TLS_KEY` | none | Serve HTTPS with this PEM certificate chain and key |
| `serve --insecure-http` | `KEEP_INSECURE_HTTP` | off | Allow plain HTTP on a non-loopback address |
| `serve --max-body-bytes` | `KEEP_MAX_BODY_BYTES` | 1 MiB | Largest accepted request body |
| | `KEEP_API_TOKEN` | none | Administrator bearer token: full access. `keep mcp --server` sends it. |
| | `KEEP_RECEIPT_KEY` | persisted | Hex key that signs receipts; overrides the stored one |
| `--model-endpoint` | `KEEP_MODEL_ENDPOINT` | off | OpenAI-compatible endpoint for fact extraction and `knowledge_summarize`; see [Model-backed extraction](#model-backed-extraction-and-summaries) |
| `--model-name` | `KEEP_MODEL_NAME` | | Model name sent in requests; required with an endpoint |
| `--model-api-key` | `KEEP_MODEL_API_KEY` | none | API key; prefer the file variant |
| `--model-api-key-file` | `KEEP_MODEL_API_KEY_FILE` | none | File holding the API key, such as a mounted secret |
| `--model-api-style` | `KEEP_MODEL_API_STYLE` | `openai` | `openai`, `azure-v1` or `azure-deployments` |
| `--model-api-version` | `KEEP_MODEL_API_VERSION` | none | `api-version` query parameter; required by `azure-deployments` |
| `--model-deployment` | `KEEP_MODEL_DEPLOYMENT` | none | Azure deployment; required by `azure-deployments`, sent as `model` by `azure-v1` |
| `--model-auth-header` | `KEEP_MODEL_AUTH_HEADER` | `api-key` for Azure styles, else `bearer` | Header that carries the key |
| `--model-served-name` | `KEEP_MODEL_SERVED_NAME` | none | Exact `model` every reply must report; a mismatch stops further calls |
| `--model-reasoning-effort` | `KEEP_MODEL_REASONING_EFFORT` | none | `reasoning_effort` for reasoning models |
| `--model-timeout-secs` | `KEEP_MODEL_TIMEOUT_SECS` | 60 | Timeout for one model call |
| `--model-concurrency` | `KEEP_MODEL_CONCURRENCY` | 4 | Model calls in flight at once in this process |
| `--model-reply-tokens` | `KEEP_MODEL_REPLY_TOKENS` | 2048 | Most output tokens per call |
| `--model-budget-calls` | `KEEP_MODEL_BUDGET_CALLS` | 100000 | Most model calls per process |
| `--model-budget-input-tokens` | `KEEP_MODEL_BUDGET_INPUT_TOKENS` | 100000000 | Most input tokens per process |
| `--model-budget-output-tokens` | `KEEP_MODEL_BUDGET_OUTPUT_TOKENS` | 20000000 | Most output tokens per process |
| `--no-model-extraction` | `KEEP_NO_MODEL_EXTRACTION` | off | Use the model for `knowledge_summarize` only; extraction stays rule-based |

### Embeddings

Knowledge search combines lexical and vector retrieval. `--embedder` picks
the vector side:

- `local`: BAAI/bge-small-en-v1.5 (384 dimensions), run in process with
  ONNX. It is compiled in only with the `local-embeddings` cargo feature,
  and is then the default. The model files, pinned by revision and SHA-256,
  are downloaded once on first start to `KEEP_MODEL_CACHE` (default
  `<data-dir>/models`); after that no network access is needed.
  `keep fetch-model --dir DIR` downloads and verifies them ahead of time;
  the container image runs it at build time, so the image never downloads
  the model. Asking for `local` in a build without the feature fails at
  start-up.
- `hash`: Keep's offline hash embedder (256 dimensions). It needs no model
  and suits tests, but its semantic recall is weak.

Unless a chat model is configured (see below), the server never calls a
hosted model API. Besides that one-time download it talks only to the
configured Postgres and Qdrant.

With a Postgres profile the embedder and its dimension are recorded per
collection. A server started with a different embedder refuses to start
rather than mix vectors. To switch, either keep the old `--embedder`, or
start with the new embedder and a new, empty `--collection`, then call
`knowledge_rebuild_projection` once for each user to re-embed from the
records in Postgres. `GET /v1/health` reports the embedder in use.

### Model-backed extraction and summaries

By default Knowledge extracts facts with deterministic rules. Set
`KEEP_MODEL_ENDPOINT` and `KEEP_MODEL_NAME` to extract them with an
OpenAI-compatible chat model instead, the same extractor the benchmark
harness uses:

```sh
export KEEP_MODEL_ENDPOINT=https://api.example.com/v1
export KEEP_MODEL_NAME=small-chat-model
export KEEP_MODEL_API_KEY_FILE=/run/secrets/model_api_key
```

For Azure OpenAI use `KEEP_MODEL_API_STYLE=azure-v1` with
`KEEP_MODEL_ENDPOINT=https://<resource>.openai.azure.com` and the deployment
in `KEEP_MODEL_DEPLOYMENT`, or `azure-deployments` with
`KEEP_MODEL_API_VERSION` as well. Keys go in an `api-key` header for the
Azure styles unless `KEEP_MODEL_AUTH_HEADER` says otherwise.

- Extraction runs in the job queue: each episode is read on its own, cut at
  whitespace into turns of at most 4,000 characters and calls of at most 8
  turns, and every fact cites the episode it came from. There is no
  cross-episode context. Up to `KEEP_MODEL_CONCURRENCY` calls run at once.
- `knowledge_ingest` waits for the jobs, so with a model it returns after
  the model calls. Calls that fail are explicit job states, never silently
  dropped: a timeout, rate limit, transport error or HTTP 5xx is retried
  (`enrichment.state` is `pending`, and the background worker retries it);
  a malformed reply, a configuration error or an exhausted budget fails the
  job after its attempts (`failed`). The ingested text stays searchable
  either way.
- `KEEP_MODEL_BUDGET_*` caps the calls and tokens one process may use; past
  a cap every call fails with `budget exhausted` until restart. Nothing
  converts tokens to money, so set the caps from your price.
- Bad configuration, such as an endpoint without a model name, both key
  variables, or a zero budget, stops the server at start-up.
- `GET /v1/health` reports the model name, endpoint host and style, and the
  calls and tokens used so far. It never shows the key.

The same model serves `knowledge_summarize`. Without a model configured that
tool returns `unavailable` with reason `MODEL_NOT_CONFIGURED`.

### Receipts and the data directory

Forget receipts and idempotency receipts are signed with a key, so they
verify across restarts:

1. `KEEP_RECEIPT_KEY` (hex, at least 16 bytes), if set.
2. With a Postgres profile, a key generated on first start and stored in
   `keep_server_settings`. Every server on that database shares it.
3. In memory, `<data-dir>/receipt.key`, generated on first start with
   permissions `0600` in a `0700` directory. Keep warns if the file is
   readable by others.

The data directory is `--data-dir`, else `$XDG_DATA_HOME/keep`, else
`~/Library/Application Support/keep` on macOS, `%APPDATA%\keep` on Windows
and `~/.local/share/keep` elsewhere. Without one, receipts fall back to a
key for that process only, with a warning.

### Evidence budget

Each `knowledge_search` and `knowledge_fetch` call already has its own
evidence allowance (`budget_tokens`). Set `--evidence-budget-tokens` to also
cap the evidence each caller draws over a rolling window
(`--evidence-budget-window-secs`, default one hour). A call's allowance is
cut to what is left, so a nearly spent budget returns fewer hits; once it is
spent the call fails with `budget_exceeded` (reason
`EVIDENCE_BUDGET_EXHAUSTED`, with `details.retryAfterSeconds`). Only the
evidence actually returned is charged.

With `--evidence-budget-per token` (the default) each bearer token, or the
local process for `keep mcp`, has one budget. With `session`, each
`keep mcp --server` process has its own, keyed by the `x-keep-session`
header it sends. Budgets live in the server's memory and reset when it
restarts.

## Scopes

Every tool takes an optional `scope: {"user": "...", "project": "..."}`.
Fields you leave out fall back to `--user` and `--project`. Pass
`"project": ""` to force the personal scope when the server has a default
project. Ids are 1 to 128 letters, digits, `.`, `_` or `-`, so a host can use
ids such as `ws-<uuid>` or `ws-<uuid>-m-<uuid>` (78 characters) directly.

| | Project scope (`alex` + `billing-api`) | Personal scope (`alex`) |
| --- | --- | --- |
| Memory tree | `user/alex/project/billing-api` | `user/alex` |
| Knowledge writes | partition `project:billing-api` in corpus `user:alex` | partition `user` in corpus `user:alex` |
| Knowledge reads | its own partition and the personal one | the personal partition only |

All agents that use the same scope share the same files and Knowledge.
Projects never see each other's Memory or Knowledge. Personal Knowledge is
visible from every project of that user.

`memory_context` for a project scope also includes the user's personal tree,
within the same standing budget (see [Memory](memory.md)). Each file it
lists carries an `origin` of `project` or `personal`; when both trees have a
file at the same path, the project's wins. Pass `"include_personal": false`
for the project tree alone.

A source can be forgotten only from the scope that ingested it. From any other
scope the source is reported as `not_found`.

## Tools

The MCP tool names and HTTP routes map one to one. HTTP routes take a JSON
object body: `POST /v1/memory/write` runs `memory_write`. Each tool's full
input schema is in `tools/list`, or `GET /v1/tools` over HTTP.

| Tool | HTTP route | What it does |
| --- | --- | --- |
| `memory_context` | `POST /v1/memory/context` | Standing context bundle: profile, file directory, relevant files and a Knowledge table of contents. Default budget 4,096 tokens, maximum 8,192. |
| `memory_list` | `POST /v1/memory/list` | Categorised file tree, with ids, revision ids and sizes |
| `memory_read` | `POST /v1/memory/read` | One file by `path` or `memory_id`, optionally an older `revision_id` |
| `memory_write` | `POST /v1/memory/write` | Create a file with `text`, or change one with `text` or an `edit`, using compare-and-swap |
| `memory_update_metadata` | `POST /v1/memory/update_metadata` | Change a file's `description` or `display_name` with compare-and-swap. Recorded in history and undoable. |
| `memory_history` | `POST /v1/memory/history` | A file's revisions, newest first |
| `memory_undo` | `POST /v1/memory/undo` | Undo the latest change, or go back to `target_revision_id` |
| `memory_delete` | `POST /v1/memory/delete` | Move a file to the trash |
| `memory_restore` | `POST /v1/memory/restore` | Restore a file from the trash |
| `memory_export` | `POST /v1/memory/export` | The scope's files as plain Markdown, with a manifest. `instruction_files` is `refuse` (default), `skip` or `allow_marked` for files that look like agent instructions. Writes nothing to disk. |
| `memory_forget` | `POST /v1/memory/forget` | Erase a file, an entry or some text everywhere, including history and trash. Returns a receipt. |
| `knowledge_ingest` | `POST /v1/knowledge/ingest` | Add `text` (a document) or `episodes` (a conversation). Returns `captured: true` once the records are stored, and an `enrichment` state; see [Operation states](#operation-states). |
| `knowledge_search` | `POST /v1/knowledge/search` | Cited passages and facts within an evidence budget (default 2,000, maximum 8,000), optionally `as_of` or `valid_at` a time |
| `knowledge_fetch` | `POST /v1/knowledge/fetch` | Full evidence for a hit's `item.id` |
| `knowledge_cite` | `POST /v1/knowledge/cite` | Exact source citations (quote, offsets, speaker, time) for a hit's `item.id` or an evidence id |
| `knowledge_traverse` | `POST /v1/knowledge/traverse` | Facts around an entity, 1 or 2 hops from `start`, optionally `as_of` or `valid_at` a time |
| `knowledge_index` | `POST /v1/knowledge/index` | Book-style index: level 0 collections, level 1 sources and topics, level 2 entities |
| `knowledge_forget_source` | `POST /v1/knowledge/forget_source` | Erase one source and everything derived only from it. Returns a receipt. |
| `knowledge_status` | `POST /v1/knowledge/status` | Background job counts and enrichment state for the scope only |
| `knowledge_summarize` | `POST /v1/knowledge/summarize` | A few sentences summarizing one readable `source_id`, some `text`, or `episodes` (at most 200 turns, 200,000 characters), written by the configured model; see [Model-backed extraction](#model-backed-extraction-and-summaries). Needs `read`. |
| `knowledge_rebuild_projection` | `POST /v1/knowledge/rebuild_projection` | Re-embed the scope user's whole corpus into the vector index from the records. Needs the `admin` permission. |

`GET /v1/health` (also `/v1/health/live`) reports the version, storage
profile, tokenizer, embedding method and default scope; it answers while the
process runs. `GET /v1/health/ready` checks storage and returns 200 with
`"status": "ready"` or 503 with `"status": "unavailable"`, and `checks.ledger`
(`ok` or `unavailable`) and `checks.vectors` (`ok`, `degraded` or `none`).
Vectors being down leaves the server ready: search falls back to lexical
results. Neither endpoint needs a token or includes error text.
`keep healthcheck [--url URL]` exits non-zero unless the ready endpoint
returns 200; container healthchecks use it. `GET /v1/tools` lists every
tool with its route and input schema; it needs any valid token.

### Sources for a host application

A host application that indexes its own documents can address them by its
own ids and keep each user's reads to the documents that user may see.

**Ingest.** `knowledge_ingest` takes `source_id`, `title`, `collection`,
`kind` (`document` or `conversation`) and an optional `metadata` object: at
most 16 string keys of up to 64 bytes, with string values of up to 1,024
bytes, such as the document's URL or id in the host. Metadata is stored with
the source and returned on every hit's citation as `source_metadata`.

**New revisions.** Ingesting a `source_id` again with different text fails
with `conflict` unless the call passes `"replace": true`. Replace removes the
source's old episodes, chunks, facts derived only from it and its vectors,
and stores the new revision with its new title, collection and metadata, in
one ledger transaction: a search sees the old revision or the new one, never
both. The response's `replaced` is `true` when the text, title, collection or metadata changed and
`false` when it was identical (then nothing is rewritten). Facts also
supported by other sources keep that other evidence. Replace does not issue
a forget receipt; use `knowledge_forget_source` to erase a source.

**Allowlists.** `knowledge_search`, `knowledge_fetch`, `knowledge_cite`,
`knowledge_traverse` and `knowledge_index` take optional `source_ids` and
`collections` lists (at most 1,000 entries each):

- Only passages, facts and index entries derived from a listed source are
  returned. With both lists, a source must be in both. An empty list
  matches nothing.
- The filter runs inside retrieval, before ranking, limits and the evidence
  budget, for both lexical and vector search and in every storage profile
  (a pgvector `WHERE` clause or a Qdrant payload filter), so a narrow
  allowlist still gets a full page of results.
- A fact supported by several sources is returned when at least one of them
  is allowed, and shows only the allowed evidence.
- Results read as if the other sources did not exist. A fact superseded
  only by a fact from a disallowed source stays active, with no `valid_to`
  or version caused by it, including for `valid_at` reads. Entity aliases
  in `knowledge_index` and `knowledge_traverse` come only from allowed
  sources, and an alias known only from a disallowed source does not start
  a traversal.
- `knowledge_fetch` or `knowledge_cite` of an id outside the allowlist fails
  with the same `not_found` error as an id that does not exist.
- `knowledge_summarize` takes the same lists for `source_id`: a source
  outside them is `not_found`. They do not apply to `text` or `episodes`,
  which the caller supplies.
- The lists narrow what the scope can already read; they never widen it.

Vectors written before this release carry no source tags, so the vector
side of a filtered search misses them until `knowledge_rebuild_projection`
re-embeds the corpus; lexical search finds them meanwhile. Filtered
`knowledge_index` views are computed per call rather than cached.

### Operation states

Keep never reports success for work it has not done.

- `knowledge_ingest` stores the source and its episodes in Postgres first,
  then extracts facts and writes vectors. If that second step fails, for
  example because Qdrant is down, the call still succeeds with
  `captured: true` and `enrichment.state` set to `pending`; the work is kept
  as jobs and retried. If storing the records fails, the call fails with
  `unavailable`.
- `enrichment.state` is `complete`, `pending` (jobs waiting or retrying),
  `failed` (jobs gave up after 3 attempts; `knowledge_rebuild_projection`
  redoes them) or `unknown` (the job counts could not be read).
- `keep serve` and `keep mcp` run a background worker every 5 seconds that
  retries due jobs for every user. It logs only when its state changes.
- `knowledge_status` shows the counts (`pending`, `running`,
  `retryScheduled`, `succeeded`, `failed`) for the caller's own scope, so a
  token limited to one project learns nothing about other projects.
  `knowledge_ingest` returns the same counts without `succeeded`: finished
  jobs pile up as a corpus grows, and counting them on every write would
  slow each write down.

### Compare-and-swap writes

`memory_write` without `expected_revision_id` only creates files. To change a
file, pass the `revisionId` from your last `memory_read` or `memory_write`.
If someone else changed it first, you get a `conflict` error. Its
`details.current.revisionId` tells you what to re-read. Set `"merge": true` to
let Keep merge non-overlapping entry edits instead.

Pass an `idempotency_key` on creates so a retried request replays the first
result instead of failing, also after a server restart.

## Errors

Failures return a stable code. Over HTTP the status matches the code. Over
MCP the tool result has `isError: true` and the same object as
`structuredContent`.

```json
{
  "error": {
    "code": "conflict",
    "message": "the file changed since revision ...",
    "retryable": false,
    "reason": "VERSION_CONFLICT",
    "details": { "current": { "revisionId": "...", "version": "..." } }
  }
}
```

| Code | HTTP | Meaning |
| --- | --- | --- |
| `invalid` | 400 | Malformed request or argument |
| `unauthorized` | 401 | Missing or wrong bearer token |
| `permission_denied` | 403 | The token may not use this scope or tool (reason `SCOPE_NOT_ALLOWED`), or the engine refused the operation, such as a profile write |
| `not_found` | 404 | No such file, revision, source, item or route |
| `conflict` | 409 | Stale revision, path already exists, or source id used elsewhere |
| `payload_too_large` | 413 | Body larger than `--max-body-bytes` |
| `limit_exceeded` | 422 | A size limit, such as file or entry length |
| `budget_exceeded` | 422 | Requested budget above the maximum, required context does not fit, or the caller's evidence budget is spent |
| `unavailable` | 503 | Storage unreachable; `retryable` is true |
| `internal` | 500 | Anything else |

`reason` carries the engine's own code when there is one.

## Security

- HTTP binds to `127.0.0.1` by default. `keep serve` refuses to start on any
  other address unless it has a token (`KEEP_API_TOKEN` or a tokens file).
  With a token, every route except `/v1/health` needs
  `Authorization: Bearer <token>`.
- On a non-loopback address it also needs TLS, or an explicit
  `--insecure-http`, which prints a warning.
- A request body must be sent as `Content-Type: application/json`, so a web
  page cannot post to the server without a CORS preflight, which it never
  grants.
- Without a token, the server also refuses requests that carry an `Origin`
  header, or whose `Host` is not `localhost` or a loopback address. A web
  page, including one reached through DNS rebinding, can neither write
  nor read memory.
- Request bodies are capped by `--max-body-bytes`.
- The server logs only the tool name, outcome and duration to stderr. It never
  logs Memory or Knowledge contents. In MCP mode stdout carries only the
  protocol.
- Memory and Knowledge text is untrusted data, not instructions. Agents
  should treat it that way.

### Tokens and scopes

`KEEP_API_TOKEN` is an administrator token: it may use every scope and tool.
For anything narrower, list tokens in a TOML file and start the daemon with
`--tokens-file` (or `KEEP_TOKENS_FILE`). The file holds only SHA-256 hashes:

```sh
keep token new                   # prints a new token, then its hash
printf %s "$TOKEN" | keep token hash   # hash an existing token
```

```toml
[[token]]
name = "billing-agents"          # shown in logs instead of the token
sha256 = "<64 hex characters>"
users = ["alex"]                 # or ["*"]
projects = ["billing-api"]       # or ["*"]; [] for none
personal = false                 # may it name the personal scope itself?
permissions = ["read", "write"]  # "read", "write", "admin"

[[token]]
name = "dashboards"
sha256 = "<64 hex characters>"
users = ["*"]
projects = ["*"]
personal = true
permissions = ["read"]
```

- Read-only tools need `read`, the rest `write`, and
  `knowledge_rebuild_projection` needs `admin`.
- A call naming a user, project or tool outside the token's grant fails with
  `permission_denied` before it reaches storage.
- A scope field left out is filled from the token when its grant allows
  exactly one value, else from the daemon's defaults, and then checked.
- A project scope reads the user's personal Knowledge and personal Memory
  context as usual. Naming the personal scope itself needs `personal = true`.
- Presented tokens are hashed and compared with every entry in constant time.
  Tokens are never logged.

### TLS

Pass `--tls-cert chain.pem --tls-key key.pem` to serve HTTPS directly
(rustls, TLS 1.2 and 1.3). Agents connect with
`keep mcp --server https://host:7357`, adding `--server-ca ca.pem` for a
private CA.

Alternatively, keep `keep serve` on `127.0.0.1` and put it behind a reverse
proxy that terminates TLS, such as Caddy or nginx. Use `--insecure-http` only
when the network path to the daemon is already protected, for example inside
a private container network.

## Run with Docker Compose

`deploy/compose.yaml` runs Keep, Postgres and optionally Qdrant with
generated secrets. Pick one profile:

```sh
docker compose -f deploy/compose.yaml --profile postgres-vector up -d --wait
docker compose -f deploy/compose.yaml --profile postgres-qdrant up -d --wait
```

- A one-shot `secrets` service generates the Postgres password, the Keep
  administrator token and the Qdrant API key into the `keep-secrets` volume.
  Read the token with
  `docker compose -f deploy/compose.yaml run --rm --entrypoint cat keep-vector /run/keep-secrets/api_token`
  (use `keep-qdrant` for the other profile).
- Keep listens on `127.0.0.1:7357` only (`KEEP_PORT` changes the port). The
  databases publish no ports.
- Each service waits until the one before it is healthy. Keep's healthcheck
  is `keep healthcheck`.
- Images are pinned by digest. Keep runs as a non-root user with a read-only
  root filesystem and no capabilities; its data directory is the `keep-data`
  volume.
- The image uses the `hash` embedder. Build with
  `--build-arg KEEP_FEATURES=local-embeddings` for the local model, which is
  then baked into the image.
- `docker compose ... down` keeps the data; `down -v` deletes it, secrets
  included.

`scripts/compose-smoke.sh` builds the image and checks both profiles end to
end on spare ports, then removes everything it created.

### Published image

The `Image` workflow publishes `ghcr.io/tulipfarm/keep`, built with
`local-embeddings` and the pinned model baked in, for `linux/amd64` and
`linux/arm64`:

- `main` and `sha-<short sha>` for each push to `main`;
- `X.Y.Z`, `X.Y` and (from 1.0) `X` for each `vX.Y.Z` tag.

It runs as user `keep` (uid 10001) and starts `keep serve`. Each build is
smoke-tested with no network: the server must become ready, by
`keep healthcheck`, from the baked model. The workflow prints the
multi-platform digest; pin it in deployments:

```sh
docker run --rm -p 127.0.0.1:7357:7357 \
  -e KEEP_BIND=0.0.0.0:7357 -e KEEP_API_TOKEN=... -e KEEP_INSECURE_HTTP=1 \
  ghcr.io/tulipfarm/keep@sha256:<digest>
```

## Try it over HTTP

```sh
keep --project billing-api serve &
curl -s localhost:7357/v1/memory/write -H 'content-type: application/json' \
  -d '{"path": "/topics/style.md", "text": "- Prefer small, reviewed commits.\n"}'
curl -s localhost:7357/v1/memory/context -H 'content-type: application/json' -d '{}'
curl -s localhost:7357/v1/knowledge/ingest -H 'content-type: application/json' \
  -d '{"title": "Release notes", "text": "Releases are cut from main every Friday."}'
curl -s localhost:7357/v1/knowledge/search -H 'content-type: application/json' \
  -d '{"query": "when are releases cut"}'
```

## Connect a coding agent

The snippets below start `keep mcp` for one project against shared Postgres
storage. To use a daemon instead, replace the storage variables with
`KEEP_SERVER_URL` (for example `http://127.0.0.1:7357`) and `KEEP_API_TOKEN`.
Use the binary's absolute path if `keep` is not on the agent's `PATH`. Give
each agent its own `KEEP_AGENT` so history shows who changed what. Keep
passwords and tokens out of committed files: export them in your shell
instead.

### Claude Code

```sh
claude mcp add keep --scope project \
  -e KEEP_STORAGE=postgres-vector \
  -e KEEP_POSTGRES_URL='postgres://keep:keep-local-only@127.0.0.1:15433/keep' \
  -e KEEP_AGENT=claude-code \
  -- keep --project billing-api mcp
```

Or commit a `.mcp.json` at the project root. Claude Code expands `${VAR}`
from your environment:

```json
{
  "mcpServers": {
    "keep": {
      "command": "keep",
      "args": ["--project", "billing-api", "mcp"],
      "env": {
        "KEEP_STORAGE": "postgres-vector",
        "KEEP_POSTGRES_URL": "${KEEP_POSTGRES_URL}",
        "KEEP_AGENT": "claude-code"
      }
    }
  }
}
```

### Codex

In `~/.codex/config.toml`:

```toml
[mcp_servers.keep]
command = "keep"
args = ["--project", "billing-api", "mcp"]
env = { KEEP_STORAGE = "postgres-vector", KEEP_POSTGRES_URL = "postgres://keep:keep-local-only@127.0.0.1:15433/keep", KEEP_AGENT = "codex" }
```

### Copilot CLI

In `~/.copilot/mcp-config.json`:

```json
{
  "mcpServers": {
    "keep": {
      "type": "local",
      "command": "keep",
      "args": ["--project", "billing-api", "mcp"],
      "env": {
        "KEEP_STORAGE": "postgres-vector",
        "KEEP_POSTGRES_URL": "postgres://keep:keep-local-only@127.0.0.1:15433/keep",
        "KEEP_AGENT": "copilot-cli"
      },
      "tools": ["*"]
    }
  }
}
```

### Other clients

Any MCP client that can start a stdio server works the same way. Run
`keep --project <id> mcp` with the storage variables set, or
`keep --project <id> mcp --server <url>`.

## Limitations

This is an alpha.

- In-memory storage lives in one process. Agents share it only through a
  `keep serve` daemon, and it is lost when that daemon stops.
- Evidence budgets are kept in the server's memory: they reset on restart and
  are not shared between several daemons.
- The local embedder is English only. Switching embedders needs a new
  collection and a projection rebuild.
- Token grants are per user and project, not per file or source.
- The tokens file is read at start-up; restart the daemon to change it.
- TLS certificates are read at start-up; restart to rotate them.
- Memory forget erases synchronously and returns a receipt; there is no
  separate erasure journal replayed at start-up yet.
- Ingest does not return an operation id; poll `knowledge_status` instead.
- Model-backed extraction reads each episode alone, and `knowledge_ingest`
  waits for it. Summaries are not stored.
