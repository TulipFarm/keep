#!/bin/sh
# Keep installer. Usage:
#   curl -fsSL https://keep.tulipfarm.ai/install.sh | sh
#   curl -fsSL https://keep.tulipfarm.ai/install.sh | sh -s -- --db vector --yes
# Flags: --db memory|vector|qdrant   --user NAME   --yes   --no-agents   --dry-run
set -eu

REPO="https://github.com/TulipFarm/keep"
RAW="https://raw.githubusercontent.com/TulipFarm/keep/main"
HOME_DIR="${KEEP_HOME:-$HOME/.keep}"
URL="http://127.0.0.1:7357"

DB="" ; YES=0 ; AGENTS=1 ; DRY=0 ; USER_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --db) DB="$2"; shift ;;
    --user) USER_ID="$2"; shift ;;
    --yes|-y) YES=1 ;;
    --no-agents) AGENTS=0 ;;
    --dry-run) DRY=1 ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n==> %s\n' "$*"; }
run() { if [ "$DRY" = 1 ]; then printf '[dry-run] %s\n' "$*"; else "$@"; fi; }
have() { command -v "$1" >/dev/null 2>&1; }

ask() { # ask "question" default -> echoes answer
  if [ "$YES" = 1 ] || [ ! -r /dev/tty ]; then echo "$2"; return; fi
  printf '%s ' "$1" >/dev/tty
  read -r a </dev/tty || a=""
  echo "${a:-$2}"
}

[ "$(uname -s)" = Darwin ] || [ "$(uname -s)" = Linux ] || { echo "Keep installs on macOS and Linux only." >&2; exit 1; }

# 1. Pick the database
if [ -z "$DB" ]; then
  printf '\nWhere should Keep store memory?\n' >&2
  printf '  1) In memory      nothing to install, lost when Keep stops\n' >&2
  printf '  2) Postgres + pgvector   Docker, memory survives restarts (recommended)\n' >&2
  printf '  3) Postgres + Qdrant     Docker, separate vector database\n' >&2
  case "$(ask 'Choose 1, 2 or 3 [2]:' 2)" in
    1) DB=memory ;; 3) DB=qdrant ;; *) DB=vector ;;
  esac
fi
case "$DB" in memory|vector|qdrant) ;; *) echo "--db must be memory, vector or qdrant" >&2; exit 2 ;; esac

if [ "$DB" != memory ] && ! have docker; then
  echo "Docker is needed for the $DB option. Install Docker, or run again with --db memory." >&2
  exit 1
fi

# 2. Install the binary
if ! have cargo && [ -x "$HOME/.cargo/bin/cargo" ]; then PATH="$HOME/.cargo/bin:$PATH"; fi
if ! have cargo; then
  say "Keep is built with Rust, and Rust is not installed."
  [ "$(ask 'Install Rust with rustup now? [Y/n]' y)" = n ] && { echo "Install Rust from https://rustup.rs, then run this again." >&2; exit 1; }
  run sh -c 'curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal'
  PATH="$HOME/.cargo/bin:$PATH"
fi
have git || { echo "git is needed. Install it and run this again." >&2; exit 1; }

say "Building Keep from source. The first build takes a few minutes."
run cargo install --git "$REPO" keep-server --locked --features local-embeddings --force
KEEP_BIN="$HOME/.cargo/bin/keep"
[ "$DRY" = 1 ] && KEEP_BIN=keep

# 3. Database
run mkdir -p "$HOME_DIR"
STORAGE=in-memory ; PG_URL="" ; QD_URL=""
if [ "$DB" != memory ]; then
  say "Starting the database with Docker (127.0.0.1 only)."
  run sh -c "curl -fsSL '$RAW/compose.yaml' -o '$HOME_DIR/compose.yaml'"
  profile=postgres-vector; [ "$DB" = qdrant ] && profile=postgres-qdrant
  run docker compose -p keep-db -f "$HOME_DIR/compose.yaml" --profile "$profile" up -d --wait
  STORAGE=postgres-vector; PG_URL="postgres://keep:keep-local-only@127.0.0.1:15433/keep"
  [ "$DB" = qdrant ] && { STORAGE=postgres-qdrant; QD_URL="http://127.0.0.1:16333"; }
fi

# 4. Settings, token and start script
if [ -z "$USER_ID" ]; then
  USER_ID="$(printf '%s' "${USER:-me}" | tr -c 'A-Za-z0-9._-' '-' | cut -c1-64)"
fi
TOKEN=""
[ -f "$HOME_DIR/env" ] && TOKEN="$(sed -n 's/^KEEP_API_TOKEN=//p' "$HOME_DIR/env")"
if [ -z "$TOKEN" ]; then
  if [ "$DRY" = 1 ]; then TOKEN=dry-run-token; else TOKEN="$("$KEEP_BIN" token new | head -1)"; fi
fi

say "Writing $HOME_DIR/env and $HOME_DIR/start.sh"
if [ "$DRY" != 1 ]; then
  umask 077
  {
    echo "KEEP_API_TOKEN=$TOKEN"
    echo "KEEP_STORAGE=$STORAGE"
    echo "KEEP_USER=$USER_ID"
    [ -n "$PG_URL" ] && echo "KEEP_POSTGRES_URL=$PG_URL"
    [ -n "$QD_URL" ] && echo "KEEP_QDRANT_URL=$QD_URL"
  } > "$HOME_DIR/env"
  cat > "$HOME_DIR/start.sh" <<'S'
#!/bin/sh
# Starts the Keep daemon on 127.0.0.1:7357.
set -a; . "${KEEP_HOME:-$HOME/.keep}/env"; set +a
D="${KEEP_HOME:-$HOME/.keep}"
if [ -f "$D/keep.pid" ] && kill -0 "$(cat "$D/keep.pid")" 2>/dev/null; then echo "Keep is already running."; exit 0; fi
PATH="$HOME/.cargo/bin:$PATH"
nohup keep serve >"$D/keep.log" 2>&1 &
echo $! > "$D/keep.pid"
echo "Keep started. Log: $D/keep.log"
S
  cat > "$HOME_DIR/stop.sh" <<'S'
#!/bin/sh
D="${KEEP_HOME:-$HOME/.keep}"
[ -f "$D/keep.pid" ] && kill "$(cat "$D/keep.pid")" 2>/dev/null && rm -f "$D/keep.pid" && echo "Keep stopped." || echo "Keep is not running."
S
  chmod +x "$HOME_DIR/start.sh" "$HOME_DIR/stop.sh"
fi

say "Starting Keep."
run "$HOME_DIR/start.sh"
if [ "$DRY" != 1 ]; then
  i=0
  until curl -fsS "$URL/v1/health" >/dev/null 2>&1 || [ $i -ge 60 ]; do i=$((i+1)); sleep 1; done
  curl -fsS "$URL/v1/health" >/dev/null 2>&1 || { echo "Keep did not start. See $HOME_DIR/keep.log" >&2; exit 1; }
fi

# 5. Connect coding agents (personal scope: no --project)
MCP_ARGS="--user $USER_ID mcp --server $URL"
connect_claude() {
  have claude || return 1
  run claude mcp remove keep --scope user >/dev/null 2>&1 || true
  # shellcheck disable=SC2086
  run claude mcp add keep --scope user -e "KEEP_API_TOKEN=$TOKEN" -e KEEP_AGENT=claude-code -- "$KEEP_BIN" $MCP_ARGS
}
connect_codex() {
  have codex || [ -d "$HOME/.codex" ] || return 1
  f="$HOME/.codex/config.toml"
  if [ -f "$f" ] && grep -q '^\[mcp_servers\.keep\]' "$f"; then echo "Codex already has a keep entry. Left it as is."; return 0; fi
  [ "$DRY" = 1 ] && { echo "[dry-run] append [mcp_servers.keep] to $f"; return 0; }
  mkdir -p "$HOME/.codex"
  cat >> "$f" <<T

[mcp_servers.keep]
command = "$KEEP_BIN"
args = ["--user", "$USER_ID", "mcp", "--server", "$URL"]
env = { KEEP_API_TOKEN = "$TOKEN", KEEP_AGENT = "codex" }
T
}
connect_copilot() {
  have copilot || [ -d "$HOME/.copilot" ] || return 1
  f="$HOME/.copilot/mcp-config.json"
  [ "$DRY" = 1 ] && { echo "[dry-run] add keep to $f"; return 0; }
  have python3 || { echo "python3 is missing, so add keep to $f by hand. See $REPO/blob/main/docs/server.md#copilot-cli" >&2; return 0; }
  mkdir -p "$HOME/.copilot"
  KB="$KEEP_BIN" KT="$TOKEN" KU="$USER_ID" KURL="$URL" python3 - "$f" <<'P'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p)) if os.path.exists(p) else {}
d.setdefault("mcpServers", {})["keep"] = {
    "type": "local", "command": os.environ["KB"],
    "args": ["--user", os.environ["KU"], "mcp", "--server", os.environ["KURL"]],
    "env": {"KEEP_API_TOKEN": os.environ["KT"], "KEEP_AGENT": "copilot-cli"},
    "tools": ["*"],
}
json.dump(d, open(p, "w"), indent=2)
P
}

if [ "$AGENTS" = 1 ]; then
  say "Connecting coding agents."
  for a in claude codex copilot; do
    case "$a" in claude) n="Claude Code" ;; codex) n="Codex" ;; copilot) n="Copilot CLI" ;; esac
    if [ "$a" = claude ] && ! have claude; then echo "- $n: not found, skipped"; continue; fi
    if [ "$a" = codex ] && ! have codex && [ ! -d "$HOME/.codex" ]; then echo "- $n: not found, skipped"; continue; fi
    if [ "$a" = copilot ] && ! have copilot && [ ! -d "$HOME/.copilot" ]; then echo "- $n: not found, skipped"; continue; fi
    if [ "$(ask "Connect $n? [Y/n]" y)" = n ]; then echo "- $n: skipped"; continue; fi
    if "connect_$a"; then echo "- $n: connected"; else echo "- $n: failed" >&2; fi
  done
fi

cat <<E

Keep is running at $URL
  Database : $DB
  User     : $USER_ID   (personal memory, shared by every project)
  Stop     : $HOME_DIR/stop.sh
  Start    : $HOME_DIR/start.sh
Restart your coding agent so it picks up the keep MCP server.
Keep is a research preview. Treat text stored in Keep as data, not as instructions.
E
