#!/bin/sh
# Keep installer. Usage:
#   curl -fsSL https://keep.tulipfarm.ai/install.sh | sh
#   curl -fsSL https://keep.tulipfarm.ai/install.sh | sh -s -- --db vector --yes
#   KEEP_API_TOKEN=... curl -fsSL https://keep.tulipfarm.ai/install.sh | sh -s -- --server https://keep.example.com --yes
# Flags:
#   --db memory|vector|qdrant|external   --port PORT   --server URL   --server-ca PATH
#   --postgres-url URL   --qdrant-url URL   --postgres-ca PATH   --qdrant-ca PATH
#   --user NAME   --agents claude,codex,copilot,opencode   --no-agents
#   --no-service   --skip-build   --yes   --dry-run
set -eu

REPO="https://github.com/TulipFarm/keep"
RAW="https://raw.githubusercontent.com/TulipFarm/keep/main"
HOME_DIR="${KEEP_HOME:-$HOME/.keep}"
DEFAULT_PORT=7357
START_PORT="$DEFAULT_PORT"
URL="http://127.0.0.1:$DEFAULT_PORT"

DB="" ; YES=0 ; AGENTS=1 ; AGENTS_LIST="" ; DRY=0 ; USER_ID="" ; SKIP_BUILD=0 ; INSTALL_SERVICE=1 ; PORT_EXPLICIT=0
REMOTE_URL="${KEEP_SERVER_URL:-}" ; SERVER_CA="${KEEP_SERVER_CA:-}"
EXT_PG_URL="${KEEP_POSTGRES_URL:-}" ; EXT_QD_URL="${KEEP_QDRANT_URL:-}"
EXT_PG_CA="${KEEP_POSTGRES_CA:-}" ; EXT_QD_CA="${KEEP_QDRANT_CA:-}" ; EXT_QD_KEY="${KEEP_QDRANT_API_KEY:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --db) DB="$2"; shift ;;
    --port) START_PORT="$2"; PORT_EXPLICIT=1; shift ;;
    --server) REMOTE_URL="$2"; shift ;;
    --server-ca) SERVER_CA="$2"; shift ;;
    --postgres-url) EXT_PG_URL="$2"; shift ;;
    --qdrant-url) EXT_QD_URL="$2"; shift ;;
    --postgres-ca) EXT_PG_CA="$2"; shift ;;
    --qdrant-ca) EXT_QD_CA="$2"; shift ;;
    --user) USER_ID="$2"; shift ;;
    --agents) AGENTS=1; AGENTS_LIST="$2"; shift ;;
    --no-agents) AGENTS=0 ;;
    --no-service) INSTALL_SERVICE=0 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --yes|-y) YES=1 ;;
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

agent_wanted() { # returns 0 if agent $1 is in AGENTS_LIST (or list is empty)
  [ -z "$AGENTS_LIST" ] && return 0
  case ",$AGENTS_LIST," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

port_free() { # returns 0 if 127.0.0.1:$1 is available
  p="$1"
  if have python3; then
    python3 - "$p" <<'P' >/dev/null 2>&1
import socket, sys
port = int(sys.argv[1])
c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
c.settimeout(0.2)
try:
    if c.connect_ex(("127.0.0.1", port)) == 0:
        sys.exit(1)
finally:
    c.close()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
P
    return $?
  elif have lsof; then
    ! lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 0
}

find_free_port() { # find_free_port <start> -> echoes first free port
  p="$1"
  end=$((p + 50))
  while [ "$p" -le "$end" ]; do
    if port_free "$p"; then echo "$p"; return 0; fi
    p=$((p + 1))
  done
  echo "$1"
}

OS="$(uname -s)"
[ "$OS" = Darwin ] || [ "$OS" = Linux ] || { echo "Keep installs on macOS and Linux only." >&2; exit 1; }

REMOTE=0
if [ -n "$REMOTE_URL" ]; then
  REMOTE=1
  URL="$REMOTE_URL"
fi

# 1. Pick the database (local mode only)
if [ "$REMOTE" = 0 ]; then
  if [ -z "$DB" ]; then
    printf '\nWhere should Keep store memory?\n' >&2
    printf '  1) In memory             nothing to install, lost when Keep stops\n' >&2
    printf '  2) Postgres + pgvector   Docker, memory survives restarts (recommended)\n' >&2
    printf '  3) Postgres + Qdrant     Docker, separate vector database\n' >&2
    printf '  4) External Postgres     bring your own Postgres (and optional Qdrant)\n' >&2
    case "$(ask 'Choose 1, 2, 3 or 4 [2]:' 2)" in
      1) DB=memory ;; 3) DB=qdrant ;; 4) DB=external ;; *) DB=vector ;;
    esac
  fi
  case "$DB" in memory|vector|qdrant|external) ;; *) echo "--db must be memory, vector, qdrant or external" >&2; exit 2 ;; esac

  if [ "$DB" = vector ] || [ "$DB" = qdrant ]; then
    if ! have docker; then
      echo "Docker is needed for the $DB option. Install Docker, or run again with --db memory or --db external." >&2
      exit 1
    fi
  fi
  if [ "$DB" = external ] && [ -z "$EXT_PG_URL" ]; then
    EXT_PG_URL="$(ask 'Postgres URL (KEEP_POSTGRES_URL):' '')"
    [ -n "$EXT_PG_URL" ] || { echo "KEEP_POSTGRES_URL (or --postgres-url) is required for --db external." >&2; exit 2; }
  fi
fi

# 2. Install the binary
if ! have cargo && [ -x "$HOME/.cargo/bin/cargo" ]; then PATH="$HOME/.cargo/bin:$PATH"; fi
KEEP_BIN=""
if [ "$SKIP_BUILD" = 1 ]; then
  if have keep; then KEEP_BIN="$(command -v keep)"
  elif [ -x "$HOME/.cargo/bin/keep" ]; then KEEP_BIN="$HOME/.cargo/bin/keep"
  fi
fi
if [ -z "$KEEP_BIN" ]; then
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
else
  say "Using existing keep binary: $KEEP_BIN"
fi

# 3. Database and ports (local mode only)
run mkdir -p "$HOME_DIR"
STORAGE=in-memory ; PG_URL="" ; QD_URL="" ; PG_PORT="" ; QD_PORT="" ; KEEP_PORT="$START_PORT"
if [ "$REMOTE" = 0 ]; then
  # Pick Keep HTTP port: reuse our existing running Keep port if present, else pick the first free port.
  EXISTING_PORT=""
  if [ -f "$HOME_DIR/env" ]; then
    EXISTING_BIND="$(sed -n 's/^KEEP_BIND=//p' "$HOME_DIR/env" | head -1)"
    EXISTING_PORT="${EXISTING_BIND##*:}"
  fi
  if [ "$PORT_EXPLICIT" = 0 ] && [ -n "$EXISTING_PORT" ] && curl -fsS "http://127.0.0.1:$EXISTING_PORT/v1/health" >/dev/null 2>&1; then
    KEEP_PORT="$EXISTING_PORT"
  else
    KEEP_PORT="$(find_free_port "$START_PORT")"
    if [ "$KEEP_PORT" != "$START_PORT" ]; then
      echo "Port $START_PORT is in use; using port $KEEP_PORT instead."
    fi
  fi
  BIND="127.0.0.1:$KEEP_PORT"
  URL="http://127.0.0.1:$KEEP_PORT"

  if [ "$DB" = vector ] || [ "$DB" = qdrant ]; then
    say "Starting the database with Docker (127.0.0.1 only)."
    run sh -c "curl -fsSL '$RAW/compose.yaml' -o '$HOME_DIR/compose.yaml'"
    profile=postgres-vector; [ "$DB" = qdrant ] && profile=postgres-qdrant

    # Reuse ports if keep-db is already running, else pick free ports.
    PG_PORT=""
    if [ "$DRY" != 1 ] && [ -f "$HOME_DIR/compose.yaml" ]; then
      pg_pub="$(docker compose -p keep-db -f "$HOME_DIR/compose.yaml" port postgres 5432 2>/dev/null | head -1 || true)"
      [ -n "$pg_pub" ] && PG_PORT="${pg_pub##*:}"
    fi
    if [ -z "$PG_PORT" ]; then
      PG_PORT="$(find_free_port 15433)"
      [ "$PG_PORT" != 15433 ] && echo "Port 15433 is in use; using Postgres port $PG_PORT instead."
    fi

    if [ "$DB" = qdrant ]; then
      QD_PORT=""
      if [ "$DRY" != 1 ] && [ -f "$HOME_DIR/compose.yaml" ]; then
        qd_pub="$(docker compose -p keep-db -f "$HOME_DIR/compose.yaml" port qdrant 6333 2>/dev/null | head -1 || true)"
        [ -n "$qd_pub" ] && QD_PORT="${qd_pub##*:}"
      fi
      if [ -z "$QD_PORT" ]; then
        QD_PORT="$(find_free_port 16333)"
        [ "$QD_PORT" != 16333 ] && echo "Port 16333 is in use; using Qdrant port $QD_PORT instead."
      fi
    fi

    # Patch downloaded compose.yaml if it still has fixed ports, then pass KEEP_PG_PORT/KEEP_QDRANT_PORT.
    if [ "$DRY" != 1 ] && [ -f "$HOME_DIR/compose.yaml" ]; then
      sed -i.bak \
        -e 's/127\.0\.0\.1:15433:5432/127.0.0.1:${KEEP_PG_PORT:-15433}:5432/g' \
        -e 's/127\.0\.0\.1:16333:6333/127.0.0.1:${KEEP_QDRANT_PORT:-16333}:6333/g' \
        "$HOME_DIR/compose.yaml" && rm -f "$HOME_DIR/compose.yaml.bak"
    fi
    KEEP_PG_PORT="$PG_PORT" KEEP_QDRANT_PORT="${QD_PORT:-16333}" \
      run docker compose -p keep-db -f "$HOME_DIR/compose.yaml" --profile "$profile" up -d --wait
    STORAGE=postgres-vector; PG_URL="postgres://keep:keep-local-only@127.0.0.1:$PG_PORT/keep"
    [ "$DB" = qdrant ] && { STORAGE=postgres-qdrant; QD_URL="http://127.0.0.1:$QD_PORT"; }
  elif [ "$DB" = external ]; then
    PG_URL="$EXT_PG_URL"
    if [ -n "$EXT_QD_URL" ]; then
      STORAGE=postgres-qdrant ; QD_URL="$EXT_QD_URL"
    else
      STORAGE=postgres-vector
    fi
  fi
fi

# 4. Settings, token, background service, and inspect helper
if [ -z "$USER_ID" ]; then
  USER_ID="$(printf '%s' "${USER:-me}" | tr -c 'A-Za-z0-9._-' '-' | cut -c1-64)"
fi
TOKEN="${KEEP_API_TOKEN:-}"
[ -z "$TOKEN" ] && [ -f "$HOME_DIR/env" ] && TOKEN="$(sed -n 's/^KEEP_API_TOKEN=//p' "$HOME_DIR/env")"
if [ -z "$TOKEN" ]; then
  if [ "$REMOTE" = 1 ]; then
    if [ "$DRY" = 1 ]; then
      TOKEN=dry-run-token
    else
      TOKEN="$(ask 'Remote Keep API token (KEEP_API_TOKEN):' '')"
      [ -n "$TOKEN" ] || { echo "KEEP_API_TOKEN is required for --server mode." >&2; exit 2; }
    fi
  elif [ "$DRY" = 1 ]; then
    TOKEN=dry-run-token
  else
    TOKEN="$("$KEEP_BIN" token new | head -1)"
  fi
fi

write_inspect_script() {
  [ "$DRY" = 1 ] && { echo "[dry-run] write $HOME_DIR/inspect.sh"; return 0; }
  cat > "$HOME_DIR/inspect.sh" <<'I'
#!/bin/sh
# Inspect your Keep memory files and knowledge graph.
# Usage:
#   ~/.keep/inspect.sh                 # list memory files + knowledge collections/entities
#   ~/.keep/inspect.sh --full          # also print the full text of every memory file
#   ~/.keep/inspect.sh --entity NAME   # traverse the knowledge graph around an entity
set -eu
D="${KEEP_HOME:-$HOME/.keep}"
[ -f "$D/env" ] || { echo "Keep env not found at $D/env" >&2; exit 1; }
set -a; . "$D/env"; set +a
URL="${KEEP_SERVER_URL:-http://${KEEP_BIND:-127.0.0.1:7357}}"
command -v python3 >/dev/null 2>&1 || { echo "python3 is required for inspect.sh" >&2; exit 1; }
KEEP_INSPECT_URL="$URL" python3 - "$@" <<'P'
import json, os, ssl, sys, urllib.request, urllib.error

url = os.environ["KEEP_INSPECT_URL"].rstrip("/")
token = os.environ.get("KEEP_API_TOKEN", "")
ca = os.environ.get("KEEP_SERVER_CA")
ctx = ssl.create_default_context(cafile=ca) if ca else None

full = "--full" in sys.argv
entity = None
if "--entity" in sys.argv:
    i = sys.argv.index("--entity")
    if i + 1 < len(sys.argv):
        entity = sys.argv[i + 1]

def call(path, body):
    req = urllib.request.Request(
        f"{url}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
        return json.loads(r.read().decode())

try:
    if entity:
        res = call("/v1/knowledge/traverse", {"start": entity, "depth": 2})
        print(f"=== Knowledge graph around: {entity} ===")
        print(json.dumps(res, indent=2))
        sys.exit(0)

    print(f"Keep server: {url}\n")
    print("=== Memory files ===")
    ml = call("/v1/memory/list", {})
    files = []
    for cat in ml.get("categories", []):
        for f in cat.get("files", []):
            files.append(f)
    if not files:
        print("  (no memory files yet)")
    for f in files:
        path = f.get("path", "")
        size = f.get("sizeBytes", f.get("size", ""))
        desc = f.get("description") or ""
        line = f"  - {path} ({size} B)" + (f" — {desc}" if desc else "")
        print(line)
        if full:
            doc = call("/v1/memory/read", {"path": path})
            text = doc.get("text", "")
            for tline in text.splitlines():
                print(f"      {tline}")
            print()

    print("\n=== Knowledge graph (collections & entities) ===")
    cols = call("/v1/knowledge/index", {"level": 0})
    entries = cols.get("entries") or cols.get("collections") or []
    if not entries:
        print("  (no knowledge sources yet)")
    else:
        for e in entries:
            print("  Collection:", json.dumps(e))
        ents = call("/v1/knowledge/index", {"level": 2})
        eitems = ents.get("entries") or ents.get("entities") or []
        if eitems:
            print("\n  Entities:")
            for e in eitems[:50]:
                name = e.get("name") or e.get("title") or e.get("id") or str(e)
                print(f"    - {name}")
            print('\n  Tip: run ~/.keep/inspect.sh --entity "<name>" to inspect facts around an entity.')
except urllib.error.URLError as err:
    print(f"Could not reach Keep at {url}: {err}", file=sys.stderr)
    sys.exit(1)
P
I
  chmod +x "$HOME_DIR/inspect.sh"
}

SERVICE_MODE="manual"
if [ "$REMOTE" = 1 ]; then
  say "Writing $HOME_DIR/env (remote server: $URL)"
  if [ "$DRY" != 1 ]; then
    umask 077
    {
      echo "KEEP_SERVER_URL=$URL"
      echo "KEEP_API_TOKEN=$TOKEN"
      echo "KEEP_USER=$USER_ID"
      [ -n "$SERVER_CA" ] && echo "KEEP_SERVER_CA=$SERVER_CA"
    } > "$HOME_DIR/env"
  fi
  write_inspect_script
  say "Checking remote Keep server."
  if [ "$DRY" != 1 ]; then
    if [ -n "$SERVER_CA" ]; then
      curl -fsS --cacert "$SERVER_CA" "$URL/v1/health" >/dev/null 2>&1 || { echo "Could not reach Keep at $URL/v1/health" >&2; exit 1; }
    else
      curl -fsS "$URL/v1/health" >/dev/null 2>&1 || { echo "Could not reach Keep at $URL/v1/health" >&2; exit 1; }
    fi
  fi
else
  say "Writing $HOME_DIR/env and helper scripts (port $KEEP_PORT)"
  if [ "$DRY" != 1 ]; then
    umask 077
    {
      echo "KEEP_API_TOKEN=$TOKEN"
      echo "KEEP_STORAGE=$STORAGE"
      echo "KEEP_USER=$USER_ID"
      echo "KEEP_BIND=$BIND"
      echo "KEEP_SERVER_URL=$URL"
      [ -n "$PG_URL" ] && echo "KEEP_POSTGRES_URL=$PG_URL"
      [ -n "$QD_URL" ] && echo "KEEP_QDRANT_URL=$QD_URL"
      [ -n "$PG_PORT" ] && echo "KEEP_PG_PORT=$PG_PORT"
      [ -n "$QD_PORT" ] && echo "KEEP_QDRANT_PORT=$QD_PORT"
      [ -n "$EXT_PG_CA" ] && echo "KEEP_POSTGRES_CA=$EXT_PG_CA"
      [ -n "$EXT_QD_CA" ] && echo "KEEP_QDRANT_CA=$EXT_QD_CA"
      [ -n "$EXT_QD_KEY" ] && echo "KEEP_QDRANT_API_KEY=$EXT_QD_KEY"
    } > "$HOME_DIR/env"
    cat > "$HOME_DIR/run.sh" <<'R'
#!/bin/sh
# Foreground runner used by launchd / systemd.
set -eu
D="${KEEP_HOME:-$HOME/.keep}"
set -a; . "$D/env"; set +a
PATH="$HOME/.cargo/bin:$PATH"
exec keep serve
R
    cat > "$HOME_DIR/start.sh" <<'S'
#!/bin/sh
# Starts the Keep daemon (via launchd / systemd if installed, else background process).
D="${KEEP_HOME:-$HOME/.keep}"
if [ "$(uname -s)" = Darwin ] && [ -f "$HOME/Library/LaunchAgents/ai.tulipfarm.keep.plist" ]; then
  UID_NUM="$(id -u)"
  launchctl bootstrap "gui/$UID_NUM" "$HOME/Library/LaunchAgents/ai.tulipfarm.keep.plist" >/dev/null 2>&1 || true
  launchctl kickstart -k "gui/$UID_NUM/ai.tulipfarm.keep" >/dev/null 2>&1 || true
  echo "Keep started (launchd). Log: $D/keep.log"
  exit 0
fi
if [ "$(uname -s)" = Linux ] && [ -f "$HOME/.config/systemd/user/keep.service" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl --user start keep.service
  echo "Keep started (systemd --user). Log: journalctl --user -u keep.service"
  exit 0
fi
set -a; . "$D/env"; set +a
if [ -f "$D/keep.pid" ] && kill -0 "$(cat "$D/keep.pid")" 2>/dev/null; then echo "Keep is already running."; exit 0; fi
PATH="$HOME/.cargo/bin:$PATH"
nohup keep serve >"$D/keep.log" 2>&1 &
echo $! > "$D/keep.pid"
echo "Keep started. Log: $D/keep.log"
S
    cat > "$HOME_DIR/stop.sh" <<'S'
#!/bin/sh
D="${KEEP_HOME:-$HOME/.keep}"
if [ "$(uname -s)" = Darwin ] && [ -f "$HOME/Library/LaunchAgents/ai.tulipfarm.keep.plist" ]; then
  launchctl bootout "gui/$(id -u)/ai.tulipfarm.keep" >/dev/null 2>&1 || true
  echo "Keep stopped (launchd)."
  exit 0
fi
if [ "$(uname -s)" = Linux ] && [ -f "$HOME/.config/systemd/user/keep.service" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl --user stop keep.service >/dev/null 2>&1 || true
  echo "Keep stopped (systemd --user)."
  exit 0
fi
[ -f "$D/keep.pid" ] && kill "$(cat "$D/keep.pid")" 2>/dev/null && rm -f "$D/keep.pid" && echo "Keep stopped." || echo "Keep is not running."
S
    chmod +x "$HOME_DIR/run.sh" "$HOME_DIR/start.sh" "$HOME_DIR/stop.sh"
  fi
  write_inspect_script

  # Install OS background service so Keep starts automatically on login and restarts on crash.
  if [ "$INSTALL_SERVICE" = 1 ]; then
    if [ "$OS" = Darwin ] && have launchctl; then
      PLIST="$HOME/Library/LaunchAgents/ai.tulipfarm.keep.plist"
      say "Installing macOS background service ($PLIST)."
      if [ "$DRY" = 1 ]; then
        echo "[dry-run] write $PLIST and load with launchctl"
      else
        mkdir -p "$HOME/Library/LaunchAgents"
        cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.tulipfarm.keep</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/sh</string>
    <string>$HOME_DIR/run.sh</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$HOME_DIR/keep.log</string>
  <key>StandardErrorPath</key>
  <string>$HOME_DIR/keep.log</string>
</dict>
</plist>
PL
        UID_NUM="$(id -u)"
        launchctl bootout "gui/$UID_NUM/ai.tulipfarm.keep" >/dev/null 2>&1 || true
        if launchctl bootstrap "gui/$UID_NUM" "$PLIST" >/dev/null 2>&1; then
          launchctl kickstart -k "gui/$UID_NUM/ai.tulipfarm.keep" >/dev/null 2>&1 || true
          SERVICE_MODE="launchd"
        else
          echo "Could not load LaunchAgent; falling back to start.sh." >&2
          rm -f "$PLIST"
        fi
      fi
      [ "$DRY" = 1 ] && SERVICE_MODE="launchd"
    elif [ "$OS" = Linux ] && have systemctl && systemctl --user show-environment >/dev/null 2>&1; then
      UNIT="$HOME/.config/systemd/user/keep.service"
      say "Installing Linux user service ($UNIT)."
      if [ "$DRY" = 1 ]; then
        echo "[dry-run] write $UNIT and enable with systemctl --user"
      else
        mkdir -p "$HOME/.config/systemd/user"
        cat > "$UNIT" <<UN
[Unit]
Description=Keep memory server
After=network-online.target

[Service]
Type=simple
ExecStart=/bin/sh $HOME_DIR/run.sh
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
UN
        systemctl --user daemon-reload
        systemctl --user enable --now keep.service
        SERVICE_MODE="systemd --user"
      fi
      [ "$DRY" = 1 ] && SERVICE_MODE="systemd --user"
    fi
  fi

  if [ "$SERVICE_MODE" = manual ]; then
    say "Starting Keep."
    run "$HOME_DIR/start.sh"
  fi
  if [ "$DRY" != 1 ]; then
    i=0
    until curl -fsS "$URL/v1/health" >/dev/null 2>&1 || [ $i -ge 60 ]; do i=$((i+1)); sleep 1; done
    curl -fsS "$URL/v1/health" >/dev/null 2>&1 || { echo "Keep did not start. See $HOME_DIR/keep.log" >&2; exit 1; }
  fi
fi

# 5. Connect coding agents (personal scope: no --project)
MCP_ARGS="--user $USER_ID mcp --server $URL"
connect_claude() {
  have claude || return 1
  run claude mcp remove keep --scope user >/dev/null 2>&1 || true
  if [ -n "$SERVER_CA" ]; then
    # shellcheck disable=SC2086
    run claude mcp add keep --scope user -e "KEEP_API_TOKEN=$TOKEN" -e KEEP_AGENT=claude-code -e "KEEP_SERVER_CA=$SERVER_CA" -- "$KEEP_BIN" $MCP_ARGS
  else
    # shellcheck disable=SC2086
    run claude mcp add keep --scope user -e "KEEP_API_TOKEN=$TOKEN" -e KEEP_AGENT=claude-code -- "$KEEP_BIN" $MCP_ARGS
  fi
}
connect_codex() {
  have codex || [ -d "$HOME/.codex" ] || return 1
  f="$HOME/.codex/config.toml"
  if [ -f "$f" ] && grep -q '^\[mcp_servers\.keep\]' "$f"; then echo "Codex already has a keep entry. Left it as is."; return 0; fi
  [ "$DRY" = 1 ] && { echo "[dry-run] append [mcp_servers.keep] to $f"; return 0; }
  mkdir -p "$HOME/.codex"
  ca_kv=""
  [ -n "$SERVER_CA" ] && ca_kv=", KEEP_SERVER_CA = \"$SERVER_CA\""
  cat >> "$f" <<T

[mcp_servers.keep]
command = "$KEEP_BIN"
args = ["--user", "$USER_ID", "mcp", "--server", "$URL"]
env = { KEEP_API_TOKEN = "$TOKEN", KEEP_AGENT = "codex"$ca_kv }
T
}
connect_copilot() {
  have copilot || [ -d "$HOME/.copilot" ] || return 1
  f="$HOME/.copilot/mcp-config.json"
  [ "$DRY" = 1 ] && { echo "[dry-run] add keep to $f"; return 0; }
  have python3 || { echo "python3 is missing, so add keep to $f by hand. See $REPO/blob/main/docs/server.md#copilot-cli" >&2; return 0; }
  mkdir -p "$HOME/.copilot"
  KB="$KEEP_BIN" KT="$TOKEN" KU="$USER_ID" KURL="$URL" KCA="$SERVER_CA" python3 - "$f" <<'P'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p)) if os.path.exists(p) else {}
env = {"KEEP_API_TOKEN": os.environ["KT"], "KEEP_AGENT": "copilot-cli"}
if os.environ.get("KCA"):
    env["KEEP_SERVER_CA"] = os.environ["KCA"]
d.setdefault("mcpServers", {})["keep"] = {
    "type": "local", "command": os.environ["KB"],
    "args": ["--user", os.environ["KU"], "mcp", "--server", os.environ["KURL"]],
    "env": env,
    "tools": ["*"],
}
json.dump(d, open(p, "w"), indent=2)
P
}
connect_opencode() {
  have opencode || [ -d "$HOME/.config/opencode" ] || return 1
  if have opencode; then
    if [ -n "$SERVER_CA" ]; then
      # shellcheck disable=SC2086
      run opencode mcp add keep --global --env "KEEP_API_TOKEN=$TOKEN" --env KEEP_AGENT=opencode --env "KEEP_SERVER_CA=$SERVER_CA" -- "$KEEP_BIN" $MCP_ARGS && return 0
    else
      # shellcheck disable=SC2086
      run opencode mcp add keep --global --env "KEEP_API_TOKEN=$TOKEN" --env KEEP_AGENT=opencode -- "$KEEP_BIN" $MCP_ARGS && return 0
    fi
  fi
  f="$HOME/.config/opencode/opencode.json"
  [ "$DRY" = 1 ] && { echo "[dry-run] add keep to $f"; return 0; }
  if [ -f "$HOME/.config/opencode/opencode.jsonc" ] && [ ! -f "$f" ]; then
    echo "OpenCode uses opencode.jsonc; add keep with 'opencode mcp add keep --global ...' or by hand." >&2
    return 0
  fi
  have python3 || { echo "python3 is missing, so add keep to $f by hand. See $REPO/blob/main/docs/server.md#opencode" >&2; return 0; }
  mkdir -p "$HOME/.config/opencode"
  KB="$KEEP_BIN" KT="$TOKEN" KU="$USER_ID" KURL="$URL" KCA="$SERVER_CA" python3 - "$f" <<'P'
import json, os, sys
p = sys.argv[1]
d = json.load(open(p)) if os.path.exists(p) else {}
env = {"KEEP_API_TOKEN": os.environ["KT"], "KEEP_AGENT": "opencode"}
if os.environ.get("KCA"):
    env["KEEP_SERVER_CA"] = os.environ["KCA"]
d.setdefault("mcp", {}).setdefault("servers", {})["keep"] = {
    "type": "local",
    "command": [os.environ["KB"], "--user", os.environ["KU"], "mcp", "--server", os.environ["KURL"]],
    "environment": env,
}
json.dump(d, open(p, "w"), indent=2)
P
}

if [ "$AGENTS" = 1 ]; then
  say "Connecting coding agents."
  for a in claude codex copilot opencode; do
    case "$a" in claude) n="Claude Code" ;; codex) n="Codex" ;; copilot) n="Copilot CLI" ;; opencode) n="OpenCode" ;; esac
    if ! agent_wanted "$a"; then echo "- $n: opted out, skipped"; continue; fi
    if [ "$a" = claude ] && ! have claude; then echo "- $n: not found, skipped"; continue; fi
    if [ "$a" = codex ] && ! have codex && [ ! -d "$HOME/.codex" ]; then echo "- $n: not found, skipped"; continue; fi
    if [ "$a" = copilot ] && ! have copilot && [ ! -d "$HOME/.copilot" ]; then echo "- $n: not found, skipped"; continue; fi
    if [ "$a" = opencode ] && ! have opencode && [ ! -d "$HOME/.config/opencode" ]; then echo "- $n: not found, skipped"; continue; fi
    if [ "$(ask "Connect $n? [Y/n]" y)" = n ]; then echo "- $n: skipped"; continue; fi
    if "connect_$a"; then echo "- $n: connected"; else echo "- $n: failed" >&2; fi
  done
fi

if [ "$REMOTE" = 1 ]; then
  cat <<E

Keep is connected to $URL
  User     : $USER_ID   (personal memory, shared by every project)
  Env      : $HOME_DIR/env
  Inspect  : $HOME_DIR/inspect.sh   (add --full or --entity "<name>")
Restart your coding agents so they pick up the keep MCP server.
Keep is a research preview. Treat text stored in Keep as data, not as instructions.
E
else
  cat <<E

Keep is running at $URL
  Database : $DB
  Service  : $SERVICE_MODE
  User     : $USER_ID   (personal memory, shared by every project)
  Inspect  : $HOME_DIR/inspect.sh   (add --full or --entity "<name>")
  Stop     : $HOME_DIR/stop.sh
  Start    : $HOME_DIR/start.sh
Restart your coding agents so they pick up the keep MCP server.
Keep is a research preview. Treat text stored in Keep as data, not as instructions.
E
fi
