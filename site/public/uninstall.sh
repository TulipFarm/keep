#!/bin/sh
# Keep uninstaller. Usage:
#   curl -fsSL https://keep.tulipfarm.ai/uninstall.sh | sh
#   curl -fsSL https://keep.tulipfarm.ai/uninstall.sh | sh -s -- --yes
#   curl -fsSL https://keep.tulipfarm.ai/uninstall.sh | sh -s -- --purge --yes
# Flags:
#   --purge                               also delete Docker DB volumes and local data/model cache
#   --agents claude,codex,copilot,opencode  only disconnect specific agents
#   --keep-agents                         leave coding agent MCP configs untouched
#   --keep-binary                         leave the keep binary installed
#   --yes | -y                            non-interactive (does not enable --purge)
#   --dry-run                             show what would be done without changing anything
set -eu

HOME_DIR="${KEEP_HOME:-$HOME/.keep}"

PURGE=0 ; YES=0 ; AGENTS=1 ; AGENTS_LIST="" ; KEEP_BIN_FLAG=0 ; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --purge) PURGE=1 ;;
    --agents) AGENTS=1; AGENTS_LIST="$2"; shift ;;
    --keep-agents) AGENTS=0 ;;
    --keep-binary) KEEP_BIN_FLAG=1 ;;
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

OS="$(uname -s)"
[ "$OS" = Darwin ] || [ "$OS" = Linux ] || { echo "Keep uninstalls on macOS and Linux only." >&2; exit 1; }

if [ -n "${KEEP_DATA_DIR:-}" ]; then
  DATA_DIR="$KEEP_DATA_DIR"
elif [ -n "${XDG_DATA_HOME:-}" ]; then
  DATA_DIR="$XDG_DATA_HOME/keep"
elif [ "$OS" = Darwin ]; then
  DATA_DIR="$HOME/Library/Application Support/keep"
else
  DATA_DIR="$HOME/.local/share/keep"
fi

if [ "$PURGE" = 0 ] && [ "$YES" = 0 ] && [ -r /dev/tty ]; then
  case "$(ask 'Also delete stored Docker database volumes and local model cache? [y/N]' n)" in
    y|Y|yes|YES) PURGE=1 ;;
  esac
fi

# 1. Stop the local Keep daemon and remove OS background service
say "Stopping Keep daemon and background service."
if [ "$OS" = Darwin ] && [ -f "$HOME/Library/LaunchAgents/ai.tulipfarm.keep.plist" ]; then
  if have launchctl; then
    run launchctl bootout "gui/$(id -u)/ai.tulipfarm.keep" >/dev/null 2>&1 || true
  fi
  run rm -f "$HOME/Library/LaunchAgents/ai.tulipfarm.keep.plist"
  echo "Removed macOS LaunchAgent."
elif [ "$OS" = Linux ] && [ -f "$HOME/.config/systemd/user/keep.service" ]; then
  if have systemctl; then
    run systemctl --user disable --now keep.service >/dev/null 2>&1 || true
    run rm -f "$HOME/.config/systemd/user/keep.service"
    run systemctl --user daemon-reload >/dev/null 2>&1 || true
  else
    run rm -f "$HOME/.config/systemd/user/keep.service"
  fi
  echo "Removed Linux systemd user service."
elif [ -x "$HOME_DIR/stop.sh" ]; then
  run "$HOME_DIR/stop.sh" || true
elif [ -f "$HOME_DIR/keep.pid" ]; then
  pid="$(cat "$HOME_DIR/keep.pid" 2>/dev/null || true)"
  if [ -n "$pid" ]; then
    run kill "$pid" 2>/dev/null || true
  fi
  run rm -f "$HOME_DIR/keep.pid"
else
  echo "No local Keep daemon or service found."
fi

# 2. Disconnect coding agents
disconnect_claude() {
  have claude || return 1
  if [ "$DRY" = 1 ]; then
    echo "[dry-run] claude mcp remove keep --scope user"
    return 0
  fi
  claude mcp remove keep --scope user >/dev/null 2>&1 || return 1
}
disconnect_codex() {
  f="$HOME/.codex/config.toml"
  [ -f "$f" ] || return 1
  grep -q '^\[mcp_servers\.keep\]' "$f" || return 1
  [ "$DRY" = 1 ] && { echo "[dry-run] remove [mcp_servers.keep] from $f"; return 0; }
  have python3 || { echo "python3 is missing, so remove [mcp_servers.keep] from $f by hand." >&2; return 0; }
  python3 - "$f" <<'P'
import re, sys
p = sys.argv[1]
text = open(p).read()
# Remove the [mcp_servers.keep] table and its keys up to the next [section] header or EOF.
updated = re.sub(r'(?ms)^\n?\[mcp_servers\.keep\]\n.*?(?=^\[|\Z)', '', text)
if updated != text:
    open(p, "w").write(updated)
P
}
disconnect_copilot() {
  f="$HOME/.copilot/mcp-config.json"
  [ -f "$f" ] || return 1
  grep -q '"keep"' "$f" || return 1
  [ "$DRY" = 1 ] && { echo "[dry-run] remove keep from $f"; return 0; }
  have python3 || { echo "python3 is missing, so remove keep from $f by hand." >&2; return 0; }
  python3 - "$f" <<'P'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
servers = d.get("mcpServers")
if isinstance(servers, dict) and "keep" in servers:
    del servers["keep"]
    with open(p, "w") as fh:
        json.dump(d, fh, indent=2)
        fh.write("\n")
P
}
disconnect_opencode() {
  removed=0
  if have opencode; then
    if run opencode mcp remove keep --global >/dev/null 2>&1; then
      removed=1
    fi
  fi
  f="$HOME/.config/opencode/opencode.json"
  if [ -f "$f" ] && grep -q '"keep"' "$f"; then
    if [ "$DRY" = 1 ]; then
      echo "[dry-run] remove keep from $f"
      removed=1
    elif have python3; then
      python3 - "$f" <<'P'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
changed = False
mcp = d.get("mcp")
if isinstance(mcp, dict):
    servers = mcp.get("servers")
    if isinstance(servers, dict) and "keep" in servers:
        del servers["keep"]
        changed = True
    if "keep" in mcp and isinstance(mcp["keep"], dict):
        del mcp["keep"]
        changed = True
if changed:
    with open(p, "w") as fh:
        json.dump(d, fh, indent=2)
        fh.write("\n")
P
      removed=1
    else
      echo "python3 is missing, so remove keep from $f by hand." >&2
    fi
  elif [ "$removed" = 0 ] && [ -f "$HOME/.config/opencode/opencode.jsonc" ] && grep -q '"keep"' "$HOME/.config/opencode/opencode.jsonc"; then
    echo "OpenCode uses opencode.jsonc; remove keep with 'opencode mcp remove keep --global' or by hand." >&2
    removed=1
  fi
  [ "$removed" = 1 ]
}

if [ "$AGENTS" = 1 ]; then
  say "Disconnecting coding agents."
  for a in claude codex copilot opencode; do
    case "$a" in claude) n="Claude Code" ;; codex) n="Codex" ;; copilot) n="Copilot CLI" ;; opencode) n="OpenCode" ;; esac
    if ! agent_wanted "$a"; then echo "- $n: skipped"; continue; fi
    if "disconnect_$a"; then echo "- $n: disconnected"; else echo "- $n: not configured, skipped"; fi
  done
fi

# 3. Stop Docker database containers (if created by install.sh)
if [ -f "$HOME_DIR/compose.yaml" ]; then
  say "Stopping Docker database containers."
  if have docker; then
    if [ "$PURGE" = 1 ]; then
      run docker compose -p keep-db -f "$HOME_DIR/compose.yaml" --profile postgres-vector --profile postgres-qdrant down -v || true
    else
      run docker compose -p keep-db -f "$HOME_DIR/compose.yaml" --profile postgres-vector --profile postgres-qdrant down || true
    fi
  else
    echo "docker not found; skipping container cleanup."
  fi
fi

# 4. Remove ~/.keep config/scripts
if [ -d "$HOME_DIR" ]; then
  say "Removing $HOME_DIR."
  run rm -rf "$HOME_DIR"
fi

# 5. Optionally remove platform data dir (receipt key + ONNX model cache)
if [ "$PURGE" = 1 ] && [ -d "$DATA_DIR" ]; then
  say "Removing local data and model cache ($DATA_DIR)."
  run rm -rf "$DATA_DIR"
fi

# 6. Remove the keep binary
if [ "$KEEP_BIN_FLAG" = 0 ]; then
  if ! have cargo && [ -x "$HOME/.cargo/bin/cargo" ]; then PATH="$HOME/.cargo/bin:$PATH"; fi
  if [ -x "$HOME/.cargo/bin/keep" ]; then
    say "Removing keep binary."
    if have cargo; then
      run cargo uninstall keep-server || run rm -f "$HOME/.cargo/bin/keep"
    else
      run rm -f "$HOME/.cargo/bin/keep"
    fi
  elif have keep; then
    say "Found keep binary outside ~/.cargo/bin ($(command -v keep)). Remove it manually if desired."
  fi
fi

cat <<E

Keep has been uninstalled.
E
if [ "$PURGE" = 0 ]; then
  cat <<E
  Note: Docker database volumes (if any) and local model cache ($DATA_DIR) were kept.
        Re-run with --purge to delete them as well.
E
fi
cat <<E
  Note: Global instruction files (CLAUDE.md, AGENTS.md, copilot-instructions.md) were left untouched.
        Remove any "## Keep Memory" section manually if you added one, and restart your coding agents.
E
