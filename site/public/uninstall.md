# Uninstall Keep

You are helping someone remove Keep from their computer.

**Important — who you are talking to:**
The person you are helping may not be a programmer. Do not use jargon like "launchd", "systemd", "Docker volumes", "MCP", "TOML", or "stdio" unless they ask. Speak in plain, friendly English. Never delete their saved memories permanently unless they explicitly ask you to.

Do the steps below in order.

## 1. Check what is installed (before asking anything)

Run quiet read-only checks first so you only ask about things that actually exist:

- **Keep installation**: check `~/.keep`, `~/Library/LaunchAgents/ai.tulipfarm.keep.plist` (macOS), `~/.config/systemd/user/keep.service` (Linux), `command -v keep` / `~/.cargo/bin/keep`, and `docker ps -a --filter name=keep-db` (if Docker is installed).
- **Connected apps**: check for a `"keep"` entry in:
  - Claude Code (`claude mcp list` or `~/.claude.json`)
  - Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `~/.config/Claude/claude_desktop_config.json` on Linux)
  - Codex (`~/.codex/config.toml`)
  - Copilot CLI (`~/.copilot/mcp-config.json`)
  - OpenCode (`~/.config/opencode/opencode.json` or `opencode.jsonc`)
  - Cursor (`~/.cursor/mcp.json`)
  - Windsurf (`~/.codeium/windsurf/mcp_config.json`)
- **Instruction notes**: check if `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, `~/.copilot/copilot-instructions.md`, or `~/.config/opencode/AGENTS.md` contain a Keep reminder added during setup.

If nothing from Keep is found, tell the user clearly that Keep does not appear to be installed and stop.

## 2. Ask in plain English

Briefly list what you found (e.g., "Keep is running in the background and connected to Claude Desktop, Claude Code, and Codex"), then ask:

1. **What should I do with your saved Keep memories?**
   - **Keep them safe (recommended)** — removes Keep and disconnects your apps, but leaves your saved database intact in case you ever reinstall.
   - **Save a Markdown backup first, then remove Keep** — exports your Keep memories to a readable folder on your computer (e.g. `~/keep-backup`) before uninstalling.
   - **Delete everything permanently** — erases Keep's local database and downloaded model files too. *(Reassure them: your AI apps' own original memory files were never deleted by Keep and will stay untouched.)*

2. *(Only if you found Keep notes in global instruction files)* **Remove the Keep note from your AI apps' instruction files?**
   - Default: **Yes** — removes only the Keep reminder, leaving all your other instructions untouched.

Wait for their answer before changing anything.

## 3. Export first (only if they asked for a backup)

If they chose to save a Markdown backup and Keep is running (or can be queried via `~/.keep/env`):

- Export their memories via `memory_export` (or `GET $KEEP_URL/v1/memory/export` with the token from `~/.keep/env`) and write the Markdown files to `~/keep-backup` (or another folder they chose).
- Show them the folder path and the list of exported files before continuing.

## 4. Run the uninstaller

Run the official uninstall script silently behind the scenes:

- **Keeping saved data**:
  ```sh
  curl -fsSL https://keep.tulipfarm.ai/uninstall.sh | sh -s -- --yes
  ```
- **Deleting everything permanently** (only if the user explicitly chose this):
  ```sh
  curl -fsSL https://keep.tulipfarm.ai/uninstall.sh | sh -s -- --yes --purge
  ```

`uninstall.sh` stops and removes the background service (`launchd` / `systemd`), disconnects Claude Code, Codex, Copilot CLI, and OpenCode, stops the local database container (keeping volumes unless `--purge`), and removes `~/.keep` and `~/.cargo/bin/keep`.

## 5. Disconnect desktop apps & clean up instruction notes

1. **Desktop apps** (`Claude Desktop`, `Cursor`, `Windsurf`): if any of their config files have a `"keep"` entry under `mcpServers`, remove **only** the `"keep"` key — preserve every other app/server in the JSON file.
2. **Instruction notes** (only if confirmed in step 2): remove **only** the short Keep pointer lines from `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, `~/.copilot/copilot-instructions.md`, or `~/.config/opencode/AGENTS.md`. Do not modify any other instructions.

## 6. Verify and report clearly

Confirm everything is clean, then give a short, friendly summary:

- Keep has been stopped and removed from background startup.
- List every app that was disconnected.
- State clearly whether their saved database was kept, backed up (with the folder path), or permanently erased.
- Remind them to restart any open AI apps or terminal sessions so the disconnect takes effect.
