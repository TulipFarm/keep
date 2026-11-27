import github from "../assets/logos/github.svg?raw";
import claude from "../assets/logos/claude-code.svg?raw";
import codex from "../assets/logos/codex.svg?raw";
import copilot from "../assets/logos/copilot.svg?raw";
import mcp from "../assets/logos/mcp.svg?raw";
import postgres from "../assets/logos/postgresql.svg?raw";
import qdrant from "../assets/logos/qdrant.svg?raw";
import docker from "../assets/logos/docker.svg?raw";

export const logos: Record<string, string[]> = {
  "Claude Code": [claude],
  "Claude Desktop": [claude],
  Codex: [codex],
  "Copilot CLI": [copilot],
  "Any MCP client": [mcp],
  "Postgres + pgvector": [postgres],
  "Postgres + Qdrant": [postgres, qdrant],
  GitHub: [github],
  Docker: [docker],
};
