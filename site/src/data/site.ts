export const repo = "https://github.com/TulipFarm/keep";
export const docs = {
  server: `${repo}/blob/main/docs/server.md`,
  memory: `${repo}/blob/main/docs/memory.md`,
  knowledge: `${repo}/blob/main/docs/knowledge.md`,
  method: `${repo}/blob/main/docs/benchmarks/methodology.md`,
  issues: `${repo}/issues`,
};

export const meta = {
  title: "Keep by TulipFarm: open-source memory for AI agents",
  description:
    "Keep is open-source memory and knowledge for AI agents, over MCP. Old facts are marked, not erased. Apache-2.0 research preview. Runs on your machine.",
};

export const setupPrompt = "Read https://keep.tulipfarm.ai/install.md and install Keep for me.";

export const install = {
  heading: "Install in one step",
  lead: "Keep asks where to store memory, then connects your coding agents. Runs on macOS and Linux.",
  agent: { title: "Ask your agent", cmd: setupPrompt, note: "Paste this into Claude Code, Codex or Copilot CLI. Your agent runs the installer, then checks that Keep is working." },
  script: { title: "Or run the script", cmd: "curl -fsSL https://keep.tulipfarm.ai/install.sh | sh", note: "Installs Rust if needed and builds Keep from source. The first build takes a few minutes. The Postgres options need Docker." },
  links: [["Read install.md", "/install.md"], ["View install.sh", "/install.sh"]] as [string, string][],
  copy: "Copy",
  copied: "Copied",
};

export const hero = {
  badge: "Open source · Apache-2.0 · Research preview",
  headline: "Memory your agents can't quietly rewrite.",
  sub: "Keep is open-source memory and knowledge for AI agents. Keep gives Claude Code, Codex and Copilot CLI one shared memory per project. Every change is a revision. When a fact changes, the old one is marked superseded, not deleted.",
  note: "Keep is a research preview. It works, but it is early.",
  primary: "Get Keep on GitHub",
  secondary: "Read the setup docs",
  copy: "Copy setup prompt",
  worksWith: ["Claude Code", "Codex", "Copilot CLI", "Any MCP client"],
  storesIn: ["In memory", "Postgres + pgvector", "Postgres + Qdrant"],
  snippetLabel: "Install from source, then connect any MCP agent",
  snippet: [
    "cargo install --path crates/keep-server --features local-embeddings",
    "keep --user me serve &",
    "keep --project my-app mcp --server http://127.0.0.1:7357",
  ],
};

export const flow = {
  caption: "What goes in, what Keep stores, what it derives, what your agent gets.",
  steps: [
    ["What goes in", "Your notes, documents and conversations."],
    ["What Keep stores", "Memory files as revisions. Episodes that are never edited."],
    ["What Keep derives", "Facts, each linked to the exact text behind it."],
    ["What your agent gets", "Memory at session start. Cited search on demand."],
  ] as [string, string][],
};

export const does: [string, string][] = [
  ["Remembers what you decided.", "Short Markdown files your agent reads at the start of every session."],
  ["Tracks what changed.", "When a fact changes, the old one is closed, not deleted."],
  ["Shows where it came from.", "Every fact links to the text that supports it."],
];

export const fandom = {
  heading: "See how Keep files a fandom",
  lead: "We gave Keep 142 pages of the Harry Potter Wiki. It read them into the people, places and things they name, the facts about each, and the page each fact came from. Click any dot.",
  hint: "Scroll to zoom · drag to move · click a dot",
};

export const where = {
  heading: "Where it runs",
  lead: "On your machine. Keep calls no hosted service.",
  items: [
    ["On your laptop", "Memory stays in the server's process. Nothing else to install.", "keep --user me serve &"],
    ["With your own Postgres", "Memory survives restarts. Use pgvector, or add Qdrant.", "docker compose --profile postgres-vector up -d"],
    ["In Docker Compose", "Run Keep itself in Compose, with generated secrets.", "deploy/compose.yaml"],
  ] as [string, string, string][],
  link: ["Run with Docker Compose", `${repo}/blob/main/docs/server.md#run-with-docker-compose`] as [string, string],
};

export const demo = {
  heading: "Pick a date. See what was true.",
  steps: ["Pick a date", "See what was true", "See what changed"],
  legend: ["True on this date", "Replaced later. Still stored.", "Not recorded yet"],
  label: "Example data",
  caption: "Sample data from an example project.",
  empty: "Nothing was recorded for this date. Pick a later date.",
};

export type Section = {
  id: string;
  n: number;
  eyebrow: string;
  title: string;
  lead: string;
  body: string;
  points: [string, string][];
  link?: [string, string];
};

export const sections: Section[] = [
  {
    id: "memory",
    n: 1,
    eyebrow: "Memory files",
    title: "Memory files",
    lead: "Short Markdown files your agent reads at the start of every session.",
    body: "",
    points: [
      ["Plain Markdown.", "Saved byte for byte."],
      ["Safe edits.", "A stale write fails with a conflict."],
      ["Undo is a new revision.", "It never rewrites history."],
      ["Revisions kept.", "Up to 100 per file, for 30 days."],
    ],
    link: ["How Memory works", docs.memory],
  },
  {
    id: "knowledge",
    n: 2,
    eyebrow: "Knowledge",
    title: "Knowledge",
    lead: "A record of what happened, and the facts drawn from it.",
    body: "",
    points: [
      ["Episodes are never edited.", ""],
      ["Sources.", "Each fact links to the exact text behind it."],
      ["Two kinds of time.", "Ask what was true on a date, or what Keep knew."],
      ["Disputes stay visible.", "If order is unclear, Keep keeps both."],
    ],
    link: ["How Knowledge works", docs.knowledge],
  },
  {
    id: "connect",
    n: 3,
    eyebrow: "Connect",
    title: "One memory, every agent",
    lead: "Agents on the same project share the same memory.",
    body: "",
    points: [
      ["MCP over stdio", "for coding agents."],
      ["HTTP", "for scripts and dashboards."],
      ["Local.", "Listens on 127.0.0.1 by default."],
      ["Scoped tokens", "limit what each agent can reach."],
    ],
    link: ["Copy-paste setup for each agent", docs.server],
  },
  {
    id: "status",
    n: 4,
    eyebrow: "Status",
    title: "Status",
    lead: "Keep is a research preview. We have not yet shown it beats simple baselines.",
    body: "",
    points: [
      ["Runs today:", "Memory, Knowledge and the server."],
      ["Alpha.", "The server is an alpha."],
      ["No benchmark result", "is published yet."],
      ["Not passed.", "Our development runs have not passed the gate. We quote no numbers."],
    ],
    link: ["Read the methodology", docs.method],
  },
  {
    id: "open-source",
    n: 5,
    eyebrow: "Open source",
    title: "Open source, Apache-2.0",
    lead: "Read the code, run the benchmark, check our work.",
    body: "",
    points: [
      ["Written in Rust.", ""],
      ["One public repository", "holds Memory, Knowledge, the server and the benchmark."],
      ["Datasets keep their own licences.", "You download them yourself."],
    ],
    link: ["View the repository", repo],
  },
];

export const faq: [string, string][] = [
  [
    "What is Keep?",
    "Keep is an open-source memory and knowledge system for AI agents, written in Rust. It stores Memory files and a record of facts, and serves them to coding agents over MCP. It is a research preview.",
  ],
  [
    "How is Keep different from RAG or a vector database?",
    "Keep does use search, both keyword and vector. It also stores facts with their source and the time range when they were true. When a fact changes, the old version is closed, not overwritten, so you can ask what was true on a date.",
  ],
  [
    "Which agents and models does Keep work with?",
    "Claude Code, Codex and Copilot CLI. The installer connects each one that is installed. Any MCP client that can start a stdio server works too. Keep sets no chat model for you. Your agent keeps its own.",
  ],
  [
    "Can I run Keep myself?",
    "Yes. Ask your coding agent to follow install.md, or run install.sh yourself. Both build Keep on your machine and store memory in process or in a local Postgres. To build by hand, use `cargo install`. Docker Compose files are in the repo.",
  ],
  ["Is Keep open source?", "Yes. Keep is Apache-2.0 and the code is public on GitHub."],
  [
    "How much does Keep cost?",
    "Keep is open source and you run it yourself. The repo states no hosted service and no price.",
  ],
  [
    "Is Keep ready for production?",
    "No. Keep is a research preview and the server is an alpha. The benchmark that decides whether Keep is worth using is not finished. Use it where losing a note is acceptable.",
  ],
  [
    "Does Keep send my data to a model provider?",
    "No. The server never calls a hosted model API. It embeds Knowledge with a local model, downloaded once on first start. It talks only to the Postgres and Qdrant you configure.",
  ],
  [
    "Does Keep ever delete anything?",
    "Only when you ask. Edits, undos and changed facts become new revisions or fact versions. Old revisions are kept for up to 30 days, and the forget operation erases content for good.",
  ],
];

export const footer = "Keep is open source under Apache-2.0 and is a research preview from TulipFarm.";

export const facts = [
  { old: "Run tests with npm test", oldWhen: "2026-03-02 → 2026-09-14", now: "Run tests with pnpm vitest run", nowWhen: "2026-09-14 → now" },
  { old: "Deploy target is Heroku", oldWhen: "2026-01-20 → 2026-06-03", now: "Deploy target is Fly.io", nowWhen: "2026-06-03 → now" },
  { old: "Billing API uses Node 18", oldWhen: "2026-02-11 → 2026-10-08", now: "Billing API uses Node 22", nowWhen: "2026-10-08 → now" },
];

export type Line = ["", string] | ["add", string] | ["del", string];
export const revisions: Line[][] = [
  [["", "# my-app"], ["", "- Run tests with npm test"], ["", "- Deploy target is Heroku"]],
  [["", "# my-app"], ["del", "- Run tests with npm test"], ["add", "- Run tests with pnpm vitest run"], ["", "- Deploy target is Heroku"]],
  [["", "# my-app"], ["", "- Run tests with pnpm vitest run"], ["del", "- Deploy target is Heroku"], ["add", "- Deploy target is Fly.io"], ["add", "- Never commit .env files"]],
];
