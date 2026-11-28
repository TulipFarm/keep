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
    "Keep is open-source memory for your AI assistants. When something changes, the old version is kept and marked as replaced, so you can see what was true, when, and where it came from. Runs on your machine. Apache-2.0 research preview.",
};

export const setupPrompt = "Read https://keep.tulipfarm.ai/install.md and install Keep for me.";

export const install = {
  heading: "Install in one step",
  lead: "Paste one sentence into an AI coding assistant, or run the install script. Works on macOS and Linux.",
  agent: { title: "Ask your agent", cmd: setupPrompt, note: "Paste into Claude Code, Codex, Copilot CLI, or OpenCode. It installs Keep and connects your apps." },
  script: { title: "Or run the script", cmd: "curl -fsSL https://keep.tulipfarm.ai/install.sh | sh", note: "Picks free ports, installs a background service (launchd / systemd), and adds ~/.keep/inspect.sh." },
  links: [["Read install.md", "/install.md"], ["View install.sh", "/install.sh"], ["Read uninstall.md", "/uninstall.md"], ["View uninstall.sh", "/uninstall.sh"]] as [string, string][],
  copy: "Copy",
  copied: "Copied",
};

export const hero = {
  badge: "Open source · Local-first · Research preview",
  headline: "Memory your AI can't quietly rewrite.",
  sub: "One shared memory for all your AI assistants. Old facts are marked replaced, never erased — with dates and sources intact.",
  hint: "Paste the prompt into Claude Code, Codex, or Copilot CLI to install Keep.",
  note: "Keep is a research preview. It works, but it is early.",
  primary: "Get Keep on GitHub",
  secondary: "Setup docs",
  copy: "Copy setup prompt",
  worksWith: ["Claude Code", "Claude Desktop", "Codex", "Copilot CLI", "OpenCode", "Cursor", "Local MCP apps"],
  storesIn: ["In memory", "Postgres + pgvector", "Postgres + Qdrant"],
  snippetLabel: "Install from source, then connect a local MCP agent",
  snippet: [
    "cargo install --path crates/keep-server --features local-embeddings",
    "keep --user me serve &",
    "keep --project my-app mcp --server http://127.0.0.1:7357",
  ],
};

export const flow = {
  caption: "Your AI tools share one local vault: versioned notes on top, timestamped facts below.",
  steps: [
    ["Work like you already do", "Write notes and chat with your AI as usual."],
    ["Keep saves every version", "Every edit becomes a new revision."],
    ["Keep extracts the facts", "Each fact keeps its dates and exact source."],
    ["Your AI starts informed", "Every new chat loads your notes and searches history."],
  ] as [string, string][],
};

export const does: [string, string][] = [
  ["Stop repeating yourself.", "Your rules and preferences load at the start of every chat."],
  ["No silent rewrites.", "Changed facts are marked replaced — never quietly erased."],
  ["Receipts for every answer.", "Every fact links back to the note or chat behind it."],
];

export const where = {
  heading: "Runs where you want",
  lead: "On your laptop or your own server. No hosted service.",
  items: [
    ["In memory", "Zero setup. Memory clears on restart.", "keep --user me serve &"],
    ["With Postgres", "Survives restarts. Uses pgvector or Qdrant.", "docker compose --profile postgres-vector up -d"],
    ["Docker Compose", "Full stack with generated secrets.", "deploy/compose.yaml"],
  ] as [string, string, string][],
  link: ["Run with Docker Compose", `${repo}/blob/main/docs/server.md#run-with-docker-compose`] as [string, string],
};

export const demo = {
  heading: "Drag the date. See what was true.",
  lead: "Pick a topic and slide through time to see how Keep separates what's true today from what changed.",
  steps: ["Pick a topic", "Drag the date", "See what was true"],
  legend: ["True on this date", "Replaced later. Still stored.", "Not recorded yet"],
  label: "Example data",
  caption: "Sample data from an example workspace.",
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
    eyebrow: "Memory",
    title: "Notes your AI reads every time",
    lead: "Plain Markdown notes loaded at the start of every chat.",
    body: "",
    points: [
      ["No accidental overwrites.", "Outdated edits are rejected automatically."],
      ["Safe undo.", "Undoing creates a new version — history stays intact."],
      ["30-day history.", "Up to 100 revisions per file, saved byte for byte."],
    ],
    link: ["How Memory works", docs.memory],
  },
  {
    id: "knowledge",
    n: 2,
    eyebrow: "Knowledge",
    title: "Temporal facts with sources",
    lead: "Every fact tracks when it was true and where it came from.",
    body: "",
    points: [
      ["Append-only.", "Old facts are closed, never erased."],
      ["Two kinds of time.", "Ask what was true then, or what Keep knew then."],
      ["Conflicts stay visible.", "If facts disagree, Keep shows both."],
    ],
    link: ["How Knowledge works", docs.knowledge],
  },
  {
    id: "connect",
    n: 3,
    eyebrow: "Connect",
    title: "One local hub for every AI",
    lead: "Switch between Claude, Codex, Copilot, or Cursor with one shared memory.",
    body: "",
    points: [
      ["Local MCP + HTTP.", "Works with any local MCP app, plus HTTP for scripts."],
      ["Scoped keys.", "One key per tool, private to 127.0.0.1 by default."],
    ],
    link: ["Setup for each agent", docs.server],
  },
  {
    id: "status",
    n: 4,
    eyebrow: "Status",
    title: "Honest status",
    lead: "Research preview. Not yet shown to beat simpler approaches.",
    body: "",
    points: [
      ["Works today:", "Memory, Knowledge, and the alpha server."],
      ["No numbers yet.", "Our test runs haven't passed our own bar."],
    ],
    link: ["Read the methodology", docs.method],
  },
  {
    id: "open-source",
    n: 5,
    eyebrow: "Open source",
    title: "Open source, Apache-2.0",
    lead: "Written in Rust. One public repo for server, memory, knowledge, and benchmarks.",
    body: "",
    points: [
      ["Audit everything.", "Datasets keep their own licences; you download them yourself."],
    ],
    link: ["View the repository", repo],
  },
];

export const faq: [string, string][] = [
  [
    "What is Keep?",
    "Keep is a shared memory for your AI assistants. It stores short notes your AI reads in every chat, plus a record of facts with their sources and dates, so you can see what changed and when. It is open source, written in Rust, connects to AI apps over MCP, and is a research preview.",
  ],
  [
    "Do I need to be a developer to use Keep?",
    "Not to use it, but setup happens on the command line today. The easiest way is to paste the setup prompt into an AI coding assistant such as Claude Code and let it install Keep for you. After that, your connected AI apps use Keep on their own.",
  ],
  [
    "What is MCP?",
    "MCP (Model Context Protocol) is an open standard that lets AI apps such as Claude, Codex and Cursor connect to tools and data. Keep is one of those tools, so any MCP app that can run a local server on your computer can use it.",
  ],
  [
    "How is Keep different from RAG or a vector database?",
    "Keep does use search, both keyword and vector. It also stores facts with their source and the time range when they were true. When a fact changes, the old version is closed, not overwritten, so you can ask what was true on a date.",
  ],
  [
    "Which agents and models does Keep work with?",
    "Claude Code, Claude Desktop, Codex, Copilot CLI, OpenCode, Cursor, and any MCP client that can start a stdio server. The agentic installer detects your apps, connects the ones you want, and offers to import existing memories. Keep sets no chat model for you. Your agent keeps its own.",
  ],
  [
    "Can I run Keep myself?",
    "Yes. Paste the install.md prompt into your coding agent, or run install.sh yourself. Both set up Keep locally or connect to a remote server, pick free ports automatically, and install a background service (launchd / systemd). Remove it anytime with uninstall.md or uninstall.sh.",
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

export const inside = {
  heading: "What's inside",
  lead: "Two kinds of memory, one local server, and an honest look at where things stand.",
};

export const footer = "Keep is open source under Apache-2.0 and is a research preview from TulipFarm.";

export const facts = [
  { old: "Pro plan is $29/mo", oldWhen: "2026-01-20 → 2026-06-03", now: "Pro plan is $39/mo", nowWhen: "2026-06-03 → now" },
  { old: "Notes live in Notion", oldWhen: "2026-03-02 → 2026-09-14", now: "Notes in Google Docs", nowWhen: "2026-09-14 → now" },
  { old: "Uses Node 22", oldWhen: "2026-02-11 → 2026-10-08", now: "Uses Node 26", nowWhen: "2026-10-08 → now" },
];

export type Line = ["", string] | ["add", string] | ["del", string];
export const revisions: Line[][] = [
  [["", "# project-notes"], ["", "- Write replies in short, plain English"], ["", "- Pro plan is $29/mo"], ["", "- Uses Node 22"]],
  [["", "# project-notes"], ["", "- Write replies in short, plain English"], ["", "- Pro plan is $29/mo"], ["del", "- Uses Node 22"], ["add", "- Uses Node 26"]],
  [["", "# project-notes"], ["", "- Write replies in short, plain English"], ["del", "- Pro plan is $29/mo"], ["add", "- Pro plan is $39/mo"], ["", "- Uses Node 26"], ["add", "- Never share customer emails"]],
];
