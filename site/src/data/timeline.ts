export const start = "2026-01-01";
export const end = "2026-11-20";
export const day = (s: string) => Math.round((Date.parse(s) - Date.parse(start)) / 864e5);
export const span = day(end);
export const iso = (d: number) => new Date(Date.parse(start) + d * 864e5).toISOString().slice(0, 10);

export const inks = { claude: "#e8673a", codex: "#2f7de1", copilot: "#8a4fe0" } as const;
export type Agent = keyof typeof inks;
export type Fact = { key: string; text: string; from: string; to: string | null; by: Agent };

export const keys: Record<string, string> = {
  deploy: "deploy target",
  tests: "test command",
  node: "billing runtime",
};

export const timeline: Fact[] = [
  { key: "deploy", text: "Deploy target is Heroku", from: "2026-01-20", to: "2026-06-03", by: "claude" },
  { key: "deploy", text: "Deploy target is Fly.io", from: "2026-06-03", to: null, by: "claude" },
  { key: "tests", text: "Run tests with npm test", from: "2026-03-02", to: "2026-09-14", by: "codex" },
  { key: "tests", text: "Run tests with pnpm vitest run", from: "2026-09-14", to: null, by: "copilot" },
  { key: "node", text: "Billing API uses Node 18", from: "2026-02-11", to: "2026-10-08", by: "codex" },
  { key: "node", text: "Billing API uses Node 22", from: "2026-10-08", to: null, by: "claude" },
];

export const revAgents: Agent[] = ["codex", "copilot", "claude"];
