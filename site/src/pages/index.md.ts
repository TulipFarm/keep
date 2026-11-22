import { hero, flow, where, sections, faq, meta, repo, setupPrompt, install } from "../data/site";

const md = `# Keep

${hero.sub}

${hero.note}

Source: ${repo}

## Install

Tell your coding agent: \`${setupPrompt}\`

Or run \`${install.script.cmd}\`. Runs on macOS and Linux.

## How it works

${flow.steps.map(([t, d]) => `- **${t}.** ${d}`).join("\n")}

## ${where.heading}

${where.lead}

${where.items.map(([t, d, c]) => `- **${t}.** ${d} \`${c}\``).join("\n")}

${sections.map((s) => `## ${s.title}\n\n${s.lead}\n\n${s.points.map(([a, b]) => `- **${a}** ${b}`.trim()).join("\n")}\n\n[${s.link?.[0]}](${s.link?.[1]})`).join("\n\n")}

## Questions

${faq.map(([q, a]) => `### ${q}\n\n${a}`).join("\n\n")}
`;

export const GET = () => new Response(`<!-- ${meta.title} -->\n${md}`, { headers: { "Content-Type": "text/markdown; charset=utf-8" } });
