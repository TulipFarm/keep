# DESIGN.md: Keep

The only UI in this repo is the marketing site in `site/` (Astro, static).
Tokens live in `site/src/styles/global.css`; all copy lives in
`site/src/data/site.ts`.

## Product
- **What it is:** Open-source memory that every AI tool you use shares
  (Claude, Codex, Copilot, Cursor and others). It is self-hosted and keeps its
  full history.
- **Why it's different:** Old facts are marked replaced, never overwritten,
  and each one keeps its source and its dates. RAG and vector stores
  overwrite silently. With Keep you can ask what was true on a date, and
  where that came from.
- **Surfaces:**
  - Marketing site `/` (Persuade).
  - `/index.md` and `/llms.txt` (Read; generated from the same data).
  - `install.md` / `install.sh` (Operate; outside this site's design).
- **Success looks like:** the visitor understands the idea of keeping history
  and believes the honest status. Then they copy the setup prompt or the
  install script, or go to GitHub.
- **Constraints:**
  - Pristine default standard: WCAG 2.2 AA, evergreen browsers, static
    export, readable with JS off, reduced motion respected, LCP < 2.5s on
    mobile, no new fonts or deps without approval.
  - Exceptions: none.
  - Must-keep: the "research preview" status and the "not passed / not run /
    none" benchmark states.
  - Must-keep: Apache-2.0, plus the FAQ JSON-LD built from `site.ts`.
  - Must-keep: the install, uninstall and methodology links.
  - Fixtures and demo data use invented projects only.

## Personas
### Dev: developer using several AI coding agents (primary)
- **Context:** On a laptop, switching between Claude Code, Codex and Copilot
  during the day. Also stands for the team lead who self-hosts Keep on
  Postgres or Docker Compose for a team.
- **Job:** Give every agent the same memory of decisions and conventions,
  and see when and why something changed.
- **Expertise:** Expert. Reads code and docs, and is suspicious of AI hype.
- **Must feel in 5 seconds:** This is honest and careful. It shows exactly
  what it does and what's unproven, and nothing is hidden or hyped.
- **Deal-breakers:**
  - Vague claims.
  - A hosted service in disguise.
  - Not being able to find the source or the install command.
- **Delighted by:** the time-machine demo, one-line install, a readable
  Markdown memory file, and the real request and response.

### Nia: non-engineer who uses AI assistants daily
- **Context:** On a laptop. Uses an AI assistant that can run commands, but
  doesn't write code.
- **Job:** Make the assistant remember preferences and decisions, and set it
  up without a terminal tutorial.
- **Expertise:** Novice with tools. Doesn't know what MCP, RAG or a vector DB
  is.
- **Must feel in 5 seconds:** I get what this does, and I can install it by
  pasting one sentence.
- **Deal-breakers:**
  - Jargon without explanation.
  - Install paths that assume a shell.
- **Delighted by:**
  - The "Copy setup prompt" button.
  - Plain-language FAQ answers ("Do I need to be a developer?", "What is
    MCP?").

### Ravi: evaluator or researcher
- **Context:** On a desktop, comparing memory systems before trusting or
  citing one.
- **Job:** Check claims against the source, the methodology and the
  benchmark status.
- **Expertise:** Expert in the domain.
- **Must feel in 5 seconds:** The status is stated plainly, and every claim
  links to evidence.
- **Deal-breakers:** unsupported numbers, or hiding the fact that it hasn't
  beaten simpler baselines yet.
- **Delighted by:** a status tile that says "Not passed", and a methodology
  link one click away.

## Brand
- **Personality:** Plain, trustworthy, quietly confident.
- **Voice:**
  - Short declarative sentences, second person, with jargon explained on
    first use.
  - State the status honestly ("research preview", "not yet shown").
  - Don't use: "revolutionary", "blazing", "seamless", "supercharge", or
    invented numbers.
  - Every claim must be supported by `README.md` or `docs/`.
- **Admired references:** Supermemory, Mem0.
- **Anti-references:** TBD.

## Design System
### Color
Light and dark follow the OS (`prefers-color-scheme`), and `color-scheme` is
`light dark`.

| Token | Light | Dark | Use |
|---|---|---|---|
| --bg | #ffffff | #0c0e11 | page background |
| --surface | #f4f5f7 | #14171b | tiles, panels |
| --raised | #ffffff | #191d22 | cards on a surface, button fill |
| --line | #e3e6ea | #252a31 | hairlines, card borders |
| --line-strong | #c7ccd3 | #39404a | button borders, dashed "future" bars |
| --fg | #111418 | #eef0f3 | text |
| --muted | #565e69 | #9ca5af | secondary text (AA on bg and surface) |
| --accent | #ff5a1f | #ff6a33 | fills only: primary button, "true now" bar, playhead |
| --accent-ink | #111418 | #111418 | text on --accent |
| --accent-text | #c2410c | #ff8f63 | accent as text: links, eyebrows, stamps, focus ring |
| --accent-soft | #ffeadf | #3a1c10 | removed-line and current-step tint |
| --ok / --ok-soft | #1d6b38 / #e2f3e7 | #82d39d / #14301e | added lines in diffs |
| --band, --band-fg, --band-muted, --band-line, --band-code | #111418, #f3f4f6, #a7afb9, #2a3038, #1b2027 | #14171b, #eef0f3, #9ca5af, #2c323a, #0c0e11 | dark install band and terminals |

Rules:
- There is one accent. Use it only for the main action, the "true now"
  state, links and stamps, and keep it to ≤ 10% of any screen.
- Never put text in `--accent` on light backgrounds. Use `--accent-text`.
- A superseded fact is shown by a strikethrough in `--accent-text` plus a
  "Superseded" stamp, never by colour alone.
- Never lower the contrast of text with `opacity`. Dim the shape instead.

### Typography
- **Families:** Inter Variable (display and body) and Geist Mono Variable
  (labels, data, code). Both load through `@fontsource-variable` and are
  preloaded in `Layout.astro`.
- **Scale:**
  - h1: clamp(2.6rem, 6.4vw, 5.4rem)/0.98, weight 750, tracking −0.045em.
  - h2: clamp(2rem, 3.6vw, 3rem)/1.05, weight 750, tracking −0.035em.
  - h3: 1.35rem/1.1, weight 700.
  - Body: 17px/1.6.
  - Eyebrow: mono 0.72rem, weight 600, uppercase, tracking 0.08em, in
    `--accent-text`.
  - Small mono: 0.75–0.85rem.
- **Rules:**
  - Headings use `text-wrap: balance`, except the hero h1, which uses
    `wrap`.
  - Paragraphs use `pretty`.
  - Keep the measure at ≤ 70ch.
  - Dates and numbers use `tabular-nums`.
  - Inter features `cv11` and `ss01` are on.

### Spacing & Layout
- **Scale:** in practice 4/8/10/12/16/20/24/32/40/48px. Section padding is
  48–96px.
- **Container:** `.wrap` = min(1200px, 100% − 40px).
- **Breakpoints:** 1000px (bento and hero collapse), 720px (single column),
  760px (time machine).
- **Grids:** always use `minmax(0, 1fr)` tracks so content never forces
  horizontal overflow at 390px.

### Shape & Elevation
- **Radii:** `--radius-sm` 6px, `--radius` 10px, `--radius-lg` 16px.
- **Elevation:** use 1px borders (`--line`) and tonal surfaces. Don't use
  drop shadows.

### Motion
- **Durations:** 100–200ms, for colour, border and small transforms only.
  For new work use `--dur-press` 140ms and `--dur-small` 180ms.
- **Easing:** `ease` for colour changes, and `--ease-out`
  cubic-bezier(0.23, 1, 0.32, 1) for new transforms.
- **Reduced motion:** a global reset in `global.css` turns off all
  animations and transitions, and turns off smooth scrolling.
- Don't use scroll-triggered reveals or decorative loops.

### Components
- **`.btn` / `.btn.primary`:**
  - 44px minimum height, `--radius`.
  - Primary is an `--accent` fill with `--accent-ink` text.
  - Hover only on `(hover: hover) and (pointer: fine)`.
  - `:active` moves the button down 1px.
- **`.link`:** `--accent-text` with a CSS-mask chevron, used for "Read more"
  links.
- **`.eyebrow`:** labels sections and tiles.
- **`.stamp`:** "Superseded" badge, not rotated.
- **`.card`, `.fact`, `.term`, `.memfile`:** the evidence vocabulary (facts
  with dates, terminal, revisioned Markdown file).
- **`Copy.astro`:**
  - A copy button that announces through a shared `role=status` live region.
  - Takes a `fallback` message.
  - The hero shows a `noscript` link to `#install` instead.
- **`TimeMachine.astro`:**
  - The signature moment.
  - The server pre-renders the state for 2026-05-01, so the page is correct
    with JS off.
  - The slider is aligned to the Gantt chart.
- **`Faq.astro`:** `<details>` items that also feed the JSON-LD.
- **Icon set:** inline SVG only, 2px stroke, round caps.
- **Focus:** 2px `--accent-text` outline with a 2px offset on every
  interactive element.

## Do / Don't
- **Do:**
  - Show the product's real output (facts, diffs, requests) instead of
    illustrations.
  - Explain jargon in plain words.
  - Keep the status and benchmark state visible.
  - Pre-render every interactive demo.
- **Don't:**
  - Gradients, glows, stock AI imagery.
  - A second accent colour.
  - Hype words or uncited numbers.
  - Hover-only meaning.
  - `opacity` on text.
  - Shadows used for depth.
