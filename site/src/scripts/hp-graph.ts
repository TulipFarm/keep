import { type GraphData, type GraphNode, mountGraph, type Viewer } from "../../../crates/keep-server/dev-ui/src/viewer";

export type { Viewer } from "../../../crates/keep-server/dev-ui/src/viewer";

export interface Demo extends GraphData {
  attribution: { text: string; url: string; licence: string; licence_url: string; trademark: string };
}

export const DEMO_URL = "/demo/harry-potter.json";

export async function loadDemo(): Promise<Demo> {
  const res = await fetch(DEMO_URL);
  if (!res.ok) throw new Error(`${res.status} loading the demo graph`);
  return res.json();
}

export function mount(el: HTMLElement, data: Demo, onSelect?: (n: GraphNode | null) => void): Viewer {
  // The home page is light only.
  return mountGraph(el, data, { dark: false, onSelect });
}

const SINGULAR: Record<string, string> = { people: "person", places: "place", objects: "object", events: "event", groups: "group" };

/** Mount when the element scrolls near the view, so the page loads without the graph file. */
export function whenVisible(el: Element, fn: () => void) {
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      io.disconnect();
      fn();
    }
  }, { rootMargin: "300px" });
  io.observe(el);
}

export function counts(data: Demo) {
  const n = (k: string) => data.nodes.filter((x) => x.kind === k).length;
  const text = data.nodes.reduce((s, x) => s + ((x.data?.attributes as unknown[] | undefined)?.length ?? 0), 0);
  const clusters = new Set(data.nodes.filter((x) => x.kind === "entity").map((x) => x.cluster)).size;
  return { sources: n("source"), entities: n("entity"), facts: text + data.edges.filter((e) => e.kind === "fact").length, clusters, files: n("file") };
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

/** HTML for a clicked dot: what Keep knows about it and where it read that. */
export function card(viewer: Viewer, n: GraphNode): string {
  const near = viewer.neighbours(n.id);
  const attrs = ((n.data?.attributes as { predicate: string; object: string; status: string }[] | undefined) ?? []).filter((a) => a.status !== "superseded");
  const links = near.filter((x) => ["fact", "names", "mentions", "cites"].includes(x.edge.kind));
  const sources = near.filter((x) => x.node.kind === "source");
  // Extraction from documents does not type entities well; the wiki page of the same name says what it is.
  const page = sources.find((x) => x.node.label.toLowerCase() === n.label.toLowerCase());
  const kind = n.kind === "entity" ? SINGULAR[page?.node.group ?? ""] : n.kind === "file" ? "Memory file" : n.kind === "source" ? "wiki page" : n.kind;
  const rows = [
    ...attrs.slice(0, 5).map((a) => `<li><span>${esc(a.predicate)}</span> ${esc(a.object)}</li>`),
    ...links.slice(0, 4).map((x) => `<li><span>${esc(x.outgoing ? x.edge.label || x.edge.kind : x.edge.kind === "names" ? "named by" : x.edge.label || x.edge.kind)}</span> <button type="button" data-go="${esc(x.node.id)}">${esc(x.node.label)}</button></li>`),
  ];
  const from = sources.slice(0, 3).map((x) => {
    const url = x.node.data?.url as string | undefined;
    return url ? `<a href="${esc(url)}" rel="noopener" target="_blank">${esc(x.node.label)} ↗</a>` : esc(x.node.label);
  });
  const more = attrs.length + links.length - rows.length;
  return `${kind ? `<p class="k">${esc(kind)}</p>` : ""}<h3>${esc(n.label)}</h3>
${rows.length ? `<ul>${rows.join("")}</ul>` : ""}${more > 0 ? `<p class="more">+ ${more} more</p>` : ""}
${from.length ? `<p class="from">From ${from.join(", ")}</p>` : ""}`;
}

export function credit(data: Demo): string {
  const a = data.attribution;
  return `${esc(a.text)} (<a href="${esc(a.url)}" rel="noopener">source</a>), <a href="${esc(a.licence_url)}" rel="noopener">${esc(a.licence)}</a>. ${esc(a.trademark)}`;
}
