import type { Cluster, EdgeKind, GraphData, GraphEdge, GraphNode, NodeKind, Viewer } from "./viewer";
import { EDGE_KIND_LABELS, NODE_KIND_LABELS } from "./viewer";

type Child = Node | string | null | undefined | false;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | boolean | ((e: Event) => void)> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (typeof v === "function") el.addEventListener(k.replace(/^on/, ""), v);
    else if (v === true) el.setAttribute(k, "");
    else if (v !== false) el.setAttribute(k, v);
  }
  for (const c of children) if (c) el.append(c);
  return el;
}

const KIND_NAMES: Record<NodeKind, string> = {
  entity: "Entity",
  source: "Source",
  collection: "Collection",
  file: "Memory file",
  category: "Memory category",
};

export function counts(data: GraphData) {
  const nodes = new Map<string, number>();
  const edges = new Map<string, number>();
  for (const n of data.nodes) nodes.set(n.kind, (nodes.get(n.kind) ?? 0) + 1);
  for (const e of data.edges) edges.set(e.kind, (edges.get(e.kind) ?? 0) + 1);
  return { nodes, edges };
}

export function searchBox(viewer: Viewer): HTMLElement {
  const list = h("ul", { class: "hits", role: "listbox" });
  let hits: GraphNode[] = [];
  let active = 0;
  const render = () => {
    list.replaceChildren(
      ...hits.map((n, i) =>
        h(
          "li",
          { role: "option", "aria-selected": i === active ? "true" : "false", onclick: () => choose(n) },
          h("span", { class: `dot k-${n.kind}` }),
          h("span", { class: "hit-label" }, n.label),
          h("span", { class: "hit-kind" }, n.kind === "entity" ? n.group : KIND_NAMES[n.kind]),
        ),
      ),
    );
  };
  const input = h("input", {
    type: "search",
    placeholder: "Find anything  /",
    "aria-label": "Find a node",
    autocomplete: "off",
    spellcheck: "false",
  });
  const choose = (n: GraphNode) => {
    viewer.select(n.id);
    input.value = n.label;
    hits = [];
    render();
  };
  input.addEventListener("input", () => {
    hits = viewer.search(input.value);
    active = 0;
    render();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") active = Math.min(active + 1, hits.length - 1);
    else if (e.key === "ArrowUp") active = Math.max(active - 1, 0);
    else if (e.key === "Enter" && hits[active]) choose(hits[active]);
    else if (e.key === "Escape") {
      hits = [];
      input.blur();
    } else return;
    e.preventDefault();
    render();
  });
  return h("div", { class: "search" }, input, list);
}

interface Layer {
  title: string;
  hint: string;
  nodes: NodeKind[];
  edges: EdgeKind[];
}

const LAYERS: Layer[] = [
  {
    title: "Knowledge",
    hint: "Facts that Keep read from your documents",
    nodes: ["entity", "source", "collection"],
    edges: ["fact", "evidence", "in_collection"],
  },
  {
    title: "Memory",
    hint: "Files the agent wrote about you",
    nodes: ["file", "category"],
    edges: ["in_category", "mentions", "cites"],
  },
];

const KIND_HINTS: Partial<Record<NodeKind | EdgeKind, string>> = {
  entity: "colour shows its cluster",
  source: "a document Keep read",
  collection: "a group of sources",
  file: "a Memory Markdown file",
  category: "a Memory folder",
  fact: "entity to entity",
  evidence: "entity to the source that says it",
  mentions: "Memory file names an entity",
  cites: "Memory entry cites a fact",
};

export function stats(data: GraphData, viewer: Viewer): HTMLElement {
  const { nodes, edges } = counts(data);
  const cell = (n: number, label: string) => h("div", {}, h("strong", {}, n.toLocaleString()), h("span", {}, label));
  return h(
    "div",
    { class: "stats" },
    cell(nodes.get("entity") ?? 0, "entities"),
    cell(edges.get("fact") ?? 0, "facts"),
    cell(nodes.get("source") ?? 0, "sources"),
    cell(viewer.clusters.length, "clusters"),
    cell(nodes.get("file") ?? 0, "memory files"),
    cell((edges.get("mentions") ?? 0) + (edges.get("cites") ?? 0), "memory links"),
  );
}

/** Layers, grouped as Knowledge and Memory. Each row is also the legend for that kind. */
export function layerToggles(viewer: Viewer, data: GraphData): HTMLElement {
  const { nodes, edges } = counts(data);
  const hidden = new Set<NodeKind | EdgeKind>();
  const apply = () => viewer.setHiddenKinds(hidden);
  const groups = LAYERS.map((layer) => {
    const kinds = [
      ...layer.nodes.filter((k) => nodes.get(k)).map((k) => [k, NODE_KIND_LABELS[k], nodes.get(k)!, true] as const),
      ...layer.edges.filter((k) => edges.get(k)).map((k) => [k, EDGE_KIND_LABELS[k], edges.get(k)!, false] as const),
    ];
    if (!kinds.length) return null;
    const boxes: HTMLInputElement[] = [];
    const rows = kinds.map(([kind, label, n, isNode]) => {
      const box = h("input", { type: "checkbox", checked: true });
      boxes.push(box);
      box.addEventListener("change", () => {
        if (box.checked) hidden.delete(kind);
        else hidden.add(kind);
        master.checked = boxes.every((b) => b.checked);
        master.indeterminate = !master.checked && boxes.some((b) => b.checked);
        apply();
      });
      return h(
        "label",
        { class: "toggle" },
        box,
        isNode ? h("span", { class: `dot k-${kind}` }) : h("span", { class: `line e-${kind}` }),
        h("span", { class: "toggle-text" }, label, KIND_HINTS[kind] ? h("small", {}, KIND_HINTS[kind]) : null),
        h("span", { class: "n" }, n.toLocaleString()),
      );
    });
    const master = h("input", { type: "checkbox", checked: true, "aria-label": `Show ${layer.title}` });
    master.addEventListener("change", () => {
      for (const [i, [kind]] of kinds.entries()) {
        boxes[i].checked = master.checked;
        if (master.checked) hidden.delete(kind);
        else hidden.add(kind);
      }
      apply();
    });
    return h(
      "details",
      { class: "layer" },
      h("summary", {}, master, h("span", { class: "toggle-text" }, layer.title, h("small", {}, layer.hint))),
      ...rows,
    );
  });
  return h("section", { class: "layers" }, h("h3", {}, "Show"), ...groups);
}

export function clusterList(viewer: Viewer): HTMLElement {
  let current: number | null = null;
  const reset = h("button", { type: "button", class: "reset", hidden: true }, "Show all");
  viewer.onCluster((id) => {
    current = id;
    for (const b of wrap.querySelectorAll<HTMLElement>(".cluster")) b.classList.toggle("on", Number(b.dataset.id) === current);
    reset.hidden = current === null;
  });
  const pick = (id: number | null) => viewer.setCluster(id);
  reset.addEventListener("click", () => pick(null));
  const items = viewer.clusters.slice(0, 24).map((c: Cluster) =>
    h(
      "button",
      { class: "cluster", type: "button", "data-id": String(c.id), title: `Zoom to ${c.top.join(", ")}`, onclick: () => pick(current === c.id ? null : c.id) },
      h("span", { class: "swatch", style: `background:${c.color}` }),
      h("span", { class: "cluster-name" }, c.top.slice(0, 2).join(" · ") || `Cluster ${c.id + 1}`),
      h("span", { class: "n" }, String(c.size)),
    ),
  );
  const wrap = h(
    "section",
    { class: "clusters" },
    h("h3", {}, "Clusters", reset),
    h("p", { class: "hint" }, "Groups of entities that share many facts. Click one to zoom in."),
    ...items,
  );
  return wrap;
}

type Link = { node: GraphNode; edge: GraphEdge; outgoing: boolean };

/** Group fact links by direction and predicate: "works on: A, B". */
function groupLinks(links: Link[]) {
  const groups = new Map<string, { label: string; outgoing: boolean; nodes: GraphNode[] }>();
  for (const x of links) {
    const reverse: Partial<Record<EdgeKind, string>> = { mentions: "mentioned in", cites: "cited by" };
    const base = x.edge.label || x.edge.kind.replace(/_/g, " ");
    const label = x.outgoing ? base : (reverse[x.edge.kind] ?? base);
    const key = `${x.outgoing}|${label}`;
    const g = groups.get(key) ?? { label, outgoing: x.outgoing, nodes: [] };
    if (!g.nodes.some((n) => n.id === x.node.id)) g.nodes.push(x.node);
    groups.set(key, g);
  }
  return [...groups.values()].sort((a, b) => Number(b.outgoing) - Number(a.outgoing) || b.nodes.length - a.nodes.length);
}

export function inspector(viewer: Viewer, data: GraphData) {
  const el = h("aside", { class: "inspector", "aria-live": "polite" });
  const history: string[] = [];
  let current: GraphNode | null = null;
  const chip = (m: GraphNode) =>
    h(
      "button",
      { type: "button", class: "chip", onclick: () => viewer.select(m.id) },
      h("span", { class: `dot k-${m.kind}`, style: `background:${viewer.graph.getNodeAttribute(m.id, "baseColor") ?? ""}` }),
      m.label,
    );

  const overview = () => {
    const top = data.nodes
      .filter((n) => n.kind === "entity" && viewer.graph.hasNode(n.id))
      .sort((a, b) => viewer.graph.degree(b.id) - viewer.graph.degree(a.id))
      .slice(0, 10);
    const files = data.nodes.filter((n) => n.kind === "file").slice(0, 8);
    fill(
      el,
      h("p", { class: "eyebrow" }, "Overview"),
      h("h2", {}, "What Keep knows"),
      h("p", { class: "desc" }, "Each dot is a thing Keep knows about. Each line is a fact or a link. Click a dot to see its facts and where they came from."),
      top.length ? h("section", {}, h("h3", {}, "Most connected"), h("div", { class: "chips" }, ...top.map(chip))) : null,
      files.length ? h("section", {}, h("h3", {}, "Memory files"), h("div", { class: "chips" }, ...files.map(chip))) : null,
      h(
        "section",
        { class: "keys" },
        h("h3", {}, "Keys"),
        h("p", {}, h("kbd", {}, "/"), " find  ", h("kbd", {}, "Esc"), " clear  ", h("kbd", {}, "Backspace"), " back"),
        h("p", {}, "Scroll to zoom. Drag to pan. Hover a dot to see its links."),
      ),
    );
  };

  const back = () => {
    history.pop();
    viewer.select(history.pop() ?? null);
  };

  const show = (n: GraphNode | null) => {
    if (n && current?.id !== n.id) history.push(n.id);
    if (!n) history.length = 0;
    current = n;
    el.dataset.mode = n ? "node" : "overview";
    if (!n) return overview();
    const near = viewer.neighbours(n.id);
    const isLink = (x: Link) => x.edge.kind === "fact" || x.edge.kind === "cites" || x.edge.kind === "mentions";
    const links = groupLinks(near.filter(isLink));
    const other = near.filter((x) => !isLink(x));
    const attrs = (n.data?.attributes as { predicate: string; object: string; status: string }[] | undefined) ?? [];
    const aliases = (n.data?.aliases as string[] | undefined) ?? [];
    const cluster = viewer.clusters[viewer.graph.getNodeAttribute(n.id, "cluster") as number];
    const parts: Child[] = [
      h(
        "div",
        { class: "nav" },
        history.length > 1 ? h("button", { type: "button", class: "back", onclick: back }, "← Back") : h("span"),
        h("button", { type: "button", class: "close", "aria-label": "Close", onclick: () => viewer.select(null) }, "×"),
      ),
      h(
        "header",
        {},
        h("span", { class: `dot k-${n.kind}`, style: `background:${viewer.graph.getNodeAttribute(n.id, "baseColor") ?? ""}` }),
        h(
          "div",
          {},
          h("p", { class: "eyebrow" }, n.kind === "entity" ? `Entity · ${n.group}` : KIND_NAMES[n.kind]),
          h("h2", {}, n.label),
          n.kind === "entity" && cluster ? h("p", { class: "in-cluster" }, "In cluster ", h("span", { class: "swatch", style: `background:${cluster.color}` }), cluster.top.slice(0, 2).join(" · ")) : null,
        ),
      ),
      aliases.length ? h("p", { class: "aliases" }, "Also known as ", aliases.join(", ")) : null,
      typeof n.data?.description === "string" ? h("p", { class: "desc" }, n.data.description) : null,
      typeof n.data?.path === "string" ? h("p", { class: "mono" }, n.data.path) : null,
      typeof n.data?.url === "string" && /^https?:\/\//.test(n.data.url)
        ? h("p", {}, h("a", { href: n.data.url, target: "_blank", rel: "noopener noreferrer" }, "Open source ↗"))
        : null,
      attrs.length
        ? h(
            "section",
            {},
            h("h3", {}, "Attributes"),
            h("dl", { class: "attrs" }, ...attrs.flatMap((a) => [h("dt", {}, a.predicate.replace(/_/g, " ")), h("dd", { class: a.status !== "active" ? "stale" : "" }, a.object, a.status !== "active" ? h("span", { class: "tag" }, a.status) : null)])),
          )
        : null,
      links.length
        ? h(
            "section",
            {},
            h("h3", {}, n.kind === "file" ? "Links to Knowledge" : "Facts"),
            h(
              "dl",
              { class: "links" },
              ...links.flatMap((g) => [h("dt", {}, g.outgoing || /^(mentioned in|cited by)$/.test(g.label) ? "" : "← ", g.label), h("dd", { class: "chips" }, ...g.nodes.slice(0, 30).map(chip))]),
            ),
          )
        : null,
      other.length
        ? h(
            "section",
            {},
            h("h3", {}, n.kind === "entity" ? "Found in" : n.kind === "collection" || n.kind === "category" ? "Contains" : "Connected"),
            h("div", { class: "chips" }, ...dedupe(other.map((x) => x.node)).slice(0, 60).map(chip)),
          )
        : null,
    ];
    fill(el, ...parts);
    el.scrollTop = 0;
  };
  overview();
  return { el, show, back };
}

function fill(el: HTMLElement, ...parts: Child[]) {
  el.replaceChildren(...parts.filter((p): p is Node | string => !!p));
}

function dedupe(nodes: GraphNode[]) {
  const seen = new Set<string>();
  return nodes.filter((n) => !seen.has(n.id) && seen.add(n.id));
}

export function canvasTools(viewer: Viewer): HTMLElement {
  let labels = true;
  const labelBtn = h("button", { type: "button", "aria-pressed": "true", title: "Show labels" }, "Aa");
  labelBtn.addEventListener("click", () => {
    labels = !labels;
    labelBtn.setAttribute("aria-pressed", String(labels));
    viewer.setLabels(labels);
  });
  return h(
    "div",
    { class: "tools", role: "toolbar", "aria-label": "Graph view" },
    h("button", { type: "button", title: "Zoom in", "aria-label": "Zoom in", onclick: () => viewer.zoom(1.5) }, "+"),
    h("button", { type: "button", title: "Zoom out", "aria-label": "Zoom out", onclick: () => viewer.zoom(1 / 1.5) }, "−"),
    h("button", { type: "button", title: "Fit the whole graph", onclick: () => viewer.fit() }, "Fit"),
    labelBtn,
  );
}

export function scopeForm(scope: string): HTMLElement {
  const params = new URLSearchParams(location.search);
  const user = h("input", { name: "user", placeholder: "user", value: params.get("user") ?? "", "aria-label": "User" });
  const project = h("input", { name: "project", placeholder: "project", value: params.get("project") ?? "", "aria-label": "Project" });
  const form = h("form", { class: "scope" }, h("span", { class: "scope-label mono" }, scope), user, project, h("button", { type: "submit" }, "Load"));
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const next = new URLSearchParams(location.search);
    for (const [k, input] of [["user", user], ["project", project]] as const) {
      if (input.value.trim()) next.set(k, input.value.trim());
      else next.delete(k);
    }
    location.search = next.toString();
  });
  return form;
}

export function emptyState(data: GraphData): HTMLElement | null {
  if (data.nodes.length) return null;
  return h(
    "div",
    { class: "empty" },
    h("h2", {}, "Nothing to draw yet"),
    h("p", {}, "This scope has no Knowledge facts or Memory files. Ingest a document or write a memory file, then reload."),
    h("pre", { class: "mono" }, `curl -s localhost:7357/v1/knowledge/ingest \\\n  -H 'content-type: application/json' \\\n  -d '{"title":"Notes","text":"Mira Sato works at Northwind."}'`),
  );
}
