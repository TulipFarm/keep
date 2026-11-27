import type Graph from "graphology";
import forceAtlas2 from "graphology-layout-forceatlas2";
import FA2Layout from "graphology-layout-forceatlas2/worker";
import Sigma from "sigma";
import { type Cluster, type EdgeKind, FA2_SETTINGS, type GraphData, type GraphEdge, type GraphNode, type NodeKind, buildGraph, clustersOf, layout } from "./layout";

export type { Cluster, EdgeKind, GraphData, GraphEdge, GraphNode, NodeKind } from "./layout";
export { buildGraph } from "./layout";

export interface ViewerOptions {
  dark?: boolean;
  /** Let ForceAtlas2 run visibly for this many milliseconds after the first layout. */
  animateMs?: number;
  onSelect?: (node: GraphNode | null) => void;
  onHover?: (node: GraphNode | null) => void;
}

export interface Viewer {
  graph: Graph;
  sigma: Sigma;
  clusters: Cluster[];
  node(id: string): GraphNode | undefined;
  neighbours(id: string): { node: GraphNode; edge: GraphEdge; outgoing: boolean }[];
  setHiddenKinds(kinds: Set<NodeKind | EdgeKind>): void;
  /** Show one cluster and what it links to, and move the camera to it. */
  setCluster(cluster: number | null): void;
  /** Call `fn` each time the shown cluster changes. */
  onCluster(fn: (cluster: number | null) => void): void;
  setLabels(on: boolean): void;
  zoom(factor: number): void;
  select(id: string | null): void;
  search(query: string, limit?: number): GraphNode[];
  fit(): void;
  destroy(): void;
}

export const KIND_COLORS = {
  light: { source: "#8c8472", collection: "#1d1a14", category: "#2f6b3a", file: "#d9a521", entity: "#5e574a" },
  dark: { source: "#7d7562", collection: "#f2ecdc", category: "#8fd09b", file: "#e8c04a", entity: "#b0a78f" },
} as const;

/** Blend `fg` over `bg` as a solid colour; WebGL edges draw solid colours best. */
function mix(fg: string, bg: string, alpha: number): string {
  const c = (hex: string, i: number) => parseInt(hex.slice(1 + i * 2, 3 + i * 2), 16);
  const out = [0, 1, 2].map((i) => Math.round(c(fg, i) * alpha + c(bg, i) * (1 - alpha)));
  return `#${out.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}

const INK = { light: "#1d1a14", dark: "#f2ecdc" };
const PAPER = { light: "#f7f2e6", dark: "#16130e" };
const MAX_NAMED = 24;

function style(g: Graph, clusters: Cluster[], dark: boolean) {
  const kinds = dark ? KIND_COLORS.dark : KIND_COLORS.light;
  g.forEachNode((id, a) => {
    const degree = g.degree(id);
    const kind = a.kind as NodeKind;
    let size = 2.5 + Math.sqrt(degree) * 1.4;
    let color: string = kinds[kind];
    if (kind === "entity") color = clusters[a.cluster]?.color ?? kinds.entity;
    if (kind === "collection" || kind === "category") size = Math.max(size, 9);
    if (kind === "source") size = Math.min(size, 6);
    g.mergeNodeAttributes(id, { size: Math.min(size, 22), color, baseColor: color });
  });
  const tone = dark ? "dark" : "light";
  const faint = mix(INK[tone], PAPER[tone], 0.08);
  const strong = mix(INK[tone], PAPER[tone], 0.22);
  g.forEachEdge((id, a) => {
    const kind = a.kind as EdgeKind;
    const isFact = kind === "fact" || kind === "names" || kind === "cites";
    g.mergeEdgeAttributes(id, {
      size: isFact ? 0.8 : 0.5,
      color: isFact ? strong : faint,
    });
  });
}

export function mountGraph(container: HTMLElement, data: GraphData, opts: ViewerOptions = {}): Viewer {
  const dark = opts.dark ?? matchMedia("(prefers-color-scheme: dark)").matches;
  const g = buildGraph(data);
  const clusters = clustersOf(g);
  style(g, clusters, dark);
  layout(g);

  let hovered: string | null = null;
  let selected: string | null = null;
  let hidden = new Set<string>();
  let onlyCluster: number | null = null;
  const clusterListeners: ((cluster: number | null) => void)[] = [];
  const inCluster = new Set<string>();
  let labels = true;
  const lit = new Set<string>();
  // A hub can have hundreds of neighbours; naming them all draws an unreadable pile.
  const named = new Set<string>();

  const refreshLit = () => {
    lit.clear();
    named.clear();
    const focus = hovered ?? selected;
    if (!focus || !g.hasNode(focus)) return;
    lit.add(focus);
    g.forEachNeighbor(focus, (n) => lit.add(n));
    named.add(focus);
    [...lit]
      .sort((a, b) => g.getNodeAttribute(b, "size") - g.getNodeAttribute(a, "size"))
      .slice(0, MAX_NAMED)
      .forEach((n) => named.add(n));
  };

  const dim = dark ? "#2b261d" : "#e4dcc8";
  const ink = dark ? "#f2ecdc" : "#1d1a14";
  const sigma = new Sigma(g, container, {
    renderEdgeLabels: true,
    edgeLabelFont: "system-ui, sans-serif",
    edgeLabelSize: 11,
    edgeLabelColor: { color: dark ? "#b0a78f" : "#5e574a" },
    labelFont: "Inter Variable, system-ui, sans-serif",
    labelColor: { color: ink },
    labelSize: 12,
    labelWeight: "500",
    labelDensity: 0.6,
    labelGridCellSize: 90,
    labelRenderedSizeThreshold: 7,
    defaultEdgeType: "line",
    defaultDrawNodeHover: (ctx, d, settings) => {
      const size = settings.labelSize;
      ctx.font = `600 ${size}px ${settings.labelFont}`;
      const w = d.label ? ctx.measureText(d.label).width + 12 : 0;
      ctx.fillStyle = dark ? "#211d15" : "#fffdf7";
      ctx.strokeStyle = dark ? "#3a3426" : "#d8cfb8";
      ctx.beginPath();
      ctx.roundRect(d.x + d.size + 3, d.y - size / 2 - 5, w, size + 10, 5);
      if (w) {
        ctx.fill();
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.size + 3, 0, Math.PI * 2);
      ctx.fillStyle = ink;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.size, 0, Math.PI * 2);
      ctx.fillStyle = d.color;
      ctx.fill();
      if (d.label) {
        ctx.fillStyle = ink;
        ctx.fillText(d.label, d.x + d.size + 9, d.y + size / 3);
      }
    },
    zIndex: true,
    minCameraRatio: 0.05,
    maxCameraRatio: 4,
    nodeReducer: (id, a) => {
      const out = { ...a };
      if (hidden.has(a.kind) || (onlyCluster !== null && !inCluster.has(id))) {
        out.hidden = true;
        return out;
      }
      if (!labels && !lit.has(id)) out.label = "";
      if (lit.size > 0) {
        if (lit.has(id)) {
          out.zIndex = 2;
          out.forceLabel = named.has(id);
          if (id === (hovered ?? selected)) out.highlighted = true;
        } else {
          out.color = dim;
          out.label = "";
          out.zIndex = 0;
        }
      }
      return out;
    },
    edgeReducer: (id, a) => {
      const out = { ...a };
      out.label = "";
      const [s, t] = g.extremities(id);
      const sa = g.getNodeAttributes(s);
      const ta = g.getNodeAttributes(t);
      if (
        hidden.has(a.kind) ||
        hidden.has(sa.kind) ||
        hidden.has(ta.kind) ||
        (onlyCluster !== null && (!inCluster.has(s) || !inCluster.has(t)))
      ) {
        out.hidden = true;
        return out;
      }
      if (lit.size > 0) {
        const focus = hovered ?? selected;
        if (s === focus || t === focus) {
          out.color = mix(INK[dark ? "dark" : "light"], PAPER[dark ? "dark" : "light"], 0.65);
          out.size = 1.5;
          out.zIndex = 2;
          out.label = named.has(s) && named.has(t) ? a.label : "";
        } else {
          out.hidden = true;
        }
      }
      return out;
    },
  });

  sigma.on("enterNode", ({ node }) => {
    hovered = node;
    refreshLit();
    container.style.cursor = "pointer";
    opts.onHover?.(g.getNodeAttribute(node, "raw"));
    sigma.refresh({ skipIndexation: true });
  });
  sigma.on("leaveNode", () => {
    hovered = null;
    refreshLit();
    container.style.cursor = "";
    opts.onHover?.(null);
    sigma.refresh({ skipIndexation: true });
  });
  sigma.on("clickNode", ({ node }) => api.select(node));
  sigma.on("clickStage", () => api.select(null));

  let worker: FA2Layout | null = null;
  if (opts.animateMs && g.order > 0 && typeof Worker !== "undefined") {
    try {
      worker = new FA2Layout(g, {
        getEdgeWeight: "weight",
        settings: { ...forceAtlas2.inferSettings(g), ...FA2_SETTINGS, barnesHutOptimize: g.order > 400, slowDown: 8 },
      });
      worker.start();
      setTimeout(() => worker?.stop(), opts.animateMs);
    } catch {
      worker = null;
    }
  }

  /** Move the camera so these nodes fill the view. */
  const frame = (ids: string[]) => {
    const pts = ids.map((id) => sigma.getNodeDisplayData(id)).filter((d) => d !== undefined);
    if (!pts.length) return;
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const ratio = Math.min(1, Math.max(0.12, Math.max(x1 - x0, y1 - y0) * 1.3));
    sigma.getCamera().animate({ x: (x0 + x1) / 2, y: (y0 + y1) / 2, ratio }, { duration: 450 });
  };

  const api: Viewer = {
    graph: g,
    sigma,
    clusters,
    node: (id) => (g.hasNode(id) ? g.getNodeAttribute(id, "raw") : undefined),
    neighbours(id) {
      if (!g.hasNode(id)) return [];
      const out: { node: GraphNode; edge: GraphEdge; outgoing: boolean }[] = [];
      g.forEachOutEdge(id, (_e, a, _s, t) => {
        for (const raw of a.all as GraphEdge[]) out.push({ node: g.getNodeAttribute(t, "raw"), edge: raw, outgoing: true });
      });
      g.forEachInEdge(id, (_e, a, s) => {
        for (const raw of a.all as GraphEdge[]) out.push({ node: g.getNodeAttribute(s, "raw"), edge: raw, outgoing: false });
      });
      return out;
    },
    setHiddenKinds(kinds) {
      hidden = new Set(kinds);
      sigma.refresh({ skipIndexation: true });
    },
    setCluster(cluster) {
      onlyCluster = cluster;
      inCluster.clear();
      if (cluster !== null) {
        g.forEachNode((id, a) => {
          if (a.kind !== "entity" || a.cluster !== cluster) return;
          inCluster.add(id);
          g.forEachNeighbor(id, (n) => inCluster.add(n));
        });
      }
      sigma.refresh({ skipIndexation: true });
      for (const fn of clusterListeners) fn(cluster);
      if (cluster === null) return api.fit();
      frame([...inCluster]);
    },
    onCluster(fn) {
      clusterListeners.push(fn);
    },
    setLabels(on) {
      labels = on;
      sigma.refresh({ skipIndexation: true });
    },
    zoom(factor) {
      const camera = sigma.getCamera();
      camera.animate({ ratio: camera.ratio / factor }, { duration: 250 });
    },
    select(id) {
      selected = id && g.hasNode(id) ? id : null;
      if (selected && onlyCluster !== null && !inCluster.has(selected)) api.setCluster(null);
      refreshLit();
      sigma.refresh({ skipIndexation: true });
      opts.onSelect?.(selected ? g.getNodeAttribute(selected, "raw") : null);
      if (selected) frame([selected, ...g.neighbors(selected)]);
    },
    search(query, limit = 8) {
      const q = query.trim().toLowerCase();
      if (!q) return [];
      const hits: [number, GraphNode][] = [];
      g.forEachNode((id, a) => {
        const raw = a.raw as GraphNode;
        const names = [raw.label, ...((raw.data?.aliases as string[] | undefined) ?? [])].map((s) => s.toLowerCase());
        const best = Math.min(...names.map((n) => (n === q ? 0 : n.startsWith(q) ? 1 : n.includes(q) ? 2 : 9)));
        if (best < 9) hits.push([best * 1000 - g.degree(id), raw]);
      });
      return hits.sort((a, b) => a[0] - b[0]).slice(0, limit).map((h) => h[1]);
    },
    fit() {
      sigma.getCamera().animatedReset({ duration: 400 });
    },
    destroy() {
      worker?.kill();
      sigma.kill();
    },
  };
  return api;
}

export const NODE_KIND_LABELS: Record<NodeKind, string> = {
  entity: "Entities",
  source: "Sources",
  collection: "Collections",
  file: "Memory files",
  category: "Memory categories",
};

export const EDGE_KIND_LABELS: Record<EdgeKind, string> = {
  fact: "Facts",
  names: "Names",
  evidence: "Evidence",
  in_collection: "In collection",
  in_category: "In category",
  mentions: "Mentions",
  cites: "Cites",
};
