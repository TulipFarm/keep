import Graph from "graphology";
import louvain from "graphology-communities-louvain";
import forceAtlas2 from "graphology-layout-forceatlas2";

export type NodeKind = "entity" | "source" | "collection" | "category" | "file";
export type EdgeKind = "fact" | "names" | "evidence" | "in_collection" | "in_category" | "mentions" | "cites";

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  group: string;
  data?: Record<string, unknown>;
  /** Position and cluster saved by `bake`, so a page need not lay out. */
  x?: number;
  y?: number;
  cluster?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: EdgeKind;
  label?: string;
}

export interface GraphData {
  scope: string;
  truncated: boolean;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface Cluster {
  id: number;
  color: string;
  size: number;
  /** The best-connected entities, as a name for the cluster. */
  top: string[];
}

const CLUSTER_COLORS = [
  "#c2553a", "#3f7cac", "#5b8c3a", "#b08a1e", "#8a4f9e", "#2f8f83",
  "#c27a3a", "#5a6abf", "#a0475e", "#6f7d2f", "#3a8fb0", "#9e6b4f",
];

const EDGE_WEIGHT: Record<EdgeKind, number> = {
  fact: 1,
  names: 0.8,
  cites: 0.8,
  mentions: 0.5,
  evidence: 0.3,
  in_collection: 0.15,
  in_category: 0.15,
};

/** Deterministic position from an id, so the same data always settles the same way. */
function seed(id: string): [number, number] {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
  const a = ((h >>> 0) % 10007) / 10007;
  const b = ((Math.imul(h, 2654435761) >>> 0) % 10007) / 10007;
  return [a * 100 - 50, b * 100 - 50];
}

export function buildGraph(data: GraphData): Graph {
  const g = new Graph({ type: "directed", multi: false, allowSelfLoops: false });
  const baked = data.nodes.length > 0 && data.nodes.every((n) => n.x !== undefined && n.y !== undefined && n.cluster !== undefined);
  g.setAttribute("baked", baked);
  for (const n of data.nodes) {
    if (g.hasNode(n.id)) continue;
    const [x, y] = baked ? [n.x!, n.y!] : seed(n.id);
    g.addNode(n.id, { x, y, raw: n, kind: n.kind, label: n.label, community: n.cluster });
  }
  for (const e of data.edges) {
    if (!g.hasNode(e.source) || !g.hasNode(e.target) || e.source === e.target) continue;
    const key = g.edge(e.source, e.target);
    if (key !== undefined) {
      const attrs = g.getEdgeAttributes(key);
      attrs.all.push(e);
      if (e.label) attrs.labels.push(e.label);
      g.setEdgeAttribute(key, "weight", attrs.weight + EDGE_WEIGHT[e.kind]);
      g.setEdgeAttribute(key, "label", attrs.labels.join(", "));
      continue;
    }
    g.addDirectedEdgeWithKey(e.id, e.source, e.target, {
      raw: e,
      all: [e],
      labels: e.label ? [e.label] : [],
      kind: e.kind,
      label: e.label ?? "",
      weight: EDGE_WEIGHT[e.kind],
    });
  }
  return g;
}

export function clustersOf(g: Graph): Cluster[] {
  if (g.size === 0) return [];
  if (!g.getAttribute("baked")) louvain.assign(g, { getEdgeWeight: "weight", nodeCommunityAttribute: "community", rng: mulberry(7) });
  const groups = new Map<number, string[]>();
  g.forEachNode((id, a) => {
    if (a.kind !== "entity") return;
    const list = groups.get(a.community) ?? [];
    list.push(id);
    groups.set(a.community, list);
  });
  const ordered = [...groups.entries()].sort((a, b) => b[1].length - a[1].length);
  const renumber = new Map<number, number>();
  const clusters = ordered.map(([community, ids], i) => {
    renumber.set(community, i);
    const top = [...ids].sort((a, b) => g.degree(b) - g.degree(a)).slice(0, 3);
    return {
      id: i,
      color: i < CLUSTER_COLORS.length ? CLUSTER_COLORS[i] : "#9a927e",
      size: ids.length,
      top: top.map((id) => g.getNodeAttribute(id, "label") as string),
    };
  });
  g.forEachNode((id, a) => g.setNodeAttribute(id, "cluster", renumber.get(a.community) ?? -1));
  return clusters;
}

function mulberry(a: number): () => number {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Start each cluster in its own sector, so the layout separates them instead of untangling one ball. */
function seedByCluster(g: Graph) {
  const communities = new Map<number, number>();
  g.forEachNode((_, a) => communities.set(a.community, (communities.get(a.community) ?? 0) + 1));
  const order = [...communities.entries()].sort((a, b) => b[1] - a[1]).map(([c]) => c);
  const angle = new Map(order.map((c, i) => [c, (i / order.length) * Math.PI * 2]));
  g.forEachNode((id, a) => {
    const t = angle.get(a.community) ?? 0;
    const r = order.length > 1 ? 100 : 0;
    const [jx, jy] = seed(id);
    g.mergeNodeAttributes(id, { x: Math.cos(t) * r + jx * 0.6, y: Math.sin(t) * r + jy * 0.6 });
  });
}

export const FA2_SETTINGS = { linLogMode: true, outboundAttractionDistribution: true, adjustSizes: false, gravity: 0.6, scalingRatio: 4 };

export function layout(g: Graph) {
  if (g.order === 0 || g.getAttribute("baked")) return;
  seedByCluster(g);
  const settings = forceAtlas2.inferSettings(g);
  const iterations = g.order < 500 ? 400 : g.order < 3000 ? 200 : 80;
  forceAtlas2.assign(g, {
    iterations,
    getEdgeWeight: "weight",
    settings: { ...settings, ...FA2_SETTINGS, barnesHutOptimize: g.order > 400 },
  });
}


/** The same data with each node's position and cluster saved, rounded. */
export function bake(data: GraphData): GraphData {
  const g = buildGraph(data);
  clustersOf(g);
  layout(g);
  const r = (v: number) => Math.round(v * 100) / 100;
  const nodes = data.nodes.map((n) => {
    const a = g.getNodeAttributes(n.id);
    return { ...n, x: r(a.x), y: r(a.y), cluster: a.cluster as number };
  });
  return { ...data, nodes };
}
