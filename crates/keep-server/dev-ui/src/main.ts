import { canvasTools, clusterList, emptyState, h, inspector, layerToggles, scopeForm, searchBox, stats } from "./ui";
import { type GraphData, mountGraph } from "./viewer";

async function load(): Promise<GraphData> {
  const res = await fetch(`graph.json${location.search}`, { headers: { accept: "application/json" } });
  if (!res.ok) throw new Error(`${res.status} ${(await res.text()).slice(0, 300)}`);
  return res.json();
}

function render(root: HTMLElement, data: GraphData) {
  const canvas = h("div", { class: "canvas" });
  const side = h("nav", { class: "side", "aria-label": "Graph controls" });
  const slot = h("aside", { class: "inspector" });
  const shell = h(
    "div",
    { class: "wb" },
    h("header", { class: "bar" }, h("strong", { class: "brand" }, "Keep", h("span", {}, " graph")), scopeForm(data.scope)),
    side,
    canvas,
    slot,
  );
  root.replaceChildren(shell);
  const empty = emptyState(data);
  if (empty) {
    slot.remove();
    return canvas.append(empty);
  }

  const viewer = mountGraph(canvas, data, { onSelect: (n) => panel.show(n) });
  const panel = inspector(viewer, data);
  const search = searchBox(viewer);
  slot.replaceWith(panel.el);
  canvas.append(canvasTools(viewer));
  if (data.truncated) {
    canvas.append(h("p", { class: "notice" }, "Showing the newest facts only. Add ", h("code", {}, "?max_facts="), " to the URL to see more."));
  }
  side.append(search, stats(data, viewer), clusterList(viewer), layerToggles(viewer, data));

  addEventListener("keydown", (e) => {
    const typing = (e.target as HTMLElement).closest("input, textarea");
    if (e.key === "/" && !typing) {
      e.preventDefault();
      search.querySelector("input")?.focus();
    } else if (e.key === "Escape" && !typing) {
      viewer.select(null);
    } else if (e.key === "Backspace" && !typing) {
      panel.back();
    }
  });
}

const root = document.getElementById("app")!;
load().then(
  (data) => render(root, data),
  (err: Error) => root.replaceChildren(h("div", { class: "empty" }, h("h2", {}, "Could not load the graph"), h("pre", { class: "mono" }, err.message))),
);
