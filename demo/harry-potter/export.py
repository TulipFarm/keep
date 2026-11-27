#!/usr/bin/env python3
"""Save the demo graph from a `keep serve --dev` server as a static file.

    python3 demo/harry-potter/export.py http://127.0.0.1:7357

Writes site/public/demo/harry-potter.json, the same shape as
/dev/graph.json plus the credit the CC BY-SA text needs, then saves each
node's position and cluster into it (`pnpm bake` in the dev UI), so the
page draws it without a layout pass. Needs `pnpm install` in
crates/keep-server/dev-ui.
"""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "public" / "demo" / "harry-potter.json"
DEV_UI = ROOT / "crates" / "keep-server" / "dev-ui"
ALIASES = Path(__file__).with_name("aliases.json")

ATTRIBUTION = {
    "text": "Facts extracted from the Harry Potter Wiki on Fandom",
    "url": "https://harrypotter.fandom.com/",
    "licence": "CC BY-SA 3.0",
    "licence_url": "https://creativecommons.org/licenses/by-sa/3.0/",
    "trademark": "Harry Potter names are trademarks of Warner Bros. Entertainment Inc. "
    "This demo is not affiliated with or endorsed by Warner Bros. or J.K. Rowling.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("server")
    ap.add_argument("--max-facts", type=int, default=50_000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out.resolve()
    url = f"{args.server.rstrip('/')}/dev/graph.json?max_facts={args.max_facts}"
    with urllib.request.urlopen(url, timeout=120) as r:
        graph = json.load(r)
    if graph.get("truncated"):
        raise SystemExit("the graph was cut; raise --max-facts")
    merged = merge_aliases(graph, json.loads(ALIASES.read_text()))
    graph["attribution"] = ATTRIBUTION
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, ensure_ascii=False, separators=(",", ":")))
    subprocess.run(["pnpm", "--silent", "bake", str(out)], cwd=DEV_UI, check=True)
    kinds = {}
    for n in graph["nodes"]:
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    print(f"{out}: {len(graph['nodes'])} nodes {kinds}, "
          f"{len(graph['edges'])} edges, {out.stat().st_size // 1024} KiB, "
          f"{merged} aliases merged")


def merge_aliases(graph, aliases):
    """Fold each alias entity into the entity of its full name.

    Extraction files "Harry" and "Harry Potter" as two entities. The demo
    joins them here; Keep's own data is not changed.
    """
    entities = {n["label"]: n for n in graph["nodes"] if n["kind"] == "entity"}
    into = {}
    for alias, name in aliases.items():
        a = entities.get(alias)
        if a is None:
            continue
        b = entities.get(name)
        if b is None:
            a["label"] = name
            entities[name] = a
            continue
        into[a["id"]] = b["id"]
        seen = {(x["predicate"], x["object"]) for x in b["data"]["attributes"]}
        for x in a["data"]["attributes"]:
            if (x["predicate"], x["object"]) not in seen:
                seen.add((x["predicate"], x["object"]))
                b["data"]["attributes"].append(x)
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] not in into]
    edges, keys = [], set()
    for e in graph["edges"]:
        s, t = into.get(e["source"], e["source"]), into.get(e["target"], e["target"])
        if s == t:
            continue
        key = (e["kind"], *sorted((s, t))) if e["kind"] == "names" else (e["kind"], s, t)
        if key in keys:
            continue
        keys.add(key)
        for old, new in ((e["source"], s), (e["target"], t)):
            e["id"] = e["id"].replace(old, new)
        e["source"], e["target"] = s, t
        edges.append(e)
    graph["edges"] = edges
    return len(into)


if __name__ == "__main__":
    main()
