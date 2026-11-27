#!/usr/bin/env python3
"""Load the fetched Harry Potter pages, plus a fan's Memory files, into Keep.

Each page becomes one Knowledge source in a collection named after its group
in pages.txt (people, places, groups, objects, events). The infobox and the
prose are cut into episodes of at most --episode-chars, so with a model
configured each episode is one extraction call.

    python3 demo/harry-potter/ingest.py --count          # episodes = model calls
    python3 demo/harry-potter/ingest.py http://127.0.0.1:7357 [--token-env KEEP_API_TOKEN]
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = HERE.parents[1] / "data" / "harry-potter" / "pages"
LICENCE = "CC BY-SA 3.0"
CREDIT = "Harry Potter Wiki on Fandom"
MAX_ITEMS = 10


def infobox_text(title: str, rows: list[list[str]]) -> str:
    lines = [f"{title} facts:"]
    for label, value in rows:
        items = [v.strip() for v in value.split(";") if v.strip()]
        lines.append(f"- {label}: {'; '.join(items[:MAX_ITEMS])}")
    return "\n".join(lines)


def episodes(page: dict, limit: int) -> list[dict]:
    blocks = []
    if page["infobox"]:
        blocks.append(infobox_text(page["title"], page["infobox"]))
    blocks.extend(page["paragraphs"])
    out, cur = [], ""
    for block in blocks:
        block = block[:limit]
        if cur and len(cur) + 2 + len(block) > limit:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{block}" if cur else block
    if cur:
        out.append(cur)
    return [{"id": f"part-{i + 1}", "speaker": CREDIT, "text": t} for i, t in enumerate(out)]


def load(limit: int) -> list[dict]:
    reqs = []
    for path in sorted(PAGES.glob("*.json")):
        page = json.loads(path.read_text())
        reqs.append(
            {
                "source_id": f"hp-wiki-{path.stem}",
                "title": page["title"],
                "collection": page["group"] or "general",
                "kind": "document",
                "episodes": episodes(page, limit),
                "metadata": {
                    "url": page["url"],
                    "revision": str(page["revid"]),
                    "licence": LICENCE,
                    "credit": CREDIT,
                },
            }
        )
    return reqs


# A reader's own notes, written for this demo. They show Memory beside
# Knowledge: Keep links each file to the entities it names.
MEMORY = {
    "/profile.md": """# Reader profile

- Reading the series for the first time, one book a month.
- Favourite character so far: Luna Lovegood. Least favourite: Dolores Umbridge.
- Wants spoiler-free answers past the current book.
""",
    "/topics/reading-log.md": """# Reading log

- Finished Philosopher's Stone. Loved Hagrid and the Mirror of Erised chapter.
- Halfway through Chamber of Secrets. Suspects Gilderoy Lockhart is a fraud.
- Next: Prisoner of Azkaban. Heard Sirius Black and Remus Lupin matter a lot.
""",
    "/topics/theories.md": """# Theories to check later

- Severus Snape is protecting Harry Potter, not hunting him.
- The Marauder's Map and the Cloak of Invisibility will come back.
- Tom Riddle's Diary felt like more than a diary.
""",
    "/topics/questions.md": """# Open questions

- Why does Albus Dumbledore trust Severus Snape?
- What happened between Peter Pettigrew and James Potter?
- How does the Sorting Hat choose between Gryffindor and Slytherin?
""",
}


def post(base: str, token: str | None, op: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{base}/v1/{op}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
        | ({"Authorization": f"Bearer {token}"} if token else {}),
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("server", nargs="?")
    ap.add_argument("--count", action="store_true", help="print the episode count and exit")
    ap.add_argument("--episode-chars", type=int, default=3800)
    ap.add_argument("--token-env", default="KEEP_API_TOKEN")
    ap.add_argument("--only", action="append", metavar="SLUG", help="ingest only these pages, and no Memory files")
    args = ap.parse_args()
    reqs = load(args.episode_chars)
    if args.only:
        reqs = [r for r in reqs if r["source_id"].removeprefix("hp-wiki-") in args.only]
    n = sum(len(r["episodes"]) for r in reqs)
    chars = sum(len(e["text"]) for r in reqs for e in r["episodes"])
    print(f"{len(reqs)} sources, {n} episodes, {chars} characters")
    if args.count or not args.server:
        return
    base, token = args.server.rstrip("/"), os.environ.get(args.token_env)
    for i, r in enumerate(reqs, 1):
        res = post(base, token, "knowledge/ingest", r)
        state = res.get("enrichment", {}).get("state", "?")
        print(f"[{i}/{len(reqs)}] {r['title']}: {state}", flush=True)
    if args.only:
        return
    for path, text in MEMORY.items():
        post(base, token, "memory/write", {"path": path, "text": text})
        print(f"memory {path}")


if __name__ == "__main__":
    main()
