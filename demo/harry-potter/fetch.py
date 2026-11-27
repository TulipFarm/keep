#!/usr/bin/env python3
"""Fetch the pages in pages.txt from the Harry Potter Wiki on Fandom.

Writes one JSON file per page to data/harry-potter/pages/ (gitignored): the
infobox as label/value pairs and the opening prose, cut to --max-chars.
The text is CC BY-SA 3.0; see README.md for the credit it needs.

    python3 demo/harry-potter/fetch.py [--max-chars 8000]
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

API = "https://harrypotter.fandom.com/api.php"
WIKI = "https://harrypotter.fandom.com/wiki/"
AGENT = "keep-demo/0.1 (https://github.com/TulipFarm/keep)"
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "data" / "harry-potter" / "pages"

SKIP_TAGS = {"table", "style", "script", "sup", "figure", "noscript"}
SKIP_CLASSES = {"navbox", "toc", "reference", "mw-editsection", "quote", "dablink", "notice"}
BOILERPLATE = re.compile(
    r"^(At least some content|This article|The subject of this article|For other uses|"
    r"This page|Spoiler|Warning:|\"[^\"]{0,400}\"\s*[—–-])"
)
VOID = {"br", "img", "hr", "input", "meta", "link", "wbr", "source", "col", "area"}


class Page(HTMLParser):
    """Collects infobox pairs and top-level paragraphs, skipping clutter."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []  # (tag, kind) where kind is skip/aside/label/value/p or None
        self.infobox = []
        self.paragraphs = []
        self.buf = []
        self.label = None

    def kinds(self):
        return {k for _, k in self.stack}

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            if tag == "br" and self.stack and self.stack[-1][1] == "value":
                self.buf.append("; ")
            return
        cls = set((dict(attrs).get("class") or "").split())
        kind = None
        if tag in SKIP_TAGS or cls & SKIP_CLASSES:
            kind = "skip"
        elif tag == "aside" and "portable-infobox" in cls:
            kind = "aside"
        elif "pi-data-label" in cls:
            kind = "label"
        elif "pi-data-value" in cls:
            kind = "value"
        elif tag == "p" and "aside" not in self.kinds():
            kind = "p"
        elif tag == "li" and "value" in self.kinds():
            self.buf.append("; ")
        if kind in ("label", "value", "p"):
            self.buf = []
        self.stack.append((tag, kind))

    def handle_endtag(self, tag):
        while self.stack:
            open_tag, kind = self.stack.pop()
            if kind == "label":
                self.label = clean(" ".join(self.buf))
            elif kind == "value" and self.label:
                value = clean(" ".join(self.buf)).strip("; ")
                value = re.sub(r"(;\s*)+", "; ", value)
                if value:
                    self.infobox.append([self.label, value])
                self.label = None
            elif kind == "p":
                text = clean(" ".join(self.buf))
                if len(text) > 40 and not BOILERPLATE.match(text):
                    self.paragraphs.append(text)
            if open_tag == tag:
                break

    def handle_data(self, data):
        kinds = self.kinds()
        if "skip" in kinds:
            return
        if kinds & {"label", "value", "p"}:
            self.buf.append(data)


def clean(s: str) -> str:
    s = re.sub(r"\[\d+\]|\[citation needed\]", "", s)
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"\s+([,.;:)])", r"\1", s).strip()


def get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def pages() -> list[tuple[str, str]]:
    group, out = "", []
    for line in (HERE / "pages.txt").read_text().splitlines():
        if line.startswith("# ") and len(line.split()) == 2:
            group = line[2:].strip().lower()
        elif line.strip() and not line.startswith("#"):
            out.append((group, line.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--refetch", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for group, title in pages():
        path = OUT / f"{slug(title)}.json"
        if path.exists() and not args.refetch:
            continue
        d = get({"action": "parse", "page": title, "prop": "text|revid", "redirects": "1"})
        if "error" in d:
            print(f"skip {title}: {d['error'].get('info')}")
            continue
        p = Page()
        p.feed(d["parse"]["text"])
        text, used = [], 0
        for para in p.paragraphs:
            if used + len(para) > args.max_chars and text:
                break
            text.append(para)
            used += len(para) + 2
        name = d["parse"]["title"]
        path.write_text(
            json.dumps(
                {
                    "title": name,
                    "group": group,
                    "url": WIKI + urllib.parse.quote(name.replace(" ", "_")),
                    "revid": d["parse"]["revid"],
                    "infobox": p.infobox,
                    "paragraphs": text,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        print(f"{name}: {len(p.infobox)} infobox rows, {used} chars")
        time.sleep(1)


if __name__ == "__main__":
    main()
