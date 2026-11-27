# Harry Potter demo graph

This directory builds the graph on Keep's site: 142 pages of the Harry
Potter Wiki read into Knowledge, plus four invented reader notes in Memory.
The result is `site/public/demo/harry-potter.json`.

The wiki text is not in this repository. `fetch.py` saves it under
`data/harry-potter/`, which git ignores.

## Credit and licence

The page text comes from the [Harry Potter Wiki](https://harrypotter.fandom.com/)
on Fandom, under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
Each source keeps its page URL and revision in its metadata, and the exported
file carries the credit. Anything that shows the graph must show that credit.

Harry Potter names are trademarks of Warner Bros. Entertainment Inc. This demo
is not made, approved or endorsed by Warner Bros. or J.K. Rowling.

## Run it

```sh
python3 demo/harry-potter/fetch.py           # pages.txt -> data/harry-potter/pages/
python3 demo/harry-potter/ingest.py --count  # sources, episodes (one model call each)
```

Start a dev server with model extraction. The demo used the Copilot CLI,
`gpt-5.6-luna` and a Postgres + Qdrant profile:

```sh
cargo build --release -p keep-server --features local-embeddings --locked
KEEP_MODEL_API_STYLE=copilot-cli KEEP_MODEL_NAME=gpt-5.6-luna \
KEEP_MODEL_BUDGET_CREDITS=200 KEEP_MODEL_BUDGET_CALLS=450 \
KEEP_MODEL_TIMEOUT_SECS=300 KEEP_MODEL_REASONING_EFFORT=low \
./target/release/keep --project harry-potter --storage postgres-qdrant \
  --collection hp_knowledge serve --dev --bind 127.0.0.1:7461
```

`--storage postgres-qdrant` reads `KEEP_POSTGRES_URL`, `KEEP_QDRANT_URL` and
`KEEP_QDRANT_API_KEY`. Keep them in a file outside the repository.

Then read the pages in and save the graph:

```sh
python3 demo/harry-potter/ingest.py http://127.0.0.1:7461
python3 demo/harry-potter/export.py http://127.0.0.1:7461
```

`ingest.py --only <slug>` reads single pages, for a cheap first check.
Reading a source again only extracts its new episodes. `export.py` refuses
a truncated graph. Open `http://127.0.0.1:7461/dev/graph` to look at it.

Extraction files short and full names as separate things ("Harry" and
"Harry Potter"). `export.py` joins the names in `aliases.json` into one
dot, in the exported file only. Keep's own data does not change.
