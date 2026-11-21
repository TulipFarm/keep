# Memory

[Keep](https://keep.tulipfarm.ai) · [GitHub](https://github.com/TulipFarm/keep) · [TulipFarm](https://tulipfarm.ai)

Memory is the part of Keep that an agent reads at the start of every
session. It is a small notebook of Markdown files: who the user is, how they
like to work, what a project has decided. People and agents edit the same
files, and every change can be reviewed, undone or permanently erased.

This page explains the model. The code lives in `crates/keep-memory` (the
model, operations, context assembly and export) and `crates/keep-ledger`
(the Postgres store).

## Files and categories

A Memory tree belongs to one scope, such as a user. It holds Markdown files
at stable paths:

| Path | Category | Used for |
| --- | --- | --- |
| `/profile.md` | You | Standing facts and preferences, always considered for context |
| `/you/<slug>.md` | You | More standing files about the user, such as who they are |
| `/topics/<slug>.md` | Topics | Subjects that come up repeatedly |
| `/areas/<slug>.md` | Areas | Ongoing responsibilities |
| `/people/<slug>.md` | People | Notes about people the user works with |
| `/projects/<uuid>/index.md` | Projects | A project's overview, decisions and ways of working |
| anything else | Other | Uncategorised notes |

Paths are case-insensitively unique within a tree, must end in `.md`, and
cannot contain hidden or relative segments. Only callers allowed to pin
profile content may create or move files into the You category, because that
content reaches every conversation.

Each file has metadata: an id that survives renames, its path, category,
display name, a short description, an optional project id, its rendered size,
when it was last updated, and a **version** hash. The version covers
everything a person can see: the text, the metadata and the annotations.

## What you see and what is stored

You always see and edit plain Markdown. Underneath, each save is stored as an
immutable revision, and the text of that revision is cut into **entries**:

- By default, one entry per heading section. Text before the first heading
  and front matter are entries of their own.
- A file can opt in to bullet entries, which also gives each top-level list
  item its own entry.
- A file can also be stored raw, as one opaque entry. The benchmark uses this
  as a baseline.

Cutting is lossless: joining the entries gives back the exact bytes that were
saved, including whitespace, line endings and odd Markdown. Property tests
check this for arbitrary input.

When a file is edited, the new text is matched against the previous entries
so that unchanged and lightly edited sections keep their identity. An entry
carries hidden data that the Markdown never shows:

- when it was created and last updated;
- **lineage**: the sources it was derived from, which can restrict who may
  read it. Lineage is protected. Editing or deleting text through Markdown
  can never remove it; a section rewritten from scratch inherits the lineage
  of the sections it replaced.
- an optional **annotation**: a confidence level, a source note and a
  validity window (`valid_from` and `valid_to`). Annotations are descriptive
  and can be changed through a separate operation.

## Operations

Every write names the version it expects to replace. If someone else saved
in the meantime, the write fails with a version conflict that carries the
current head, and nothing is changed. Creating a file can likewise require
the tree to still be at a given revision.

| Operation | What it does |
| --- | --- |
| create | Add a file at a path |
| view, list | Read one file, or page through the directory |
| edit | Replace a unique string, insert at a line, apply ordered patch hunks, or rewrite the file |
| edit with merge | As edit, but a stale edit is replayed onto the newer version when the two changed different entries |
| rename | Move a file to another path, keeping its id and history |
| update metadata, annotate | Change the display name, description or entry mode (a new revision that shows in history and can be undone), or one entry's annotation |
| delete | Move a file to the trash |
| restore | Bring a file back from the trash |
| undo | Write a new revision with an earlier text; history only grows |
| history | List retained revisions |
| forget | Permanently erase content (see below) |

**Merging.** A plain edit is strict compare-and-swap. The merging variant
compares both changes against the version the caller started from. If the
two sides changed, removed or inserted next to different entries, the
caller's changes are applied to the current version and saved as a merge.
If they touched any of the same entries, it fails with a merge conflict
listing them, and nothing is written.

**Retries.** A write can carry an idempotency key. Repeating the same request
with the same key returns the original result instead of writing twice.
Reusing a key for a different request is an error. Receipts store a keyed
digest of the request, never its text.

**History and trash.** Up to 100 earlier revisions per file are kept, for at
most 30 days; the current version does not count towards either limit.
Deleted files stay in the trash for 7 days and can be restored until a sweep
removes them.

## Forget

Undo and trash are for mistakes. Forget is for content that must be gone. It
can target a whole file, one entry across all of that file's revisions, or
every occurrence of an exact piece of text in a file or the whole tree.

Forget removes the content from every retained revision, from the trash,
from annotations and descriptions that could repeat it, and from idempotency
receipts, so retrying an old request cannot bring the text back. It cannot
be undone. A record is written to an erasure journal first. The record holds
no content: only which files and entries were affected, when, and by whom.
After restoring a database backup, replaying the journal applies those
erasures again.

## Limits

Limits keep Memory small enough to read in full and fast to assemble:

| Limit | Default |
| --- | --- |
| Rendered text per file | 8 KiB |
| Files per tree | 128 |
| Rendered text per tree | 512 KiB |
| Entries per file | 128 |
| Hidden metadata per file | 64 KiB |

Every operation that would break a limit fails with a typed error naming the
limit, its value and the attempted size.

## Context for a conversation

`memory.context` builds the block of text an agent receives at the start of
a conversation. It calls no model and gives the same output for the same
input. The default budget is 4,096 tokens:

| Part | Tokens |
| --- | --- |
| Framing | 256 |
| Profile | 768 |
| Directory of all files | 768 |
| Knowledge table of contents (supplied by the caller) | 640 |
| Selected files | 1,664 |

Unused tokens from the other parts go to the selected files. The profile
leads with a "Prompt essentials" section if it has one. The directory lists
every file when it fits. When it doesn't, it says it is partial, gives counts
per category, and tells the agent how to page through the rest. Files are
chosen by project, explicit reference, word overlap with the current request,
and recency. A file that does not fit whole is cut to its most relevant
entries, marked as an excerpt, with the byte ranges it kept. The finished
block is counted again, and it never exceeds the budget.

**Several trees in one block.** The context can merge trees in layers, such
as a project tree followed by the user's personal tree. The layers share one
budget with the same parts: there is no extra allowance per tree. Each file
block, directory entry and included-file record carries an `origin` label,
such as `project` or `personal`. If the same path exists in two layers, the
earlier layer's file is used and the later one is left out, with a note.
Among files that rank equally on project, reference and word overlap,
earlier layers come first. With a single tree and no label, the output is
unchanged.

Memory content is presented as data, not instructions. The block opens with
a warning to that effect, and each file is wrapped as a labelled JSON string,
so its text cannot pose as part of the surrounding prompt.

## Export and import

Export writes the current files to a plain folder of Markdown, plus an
optional `.memory/manifest.json` with ids, versions and checksums. Revisions,
lineage and annotations stay behind.

Some file names make coding agents treat a file as trusted instructions, for
example `AGENTS.md`, `CLAUDE.md` or anything under `.github/`. By default,
export refuses to write such files. The caller can skip them instead, or
explicitly allow them, in which case each one starts with a visible comment
saying it came from Memory.

Import never writes straight away. It first produces a preview: for each file,
whether it would be created, changed (with a line diff), left unchanged, or
rejected, and why. When the manifest shows that a file changed in Memory
since export, import reports a conflict instead of overwriting. Applying the
preview writes each change with compare-and-swap. Import never deletes, and
there is no live sync.

## Storage

The service works against a ledger interface with two implementations. The
in-memory ledger is used for tests and the benchmark. The Postgres ledger
keeps trees, files, revisions, entries, receipts and the erasure journal in
tables prefixed `memory_`, and applies its own migrations. Both pass the same
conformance suite (`keep_memory::conformance`).

```sh
KEEP_TEST_POSTGRES_URL='postgres://keep:keep-local-only@127.0.0.1:15433/keep' \
cargo test -p keep-ledger
```

Each test run uses its own Postgres schema and drops it afterwards.

## In the benchmark

The `keep-m` arms in `keep-bench` exercise this crate end to end. A simple,
deterministic curator writes the visible conversation into a Memory tree:
first-person preferences go into the profile, everything else into dated
topic notes. `memory.context` then builds the standing context under the
manifest's `standing_tokens`. The three representation arms (`keep-m-raw`,
`keep-m` for sections, `keep-m-structured-bullets`) store byte-identical
Markdown, so any difference between them comes from how it is stored and
excerpted. The curator stands in for a model-driven one, so these arms
measure the storage and selection machinery, not the quality of curation.

The `keep-mk0` and `keep-mk1` arms put the same Memory context in front of
Knowledge evidence. For `keep-mk1` the bench fetches the Knowledge table of
contents first and hands it to `memory.context` through the Knowledge
callback, so it is charged to the standing budget like any other block.
