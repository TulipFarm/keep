# Knowledge

Knowledge is the part of Keep that remembers what happened and what was
learned from it. It is searched on demand, unlike Memory files, which an
agent reads at the start of every session. The code lives in
`crates/keep-knowledge`.

This page explains the ideas. The Rust API docs (`cargo doc -p
keep-knowledge --open`) cover the details.

## What is stored

| Record | What it is |
| --- | --- |
| **Source** | Where content came from: a conversation, a document or an import. Each source has one *partition*, its visibility label. |
| **Episode** | One timestamped message or document section in a source. Episodes are never edited. Re-sending the same episode is a no-op. Sending different text under the same id is an error. |
| **Chunk** | A searchable slice of an episode, about 1,200 characters, cut at sentence or word boundaries. |
| **Entity** | A person, project, place or other thing, with aliases ("Tomas Reyes", also known as "Tom"). |
| **Fact** | A claim about an entity, such as *Priya works at Northwind*. A fact has versions; it is never overwritten. |
| **Evidence** | A link from a fact version to the exact span of the episode that supports it. |
| **Job** | Background work (chunk, embed, extract) with explicit states: queued, running, done or failed, with bounded retries. |

Everything derived from a source carries that source's partition. A chunk,
fact or index entry is visible only when the caller may read its partition.
Unknown partitions are denied: access fails closed.

A *corpus* is a hard boundary, such as one user or one benchmark history.
Nothing is shared or searched across corpora.

## Two kinds of time

Every fact version records two time ranges:

- **Valid time** (`valid_from`, `valid_to`): when the fact was true in the
  world. "Priya worked at Northwind from March to June."
- **Known time** (`recorded_at`, `invalidated_at`): when Keep learned it and
  when Keep stopped believing it.

So two different questions can be asked:

- *What was true on 1 May?* Filter by valid time.
- *What did we know on 1 May?* Filter by known time. Facts learned later are
  hidden, and so is evidence that arrived later.

Searches accept both (`AsOf { valid_at, known_at }`). Without them, the
latest knowledge is used, and superseded facts are labelled as such rather
than hidden.

## Ingestion

1. `ingest` stores the source and its new episodes and queues jobs. It is
   idempotent: replaying a batch changes nothing.
2. `run_jobs` chunks the episodes, embeds the chunks through an `Embedder`,
   and extracts facts through a `FactExtractor`.
3. The default extractor is a small, deterministic rule set. It is
   deliberately conservative: it only records clear statements such as
   "X works at Y", "X lives in Y", "X prefers Y" or "X is known as Y". Each
   claim must quote an exact span of the episode and name its subject.
   Pronouns are not resolved, and anything that does not match is dropped.
   An LLM extractor can be
   plugged in behind the same trait. A prompt template exists, but nothing in
   this crate calls a model.
4. Session-level extraction ([`session_extract`](../crates/keep-knowledge/src/session_extract.rs))
   lets a model read a whole session at once, which the per-episode rules
   cannot. See [Session fact extraction](#session-fact-extraction).
   `Knowledge::with_session_extractor` puts the same extractor in the job
   queue instead of the rules: each episode is read alone, in windows of at
   most 8 turns of 4,000 characters, and model failures become retried or
   failed jobs. The server enables it when a chat model is configured
   ([server docs](server.md#model-backed-extraction-and-summaries)).
5. `ingest_revision` replaces a source whose text or metadata changed: the
   old episodes, chunks, vectors and evidence go and the new ones arrive in
   one ledger transaction, so reads never see both revisions. Plain
   `ingest` refuses changed text under an existing episode id.

### Contradictions

Some predicates have one value at a time (`lives_in`, `works_at`, `prefers`,
`role`). When a newer statement disagrees with an older one, the older version
is closed, not deleted:

- If the new statement is later in world time, the old fact is
  **superseded**. Its valid time ends where the new one begins.
- If the timing cannot be ordered, both are kept and marked **disputed**.

Either way, the old version keeps its evidence and stays available to
"as of" queries.

### Session fact extraction

`SessionExtractor` sends one session (its date and every turn with its id)
to a chat model with a fixed prompt (`session-facts-v2`) and parses a JSON
list of facts. Each fact has:

- `subject`, an entity name ("user" for the speaker), and `attribute`, a
  short snake_case key normalised by `normalize_attribute`;
- `value` and a one-sentence `statement` that stands on its own;
- `kind`: `state` (one current value, such as where someone lives),
  `event` (something that happened on a date), or `preference` (a taste,
  habit, opinion, interest, goal, constraint or the tools and brands
  someone uses);
- `when`, the date the fact became true, and `turns`, the turn ids it
  rests on, plus a confidence.

The model is told the session date and asked to resolve relative dates
("last Tuesday", "two weeks ago") itself. `resolve_when` then checks the
reply deterministically: an absolute date is kept, and a phrase the model
left relative is resolved against the session date with the same rules as
[`time_ref`](../crates/keep-knowledge/src/time_ref.rs). Facts with no turn
ids, unknown turn ids or an unknown kind are dropped, not guessed.

Replies are cached on disk under `(model id, prompt version, SHA-256 of the
rendered session)`, so a rerun, another arm or another manifest with the
same model and prompt reads the cache instead of calling the model.
Changing the prompt text requires a new prompt version.

`Knowledge::apply_claims` stores the facts in the usual `facts`,
`fact_versions` and `evidence` tables; no migration is needed. Each claim
cites whole turns as evidence. Supersession is a **deterministic key
match**, not a model decision: `state` facts are single-valued on
(subject entity, attribute), so a new value closes the previous version
under the [rules above](#contradictions) (later start supersedes, earlier
start closes the incoming fact, equal or unknown starts are disputed).
`event` facts carry their date in the object ("met Emma @ 2023-04-11") and
`preference` facts are multi-valued, so neither replaces another.

### Session summaries

[`SessionSummarizer`](../crates/keep-knowledge/src/session_summary.rs)
asks the same model for two to four sentences per session (prompt
`session-summary-v1`): who did, planned or felt what, with names, places,
numbers and absolute dates, each point attributed to its person. Facts
answer "what is X?" well but lose which events belong to one occasion and
how a session hangs together; summaries keep that for questions that join
several sessions. A session with nothing about the people summarizes to
an empty string. Summaries are rendered from the same session text as fact
extraction and cached in the same on-disk cache under their own prompt
version, so a summary and an extraction of one session never collide.

### Who a question is about

[`subject::Grounding`](../crates/keep-knowledge/src/subject.rs) finds the
participants a question names, by whole word and case-insensitively, among
the history's speakers. When it names some but not all of them, `about`
classifies a fact's subject as the person asked about, another
participant, or neither. A subject naming the person asked about counts as
theirs even when it names someone else too ("Ada's brother Ben"). Histories
between a user and an assistant, and questions naming no participant or
every participant, have no grounding.

### Question time references

[`time_ref::question_window`](../crates/keep-knowledge/src/time_ref.rs)
turns a question's time reference into a date window, using the question
date: "last week", "past 3 months", "two weeks ago" (with slack), "this
year", "yesterday", "last Saturday", "last summer", "in May", "in 2022".
Questions without one return no window. The fact arms in the benchmark use
the window to rank facts and evidence from it higher; nothing is filtered
out.

## Retrieval

`search` combines two rankings:

- **Lexical**: BM25 in memory, or Postgres full-text search.
- **Vector**: pgvector or Qdrant, depending on the storage profile.

The two are merged with **reciprocal rank fusion** (RRF): each result scores
`1 / (60 + rank)` in each list, and the scores are added. Fusion uses ranks,
not raw scores, so the two retrievers need no calibration.

Permissions are applied **before** ranking. Each retriever only searches the
partitions the caller may read, and every candidate is checked against the
ledger again before a snippet is produced.

#### Visible-first hybrid search

`KnowledgeConfig::lexical` picks the lexical retriever. The default,
`Ledger`, is the ledger search above: both pools are cut to
`candidate_pool` first, and time filtering ("as of" reads) happens after
that, so a question about an early date can lose every candidate to later
items it cannot see.

`LexicalMode::Bm25` fixes that. It loads the corpus into an in-process
snapshot (reloaded when a partition's generation changes) and decides which
chunks and facts the reader may see at the requested time **before**
ranking:

- **Lexical**: BM25 over exactly the visible items. A chunk is indexed as a
  hit shows it, `[date] speaker: text`, so a question that names a speaker
  matches that speaker's turns. Facts are indexed by their claim.
- **Vector**: the vector search asks the index for the pool size plus the
  number of invisible items, drops the invisible ones, and keeps the first
  `candidate_pool` visible points.
- **Fusion**: the two pools are fused with RRF (`k = 60`), as above.
- **Reranking** (optional): `Knowledge::with_reranker(reranker, top_n)`
  reorders the fused top `top_n` with a cross-encoder that scores each
  candidate's displayed text against the query. Equal scores keep their
  fused order, so ranking stays deterministic. The rest of the list is
  left in fused order. The method string records the reranker's model and
  revision.

Source titles are fetched once per search, not once per hit.

`keep-providers` includes a local ONNX cross-encoder (`local-onnx`, behind
the `local-embeddings` feature) with the same pinned-download and digest
checks as the local embedder. Tests use mock rerankers and never download
a model.

Other read tools:

- `fetch`: one fact, chunk or episode by id, with its evidence.
- `cite`: stable citations (source, episode, character span, quote) for a
  result.
- `traverse`: the facts around an entity, limited in depth and fan-out.

### Source allowlists

`Access::only_sources` and `Access::only_collections` narrow a read to
evidence from some sources. Search applies them inside retrieval: lexical
queries filter by source, and vectors carry a per-source tag that pgvector
and Qdrant filter on, so the allowlist cuts candidates before ranking,
limits and the evidence allowance rather than after. A fact supported by
several sources is visible when one of them is allowed and shows only the
allowed evidence. Fetch and cite treat anything outside the allowlist as
missing. Vectors projected before tags existed need
`rebuild_projection` before vector search can match them under a filter.

### Evidence allowance

Each answer has an allowance of retained evidence, 2,000 tokens by default
and 8,000 at most. Results are added in rank order until the allowance runs
out. The response then reports how many results were left out, rather than
cutting text silently.

## The book index

`index` describes what is in the knowledge base, much like a book's front
and back matter. It never returns the content itself.

| Level | Like | Contents | Default budget |
| --- | --- | --- | --- |
| 0 | Table of contents | One line per collection, with counts and date range. Always fits a 640-token slot, so it can sit in every prompt. | 640 |
| 1 | Chapters | The sources in one collection, or in all of them, with dates and top entities, paged by cursor | 1,000 |
| 2 | Back-of-book index | Entities and aliases, with pointers to the sources that mention them | 1,500 |

The index is built deterministically, without any model. It is built
separately for each partition and merged only for the partitions the caller
may read. A caller without access never sees a restricted title, entity name
or count, not even in totals.

Every write that changes a partition bumps its *generation*. A cached index
is used only if its generation still matches, so after a delete or a
permission change, the next read rebuilds from current data. A stale index
cannot reveal a removed source.

Community detection and LLM-written summaries are interfaces only. They are
disabled by default (`DisabledSummarizer`).

## Forgetting

`forget_source` erases a source and everything that exists only because of it:

- its episodes, chunks and their vectors, in pgvector or Qdrant
- the evidence it supplied
- facts left with no evidence, and their vectors
- index entries, by bumping the generation and dropping caches

Facts that are also supported by other sources stay, with only their
remaining evidence. An entry is written to an erasure journal, holding
the time, counts and SHA-256 digests of the ids, but no content.

`delete_corpus` removes a whole corpus.

## Storage profiles

| Profile | Ledger | Vectors |
| --- | --- | --- |
| `in-memory` | In process | In process |
| `postgres-vector` | Postgres | pgvector |
| `postgres-qdrant` | Postgres | Qdrant |

The Postgres ledger (tables prefixed `knowledge_`) is the source of truth.
It can share a database and schema with the Memory ledger: the two use
disjoint table prefixes, separate migration records
(`knowledge_schema_migrations`, `memory_schema_migrations`) and no foreign
keys between them, so either may migrate first. A test in `keep-bench`
checks both orders.
The vector store is a projection: `rebuild_projection` recreates it from the
ledger. If no vector store is configured, search falls back to lexical only.

## Testing

```sh
cargo test -p keep-knowledge          # unit tests and the in-memory profile
```

The same conformance suite (`tests/conformance.rs`) runs against all three
profiles. To include the database profiles, start the local stack and set:

```sh
docker compose --profile postgres-qdrant up -d
KEEP_TEST_POSTGRES_URL='postgres://keep:keep-local-only@127.0.0.1:15433/keep' \
KEEP_TEST_QDRANT_URL='http://127.0.0.1:16333' \
cargo test -p keep-knowledge -p keep-bench
```

Each test uses its own Postgres schema and Qdrant collection and removes
them afterwards.

## In the benchmark harness

`keep-bench` runs Knowledge as two ablation arms:

- **K0**: hybrid search only, within the evidence allowance.
- **K1**: the same, plus the level-0 table of contents in the standing slot.
- **Mk0**: the Memory context (see [memory.md](memory.md)) followed by K0's
  evidence.
- **Mk1**: Mk0 with the level-0 table of contents placed in the Memory
  context's Knowledge slot (640 of the 4,096 standing tokens, scaled to the
  manifest's `standing_tokens`). Any of the slot the contents leave unused
  goes to Memory files; evidence keeps its own allowance.
- **K2**: visible-first hybrid search (pools of 800, 200 results) with the
  manifest's reranker, if any, filling `evidence_tokens`.
- **Mk2**: one ranking over Knowledge results and Memory entries, filling
  `standing_tokens + evidence_tokens`, the same budget as the 6K retrieval
  baselines. Three lists are fused with RRF (`k = 60`): the K2 search, BM25
  over the Memory lines, and the cosine similarity between the question and
  each Memory entry's source turn (embedded as `[date] Role: text`, the
  form the dense baseline embeds, so the vectors come from the embedding
  cache). The reranker then reorders the fused top `top_n`.

- **K3** and **Mk3**: K2 and Mk2 with model-extracted session facts first.
  The facts known at the question date (`fact_as_of`) are ranked by RRF over
  BM25 and dense similarity to the question, grouped by subject and
  attribute, and printed as a dated block within `extraction.facts_tokens`:

  ```text
  Facts extracted from earlier sessions (dates are when each held or happened; "current" marks the latest value of something that changed):
  - user / gym class time: current value 6 pm (since 2023-05-10); 1 earlier value(s):
    [2023-03-01 until 2023-05-10, superseded] The user's gym class is at 7 pm.
    [as of 2023-05-10] The user's gym class moved to 6 pm.
  ```

  K2's or Mk2's ranked fill takes the rest of the same budget, after
  `Conversation excerpts:`, so evidence recall can only
  shrink by the facts block's size. Three options, set per arm:
  - `temporal`: the question's time window (above) is an extra RRF list
    for facts that started in it and for evidence from sessions in it; the
    block opens with `Question time reference: "last month" = A to B.` and
    every fact shows how many days before the question it started.
  - `preferences` (Mk3 only): the extracted preferences are written as dated
    bullets into a Memory tree (`/profile/preferences/NNN.md`), read back,
    and shown first as a standing block within `extraction.profile_tokens`.
    All of it when it fits, otherwise the lines most relevant to the
    question by BM25, in date order. Facts the profile shows are left out of
    the facts block.
  - `assistant_route`: facts come from what the user said, so a question
    about the assistant's own earlier reply ("what did you recommend",
    "remind me of our previous conversation", per
    [`query_intent`](../crates/keep-knowledge/src/query_intent.rs)) gets
    K2's or Mk2's plain ranked context over the whole budget instead, and
    the method is prefixed `route=assistant-reply(ranked only)`. Without
    it the profile and facts blocks pushed the assistant turns out of Mk3's
    context on the development subset.
  - `answer_prompt = "answer-v3"`: see
    [methodology](benchmarks/methodology.md#arms).

  Version 4 adds three more options. `keep-k4` and `keep-mk4` in the G0 v4
  manifests use `subjects` and `summaries`; `expand` did not help on the
  development questions and is kept for the ablation arms only (see
  [running-g0.md](benchmarks/running-g0.md#version-4-grounding-and-summaries)):
  - `subjects`: when the question has a [grounding](#who-a-question-is-about),
    the block opens with `The question asks about A. Facts and excerpts
    about B are not about A.`, and fact groups whose subject is another
    participant (and no line of which names A) move under `Facts about B,
    not about A:` after A's facts. Both sections share the facts budget and
    the rank order still decides what fits. The method records
    `subjects(asked=A;others=B)` or `subjects(none)`.
  - `expand`: one more RRF list for facts that share a capitalized named
    entity (not a participant, weekday or month) with the top five fused
    facts, or the same subject and start date (the same occasion), in BM25
    order. The method records `expand(seeds=5,added=N)`.
  - `summaries`: the most relevant [session summaries](#session-summaries)
    visible at the question date, ranked by RRF over BM25, dense similarity
    and, with `temporal`, the question's window, fill up to
    `extraction.summary_tokens` (at most half of what the profile and facts
    leave) and are shown oldest first under `Session summaries`, after the
    facts and before the excerpts. They take budget from the excerpts, not
    from the facts, so the total budget is unchanged. `keep-bench extract`
    writes them alongside the facts.

K2 and Mk2 fill their budget greedily in ranked order. A Knowledge chunk
that is a whole turn, and a Memory entry, both stand for that source turn:
each turn appears at most once, shown whole (`[date] Role: text`) when it
fits and as its shorter Memory line otherwise. Facts are shown with their
validity window. What was kept is then printed in conversation order. The
context is one `keep-ranked` block, counted against the evidence slot.

### Version 5 source-grounded facts

The v5 manifests add `keep-k5` and `keep-mk5` without changing the old
arms. They use the existing fact paths with four opt-in switches:

- `provenance` puts the original cited turn beside each fact, with its
  speaker, turn id and **session** date. A snippet is at most 600
  characters; cuts are marked. These words use the existing facts budget,
  not extra context. Preference facts can also enter this sourced block.
  Attribution is decided before adding the quote: addressing another
  participant in a source does not make the fact theirs.
- `relative_dates` stores source-time annotations at fact ingestion.
  `relative_time` keeps the original phrase and its session anchor.
  Calendar months and years are resolved; imprecise weeks and weekends
  retain their relative phrasing and anchor. The old extraction parser,
  cached model replies and fact validity dates are unchanged.
- `fact_selection` reranks individual facts with the configured local
  reranker, deduplicates matching subject/attribute/value/validity claims,
  and fills by relevance rather than by a group's newest items.
  Facts about the requested participant get first use of the budget.
- `decompose` makes at most three deterministic question facets from
  clauses and named participants. Each facet searches facts and original
  dialogue separately; reciprocal rank fusion merges the rankings.
  Source-turn deduplication and the same final token cap still apply.
  This makes no extra chat-model call and never sees gold answers or
  category labels. It does not attempt unrestricted entity inference.

`answer-v5` asks the model to check the original turn, distinguish plans
from events, anchor relative phrases to session dates rather than the
wall clock, and return complete but focused lists. Factual questions still
require evidence for the requested claim. Inference is allowed only for
questions explicitly asking for a likely preference, recommendation or
counterfactual. The control baselines use this same prompt.

### Version 6 answer notes

The v6 manifests add `keep-k6` and `keep-mk6`, which are `keep-k4` and
`keep-mk4` with the same retrieval and context, answered with
`answer-v6`. Nothing in the memory or knowledge path changes. The prompt
asks the model to write at most eight lines of working notes before a
final line that starts with `Answer:`:

1. who the question is about and what kind of answer it wants;
2. the dated evidence, listing each link when facts from different
   sessions have to be joined;
3. a person check: evidence about someone else, or a shared topic, does
   not answer a question about this person;
4. the working: relative dates resolved against their session date,
   dates derived from stated durations, counts and lists gathered across
   every session, the latest value used unless an earlier one is asked
   for.

Only the text after the last `Answer:` line is scored and judged; it is
stored as `final_answer` beside the full reply. A reply without the
marker is scored whole. "I don't know." is reserved for questions whose
only matching information is about someone else, or that ask about
something never connected to the person; the model must not correct the
question's premise instead. Questions asking what someone would, could or
might do, or what is likely, get an inference starting with "Likely"
when the memory says anything relevant about them. The `dense-6k-v6` and
`full-context-v6` controls use the same prompt.

### Version 7 answer bridging

The v7 manifests add `keep-mk7`: `keep-mk6`'s retrieval and context,
answered with `answer-v7`. Nothing in the memory or knowledge path
changes. The G0 v6 records showed that, where mk6 lost to full context,
the evidence was usually already in mk6's context; the answer misused it.
answer-v7 keeps answer-v6's notes, person check and final `Answer:` line,
and adds:

1. a bridge step: common world knowledge may link a fact from memory to
   what is asked (a city to its country, a game to its console, a park
   to its state), but the fact about the person must come from memory;
2. answers at the level of detail asked (a country when a country is
   asked, not the city);
3. inferences give the conclusion and the main reason from memory;
4. the memory's most specific wording, qualifiers kept, without
   reinterpreting it; `[shared an image: …]` captions count as evidence;
5. no "I don't know." when the memory holds the answer in other words or
   one bridging step away.

`dense-6k-v7` and `full-context-v7` use the same prompt.

### Version 8 focused excerpts

`keep-mk8` is `keep-mk7` with two opt-in retrieval switches, answered with
the same `answer-v7` prompt in the same 6,096-token budget. They target
long two-person histories (LoCoMo), where mk7 trailed full context.

- `focus` (`k3`, `mk3`): in a conversation between two named people,
  almost every turn by one names the other ("Wow, Caroline! …"), so the
  excerpt search for "What did Caroline research?" filled up with short
  replies that match only the name. With `focus`, the excerpt search
  (BM25, dense, the local reranker) uses the question with participant
  names taken out ("What did research?"), and, when the question asks about
  one participant, that participant's own turns join the ranking as one
  more RRF list. Facts, the profile and summaries still use the question
  as asked. Questions that name nobody, such as LongMemEval-S's user
  questions, are searched as before. The context method records
  `focus=v8(query=unnamed|as-asked,own=N)`.
- `verbatim_first` (`k3`, `mk3`): after a first build, the facts block is
  built again without the facts whose cited turns the excerpts already
  show word for word, and the excerpts take the freed room. A fact stays
  when it cites a turn not shown whole, carries its age at the question
  date, belongs to an attribute whose value changed (superseded or
  disputed), or cites nothing. No fact the first pass left out comes in.
  The method records `verbatim_first(pruned=N)`.

In the G2 development records, about one in five LoCoMo questions mk7
lost to full context had its gold turn missing from the context, and
about one in seven had only a fact's paraphrase of it. Offline, on the
LoCoMo development split, the two switches raise the share of gold turns
whose text is in the prompt from 85.0% to 89.6% (dense-6k: 90.7%), at the
same mean context size. On LongMemEval-S they leave evidence recall and
the facts block about the same, because facts there carry their age.

Evidence recall remains a **citation-based** metric: a fact citing a turn
does not prove that the entire turn, or its answer-bearing words, reached
the answer model. Inspect the saved prompt to distinguish these cases.

Every Keep record lists the tokens of each block (`memory`, `knowledge-toc`,
`knowledge-evidence`) and whether it counts against `standing_tokens` or
`evidence_tokens`; the run summary reports their means and maxima.

Each benchmark history becomes a separate corpus, with one source per
session. The runner ingests every history before the first arm, several at
a time, and executed runs keep their Postgres rows and Qdrant points in a
schema and collections of their own that they drop at the end (see
[methodology](benchmarks/methodology.md#building-contexts)). Questions are
asked "as known" at their question date. Try it
offline with:

```sh
cargo run -p keep-bench -- run fixtures/manifests/knowledge-mock.toml
cargo run -p keep-bench -- run fixtures/manifests/ablation-mock.toml   # M, K0, K1, Mk0, Mk1
```

Set the manifest's storage profile to `postgres-vector` or `postgres-qdrant`
to use a database. The connection URLs come from the environment variables
named in the manifest. The bench's Memory curator always uses the
in-memory Memory ledger, whatever the profile.
