# Benchmark methodology

This page explains how Keep is evaluated, why the harness is built the way it
is, and the limits of what any number it produces can tell you.

## Principles

1. **Preregister, then run.** Every run is described by a versioned manifest:
   dataset files (pinned by SHA-256), split, subset, holdout, model pins,
   prompts, retrieval settings, storage and embedding profiles, budget caps,
   seeds and the statistical test. The manifest is canonicalised and hashed;
   every artifact a run writes carries that hash, and `compare` refuses runs
   whose records do not match their manifest.
2. **Baselines first.** Keep has to beat simple, strong baselines on the same
   questions, the same answer model and the same token budget before any
   claim is made.
3. **Publish negative results.** A run that shows Keep losing, tying or being
   too expensive is reported exactly like a run that shows it winning.
4. **No silent failures.** Every question ends in an explicit state —
   answered, unavailable (for example, the history does not fit), error, or
   not run — with the reason recorded. Unavailable or failed answers are never
   dropped from a denominator without being reported.

## Datasets

| Dataset | Use | Notes |
| --- | --- | --- |
| LoCoMo (10 long conversations, ~2k questions) | development and gate | CC BY-NC 4.0. A published audit found roughly 6% of gold answers are wrong or unanswerable; category labels are also disputed, so categories keep their raw `cat-N` ids. |
| LongMemEval-S (500 questions, ~115k-token histories) | development and gate | MIT. Includes abstention questions (`_abs`). |
| LongMemEval-M (~1.5M-token histories) | later stress test | Shape-compatible loader only; the file is ~2.7 GB and needs a streaming loader before use. |

Loaders normalise every dataset into one schema: histories made of dated
sessions and turns, and questions with gold answers, evidence pointers,
category and an abstention flag. Loader warnings (for example evidence ids
that match no turn) are reported, not hidden.

### Wrong gold answers

Known-bad gold answers are handled as data, not by editing the dataset. A
manifest may pin an annotations file listing question ids marked
`wrong-gold` (with a corrected answer and source) or `exclude`. The shipped
list is empty: we do not invent labels. When annotations are used, results are
reported on both the original and the annotated gold track.

## Holdout and fit mask

- **Sealed holdout.** A fixed fraction of histories (not questions, so related
  questions stay together) is assigned to the holdout by a keyed hash. The
  manifest stores only a commitment to the holdout; development manifests
  cannot load holdout questions, and `--unseal` is rejected for them. Any change
  to the holdout definition changes the commitment and fails validation.
- **Length-only fit mask.** Comparisons against the full-context baseline are
  only meaningful on questions whose whole history fits the answer model's
  context window. The fit mask is computed from token lengths alone — never
  from answers or scores — then frozen and pinned by hash in the manifest.
  If too few questions fit (see `min_fit_fraction`), the full-context
  comparison is reported as not eligible rather than computed on a biased
  subset. How many questions fit depends on the tokenizer: with the
  crude `approx-v1` estimate and a 128K window only about 11% of
  LongMemEval-S histories fit, while with `tiktoken-o200k-base`, the gpt-4o
  tokenizer, all of them do (the longest is about 106.6K tokens).

## Arms

| Arm | What it gets |
| --- | --- |
| `no-memory` | The question only. Measures what the model already guesses. |
| `full-context` | The whole history, never truncated. A history that does not fit is reported as unavailable for that question. |
| `recent-window` | The most recent turns up to a token budget: the explicit truncation strategy, labelled as such, with dropped turns counted per question. |
| `bm25` | Lexical retrieval over turns or sessions under a token budget. |
| `dense` | Embedding retrieval under the same budget, on pgvector or Qdrant. |
| `bm25-6k`, `dense-6k` | The same two baselines with the combined Keep arms' context budget, `standing_tokens + evidence_tokens` (6096 by default), so a combined arm is never compared only against baselines that saw a third of its context. |
| `notes-grep` | A plain Markdown notes baseline searched by keyword. |
| `filesystem-agent` | An agent with file tools over Markdown notes (not run yet). |
| `oracle-evidence` | Only the gold evidence turns. An upper bound for retrieval, not a competitor. |
| `keep` | Keep paths `m`, `k0`, `k1`, `mk0`, `mk1` (Memory files, Knowledge, table of contents), the ranked paths `k2`, `mk2` and the fact paths `k3`, `mk3`, with `raw`, `structured` or `structured-bullets` representations. Records list each context block's tokens against the standing or evidence allowance. |

`keep-bench ablation-grid` prints the full set of Keep ablation arms.

The ranked paths change how Keep spends its budget. `k2` fills
`evidence_tokens` from visible-first BM25 + dense search fused with RRF
(`k = 60`) and, when the manifest has a `[reranker]`, a local
cross-encoder over the fused top `top_n`. `mk2` ranks Knowledge results and
Memory entries together (Memory by BM25 over its lines and by the dense
similarity of each entry's source turn), deduplicates by source turn,
prefers whole source turns over derived lines, and fills
`standing_tokens + evidence_tokens`: the same 6,096 tokens as `bm25-6k` and
`dense-6k`. A combined `mk2` arm needs the same equal-context baseline as
`mk0` and `mk1`. See [knowledge.md](../knowledge.md#in-the-benchmark-harness)
for the details. `k0`, `k1`, `mk0` and `mk1` are unchanged: a test pins a
digest of their contexts on the fixtures.

The fact paths `k3` and `mk3` put a dated block of model-extracted session
facts (`[extraction].facts_tokens`) ahead of `k2`'s or `mk2`'s ranked fill,
which takes the rest of the same budget; superseded values are marked and
dated. They need `[extraction]` and `[models.extract]`. Per-arm
options build on them, each recorded in the manifest and so in its hash:

- `temporal = true` (`k3`, `mk3`): parse the question's time reference
  ("last month", "two weeks ago", "in May") against the question date, rank
  facts and evidence from that window higher with an extra RRF list, show
  the window and each fact's age in days before the question.
- `preferences = true` (`mk3` only): open with a Memory profile of the
  extracted preferences, habits and opinions, within
  `[extraction].profile_tokens` (standing context, at most
  `standing_tokens`).
- `assistant_route = true` (`k3`, `mk3`): questions about the
  assistant's own earlier replies, found by a word rule, get the plain
  `k2` or `mk2` context, since extracted facts describe the user. The rule
  was written against development questions only.
- `answer_prompt = "answer-v3"` (any arm): answer with the v3 system
  message, which asks the model to work out dates step by step before
  answering, prefer the most recent value of anything that changed, count
  distinct instances across all sessions, and use the user's stated
  preferences for recommendations. Other arms keep `[models.answer]
  .prompt_version` (`answer-v1`), whose text is pinned by a test; each
  record's `prompt.version` says which one it used.
- `subjects = true` (`k3`, `mk3`): when the question names some but not
  all of the history's speakers, say who it asks about and set facts about
  the other participants apart under their own heading.
- `summaries = true` (`k3`, `mk3`): add the most relevant model-written
  session summaries within `[extraction].summary_tokens` of the same
  budget. Needs `summary_prompt_version` in `[extraction]`; summaries are
  extracted once per session and cached like facts, and priced and capped
  the same way.
- `expand = true` (`k3`, `mk3`): rank facts that share a named entity, or
  a subject and date, with the top five facts as one more RRF list. Used
  only by development ablation arms; it did not raise accuracy there, so
  `keep-k4` and `keep-mk4` leave it off.
- `answer_prompt = "answer-v4"` (any arm): v3's steps, plus: use only what
  the memory says about the person asked about; abstain only when nothing
  about that person bears on the question and otherwise give the best
  supported answer; answer with the specific item in the memory's words;
  and for "what/which" lists name only items that directly match. The G0
  v4 manifests give it to `keep-k4` and `keep-mk4` and to the control arms
  `dense-6k-v4` and `full-context-v4`, whose retrieval is unchanged, so the
  prompt's gain can be told apart from the memory's.
- `answer_prompt = "answer-v5"` adds source-first grounding, source-date
  anchoring when the question has no date, complete focused lists, and a
  stricter distinction between factual answers and requested inference.
  `keep-k5` and `keep-mk5` opt into `provenance`, `relative_dates`,
  `fact_selection` and `decompose`; `dense-6k-v5` and `full-context-v5`
  change only the answer prompt. See the [v5 design](../knowledge.md#version-5-source-grounded-facts).
- `focus = true` and `verbatim_first = true` (`k3`, `mk3`): rank excerpts
  with participant names taken out of the question plus the asked
  participant's own turns, and drop facts whose cited turns the excerpts
  already show whole. `keep-mk8` is `keep-mk7` with both. See the
  [v8 design](../knowledge.md#version-8-focused-excerpts).

See [knowledge.md](../knowledge.md#in-the-benchmark-harness) for the
details. A test pins a digest of the `k3` and `mk3` contexts without these
options, and the `answer-v3` text, so v3 runs stay reproducible.

Extraction is part of the method under test, so its calls are model calls
like any other: the dry run prices sessions missing from the extraction
cache, they count against the run's caps, `run.json` records the extraction
pass (sessions, cached, called, failed, tokens, credits), and each record's
context method names the extraction model and prompt version. See
[knowledge.md](../knowledge.md#session-fact-extraction).

Retrieval baselines are ordinary `bm25` and `dense` arms; the arm name
carries the budget. A confirmation manifest with a combined Memory +
Knowledge arm (`mk0` or `mk1`) must include a `bm25` or `dense` arm whose
`budget_tokens` is at least `standing_tokens + evidence_tokens`, or it fails
validation. Claims about a combined arm are made against that equal-context
baseline, and the dry run and report show each arm's mean context tokens so
the comparison can be checked.

Every arm records the context it built, the retrieved ids, token counts for
context, prompt and output, latency, cost estimate and the outcome.

## Scoring

- **Local metrics:** normalised exact match and SQuAD-style token F1. These are
  not identical to each benchmark's official scorer and are reported as
  local metrics.
- **LLM judge:** the primary accuracy metric, through a pinned judge model and
  a versioned prompt. Judges are lenient: they often accept answers with the
  wrong date or a contradicting detail. Before a judge is trusted,
  `keep-bench calibrate-judge` measures it on probes with known verdicts (correct
  answers, shifted dates, contradictions, distractor answers, answers to a
  different question, wrong and correct abstentions) and on
  human-labelled examples, and reports false-accept and false-reject rates.
- Abstention questions are scored on whether the answer declines.

### Consistent judge verdicts

The judge is a model call, so the same answer can get different verdicts
on different calls. In the v4 records, 98 questions had identical answer
text graded both correct and incorrect. An optional `[judge_consistency]`
manifest section removes that source of noise without changing the judge
model, its prompt or any verdict rule:

```toml
[judge_consistency]
cache = true            # reuse one verdict per distinct answer
votes = 3               # 1 (default) or 3
vote_on = "borderline"  # or "all"
```

- **Verdict cache.** Each verdict is stored under a key made of the judge
  provider, model, revision and prompt version, the protocol, the vote
  settings, the question id and text, the category, the gold answer, the
  abstention flag and the *normalised* answer (lower case, whitespace
  collapsed, quotes, markdown emphasis and trailing punctuation removed).
  An answer that normalises to the same text as one judged before reuses
  that verdict, in the same run (another arm) or a later one. The cache
  lives in `.keep-cache/judgments`, or `KEEP_JUDGE_CACHE`; one file per
  key, written atomically. Failed and malformed judgments are not cached.
  Concurrent jobs with the same key wait for one judge call.
- **Majority of three.** With `votes = 3`, a borderline answer gets a
  second vote, and a third when the first two disagree; the majority
  is the verdict. Borderline means not an abstention and a token F1
  strictly between 0 and 0.6 against the gold answer: the band where
  conflicting verdicts were concentrated. `vote_on = "all"` votes on every
  answer. A vote that fails or cannot be parsed counts as no vote; if no
  majority remains, the judgment is recorded as malformed.
- **Records.** Each record's `judge_consistency` holds the key, whether the
  verdict was judged or reused, every vote, and the extra calls. The
  summary reports reused verdicts and extra votes per arm; extra votes
  are billed and counted in the judge's tokens and credits. A dry run
  prices the worst case, every judgment at full votes.

Manifests without the section judge each answer once, exactly as before,
so earlier runs reproduce. A cached verdict is only as good as the call
that produced it: the cache makes verdicts *consistent*, not *correct*.
Delete the cache directory to judge everything afresh.

LoCoMo summaries also report **non-adversarial scheduled accuracy**:
exclude only the `adversarial` category, and keep unavailable, failed and
unjudged answers in the denominator. This view does not change the
overall score, judge or protocol, and is not directly comparable to a
published result with another judge.

`python3 scripts/analyze-failures.py results/<run>/records.jsonl` prints a
JSON review queue of questions every recorded arm got wrong, grouped by
category. `--subset manifests/dev/locomo-dev-v5.subset.txt` restricts the
review to the fixed dev subset. A missing, duplicate, unavailable or
unjudged arm excludes that question/seed pair from the complete queue.
The output includes each answer and the original gold, but does not open
or change any dataset. Agreement on failure is **not proof** of an invalid
label: keep these candidates out of tuning targets until a separate
human evidence review establishes whether the label, retrieval or judge
is at fault.

Judge scores from different judge models, prompts or harnesses are **not
comparable**. Numbers from other projects' papers are context, not baselines.

The above is Keep's own protocol, the default. A manifest's `[protocol]`
section can instead select a **parity protocol** that reproduces a published
evaluation's prompts, question filter and scoring: `mem0-locomo` (Mem0's
LoCoMo J score, categories 1-4) or `longmemeval-official` (LongMemEval's
per-type judge, with task-averaged and abstention accuracy). Parity runs
default to the development split; `split = "full"` scores the whole dataset
as published numbers do, which consumes the holdout and needs explicit
maintainer approval and `--consume-holdout`. `keep-bench export` writes an
arm's retrieval for Mem0's open harness to answer and judge unmodified. See
[parity.md](parity.md).

## Statistics and the superiority gate

Two runs are compared per question with a paired bootstrap, resampling whole
histories (clusters) so that questions about the same conversation are not
treated as independent. A candidate beats a comparator on a storage profile
only if:

- the mean accuracy gain is at least `min_gain_pp` percentage points (2 by
  default), and
- the one-sided lower confidence bound of the gain, adjusted with Holm's
  method over one family holding every comparator in every compared storage
  profile, is above zero, and
- at least `min_completion` (99% by default) of scheduled answers completed.

The gate must pass separately on every required storage profile
(`postgres-vector` and `postgres-qdrant`). `keep-bench compare` exits with code
2 when it does not pass.

## Budgets and model calls

The harness never calls a remote model unless the manifest sets
`dispatch.allow_model_calls = true`, the command passes `--allow-model-calls`,
every model and tokenizer is pinned, and the token, call and cost caps are
non-zero. `keep-bench dry-run` builds every context and estimates tokens,
calls and cost against the caps without calling anything.

USD-metered runs also share a cumulative cap. Each run reserves its
`max_cost_usd` in an append-only spend ledger (`results/spend-ledger.jsonl`
by default, ignored by git) before its first call, and is refused if that
would take settled spend plus open reservations past `--spend-cap-usd`
($300 by default). The reservation is settled to the charged cost when the
run ends. A model with `served_model` set must report exactly that model in
every reply; the first mismatch aborts the run, and `compare` refuses
aborted runs.

Two operational settings may be overridden for one run without a new
manifest, because they change how fast calls go out, not what is measured:
client-side pacing (`KEEP_DISPATCH_TOKENS_PER_MINUTE`,
`KEEP_DISPATCH_REQUESTS_PER_MINUTE`) and the number of calls in flight
(`KEEP_DISPATCH_CONCURRENCY`, overriding `dispatch.concurrency` for
answering, judging, session extraction and summaries). Neither enters the
manifest hash; `run.json` records the values used (`rate_limit`,
`concurrency`). Records land in question order whatever the concurrency,
and a test checks that a mock run gives the same records at 1 and 8.

[Running G0](running-g0.md) describes the first canonical runs, on Azure
OpenAI with pinned gpt-4o-mini and gpt-4o.

## Development runs through the Copilot CLI

For cheap iteration without an API key, a manifest can route answer and judge
calls through a locally installed, signed-in GitHub Copilot CLI
(`provider = "copilot-cli"`). These runs are **non-canonical**, and every
artifact says so:

- The CLI wraps every prompt in its own hidden system prompt (about 2,000 to
  3,000 tokens of instructions the answer model sees and Keep does not
  control). The harness runs it from an empty working directory with tools,
  built-in MCP servers and custom instructions disabled, but cannot remove
  that prompt.
- The upstream model is not pinned: the CLI names a model family, not a
  deployment, and may change it without notice. The revision is recorded as
  `unpinned-upstream`.
- The CLI has no sampling controls, so `temperature` and `seed` are not
  applied, and its usage counts come from the CLI rather than the model
  provider.

Guard rails:

- `copilot-cli` is accepted only with `track = "development"`, which in turn
  cannot be used in the confirmation phase. Dispatch also needs
  `dispatch.allow_model_calls = true` and `--allow-model-calls`.
- `run.json`, every line of `records.jsonl`, `summary.json`, `summary.md` and
  the dry-run report carry the banner `NON-CANONICAL: Copilot CLI development
  run (hidden harness system prompt, unpinned upstream model)`.
  `keep-bench compare` refuses any development-track run, so these numbers can
  never enter a canonical comparison or gate.
- `keep-bench compare --allow-non-canonical` is the one exception, for
  internal decisions such as the [internal G0](running-g0.md#internal-g0-on-copilot-cli).
  It accepts only non-canonical runs (never a mix with canonical ones), and
  only when every run has the same answer and judge models (provider, model,
  revision, reasoning effort and prompt version). Every output, on stdout and
  in the `comparison.md` and `comparison.json` written by `--out`, is stamped
  `INTERNAL / NON-CANONICAL` with the runs' labels and models, and the gate
  always reports a failing `canonical runs` check, so it never passes.
  Every comparison, canonical or not, refuses runs whose models differ.
- `budget.max_ai_credits` caps the run in AI credits. Each call reserves a
  worst-case charge (from `credits_per_1k_input`, `credits_per_1k_output` and
  `overhead_tokens` in `[models.*.copilot]`, and never less than the largest
  charge seen so far) and settles to the charge the CLI reports. A call whose
  reservation would cross the cap is refused and recorded as
  `budget exhausted`; the report counts those refusals instead of dropping the
  questions. A call that fails without reporting a charge keeps its
  reservation, so failures cannot hide spend.
- Systematic failures (authentication, an unknown model, rejected options, or
  repeated failures in a row) stop all further calls for the run.
- The report shows Keep's own prompt tokens and the CLI's hidden overhead
  tokens in separate columns. Accuracy from these runs has wide confidence
  intervals at development slice sizes and is only useful for spotting large
  regressions or broken arms.

Development manifests in `manifests/dev/` also use the local embedding
provider (`provider = "local-onnx"`, built with
`--features local-embeddings`). It runs a pinned ONNX model (repository,
commit and the SHA-256 of every file are in the manifest) downloaded to
`.keep-models/` or `$KEEP_MODEL_CACHE`, so dense and Knowledge arms need no
paid API. Token budgets in those manifests use `tiktoken-o200k-base`, an
approximate tokenizer: it is not the exact tokenizer of every model the CLI
may route to, and reports label it as an estimate.

## Building contexts

Context building is engineered to be fast without changing what any arm
sees. These choices affect run time only; each was checked against a run of
the previous code on the same questions, and the records' contexts,
retrieved and included ids and prompt hashes were identical.

- **Ingestion happens once, up front.** Before the first arm runs, every
  scored question's history is stored for the dense baseline and as a
  Knowledge corpus, up to `dispatch.concurrency` histories at a time.
  Histories have separate namespaces and corpora, each worker replays its
  sessions in date order on its own clock, and searches are exact, so the
  stores match what one-at-a-time ingestion leaves (a test checks every
  dense, K0 and K1 context). A history that fails here is retried, and its
  error recorded, when an arm first needs it. Because of this, a record's
  `build_ms` measures retrieval and rendering, not ingestion; the runner
  prints the ingestion time separately.
- **Local embeddings run on the CPU.** The `local-onnx` provider uses ONNX
  Runtime's CPU execution provider by default, with one session per three
  cores and three threads each, embedding one text per session run. On a
  10-core Apple silicon laptop the CoreML execution provider was 2 to 7
  times slower, depending on call size, and its vectors differ from the
  CPU's by up to about 1.8e-7 per component.
  `KEEP_ONNX_DEVICE=auto` restores the platform's accelerator;
  `KEEP_ONNX_SESSIONS` and `KEEP_ONNX_THREADS` override the pool. The device
  is part of the embedding cache key, so CPU and accelerator vectors never
  mix. Vectors do not depend on the session count, thread count or how texts
  are grouped into calls; a test checks they are bitwise identical.
  Embedding texts singly rather than as one padded batch avoids a known
  difference: padded batches move vector components by up to about 1.2e-7.
  Earlier runs on Apple silicon embedded with CoreML, so their vectors can
  differ from current ones by these amounts, enough to reorder near ties.
- **Requests are large.** For `local-onnx`, `embedding.batch_size` does not
  limit a request: the dense baseline embeds a whole history per call and
  stores it in one upsert. Remote providers still use `batch_size`.
- **Embeddings are cached.** Local vectors are cached by content: the key is
  a SHA-256 over the provider, model, revision, dimensions, pooling, maximum
  length, model file digests, device and the exact text. Texts are not
  normalised first, because the cached vector must be the one the model
  would have produced for that text. The cache lives in
  `$KEEP_EMBED_CACHE`, or `embeddings/` under the model cache, which is
  ignored by git; `KEEP_EMBED_CACHE=off` disables it. It is shared by all
  arms and later runs.
- **Writes are batched.** Knowledge stores a session's episodes, chunks and
  jobs with multi-row inserts, reads projections and completes jobs in one
  statement per batch, and caches Qdrant collection sizes.
- **Each run has its own storage.** An executed run against Postgres
  creates the schema `keepbench_<run id>` and puts it first in its
  `search_path` (then `public`, for extension types). Qdrant runs use the
  collections `keepbench_<run id>` and `keepbench_k_<run id>`. The run drops
  them when it ends; an interrupted run leaves only them behind. A Postgres
  URL that already sets `options` is used as given.

`keep-bench run --timings` prints per-phase timings for each arm and for
ingestion. Phases timed in concurrent workers add up across workers, so
their sum can exceed the wall time.
`manifests/dev/longmemeval-s-ingest-timing.toml` is a two-question,
mock-model reproduction of the development manifest for timing work.

## Evidence recall without model calls

Evidence recall and context size depend only on the contexts, not on the
answer model. `keep-bench contexts` builds every arm's context for every
development question with the manifest's embedder, reranker and storage,
and scores only evidence recall and context tokens. It never constructs an
answer or judge model, so it needs no `--allow-model-calls`, makes no model
calls and spends nothing; it refuses manifests whose embedder is a remote
API.

```sh
keep-bench contexts manifests/g0/locomo-copilot-v2.toml \
  [--arms bm25-6k,dense-6k,keep-mk2] [--questions ids.txt] \
  [--out target/keep-contexts] [--timings]
```

`--questions` takes a file of question ids, one per line; every id must be
in the development split, so the holdout is never loaded. Records have the
`estimated` outcome, no answer and no accuracy; the summary is stamped
"Contexts only". `manifests/dev/longmemeval-s-ranked-dev.subset.txt` is a
fixed 60-question LongMemEval-S development subset (ten per category,
chosen by a SHA-256 of each id) for quick iteration; LoCoMo is fast enough
to run whole. `manifests/dev/longmemeval-s-facts-dev.toml` runs answers on
that subset for the fact-arm ablation (`mk3` option by option against `mk2`,
`dense-6k` and `full-context`); `longmemeval-s-facts-dev-v2.toml` repeats
`mk2` and `mk3` and adds `mk2` with `answer-v3` and `mk3` with
`assistant_route`. Results are in
[running-g0.md](running-g0.md#version-3-fact-arms).

## What these benchmarks cannot tell you

- Conversational QA benchmarks test recall of chat history. They say little
  about curated, long-lived memory, multi-agent use or real workloads.
- Results depend heavily on the answer model; a change of model can outweigh
  any memory system difference.
- Mock providers exist for testing the harness. Their output is not a result.
- Copilot CLI development runs measure Keep's context under a hidden harness
  prompt and an unpinned model. Use them to find bugs and large effects, not
  to claim results.
