# Parity protocols: scoring like published memory benchmarks

Keep's own protocol ([methodology](methodology.md)) uses its own answer and
judge prompts and scores every LoCoMo category. Published numbers from other
memory systems use different prompts, judges and question sets, so the two
are not comparable. A **parity protocol** runs Keep's arms under the answer
prompt, judge prompts, question filter and scoring of a published evaluation,
so that the protocol, at least, matches.

Two protocols exist, selected in the manifest:

```toml
[protocol]
name = "mem0-locomo"           # or "longmemeval-official"; default "keep"
split = "dev"                  # or "full"; default "dev"
```

Without a `[protocol]` section (or with `name = "keep"`) nothing changes:
the manifest hash, every artifact and every report are byte-identical to
before protocols existed.

## `mem0-locomo`

What Mem0 reports as the LoCoMo "J" score: the
[Mem0 paper](https://arxiv.org/abs/2504.19413) and Mem0's open
[`memory-benchmarks`](https://github.com/mem0ai/memory-benchmarks) harness.

- **Questions:** categories 1-4 only (multi-hop, temporal, open-domain,
  single-hop). The adversarial category 5 is excluded, as upstream does. On
  the full dataset this is 1,540 questions in 10 conversations.
- **Answer prompt:** Mem0's `ANSWER_GENERATION_PROMPT`, one user message.
  The arm's context is split into memories (see below), shown oldest first as
  `(Weekday, Month DD, YYYY) memory`, at most 200, with the date of the
  conversation's last session as the reference date. The answer is the text
  after the last `ANSWER:`.
- **Judge:** Mem0's `JUDGE_SYSTEM_PROMPT` and unified `JUDGE_PROMPT`, which
  replies with JSON `{"reasoning", "label"}`; `CORRECT` scores 1 and anything
  else 0. For open-domain questions the gold answer is cut at its first `;`,
  as upstream's `preprocess_answer` does.
- **Report:** overall J and J per category name, with counts. Questions an
  arm could not answer, and unparseable verdicts, count as wrong (as upstream)
  and are also listed as `unscored`.

Source: `mem0ai/memory-benchmarks` at commit
`4b61c5d31b9c668a12b4f5e78064248a02c82d2b`, `benchmarks/locomo/prompts.py`
and `run.py`, Apache License 2.0.

## `longmemeval-official`

The official evaluation of
[LongMemEval](https://github.com/xiaowu0162/LongMemEval)
(`src/evaluation/evaluate_qa.py`), used by Supermemory, Zep, Hindsight and
others for LongMemEval-S numbers.

- **Questions:** every question. On LongMemEval-S that is 500.
- **Reader prompt:** the chain-of-thought reader prompt without fact
  expansion from `src/generation/run_generation.py`, filled with the arm's
  context, the question date and the question. The whole reply is the
  hypothesis, as upstream.
- **Judge:** the official per-type templates: the default template for
  `single-session-user`, `single-session-assistant` and `multi-session`; an
  off-by-one-day tolerance for `temporal-reasoning`; credit for answers that
  mention an old value beside the updated one for `knowledge-update`; a
  rubric for `single-session-preference`; and the abstention template for
  every question whose id contains `_abs`. A verdict is correct when the
  reply contains `yes` (case-insensitive), as upstream.
- **Report:** accuracy per question type, task-averaged accuracy (the mean
  of the six per-type accuracies), overall accuracy and abstention accuracy,
  as `print_qa_metrics.py` prints them.

Source: `xiaowu0162/LongMemEval` at commit
`9e0b455f4ef0e2ab8f2e582289761153549043fc`, MIT licence.

## Prompts and audit trail

The prompts are vendored byte-for-byte in
[`crates/keep-bench/protocols/`](../../crates/keep-bench/protocols/README.md),
with attribution in [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
They are compiled into the binary and checked against their recorded SHA-256
by a unit test. Each run under a parity protocol records, in `run.json` and
`summary.json`, the protocol, split, prompt versions, upstream repository,
commit, licence and files, and the SHA-256 of every prompt file. `summary.md`
starts its results with the same information. `keep-bench compare` refuses
to compare runs under different protocols.

The manifest's `prompt_version` values must name the protocol's prompts:

| Protocol | Answer `prompt_version` | Judge `prompt_version` |
|---|---|---|
| `mem0-locomo` | `mem0-locomo-answer-v1` | `mem0-locomo-judge-v1` |
| `longmemeval-official` | `longmemeval-official-reader-cot-v1` | `longmemeval-official-judge-v1` |

## Splits and the holdout

`split = "dev"` (the default) scores the development split only, like every
other development run. Its numbers use the published protocol but not the
published question set, so they are **not** directly comparable to published
figures; they are for tracking.

`split = "full"` scores the whole dataset, which is what published numbers
use. That includes the sealed holdout, so it **consumes the holdout**:

> Full-set parity runs need explicit maintainer approval before they are run.
> Once run, holdout questions have been seen and the holdout can no longer
> confirm a result unseen.

Safeguards: `split = "full"` is accepted only in the development phase and
only under a parity protocol; the holdout commitment must still verify; and
`keep-bench run`, `contexts` and `export` refuse a full-set manifest unless
`--consume-holdout` is passed. `--unseal` cannot be combined with it. Dry runs
need no flag because they call no model; their report says "full set,
holdout included".

## Manifests

In [`manifests/parity/`](../../manifests/parity/). Every manifest has the
arms `full-context`, `dense-6k`, `bm25-6k`, `keep-mk2` and `keep-mk3`, and
the same embedding, reranker, storage and retrieval
settings as the internal G0 v2 manifests.

| File | Protocol | Split | Models |
|---|---|---|---|
| `locomo-mem0-copilot.toml` | `mem0-locomo` | dev (971 questions) | Copilot CLI, non-canonical |
| `locomo-mem0-full-copilot.toml` | `mem0-locomo` | full (1,540) | Copilot CLI, non-canonical |
| `locomo-mem0-pinned.toml` | `mem0-locomo` | dev | pinned gpt-4o-mini answer and judge |
| `locomo-mem0-full-pinned.toml` | `mem0-locomo` | full | pinned gpt-4o-mini answer and judge |
| `longmemeval-s-official-copilot.toml` | `longmemeval-official` | dev (345) | Copilot CLI, non-canonical |
| `longmemeval-s-official-full-copilot.toml` | `longmemeval-official` | full (500) | Copilot CLI, non-canonical |
| `longmemeval-s-official-pinned.toml` | `longmemeval-official` | dev | pinned gpt-4o-2024-08-06 reader and judge |
| `longmemeval-s-official-full-pinned.toml` | `longmemeval-official` | full | pinned gpt-4o-2024-08-06 reader and judge |

`keep-mk3` is the fact arm of [methodology.md](methodology.md#arms) with
`temporal`, `preferences` and `assistant_route`. It answers with the
protocol's own prompt like every other arm, so the protocol comparison
stays fair; `answer-v3` is not allowed under a parity protocol. Its
`[extraction]` and `[models.extract]` blocks are the internal G0 v3 ones on
the Copilot manifests; the pinned manifests extract with
`gpt-4o-mini-2024-07-18` through the Azure OpenAI v1 route
(`KEEP_AZURE_EXTRACT_DEPLOYMENT`), also on LongMemEval, where the reader is
gpt-4o. Run `keep-bench extract` first, or the run extracts every
uncached session itself. Adding the arm changed every manifest hash, so
runs made before stay attributable.

### Models and how they differ from published setups

| Setup | Answer model | Judge model |
|---|---|---|
| Mem0 paper (LoCoMo) | gpt-4o-mini | gpt-4o-mini |
| Mem0 2026 numbers (LoCoMo, LongMemEval) | gpt-5 | gpt-5 |
| LongMemEval official judge | (system's own reader) | gpt-4o-2024-08-06, 10 output tokens |
| Keep Copilot manifests | gpt-5.6-luna, low reasoning effort | gpt-5.6-luna, low reasoning effort |
| Keep pinned LoCoMo manifests | gpt-4o-mini-2024-07-18 | gpt-4o-mini-2024-07-18 |
| Keep pinned LongMemEval manifests | gpt-4o-2024-08-06 | gpt-4o-2024-08-06, 10 output tokens |

The Copilot CLI offers none of gpt-4o, gpt-4o-mini, gpt-5 or gpt-4.1, so the
Copilot manifests use gpt-5.6-luna, the closest GPT-5-family model, for both
roles. On top of the model mismatch, the CLI adds a hidden system prompt,
does not pin the model and ignores temperature and output-token limits (see
[Development runs through the Copilot CLI](methodology.md#development-runs-through-the-copilot-cli)).
Mem0's harness asks its judge for structured JSON output; the CLI has no
JSON mode, so the verdict is parsed from the reply text. Every Copilot
artifact is stamped NON-CANONICAL. Use the pinned manifests (after approval
for the full set) for numbers to publish.

## Context as memories

Keep's arms build a context bundle, not a list of memories. For Mem0's
prompt the bundle is split into memories (segmentation `memories-v1`):

- each `[date] text` line starts a memory with that date;
- a `[Session id | date]` header starts one memory holding the session;
- other lines continue the current memory, and text before the first header
  is one undated memory (shown as `(unknown date)`);
- if there are more than 200 memories, neighbours with the same date are
  merged, then neighbouring pairs, so the limit never drops text.

Mem0's harness passes its top 200 search results; Keep passes everything
the arm built within its token budget. The ordering differs too: Mem0 sorts
by creation time, which is what the date column above provides.

## Export for the unmodified external harnesses

To have the upstream harnesses (with their own OpenAI calls) answer and
judge Keep's retrieval, export it. Export builds contexts with the
manifest's local embedder, reranker and storage and calls no answer or judge
model:

```sh
cargo build --release -p keep-bench --features local-embeddings
./target/release/keep-bench export manifests/parity/locomo-mem0-copilot.toml \
  --arms dense-6k,keep-mk2 --out target/export/locomo --project keep
```

Options: `--questions FILE` restricts to listed development question ids,
`--single-memory` passes each context as one memory instead of segmenting
it, and `--consume-holdout` is required for a full-set manifest. The output
directory must not exist or be empty.

Layout:

```text
export.json                       manifest hash, protocol, prompt hashes,
                                  segmentation, gaps and the commands below
id_map.json                       Keep question id -> harness question id
dataset.json                      reference data for the harness
<arm>/predicted_<project>/*.json  one retrieval file per question
```

Each retrieval file has the fields Mem0's harness writes after its search
step (`question_id`, `question`, `category`, `ground_truth_answer`,
`reference_date`, `retrieval.search_results` with `memory`, `score`, `id`
and `created_at`, and so on). For LoCoMo, `dataset.json` holds only the
exported conversations, in their original order, with `qa` narrowed to the
exported questions, so the harness's `conv{i}_q{j}` ids line up;
`id_map.json` maps them back. Questions an arm could not build a context for
are listed under `missing` in `export.json` and get no file, so the harness
refuses to score an incomplete export rather than scoring a gap as wrong.

Then, in a checkout of `mem0ai/memory-benchmarks` at the commit above with
`pip install -r requirements.txt` and `OPENAI_API_KEY` set, run the command
`export.json` lists for each arm. For LoCoMo:

```sh
python -m benchmarks.locomo.run --project-name keep --evaluate-only \
  --dataset-path /abs/target/export/locomo/dataset.json \
  --output-dir /abs/target/export/locomo/keep-mk2 \
  --conversations 0,1,2,3,4,5 --categories 1,2,3,4 --top-k-cutoffs 200 \
  --answerer-model gpt-4o-mini --judge-model gpt-4o-mini
```

For LongMemEval (scored with Mem0's own judge, not the official one):

```sh
python -m benchmarks.longmemeval.run --project-name keep --evaluate-only \
  --all-questions --dataset-path /abs/target/export/lme/dataset.json \
  --output-dir /abs/target/export/lme/keep-mk2 --top-k-cutoffs 200
```

Set `--answerer-model` and `--judge-model` to `gpt-5` to match Mem0's 2026
numbers.

### LongMemEval hypothesis file

For the official LongMemEval script, which judges answers rather than
retrieval, export a finished run's answers for one arm:

```sh
./target/release/keep-bench hypotheses results/<run-dir> --arm keep-mk2 \
  --out target/export/keep-mk2.jsonl
```

Each line is `{"question_id": ..., "hypothesis": ...}`, the reader's whole
reply. Questions without an answer get an empty hypothesis (scored wrong);
the command prints how many. Then, in a checkout of `xiaowu0162/LongMemEval`
at the commit above with `OPENAI_API_KEY` set:

```sh
cd src/evaluation
python3 evaluate_qa.py gpt-4o /abs/target/export/keep-mk2.jsonl \
  /abs/data/longmemeval/longmemeval_s_cleaned.json
python3 print_qa_metrics.py /abs/target/export/keep-mk2.jsonl.eval-results-gpt-4o \
  /abs/data/longmemeval/longmemeval_s_cleaned.json
```

`print_qa_metrics.py` expects all six question types to be present; on a
small subset it prints `nan` for missing types.

## Cost estimates

Dry-run worst cases (every reply at `max_output_tokens`, plus the CLI's
hidden overhead), one seed, all five arms, including fact extraction for
every session not yet in the extraction cache (the Copilot figures assume
the development cache built for this work: 61 LoCoMo and 2,840
LongMemEval-S sessions):

| Manifest | Questions | Extraction calls | Extraction worst case | Total worst-case AI credits |
|---|---:|---:|---:|---:|
| `locomo-mem0-copilot.toml` | 971 | 101 | 74 | 5,091 |
| `locomo-mem0-full-copilot.toml` | 1,540 | 211 | 155 | 8,138 |
| `longmemeval-s-official-copilot.toml` | 345 | 13,583 | 10,028 | 11,950 |
| `longmemeval-s-official-full-copilot.toml` | 500 | 20,761 | 15,330 | 18,100 |

| Manifest | Questions | Extraction calls | Extraction worst case | Total worst-case USD |
|---|---:|---:|---:|---:|
| `locomo-mem0-pinned.toml` | 971 | 162 | $0.33 | $17.08 |
| `locomo-mem0-full-pinned.toml` | 1,540 | 272 | $0.55 | $27.25 |
| `longmemeval-s-official-pinned.toml` | 345 | 16,423 | $33.77 | $164.26 |
| `longmemeval-s-official-full-pinned.toml` | 500 | 23,601 | $48.54 | $236.31 |

Each manifest's cap is about 1.5 times its worst case. On LongMemEval-S,
whose 345 histories have about 48 sessions each, extraction dominates the
Copilot worst case; at the measured mean of about 0.15 credits per
session it would cost about 2,100 credits on the development split, not
10,000. Among the answer arms full context still dominates.

The pinned LoCoMo manifests use gpt-4o-mini for both roles; the pinned
LongMemEval manifests use gpt-4o for both, which is why they cost more
despite having fewer questions.

## Smoke runs

Before any parity run, the protocols were checked end to end on small
development-split subsets with the Copilot manifests. These numbers show
that the pipeline works. They are far too small to compare with anything.

LoCoMo under `mem0-locomo`: 22 questions (categories 1 to 4) from two
development conversations, 176 calls, 35.6 AI credits, no failed calls and
no malformed verdicts.

| Arm | J | multi-hop | open-domain | single-hop | temporal |
|---|---|---|---|---|---|
| full-context | 95.5% | 100% | 83.3% | 100% | 100% |
| dense-6k | 90.9% | 100% | 83.3% | 75.0% | 100% |
| bm25-6k | 81.8% | 100% | 66.7% | 75.0% | 83.3% |
| keep-mk2 | 86.4% | 100% | 66.7% | 75.0% | 100% |

LongMemEval-S under `longmemeval-official`: 24 development questions, four
of each question type, including two abstention questions. 192 calls, 81.6
AI credits, no failed calls and no malformed verdicts.

| Arm | Overall | Task-averaged | Abstention |
|---|---|---|---|
| full-context | 75.0% | 75.0% | 100% (2) |
| dense-6k | 75.0% | 75.0% | 100% (2) |
| bm25-6k | 66.7% | 66.7% | 100% (2) |
| keep-mk2 | 87.5% | 87.5% | 100% (2) |
