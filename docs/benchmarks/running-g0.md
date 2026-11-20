# Running G0

G0 is the first canonical run: every baseline and Keep arm on the
development split of LoCoMo and LongMemEval-S, with pinned OpenAI models
served by Azure OpenAI. It produces the reference numbers that later gates
compare against. The sealed holdout is not loaded; it is kept for the
confirmation phase.

| | |
| --- | --- |
| Manifests | `manifests/g0/locomo.toml`, `manifests/g0/longmemeval-s.toml` |
| Answer model | `gpt-4o-mini`, version `2024-07-18` |
| Judge model | `gpt-4o`, version `2024-08-06` |
| Embeddings | local ONNX `BAAI/bge-small-en-v1.5` (no paid API) |
| Storage | `postgres-qdrant` |
| Questions | all development-split questions: 1,242 LoCoMo (6 conversations), 345 LongMemEval-S |
| Seeds | 1 |
| Run caps | $45 (LoCoMo), $20 (LongMemEval-S) |
| Worst-case estimate | $38.82 + $15.45 = $54.27 |

## Steps

1. **Create two Azure deployments**, both with the **Global Standard**
   deployment type (Regional Standard is retired for these model versions):
   - `gpt-4o-mini`, model version `2024-07-18`
   - `gpt-4o`, model version `2024-08-06`

   Set the version upgrade policy to no automatic upgrade where Azure
   offers it. Either way, a reply from another version aborts the run. Give each enough tokens-per-minute quota for four
   concurrent requests of up to about 110K tokens (LongMemEval-S full
   context).

2. **Create `~/.keep/env`**, readable only by you:

   ```sh
   mkdir -p ~/.keep && touch ~/.keep/env && chmod 600 ~/.keep/env
   ```

   It must set these variables (shell `NAME=value` lines):

   | Variable | Value |
   | --- | --- |
   | `AZURE_FOUNDRY_ENDPOINT` | The resource endpoint, for example `https://<resource>.openai.azure.com/`. Only its scheme and host are used. |
   | `AZURE_FOUNDRY_API_KEY` | A key for that resource. |
   | `KEEP_AZURE_ANSWER_DEPLOYMENT` | The name of the gpt-4o-mini deployment. |
   | `KEEP_AZURE_JUDGE_DEPLOYMENT` | The name of the gpt-4o deployment. |
   | `KEEP_POSTGRES_URL` | Postgres URL for the benchmark stack. |
   | `KEEP_QDRANT_URL` | Qdrant URL for the benchmark stack. |

   Optionally set `KEEP_MODEL_CACHE` to an existing local model cache.

3. **Fetch the datasets** if `data/` is empty: `scripts/fetch-datasets.sh`.
   Start the storage stack with
   `docker compose --profile postgres-qdrant up -d` if it is not running.

4. **Run** from the repository root:

   ```sh
   scripts/run-g0.sh
   ```

   The script refuses an env file readable by group or others, checks
   that every variable is set without printing any value, builds the
   harness, validates both manifests, prints both dry-run estimates and the
   spend ledger, and asks for confirmation before any model call.
   `KEEP_SPEND_CAP_USD` (default 300) and `KEEP_SPEND_LEDGER` (default
   `results/spend-ledger.jsonl`) override the cross-run cap and ledger.

Results land in `results/<run id>/`; `summary.md` shows accuracy per arm,
the spend charged and, in `run.json`, the endpoint each model was served
from.

## Safety checks

- **Dispatch.** Nothing is sent unless the manifest sets
  `dispatch.allow_model_calls = true` and the command passes
  `--allow-model-calls`.
- **Version pin.** Every reply's `model` field must equal the manifest's
  `served_model` (`gpt-4o-mini-2024-07-18`, `gpt-4o-2024-08-06`). On the
  first mismatch the run stops sending, marks the remaining calls
  `not run`, writes its artifacts with `aborted` set, and exits with an
  error. `compare` refuses an aborted run. A deployment that serves a
  different version therefore cannot produce canonical numbers.
- **What is recorded.** `run.json` records, for each model, the API style,
  auth header kind, endpoint host, deployment name, the served models and
  the `system_fingerprint` values seen. Each record line keeps the served
  model and fingerprint of its call. The key is never logged, printed or
  written; request URLs are dropped from transport errors.
- **Per-run cap.** Before each call the harness reserves its worst case:
  input tokens plus a 5% (and 16-token) margin, and `max_output_tokens` of
  output, at the pinned prices. A call whose reservation would cross a cap
  is refused. Each reply then settles the reservation to the usage it
  reports.
- **Cross-run cap.** `results/spend-ledger.jsonl` (ignored by git) is an
  append-only record of every USD-metered run and judge calibration. A run
  first reserves its whole `max_cost_usd`; it is refused if settled spend,
  open reservations and the new reservation together would exceed the cap
  ($300 by default). When the run ends, whether it finished, failed or was
  aborted, the reservation is settled to the cost charged. `keep-bench
  spend show` prints the totals. A run that was killed leaves its
  reservation open, which keeps counting against the cap; close it with
  `keep-bench spend settle <id> --usd <billed amount>` using the amount
  Azure billed.

## Prices

Pinned per million tokens in both manifests, from the Azure Retail Prices
API (`https://prices.azure.com/api/retail/prices`, service "Foundry Models",
Global Standard meters), checked on 2026-11-13. They equal OpenAI's list
prices for the same models.

| Model | Input | Output | Meters |
| --- | --- | --- | --- |
| gpt-4o-mini 2024-07-18 | $0.15 | $0.60 | `gpt-4o-mini-0718 Inp-glbl`, `Outp-glbl` |
| gpt-4o 2024-08-06 | $2.50 | $10.00 | `gpt-4o-0806 Inp-glbl`, `Outp-glbl` |

Other deployment types can cost more; if you use one, raise the prices in
the manifests to its meters so the caps stay conservative. These model versions retire
on Azure on 2027-04-14.

## Endpoint

The manifests use the Azure OpenAI v1 API (`api_style = "azure-v1"`):
`POST {endpoint}/openai/v1/chat/completions` with the deployment name as
`model`, no `api-version` parameter, and the key in the `api-key` header.
See [Azure OpenAI API version lifecycle](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle).
The older deployment-scoped form
(`{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...`)
is available as `api_style = "azure-deployments"` with `api_version` set.
`auth_header = "bearer"` sends `Authorization: Bearer` instead.

## Sample size

The methodology does not fix a size for G0. Both manifests use every
development-split question with one seed:

- **LoCoMo** has only 6 development conversations, and the bootstrap
  resamples whole conversations, so taking fewer questions would widen the
  intervals without changing the number of clusters. All 1,242 questions
  are used.
- **LongMemEval-S** has one history per question, so precision grows with
  the question count. All 345 development questions are used.
- **One seed.** Answers use temperature 0. Repeated seeds mainly measure
  sampling noise, which the confirmation phase (three or more seeds, on the
  holdout) covers.

The worst-case estimate for both manifests is $54.27, and the run caps
reserve $65, leaving at least $235 of the $300 budget for G1 and G2. Real
spend is lower: answers and verdicts are rarely as long as
`max_output_tokens`.

## Dry-run estimate

From `keep-bench dry-run`, one seed, worst case (every answer at 256 output
tokens, every verdict at 16, input margin included). Answer and judge calls
are equal per arm. Mean context tokens show the equal-context comparison:
`bm25-6k` and `dense-6k` get the same 6,096-token budget as `keep-mk0` and
`keep-mk1`.

### LoCoMo (`g0-locomo`, 1,242 questions)

Manifest hash `sha256:8cb62c3df2b0117624d6e46b005ae3e39571abb6f83d7dcdca2ce6170415468d`.

| Arm | Answer calls | Judge calls | Answer input tokens | Judge input tokens | Mean context tokens | Input-only cost USD | Worst-case USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-memory | 1,242 | 1,242 | 109,411 | 624,904 | 0 | 1.58 | 1.97 |
| full-context | 1,242 | 1,242 | 26,506,294 | 624,904 | 20,246 | 5.54 | 5.93 |
| recent-window | 1,242 | 1,242 | 7,851,947 | 624,904 | 5,941 | 2.74 | 3.13 |
| bm25 | 1,242 | 1,242 | 1,406,124 | 624,904 | 998 | 1.77 | 2.16 |
| dense | 1,242 | 1,242 | 1,282,075 | 624,904 | 903 | 1.75 | 2.14 |
| bm25-6k | 1,242 | 1,242 | 7,875,465 | 624,904 | 5,959 | 2.74 | 3.13 |
| dense-6k | 1,242 | 1,242 | 7,870,739 | 624,904 | 5,956 | 2.74 | 3.13 |
| notes-grep | 1,242 | 1,242 | 2,643,667 | 624,904 | 1,947 | 1.96 | 2.35 |
| oracle-evidence | 1,240 | 1,240 | 219,570 | 623,891 | 89 | 1.59 | 1.98 |
| keep-m | 1,242 | 1,242 | 5,426,753 | 624,904 | 4,081 | 2.38 | 2.77 |
| keep-k0 | 1,242 | 1,242 | 1,181,509 | 624,904 | 826 | 1.74 | 2.13 |
| keep-k1 | 1,242 | 1,242 | 1,277,930 | 624,904 | 900 | 1.75 | 2.14 |
| keep-mk0 | 1,242 | 1,242 | 6,504,241 | 624,904 | 4,908 | 2.54 | 2.93 |
| keep-mk1 | 1,242 | 1,242 | 6,505,558 | 624,904 | 4,909 | 2.54 | 2.93 |
| **Total** | | 34,772 calls | 76,661,283 | 8,747,643 | | 33.37 | 38.82 |

### LongMemEval-S (`g0-longmemeval-s`, 345 questions)

Manifest hash `sha256:1b3b652ed7796d7a6270467d2b0d4746867dd0f7e70331c45ad6f1d4b6bb0349`.

| Arm | Answer calls | Judge calls | Answer input tokens | Judge input tokens | Mean context tokens | Input-only cost USD | Worst-case USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-memory | 345 | 345 | 37,166 | 175,088 | 0 | 0.44 | 0.55 |
| full-context | 345 | 345 | 35,784,413 | 175,088 | 98,685 | 5.81 | 5.91 |
| recent-window | 345 | 345 | 2,153,311 | 175,088 | 5,846 | 0.76 | 0.87 |
| bm25 | 345 | 345 | 744,419 | 175,088 | 1,956 | 0.55 | 0.66 |
| dense | 345 | 345 | 724,476 | 175,088 | 1,901 | 0.55 | 0.65 |
| bm25-6k | 345 | 345 | 2,230,345 | 175,088 | 6,058 | 0.77 | 0.88 |
| dense-6k | 345 | 345 | 2,229,234 | 175,088 | 6,055 | 0.77 | 0.88 |
| notes-grep | 345 | 345 | 748,191 | 175,088 | 1,967 | 0.55 | 0.66 |
| oracle-evidence | 331 | 331 | 180,452 | 168,022 | 421 | 0.45 | 0.55 |
| keep-m | 345 | 345 | 1,499,545 | 175,088 | 4,040 | 0.66 | 0.77 |
| keep-k0 | 345 | 345 | 724,385 | 175,088 | 1,901 | 0.55 | 0.65 |
| keep-k1 | 345 | 345 | 751,234 | 175,088 | 1,975 | 0.55 | 0.66 |
| keep-mk0 | 345 | 345 | 2,188,230 | 175,088 | 5,942 | 0.77 | 0.87 |
| keep-mk1 | 345 | 345 | 2,189,095 | 175,088 | 5,944 | 0.77 | 0.87 |
| **Total** | | 9,632 calls | 52,184,496 | 2,444,166 | | 13.94 | 15.45 |

**Both manifests:** 44,404 calls, 140,037,588 input tokens; $47.31 at input prices only, $54.27 worst case, against $65 of run caps.

Input-only cost counts input tokens at the pinned prices, without output or reservation margin; real spend should be close to it plus a few dollars of output.

## Internal G0 on Copilot CLI

The go/no-go decision (does Keep beat plain retrieval baselines?) can run
through a locally installed, signed-in GitHub Copilot CLI instead of Azure.
It needs no API key and is billed in Copilot AI credits.

| | |
| --- | --- |
| Manifests | `manifests/g0/locomo-copilot.toml` (`g0-copilot-locomo`), `manifests/g0/longmemeval-s-copilot.toml` (`g0-copilot-longmemeval-s`) |
| Answer model | `gpt-5.6-luna`, reasoning effort low |
| Judge model | `claude-haiku-4.5` |
| Everything else | same datasets, development split, arms, embedding, retrieval, storage and seed as the pinned G0 |
| Context limit | 125,000 tokens (see below) |
| Concurrency | 16 CLI processes |
| AI-credit caps | 16,000 (LoCoMo), 5,750 (LongMemEval-S) |
| Worst-case estimate | 10,569 + 3,815 = 14,384 credits |

**Why it is non-canonical.** The CLI wraps every prompt in its own hidden
system prompt (about 2,000 to 3,000 tokens) and does not pin the upstream
model, so these numbers are not comparable to the pinned G0 or to any
published result. Both manifests are `track = "development"` and every
artifact carries the non-canonical banner.

**Comparing the runs.** `keep-bench compare` refuses non-canonical runs
unless `--allow-non-canonical` is passed. With it, all runs must be
non-canonical and share the same answer and judge models, every output is
stamped `INTERNAL / NON-CANONICAL` with the runs' labels and models, and
the gate always fails its `canonical runs` check. `--out DIR` also writes
`comparison.md` and `comparison.json`.

```sh
keep-bench compare --allow-non-canonical --run results/<run id> \
  --candidate keep-mk1 --against bm25-6k --against dense-6k \
  --out results/internal-g0-copilot-locomo
```

**Running it.** Start the storage stack, set the storage URLs, and run the
script from the repository root. With the local compose stack the URLs are:

```sh
docker compose --profile postgres-qdrant up -d
export KEEP_POSTGRES_URL='postgres://keep:keep-local-only@127.0.0.1:15433/keep'
export KEEP_QDRANT_URL='http://127.0.0.1:16333'
scripts/run-g0-copilot.sh
```

The script checks that `copilot` (or `$KEEP_COPILOT_BIN`) is on `PATH` and
that both URLs are set, without printing them. It sets `KEEP_MODEL_CACHE`
to `.keep-models/` and `KEEP_COPILOT_WORKDIR` to `target/copilot-work/`
unless they are already set, builds the harness with local embeddings,
prints both dry-run estimates and asks for confirmation. It then runs both
manifests in turn and writes one internal comparison per dataset to
`results/internal-g0-copilot-<dataset>-v2-<time>/`, comparing `keep-mk2`
(or `$KEEP_G0_CANDIDATE`) against full context, recent window, BM25, dense
(both budgets), notes grep and `keep-mk1`. See
[Version 2](#version-2-ranked-keep-arms) for running v1 instead.

The harness cannot resume a run; running a manifest again starts it over.
If one manifest finished and the other did not, set `KEEP_G0_LOCOMO_RUN`
or `KEEP_G0_LONGMEMEVAL_RUN` to the finished run's results directory and
the script reuses it instead of running it again.

**Context limit.** The CLI does not report a model's context window
without a model call, so the manifests assume 128,000 tokens and subtract
the hidden harness prompt (about 3,000 tokens for the answer model, 2,300
for the judge), giving 125,000 and 125,700. The longest LongMemEval-S
history is about 106,600 tokens, so full context fits every question; the
pinned fit masks (`*-copilot.fit-mask.json`) record a 100% fit for both
datasets. If the CLI's window for this model is smaller, the longest
full-context calls fail and are recorded as failures, not hidden; many
failures in a row stop the run.

**Concurrency.** `dispatch.concurrency = 16` bounds the questions in
flight and a single 16-slot gate shared by the answer and judge models, so
at most 16 CLI processes run at once. The same value sets the number of
ingest workers. Each call gets its own fresh, empty directory under
`KEEP_COPILOT_WORKDIR`, so concurrent sessions never share files. Each
process is started with the CLI's minimum `--max-ai-credits` of 30, which
is above any single call's charge.

**Credits.** From `keep-bench dry-run`, worst case (every answer at 256
output tokens, reservation rates above the observed ones, hidden overhead
included). Answer and judge calls are equal per arm. Manifest hashes:
`sha256:23c0ec5456a1e78e2333c15d964365bad1138ce332ac4116a61e180907b5e966`
(LoCoMo) and
`sha256:74d13de480d29a879be5e48e1e3144724e5919cc75d5cf0479ef3f8d003006e7`
(LongMemEval-S). Keep arm estimates vary by a few hundredths of a credit
between dry runs.

| Arm | LoCoMo calls | LoCoMo credits | LongMemEval-S calls | LongMemEval-S credits |
|---|---:|---:|---:|---:|
| no-memory | 1,242 | 601.68 | 345 | 167.51 |
| full-context | 1,242 | 1,355.89 | 345 | 1,188.86 |
| recent-window | 1,242 | 822.90 | 345 | 227.97 |
| bm25 | 1,242 | 638.74 | 345 | 187.72 |
| dense | 1,242 | 635.19 | 345 | 187.15 |
| bm25-6k | 1,242 | 823.58 | 345 | 230.17 |
| dense-6k | 1,242 | 823.44 | 345 | 230.14 |
| notes-grep | 1,242 | 674.10 | 345 | 187.83 |
| oracle-evidence | 1,240 | 603.87 | 331 | 164.86 |
| keep-m | 1,242 | 753.61 | 345 | 209.29 |
| keep-k0 | 1,242 | 632.32 | 345 | 187.15 |
| keep-k1 | 1,242 | 635.10 | 345 | 187.92 |
| keep-mk0 | 1,242 | 784.40 | 345 | 228.97 |
| keep-mk1 | 1,242 | 784.44 | 345 | 228.99 |
| **Total** | 17,386 | **10,569.3** | 4,816 | **3,814.5** |

Each cap is about 1.5 times its worst case. Real spend should be lower: in
the development runs, answer calls settled at about two thirds of the
reservation rate.

**Duration.** In the development runs, answer calls took about 5 to 6
seconds at the median and about 11 seconds with 30K-token full context.
Both manifests make about 44,400 answer and judge calls; at 16 at a time
that is roughly 4 to 5 hours of calls, plus ingest and local embedding of
every history (LongMemEval-S embeds about 38 million tokens). Expect most
of a working day, longer if the CLI is rate limited.

### Version 2: ranked Keep arms

`manifests/g0/locomo-copilot-v2.toml` (`g0-copilot-locomo-v2`) and
`manifests/g0/longmemeval-s-copilot-v2.toml` (`g0-copilot-longmemeval-s-v2`)
are new manifest versions. They keep every v1 model, setting, fit mask and
arm, so the old arms can be compared pair by pair, and add:

- `keep-k2` and `keep-mk2`, the ranked Keep arms (see
  [methodology.md](methodology.md#arms)). `keep-mk2` fills the same
  6,096-token budget as `bm25-6k` and `dense-6k`.
- A `[reranker]`: the local cross-encoder
  `cross-encoder/ms-marco-MiniLM-L6-v2` at revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`, with its file digests pinned,
  reordering the top 50 fused candidates (inputs up to 512 tokens). It is
  downloaded into `KEEP_MODEL_CACHE` on first use, like the embedding model.
  It was chosen over `BAAI/bge-reranker-base`, which scored about 68 pairs a
  second on a 10-core laptop CPU against MiniLM's 420 to 560, and would
  have added hours to a run.

The v1 manifests and their hashes are unchanged.
`scripts/run-g0-copilot.sh` runs v2 by default, with `keep-mk2` as the
candidate and `keep-mk1` added to the comparison; `KEEP_G0_VERSION=1` runs
the v1 manifests with `keep-mk1` as the candidate.

Manifest hashes:
`sha256:89039e54e49f64132fde59548faec77c7726d1198188ffaeb763d43fafbde015`
(LoCoMo) and
`sha256:0d767e9028b16d2fdfc8f8f0f229857d0e6619c85cc99e4541d17fcf69e20155`
(LongMemEval-S). The dry run builds the ranked arms without the reranker,
which changes which turns fill the budget but not its size.

| Arm | LoCoMo calls | LoCoMo credits | LongMemEval-S calls | LongMemEval-S credits |
|---|---:|---:|---:|---:|
| v1 arms (as above) | 17,386 | 10,569.3 | 4,816 | 3,814.5 |
| keep-k2 | 1,242 | 674.10 | 345 | 187.90 |
| keep-mk2 | 1,242 | 823.24 | 345 | 229.72 |
| **Total** | 19,870 | **12,066.6** | 5,506 | **4,232.2** |

The caps are 18,100 (LoCoMo) and 6,350 (LongMemEval-S), again about 1.5
times the worst case. Building the ranked contexts adds about 4 minutes on
LoCoMo and 3 minutes on LongMemEval-S on a 10-core laptop, mostly
reranking; the extra answer and judge calls add about one seventh to the
call time.

Before spending credits, `keep-bench contexts` (see
[methodology.md](methodology.md#evidence-recall-without-model-calls))
reports every arm's evidence recall and context size from the same
manifests without any model call.

### Version 3: fact arms

`manifests/g0/locomo-copilot-v3.toml` (`g0-copilot-locomo-v3`) and
`manifests/g0/longmemeval-s-copilot-v3.toml` (`g0-copilot-longmemeval-s-v3`)
keep every v2 model, setting, fit mask and arm, byte for byte, and add:

- `keep-k3` and `keep-mk3`, the fact arms (see
  [methodology.md](methodology.md#arms) and
  [knowledge.md](../knowledge.md#session-fact-extraction)), both with
  `temporal` and `assistant_route`; `keep-mk3` also has `preferences` and
  answers with `answer-v3`. Every other arm keeps `answer-v1`.
- `[extraction]` (prompt `session-facts-v2`, a 1,536-token facts block, a
  512-token profile) and `[models.extract]`: the Copilot CLI model, with
  `expected_output_tokens = 520`, the mean of the first 2,901 cached
  development replies.

`scripts/run-g0-copilot.sh` ran v3 by default until version 4, with `keep-mk3` as the
candidate and `keep-mk2` and `keep-mk1` added to the comparison;
`KEEP_G0_VERSION=2` and `1` rerun the earlier versions. To extract ahead of
the run, or to retry sessions whose reply was malformed (those are not
cached), run

```sh
keep-bench extract manifests/g0/longmemeval-s-copilot-v3.toml --allow-model-calls
```

which reads every development history (or `--questions ids.txt`) and
writes one cache file per session under `.keep-cache/extractions/`
(`KEEP_EXTRACT_CACHE` overrides it), keyed by
the model, the prompt version and a hash of the session text. A run reads
the cache and extracts only what is missing, within the same
`max_ai_credits`.

Manifest hashes:
`sha256:5ebdd1ece05d3dcb4050cebed833fa2708f3d216b327abe3d5f2a9cb7701276d`
(LoCoMo) and
`sha256:59442e3c2b2c7ce8a883259a484c630c6f4d02cd5d6eb90dd4e5558608a4dff2`
(LongMemEval-S).

| | LoCoMo calls | LoCoMo credits | LongMemEval-S calls | LongMemEval-S credits |
|---|---:|---:|---:|---:|
| v2 arms (as above) | 19,870 | 12,066.6 | 5,506 | 4,232.2 |
| keep-k3 | 1,242 | 682.54 | 345 | 190.13 |
| keep-mk3 | 1,242 | 831.81 | 345 | 231.97 |
| Fact extraction (uncached sessions) | 101 | 74.2 | 13,583 | 10,027.6 |
| **Total** | 22,455 | **13,655.2** | 19,779 | **14,681.9** |

Extraction is priced at its worst case, every reply at 3,000 output
tokens; the dry run's expected figure, at 520, is 24.1 (LoCoMo) and 3,290.4
(LongMemEval-S) credits. The development sessions already extracted cost
0.15 credits each on average, which puts the remaining LongMemEval-S
extraction nearer 2,100 credits. LongMemEval-S has about 48 sessions per
history, most of them distractors, and every one is read once. The caps
are 20,500 (LoCoMo) and 22,000 (LongMemEval-S), about 1.5 times the worst
case. The counts above assume the extraction cache built during
development (61 LoCoMo and 2,840 LongMemEval-S sessions); without it, add
those sessions back.

#### Development results

Measured with the Copilot CLI (unpinned model, non-canonical) on the
fixed 60-question LongMemEval-S development subset (ten questions per
category) and on the 26 development questions of two LoCoMo conversations
(`conv-47`, `conv-48`). Extracting the 2,901 sessions those histories
hold cost 445.4 credits.

Evidence recall and mean context tokens from `keep-bench contexts` (no
answer calls):

| Arm | LoCoMo recall | LongMemEval-S recall | LongMemEval-S tokens |
|---|---:|---:|---:|
| dense-6k | 82.7% | 89.2% | 6,050 |
| keep-mk2 | 91.0% | 91.4% | 6,011 |
| keep-k3 | 86.9% | 91.4% | 1,982 |
| keep-mk3 without `assistant_route` | 90.8% | 86.1% | 6,026 |
| keep-mk3 with `assistant_route` | | 91.4% | |

Without routing, mk3 lost recall only on single-session-assistant
questions: mk2 reaches those assistant turns at ranks beyond the
reranker's top 50, by budget alone, and the profile and facts blocks
pushed them out. `assistant_route` was added after this measurement.

Answer accuracy, judged, in two runs: run 1
(`manifests/dev/longmemeval-s-facts-dev.toml`, 437.9 credits) adds the
`mk3` options one at a time; run 2
(`manifests/dev/longmemeval-s-facts-dev-v2.toml`, 149.5 credits) repeats
`mk2` and `mk3` and adds `mk2` with `answer-v3` and `mk3` with routing.
Each cell is the percentage correct of 10 questions.

| Category | full-context | dense-6k | dense-6k + v3 | mk2 (1) | mk2 (2) | mk2 + v3 | mk3 facts | + temporal | + preferences | + v3 prompt (1) | + v3 prompt (2) | + routing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| knowledge-update | 90 | 90 | 90 | 80 | 70 | 80 | 90 | 90 | 90 | 90 | 90 | 90 |
| multi-session | 50 | 30 | 30 | 70 | 50 | 70 | 60 | 70 | 60 | 70 | 80 | 70 |
| single-session-assistant | 100 | 100 | 100 | 90 | 90 | 90 | 70 | 70 | 60 | 50 | 50 | 90 |
| single-session-preference | 10 | 20 | 50 | 30 | 20 | 30 | 30 | 20 | 0 | 60 | 10 | 40 |
| single-session-user | 90 | 100 | 90 | 90 | 90 | 90 | 90 | 90 | 100 | 90 | 80 | 90 |
| temporal-reasoning | 30 | 30 | 30 | 30 | 30 | 50 | 30 | 50 | 50 | 60 | 40 | 60 |
| **All (n = 60)** | 61.7 | 61.7 | 65.0 | 65.0 | 58.3 | 68.3 | 61.7 | 65.0 | 60.0 | 70.0 | 58.3 | **73.3** |

`+ routing` is the configuration of `keep-mk3` in the v3 manifests. On the
same questions in run 2 it was right where `mk2` with `answer-v3` was
wrong 8 times and the reverse 5 times; against plain `mk2`, 11 and 2.
These numbers are small and noisy: the same `mk2` arm changed its verdict
on 6 of 60 questions between the two runs, and the same `mk3` arm on 9
(70.0% then 58.3%), so differences of under about 10 points on 60
questions, and any single category of 10, are not evidence. The
preference result in particular swung from 60% to 10% between identical
arms. The routing rule and the prompt were written while looking at these
development questions, so they may fit them better than the holdout.
Development runs spent 1,032.8 credits in total.

### Version 4: grounding and summaries

`manifests/g0/locomo-copilot-v4.toml` (`g0-copilot-locomo-v4`) and
`manifests/g0/longmemeval-s-copilot-v4.toml` (`g0-copilot-longmemeval-s-v4`)
keep every v3 model, setting, fit mask and arm, byte for byte, and add:

- `keep-k4` and `keep-mk4`: `keep-k3` and `keep-mk3` with `subjects`
  (subject grounding), `summaries` (per-session summaries) and
  `answer-v4` (see [methodology.md](methodology.md#arms) and
  [knowledge.md](../knowledge.md#who-a-question-is-about)).
- `dense-6k-v4` and `full-context-v4`: the unchanged `dense-6k` and
  `full-context` retrieval answered with `answer-v4`, so the gain of the
  prompt can be told apart from the gain of the memory.
- `summary_prompt_version = "session-summary-v1"` and
  `summary_tokens = 512` in `[extraction]`. Summaries are written by the
  extract model, once per session, and cached beside the facts;
  `keep-bench extract` fills both caches.

With `KEEP_G0_VERSION=4`, `scripts/run-g0-copilot.sh` runs v4, with `keep-mk4` as the
candidate and `dense-6k-v4`, `full-context-v4`, `keep-mk3` and `keep-mk2`
added to the comparison; `KEEP_G0_VERSION=3`, `2` and `1` rerun the earlier
versions.

Manifest hashes:
`sha256:32ba40f21ff938aabfb4eb440cf8801def94d3bdf804da4fb6dc2978ef6c2fc0`
(LoCoMo) and
`sha256:6cfe7dcb947ac13d5dceae45ec6e0d70aa0b4f2eff31139b9f7999f2c2577c48`
(LongMemEval-S).

| | LoCoMo calls | LoCoMo credits | LongMemEval-S calls | LongMemEval-S credits |
|---|---:|---:|---:|---:|
| keep-k4 | 1,242 | 693.27 | 345 | 192.85 |
| keep-mk4 | 1,242 | 843.02 | 345 | 234.69 |
| dense-6k-v4 | 1,242 | 841.29 | 345 | 235.10 |
| full-context-v4 | 1,242 | 1,373.73 | 345 | 1,193.82 |
| **Dry-run worst case, all arms and extraction** | | **17,334.2** | | **25,959.6** |

The LongMemEval-S total includes 26,649 extraction and summary calls for
sessions not yet cached (worst case 19,448.9 credits, expected 5,225.8);
the cache built during development holds 3,357 fact and 2,840 summary
sessions there and every LoCoMo session. The caps are 26,000 (LoCoMo) and
39,000 (LongMemEval-S), about 1.5 times the worst case.

#### Development results

Measured with the Copilot CLI (unpinned model, non-canonical) on 300 fixed
LoCoMo development questions (`manifests/dev/locomo-dev-v4.toml`,
`locomo-dev-v4.subset.txt`) and on the 60-question LongMemEval-S
development subset used for v3 (`manifests/dev/longmemeval-s-dev-v4.toml`).
Both manifests hold one arm per step from `keep-mk3` to `keep-mk4`. No
holdout question was read. Cells are percent correct.

| Arm | What changes |
|---|---|
| `keep-mk3-p4` | `keep-mk3` answered with `answer-v4` |
| `keep-mk3-p4s` | + `subjects` |
| `keep-mk3-p4ss` | + `summaries` (the `keep-mk4` configuration) |
| `keep-mk3-p4ssx` | + `expand` |
| `keep-k3-p4ssx` | the same on `k3` |

LoCoMo, 300 development questions. `full-context`, `dense-6k`, `keep-k3`
and `keep-mk3` are the G0 v3 run's answers to the same questions;
`keep-mk4` is a second run of the `keep-mk3-p4ss` configuration.

| Arm | adversarial (68) | multi-hop (29) | temporal (60) | open-domain (14) | single-hop (129) | All |
|---|---:|---:|---:|---:|---:|---:|
| full-context | 76.5 | 31.0 | 41.7 | 21.4 | 81.4 | 64.7 |
| dense-6k | 75.0 | 37.9 | 48.3 | 28.6 | 80.6 | 66.3 |
| keep-k3 | 76.5 | 41.4 | 61.7 | 21.4 | 77.5 | 68.0 |
| keep-mk3 | 69.1 | 41.4 | 60.0 | 35.7 | 83.7 | 69.3 |
| full-context-v4 | 63.2 | 44.8 | 58.3 | 50.0 | 89.9 | 71.3 |
| dense-6k-v4 | 52.9 | 44.8 | 70.0 | 50.0 | 80.6 | 67.3 |
| keep-mk3-p4 | 73.5 | 37.9 | 68.3 | 35.7 | 79.8 | 70.0 |
| keep-mk3-p4s | 82.4 | 34.5 | 66.7 | 28.6 | 83.7 | 72.7 |
| keep-mk3-p4ss | 83.8 | 34.5 | 70.0 | 35.7 | 85.3 | 74.7 |
| keep-mk3-p4ssx | 83.8 | 37.9 | 68.3 | 28.6 | 85.3 | 74.3 |
| **keep-mk4** (repeat of p4ss) | 85.3 | 44.8 | 70.0 | 42.9 | 86.0 | **76.7** |
| keep-k3-p4ssx | 91.2 | 37.9 | 70.0 | 42.9 | 79.8 | 74.7 |
| **keep-k4** | 88.2 | 41.4 | 73.3 | 42.9 | 79.1 | **74.7** |

LongMemEval-S, 60 development questions, 10 per category. `full-context`
is from the v3 development run on the same questions; every other arm
comes from the v4 runs. `subjects` does nothing here (the histories are
between a user and an assistant), so `keep-mk3-p4s` was not run.

| Arm | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| full-context | 90 | 50 | 100 | 10 | 90 | 30 | 61.7 |
| dense-6k | 100 | 20 | 100 | 10 | 100 | 30 | 60.0 |
| keep-mk3 | 70 | 80 | 90 | 10 | 90 | 60 | 66.7 |
| full-context-v4 | 90 | 80 | 100 | 30 | 90 | 50 | 73.3 |
| dense-6k-v4 | 100 | 60 | 100 | 40 | 90 | 30 | 70.0 |
| keep-mk3-p4 | 70 | 60 | 90 | 30 | 80 | 50 | 63.3 |
| keep-mk3-p4ss | 80 | 70 | 90 | 60 | 90 | 50 | 73.3 |
| keep-mk3-p4ssx | 80 | 70 | 90 | 10 | 90 | 70 | 68.3 |
| **keep-mk4** (repeat of p4ss) | 80 | 60 | 90 | 70 | 70 | 60 | **71.7** |
| keep-k3-p4ssx | 80 | 40 | 100 | 50 | 90 | 90 | 75.0 |
| **keep-k4** | 80 | 50 | 100 | 40 | 90 | 80 | **73.3** |

Questions each step got wrong that the previous got right, and the
reverse, on LoCoMo: `mk3` to `p4` 19 and 21; `p4` to `p4s` 10 and 18
(7 of the 18 adversarial); `p4s` to `p4ss` 9 and 15; `p4ss` to `p4ssx`
12 and 11; `mk3` to `mk4` 13 and 35; `k3` to `k4` 15 and 35;
`dense-6k-v4` to `mk4` 17 and 45; `full-context-v4` to `mk4` 23 and 39.
The two runs of the `p4ss` configuration disagree on 20 of 300 LoCoMo
questions (7 and 13) and 9 of 60 LongMemEval-S questions, so steps of
under about 3 points on LoCoMo and 10 points on LongMemEval-S, and any
single category, are within noise.

What the runs show:

- `answer-v4` alone helps on temporal questions (and, for the baselines,
  open-domain ones) but costs `dense-6k` and `full-context` 22 and 13
  points on adversarial questions: told to answer from partial evidence,
  they borrow facts about the other speaker. Subject grounding wins those
  back for Keep (`p4` to `p4s`, +9 adversarial), which the baselines
  cannot do without a fact layer.
- Session summaries help on both datasets (LoCoMo +2.0 points;
  LongMemEval-S +10 points from `p4` to `p4ss`, mostly preference
  questions).
- Entity expansion (`expand`) did not help: -0.3 points on LoCoMo
  and -5.0 on LongMemEval-S for `mk3`, and within noise for `k3`. It is
  left out of `keep-k4` and `keep-mk4` and kept only for these ablation
  arms.
- Multi-hop questions (29) stay within noise of `keep-mk3`.
- `keep-mk4` beats `keep-mk3` by 5.4 and 7.4 points on LoCoMo (two runs)
  and by 5.0 to 6.7 on LongMemEval-S, and the v4 control arms by 3 to 9
  points on LoCoMo; on LongMemEval-S it ties `full-context-v4`.

The `answer-v4` rules and the grounding were written after looking at
answers to LoCoMo development questions and the v3 LongMemEval-S
development subset, so they may fit those better than the full split.
Development runs for version 4 spent 2,814.8 AI credits: 13.9 for the
LoCoMo summaries, 645.5 for the LongMemEval-S facts and summaries
(2,712 fact and 2,840 summary calls), and 2,155.4 for five answer runs.

### Version 5: source evidence and anchored dates

The runner now defaults to `KEEP_G0_VERSION=5`. It uses
`manifests/g0/{locomo,longmemeval-s}-copilot-v5.toml`, compares `keep-mk5`
with the old arms and the `dense-6k-v5` / `full-context-v5` prompt controls,
and retains the v4 caps: 26,000 and 39,000 AI credits. These caps authorize
no automatic run. LongMemEval-S remains development-only; never pass
`--unseal` or `--consume-holdout` for this work.

The new arms add original source snippets, conservative source-time
annotations, individual-fact ranking and bounded query decomposition.
See [Knowledge](../knowledge.md#version-5-source-grounded-facts).
The answer model remains `gpt-5.6-luna`, and the judge remains
`claude-haiku-4.5`, through the same Copilot CLI provider.

`manifests/dev/locomo-dev-v5.toml` pins the same 300 development questions
as v4; `longmemeval-s-dev-v5.toml` pins the same 60-question development
subset. Both include:

- `keep-mk4`, `keep-mk5`, `keep-k5` and both v5 prompt controls;
- `keep-mk4-p5`, which changes only the answer prompt;
- `keep-mk5-no-provenance`, `-no-dates`, `-no-decompose` and
  `-no-selection`, which remove one v5 option at a time.

The LongMemEval-S dev manifest limits dispatch to four concurrent calls
and retains its 900-credit dev cap, to leave room for the LoCoMo run on
the same machine. The answer and judge settings and G0 caps are unchanged.

Select arms explicitly with `--arms`. Each run is a new billed run, not a
resume. Check `summary.json`'s `credit_usage` before the next run and count
unsettled reservations conservatively. The 3,000-credit development
allowance is separate from the G0 manifest caps.

Development comparisons are non-canonical and noisy: identical v4 runs
changed about 20 of 300 LoCoMo answers and 9 of 60 LongMemEval-S answers.
Small per-category changes are not evidence of improvement.

The initial 30-question pilot used `locomo-dev-v5-pilot.toml` (manifest id
`dev-locomo-v5`) and spent 193.6 credits. Six questions per category were
selected by a fixed hash of their ids, not by their scores. Mk4 was correct
on 18, mk5 on 19 and k5 on 21. These small differences are not evidence.
Before the larger run, the source lookup was indexed and an attribution
bug was corrected: a quoted turn addressing another person must not
reclassify the fact as about that person. The larger LoCoMo manifest is
therefore revision `dev-locomo-v5-r2`, not a silent rerun of the pilot.

#### What changed

| Change | Plain-language purpose | Main files |
|---|---|---|
| Source evidence | Show the original speaker, words, turn and session date beside each extracted fact; do not let quoted names change fact ownership. | `keep-bench/src/v5.rs`, `facts.rs`, `knowledge_arm.rs` |
| Date anchors | Keep both the relative phrase and its source date; resolve clear dates without using today's clock or inventing a day for an imprecise week. | `keep-knowledge/src/relative_time.rs` |
| Fact selection | Rank individual facts and remove repeated statements instead of filling the space with a group's newest entries. | `keep-bench/src/v5.rs`, `knowledge_arm.rs` |
| Question facets | Search the original question and up to two smaller parts, then merge and deduplicate results within the existing budget. | `keep-bench/src/v5.rs`, `knowledge_arm.rs` |
| Answer rules | Match the right person/event, complete lists, check arithmetic, allow explicitly requested inference, and still abstain on unsupported factual claims. | `keep-bench/src/prompt.rs` |
| Honest reporting | Add non-adversarial accuracy and a read-only review queue, including incomplete comparisons and inconsistent judge verdicts. | `keep-bench/src/report.rs`, `scripts/analyze-failures.py` |
| Cache reliability | Give concurrent cache writes distinct staging names, without changing cache keys or model prompts. | `keep-knowledge/src/session_extract.rs` |
| Reproducible experiments | Add v5 arms, prompt controls, independent switches, dev/G0 manifests and version-5 runner selection. | `keep-bench/src/ablation.rs`, `manifest.rs`, `manifests/`, `scripts/run-g0-copilot.sh` |

Crate paths above are under `crates/`. Source snippets are capped at 600
characters and share the existing facts budget; they can displace useful
facts. Question facets are deterministic searches, not an unrestricted
model-driven research loop. The old arms and prompt text remain unchanged.

#### LoCoMo development comparison

Run `20261116T070741.971Z-67f85490b508`, all 300 pinned dev questions,
five arms, 3,000 answer/judge calls, **1,002.962025 credits** and no
unsettled usage. Every scheduled answer completed. Values below are
percent correct; the column headings give the question counts.

| Arm | All (300) | Non-adversarial (232) | Multi-hop (29) | Temporal (60) | Open-domain (14) | Single-hop (129) | Adversarial (68) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mk4 | 76.7 | 73.7 | 41.4 | 73.3 | 35.7 | 85.3 | 86.8 |
| mk5 | 75.0 | 73.3 | 41.4 | 70.0 | 42.9 | 85.3 | 80.9 |
| k5 | 74.3 | 68.5 | 37.9 | 66.7 | 28.6 | 80.6 | 94.1 |
| dense-6k-v5 | 69.0 | 71.1 | 41.4 | 75.0 | 21.4 | 81.4 | 61.8 |
| full-context-v5 | 71.0 | 70.3 | 37.9 | 60.0 | 35.7 | 86.0 | 73.5 |

Mk5 answered **225/300** correctly versus mk4's **230/300**: 17 paired
wins and 22 losses. It tied mk4 on single-hop and multi-hop counts, gained
one open-domain answer, lost two temporal answers and lost four
adversarial answers. **This run does not show improvement or establish
that adversarial accuracy is preserved.** K5 scored 223/300.

Repeating unchanged mk4 against the earlier v4 run changed **22/300
verdicts** (11 wins, 11 losses) while leaving its total unchanged.
The new comparison also has seven questions with conflicting judge
verdicts on identical answer text. These are development diagnostics,
not canonical scores or evidence for promoting v5 over v4.

Wrong non-adversarial abstentions remain a problem: mk5 produced 16
versus mk4's six (mk5: nine temporal, four open-domain and three
single-hop). K5 produced 26. Source checks and narrower evidence can
trade unsupported guesses for abstention without improving accuracy;
these counts do not identify which option caused the change.

The follow-up `locomo-dev-v5-ablation.toml` uses 200 of these 300 questions
and an 850-credit cap. IDs are chosen by sorting
`SHA256("keep-v5-ablation-200-v1:" + id)` and taking the first 200, without
reading scores or question text. It repeats mk5 and runs prompt-only mk4
plus the four single-option removals. The smaller sample allows a paired
repeat and all four removals within the shared 3,000-credit allowance.

#### LongMemEval-S development comparison

Run `20261116T072320.459Z-e07a1338dc63`, the same pinned 60 dev
questions, five arms, 600 answer/judge calls, **322.466075 credits** and
no unsettled usage. Every scheduled answer completed. The sealed
holdout was not consumed. Each category below has ten questions.

| Arm | All (60) | Knowledge update | Multi-session | Assistant recall | Preference | User recall | Temporal |
|---|---:|---:|---:|---:|---:|---:|---:|
| mk4 | 70.0 | 80 | 60 | 90 | 50 | 80 | 60 |
| mk5 | 80.0 | 90 | 80 | 90 | 60 | 90 | 70 |
| k5 | 78.3 | 100 | 60 | 100 | 40 | 90 | 80 |
| dense-6k-v5 | 70.0 | 100 | 80 | 100 | 20 | 90 | 30 |
| full-context-v5 | 66.7 | 100 | 60 | 100 | 20 | 80 | 40 |

Mk5 scored **48/60** versus mk4's **42/60**, with seven paired wins and
one loss. K5 scored 47/60. The six-answer net gain is smaller than the
previously observed nine-answer repeat variation. It is encouraging but
does not establish an improvement. The prompt-only controls do not
isolate individual retrieval changes; that requires the ablations.

`longmemeval-s-dev-v5-ablation.toml` repeats mk5 and removes each option
on all 60 dev questions, alongside prompt-only mk4. It keeps four
workers and has a separate 500-credit cap within the same overall
allowance. Apart from the named ablations, model settings, token budgets
and data selections match the main comparison. K5 retains its 2,000-token
v4 budget; mk5 and dense use 6,096 tokens, while full context uses the
available history.

#### LoCoMo ablations and repeat check

Run `20261116T075217.951Z-d3f683c4b6f4`: 200 questions, 2,400
answer/judge calls, **740.110830 credits**, no unsettled usage or refused
calls. This subset has 16 multi-hop, 39 temporal, eight open-domain,
84 single-hop and 53 adversarial questions.

| Configuration | Correct / 200 | Change from mk5 repeat |
|---|---:|---:|
| mk4, original prompt (matched main-run questions) | 157 | +4 |
| mk4 with answer-v5 only | 157 | +4 |
| mk5, all options, first run | 155 | +2 |
| mk5, all options, repeated | 153 | — |
| mk5 without source snippets | 155 | +2 |
| mk5 without date annotations | 154 | +1 |
| mk5 without question facets | 153 | 0 |
| mk5 without individual-fact selection | 160 | +7 |

These are observed directions, not established effects. Adding source
snippets scored two fewer answers; adding date annotations scored one
fewer; facets tied; adding individual selection scored seven fewer than
its removal. Prompt-only mk4 tied original mk4's matched total.

The unchanged mk5 repeat changed **14/200 verdicts** (six wins, eight
losses). The strongest ablation, removing individual selection, had 15
paired wins and eight losses against that repeat. Five of its seven net
gains were adversarial and two were temporal. The other category totals
were unchanged.

A full-300 follow-up ran only `keep-mk5-no-selection` using
`locomo-dev-v5-selection-check.toml`, run
`20261116T082444.909Z-6d90ae080608`. It used 600 answer/judge calls and
**185.244935 credits**, with no unsettled usage or refused calls.

| Configuration | All (300) | Non-adversarial (232) | Multi-hop (29) | Temporal (60) | Open-domain (14) | Single-hop (129) | Adversarial (68) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mk5 without individual selection | 76.3 | 72.4 | 34.5 | 73.3 | 35.7 | 84.5 | 89.7 |

It scored **229/300**, versus mk4's 230 and full mk5's 225. Against full
mk5 it had 23 wins and 19 losses: six more adversarial answers but two
fewer non-adversarial answers. On the repeated 200 questions it again
scored 160, yet **ten verdicts changed**. The first ablation's apparent
win therefore did not establish superiority over mk4. This follow-up
overlaps the tuning subset and is not an independent confirmation test.

#### LongMemEval-S ablations

Run `20261116T081150.343Z-c3cbd400a34b`: all 60 pinned dev questions,
720 answer/judge calls, **225.531115 credits**, no unsettled usage or
budget-refused calls. These are counts, not percentages; each category has ten
questions.

| Configuration | Correct / 60 | Update | Multi-session | Assistant | Preference | User | Temporal |
|---|---:|---:|---:|---:|---:|---:|---:|
| mk5, all options, repeated | 46 | 9 | 7 | 9 | 5 | 9 | 7 |
| mk4 with answer-v5 only | 39 | 8 | 5 | 9 | 3 | 9 | 5 |
| mk5 without source snippets | 45 | 9 | 7 | 8 | 4 | 9 | 8 |
| mk5 without date annotations | 45 | 9 | 7 | 8 | 5 | 9 | 7 |
| mk5 without question facets | 46 | 9 | 7 | 9 | 5 | 8 | 8 |
| mk5 without individual-fact selection | 43 | 8 | 7 | 9 | 2 | 9 | 8 |

The mk5 repeat fell from 48 to 46, with **two changed verdicts**, both
losses. The judge refused its grading task twice instead of returning a
verdict, once each for `-no-provenance` and `-no-dates` on the same
assistant-recall question. Both are recorded as malformed and remain in
the scheduled denominator; neither was retried or imputed. Their apparent
one-answer deficits disappear on the 59 valid paired judgments:
both variants and mk5 score 45/59 there. Those are not measured benefits
of source snippets or date annotations.

Facets tied; individual selection had an observed three-answer benefit.
Prompt-only mk4 scored three fewer than original mk4's 42 in the main
comparison. Thus the main v5 result cannot be credited to the new answer
prompt alone.

**Decision:** no new option was dropped based on these scores.
Individual selection's removal traded better LoCoMo adversarial results
for weaker non-adversarial and LongMemEval-S results, and did not beat
mk4 on the full 300-question check. The other changes were small and
mixed. All four switches remain available for controlled experiments.
V5 is an experimental variant, **not a demonstrated quality upgrade**.
The runner defaults to version 5 as requested; version 4 remains
available unchanged with `KEEP_G0_VERSION=4`.

#### Development spend and validation

| Run | AI credits |
|---|---:|
| Balanced 30-question pilot | 193.604440 |
| LoCoMo 300, five-arm comparison | 1,002.962025 |
| LongMemEval-S 60, five-arm comparison | 322.466075 |
| LoCoMo 200, six-arm ablations | 740.110830 |
| LongMemEval-S 60, six-arm ablations | 225.531115 |
| LoCoMo 300, selection-removal follow-up | 185.244935 |
| **Total** | **2,669.919420** |
| **Unused allowance** | **330.080580** |

All six run summaries report zero unsettled credits and zero budget-refused
calls. All 3,960 answers completed; 3,958 received valid judge verdicts,
with the two malformed judgments above kept as explicit failures.
The total includes the pilot and follow-up, not just the two main
comparisons. Each new run's cap plus previous billed/unsettled usage and
other active runs' caps was checked against 3,000 before launch.

Validation passed: **305 tests**, including the local Postgres/Qdrant
integration tests; workspace/all-target Clippy with warnings denied,
including bench/server local-embedding features; formatting; shell
syntax; and every new manifest's model pins, subset hashes and G0 caps.
Focused synthetic regressions cover source ownership, provenance,
relative dates, facets, selection, budgets, legacy context digests,
non-adversarial denominators, incomplete failure reviews and concurrent
cache writes. No dataset labels or sealed holdout results were changed
or used for tuning.

#### Additional findings from the v3/v4 records

These are diagnostic signals, not automatically verified causes or label
corrections. The original data and old results were not changed.

| Signal | v4 mk4, all development questions | v4 mk4, fixed 300 | v3 mk3, all development questions |
|---|---:|---:|---:|
| Incorrect answers | 328 | 70 | 382 |
| Incorrect non-adversarial abstentions | 35 | 8 | 71 |
| Incorrect answers whose context omitted older fact-group entries | 64 | 14 | 76 |
| Incorrect list/count questions (conservative wording heuristic) | 47 | 12 | 43 |
| Incorrect count questions beginning “how many” | 12 | 4 | 10 |
| Incorrect answers over 40 words | 1 | 0 | 1 |
| Incorrect answers with citation-based evidence recall = 1 | 261 | 60 | 320 |

Every incorrect mk4 answer used at least 5,800 of its 6,096 context
tokens. More source text therefore replaces other evidence; it is not
free. Fact groups keeping newest entries instead of relevant entries and
incomplete lists motivated the individual selection and list instructions.
All LoCoMo question dates are unknown in these records, so the new prompt
explicitly forbids using the real-world clock for a source's “years ago”.

Across all arms, **98 v4 questions** (22 in the fixed 300) and **80 v3
questions** (19 in the fixed 300) had the same answer text graded both
correct and incorrect. This comparison folds case and whitespace only.
For example, `conv-26:q108` has conflicting grades on identical text.
The judge is unchanged; this remains a measurement limit.

Examples of the remaining reasoning gaps include `conv-26:q38` and
`conv-42:q75` (incomplete or overbroad lists), and `conv-42:q43`
(inconsistent duration arithmetic). These are examples from development
records, not special cases in the implementation. Verbosity is not a
large measured failure pattern: only one wrong answer in each full run
exceeded 40 words.

`conv-26:q94` illustrates a separate label-review issue: the question
attributes an object to the other speaker, while the cited conversation
attributes it to its owner. Some baselines follow the question's mistaken
premise and receive credit. All-arms-wrong cannot detect every faulty
label. This case was not used as a target for changing attribution rules.

The strict v4 all-arms-wrong review queue contains **93** complete
question/seed pairs: 40 multi-hop, 21 temporal, 21 open-domain and 11
single-hop; none adversarial. One more open-domain question,
`conv-26:q30`, was wrong for every judged arm but had unavailable oracle
evidence. That explains the previously reported total of 94. The tool
lists this incomplete comparison separately rather than silently counting
it as complete. The fixed 300 contain 22 complete candidates (10, 7, 4
and 1 respectively). These remain review candidates, not an exclusion
list or edited gold labels.

Validation also exposed a cache-write race: concurrent writers in one
process shared a staging filename. A synthetic regression reproduced
51 failed writes out of 128. Unique per-write staging names made all 128
succeed. Cache keys, prompts and stored replies are unchanged.

### Version 6: working notes and consistent judge verdicts

`KEEP_G0_VERSION=6` runs `manifests/g0/{locomo,longmemeval-s}-copilot-v6.toml`
with `keep-mk6` as the candidate and `dense-6k-v6`, `full-context-v6`,
`keep-mk5`, `keep-mk4` and `keep-mk3` added to the comparison. **The runner
still defaults to version 5.** The caps stay at 26,000 and 39,000 AI
credits and authorize no automatic run. LongMemEval-S remains
development-only; never pass `--unseal` or `--consume-holdout`.

Manifest hashes:
`sha256:04099975b7ba0e17ede46a1107c4bc4a65ceb5976d2eadd83dca8fa8269683fe`
(LoCoMo) and
`sha256:8dcfa6c08d5bd488084f4381cc402edd6d2972faf0c9838d2bd36e2145e5b485`
(LongMemEval-S). Each is the v5 manifest plus the four v6 arms, the
`[judge_consistency]` section and a 1,024-token answer reservation.
Models, datasets, subsets, retrieval and every earlier arm are unchanged.

`manifests/dev/locomo-dev-v6.toml` (hash `789896d4…`) and
`longmemeval-s-dev-v6.toml` (hash `a7311766…`) pin the same 300 LoCoMo and
60 LongMemEval-S development questions as v4 and v5.

#### What changed

| Change | Plain-language purpose | Main files |
|---|---|---|
| Verdict cache | The same answer to the same question always gets the same verdict, in one run and across runs. | `keep-bench/src/judge.rs`, `runner.rs` |
| Majority of three | Borderline answers (partial word overlap with the gold answer) are judged up to three times and the majority wins. | `keep-bench/src/judge.rs`, `runner.rs`, `dryrun.rs` |
| Answer-v6 prompt | The model writes up to eight short lines of working notes (who, dated evidence, person check, date and list working), then a final `Answer:` line. Only that final line is judged. | `keep-bench/src/prompt.rs` |
| Abstention and inference rules | "I don't know" only when the matching information is about someone else or never connected to the person; no premise corrections; "Likely …" inferences only for would/could/might questions. | `keep-bench/src/prompt.rs` |
| v6 arms and controls | `keep-k6` and `keep-mk6` are `keep-k4` and `keep-mk4` with answer-v6; `dense-6k-v6` and `full-context-v6` use the same prompt. | `manifests/`, `scripts/run-g0-copilot.sh` |
| Reporting | Records keep the full reply and the judged `final_answer`; summaries count reused verdicts and extra votes per arm. | `artifacts.rs`, `report.rs`, `scripts/analyze-failures.py` |

Crate paths are under `crates/`. See
[Consistent judge verdicts](methodology.md#consistent-judge-verdicts) and
[Version 6 answer notes](../knowledge.md#version-6-answer-notes).
**V6 changes no retrieval, memory or knowledge code.** The mk6 gain below
comes from the answer prompt.

#### LoCoMo development comparison

Run `20261116T175217.524Z-789896d4b0af`, all 300 pinned questions, six
arms, **817.73 credits** (0.77 of it one unsettled call), no refused
calls, malformed or failed judgments. Values are correct answers.

| Arm | All (300) | Non-adversarial (232) | Multi-hop (29) | Temporal (60) | Open-domain (14) | Single-hop (129) | Adversarial (68) | Wrong abstentions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mk4 | 225 (75.0%) | 171 | 11 | 44 | 5 | 111 | 54 | 5 |
| mk5 | 226 (75.3%) | 169 | 13 | 42 | 5 | 109 | 57 | 16 |
| **mk6** | **239 (79.7%)** | **177** | 13 | 47 | 5 | 112 | **62** | 8 |
| k6 | 235 (78.3%) | 173 | 13 | 48 | 6 | 106 | 62 | 10 |
| dense-6k-v6 | 227 (75.7%) | 170 | 12 | 49 | 7 | 102 | 57 | 9 |
| full-context-v6 | 241 (80.3%) | 180 | 12 | 49 | 7 | 112 | 61 | 6 |

"Wrong abstentions" counts non-adversarial questions answered with
"I don't know".

Paired against mk4 in the same run (wins / losses):

| Arm | Wins | Losses | Net | Where |
|---|---:|---:|---:|---|
| mk5 | 18 | 17 | +1 | |
| **mk6** | **21** | **7** | **+14** | adversarial +9/−1, temporal +4/−1, multi-hop +3/−1, single-hop +5/−4 |
| k6 | 26 | 16 | +10 | single-hop +5/−10 |
| dense-6k-v6 | 24 | 22 | +2 | single-hop +6/−15 |
| full-context-v6 | 30 | 14 | +16 | |

Mk6 beat mk5 by 25 wins to 12 losses and dense-6k-v6 by 27 to 15, and was
within noise of full-context-v6 (14 wins, 16 losses) at less than a third
of its context (6,002 against 20,281 tokens on average). A sign test on
mk6's 21/7 against mk4 gives p ≈ 0.013. Across separate runs, mk6 (final
prompt) scored 236 and 239, mk4 226 and 225.

The answer-v6 prompt also lifts both controls: full-context-v6 at 241
compares with 213 for full-context-v5 in the v5 comparison. The prompt is
the improvement, not a memory change; it helps memory most on
adversarial questions, where mk6 keeps mk4's lead.

Of mk6's 61 wrong answers, only 16 are answered correctly by
full-context-v6. The other 45, including 15 of 16 multi-hop errors, are
wrong with the whole conversation in context. Better packing of the
6,096-token context can therefore recover at most a small share; most
remaining errors are reasoning, label or judging problems.

#### LongMemEval-S development comparison

Run `20261116T174511.923Z-a731176612fb`, the pinned 60 questions, six
arms, **343.83 credits**, no unsettled usage, refusals or failed
judgments. A repeat of mk4, mk5 and mk6, run
`20261116T182830.699Z-a731176612fb`, cost **78.77 credits** (the cache
reused most verdicts). The sealed holdout was not consumed. The four
abstention questions are split out of their categories here.

| Arm | All (60) | Abstention (4) | Update (9) | Multi-session (9) | Assistant (10) | Preference (10) | User (9) | Temporal (9) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mk4 | 42 | 4 | 7 | 4 | 9 | 5 | 8 | 5 |
| mk5 | 49 | 4 | 8 | 7 | 9 | 5 | 8 | 8 |
| **mk6** | **48** | 4 | 7 | 8 | 9 | 3 | 8 | 9 |
| k6 | 46 | 3 | 8 | 6 | 10 | 4 | 8 | 7 |
| dense-6k-v6 | 43 | 4 | 9 | 9 | 10 | 1 | 8 | 2 |
| full-context-v6 | 46 | 4 | 9 | 9 | 10 | 3 | 8 | 3 |
| mk4, repeat | 42 | 3 | 7 | 5 | 9 | 4 | 8 | 6 |
| mk5, repeat | 46 | 4 | 8 | 7 | 9 | 4 | 8 | 6 |
| mk6, repeat | 48 | 4 | 7 | 8 | 8 | 3 | 9 | 9 |

Mk6 scored 48 in both runs against mk4's 42 in both: +8/−2 and +9/−3
paired. Gains are in multi-session and temporal questions; preference
questions lost one or two. Mk5 scored 49 and 46, so mk6 and mk5 are tied
on this set (mk6 against mk5: 2 wins, 3 losses). Mk6 beat
full-context-v6 by 8 to 6 and dense-6k-v6 by 9 to 4. Sixty questions are
few: each paired result alone has p ≈ 0.1–0.15, but the +6 repeated
exactly and exceeds every repeat difference measured after the judge fix
(at most 3).

#### Measured noise

Flips are questions whose verdict differs between two runs of the same
arm. "Same-text flips" are flips where both answers normalise to the same
text: pure judge disagreement.

| Pair | Questions | Flips | Same-text answers | Same-text flips |
|---|---:|---:|---:|---:|
| **Before:** mk4, two v4 dev runs, LoCoMo | 300 | 25 | 214 | 3 |
| **Before:** mk4, two v4 dev runs, LongMemEval-S | 60 | 12 | 43 | 2 |
| **After:** mk4, v6 pilot and comparison, LoCoMo | 300 | 23 | 209 | **0** |
| **After:** mk6, two final-prompt runs, LoCoMo | 300 | 19 | 191 | **0** |
| **After:** mk4, comparison and repeat, LongMemEval-S | 60 | 8 | 42 | **0** |
| **After:** mk5, comparison and repeat, LongMemEval-S | 60 | 5 | 47 | **0** |
| **After:** mk6, comparison and repeat, LongMemEval-S | 60 | 6 | 42 | **0** |

The cache removes judge disagreement on identical answers entirely, as
designed. **It does not remove most of the noise.** About a quarter to a
third of answers differ in wording between identical runs, because the
answer model is sampled, and different wording can earn a different
verdict. That variation now dominates: LoCoMo still flips about 20 of 300
verdicts, LongMemEval-S 5 to 8 of 60. The flips that remain reflect the
answer model, not grading.

Majority voting rarely mattered: the majority overruled the first vote on
6 of 383 borderline LoCoMo answers and 7 of 95 LongMemEval-S answers. It
costs up to two extra judge calls per borderline answer; the cache more
than pays for it when arms or runs repeat answers.
In a single comparison run, only 862 of 1,800 LoCoMo judgments had
distinct keys.

To shrink the remaining noise, repeat runs and pool them, or use more
questions; do not read single-run differences under about 10 of 300 or
4 of 60.

#### Prompt revisions and ablations

Answer-v6 was revised three times on the development questions. Each
revision kept the name `answer-v6`; records carry the prompt's digest
in `prompt.sha256`. Only the final one is in the code, pinned by test
(`sha256:1cc1d5ce…40d3`).

| Revision | LoCoMo mk6 / 300 | mk4 in the same run | Change |
|---|---:|---:|---|
| 1 (pilot) | 234 | 226 (+18/−10) | Working notes, final line, person check |
| 2 | 241 | — (+23/−8 against the pilot's mk4) | No premise corrections; "Likely" only on would/could/might questions; plain text |
| 3 (final) | 236, 239 | 225 in the second run (+21/−7) | Same-person evidence may be worded differently; dates from stated durations; "now" is the latest memory date when today's date is unknown |

Revision 3 against revision 2 was 8 wins and 13 losses on the first run,
within noise. It was kept because it addressed wrong abstentions found in
revision 2's records with generic rules, and it was the version run on
LongMemEval-S. Revision 2 is not available as a switch.

`keep-mk6-r5` combines the v5 retrieval (source snippets, anchored dates,
question facets, individual-fact selection) with answer-v6. Run
`20261116T173120.507Z-789896d4b0af` scored it **232/300** against mk6's
241 in the same run (+22/−16 against mk4). **Dropped:** the v5 retrieval
options do not add to answer-v6. The arm remains in the dev manifest
for reproduction; it is not in the G0 manifests.

`keep-k6` (2,000-token budget) scored 235 and 46, a few answers below
mk6 on both sets; most of its losses are single-hop questions whose
evidence did not fit.

**Not done.** These ideas were assessed but not built:

- *Iterative retrieval for multi-hop questions.* Contexts are built before
  any answer is generated, so a second, answer-driven search needs a new
  runner phase. With 15 of 16 mk6 multi-hop errors also wrong under full
  context, more retrieval would help little; the working notes left
  multi-hop at 13/29 in the final run, up from 11.
- *Relevance-ranked packing across all context blocks.* As above, at most
  16 of 61 LoCoMo errors are retrieval-limited.
- *A separate "never mentioned" check before answering.* The prompt's
  person check achieved the same effect: adversarial accuracy rose from
  54 to 62 of 68 while wrong abstentions stayed at 5 to 8.

#### Failures fixed during development

The second development run crashed partway: the answer extractor sliced
a string at a byte inside a curly apostrophe. About 240 of 600 answers
had been paid for and lost, with no summary written. The extractor now
uses character-safe slicing, with a regression test on such text. The
first launch of the same run failed before any call because the database
settings were missing from the environment.

#### Expected G0 spend

The dry run prices every answer at 1,024 output tokens and every
judgment at three votes, and cannot count cache hits. Its worst case is
**66,516.7 credits for LoCoMo (cap 26,000)** and **45,713.6 for
LongMemEval-S (cap 39,000)**, against 21,298.0 and 32,341.6 for v5. The
script shows `OVER` for both. These figures were taken with three votes;
the G0 manifests now use one, which lowers both.

Expected spend is lower. The CLI does not cap answer length, so the
1,024 tokens only size each call's reservation; v6 answers used at most
380 output tokens in the development runs (median about 110). Judge
calls cost about 0.41 credits each. In the v4 G0 records, 22 arms
produced only 8,734 distinct answers to 1,242 questions (32% of the
27,324 judgments), and about half of those distinct answers were
borderline; a borderline answer takes one extra vote, two when the
first two disagree. From these rates, the 30-arm LoCoMo run should cost
**about 16,000–20,000 credits**, and LongMemEval-S about **15,000**,
including 6,500 for extraction if its cache is empty. Each run reserves
per call and stops when its cap is reached, leaving a partial result.
To lower that risk, the G0 v6 manifests use `votes = 1`: the cache still
guarantees consistent verdicts, and judge cost falls by about a third.
The dev manifests keep the three-vote majority.

#### Development spend and validation

| Run | AI credits |
|---|---:|
| LoCoMo 300, mk4 and mk6 pilot prompt | 403.645995 |
| LoCoMo 300, mk6 revision 2 and mk6-r5 (crashed, no summary) | about 200 (estimated) |
| LoCoMo 300, mk6 revision 2 and mk6-r5 | 253.732900 |
| LoCoMo 300, mk6 final prompt | 114.833400 |
| LongMemEval-S 60, six-arm comparison | 343.833490 |
| LoCoMo 300, six-arm comparison | 817.726370 |
| LongMemEval-S 60, mk4/mk5/mk6 repeat | 78.768220 |
| **Total** | **about 2,213** |
| **Unused of the 4,000 allowance** | **about 1,787** |

All runs except the crashed one completed every answer with valid judge
verdicts: no malformed or failed judgments and no refused calls.
Further prompt tuning on the same 300 questions was stopped to avoid
fitting the development set, although credits remained.

Validation passed: **305 tests**, including the local Postgres/Qdrant
integration tests; workspace/all-target Clippy with warnings denied,
including the bench and server local-embedding features; and formatting.
New tests cover answer normalisation, cache keys, the borderline rule,
majority voting, failed votes (never cached), concurrent identical
judgments, the final-answer extractor (including non-ASCII text), the
pinned prompt digests, and the v6 manifests. The v6 manifest test checks
that every v5 arm of the G0 manifests, and the v4/v5 Keep arms of the dev
manifests, are unchanged, and that only the answer reservation and the
judge settings differ in the model settings.

### Version 7: answer bridging and specific wording

`KEEP_G0_VERSION=7` runs `manifests/g0/{locomo,longmemeval-s}-copilot-v7.toml`
with `keep-mk7` as the candidate and `dense-6k-v7`, `full-context-v7`,
`full-context-v6`, `keep-mk6`, `keep-mk5` and `keep-mk4` added to the
comparison. **The runner still defaults to version 5, and no G0 v7 run has
been launched.** Each manifest is the v6 manifest plus three answer-v7
arms; models, judge settings (one vote), datasets, retrieval and every v6
arm are unchanged. The caps grow by a tenth for the extra arms, to 28,600
and 42,900 AI credits. LongMemEval-S remains development-only; never pass
`--unseal` or `--consume-holdout`.

Manifest hashes:
`sha256:22084bb55535889a95d75b0dc2d5dc18c73cd413426b03cc6364c7912bbff889`
(LoCoMo) and
`sha256:d12357547d977b9498af7516623d094216779206ad4db87c149393e567f514d3`
(LongMemEval-S). `manifests/dev/locomo-dev-v7.toml` (hash `442d9dca…`)
and `longmemeval-s-dev-v7.toml` (hash `4e99dc8a…`) pin the same 300 and
60 development questions as v4 to v6, with the unchanged `keep-mk6` and
`full-context-v6` as paired controls.

#### Why: error analysis of the G0 v6 LoCoMo records

In G0 v6, keep-mk6 trailed full-context-v6 by 2.1 points overall, most of
it on open-domain (−8.9) and single-hop (−3.5). Of the questions full
context answered and mk6 did not, the cited evidence was usually already
in mk6's context (mean evidence recall 0.78 for open-domain and 0.82 for
single-hop losses; offline, mk6's dev contexts reach 97% evidence
recall). The losses were mostly in how the answer used it:

- **No world-knowledge bridge.** Open-domain golds such as a state, a
  country, a console or a game's name need a step beyond memory (Fort
  Wayne → Indiana, Xenoblade 2 → Switch). mk6 answered "I don't know"
  more often than full context (16 against 11 of 67) or gave the city
  when the country was asked.
- **Bare inferences.** "Likely yes." without a reason was judged wrong
  where full context's reasoned answer was judged right.
- **Reworded memory.** mk6 replaced the memory's wording ("goodies" →
  "cookies", "watercolor painting" → "painting") or ignored shared-image
  captions.
- Only about 6–8 of 39 single-hop losses lacked the evidence altogether.

#### What changed

| Change | Plain-language purpose | Main files |
|---|---|---|
| Answer-v7 prompt | answer-v6 plus a bridge step (common knowledge may link a memory fact to what is asked), answers at the level of detail asked, a reason with every inference, the memory's most specific wording, image captions as evidence, and no "I don't know" when the answer is one bridging step away. The person check is unchanged. | `keep-bench/src/prompt.rs` |
| Final-answer line | answer-v7 replies are cut to their `Answer:` line like answer-v6. | `keep-bench/src/prompt.rs`, `runner.rs` |
| v7 arms | `keep-mk7` is `keep-mk6` with answer-v7; `dense-6k-v7` and `full-context-v7` use the same prompt. | `manifests/`, `scripts/run-g0-copilot.sh` |

**V7 changes no retrieval, memory or knowledge code.** See
[Version 7 answer bridging](../knowledge.md#version-7-answer-bridging).

#### LoCoMo development comparison

Run `20261118T124208.309Z-442d9dca00de`, all 300 pinned questions, four
arms, **708.30 credits**, no failed or refused calls. Values are correct
answers; "IDK" counts non-adversarial "I don't know" answers.

| Arm | All (300) | Multi-hop (29) | Temporal (60) | Open-domain (14) | Single-hop (129) | Adversarial (68) | IDK |
|---|---:|---:|---:|---:|---:|---:|---:|
| mk6 | 231 (77.0%) | 8 | 48 | 6 | 108 | 61 | 7 |
| **mk7** | **239 (79.7%)** | 11 | 49 | 7 | 111 | 61 | 5 |
| full-context-v6 | 239 (79.7%) | 16 | 44 | 8 | 113 | 58 | 6 |
| full-context-v7 | 236 (78.7%) | 13 | 45 | 10 | 112 | 56 | 3 |

Paired (wins / losses for mk7): against mk6 18/10 (single-hop 7/4,
multi-hop 3/0, open-domain 2/1, adversarial 4/4); against
full-context-v6 17/17; against full-context-v7 22/19.

Repeat run `20261118T134228.425Z-442d9dca00de` (mk6 and mk7 only,
**213.07 credits**): mk7 237, mk6 234, paired 13/10. Pooled over both
runs mk7 scored 476 of 600 against mk6's 465 (+1.8 points, 31 wins to
20 losses). About 20 of 300 verdicts flip between identical runs, so
each single-run difference is within noise; the direction held in both.
On these questions mk7 matches full context (239 against 239 in the
same run, 241 in the v6 dev run), within the plan's 2-point guardrail.
Open-domain has only 14 development questions, too few to measure that
category's gap; G0 has 67.

#### LongMemEval-S development comparison

Run `20261118T130608.661Z-4e99dc8a7401`, the pinned 60 questions, four
arms, **394.81 credits** (10.36 of it held for one failed
full-context-v6 call).

| Arm | All (60) | Knowledge update (10) | Multi-session (10) | Assistant (10) | Preference (10) | User (10) | Temporal (10) |
|---|---:|---:|---:|---:|---:|---:|---:|
| mk6 | 46 (76.7%) | 8 | 9 | 9 | 1 | 9 | 10 |
| **mk7** | **51 (85.0%)** | 8 | 9 | 9 | 6 | 10 | 9 |
| full-context-v6 | 42 (70.0%) | 9 | 8 | 10 | 0 | 10 | 5 |
| full-context-v7 | 45 (75.0%) | 10 | 8 | 10 | 2 | 10 | 5 |

Paired, mk7 beat mk6 7/2 (preference 5/0) and full-context-v6 13/4.
The preference gain comes from the reason rule: a recommendation that
names the user's own stated preference is judged correct more often.

#### Development spend and validation

| Run | AI credits |
|---|---:|
| LoCoMo 300, four-arm comparison | 708.30 |
| LongMemEval-S 60, four-arm comparison | 394.81 |
| LoCoMo 300, mk6 and mk7 repeat | 213.07 |
| **Total** | **1,316.18** |
| **Unused of the 3,000 allowance** | **1,683.82** |

Only one hypothesis (the answer prompt) was run. A second, retrieval
change was not tried: the error analysis found too few losses with the
evidence missing to justify one, and further tuning on the same 300
questions would risk fitting the development set.

## Known limits

- Token counts use `tiktoken-o200k-base`, the gpt-4o tokenizer, without
  the chat-format overhead, so the harness labels fit and budget decisions
  as provisional. Every LongMemEval-S history fits the 128K window with
  this tokenizer (the longest is about 106.6K tokens), so full context is
  available on every question.
- One storage profile. The superiority gate needs both `postgres-vector`
  and `postgres-qdrant`; G0 runs `postgres-qdrant` only.
- LoCoMo is CC BY-NC 4.0: non-commercial use only.
