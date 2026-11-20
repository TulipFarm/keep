# Running G2

G2 is the canonical, gate-eligible run. It compares the frozen Keep
candidate with the two preregistered comparators on LoCoMo and LongMemEval-S, with three seeds, both
storage profiles, and one Azure OpenAI deployment. It runs in two phases: a
development run first, then a separate confirmation run on the sealed holdout.

| | |
| --- | --- |
| Development manifests | `manifests/g2/locomo.toml`, `manifests/g2/longmemeval-s.toml` (+ `*-pgvector.toml`); version 2 adds `-v2` to each name |
| Confirmation manifests | `manifests/g2/*-confirmation.toml` (+ `*-confirmation-pgvector.toml`); version 2 adds `-v2` |
| Answer, judge and extraction model | Azure deployment of `gpt-6-luna`, version `2026-09-22`, Data Zone Standard |
| Reasoning effort | answer `low` (max 2,048 output), judge `none` (max 16), extraction `low` (max 3,000) |
| Embeddings / reranking | the pinned local ONNX models (no Azure calls) |
| Storage | `postgres-qdrant` and `postgres-vector` (the gate requires both) |
| Seeds | 1, 2, 3 |
| Candidate arm | version 1: `keep-mk7` (the G0 v7 arm, frozen): keep-mk6 answered with `answer-v7`; version 2: `keep-mk8`, keep-mk7 with the [v8 options](../knowledge.md#version-8-focused-excerpts) |
| Gate comparators | `full-context-v7` (complete history) and `dense-6k-v7` (dense retrieval at the candidate's 6,096 context tokens), both with `answer-v7`, in both versions |
| Reported, not gated | version 1: `keep-mk6` (the previous candidate, `answer-v6`); version 2: `keep-mk7` |
| Price | $0.11 per 1M input tokens, $0.55 per 1M output tokens |

## Manifest versions

Manifests are frozen once they have results, so a new candidate gets a new
set. `scripts/run-g2-azure.sh --set v1` (the default) runs the original
files with `keep-mk7`; `--set v2` (or `KEEP_G2_SET=v2`) runs the `-v2` files
with `keep-mk8` and reports `keep-mk7`. Version 2 changes only the ids,
descriptions, arms and frozen candidate: the dataset, holdout salt and
commitment, seeds, models, prompts, budgets, caps and statistics are
version 1's (a test enforces this). Comparison folders of version 2 end
in `-v2`.

## Arms and the gate

Each manifest has exactly four arms. Section 7.6 of the design requires the
candidate to beat both a strong full-context comparator and the strongest
plain retrieval baseline, with the same answer model, judge and prompt. The
harness also refuses a confirmation manifest whose combined Memory +
Knowledge arm has no equal-context `bm25` or `dense` arm. So both
comparators use the candidate's `answer-v7` prompt, and only the memory
differs. `keep-mk6` is compared in a separate report and never enters the
gate's Holm family. The manifests write out the statistical test in
`[statistics]`. Holm adjusts all four superiority tests (two comparators in
each of two storage profiles) as one family, so the first test must clear
5%/4 = 1.25%. Each profile must also pass all its checks on its own.

## Same model for answers and judge

Only `gpt-6-luna` on Azure may be used for live runs. The answer model, the
judge and the fact extractor therefore share one pinned deployment. The judge
has its own prompt (`judge-v1`) and no reasoning, but a judge from the same
model family as the answerer may favour its answers. Every report must state
this limitation.

## Exact token counts

OpenAI has not published the gpt-6-luna tokenizer. Public token counters
label their gpt-6-luna counts as estimates (for example
tiktokenizer.com/models, checked 2026-10-09). The manifests use the
`o200k_base` encoding (`tiktoken-o200k-base`) only to make the fit decision
before the run. On its own, that encoding is an estimate and is not
gate-eligible.

Exact counts come from the provider instead:

1. Every call records the `usage.prompt_tokens` that Azure reports.
2. After a live run, the runner compares each full-context prompt that was
   sent with its reported count, against the model's input limit (922,000
   tokens). The result is stored as `fit.reported_check` in `run.json`.
3. `keep-bench compare` passes the "full-context fit" gate only when the
   tokenizer is exact or `reported_check.verified` is true. That means every
   checked decision agrees with the reported counts and at least one call
   was checked. Otherwise the gate reports the fit as approximate and not
   gate-eligible.

The largest prompts are about 110K tokens, about one eighth of the limit, so
an estimation error cannot change a fit decision in practice. The check
proves it for each run.

## Rate limits

Azure counts each request's prompt tokens plus its `max_completion_tokens`
against the deployment's tokens-per-minute (TPM) limit. It allows about 6
requests per minute for every 1,000 TPM. The runner paces requests in a
sliding one-minute window. The manifests set 95,000 TPM and 570 RPM, 95% of
capacity 100. On a 429 response, the runner waits for the time in
`retry-after-ms` or `Retry-After`, up to a bounded limit, and pauses every
worker. It retries at most `max_attempts` (6) times. Retries and failures
appear in `run.json` (`rate_limit`) and in the failure counts; none are hidden.

**Raise capacity before running LongMemEval-S.** A full-context
LongMemEval-S request (about 105K tokens plus 2,048 output tokens) is
larger than a 100K TPM deployment can accept. The dry run counts these
requests as "oversized", and the script refuses to run while any exist.
Check your subscription's quota for the model in its region, raise the deployment's capacity in the Azure portal, then pass the same
number to the script with `--capacity N`. The script then sets
`KEEP_DISPATCH_TOKENS_PER_MINUTE` to 950 × N and
`KEEP_DISPATCH_REQUESTS_PER_MINUTE` to 5.7 × N.

**Raise concurrency with capacity.** The manifests set
`dispatch.concurrency = 4`: four questions answered and judged at once, and
four sessions extracted or summarised at once. At capacity 2000 that uses
only about half the token quota. Pass `--concurrency 16` as well; the
script sets `KEEP_DISPATCH_CONCURRENCY`, which overrides
`dispatch.concurrency` for the run. Every call still goes through the same
pacer, 429 handling and budget, so a higher value cannot exceed the quota;
it only keeps more calls waiting. Like pacing, concurrency is operational:
it is not part of the manifest hash, records are identical at any value,
and `run.json` records the value used (`concurrency`).

## Estimate

All figures are from `keep-bench dry-run` at `--capacity 2000` (1.9M tokens
and 11,400 requests per minute). They are worst case: every request uses
its full output cap and every planned extraction runs. Each row is for one
storage profile.

| Manifest | Questions | Calls | Input tokens | Max output tokens | Worst USD | Hours |
| --- | --- | --- | --- | --- | --- | --- |
| LoCoMo development | 1,242 | 30,132 | 198.6M | 31.7M | $39.29 | 2.0 |
| LongMemEval-S development | 345 | 41,126 | 186.0M | 107.1M | $79.36 | 2.6 |
| LoCoMo confirmation | 744 | 18,076 | 122.6M | 19.1M | $23.98 | 1.2 |
| LongMemEval-S confirmation | 155 | 18,320 | 82.0M | 47.6M | $35.22 | 1.1 |

Both storage profiles double the answer and judge calls; extraction is
cached on disk and paid once. Across both profiles, the worst case is about
$178 for development (about 8 hours) and $92 for confirmation (about 4
hours), $270 in total. If only input tokens were paid, the floor would be
about $80 and $43. No request is larger than the per-minute quota.

Price source: Azure Retail Prices API
(`https://prices.azure.com/api/retail/prices`, filter
`serviceName eq 'Foundry Models' and armRegionName eq '<your-region>' and contains(meterName,'6-luna')`).
In a US Data Zone region, the meters "6-luna ShortCo Inp Std DZ" and "6-luna ShortCo Opt Std DZ" are
$0.11 and $0.55 per 1M tokens, effective 2026-09-01, checked 2026-11-14.

## Steps

1. **Create `~/.keep/env`** with mode 0600. It must contain these lines:

   | Variable | Value |
   | --- | --- |
   | `KEEP_AZURE_OPENAI_ENDPOINT` | The resource endpoint, for example `https://<your-resource>.openai.azure.com/` |
   | `KEEP_AZURE_OPENAI_API_KEY` | A key for that resource |
   | `KEEP_AZURE_OPENAI_DEPLOYMENT` | `gpt-6-luna` |
   | `KEEP_POSTGRES_URL` | Postgres URL for the benchmark stack |
   | `KEEP_QDRANT_URL` | Qdrant URL for the benchmark stack |

   The script prints variable names only, never values.

2. **Start storage**:
   `docker compose --profile postgres-qdrant --profile postgres-vector up -d`.

3. **Estimate**: `scripts/run-g2-azure.sh --estimate-only --capacity N`.
   This validates and dry-runs every manifest without calling any model. The
   reports go to `target/g2-logs/`.

4. **Smoke test**: `scripts/run-g2-azure.sh --smoke questions.txt --smoke-usd 5 --max-usd 300 --capacity N --concurrency 16`.
   It runs the candidate arm on a few question ids, and holds that run to
   `--smoke-usd` (through `keep-bench --max-run-usd`, default 5) instead of
   reserving the manifest's full-run budget. `--max-usd` stays the cap on
   all spend in the ledger, so set it above what earlier runs have spent. A reply
   whose model name differs from the pinned `gpt-6-luna-2026-09-22` aborts
   the run on the first call. If Azure reports a different name, check the
   deployment's version before changing `served_model` in all eight
   manifests.

5. **Development**: `scripts/run-g2-azure.sh --max-usd 300 --capacity N --concurrency 16`.
   The script prints the estimate and asks for confirmation. It then runs
   each manifest and finally runs `keep-bench compare` for each dataset:
   the gate against both comparators, plus a separate report against
   `keep-mk6`.
   `--max-usd` caps the cumulative spend recorded in
   `results/g2-spend-ledger.jsonl`.

6. **Confirmation**, only after the development results are reviewed and the
   candidate is frozen:
   `scripts/run-g2-azure.sh --phase confirmation --max-usd 700 --capacity N --concurrency 16`.
   You must type `unseal` before the holdout is read. Because the ledger is
   cumulative, this cap includes the development spend.

## Swapping the candidate

The development and confirmation manifests share every frozen setting. A
test enforces this. To change the candidate, for example to `keep-mk8`:

1. Add the arm to all eight manifests.
2. Set `frozen_candidate` in the four confirmation manifests.
3. Update the frozen-settings test in `crates/keep-bench/tests/cli.rs`.
4. Pass `--candidate keep-mk8` (or set `KEEP_G2_CANDIDATE=keep-mk8`).
