# Vendored evaluation prompts

These files are copied byte-for-byte from two published benchmark harnesses so
that `keep-bench` can score runs under the same answer and judge prompts as
the systems it is compared with. Do not edit them. The harness hashes each
file and records the hashes in `run.json` and `summary.json`; a changed file
changes the recorded hash.

To update, re-extract from a newer upstream commit, record the new commit and
hashes here and in `src/protocol.rs`, and bump the protocol prompt versions.

## `mem0-locomo/`

Source: <https://github.com/mem0ai/memory-benchmarks>, commit
`4b61c5d31b9c668a12b4f5e78064248a02c82d2b`, file
`benchmarks/locomo/prompts.py`. Licence: Apache License 2.0,
Copyright 2025 Mem0.ai. See `THIRD_PARTY_NOTICES.md` at the repository root.

| File | Upstream value | SHA-256 |
|---|---|---|
| `answer.txt` | `ANSWER_GENERATION_PROMPT` | `79c9f09bcc8d5e9e8b7e9786af587b02a67d366ab79285fc148b73fd20f6297b` |
| `judge-system.txt` | `JUDGE_SYSTEM_PROMPT` | `36c007917faf1ab84516cdca577fb523711a9b993706fbae8ae37806e6f9adcc` |
| `judge.txt` | `JUDGE_PROMPT` (the unified judge template built without evidence, as `run.py` uses it by default) | `d248e056d993725e28fba8d16ca7081f0b59deae272ef294f3c6b00d48eac02b` |

Placeholders keep Python's `str.format` names: `{reference_date}`,
`{memories}`, `{question}` in the answer prompt and `{question}`, `{answer}`,
`{response}` in the judge prompt.

## `longmemeval-official/`

Source: <https://github.com/xiaowu0162/LongMemEval>, commit
`9e0b455f4ef0e2ab8f2e582289761153549043fc`. Licence: MIT,
Copyright (c) 2024 Di Wu. See `THIRD_PARTY_NOTICES.md`.

| File | Upstream value | SHA-256 |
|---|---|---|
| `judge-default.txt` | `get_anscheck_prompt` template for `single-session-user`, `single-session-assistant` and `multi-session` (`src/evaluation/evaluate_qa.py`) | `fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc` |
| `judge-temporal-reasoning.txt` | template for `temporal-reasoning` | `8d33a5fdd83afeeb4592454a965eab43d1fcb2dedc042d1d3892f4254be6c273` |
| `judge-knowledge-update.txt` | template for `knowledge-update` | `183a9b3a6197ec620940f610cdc1207201ec98c1113dd633ea685cfc322fafac` |
| `judge-single-session-preference.txt` | template for `single-session-preference` | `741ee3bcbea7ff5e8ed359acef61d2f8ded3de021bbcff6ee13de455f2e2aa9b` |
| `judge-abstention.txt` | template for abstention questions (`_abs` in the question id) | `5c0b365a1e1d06db36377c735432b56e122ca3c428f89faf61d43a0d5a7e050b` |
| `reader-cot.txt` | chain-of-thought reader prompt without fact expansion (`src/generation/run_generation.py`) | `9e2b3110622929ab896696dd8937231c7436740ec3b9586f653f97346e19ab2c` |

Placeholders are positional `{}` as upstream: question, answer, response for
the judge; history, current date, question for the reader.
