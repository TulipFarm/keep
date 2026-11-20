# Contributing to Keep

Thanks for your interest. Keep is early and research-driven, so the most
useful contributions right now are to the benchmark harness: bugs in loaders,
metrics or statistics, new baselines, and honest reports of results.

## Workflow

1. Open an issue first for anything larger than a small fix, so we can agree
   on the approach.
2. Keep pull requests focused. One logical change per commit.
3. Run the checks before pushing:

   ```sh
   cargo fmt --all --check
   cargo clippy --workspace --all-targets --locked -- -D warnings
   cargo test --workspace --locked
   ```

4. Add tests for the behaviour you change. Tests must run offline: use the
   mock providers and the synthetic fixtures in `fixtures/`, never real
   datasets or real model endpoints.

## Benchmark changes

- Never change a sealed holdout, a fit mask or a manifest after its results
  are known; write a new manifest version instead.
- Report negative and null results the same way as positive ones.
- Fixtures must be synthetic. Do not copy real conversations or personal data
  into the repository.

## Sign-off

Contributions are accepted under the Apache-2.0 licence. Sign off each commit
(`git commit -s`) to certify the [Developer Certificate of
Origin](https://developercertificate.org/).
