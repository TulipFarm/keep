#!/usr/bin/env bash
# Download the public benchmark datasets into ./data (gitignored) and verify
# their SHA-256 checksums. Arguments are passed to `keep-bench fetch`, e.g.
#
#   scripts/fetch-datasets.sh                        # LoCoMo, LongMemEval-S, oracle
#   scripts/fetch-datasets.sh --dataset longmemeval-m
#
# Licences differ per dataset and are printed for every file:
#   LoCoMo       CC BY-NC 4.0, non-commercial use only
#   LongMemEval  MIT
# The data is never committed and never redistributed by this repository.
set -euo pipefail
cd "$(dirname "$0")/.."
exec cargo run --quiet --release -p keep-bench -- fetch "$@"
