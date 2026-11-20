#!/usr/bin/env bash
# Run the G0 canonical manifests against Azure OpenAI.
#
# Reads credentials from ~/.keep/env (override with KEEP_ENV_FILE), which must
# be readable by its owner only. Prints the dry-run estimate and the spend
# ledger, asks for confirmation, then runs each manifest in turn. Values of
# the variables are never printed. See docs/benchmarks/running-g0.md.
set -euo pipefail

cd "$(dirname "$0")/.."

env_file="${KEEP_ENV_FILE:-$HOME/.keep/env}"
manifests=(manifests/g0/locomo.toml manifests/g0/longmemeval-s.toml)
spend_ledger="${KEEP_SPEND_LEDGER:-results/spend-ledger.jsonl}"
spend_cap_usd="${KEEP_SPEND_CAP_USD:-300}"

die() {
	echo "run-g0: $*" >&2
	exit 1
}

[[ -f "$env_file" ]] || die "$env_file not found; create it with mode 0600 (see docs/benchmarks/running-g0.md)"
[[ -O "$env_file" ]] || die "$env_file is not owned by you"
if mode="$(stat -f %Lp "$env_file" 2>/dev/null)"; then :; else mode="$(stat -c %a "$env_file")"; fi
if (((8#$mode & 8#077) != 0)); then
	die "$env_file has mode $mode; it must not be readable by group or others (run: chmod 600 $env_file)"
fi

set -a
# shellcheck disable=SC1090
. "$env_file"
set +a

missing=()
for var in AZURE_FOUNDRY_ENDPOINT AZURE_FOUNDRY_API_KEY \
	KEEP_AZURE_ANSWER_DEPLOYMENT KEEP_AZURE_JUDGE_DEPLOYMENT \
	KEEP_POSTGRES_URL KEEP_QDRANT_URL; do
	if [[ -z "$(printenv "$var" || true)" ]]; then
		missing+=("$var")
	else
		echo "set: $var"
	fi
done
((${#missing[@]} == 0)) || die "not set: ${missing[*]}"
[[ "$AZURE_FOUNDRY_ENDPOINT" == https://* ]] || die "AZURE_FOUNDRY_ENDPOINT must start with https://"

cargo build --release --locked -p keep-bench --features local-embeddings
bench=target/release/keep-bench
spend=(--spend-ledger "$spend_ledger" --spend-cap-usd "$spend_cap_usd")

for m in "${manifests[@]}"; do
	"$bench" validate-manifest "$m"
done
for m in "${manifests[@]}"; do
	echo
	"$bench" dry-run "$m"
done
echo
"$bench" spend show "${spend[@]}"

echo
echo "Each run reserves its manifest's budget.max_cost_usd in $spend_ledger"
echo "and is refused if that would take committed spend past \$$spend_cap_usd."
read -r -p "Run ${manifests[*]} against Azure now? [y/N] " answer
[[ "$answer" == [yY] || "$answer" == [yY][eE][sS] ]] || die "cancelled; nothing was called"

for m in "${manifests[@]}"; do
	echo
	echo "== $m"
	"$bench" run "$m" --allow-model-calls "${spend[@]}"
done
