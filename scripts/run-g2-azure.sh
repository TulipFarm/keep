#!/usr/bin/env bash
# Run the canonical G2 manifests against the pinned gpt-6-luna deployment on
# Azure OpenAI. See docs/benchmarks/running-g2.md.
#
#   scripts/run-g2-azure.sh --estimate-only
#   scripts/run-g2-azure.sh --set v2 --estimate-only
#   scripts/run-g2-azure.sh --max-usd 250
#   scripts/run-g2-azure.sh --max-usd 300 --phase confirmation
#   scripts/run-g2-azure.sh --max-usd 300 --smoke questions.txt --smoke-usd 1
#
# Every invocation first validates and dry-runs each manifest it would run
# (no model or database call) and prints the estimates. --estimate-only stops
# there. Otherwise --max-usd is required: the hard cap on committed spend
# across every G2 run in the G2 spend ledger. Each run reserves its
# manifest's budget.max_cost_usd before its first call, and the harness
# refuses any run that would take the ledger past the cap.
#
# Options:
#   --max-usd N          hard spend cap in USD (required unless --estimate-only)
#   --phase P            development (default) or confirmation; confirmation
#                        reads and scores the sealed holdout and asks you to
#                        type "unseal" before even estimating it
#   --set S              manifest set (default $KEEP_G2_SET or v1): v1 runs
#                        manifests/g2/<dataset>[-confirmation][-pgvector].toml
#                        (candidate keep-mk7, keep-mk6 reported); v2 runs the
#                        same names ending -v2 (candidate keep-mk8, keep-mk7
#                        reported). Both use full-context-v7 and dense-6k-v7
#   --candidate ARM      the frozen candidate (default $KEEP_G2_CANDIDATE or
#                        the set's); the one setting to change to swap it. It
#                        must be an arm of every manifest run, and must equal
#                        [confirmation].frozen_candidate for confirmation runs
#   --datasets LIST      comma-separated (default longmemeval-s,locomo)
#   --profiles LIST      comma-separated storage profiles (default
#                        postgres-qdrant,postgres-vector; the gate needs both)
#   --capacity UNITS     the deployment's capacity if raised above the 100
#                        units (100,000 tokens/min) the manifests assume;
#                        paces calls at 95% of UNITS x 1,000 tokens and
#                        UNITS x 6 requests per minute. Pacing does not
#                        change what is measured; run.json records it
#   --concurrency N      calls in flight at once (questions answered and
#                        judged, sessions extracted or summarised), in
#                        place of the manifests' dispatch.concurrency (4);
#                        16 suits capacity 2000. Like pacing it does not
#                        change what is measured or the manifest hash;
#                        run.json records it
#   --smoke FILE         development only: run just the candidate on the
#                        question ids in FILE (one per line),
#                        on the first dataset and profile, to check the
#                        deployment, the version pin and the pacing first
#   --smoke-usd N        hold the smoke run to N USD (default 5); --max-usd
#                        stays the cap on all G2 spend
#   --estimate-only      print the dry runs and stop
#
# Reads credentials from ~/.keep/env (override with KEEP_ENV_FILE), which
# must be readable by its owner only; values are never printed. Runs are
# sequential: the pacer that keeps calls under the deployment's quota is
# per process, so never start two G2 runs against the same deployment at
# once.
set -euo pipefail

cd "$(dirname "$0")/.."

die() {
	echo "run-g2-azure: $*" >&2
	exit 1
}

max_usd=""
phase=development
set_name="${KEEP_G2_SET:-v1}"
candidate="${KEEP_G2_CANDIDATE:-}"
datasets=longmemeval-s,locomo
profiles=postgres-qdrant,postgres-vector
smoke=""
smoke_usd=5
capacity=""
concurrency=""
estimate_only=false
while (($#)); do
	case "$1" in
	--max-usd) max_usd="${2:-}" && shift 2 ;;
	--phase) phase="${2:-}" && shift 2 ;;
	--set) set_name="${2:-}" && shift 2 ;;
	--candidate) candidate="${2:-}" && shift 2 ;;
	--datasets) datasets="${2:-}" && shift 2 ;;
	--profiles) profiles="${2:-}" && shift 2 ;;
	--smoke) smoke="${2:-}" && shift 2 ;;
	--smoke-usd) smoke_usd="${2:-}" && shift 2 ;;
	--capacity) capacity="${2:-}" && shift 2 ;;
	--concurrency) concurrency="${2:-}" && shift 2 ;;
	--estimate-only) estimate_only=true && shift ;;
	-h | --help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//' && exit 0 ;;
	*) die "unknown argument $1 (see --help)" ;;
	esac
done

[[ "$phase" == development || "$phase" == confirmation ]] || die "--phase must be development or confirmation"
# Each set is frozen with its arms: the default candidate and the arms
# reported beside the gate. The gate's two comparators are the same in both.
case "$set_name" in
v1) suffix="" default_candidate=keep-mk7 reported=(keep-mk6) ;;
v2) suffix="-v2" default_candidate=keep-mk8 reported=(keep-mk7) ;;
*) die "--set must be v1 or v2" ;;
esac
candidate="${candidate:-$default_candidate}"
if ! $estimate_only; then
	[[ "$max_usd" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--max-usd N is required: the hard cap in USD on all G2 spend"
fi
if [[ -n "$capacity" ]]; then
	[[ "$capacity" =~ ^[1-9][0-9]*$ ]] || die "--capacity must be a whole number of deployment capacity units"
	export KEEP_DISPATCH_TOKENS_PER_MINUTE=$((capacity * 950))
	export KEEP_DISPATCH_REQUESTS_PER_MINUTE=$((capacity * 57 / 10))
	echo "pacing: $KEEP_DISPATCH_TOKENS_PER_MINUTE tokens and $KEEP_DISPATCH_REQUESTS_PER_MINUTE requests per minute"
fi
if [[ -n "$concurrency" ]]; then
	[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || die "--concurrency must be a positive whole number"
	export KEEP_DISPATCH_CONCURRENCY="$concurrency"
	echo "concurrency: $KEEP_DISPATCH_CONCURRENCY calls in flight"
fi
if [[ -n "$smoke" ]]; then
	[[ "$phase" == development ]] || die "--smoke runs the development split only"
	[[ -s "$smoke" ]] || die "$smoke is empty or missing"
	[[ "$smoke_usd" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--smoke-usd must be a USD amount"
fi

IFS=, read -r -a dataset_list <<<"$datasets"
IFS=, read -r -a profile_list <<<"$profiles"
if [[ -n "$smoke" ]]; then
	dataset_list=("${dataset_list[0]}")
	profile_list=("${profile_list[0]}")
fi

manifest() { # dataset profile
	local name="$1"
	[[ "$phase" == confirmation ]] && name+="-confirmation"
	case "$2" in
	postgres-qdrant) ;;
	postgres-vector) name+="-pgvector" ;;
	*) die "unknown storage profile $2" ;;
	esac
	echo "manifests/g2/$name$suffix.toml"
}

# The two preregistered G2 comparators; the gate's Holm family is exactly
# these. The set's reported arms never enter the gate.
comparators=(full-context-v7 dense-6k-v7)

manifests=()
for d in "${dataset_list[@]}"; do
	for p in "${profile_list[@]}"; do
		m="$(manifest "$d" "$p")"
		[[ -f "$m" ]] || die "$m not found"
		for a in "$candidate" "${comparators[@]}" "${reported[@]}"; do
			grep -q "^name = \"$a\"$" "$m" || die "$m has no arm $a"
		done
		if [[ "$phase" == confirmation ]]; then
			frozen="$(sed -n 's/^frozen_candidate = "\(.*\)"$/\1/p' "$m")"
			[[ "$frozen" == "$candidate" ]] ||
				die "$m freezes $frozen, not $candidate; confirmation measures the candidate frozen in development"
		fi
		manifests+=("$m")
	done
done

arm_args=()
if [[ -n "$smoke" ]]; then
	arm_args=(--arms "$candidate" --questions "$smoke")
fi

export KEEP_MODEL_CACHE="${KEEP_MODEL_CACHE:-$PWD/.keep-models}"
cargo build --release --locked -p keep-bench --features local-embeddings
bench=target/release/keep-bench

for m in "${manifests[@]}"; do
	"$bench" validate-manifest "$m"
done
unseal=()
if [[ "$phase" == confirmation ]]; then
	echo
	echo "Confirmation reads and scores the SEALED HOLDOUT with candidate $candidate;"
	echo "even its estimate reads it (lengths only). Score it once, after development is frozen."
	read -r -p 'Type "unseal" to continue: ' answer
	[[ "$answer" == unseal ]] || die "cancelled; nothing was read or called"
	unseal=(--unseal)
fi
oversized=0
mkdir -p target/g2-logs
for m in "${manifests[@]}"; do
	echo
	estimate="target/g2-logs/$(basename "$m" .toml)-dry-run.md"
	"$bench" dry-run "$m" ${unseal[@]+"${unseal[@]}"} | tee "$estimate"
	n="$(sed -n 's/.* \([0-9][0-9]*\) answer requests exceed the whole token quota\./\1/p' "$estimate")"
	oversized=$((oversized + ${n:-0}))
done
if $estimate_only; then
	echo
	echo "Estimates only; nothing was called."
	exit 0
fi
# Azure counts each request's prompt plus its whole output allowance
# against the per-minute quota, so a request larger than the quota cannot
# be admitted and would fail on every attempt. Smoke runs skip the
# full-context arms.
[[ -n "$smoke" ]] || ((oversized == 0)) || die "$oversized answer requests need more than the whole per-minute quota; raise the deployment's capacity and pass --capacity UNITS"

env_file="${KEEP_ENV_FILE:-$HOME/.keep/env}"
[[ -f "$env_file" ]] || die "$env_file not found; create it with mode 0600 (see docs/benchmarks/running-g2.md)"
[[ -O "$env_file" ]] || die "$env_file is not owned by you"
if mode="$(stat -f %Lp "$env_file" 2>/dev/null)"; then :; else mode="$(stat -c %a "$env_file")"; fi
if (((8#$mode & 8#077) != 0)); then
	die "$env_file has mode $mode; it must not be readable by group or others (run: chmod 600 $env_file)"
fi
set -a
# shellcheck disable=SC1090
. "$env_file"
set +a

needed=(KEEP_AZURE_OPENAI_ENDPOINT KEEP_AZURE_OPENAI_API_KEY KEEP_AZURE_OPENAI_DEPLOYMENT KEEP_POSTGRES_URL)
[[ " ${profile_list[*]} " == *" postgres-qdrant "* ]] && needed+=(KEEP_QDRANT_URL)
missing=()
for var in "${needed[@]}"; do
	if [[ -z "$(printenv "$var" || true)" ]]; then
		missing+=("$var")
	else
		echo "set: $var"
	fi
done
((${#missing[@]} == 0)) || die "not set: ${missing[*]}"
[[ "$KEEP_AZURE_OPENAI_ENDPOINT" == https://* ]] || die "KEEP_AZURE_OPENAI_ENDPOINT must start with https://"

ledger="${KEEP_G2_SPEND_LEDGER:-results/g2-spend-ledger.jsonl}"
mkdir -p results
spend=(--spend-ledger "$ledger" --spend-cap-usd "$max_usd")
# A smoke run covers a few questions, so it is held to --smoke-usd rather
# than reserving the manifest's full-run budget.
[[ -z "$smoke" ]] || spend+=(--max-run-usd "$smoke_usd")
echo
"$bench" spend "${spend[@]}" show

reserve=0
for m in "${manifests[@]}"; do
	usd="$(sed -n 's/^max_cost_usd = \([0-9.]*\)$/\1/p' "$m")"
	[[ -z "$smoke" ]] || usd="$(awk -v a="$usd" -v b="$smoke_usd" 'BEGIN { print (a < b ? a : b) }')"
	reserve="$(awk -v a="$reserve" -v b="$usd" 'BEGIN { print a + b }')"
done
echo
echo "These ${#manifests[@]} run(s) reserve \$$reserve in total, one run at a time, in $ledger."
echo "The harness refuses any run that would take committed spend past \$$max_usd."
if awk -v r="$reserve" -v c="$max_usd" 'BEGIN { exit !(r > c) }'; then
	echo "Warning: the runs' reservations exceed the cap; the later runs will be refused unless earlier ones settle well below it."
fi

read -r -p "Run ${manifests[*]} against Azure now (cap \$$max_usd)? [y/N] " answer
[[ "$answer" == [yY] || "$answer" == [yY][eE][sS] ]] || die "cancelled; nothing was called"

logs="target/g2-logs"
runs=()
for m in "${manifests[@]}"; do
	echo
	echo "== $m"
	log="$logs/$(basename "$m" .toml)-$(date -u +%Y%m%dT%H%M%SZ).log"
	"$bench" run "$m" --allow-model-calls ${unseal[@]+"${unseal[@]}"} ${arm_args[@]+"${arm_args[@]}"} "${spend[@]}" | tee "$log"
	dir="$(sed -n 's/^artifacts: //p' "$log" | tail -n 1)"
	[[ -n "$dir" && -f "$dir/run.json" ]] || die "could not find the artifacts of $m"
	runs+=("$dir")
done

[[ -z "$smoke" ]] || {
	echo
	echo "Smoke run finished: ${runs[*]}"
	exit 0
}

against=()
for b in "${comparators[@]}"; do against+=(--against "$b"); done
require=()
for p in "${profile_list[@]}"; do require+=(--require-profile "$p"); done
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
i=0
for d in "${dataset_list[@]}"; do
	run_args=()
	for _ in "${profile_list[@]}"; do
		run_args+=(--run "${runs[$i]}")
		i=$((i + 1))
	done
	out="results/g2-$phase-$d$suffix-$stamp"
	echo
	echo "== G2 $phase comparison: $d -> $out"
	# Exit code 2 means the gate did not pass (development runs never pass
	# the confirmation-phase check); the report says why.
	status=0
	"$bench" compare "${run_args[@]}" --candidate "$candidate" "${against[@]}" "${require[@]}" --out "$out" || status=$?
	((status == 0 || status == 2)) || die "comparison failed for $d"
	for b in "${reported[@]}"; do
		echo "== Reported (not gated): $candidate vs $b -> $out-vs-$b"
		status=0
		"$bench" compare "${run_args[@]}" --candidate "$candidate" --against "$b" "${require[@]}" --out "$out-vs-$b" || status=$?
		((status == 0 || status == 2)) || die "comparison with $b failed for $d"
	done
done
