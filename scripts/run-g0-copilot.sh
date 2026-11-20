#!/usr/bin/env bash
# Run the internal G0 go/no-go gate through the local GitHub Copilot CLI.
#
# These runs are NON-CANONICAL: the CLI adds a hidden system prompt and does
# not pin the upstream model. The comparison is written with
# --allow-non-canonical and stamped INTERNAL / NON-CANONICAL. No API key or
# env file is needed; the signed-in `copilot` CLI makes the calls and is
# billed in AI credits. Prints both dry-run estimates and asks for
# confirmation before any model call. See docs/benchmarks/running-g0.md.
#
# The harness cannot resume a run: a manifest that is run again starts
# over. To skip a manifest that already finished, point
# KEEP_G0_LOCOMO_RUN or KEEP_G0_LONGMEMEVAL_RUN at its results directory.
#
# KEEP_G0_VERSION picks the manifests: 5 (default) runs the v5 manifests
# with keep-mk5 and answer-v5 controls. 6 runs
# manifests/g0/*-copilot-v6.toml, whose candidate is keep-mk6 (keep-mk4
# answered with answer-v6, which writes working notes before its final
# answer line) compared with dense-6k-v6, full-context-v6, keep-mk5 and
# keep-mk4; its judge verdicts are cached and borderline answers get a
# majority of three votes. 7 runs manifests/g0/*-copilot-v7.toml, whose
# candidate is keep-mk7 (keep-mk6 answered with answer-v7, which adds
# world-knowledge bridging, answers at the level of detail asked and
# reasoned inferences) compared with dense-6k-v7, full-context-v7,
# full-context-v6, keep-mk6, keep-mk5 and keep-mk4. 4 runs
# manifests/g0/*-copilot-v4.toml, whose candidate is keep-mk4 (mk3 with
# subject grounding, session summaries and answer-v4); it also compares it
# with keep-mk3 and keep-mk2 and with dense-6k-v4 and full-context-v4, the
# baselines answered with the same answer-v4 prompt. Its run extracts facts
# and writes a summary for every session not yet in the cache before
# building contexts; those calls count against the same
# budget.max_ai_credits and the dry run prices them. 3 reruns
# manifests/g0/*-copilot-v3.toml with keep-mk3 as the candidate (compared
# with keep-mk2 and keep-mk1); 2 reruns manifests/g0/*-copilot-v2.toml with
# keep-mk2 as the candidate; 1 reruns the original
# manifests/g0/*-copilot.toml with keep-mk1 as the candidate.
set -euo pipefail

cd "$(dirname "$0")/.."

version="${KEEP_G0_VERSION:-5}"
baselines=(full-context recent-window bm25 dense bm25-6k dense-6k notes-grep)
case "$version" in
1)
	suffix=""
	candidate="${KEEP_G0_CANDIDATE:-keep-mk1}"
	;;
2)
	suffix="-v2"
	candidate="${KEEP_G0_CANDIDATE:-keep-mk2}"
	baselines+=(keep-mk1)
	;;
3)
	suffix="-v3"
	candidate="${KEEP_G0_CANDIDATE:-keep-mk3}"
	baselines+=(keep-mk2 keep-mk1)
	;;
4)
	suffix="-v4"
	candidate="${KEEP_G0_CANDIDATE:-keep-mk4}"
	baselines+=(dense-6k-v4 full-context-v4 keep-mk3 keep-mk2)
	;;
5)
	suffix="-v5"
	candidate="${KEEP_G0_CANDIDATE:-keep-mk5}"
	baselines+=(dense-6k-v5 full-context-v5 keep-mk4 keep-mk3 keep-mk2)
	;;
6)
	suffix="-v6"
	candidate="${KEEP_G0_CANDIDATE:-keep-mk6}"
	baselines+=(dense-6k-v6 full-context-v6 keep-mk5 keep-mk4 keep-mk3)
	;;
7)
	suffix="-v7"
	candidate="${KEEP_G0_CANDIDATE:-keep-mk7}"
	baselines+=(dense-6k-v7 full-context-v7 full-context-v6 keep-mk6 keep-mk5 keep-mk4)
	;;
*)
	echo "run-g0-copilot: KEEP_G0_VERSION must be 1, 2, 3, 4, 5, 6 or 7, not $version" >&2
	exit 1
	;;
esac
datasets=(locomo longmemeval-s)
prior_runs=("${KEEP_G0_LOCOMO_RUN:-}" "${KEEP_G0_LONGMEMEVAL_RUN:-}")

die() {
	echo "run-g0-copilot: $*" >&2
	exit 1
}

copilot_bin="${KEEP_COPILOT_BIN:-copilot}"
command -v "$copilot_bin" >/dev/null || die "$copilot_bin not found on PATH; install the GitHub Copilot CLI and sign in"

missing=()
for var in KEEP_POSTGRES_URL KEEP_QDRANT_URL; do
	if [[ -z "$(printenv "$var" || true)" ]]; then
		missing+=("$var")
	else
		echo "set: $var"
	fi
done
((${#missing[@]} == 0)) || die "not set: ${missing[*]} (see docs/benchmarks/running-g0.md)"

export KEEP_MODEL_CACHE="${KEEP_MODEL_CACHE:-$PWD/.keep-models}"
export KEEP_COPILOT_WORKDIR="${KEEP_COPILOT_WORKDIR:-$PWD/target/copilot-work}"
mkdir -p "$KEEP_COPILOT_WORKDIR" results

for i in "${!datasets[@]}"; do
	run="${prior_runs[$i]}"
	[[ -z "$run" || -f "$run/run.json" ]] || die "$run has no run.json"
done

cargo build --release --locked -p keep-bench --features local-embeddings
bench=target/release/keep-bench

for d in "${datasets[@]}"; do
	"$bench" validate-manifest "manifests/g0/$d-copilot$suffix.toml"
done
for i in "${!datasets[@]}"; do
	[[ -z "${prior_runs[$i]}" ]] || continue
	echo
	"$bench" dry-run "manifests/g0/${datasets[$i]}-copilot$suffix.toml"
done

echo
echo "Each run stops once its manifest's budget.max_ai_credits is reached."
read -r -p "Run the internal Copilot G0 now (spends AI credits)? [y/N] " answer
[[ "$answer" == [yY] || "$answer" == [yY][eE][sS] ]] || die "cancelled; nothing was called"

runs=()
for i in "${!datasets[@]}"; do
	d="${datasets[$i]}"
	if [[ -n "${prior_runs[$i]}" ]]; then
		echo "== reusing ${prior_runs[$i]} for $d"
		runs+=("${prior_runs[$i]}")
		continue
	fi
	echo
	echo "== manifests/g0/$d-copilot$suffix.toml"
	log="$KEEP_COPILOT_WORKDIR/run-$d.log"
	"$bench" run "manifests/g0/$d-copilot$suffix.toml" --allow-model-calls | tee "$log"
	dir="$(sed -n 's/^artifacts: //p' "$log" | tail -n 1)"
	[[ -n "$dir" && -f "$dir/run.json" ]] || die "could not find the artifacts of the $d run"
	runs+=("$dir")
done

against=()
for b in "${baselines[@]}"; do against+=(--against "$b"); done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
for i in "${!datasets[@]}"; do
	out="results/internal-g0-copilot-${datasets[$i]}$suffix-$stamp"
	echo
	echo "== internal comparison: ${datasets[$i]} -> $out"
	# Exit code 2 means the gate did not pass, which an internal comparison
	# never does; the tables are what matter.
	status=0
	"$bench" compare --allow-non-canonical --run "${runs[$i]}" \
		--candidate "$candidate" "${against[@]}" --out "$out" || status=$?
	((status == 0 || status == 2)) || die "comparison failed for ${datasets[$i]}"
done
