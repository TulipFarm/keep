#!/bin/bash
# Bring up the quick-start Compose stack for each storage profile under a
# throwaway project name and port, check it end to end over REST, then
# remove its containers and volumes.
#
#   scripts/compose-smoke.sh [postgres-vector] [postgres-qdrant]
#
# KEEP_SMOKE_PORT (default 17357) is the loopback port used for the test.
set -euo pipefail
cd "$(dirname "$0")/.."

profiles=("$@")
[ ${#profiles[@]} -gt 0 ] || profiles=(postgres-vector postgres-qdrant)
port=${KEEP_SMOKE_PORT:-17357}
export KEEP_IMAGE=keep:smoke KEEP_PORT=$port
docker build -q -f deploy/Dockerfile -t "$KEEP_IMAGE" . >/dev/null

check() {
    local profile=$1 project=$2 service=$3
    local dc=(docker compose -p "$project" -f deploy/compose.yaml --profile "$profile")
    trap '"${dc[@]}" down -v --remove-orphans >/dev/null 2>&1 || true' RETURN
    "${dc[@]}" up -d --no-build --wait --wait-timeout 180

    local base="http://127.0.0.1:$port"
    local token
    token=$("${dc[@]}" exec -T "$service" cat /run/keep-secrets/api_token)
    call() {
        curl -sS --fail-with-body -H "authorization: Bearer $token" \
            -H 'content-type: application/json' -d "$2" "$base/v1/$1"
    }

    curl -sS --fail "$base/v1/health/ready" | grep -q '"status":"ready"'
    # Writes need the generated token.
    code=$(curl -s -o /dev/null -w '%{http_code}' -H 'content-type: application/json' \
        -d '{"path":"/topics/x.md","text":"x"}' "$base/v1/memory/write")
    [ "$code" = 401 ] || { echo "unauthenticated write got $code" >&2; return 1; }
    call memory/write '{"path":"/topics/smoke.md","text":"- The smoke test ran.\n"}' >/dev/null
    call memory/read '{"path":"/topics/smoke.md"}' | grep -q 'smoke test ran'
    call knowledge/ingest '{"source_id":"smoke","text":"The quick start uses Compose profiles."}' \
        | grep -q '"state":"complete"'
    call knowledge/search '{"query":"Compose profiles"}' | grep -q '"source_id":"smoke"'
    # Databases publish no host ports.
    if "${dc[@]}" ps --format '{{.Service}} {{.Ports}}' | grep -E '^(postgres|qdrant) .*->' >/dev/null; then
        echo "a database publishes a host port" >&2
        return 1
    fi
    # A restart keeps the data and the generated secrets.
    "${dc[@]}" restart "$service" >/dev/null
    "${dc[@]}" up -d --no-build --wait --wait-timeout 120 >/dev/null
    call memory/read '{"path":"/topics/smoke.md"}' | grep -q 'smoke test ran'
    echo "ok: $profile"
}

for p in "${profiles[@]}"; do
    case $p in
    postgres-vector) check "$p" "keep-smoke-$$-vector" keep-vector ;;
    postgres-qdrant) check "$p" "keep-smoke-$$-qdrant" keep-qdrant ;;
    *) echo "unknown profile $p" >&2; exit 2 ;;
    esac
done
docker image rm "$KEEP_IMAGE" >/dev/null
