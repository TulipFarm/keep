#!/bin/sh
# Builds Keep's connection settings from the generated secrets, then runs
# `keep`. Values already set in the environment win, so the image also
# works outside this Compose file.
set -eu
secrets=${KEEP_SECRETS_DIR:-/run/keep-secrets}
read_secret() { cat "$secrets/$1"; }

if [ -d "$secrets" ]; then
    : "${KEEP_API_TOKEN:=$(read_secret api_token)}"
    export KEEP_API_TOKEN
    case "${KEEP_STORAGE:-in-memory}" in
    postgres-*)
        # Plain TCP is acceptable here: Postgres is reachable only on the
        # Compose project's private network, never on a host port.
        : "${KEEP_POSTGRES_URL:=postgres://keep:$(read_secret postgres_password)@postgres:5432/keep?sslmode=disable}"
        export KEEP_POSTGRES_URL
        ;;
    esac
    if [ "${KEEP_STORAGE:-}" = postgres-qdrant ]; then
        : "${KEEP_QDRANT_URL:=http://qdrant:6333}"
        : "${KEEP_QDRANT_API_KEY:=$(read_secret qdrant_api_key)}"
        export KEEP_QDRANT_URL KEEP_QDRANT_API_KEY
    fi
fi
exec keep "$@"
