#!/bin/bash
# Run the storage conformance and server suites against Postgres and Qdrant
# that require TLS with a private CA, the way a managed remote database
# does: `sslmode=verify-full`, `https://` Qdrant with an API key.
#
#   scripts/remote-profile-test.sh            # throwaway TLS databases
#
# To test real remote endpoints instead, skip this script and set
# KEEP_TEST_POSTGRES_URL (with sslmode=verify-full and, for a private CA,
# sslrootcert=...), KEEP_TEST_QDRANT_URL (https://), KEEP_TEST_QDRANT_API_KEY
# and KEEP_TEST_QDRANT_CA, then run `cargo test -p keep-ledger -p
# keep-knowledge -p keep-server`.
#
# Uses its own Compose project and ports (KEEP_TLS_PG_PORT, default 25433;
# KEEP_TLS_QDRANT_PORT, default 26333) and removes everything afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

pg_port=${KEEP_TLS_PG_PORT:-25433}
q_port=${KEEP_TLS_QDRANT_PORT:-26333}
project="keep-tls-test-$$"
work="$PWD/target/remote-tls-$$"
mkdir -p "$work"
password=$(openssl rand -hex 16)
qdrant_key=$(openssl rand -hex 16)

cleanup() {
    docker compose -p "$project" -f "$work/compose.yaml" down -v --remove-orphans >/dev/null 2>&1 || true
    rm -rf "$work"
}
trap cleanup EXIT

# A private CA and a server certificate for localhost.
openssl req -x509 -newkey rsa:2048 -nodes -days 2 -subj "/CN=keep test CA" \
    -keyout "$work/ca.key" -out "$work/ca.pem" 2>/dev/null
openssl req -newkey rsa:2048 -nodes -subj "/CN=localhost" \
    -keyout "$work/server.key" -out "$work/server.csr" 2>/dev/null
printf 'subjectAltName=DNS:localhost\nbasicConstraints=CA:FALSE\n' >"$work/ext.cnf"
openssl x509 -req -in "$work/server.csr" -CA "$work/ca.pem" -CAkey "$work/ca.key" \
    -CAcreateserial -days 2 -extfile "$work/ext.cnf" -out "$work/server.pem" 2>/dev/null
chmod 0644 "$work/server.key"

cat >"$work/compose.yaml" <<YAML
services:
  postgres:
    image: pgvector/pgvector:0.8.2-pg18@sha256:42e7f6b4e1eceb02ff14e3e6bc6108bbe259abbe83879dc1845d0da1ddeb555d
    environment:
      POSTGRES_USER: keep
      POSTGRES_PASSWORD: "$password"
      POSTGRES_DB: keep
      POSTGRES_HOST_AUTH_METHOD: scram-sha-256
      POSTGRES_INITDB_ARGS: --auth-host=scram-sha-256
    volumes:
      - $work:/certs:ro
    # Copy the key so it can be owned by postgres with mode 0600, and
    # accept only TLS connections from outside the container.
    entrypoint: ["/bin/bash", "-c"]
    command:
      - |
        install -o postgres -g postgres -m 0600 /certs/server.key /var/lib/postgresql/server.key
        install -o postgres -g postgres -m 0644 /certs/server.pem /var/lib/postgresql/server.pem
        cat > /docker-entrypoint-initdb.d/hba.sh <<'SH'
        echo 'local all all trust' > "\$\$PGDATA/pg_hba.conf"
        echo 'hostssl all all all scram-sha-256' >> "\$\$PGDATA/pg_hba.conf"
        SH
        exec docker-entrypoint.sh postgres -c ssl=on \
          -c ssl_cert_file=/var/lib/postgresql/server.pem \
          -c ssl_key_file=/var/lib/postgresql/server.key
    ports: ["127.0.0.1:$pg_port:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U keep -d keep -h 127.0.0.1"]
      interval: 2s
      retries: 30
  qdrant:
    image: qdrant/qdrant:v1.20.0@sha256:9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2
    environment:
      QDRANT__SERVICE__ENABLE_TLS: "true"
      QDRANT__SERVICE__API_KEY: "$qdrant_key"
      QDRANT__TLS__CERT: /certs/server.pem
      QDRANT__TLS__KEY: /certs/server.key
    volumes:
      - $work:/certs:ro
    ports: ["127.0.0.1:$q_port:6333"]
YAML

docker compose -p "$project" -f "$work/compose.yaml" up -d --wait --wait-timeout 120 >/dev/null
for _ in $(seq 1 30); do
    curl -sf --cacert "$work/ca.pem" "https://localhost:$q_port/readyz" >/dev/null && break
    sleep 1
done

# Plaintext must be refused by the server, so a passing run proves TLS.
if command -v psql >/dev/null &&
    psql "postgres://keep:$password@localhost:$pg_port/keep?sslmode=disable" -c 'select 1' >/dev/null 2>&1; then
    echo "the test Postgres accepted a plaintext connection" >&2
    exit 1
fi

export KEEP_TEST_POSTGRES_URL="postgres://keep:$password@localhost:$pg_port/keep?sslmode=verify-full&sslrootcert=$work/ca.pem"
export KEEP_TEST_QDRANT_URL="https://localhost:$q_port"
export KEEP_TEST_QDRANT_API_KEY="$qdrant_key"
export KEEP_TEST_QDRANT_CA="$work/ca.pem"
export KEEP_TEST_POSTGRES_CA="$work/ca.pem"

cargo test --locked -p keep-postgres -p keep-qdrant -p keep-ledger -p keep-knowledge -p keep-server -- --test-threads=4
echo "ok: suites passed against TLS-only Postgres (verify-full) and HTTPS Qdrant"
