#!/usr/bin/env bash
# Set the workspace version, and the keep-* entries in Cargo.lock, to X.Y.Z.
set -euo pipefail
cd "$(dirname "$0")/.."

version=${1#v}
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Usage: scripts/set-version.sh X.Y.Z" >&2; exit 2; }

VERSION=$version perl -0pi -e 's/(\[workspace\.package\]\nversion = ")[^"]+"/${1}$ENV{VERSION}"/ or die "no [workspace.package] version\n"' Cargo.toml
VERSION=$version perl -0pi -e 's/(\nname = "keep-[a-z-]+"\nversion = ")[^"]+"/${1}$ENV{VERSION}"/g' Cargo.lock
