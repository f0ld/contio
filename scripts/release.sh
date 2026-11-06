#!/usr/bin/env bash
# Cut a release: set the version everywhere, commit, tag. Pushing the tag
# makes GitHub Actions build the AppImage and publish the release
# (.github/workflows/release.yml).
#
#   scripts/release.sh 0.4.0          # then: git push origin main v0.4.0
set -euo pipefail
cd "$(dirname "$0")/.."

v="${1:?usage: scripts/release.sh <version, e.g. 0.4.0>}"
[[ "$v" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "version must look like 0.4.0" >&2; exit 1; }
[[ "$(git branch --show-current)" == main ]] || { echo "release from main" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "commit or stash your changes first" >&2; exit 1; }
git rev-parse -q --verify "refs/tags/v$v" >/dev/null && { echo "tag v$v exists" >&2; exit 1; }

# the four places that carry the version
sed -i -E "0,/^version = \".*\"/s//version = \"$v\"/" Cargo.toml apps/gui/src-tauri/Cargo.toml
sed -i -E "0,/\"version\": \".*\"/s//\"version\": \"$v\"/" apps/gui/src-tauri/tauri.conf.json apps/gui/package.json
(cd apps/gui && npm install --package-lock-only --silent)
cargo check --workspace --quiet

git add Cargo.toml Cargo.lock apps/gui/src-tauri/Cargo.toml apps/gui/src-tauri/tauri.conf.json apps/gui/package.json apps/gui/package-lock.json
git commit -q -m "Release v$v"
git tag -a "v$v" -m "contio v$v"
echo "tagged v$v; publish with: git push origin main v$v"
