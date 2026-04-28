#!/usr/bin/env bash
# Build the three Vite sub-apps into apps/static/<name>/ so the unified
# FastAPI app can serve them.
#
# Run from the ai-recruiter root:
#     bash scripts/build_frontends.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$ROOT"

build_one() {
  local name="$1"
  local dir="apps/$name"
  echo "[$name] installing deps in $dir"
  (cd "$dir" && npm ci)
  echo "[$name] building → apps/static/$name"
  (cd "$dir" && npm run build)
}

build_one screening
build_one interview
build_one behavior
echo "[ok] all frontends built into apps/static/"
