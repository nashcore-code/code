#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage: scripts/regenerate_n8.sh OUTPUT_DIRECTORY [pipeline options]

Environment shortcuts:
  JOBS=N          worker/thread count (default: detected CPU count)
  CHUNK_SIZE=N    matrices per scan chunk (default: 100000)
  RESUME=1        validate and reuse completed outputs
  MAX_M=4..8      stop after this fractional-column level
  SKIP_GMP=1      development-only: omit the independent GMP replay

Examples:
  scripts/regenerate_n8.sh /tmp/n8-full
  RESUME=1 JOBS=8 scripts/regenerate_n8.sh /tmp/n8-full
  MAX_M=5 scripts/regenerate_n8.sh /tmp/n8-smoke
EOF
  exit 2
fi

exec python3 "$ROOT/scripts/regenerate_n8.py" "$@"
