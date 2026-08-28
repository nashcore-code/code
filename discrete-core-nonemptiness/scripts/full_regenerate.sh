#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
TARGET="$1"
if [[ -e "$TARGET" ]]; then
  echo "ERROR: destination already exists: $TARGET" >&2
  exit 2
fi
mkdir -p "$TARGET"

"$ROOT/scripts/regenerate_n6.sh" "$TARGET/n6"
"$ROOT/scripts/regenerate_n7.sh" "$TARGET/n7"
"$ROOT/scripts/regenerate_n8.sh" "$TARGET/n8"
echo "PASS full_regeneration=n6,n7,n8"
