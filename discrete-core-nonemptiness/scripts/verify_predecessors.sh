#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$ROOT/scripts/verify_manifest.py" "$ROOT/SHA256SUMS"
"$ROOT/scripts/verify_n6.sh"
"$ROOT/scripts/verify_n7.sh"
echo "PASS predecessor_modules=n6,n7"

