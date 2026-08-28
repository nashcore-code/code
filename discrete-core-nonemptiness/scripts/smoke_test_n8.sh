#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE_MAX_M="${SMOKE_MAX_M:-5}"
if [[ "$SMOKE_MAX_M" != 4 && "$SMOKE_MAX_M" != 5 ]]; then
  echo "ERROR: SMOKE_MAX_M must be 4 or 5" >&2
  exit 2
fi
CLEAN=0
CLEAN_ROOT=""
if [[ $# -eq 0 ]]; then
  # The regeneration driver requires a destination that does not yet exist.
  # Create a private parent, not the destination itself.
  CLEAN_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/discrete-core-n8-smoke.XXXXXX")"
  OUT="$CLEAN_ROOT/output"
  CLEAN=1
elif [[ $# -eq 1 ]]; then
  OUT="$1"
else
  echo "Usage: $0 [NEW_OUTPUT_DIRECTORY]" >&2
  exit 2
fi
cleanup() {
  if [[ "$CLEAN" == 1 ]]; then
    rm -rf -- "$CLEAN_ROOT"
  fi
}
trap cleanup EXIT

MAX_M="$SMOKE_MAX_M" JOBS="${JOBS:-2}" "$ROOT/scripts/regenerate_n8.sh" "$OUT"
python3 - "$OUT/summaries/eight_voter_regeneration.json" "$SMOKE_MAX_M" <<'PY'
import json,sys
p=sys.argv[1]; expected=int(sys.argv[2]); d=json.load(open(p))
assert d['status']=='PASS' and d['max_m']==expected
assert d['levels']['4']['positive_kernels']==4779 and d['levels']['4']['hard_cells']==0
if expected >= 5:
    assert d['levels']['5']['positive_kernels']==56479 and d['levels']['5']['hard_cells']==0
print(f'PASS n=8 smoke expected m=4..{expected} census and zero hard cells')
PY
if [[ "$CLEAN" == 0 ]]; then
  echo "smoke_output=$OUT"
fi
