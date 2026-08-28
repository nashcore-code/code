#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
ROOT="$(artifact_root)"
CXX="${CXX:-g++}"
require_command "$CXX"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/discrete-core-n6-verify.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

"$CXX" -O2 -ftrapv -std=c++17 \
  "$ROOT/src/common/classify_cells_exact.cpp" -o "$WORK/classify_exact"
"$CXX" -O2 -std=c++17 \
  "$ROOT/src/common/classify_cells_bigint.cpp" -o "$WORK/classify_bigint"

expect_lines 2153 "$ROOT/data/n6/antichains.txt"
expect_lines 46 "$ROOT/data/n6/final_cells.txt"

"$WORK/classify_exact" 6 "$ROOT/data/n6/final_cells.txt" \
  "$WORK/bad_exact.txt" "$WORK/surplus_exact.txt" 2>"$WORK/exact.log"
"$WORK/classify_bigint" 6 "$ROOT/data/n6/final_cells.txt" \
  "$WORK/bad_bigint.txt" "$WORK/surplus_bigint.txt" 2>"$WORK/bigint.log"

cmp "$WORK/bad_exact.txt" "$WORK/bad_bigint.txt"
cmp "$WORK/surplus_exact.txt" "$WORK/surplus_bigint.txt"
cmp "$WORK/bad_exact.txt" "$ROOT/data/n6/bad_cells.empty"
cmp "$WORK/surplus_exact.txt" "$ROOT/data/n6/surplus_cells.empty"
expect_empty "$WORK/bad_exact.txt"
expect_empty "$WORK/surplus_exact.txt"

echo "PASS n=6 antichains=2153 cells=46 bad=0 surplus=0"

