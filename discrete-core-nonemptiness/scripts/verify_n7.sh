#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
ROOT="$(artifact_root)"
CXX="${CXX:-g++}"
require_command "$CXX"
require_command python3

WORK="$(mktemp -d "${TMPDIR:-/tmp}/discrete-core-n7-verify.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

"$CXX" -O2 -ftrapv -std=c++17 \
  "$ROOT/src/common/classify_cells_exact.cpp" -o "$WORK/classify_exact"
"$CXX" -O2 -std=c++17 \
  "$ROOT/src/common/classify_cells_bigint.cpp" -o "$WORK/classify_bigint"
"$CXX" -O2 -std=c++17 \
  "$ROOT/src/n7/verify_square_surplus.cpp" -o "$WORK/verify_square"
"$CXX" -O2 -std=c++17 \
  "$ROOT/src/n7/verify_rectangular_surplus.cpp" -o "$WORK/verify_rectangular"

expect_lines 29996 "$ROOT/data/n7/positive_antichains_m4_m6.txt"
expect_lines 31850 "$ROOT/data/n7/positive_antichains_m7.txt"
expect_lines 30929 "$ROOT/data/n7/final_cells.txt"
expect_lines 298 "$ROOT/data/n7/surplus_cells.txt"

"$WORK/classify_exact" 7 "$ROOT/data/n7/final_cells.txt" \
  "$WORK/bad_exact.txt" "$WORK/surplus_exact.txt" 2>"$WORK/exact.log"
"$WORK/classify_bigint" 7 "$ROOT/data/n7/final_cells.txt" \
  "$WORK/bad_bigint.txt" "$WORK/surplus_bigint.txt" 2>"$WORK/bigint.log"

cmp "$WORK/bad_exact.txt" "$WORK/bad_bigint.txt"
cmp "$WORK/surplus_exact.txt" "$WORK/surplus_bigint.txt"
cmp "$WORK/bad_exact.txt" "$ROOT/data/n7/bad_cells.empty"
cmp "$WORK/surplus_exact.txt" "$ROOT/data/n7/surplus_cells.txt"
expect_empty "$WORK/bad_exact.txt"

python3 "$ROOT/src/n7/verify_puncture_certificates.py" \
  "$ROOT/data/n7/puncture_certificates.json"
"$WORK/verify_square" "$ROOT/data/n7/surplus_cells.txt" 2>"$WORK/square.log"
"$WORK/verify_rectangular" "$ROOT/data/n7/surplus_cells.txt" 2>"$WORK/rectangular.log"

echo "PASS n=7 antichains=61846 cells=30929 surplus=298 bad=0"

