#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
ROOT="$(artifact_root)"
CXX="${CXX:-g++}"
require_command "$CXX"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
WORK="$1"
require_new_directory "$WORK"
mkdir -p "$WORK/bin" "$WORK/data" "$WORK/logs"

compile() {
  "$CXX" -O3 -std=c++17 "$ROOT/src/common/enumerate_antichains.cpp" -o "$WORK/bin/enumerate_antichains"
  "$CXX" -O3 -std=c++17 "$ROOT/src/common/generate_minimal_holes.cpp" -o "$WORK/bin/generate_minimal_holes"
  "$CXX" -O3 -std=c++17 "$ROOT/src/common/filter_holes_subset.cpp" -o "$WORK/bin/filter_holes_subset"
  "$CXX" -O3 -std=c++17 "$ROOT/src/common/filter_holes_coverdual.cpp" -o "$WORK/bin/filter_holes_coverdual"
  "$CXX" -O3 -std=c++17 "$ROOT/src/common/generate_upper_cells.cpp" -o "$WORK/bin/generate_upper_cells"
  "$CXX" -O3 -std=c++17 "$ROOT/src/common/filter_cells_coverdual.cpp" -o "$WORK/bin/filter_cells_coverdual"
  "$CXX" -O2 -ftrapv -std=c++17 "$ROOT/src/common/classify_cells_exact.cpp" -o "$WORK/bin/classify_exact"
  "$CXX" -O2 -std=c++17 "$ROOT/src/common/classify_cells_bigint.cpp" -o "$WORK/bin/classify_bigint"
}

compile

"$WORK/bin/enumerate_antichains" 6 4 6 "$WORK/data/antichains.txt" 2>"$WORK/logs/enumerate.log"
expect_lines 2153 "$WORK/data/antichains.txt"
awk '$1==4 {n++} END {exit n==279 ? 0 : 1}' "$WORK/data/antichains.txt"
awk '$1==5 {n++} END {exit n==712 ? 0 : 1}' "$WORK/data/antichains.txt"
awk '$1==6 {n++} END {exit n==1162 ? 0 : 1}' "$WORK/data/antichains.txt"
cmp "$WORK/data/antichains.txt" "$ROOT/data/n6/antichains.txt"

"$WORK/bin/generate_minimal_holes" 6 "$WORK/data/antichains.txt" \
  "$WORK/data/minimal_holes.txt" 2>"$WORK/logs/minimal_holes.log"
expect_lines 32286 "$WORK/data/minimal_holes.txt"

"$WORK/bin/filter_holes_subset" 6 "$WORK/data/minimal_holes.txt" \
  "$WORK/data/subset_survivors.txt" 2>"$WORK/logs/subset_filter.log"
expect_lines 6053 "$WORK/data/subset_survivors.txt"

"$WORK/bin/filter_holes_coverdual" 6 "$WORK/data/subset_survivors.txt" \
  "$WORK/data/minimal_cover_survivors.txt" 2>"$WORK/logs/minimal_cover_filter.log"
expect_lines 46 "$WORK/data/minimal_cover_survivors.txt"

"$WORK/bin/generate_upper_cells" 6 "$WORK/data/minimal_cover_survivors.txt" \
  "$WORK/data/upper_cells.txt" 2>"$WORK/logs/upper_cells.log"
expect_lines 597 "$WORK/data/upper_cells.txt"

"$WORK/bin/filter_cells_coverdual" 6 "$WORK/data/upper_cells.txt" \
  "$WORK/data/final_cells.txt" 2>"$WORK/logs/final_cover_filter.log"
expect_lines 46 "$WORK/data/final_cells.txt"
cmp "$WORK/data/final_cells.txt" "$ROOT/data/n6/final_cells.txt"

"$WORK/bin/classify_exact" 6 "$WORK/data/final_cells.txt" \
  "$WORK/data/bad_exact.txt" "$WORK/data/surplus_exact.txt" 2>"$WORK/logs/classify_exact.log"
"$WORK/bin/classify_bigint" 6 "$WORK/data/final_cells.txt" \
  "$WORK/data/bad_bigint.txt" "$WORK/data/surplus_bigint.txt" 2>"$WORK/logs/classify_bigint.log"
cmp "$WORK/data/bad_exact.txt" "$WORK/data/bad_bigint.txt"
cmp "$WORK/data/surplus_exact.txt" "$WORK/data/surplus_bigint.txt"
expect_empty "$WORK/data/bad_exact.txt"
expect_empty "$WORK/data/surplus_exact.txt"

echo "PASS regenerate n=6 antichains=2153 targets=32286 cells=46 bad=0 surplus=0"

