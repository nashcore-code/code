#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
ROOT="$(artifact_root)"
CXX="${CXX:-g++}"
JOBS="${JOBS:-5}"
require_command "$CXX"
require_command python3

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
WORK="$1"
require_new_directory "$WORK"
mkdir -p "$WORK/bin" "$WORK/data" "$WORK/logs" "$WORK/chunks"

"$CXX" -O3 -std=c++17 "$ROOT/src/common/enumerate_antichains.cpp" -o "$WORK/bin/enumerate_antichains"
"$CXX" -O3 -fopenmp -std=c++17 "$ROOT/src/n7/enumerate_positive_dual.cpp" -o "$WORK/bin/enumerate_positive_dual"
"$CXX" -O3 -std=c++17 "$ROOT/src/n7/filter_positive_alpha.cpp" -o "$WORK/bin/filter_positive_alpha"
"$CXX" -O3 -std=c++17 "$ROOT/src/common/generate_minimal_holes.cpp" -o "$WORK/bin/generate_minimal_holes"
"$CXX" -O3 -std=c++17 "$ROOT/src/common/filter_holes_subset.cpp" -o "$WORK/bin/filter_holes_subset"
"$CXX" -O3 -std=c++17 "$ROOT/src/common/filter_holes_coverdual.cpp" -o "$WORK/bin/filter_holes_coverdual"
"$CXX" -O3 -std=c++17 "$ROOT/src/common/generate_upper_cells.cpp" -o "$WORK/bin/generate_upper_cells"
"$CXX" -O3 -std=c++17 "$ROOT/src/common/filter_cells_coverdual.cpp" -o "$WORK/bin/filter_cells_coverdual"
"$CXX" -O2 -ftrapv -std=c++17 "$ROOT/src/common/classify_cells_exact.cpp" -o "$WORK/bin/classify_exact"
"$CXX" -O2 -std=c++17 "$ROOT/src/common/classify_cells_bigint.cpp" -o "$WORK/bin/classify_bigint"
"$CXX" -O2 -std=c++17 "$ROOT/src/n7/verify_square_surplus.cpp" -o "$WORK/bin/verify_square"
"$CXX" -O2 -std=c++17 "$ROOT/src/n7/verify_rectangular_surplus.cpp" -o "$WORK/bin/verify_rectangular"

"$WORK/bin/enumerate_antichains" 7 4 6 "$WORK/data/all_m4_m6.txt" 2>"$WORK/logs/enumerate_m4_m6.log"
OMP_NUM_THREADS="$JOBS" "$WORK/bin/enumerate_positive_dual" \
  "$WORK/data/positive_m7.txt" 2>"$WORK/logs/enumerate_m7.log"
"$WORK/bin/filter_positive_alpha" 7 "$WORK/data/all_m4_m6.txt" \
  "$WORK/data/positive_m4_m6.txt" 2>"$WORK/logs/positive_filter.log"
expect_lines 29996 "$WORK/data/positive_m4_m6.txt"
expect_lines 31850 "$WORK/data/positive_m7.txt"
cmp "$WORK/data/positive_m4_m6.txt" "$ROOT/data/n7/positive_antichains_m4_m6.txt"
cmp "$WORK/data/positive_m7.txt" "$ROOT/data/n7/positive_antichains_m7.txt"

cp "$WORK/data/positive_m4_m6.txt" "$WORK/data/positive_all.txt"
cat "$WORK/data/positive_m7.txt" >> "$WORK/data/positive_all.txt"
expect_lines 61846 "$WORK/data/positive_all.txt"

"$WORK/bin/generate_minimal_holes" 7 "$WORK/data/positive_all.txt" \
  "$WORK/data/minimal_holes.txt" 2>"$WORK/logs/minimal_holes.log"
expect_lines 3200557 "$WORK/data/minimal_holes.txt"
"$WORK/bin/filter_holes_subset" 7 "$WORK/data/minimal_holes.txt" \
  "$WORK/data/subset_survivors.txt" 2>"$WORK/logs/subset_filter.log"
expect_lines 888719 "$WORK/data/subset_survivors.txt"
"$WORK/bin/filter_holes_coverdual" 7 "$WORK/data/subset_survivors.txt" \
  "$WORK/data/minimal_cover_survivors.txt" 2>"$WORK/logs/minimal_cover_filter.log"
expect_lines 16854 "$WORK/data/minimal_cover_survivors.txt"
"$WORK/bin/generate_upper_cells" 7 "$WORK/data/minimal_cover_survivors.txt" \
  "$WORK/data/upper_cells.txt" 2>"$WORK/logs/upper_cells.log"
expect_lines 1579161 "$WORK/data/upper_cells.txt"
"$WORK/bin/filter_cells_coverdual" 7 "$WORK/data/upper_cells.txt" \
  "$WORK/data/final_cells.txt" 2>"$WORK/logs/final_cover_filter.log"
expect_lines 30929 "$WORK/data/final_cells.txt"
cmp "$WORK/data/final_cells.txt" "$ROOT/data/n7/final_cells.txt"

"$WORK/bin/classify_exact" 7 "$WORK/data/final_cells.txt" \
  "$WORK/data/bad_exact.txt" "$WORK/data/surplus_exact.txt" 2>"$WORK/logs/classify_exact.log"
"$WORK/bin/classify_bigint" 7 "$WORK/data/final_cells.txt" \
  "$WORK/data/bad_bigint.txt" "$WORK/data/surplus_bigint.txt" 2>"$WORK/logs/classify_bigint.log"
cmp "$WORK/data/bad_exact.txt" "$WORK/data/bad_bigint.txt"
cmp "$WORK/data/surplus_exact.txt" "$WORK/data/surplus_bigint.txt"
expect_empty "$WORK/data/bad_exact.txt"
expect_lines 298 "$WORK/data/surplus_exact.txt"
cmp "$WORK/data/surplus_exact.txt" "$ROOT/data/n7/surplus_cells.txt"

python3 "$ROOT/src/n7/verify_puncture_certificates.py" "$ROOT/data/n7/puncture_certificates.json"
"$WORK/bin/verify_square" "$WORK/data/surplus_exact.txt" 2>"$WORK/logs/verify_square.log"
"$WORK/bin/verify_rectangular" "$WORK/data/surplus_exact.txt" 2>"$WORK/logs/verify_rectangular.log"

echo "PASS regenerate n=7 kernels=61846 cells=30929 surplus=298 bad=0"
