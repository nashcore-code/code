#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
ROOT="$(artifact_root)"
CXX="${CXX:-g++}"
require_command "$CXX"
require_command python3

WORK="$(mktemp -d "${TMPDIR:-/tmp}/discrete-core-n8-available.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

"$CXX" -O2 -std=c++20 "$ROOT/src/n8/binary_format_selftest.cpp" -o "$WORK/format_check"
"$CXX" -O2 -std=c++20 "$ROOT/src/n8/m8_record_checker.cpp" -o "$WORK/record_check"
"$CXX" -O2 -std=c++20 -pthread -DMM=4 \
  "$ROOT/src/n8/eight_row_floor_cell_scanner_template.cpp" -o "$WORK/scan4"
"$CXX" -O2 -std=c++20 -pthread -DMM=5 \
  "$ROOT/src/n8/eight_row_floor_cell_scanner_template.cpp" -o "$WORK/scan5"
"$CXX" -O2 -std=c++20 "$ROOT/src/n8/m6_exact_certificate_checker.cpp" \
  -lgmpxx -lgmp -o "$WORK/check_m6"
"$CXX" -O2 -DMM=7 -std=c++20 "$ROOT/src/n8/m8_certificate_proposer.cpp" \
  -o "$WORK/propose_m7"
"$CXX" -O2 -DMM=7 -std=c++20 "$ROOT/src/n8/exact_cps_fullsat_checker.cpp" \
  -o "$WORK/check_m7"

"$WORK/format_check" > "$WORK/format.log"
grep -q 'PASS hard_record_size=24 certificate_record_size=64' "$WORK/format.log"

python3 - "$ROOT" <<'PY'
from pathlib import Path
import struct, sys
root=Path(sys.argv[1])
expected={4:(5060,4779),5:(69814,56479)}
def key_count(p):
    raw=p.read_bytes()
    if len(raw)<8: raise SystemExit(f'truncated {p}')
    n=struct.unpack_from('<Q',raw,0)[0]
    if len(raw)!=8+8*n: raise SystemExit(f'bad key length {p}')
    return n
def hard_count(p):
    raw=p.read_bytes()
    if len(raw)<16: raise SystemExit(f'truncated {p}')
    magic,n=struct.unpack_from('<QQ',raw,0)
    if magic!=0x3843454c4c533031 or len(raw)!=16+24*n: raise SystemExit(f'bad hard file {p}')
    return n
for m,(full,pos) in expected.items():
    d=root/f'data/n8/m{m}'
    all_name='n8m4_all.bin' if m==4 else 'n8m5_all_direct.bin'
    if key_count(d/all_name)!=full: raise SystemExit(f'm={m} direct count mismatch')
    if key_count(d/f'n8m{m}_pos.bin')!=pos: raise SystemExit(f'm={m} positive count mismatch')
    if hard_count(d/f'n8m{m}_hard.bin')!=0: raise SystemExit(f'm={m} hard file not empty')
if (root/'data/n8/m5/n8m5_pos_direct.bin').read_bytes() != (root/'data/n8/m5/n8m5_pos_augmented.bin').read_bytes():
    raise SystemExit('m=5 direct and augmented lists differ')
print('PASS m=4,5 canonical counts and direct/augmentation agreement')
PY

"$WORK/scan4" "$ROOT/data/n8/m4/n8m4_pos.bin" "$WORK/m4_hard.bin" 1 \
  > "$WORK/m4_scan.log" 2> "$WORK/m4_scan.err"
cmp "$WORK/m4_hard.bin" "$ROOT/data/n8/m4/n8m4_hard.bin"
grep -q 'cell_feasible=22 unresolved_by_tight=0' "$WORK/m4_scan.log"

"$WORK/scan5" "$ROOT/data/n8/m5/n8m5_pos.bin" "$WORK/m5_hard.bin" 1 \
  > "$WORK/m5_scan.log" 2> "$WORK/m5_scan.err"
cmp "$WORK/m5_hard.bin" "$ROOT/data/n8/m5/n8m5_hard.bin"
[[ "$(grep -o 'cell_feasible=42 unresolved_by_tight=0' "$WORK/m5_scan.log" | wc -l | tr -d ' ')" == "2" ]]

"$WORK/record_check" "$ROOT/data/n8/m4/n8m4_hard.bin" \
  "$ROOT/data/n8/m4/cps4_all.bin" "$ROOT/data/n8/m4/cps4_fail.bin" > "$WORK/m4_records.log"
"$WORK/record_check" "$ROOT/data/n8/m5/n8m5_hard.bin" \
  "$ROOT/data/n8/m5/cps5_all.bin" "$ROOT/data/n8/m5/cps5_fail.bin" > "$WORK/m5_records.log"

"$WORK/check_m6" "$ROOT/data/n8/m6/n8m6_hard_exact.bin" \
  "$ROOT/data/n8/m6/m6_certificates_exact.bin" > "$WORK/m6.log"
grep -q 'PASS certs=168 fixed=163 adaptive=5' "$WORK/m6.log"

# The proposer is deliberately non-authoritative. Simplex pivoting and
# floating-point tie-breaking may select a different valid puncture on another
# platform. Verify stored and regenerated streams by hard-record coverage and
# exact replay; do not compare committee choices.
"$WORK/record_check" "$ROOT/data/n8/m7/n8m7_unresolved_exact_ll.bin" \
  "$ROOT/data/n8/m7/cps7_fullsat_all.bin" "$ROOT/data/n8/m7/cps7_fail_zero.bin" \
  > "$WORK/m7_deposited_records.log"
"$WORK/check_m7" "$ROOT/data/n8/m7/cps7_fullsat_all.bin" \
  > "$WORK/m7_deposited_exact.log" 2> "$WORK/m7_deposited_exact.err"
python3 - "$WORK/m7_deposited_exact.log" <<'PYDEPOSIT'
import re, sys
hits=[]
for line in open(sys.argv[1], encoding='utf-8'):
    m=re.match(r"PASS certs=(\d+) fixed=(\d+) adaptive=(\d+)\b", line)
    if m: hits.append(tuple(map(int,m.groups())))
if len(hits)!=1:
    raise SystemExit(f"expected one deposited exact PASS line, found {len(hits)}")
certs,fixed,adaptive=hits[0]
if certs!=36128 or fixed+adaptive!=certs:
    raise SystemExit(f"invalid deposited exact replay census: {hits[0]}")
print(f"PASS m=7 deposited exact validity certs={certs} fixed={fixed} adaptive={adaptive}")
PYDEPOSIT

"$WORK/propose_m7" "$ROOT/data/n8/m7/n8m7_unresolved_exact_ll.bin" \
  "$WORK/cps7_regenerated.bin" "$WORK/cps7_regenerated_fail.bin" \
  > "$WORK/m7_proposer.log" 2> "$WORK/m7_proposer.err"
python3 - "$WORK/m7_proposer.log" <<'PY'
import re, sys
hits=[]
for line in open(sys.argv[1], encoding='utf-8'):
    m=re.fullmatch(r"records=(\d+) fixed=(\d+) adaptive=(\d+) failures=(\d+)\n?", line)
    if m: hits.append(tuple(map(int,m.groups())))
if len(hits)!=1:
    raise SystemExit(f"expected one proposer census line, found {len(hits)}")
records,fixed,adaptive,failures=hits[0]
if records!=36128 or failures!=0 or fixed+adaptive!=records:
    raise SystemExit(f"invalid proposer census: {hits[0]}")
print(f"PASS m=7 proposal coverage records={records} fixed={fixed} adaptive={adaptive}")
PY
"$WORK/record_check" "$ROOT/data/n8/m7/n8m7_unresolved_exact_ll.bin" \
  "$WORK/cps7_regenerated.bin" "$WORK/cps7_regenerated_fail.bin" \
  > "$WORK/m7_regenerated_records.log"
"$WORK/check_m7" "$WORK/cps7_regenerated.bin" \
  > "$WORK/m7_regenerated_exact.log" 2> "$WORK/m7_regenerated_exact.err"
python3 - "$WORK/m7_regenerated_exact.log" <<'PY'
import re, sys
hits=[]
for line in open(sys.argv[1], encoding='utf-8'):
    m=re.match(r"PASS certs=(\d+) fixed=(\d+) adaptive=(\d+)\b", line)
    if m: hits.append(tuple(map(int,m.groups())))
if len(hits)!=1:
    raise SystemExit(f"expected one exact PASS line, found {len(hits)}")
certs,fixed,adaptive=hits[0]
if certs!=36128 or fixed+adaptive!=certs:
    raise SystemExit(f"invalid exact replay census: {hits[0]}")
print(f"PASS m=7 regenerated exact validity certs={certs} fixed={fixed} adaptive={adaptive}")
PY

# The checker itself must reject both rank-deficient and full-rank-but-empty
# positive-dual domains, before any vacuous implication can be accepted.
python3 - "$ROOT/data/n8/m7/cps7_fullsat_all.bin" \
  "$WORK/cps7_rank_deficient.bin" "$WORK/cps7_empty_dual.bin" <<'PY'
from pathlib import Path
import struct, sys
src=Path(sys.argv[1]).read_bytes()
if len(src)<72:
    raise SystemExit('truncated source certificate stream')
record=src[8:72]
rank_bad=bytearray(struct.pack('<Q',1)+record)
struct.pack_into('<Q',rank_bad,8,0)
Path(sys.argv[2]).write_bytes(rank_bad)
# Key 53221801103618419 is a rank-7 antichain, but A^T alpha=1 has no
# strictly positive solution (its line collapses against a positivity boundary).
empty_dual=bytearray(struct.pack('<Q',1)+record)
struct.pack_into('<Q',empty_dual,8,53221801103618419)
Path(sys.argv[3]).write_bytes(empty_dual)
PY
for bad in rank_deficient empty_dual; do
  if "$WORK/check_m7" "$WORK/cps7_${bad}.bin" \
      > "$WORK/m7_${bad}.out" 2> "$WORK/m7_${bad}.err"; then
    echo "ERROR: m=7 checker accepted invalid dual-domain certificate: $bad" >&2
    exit 1
  fi
  grep -q 'invalid dual kernel' "$WORK/m7_${bad}.err"
done

"$ROOT/scripts/verify_n8_adaptive_smoke.sh" > "$WORK/m8_adaptive_smoke.log"
grep -q 'PASS m=8 adaptive-certificate proposal' "$WORK/m8_adaptive_smoke.log"

echo "PASS n=8 available branches m4,m5,m6,m7 plus m8 adaptive exact smoke"
