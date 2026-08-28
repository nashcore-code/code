#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
ROOT="$(artifact_root)"
CXX="${CXX:-g++}"
require_command "$CXX"
require_command python3

file_sha256() {
  python3 - "$1" <<'PYHASH'
from pathlib import Path
import hashlib, sys
h=hashlib.sha256()
with Path(sys.argv[1]).open('rb') as stream:
    for block in iter(lambda: stream.read(1 << 20), b''):
        h.update(block)
print(h.hexdigest())
PYHASH
}

WORK="$(mktemp -d "${TMPDIR:-/tmp}/discrete-core-n8-regressions.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# 1. The no-argument smoke wrapper must hand the driver a nonexistent child.
SMOKE_MAX_M=4 JOBS=2 "$ROOT/scripts/smoke_test_n8.sh" \
  > "$WORK/smoke.out" 2> "$WORK/smoke.err"
grep -q 'PASS n=8 smoke expected m=4..4' "$WORK/smoke.out"

# 2. The m=7 exact checker must validate the dual domain itself.
"$CXX" -O2 -DMM=7 -std=c++20 \
  "$ROOT/src/n8/exact_cps_fullsat_checker.cpp" -o "$WORK/check_m7"
python3 - "$ROOT/data/n8/m7/cps7_fullsat_all.bin" "$WORK/valid.bin" \
  "$WORK/rank_bad.bin" "$WORK/empty_dual.bin" <<'PY'
from pathlib import Path
import struct, sys
src=Path(sys.argv[1]).read_bytes()
if len(src) < 72:
    raise SystemExit('truncated source certificate stream')
record=src[8:72]
Path(sys.argv[2]).write_bytes(struct.pack('<Q', 1) + record)
rank_bad=bytearray(struct.pack('<Q', 1) + record)
struct.pack_into('<Q', rank_bad, 8, 0)
Path(sys.argv[3]).write_bytes(rank_bad)
empty_dual=bytearray(struct.pack('<Q', 1) + record)
struct.pack_into('<Q', empty_dual, 8, 53221801103618419)
Path(sys.argv[4]).write_bytes(empty_dual)
PY
"$WORK/check_m7" "$WORK/valid.bin" > "$WORK/m7_valid.out" 2> "$WORK/m7_valid.err"
grep -q 'PASS certs=1' "$WORK/m7_valid.out"
for bad in rank_bad empty_dual; do
  if "$WORK/check_m7" "$WORK/${bad}.bin" > "$WORK/${bad}.out" 2> "$WORK/${bad}.err"; then
    echo "ERROR: invalid m=7 dual-domain certificate was accepted: $bad" >&2
    exit 1
  fi
  grep -q 'invalid dual kernel' "$WORK/${bad}.err"
done

# 3. Platform-dependent proposal choices must not be compared to one stored
# committee stream. Coverage plus exact validity is the verification rule.
if grep -q 'compare_certificate_semantics.py' "$ROOT/scripts/verify_n8_available.sh"; then
  echo 'ERROR: available-data verifier still requires proposal identity' >&2
  exit 1
fi
grep -q 'm7_regenerated_exact' "$ROOT/scripts/verify_n8_available.sh"
if grep -q 'check_sha256 e529d6ca82525073c90d374dff789f3fdc750555cc480ec55fc9304f37046c12' \
    "$ROOT/scripts/verify_n8.sh"; then
  echo 'ERROR: full verifier still treats one proposal hash as mathematical authority' >&2
  exit 1
fi
grep -q 'parallel_m8_exact_replay.py' "$ROOT/scripts/verify_n8.sh"
grep -q 'fixed + adaptive' "$ROOT/scripts/parallel_m8_exact_replay.py"

# 4. Resume must bind every reused key list and scan chunk to SHA-256 provenance.
OUT="$WORK/resume"
MAX_M=4 JOBS=2 CHUNK_SIZE=1000 "$ROOT/scripts/regenerate_n8.sh" "$OUT" \
  > "$WORK/resume_initial.out" 2> "$WORK/resume_initial.err"
initial_pos_sha="$(file_sha256 "$OUT/data/m4/n8m4_pos.bin")"

# Reproduce the reported failure mode exactly: change one byte in the canonical
# m=4 positive list and resume. The key stage must reject the modified output,
# regenerate it, and restore the original hash. Once restored, an old scan chunk
# cryptographically bound to that same canonical hash remains valid to reuse.
python3 - "$OUT/data/m4/n8m4_pos.bin" <<'PYKEY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); data=bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
PYKEY
RESUME=1 MAX_M=4 JOBS=2 CHUNK_SIZE=1000 "$ROOT/scripts/regenerate_n8.sh" "$OUT" \
  > "$WORK/resume_key_mutated.out" 2> "$WORK/resume_key_mutated.err"
grep -q 'mismatched hash-binding provenance; regenerating' "$WORK/resume_key_mutated.err"
restored_pos_sha="$(file_sha256 "$OUT/data/m4/n8m4_pos.bin")"
[[ "$restored_pos_sha" == "$initial_pos_sha" ]]

# Independently corrupt a scan output. Its output hash no longer matches its
# sidecar, so the exact scanner must run again rather than accepting stale data.
python3 - "$OUT/chunks/m4/scan/chunk_0000000.bin" <<'PYSCAN'
from pathlib import Path
import sys
p=Path(sys.argv[1]); data=bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
PYSCAN
RESUME=1 MAX_M=4 JOBS=2 CHUNK_SIZE=1000 "$ROOT/scripts/regenerate_n8.sh" "$OUT" \
  > "$WORK/resume_scan_mutated.out" 2> "$WORK/resume_scan_mutated.err"
grep -q 'n8_scan_m4 .*chunk_0000000.bin' "$WORK/resume_scan_mutated.err"

printf '%s\n' \
  'PASS n=8 release regressions: smoke destination, m=7 domain validation,' \
  'proposal non-identity, and SHA-256-bound resume'
