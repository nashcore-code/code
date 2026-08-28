#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
ROOT="$(artifact_root)"
CXX="${CXX:-g++}"
require_command "$CXX"
require_command python3

M8="$ROOT/data/n8/m8"
required=(n8m8_pos.bin n8m8_hard.bin cps8_all.bin cps8_fail.bin)
missing=0
for name in "${required[@]}"; do
  if [[ ! -f "$M8/$name" ]]; then
    echo "MISSING data/n8/m8/$name" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "ERROR: full n=8 verification needs the large m=8 data; see data/n8/m8/REQUIRED_FILES.md" >&2
  exit 2
fi

# Verify every compactly deposited branch before the square-stage replay.
"$ROOT/scripts/verify_n8_available.sh"

# The canonical kernel and hard-cell universes are fixed audited inputs.
check_sha256 4bff2f6e4af42bb2ff8517e08fd7ceff36767f8a6a7bd4b95f972799a7f597d0 "$M8/n8m8_pos.bin"
check_sha256 f22c6d2ee7f8359e99ac84030a27f17588ae46c069831b57cd7f0f5f97d77ff4 "$M8/n8m8_hard.bin"
# A successful failure file has a unique eight-byte zero-count encoding.
check_sha256 af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc "$M8/cps8_fail.bin"

# The proposal program is not authoritative. An alternative certificate stream
# may choose different valid punctures and therefore have a different byte
# hash. Report departure from the audited reference, but let complete record
# coverage and the two exact replayers decide mathematical validity.
python3 - "$M8/cps8_all.bin" <<'PYCERTHASH'
import hashlib, pathlib, sys
p=pathlib.Path(sys.argv[1])
h=hashlib.sha256()
with p.open('rb') as stream:
    for block in iter(lambda: stream.read(1 << 20), b''):
        h.update(block)
actual=h.hexdigest()
reference='e529d6ca82525073c90d374dff789f3fdc750555cc480ec55fc9304f37046c12'
if actual != reference:
    print(f'NOTE: {p.name} differs from the audited proposal hash; exact validity will be checked ({actual})', file=sys.stderr)
else:
    print(f'PASS audited certificate reference hash={actual}')
PYCERTHASH

WORK="$(mktemp -d "${TMPDIR:-/tmp}/discrete-core-n8-verify.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

"$CXX" -O2 -std=c++20 "$ROOT/src/n8/m8_kernel_list_verifier.cpp" -o "$WORK/kernel_check"
"$CXX" -O2 -std=c++20 "$ROOT/src/n8/m8_record_checker.cpp" -o "$WORK/record_check"
"$CXX" -O2 -std=c++20 "$ROOT/src/n8/m8_exact_checker_ll.cpp" -o "$WORK/check_ll"
"$CXX" -O2 -std=c++20 "$ROOT/src/n8/m8_exact_checker_gmp.cpp" \
  -lgmpxx -lgmp -o "$WORK/check_gmp"

"$WORK/kernel_check" "$M8/n8m8_pos.bin" > "$WORK/kernel.log"
grep -q 'PASS kernels=9105190 sorted_unique=1 antichain=1 full_rank=1 positive_dual=1' "$WORK/kernel.log"
"$WORK/record_check" "$M8/n8m8_hard.bin" "$M8/cps8_all.bin" "$M8/cps8_fail.bin" \
  > "$WORK/record.log"
# Certificate records are independent proof obligations.  Make an exact,
# contiguous partition of the complete 64-byte record stream and replay all
# parts with each backend.  Dynamic chunk scheduling gives a referee-scale
# wall time without changing the accepted arithmetic or aggregate verdict.
replay_args=()
if [[ -n "${REPLAY_JOBS:-}" ]]; then
  replay_args+=(--jobs "$REPLAY_JOBS")
fi
python3 "$ROOT/scripts/parallel_m8_exact_replay.py" \
  "$M8/cps8_all.bin" "$WORK/check_ll" "$WORK/check_gmp" \
  "$WORK/parallel_replay" "${replay_args[@]}"

echo "PASS n=8 m4,m5,m6,m7,m8 exact replay and record coverage"
