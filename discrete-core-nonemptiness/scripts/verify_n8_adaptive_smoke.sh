#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/lib.sh"
ROOT="$(artifact_root)"
CXX="${CXX:-g++}"
require_command "$CXX"
require_command python3
WORK="$(mktemp -d "${TMPDIR:-/tmp}/discrete-core-m8-adaptive.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

"$CXX" -O2 -std=c++20 -pthread -DMM=8 \
  "$ROOT/src/n8/eight_row_floor_cell_scanner_template.cpp" -o "$WORK/scan_template"
"$CXX" -O2 -std=c++20 -pthread \
  "$ROOT/src/n8/m8_floor_cell_scanner.cpp" -o "$WORK/scan_specialized"
"$CXX" -O2 -std=c++20 -pthread -DMM=8 \
  "$ROOT/src/n8/m8_certificate_proposer.cpp" -o "$WORK/propose"
"$CXX" -O2 -std=c++20 "$ROOT/src/n8/m8_record_checker.cpp" -o "$WORK/record"
"$CXX" -O2 -std=c++20 "$ROOT/src/n8/m8_exact_checker_ll.cpp" -o "$WORK/check_ll"
"$CXX" -O2 -std=c++20 "$ROOT/src/n8/m8_exact_checker_gmp.cpp" \
  -lgmpxx -lgmp -o "$WORK/check_gmp"

# The ten adaptive hard records lie on nine canonical square kernels. Rebuild
# just that key list, then rerun both copies of the m=8 scanner from first
# principles. The resulting 48-record subproblem contains all ten exceptional
# records plus 38 fixed-certificate records.
python3 - "$ROOT/summaries/n8/m8_adaptive_certificates.json" "$WORK/keys.bin" <<'PYKEY'
import json,struct,sys
items=json.load(open(sys.argv[1],encoding='utf-8'))
keys=sorted({int(z['key']) for z in items})
if len(items)!=10 or len(keys)!=9:
    raise SystemExit(f'unexpected adaptive/key census: {len(items)} records on {len(keys)} keys')
open(sys.argv[2],'wb').write(struct.pack('<Q',len(keys))+b''.join(struct.pack('<Q',x) for x in keys))
PYKEY

"$WORK/scan_template" "$WORK/keys.bin" "$WORK/hard_template.bin" 1 \
  > "$WORK/scan_template.log" 2> "$WORK/scan_template.err"
"$WORK/scan_specialized" "$WORK/keys.bin" "$WORK/hard_specialized.bin" 1 \
  > "$WORK/scan_specialized.log" 2> "$WORK/scan_specialized.err"
cmp "$WORK/hard_template.bin" "$WORK/hard_specialized.bin"
grep -v '^seconds=' "$WORK/scan_template.log" > "$WORK/scan_template.stable"
grep -v '^seconds=' "$WORK/scan_specialized.log" > "$WORK/scan_specialized.stable"
diff -u "$WORK/scan_template.stable" "$WORK/scan_specialized.stable"

python3 - "$ROOT/summaries/n8/m8_adaptive_certificates.json" "$WORK/hard_template.bin" <<'PYCHECK'
import json,struct,sys
items=json.load(open(sys.argv[1],encoding='utf-8'))
raw=open(sys.argv[2],'rb').read()
if len(raw)<16: raise SystemExit('truncated scanner output')
magic,n=struct.unpack_from('<QQ',raw,0)
rec=struct.Struct('<QIBBBBd')
if magic!=0x3843454c4c533031 or len(raw)!=16+n*rec.size:
    raise SystemExit('bad scanner output format')
ids=set()
last=None
for i in range(n):
    key,h,k,bmask,flags,reserved,eps=rec.unpack_from(raw,16+i*rec.size)
    if reserved: raise SystemExit(f'nonzero reserved byte at scanner record {i}')
    rid=(key,k,h,bmask)
    order=(key,k,h)
    if last is not None and order<=last: raise SystemExit('scanner output is not strictly ordered')
    last=order
    ids.add(rid)
want=[]
for z in items:
    h=sum(int(v) << (3*i) for i,v in enumerate(z['h']))
    want.append((int(z['key']),int(z['kappa']),h,int(z['Bmask'])))
missing=[x for x in want if x not in ids]
if n!=48 or missing:
    raise SystemExit(f'unexpected scanner subproblem: records={n}, missing={missing}')
print('PASS m=8 scanner smoke records=48 adaptive_targets=10')
PYCHECK

# Exercise the same scan merger, certificate merger, record-bijection check,
# exact backends, and aggregate-log merger used by the full computation.
mkdir -p "$WORK/scan_chunks" "$WORK/cert_chunks" "$WORK/exact_logs"
cp "$WORK/hard_template.bin" "$WORK/scan_chunks/chunk_0000000.bin"
cp "$WORK/scan_template.log" "$WORK/scan_chunks/chunk_0000000.out"
python3 "$ROOT/src/n8/merge_and_summarize.py" \
  "$WORK/scan_chunks" "$WORK/hard_merged.bin" "$WORK/scan_summary.json" \
  --total 9 > "$WORK/merge_scan.log"
cmp "$WORK/hard_template.bin" "$WORK/hard_merged.bin"

"$WORK/propose" "$WORK/hard_merged.bin" \
  "$WORK/cert_chunks/cert_0000000.bin" "$WORK/cert_chunks/fail_0000000.bin" \
  > "$WORK/propose.log"
grep -q '^records=48 fixed=38 adaptive=10 failures=0$' "$WORK/propose.log"
python3 "$ROOT/src/n8/merge_certificates.py" \
  "$WORK/scan_summary.json" "$WORK/cert_chunks" \
  "$WORK/cert_merged.bin" "$WORK/fail_merged.bin" "$WORK/cert_summary.json" \
  > "$WORK/merge_certificates.log"
cmp "$WORK/cert_chunks/cert_0000000.bin" "$WORK/cert_merged.bin"
cmp "$WORK/cert_chunks/fail_0000000.bin" "$WORK/fail_merged.bin"

"$WORK/record" "$WORK/hard_merged.bin" "$WORK/cert_merged.bin" "$WORK/fail_merged.bin" \
  > "$WORK/record.log"
"$WORK/check_ll" "$WORK/cert_merged.bin" > "$WORK/exact_logs/check_ll_0000000.out"
"$WORK/check_gmp" "$WORK/cert_merged.bin" > "$WORK/exact_logs/check_gmp_0000000.out"
diff -u "$WORK/exact_logs/check_ll_0000000.out" "$WORK/exact_logs/check_gmp_0000000.out"
grep -q '^PASS certs=48 fixed=38 adaptive=10 ' "$WORK/exact_logs/check_gmp_0000000.out"
python3 "$ROOT/src/n8/aggregate_checker_logs.py" \
  "$WORK/exact_logs" 'check_ll_*.out' "$WORK/exact_ll_summary.json" \
  > "$WORK/aggregate_ll.log"
python3 "$ROOT/src/n8/aggregate_checker_logs.py" \
  "$WORK/exact_logs" 'check_gmp_*.out' "$WORK/exact_gmp_summary.json" \
  > "$WORK/aggregate_gmp.log"
python3 - "$WORK/exact_ll_summary.json" "$WORK/exact_gmp_summary.json" <<'PYAGG'
import json,sys
left=json.load(open(sys.argv[1],encoding='utf-8'))
right=json.load(open(sys.argv[2],encoding='utf-8'))
left.pop('logs',None)
right.pop('logs',None)
if left!=right:
    raise SystemExit(f'exact aggregate mismatch: {left} != {right}')
if (left['certs'],left['fixed'],left['adaptive'])!=(48,38,10):
    raise SystemExit(f'unexpected exact aggregate: {left}')
print('PASS m=8 chunk/certificate merge and exact aggregate agreement')
PYAGG

echo "PASS m=8 adaptive-certificate proposal, dual-scanner agreement, and independent exact replay on the 48-record exceptional-kernel subproblem"
