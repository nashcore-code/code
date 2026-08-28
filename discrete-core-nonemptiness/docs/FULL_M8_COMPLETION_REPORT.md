# Full m=8 proof-data completion report

Date: 2026-08-07 UTC

## Result

- Kernel audit: PASS (9,105,190 sorted, unique, full-rank antichains with positive dual).
- Hard records: 1,049,187.
- Certificates: 1,049,187 = 1,049,177 fixed + 10 adaptive.
- Failure records: 0.
- Signed-rational exact replay: PASS.
- GMP exact replay: PASS.
- Proof-relevant aggregate agreement: PASS.
- Minimum singleton/adaptive-sum margin: 1/1008.
- Minimum exact price margin: 1/11.

## Files

| File | Bytes | SHA-256 |
|---|---:|---|
| `n8m8_pos.bin` | 72,841,528 | `4bff2f6e4af42bb2ff8517e08fd7ceff36767f8a6a7bd4b95f972799a7f597d0` |
| `n8m8_hard.bin` | 25,180,504 | `f22c6d2ee7f8359e99ac84030a27f17588ae46c069831b57cd7f0f5f97d77ff4` |
| `cps8_all.bin` | 67,147,976 | `527d8237a2e4aaa79a5767fc5b6cfed8e9094f11e62f413e49a023dff683021b` |
| `cps8_fail.bin` | 8 | `af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc` |

## Exact aggregate counters

- Saturation exclusions: 122,280,434
- Open-floor checks: 6,543,104
- Exact price LPs: 4,430,958

## Reproduction and verification

The source run regenerated the square kernel universe, scanned all residual
budgets, generated certificates, proved hard-record coverage, and replayed
every certificate with two independent exact-arithmetic implementations.
The final release additionally runs `scripts/quick_verify.sh` on the packaged
data and embeds that transcript under `logs/n8/`.

## Certificate-stream portability

The release certificate file is an exact-valid alternative to the earlier
floating-point proposal stream.  Its hash and two route-specific aggregate
counters therefore differ, while record coverage, fixed/adaptive counts,
minimum margins, and both exact replays agree.  See
`docs/CERTIFICATE_STREAM_PORTABILITY.md` for the complete comparison and the
recommended manuscript wording.

## Independent raw-binary audit

A separate Python parser, independent of the C++ record checker, validates the
four deposited square-stage streams. It checks exact byte lengths and headers,
sorted/unique hard and certificate keys, equality of the hard and certificate
key sequences, the zero failure count, certificate type and reserved-byte
fields, and the budget-by-budget census. It reports PASS in
`logs/n8/proof_complete_independent_binary_audit.log`.

## Packaged referee-scale replay

`REPLAY_JOBS=5 scripts/quick_verify.sh` passed on the completed artifact. It
verified the package manifest, predecessor modules, every eight-row branch, the
9,105,190-kernel square audit, exact hard-record coverage, and complete
gap-free contiguous replays of all certificates with both exact arithmetic
backends. Wall time: `23:31.65`; peak RSS:
`642728` KiB. The transcript is
`logs/n8/proof_complete_quick_verify.log`.

## Referee-scale replay of the immutable proof snapshot

`scripts/quick_verify.sh` passed on an immutable snapshot containing the final
source and all proof data. It verified the package manifest, predecessor
modules, all eight-row branches, the 9,105,190-kernel audit, exact record
coverage, and both complete-stream exact replays over gap-free contiguous
partitions. Final packaging adds only this transcript and completion metadata.
Wall time: `23:31.65`; peak RSS:
`642728` KiB. The complete transcript is
`logs/n8/proof_complete_quick_verify.log`.

## Publication-tree note

The recorded quick-verification transcript predates the removal of duplicate
summaries, low-level build logs, and internal release-management documents, so
its manifest line reports 179 files. The public tree now has a 131-entry
manifest. No proof data, proof source, verifier, or reproduction script changed
during that publication-only cleanup.
