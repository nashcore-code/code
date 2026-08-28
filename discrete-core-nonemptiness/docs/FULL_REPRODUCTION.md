# Full reproduction

## Prerequisites

- little-endian host with IEEE-754 binary64;
- GNU-compatible C++20 compiler;
- Boost.Rational and Boost.Multiprecision headers;
- GMP and GMPXX;
- Python 3.10+;
- OpenMP for the seven-voter predecessor enumerator.

The eight-row build runs a compile-time and executable binary-layout audit before
starting proof computation.

## Six voters

```bash
scripts/regenerate_n6.sh /new/output/path/n6
```

The destination must not exist. The script regenerates the antichain universe,
minimal holes, exact filters, final cells, and both exact classifier outputs.

## Seven voters

```bash
JOBS=5 scripts/regenerate_n7.sh /new/output/path/n7
```

This regenerates the positive-dual universe and floor-cell classification, then
runs all square/rectangular checks. The final one-deficit witness check reads
the deposited `data/n7/puncture_certificates.json`; the package verifies but
does not generate that JSON file.

## Eight voters

Run the self-cleaning smoke test:

```bash
SMOKE_MAX_M=5 scripts/smoke_test_n8.sh
```

Then run or resume the complete computation:

```bash
JOBS=8 CHUNK_SIZE=100000 scripts/regenerate_n8.sh /new/output/path/n8
RESUME=1 JOBS=8 CHUNK_SIZE=100000 \
  scripts/regenerate_n8.sh /new/output/path/n8
```

Resume requires the original chunk layout and reuses only outputs whose
SHA-256 provenance binds the exact executable, arguments, inputs, parameters,
and output. A changed input invalidates its dependent chunks; a changed layout
is rejected; legacy unbound chunks are discarded.

The pipeline enumerates `m=4,...,8`, scans every floor cell, constructs all
fixed/adaptive certificates, runs exact `m=6` and `m=7` replay, and compares the
signed-rational and GMP `m=8` replays. A different platform may generate a
different exact-valid committee stream; record coverage and exact replay, not
proposal identity, determine success.

For `m=6`, the hard-record stream is regenerated, while the exact certificate
stream is serialized from the deposited
`data/n8/m6/m6_unresolved_certificate_summary.csv` before replay. The `m=7`
and `m=8` certificate proposal streams are generated during the run.

## Complete theorem artifact

```bash
scripts/full_regenerate.sh /new/output/path/all-modules
```

This creates separate `n6`, `n7`, and `n8` subruns and never overwrites an
existing destination. It is the complete packaged regeneration interface, but
because of the two deposited certificate inputs described above it is not a
fully source-independent reconstruction of every certificate datum.

The distributed artifact includes the complete square-stage outputs. Run
`scripts/quick_verify.sh` for a referee-scale replay of the deposited proof.
The complete `m=8` certificate stream is partitioned into exact contiguous
intervals and checked by both arithmetic backends; set `REPLAY_JOBS=N` to
control parallelism. Fresh output regeneration is available through the
commands above, subject to the disclosed deposited-input scope.
