# Eight-voter regeneration pipeline

`scripts/regenerate_n8.sh` is the supported entry point. Its Python driver is
`scripts/regenerate_n8.py`; the shell wrapper preserves the interface used by
`scripts/full_regenerate.sh`.

## Stages

1. **Build and format audit.** Compile all programs reachable under the selected
   `MAX_M`; a default `MAX_M=8` run compiles the complete eight-row toolset. Run
   the binary-layout self-test.
2. **Canonical enumeration.** Directly enumerate `m=4`; directly enumerate
   `m=5` and independently obtain its positive list by canonical augmentation
   from `m=4`; require byte-for-byte agreement. Extend the positive hierarchy
   through the selected maximum `m`.
3. **Floor-cell scan.** For every selected `m`, scan all residual budgets
   `k=2,...,m-2` in contiguous chunks. Merge only after checking exact coverage,
   deterministic order, counts, and record sizes.
4. **Certificate construction.** `m=4,5` have no hard records. The regenerated
   `m=6` hard stream is paired with a certificate table serialized from the
   deposited `data/n8/m6/m6_unresolved_certificate_summary.csv`, then replayed
   exactly. At `m=7`, the proposer handles the affine dual line; at `m=8`, the
   dual is unique. Floating point chooses only candidate integral certificates.
5. **Exact acceptance.** Check the hard/certificate bijection and empty failure
   file. Reconstruct every certificate exactly. The `m=7` checker first proves
   that the kernel itself is a full-rank antichain with nonempty strictly
   positive dual domain. The two `m=8` arithmetic backends must agree. A
   platform may choose a different exact-valid committee; proposal byte
   identity, fixed/adaptive split, and reference minimum margins are not proof
   obligations.
6. **Audit and manifest.** Verify the square-kernel list, expected hard-cell
   census, and canonical positive/hard reference hashes. Write a JSON result and
   SHA-256 manifest.

## Full run

```bash
JOBS=8 CHUNK_SIZE=100000 scripts/regenerate_n8.sh /new/path/n8
```

The destination must be new. A full run is long and creates substantial
intermediate data. Resume it with the same chunk layout:

```bash
RESUME=1 JOBS=8 CHUNK_SIZE=100000 \
  scripts/regenerate_n8.sh /new/path/n8
```

### Resume invariants

Each reusable stage has a `.provenance.json` sidecar containing:

- a provenance-format version;
- the stage name;
- executable path, size, and SHA-256;
- the complete argument vector;
- named input paths, sizes, and SHA-256 values;
- stage parameters;
- named output paths, sizes, and SHA-256 values.

A sidecar mismatch, a mutated output, a changed input, or a changed executable
forces regeneration. Scan directories additionally store `scan_plan.json`,
which fixes `m`, total kernel count, chunk size, exact intervals, positive-list
hash, and scanner hash. Changing only the chunk layout is rejected so that a
resume cannot silently mix partitions. When an old scan directory has no plan
or provenance, its chunks are discarded rather than adopted. Merged hard files
and final summaries are always rebuilt from the validated chunks.

## Fast smoke runs

The self-cleaning wrapper is:

```bash
SMOKE_MAX_M=5 scripts/smoke_test_n8.sh
```

With an explicit destination:

```bash
MAX_M=5 scripts/regenerate_n8.sh /new/path/n8-smoke
```

The smoke path compiles only the stages reachable through `m=5`, independently
constructs the `m=5` positive list in two ways, and reproduces the exact `m=4,5`
zero-hard-cell censuses.

## Development switches

- `JOBS=N`: threads for enumeration/scanning and process workers for certificate
  proposal/replay.
- `CHUNK_SIZE=N`: matrices per scan chunk.
- `MAX_M=4,...,8`: stop the mathematical pipeline at the selected level.
- `RESUME=1`: reuse only hash-bound, validated outputs.
- `SKIP_GMP=1`: omit the independent GMP replay; development-only.

The Python driver also supports `--stop-after build|enumerate|scan|certify|verify`
and `--no-reference-hashes` for diagnostic runs.

## Large-data-independent `m=8` test

The ten audited adaptive records lie on nine canonical square kernels. This
command reconstructs that nine-key input, runs both scanner implementations,
requires byte-identical hard records, checks occurrence of all ten exceptional
records in the resulting 48-record subproblem, reruns proposal, exercises the
production mergers, and replays all 48 records with both exact backends:

```bash
scripts/verify_n8_adaptive_smoke.sh
```

This is not a substitute for the full 9,105,190-kernel scan.

## Completed production execution

The full square stage has now been executed. It regenerated and audited all
9,105,190 `m=8` kernels, scanned every residual budget, produced 1,049,187
hard records, constructed 1,049,177 fixed and 10 adaptive certificates, found
zero failures, and passed both exact replay backends. The proof-relevant files
are deposited under `data/n8/m8`; see `summaries/n8/m8_full_completion.json`.

The deposited-data verifier uses the same exact C++ replay programs but makes a
checked, gap-free partition of the complete certificate stream for dynamic
parallel scheduling.  Both backends process every interval, and the summed
integer counters and rational minima must agree exactly.

This pipeline regenerates every matrix universe, scan, and `m=7,8` proposal,
but it is not a source-independent generator for the deposited `m=6`
certificate-summary CSV.
