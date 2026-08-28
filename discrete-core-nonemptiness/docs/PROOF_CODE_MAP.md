# Proof-to-code map

The manuscript is stored separately. This document identifies the artifact
components corresponding to its finite computer lemmas.

## Dependency chain

```text
published n<=5 result
        |
        v
six-row tight-cell classification (n6)
        |
        v
seven-row separation classification (n7)
        |
        v
eight-row price-saturation classification (n8, m=4,...,8)
```

## Six-row tight-cell classification

| Mathematical obligation | Implementation | Proof-relevant data |
|---|---|---|
| Canonical full-rank antichain enumeration | `src/common/enumerate_antichains.cpp` | `data/n6/antichains.txt` |
| Integral utilities and minimal nonimplementable floors | `src/common/generate_minimal_holes.cpp` | regenerated intermediate data |
| Voter-subset exclusion | `src/common/filter_holes_subset.cpp` | regenerated intermediate data |
| Exactly checked cover-dual exclusion | `src/common/filter_holes_coverdual.cpp` | regenerated intermediate data |
| Upper-cone floor generation | `src/common/generate_upper_cells.cpp` | regenerated intermediate data |
| Upper-floor cover exclusion | `src/common/filter_cells_coverdual.cpp` | `data/n6/final_cells.txt` |
| Exact strict-cell classification | `src/common/classify_cells_exact.cpp` and `classify_cells_bigint.cpp` | empty `bad_cells` and `surplus_cells` outputs |

The final conclusion is `PASS n=6 cells=46 bad=0 surplus=0`. This proves that
every surviving six-row floor cell has a usable tight row. Turning that row into
a core committee is a mathematical argument in the separate manuscript.

## Seven-row classification

| Mathematical obligation | Implementation | Proof-relevant data |
|---|---|---|
| Positive-dual kernel universe | `src/n7/enumerate_positive_dual.cpp`, `filter_positive_alpha.cpp` | `data/n7/positive_antichains_*.txt` |
| Shared minimal-hole and floor pipeline | `src/common/*` | `data/n7/final_cells.txt` |
| Exact bad/surplus classification | both common classifiers | `data/n7/surplus_cells.txt`, empty bad output |
| Explicit puncture integrity | `src/n7/verify_puncture_certificates.py` | `data/n7/puncture_certificates.json` |
| Square-cell separation | `src/n7/verify_square_surplus.cpp` | 294 square cells |
| Rectangular-cell separation | `src/n7/verify_rectangular_surplus.cpp` | four rectangular cells |

The final conclusion is `PASS n=7 cells=30929 surplus=298 bad=0`.

## Eight-row naming rule

All eight-row files are under `data/n8/m*`. For example,
`data/n8/m6/n8m6_hard_exact.bin` means eight voters and six fractional columns;
it is unrelated to the six-voter induction base in `data/n6`.


## Eight-row price-saturation classification

| Mathematical obligation | Implementation | Proof-relevant output |
|---|---|---|
| Canonical full-rank antichain orbits and positive-dual filtering | `src/n8/m8_canonical_enumerator.cpp` | `n8m*_pos.bin`; direct/augmentation agreement at `m=5` |
| Complete floor-cell census for every residual budget | `src/n8/eight_row_floor_cell_scanner_template.cpp` | `n8m*_hard.bin` and scan summaries |
| Exact coverage and deterministic merge of scan chunks | `src/n8/merge_and_summarize.py` | coverage interval, aggregate counts, hard-file hash |
| Fixed/adaptive integral certificate proposal for `m=7,8` | `src/n8/m8_certificate_proposer.cpp` | affine-line endpoint proposal at `m=7`, unique-dual proposal at `m=8`; proposal-only doubles are untrusted |
| Exact two-dimensional `m=6` replay | `src/n8/m6_make_certificates.py`, `m6_exact_certificate_checker.cpp` | 168 exact certificates, zero uncovered records |
| Exact one-dimensional `m=7` replay and checker-side dual-domain validation | `src/n8/exact_cps_fullsat_checker.cpp` with `MM=7` | full-rank antichain, nonempty strictly positive dual line, 36,128 exact certificates, zero failures |
| Exact square `m=8` replay and checker-side unique-dual validation | `src/n8/m8_exact_checker_ll.cpp`, `m8_exact_checker_gmp.cpp` | 1,049,187 exact certificates, positive dual equations, arithmetic agreement |
| Hard/certificate bijection and zero failure file | `src/n8/m8_record_checker.cpp` | strict record-set equality and no duplicates |
| Full square-kernel audit | `src/n8/m8_kernel_list_verifier.cpp` | sorted/unique, antichain, full rank, positive dual |
| Binary-layout audit | `src/n8/n8_binary_format.hpp`, `binary_format_selftest.cpp` | 24-byte hard and 64-byte certificate records |
| End-to-end orchestration | `scripts/regenerate_n8.sh`, `regenerate_n8.py` | resumable complete run, JSON result, SHA-256 manifest |

The fast deposited branches conclude:

```text
m=4: kernels=4779  feasible=22  hard=0
m=5: kernels=56479 feasible=84  hard=0
m=6: kernels=561445 feasible=5766 hard=168  fixed=163 adaptive=5
m=7: kernels=3541727 feasible=89286 hard=36128 fixed=36119 adaptive=9
```

The deposited and exactly replayed `m=8` census is:

```text
kernels=9105190 hard=1049187 fixed=1049177 adaptive=10 failures=0
```


Certificate proposal is non-authoritative. The fixed/adaptive split shown above
identifies the deposited stream; an alternative stream may use different valid
committees or a different split, provided it covers every hard record, has zero
failures, and passes both exact replay backends. All four proof-relevant `m=8`
files are included under `data/n8/m8`.

The release verifier makes an exact contiguous partition of the complete
certificate record stream and runs both independent exact backends on every
part.  The partition is checked for full, gap-free coverage; summed counters and
rational minima must give byte-identical aggregate PASS lines.  Parallel chunk
scheduling changes wall time only.
