# Eight voters, four fractional columns

This directory now contains the complete `m=4` census inputs and outputs.

- `n8m4_all.bin`: 5,060 canonical full-rank antichain kernels.
- `n8m4_pos.bin`: 4,779 positive-dual kernels.
- `n8m4_hard.bin`: hard-cell file with a valid header and zero records.
- `cps4_all.bin`, `cps4_fail.bin`: zero-record certificate/failure files.

The scanner checks one residual budget (`k=2`), finds 22 feasible
nonroundable floor cells, and resolves all of them by a usable tight voter.
The exact command output is preserved under `logs/n8/m4/`.
