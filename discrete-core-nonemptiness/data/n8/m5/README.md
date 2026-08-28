# Eight voters, five fractional columns

This directory contains the complete `m=5` census inputs and outputs.

- `n8m5_all_direct.bin`: 69,814 canonical full-rank antichain kernels.
- `n8m5_pos_direct.bin`: direct positive-dual filter result.
- `n8m5_pos_augmented.bin`: canonical augmentation from the `m=4` list.
- `n8m5_pos.bin`: canonical released copy; the two constructions agree byte-for-byte and contain 56,479 kernels.
- `n8m5_hard.bin`: hard-cell file with a valid header and zero records.
- `cps5_all.bin`, `cps5_fail.bin`: zero-record certificate/failure files.

For residual budgets `k=2,3`, the scanner finds 84 feasible nonroundable
floor cells in total and resolves all of them by a usable tight voter.  The
exact command output is preserved under `logs/n8/m5/`.
