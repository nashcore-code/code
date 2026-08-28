# Eight-voter proof data

These directories are indexed by `m`, the number of fractional columns in an
eight-row residual kernel; they are unrelated to the number of voters in the
predecessor modules.

- `m4`: complete direct enumeration, positive list, and zero-hard-cell output.
- `m5`: complete direct and canonical-augmentation enumerations, which agree,
  and zero-hard-cell output.
- `m6`: 561,445 positive kernels, 168 hard records, and exact certificates.
- `m7`: 3,541,727 positive kernels, 36,128 hard records, and exact certificates.
- `m8`: complete 9,105,190-kernel list, 1,049,187 hard records, 1,049,187
  accepted certificates, and the zero-record failure stream.

Use `scripts/verify_n8.sh` for complete eight-row verification or
`scripts/quick_verify.sh` for the theorem-wide audit including the predecessor
modules. Exact names, sizes, and hashes of the square-stage files are recorded
in `m8/REQUIRED_FILES.md`.
