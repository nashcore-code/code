# Optional independent six-voter cross-check

This is the older 3,209-case integer-only six-voter computation. It supports an
independent proof route but is not the six-row base used by the unified
at-most-eight theorem. Only source code and verification reports are included;
the associated manuscript is intentionally excluded from this repository.

Both programs accept an optional report pathname as their first argument. If it
is omitted, they write `six_voter_exact_finite_report.txt` and
`six_voter_exact_coverage_report.txt`, respectively, in the current directory.
