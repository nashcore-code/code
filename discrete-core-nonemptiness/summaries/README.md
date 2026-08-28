# Machine-readable summaries

- `bounded_voter_results.json`: consolidated module and release-state metadata.
- `n8/eight_voter_results.json`: complete `m=4,...,8` census and theorem-artifact
  status.
- `n8/m8_full_completion.json`: full square-stage file hashes, execution
  environment metrics, kernel audit, record census, and exact replay aggregates.
- `n8/m8_scan_summary.json`: complete floor-cell census by residual budget.
- `n8/m8_certificate_summary.json`: accepted/failure record census and hashes.
- `n8/m8_exact_ll_summary.json` and `m8_exact_gmp_summary.json`: independent
  exact-replay aggregates, which must agree.
- `n8/m8_adaptive_certificates.json`: the ten exceptional adaptive records used
  by the large-data-independent smoke test.

A fresh `scripts/regenerate_n8.sh` run writes corresponding summaries and a
SHA-256 manifest in its chosen output directory.
