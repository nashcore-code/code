# Eight-row source modules

The supported build/orchestration interface is `scripts/regenerate_n8.sh`.
Individual programs remain usable for audit and development.

- `m8_canonical_enumerator.cpp`: direct and canonical-augmentation kernel
  enumeration plus exact positive-dual filtering for `m=4,...,8`.
- `eight_row_floor_cell_scanner_template.cpp`: compile with `-DMM=m` for the
  complete floor-cell scan at a selected fractional-column count.
- `m8_floor_cell_scanner.cpp`: independently copied `m=8` specialization; the
  pipeline compares it with the template on a prefix.
- `m8_certificate_proposer.cpp`: compile with `-DMM=7` or `8`; it supports the
  affine dual line at `m=7` and the unique square dual at `m=8`. Floating point
  is used only to propose integral fixed/adaptive committees.
- `m6_exact_certificate_checker.cpp`, `exact_cps_fullsat_checker.cpp`,
  `m8_exact_checker_ll.cpp`, `m8_exact_checker_gmp.cpp`: exact replays for the
  two-dimensional, one-dimensional, and square dual cases. The `m=7` checker
  validates full rank and nonempty strictly positive dual domain itself; both
  square checkers validate their unique positive dual equations and structural
  metadata.
- `m8_record_checker.cpp`: strict hard/certificate record bijection and failure
  count validation.
- `merge_and_summarize.py`, `merge_certificates.py`,
  `aggregate_checker_logs.py`: coverage-preserving chunk aggregation.
- `compare_certificate_semantics.py`: optional diagnostic comparison of fields
  consumed by exact replay. It is **not** an acceptance test: a different
  exact-valid proposal is permitted and need not match a deposited committee.
- `n8_binary_format.hpp`, `binary_format_selftest.cpp`: authoritative C++
  layouts and representation checks.

See `docs/N8_PIPELINE.md`, `docs/TRUST_BOUNDARY.md`, and
`docs/BINARY_FORMAT.md`.
