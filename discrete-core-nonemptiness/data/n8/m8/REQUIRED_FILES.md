# Required full eight-column files

All proof-relevant square-stage files are present in this artifact.

| File | Status | Bytes | SHA-256 |
|---|---|---:|---|
| `n8m8_pos.bin` | present; matches audited reference | 72,841,528 | `4bff2f6e4af42bb2ff8517e08fd7ceff36767f8a6a7bd4b95f972799a7f597d0` |
| `n8m8_hard.bin` | present; matches audited reference | 25,180,504 | `f22c6d2ee7f8359e99ac84030a27f17588ae46c069831b57cd7f0f5f97d77ff4` |
| `cps8_all.bin` | present; exact-valid alternative stream | 67,147,976 | `527d8237a2e4aaa79a5767fc5b6cfed8e9094f11e62f413e49a023dff683021b` |
| `cps8_fail.bin` | present; matches audited reference | 8 | `af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc` |

The failure file is an eight-byte little-endian zero-record header, not a
zero-length file. Zero failures are established jointly by the record checker,
which proves that every hard-cell key occurs exactly once in the accepted or
failure stream, and by the zero record count.

Run:

```bash
scripts/verify_n8.sh
```

The verifier audits the canonical kernel and hard-cell hashes, checks record
coverage, and replays all certificates independently with signed-rational and
GMP arithmetic. Proposal-file byte identity is diagnostic rather than a proof
obligation; exact replay is authoritative.

The certificate hash is intentionally the exact-valid release-stream hash, not
a requirement to reproduce one floating-point proposer choice. See
`docs/CERTIFICATE_STREAM_PORTABILITY.md`.
