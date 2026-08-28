# Exact-valid alternative `m=8` certificate stream

## Mathematical status

The canonical square-kernel stream and the hard-cell stream are unique audited
objects in this release and match the hashes recorded in the manuscript.  The
certificate proposer, by contrast, is a floating-point **search heuristic**.  A
hard cell can admit more than one valid integral puncture, so different
platforms or compiler choices can legitimately select different committees.

This artifact deposits the following stream:

```text
cps8_all.bin
bytes   = 67,147,976
SHA-256 = 527d8237a2e4aaa79a5767fc5b6cfed8e9094f11e62f413e49a023dff683021b
records = 1,049,187 = 1,049,177 fixed + 10 adaptive
```

It is not byte-identical to the earlier proposal stream whose recorded hash was
`e529d6ca82525073c90d374dff789f3fdc750555cc480ec55fc9304f37046c12`.
That difference is not accepted on trust.  The release verifier proves all of
the following from the deposited records:

1. the certificate and zero-failure files form an exact, duplicate-free
   partition of all 1,049,187 hard-cell identifiers;
2. every committee mask and all structural metadata are valid;
3. every defining dual equation and positivity condition holds exactly;
4. every singleton/adaptive-sum and coalition implication has a strictly
   positive exact margin; and
5. the signed-rational and independent GMP implementations return identical
   proof-relevant aggregates.

The exact results for this deposited stream are:

```text
saturation_skips     = 122,280,434
open_floor_checks    =   6,543,104
exact_price_LPs      =   4,430,958
min_singleton_or_sum = 1/1008
min_exact_price      = 1/11
```

The earlier proposal stream reported 6,544,201 open-floor checks and 4,429,861
exact price LPs.  Exactly 1,097 checks move between those two valid proof routes
in this release, while their sum remains 10,974,062.  The fixed/adaptive census,
zero-failure result, and exact minimum margins are unchanged.

## Manuscript alignment before public release

Any manuscript sentence that binds `cps8_all.bin` to one specific proposal hash
should either use the release hash above or explicitly label the old hash as a
reference-run hash.  Likewise, a displayed exact-replay transcript should use
the release counters above.  A stream-independent formulation is preferable:

> The deposited certificate stream covers every hard record exactly once, the
> failure stream is empty, and the signed-rational and GMP replayers agree on
> all proof-relevant aggregates with minimum margins `1/1008` and `1/11`.

No theorem statement or mathematical lemma changes; this is alignment of
noncanonical search-output metadata with the exact-valid stream actually
released.
