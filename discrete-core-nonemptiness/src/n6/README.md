# Six-row implementation

Computer Lemma 6 uses the generic programs in `../common` with `n=6`. There is
no separate six-row algorithm and no positive-dual preprocessing step: the
computation deliberately classifies every full-column-rank six-row antichain.

The executable pipeline is defined by `scripts/regenerate_n6.sh`, and the fast
stored-cell replay is defined by `scripts/verify_n6.sh`.

