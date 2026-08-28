# Trust boundary

## Trusted ingredients

- the mathematical reductions in the manuscript;
- the source files in this archive;
- a conforming C++20/Python/GMP toolchain;
- the operating system and hardware;
- SHA-256 for file and provenance identity;
- the released proof-relevant records after exact format, coverage, and replay
  validation.

## Not trusted

- floating-point feasibility, objective values, or margin values;
- the proposal program's choice of deficit voter or committee;
- byte identity with one platform's proposal output;
- aggregate counts without record-level coverage checks;
- a dual-domain emptiness conclusion inferred from upstream enumeration rather
  than checked at the exact-replay boundary;
- resumed output not cryptographically bound to its exact inputs and tool;
- malformed binary files, trailing data, nonzero reserved bytes, duplicates, or
  a zero-length file used as a purported zero-failure artifact.

## Enforcement in code

1. Floating point only proposes integral fixed/adaptive committees. Exact
   replayers reconstruct the matrix and floor and recompute every relevant
   inequality over exact rationals.
2. The `m=7` exact checker independently verifies antichainness, full column
   rank, `A^T alpha=1`, and existence of a strictly positive point on the affine
   dual line before any certificate can be accepted. Empty domains are safe
   only for a restricted coalition implication whose strict saturation
   antecedent has no feasible price; they are never a vacuous proof of a
   singleton or adaptive-sum margin.
3. Both square `m=8` replayers verify the unique positive dual equations,
   antichainness, residual-budget range, committee masks, and fixed/adaptive
   metadata. The signed-rational replay is independently duplicated with GMP,
   and all proof-relevant aggregate fields must agree.
4. The hard/certificate checker requires a bijection of record identifiers,
   rejects duplicates, validates reserved bytes and exact lengths, and requires
   an eight-byte zero-count failure file.
5. Proposal identity is not an acceptance condition. A different committee
   stream is accepted only after complete record coverage, zero failures, and
   exact replay. The audited proposal hash is diagnostic; the canonical kernel
   and hard-cell hashes remain authoritative for the enumerated universe.
6. Chunk mergers require exact interval coverage, no overlap or gap,
   deterministic global record order, and the expected total kernel count.
7. Resume sidecars bind each reusable stage to the executable SHA-256, full
   command, parameters, exact input hashes, and output hashes. `scan_plan.json`
   separately fixes the chunk layout. A changed fingerprint invalidates the
   affected chunks; legacy unbound chunks are deleted.
8. C++ binary readers reject non-little-endian or non-IEEE-754 hosts at compile
   time; the format self-test verifies all field offsets and record sizes.
9. A complete regeneration audits the canonical `m=8` positive and hard-cell
   reference hashes and emits a complete output manifest.
