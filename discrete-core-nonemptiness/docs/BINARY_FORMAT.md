# Binary record formats

All eight-row binary files use **little-endian** fixed-width integers.  Stored
floating-point fields use IEEE-754 binary64 and are proposal diagnostics only;
no proof verdict depends on their rounded value.  The exact C++ readers include
compile-time checks for little-endian, 8-bit bytes, the fixed integer widths,
and IEEE-754 binary64.  Python readers use explicit `struct` formats beginning
with `<`.

The format self-test is:

```bash
g++ -std=c++20 src/n8/binary_format_selftest.cpp -o format_check
./format_check
```

It must print `hard_record_size=24` and `certificate_record_size=64`.

## Canonical kernel list

Files such as `n8m5_pos.bin` and `n8m8_pos.bin` have no magic number because
this is the original development format.

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `uint64` | number `q` of keys |
| 8 | `q * uint64` | strictly increasing canonical keys |

For eight voters and `m` fractional columns, a key is the concatenation of the
eight sorted `m`-bit row patterns.  Row 0 occupies the most significant used
block and row 7 the least significant block.  Within a row, bit `c` is one iff
the voter approves fractional column `c`.  The file length must be exactly
`8 + 8q` bytes.

## Hard-cell file

Python format: header `<QQ>`, records `<QI4Bd>`.

### Header

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `uint64` | magic `0x3843454c4c533031` |
| 8 | `uint64` | number `q` of hard records |

### Record (24 bytes)

| Record offset | Type | Field | Meaning |
|---:|---|---|---|
| 0 | `uint64` | `key` | canonical matrix key |
| 8 | `uint32` | `h` | eight 3-bit floor coordinates; voter 0 is least significant |
| 12 | `uint8` | `k` | residual committee budget |
| 13 | `uint8` | `Bmask` | usable/deficit-voter mask |
| 14 | `uint8` | `flags` | legacy flag byte; bit 0 records an empty `Bmask` |
| 15 | `uint8` | `reserved` | must be zero |
| 16 | `binary64` | `eps` | proposal diagnostic; current scanners store `1.0` |

The file length must be exactly `16 + 24q` bytes.  Records are strictly ordered
by `(key, k, h)`; the record checker additionally verifies that record IDs are
unique.

## Fixed/adaptive certificate and failure files

Python validation format: header `<Q>`, records `<QI4BdHbB4sdd8H>` (the `4s` field must be four zero bytes).

### Header

| Offset | Type | Meaning |
|---:|---|---|
| 0 | `uint64` | number `q` of records |

### Record (64 bytes)

| Record offset | Type | Field | Meaning |
|---:|---|---|---|
| 0 | 24-byte hard record | `r` | hard-cell identifier and metadata |
| 24 | `uint16` | `committee` | fixed committee; for adaptive records, voter mask `E` |
| 26 | `int8` | `deficit` | fixed deficit voter; `-1` for adaptive records |
| 27 | `uint8` | `type` | `0` fixed, `1` adaptive |
| 28 | 4 bytes | `reserved` | all zero; explicit replacement for former ABI padding |
| 32 | `binary64` | `sg` | proposed singleton/adaptive-sum margin |
| 40 | `binary64` | `coal` | proposed coalition margin |
| 48 | `8 * uint16` | `allcm` | adaptive puncture committee for each voter |

The binary64 fields are not trusted.  Exact replay recomputes every relevant
margin from the matrix, floor, and integral committee fields.  The file length
must be exactly `8 + 64q` bytes.

A successful failure file is **eight bytes**, containing a zero `uint64` record
count.  It is not a zero-length file.

## Six-column exact certificate file

The `m=6` branch has a separate exact format because its positive-dual space is
two-dimensional.  Header magic is the eight ASCII bytes `M6CERT01`, followed by
a little-endian `uint64` count.  Each record is 36 bytes with Python format
`<QIBBBbHBB8H>`:

| Record offset | Type | Meaning |
|---:|---|---|
| 0 | `uint64` | key |
| 8 | `uint32` | packed floor `h` |
| 12 | `uint8` | residual budget `k` |
| 13 | `uint8` | usable mask |
| 14 | `uint8` | certificate type: `0` fixed, `1` adaptive |
| 15 | `int8` | fixed deficit, or `-1` for adaptive |
| 16 | `uint16` | fixed committee |
| 18 | `uint8` | adaptive voter mask `E` (zero for fixed records) |
| 19 | `uint8` | reserved; must be zero |
| 20 | `8 * uint16` | adaptive committees |

The file length is exactly `16 + 36q` bytes.

## Text records for six- and seven-voter predecessor modules

### Antichain record

```text
m support_mask_1 ... support_mask_m
```

Each support is an `n`-bit unsigned integer.  Bit `i` is one exactly when voter
`i` approves the corresponding fractional candidate.  Supports are in
canonical sorted order.

### Floor-cell record

```text
P|N m kappa support_mask_1 ... support_mask_m |h_1...h_n
```

`P` denotes a coordinatewise-minimal nonimplementable floor, `N` a generated
upper-cone floor, `kappa` the residual budget, and the final digit string the
integral floor vector.
