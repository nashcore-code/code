#!/usr/bin/env python3
"""Compare proof-relevant fields of two n8 certificate streams.

The floating-point proposal margins are diagnostics only.  For fixed
certificates, the per-voter committee table is also unused by exact replay;
for adaptive certificates only entries selected by the adaptive voter mask are
proof-relevant.  This utility compares exactly the fields consumed by the
exact checkers and validates all reserved bytes.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

CERT = struct.Struct("<QI4BdHbB4sdd8H")


def read(path: Path) -> list[tuple]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"truncated certificate header: {path}")
    count = struct.unpack_from("<Q", raw, 0)[0]
    expected = 8 + count * CERT.size
    if len(raw) != expected:
        raise ValueError(
            f"certificate length mismatch for {path}: got {len(raw)}, expected {expected}"
        )
    out: list[tuple] = []
    for index in range(count):
        values = CERT.unpack_from(raw, 8 + index * CERT.size)
        key, h, k, bmask, flags, hard_reserved, eps = values[:7]
        committee, deficit, cert_type, cert_reserved = values[7:11]
        allcm = values[13:]
        if hard_reserved != 0 or cert_reserved != b"\0\0\0\0":
            raise ValueError(f"nonzero reserved bytes in {path}, record {index}")
        if cert_type not in (0, 1):
            raise ValueError(f"unknown certificate type {cert_type} in {path}, record {index}")
        # The embedded hard record is identified by the discrete fields.  Its
        # binary64 ``eps`` member is a scanner diagnostic and is intentionally
        # excluded, just like the certificate proposal margins below.
        hard = (key, h, k, bmask, flags)
        if cert_type == 0:
            semantic = (hard, committee, deficit, cert_type)
        else:
            selected = tuple(
                (voter, allcm[voter])
                for voter in range(8)
                if (committee >> voter) & 1
            )
            semantic = (hard, committee, deficit, cert_type, selected)
        out.append(semantic)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    left = read(args.left)
    right = read(args.right)
    if len(left) != len(right):
        raise SystemExit(f"count mismatch: {len(left)} != {len(right)}")
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a != b:
            raise SystemExit(f"proof-relevant mismatch at record {index}:\nleft={a}\nright={b}")
    print(f"PASS proof-relevant certificate semantics records={len(left)}")


if __name__ == "__main__":
    main()
