#!/usr/bin/env python3
"""Independent structural audit of the four deposited m=8 binary streams.

This deliberately does not import the production serializer or parser.  It
checks the platform-independent little-endian layout, exact record counts,
strict hard-record order, exact hard/certificate identifier equality, reserved
bytes, certificate kinds, zero failure count, and budget-by-budget census.
It complements (but does not replace) the mathematical exact replay programs.
"""
from __future__ import annotations

import collections
import struct
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
DATA = ARTIFACT_ROOT / "data" / "n8" / "m8"
POS = DATA / "n8m8_pos.bin"
HARD = DATA / "n8m8_hard.bin"
CERT = DATA / "cps8_all.bin"
FAIL = DATA / "cps8_fail.bin"

HARD_MAGIC = 0x3843454C4C533031
HARD_RECORD_SIZE = 24
CERT_RECORD_SIZE = 64


def read_exact(stream, size: int, label: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise RuntimeError(f"truncated {label}: got {len(data)} of {size} bytes")
    return data


def main() -> None:
    with POS.open("rb") as stream:
        q_pos = struct.unpack("<Q", read_exact(stream, 8, "positive header"))[0]
    with HARD.open("rb") as stream:
        magic, q_hard = struct.unpack("<QQ", read_exact(stream, 16, "hard header"))
    with CERT.open("rb") as stream:
        q_cert = struct.unpack("<Q", read_exact(stream, 8, "certificate header"))[0]
    with FAIL.open("rb") as stream:
        q_fail = struct.unpack("<Q", read_exact(stream, 8, "failure header"))[0]
        if stream.read(1):
            raise RuntimeError("failure file has bytes after its count header")

    pos_size_ok = POS.stat().st_size == 8 + 8 * q_pos
    hard_size_ok = HARD.stat().st_size == 16 + HARD_RECORD_SIZE * q_hard
    cert_size_ok = CERT.stat().st_size == 8 + CERT_RECORD_SIZE * q_cert
    print(f"positive_count={q_pos} size_ok={pos_size_ok}")
    print(f"hard_magic={magic:#x} hard_count={q_hard} size_ok={hard_size_ok}")
    print(f"certificate_count={q_cert} size_ok={cert_size_ok}")
    print(f"failure_count={q_fail} size={FAIL.stat().st_size}")

    if q_pos != 9_105_190 or not pos_size_ok:
        raise RuntimeError("invalid positive-kernel stream")
    if magic != HARD_MAGIC or q_hard != 1_049_187 or not hard_size_ok:
        raise RuntimeError("invalid hard-record stream")
    if q_cert != q_hard or not cert_size_ok:
        raise RuntimeError("invalid certificate stream")
    if q_fail != 0 or FAIL.stat().st_size != 8:
        raise RuntimeError("failure stream is not the canonical zero-count file")

    hard_by_k: collections.Counter[int] = collections.Counter()
    cert_by_k_type: collections.Counter[tuple[int, int]] = collections.Counter()
    bad_reserved = 0
    bad_type = 0

    with HARD.open("rb") as hard_stream, CERT.open("rb") as cert_stream:
        read_exact(hard_stream, 16, "hard header")
        read_exact(cert_stream, 8, "certificate header")
        previous: tuple[int, int, int] | None = None
        for index in range(q_hard):
            hard_record = read_exact(
                hard_stream, HARD_RECORD_SIZE, f"hard record {index}"
            )
            cert_record = read_exact(
                cert_stream, CERT_RECORD_SIZE, f"certificate record {index}"
            )
            # The binary identifier order is (canonical matrix key, k, floor).
            record_id = (
                struct.unpack_from("<Q", hard_record, 0)[0],
                hard_record[12],
                struct.unpack_from("<I", hard_record, 8)[0],
            )
            if previous is not None and record_id <= previous:
                raise RuntimeError(
                    f"hard-record order failure at {index}: {previous} then {record_id}"
                )
            previous = record_id
            if hard_record != cert_record[:HARD_RECORD_SIZE]:
                raise RuntimeError(f"hard/certificate identifier mismatch at {index}")

            residual_budget = hard_record[12]
            cert_kind = cert_record[27]
            hard_by_k[residual_budget] += 1
            cert_by_k_type[(residual_budget, cert_kind)] += 1
            bad_reserved += cert_record[28:32] != bytes(4)
            bad_type += cert_kind not in (0, 1)

        if hard_stream.read(1) or cert_stream.read(1):
            raise RuntimeError("unexpected trailing binary data")

    print("hard_by_k=" + repr(dict(sorted(hard_by_k.items()))))
    print("cert_by_k_type=" + repr(dict(sorted(cert_by_k_type.items()))))
    print(f"bad_type={bad_type} bad_reserved={bad_reserved}")

    expected_hard = {2: 59_358, 3: 258_574, 4: 413_323, 5: 258_574, 6: 59_358}
    expected_cert = collections.Counter(
        {
            (2, 0): 59_350,
            (2, 1): 8,
            (3, 0): 258_572,
            (3, 1): 2,
            (4, 0): 413_323,
            (5, 0): 258_574,
            (6, 0): 59_358,
        }
    )
    if dict(hard_by_k) != expected_hard:
        raise RuntimeError("hard-record budget census mismatch")
    if cert_by_k_type != expected_cert:
        raise RuntimeError("certificate budget/type census mismatch")
    if bad_type or bad_reserved:
        raise RuntimeError("invalid certificate type or nonzero reserved bytes")

    print(
        "PASS independent binary layout, ordered record-bijection, "
        "and budget census audit"
    )


if __name__ == "__main__":
    main()
