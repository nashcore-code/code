#!/usr/bin/env python3
import argparse, csv, re, struct
from pathlib import Path

HARD_MAGIC = 0x3843454C4C533031
CERT_MAGIC = b"M6CERT01"
HARD_REC = struct.Struct("<QIBBBBd")
CERT_REC = struct.Struct("<QIBBBbHBB8H")


def pack_floor(text: str) -> int:
    vals = [int(x) for x in text.split()]
    if len(vals) != 8 or any(v < 0 or v > 7 for v in vals):
        raise ValueError(f"bad floor: {text!r}")
    out = 0
    for i, v in enumerate(vals):
        out |= v << (3 * i)
    return out


def mask_from_columns(text: str) -> int:
    cols = [int(x) for x in text.replace(",", " ").split()]
    mask = 0
    for c in cols:
        if not 1 <= c <= 6:
            raise ValueError(f"bad column {c} in {text!r}")
        mask |= 1 << (c - 1)
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hard")
    ap.add_argument("csv")
    ap.add_argument("output")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    lookup = {}
    for row in rows:
        key = (int(row["matrix_key"]), int(row["kappa"]), pack_floor(row["floor_h"]))
        if key in lookup:
            raise RuntimeError(f"duplicate CSV key {key}")
        lookup[key] = row

    raw = Path(args.hard).read_bytes()
    if len(raw) < 16:
        raise RuntimeError("hard file too short")
    magic, count = struct.unpack_from("<QQ", raw, 0)
    if magic != HARD_MAGIC:
        raise RuntimeError(f"bad hard-file magic {magic:#x}")
    if len(raw) != 16 + count * HARD_REC.size:
        raise RuntimeError("hard-file length mismatch")

    certs = []
    used = set()
    fixed = adaptive = 0
    for idx in range(count):
        key, h, k, bmask, flags, reserved, eps = HARD_REC.unpack_from(raw, 16 + idx * HARD_REC.size)
        if reserved != 0:
            raise RuntimeError(f"nonzero hard-record reserved byte at index {idx}")
        ident = (key, k, h)
        row = lookup.get(ident)
        if row is None:
            raise RuntimeError(f"missing CSV certificate for {ident}")
        used.add(ident)
        typetext = row["certificate_type"]
        allcm = [0] * 8
        if typetext.startswith("all-one averaging"):
            typ = 1
            deficit = -1
            committee = 0
            emask = 0
            parts = [p.strip() for p in row["certificate_detail"].split(";") if p.strip()]
            for part in parts:
                m = re.fullmatch(r"v(\d+)->(.+)", part)
                if not m:
                    raise RuntimeError(f"bad adaptive mapping {part!r}")
                voter = int(m.group(1))
                if not 1 <= voter <= 8:
                    raise RuntimeError(f"bad voter {voter}")
                cm = mask_from_columns(m.group(2))
                allcm[voter - 1] = cm
                emask |= 1 << (voter - 1)
            if emask == 0:
                raise RuntimeError("empty adaptive family")
            if emask & ~bmask:
                raise RuntimeError(f"adaptive voter outside Bmask for {ident}")
            adaptive += 1
        else:
            typ = 0
            deficit = int(row["deficit_voter"]) - 1
            if not 0 <= deficit < 8:
                raise RuntimeError(f"bad deficit voter for {ident}")
            committee = mask_from_columns(row["committee"])
            emask = 0
            if not ((bmask >> deficit) & 1):
                raise RuntimeError(f"fixed deficit outside Bmask for {ident}")
            fixed += 1
        certs.append(CERT_REC.pack(key, h, k, bmask, typ, deficit, committee, emask, 0, *allcm))

    extras = set(lookup) - used
    if extras:
        raise RuntimeError(f"CSV has {len(extras)} unused records")
    Path(args.output).write_bytes(CERT_MAGIC + struct.pack("<Q", count) + b"".join(certs))
    print(f"PASS certificates={count} fixed={fixed} adaptive={adaptive} record_size={CERT_REC.size}")


if __name__ == "__main__":
    main()
