#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


manifest = Path(sys.argv[1] if len(sys.argv) > 1 else "SHA256SUMS").resolve()
root = manifest.parent
checked = 0
for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
    if not raw:
        continue
    try:
        expected, relative = raw.split("  ", 1)
    except ValueError as exc:
        raise SystemExit(f"malformed manifest line {line_number}: {raw!r}") from exc
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise SystemExit(f"invalid SHA-256 on manifest line {line_number}")
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"missing manifest file: {relative}")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch: {relative}")
    checked += 1
print(f"Verified SHA-256 for {checked} files.")
