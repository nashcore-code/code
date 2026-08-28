#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


root = Path(__file__).resolve().parent.parent
manifest = root / "SHA256SUMS"
lines: list[str] = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path == manifest:
        continue
    relative = path.relative_to(root)
    if (
        ".git" in relative.parts
        or relative.name == ".DS_Store"
        or "__pycache__" in relative.parts
        or path.suffix == ".pyc"
    ):
        continue
    lines.append(f"{sha256(path)}  {relative.as_posix()}")
manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {manifest} with {len(lines)} entries.")
