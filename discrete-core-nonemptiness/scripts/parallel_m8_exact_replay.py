#!/usr/bin/env python3
"""Parallel, exact replay of a complete m=8 certificate stream.

The C++ checkers validate certificate records independently.  This driver makes
an exact contiguous partition of the 64-byte record stream, runs each backend
on every part, aggregates integer counters and rational minima, and requires
byte-identical aggregate PASS lines from the signed-rational and GMP backends.
No record is omitted or duplicated: the splitter validates the source length,
records each interval, and checks that the per-part certificate counts sum to
the source header.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from typing import Any

CERT_RECORD_SIZE = 64
READ_BLOCK = 1 << 20
PASS_RE = re.compile(
    r"^PASS certs=(\d+) fixed=(\d+) adaptive=(\d+) "
    r"saturation_skips=(\d+) open_floor_checks=(\d+) exact_price_LPs=(\d+) "
    r"min_singleton_or_sum=(\S+) min_exact_price=(\S+)$"
)
PROOF_FIELDS = (
    "certs",
    "fixed",
    "adaptive",
    "saturation_skips",
    "open_floor_checks",
    "exact_price_LPs",
    "min_singleton_or_sum",
    "min_exact_price",
)


@dataclass(frozen=True)
class Chunk:
    index: int
    first_record: int
    count: int
    path: Path


def eprint(*args: object, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, flush=True, **kwargs)


def read_count(path: Path) -> int:
    with path.open("rb") as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError(f"truncated certificate header: {path}")
    count = struct.unpack("<Q", raw)[0]
    expected = 8 + CERT_RECORD_SIZE * count
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"bad certificate stream length: {path}: {actual} bytes, expected {expected}"
        )
    return count


def copy_exact(source, target, remaining: int) -> None:
    while remaining:
        block = source.read(min(READ_BLOCK, remaining))
        if not block:
            raise ValueError("certificate stream ended while making exact replay chunks")
        target.write(block)
        remaining -= len(block)


def split_stream(source: Path, directory: Path, requested_chunks: int) -> list[Chunk]:
    total = read_count(source)
    if total <= 0:
        raise ValueError("the complete m=8 certificate stream must be nonempty")
    number = min(total, max(1, requested_chunks))
    base, extra = divmod(total, number)

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    chunks: list[Chunk] = []
    first = 0
    with source.open("rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"truncated certificate header: {source}")
        for index in range(number):
            count = base + (1 if index < extra else 0)
            path = directory / f"cert_{index:04d}.bin"
            with path.open("wb") as out:
                out.write(struct.pack("<Q", count))
                copy_exact(stream, out, count * CERT_RECORD_SIZE)
            expected_size = 8 + count * CERT_RECORD_SIZE
            if path.stat().st_size != expected_size:
                raise ValueError(f"internal chunk-size error: {path}")
            chunks.append(Chunk(index, first, count, path))
            first += count
        if stream.read(1):
            raise ValueError("trailing bytes remained after the exact certificate partition")

    if first != total:
        raise ValueError(f"partition covers {first} records, expected {total}")
    intervals = [(chunk.first_record, chunk.first_record + chunk.count) for chunk in chunks]
    if intervals[0][0] != 0 or intervals[-1][1] != total:
        raise ValueError("partition endpoints do not cover the full stream")
    for left, right in zip(intervals, intervals[1:]):
        if left[1] != right[0]:
            raise ValueError("gap or overlap in certificate partition")

    manifest = {
        "source": str(source.resolve()),
        "records": total,
        "record_size": CERT_RECORD_SIZE,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "file": chunk.path.name,
                "first_record": chunk.first_record,
                "records": chunk.count,
            }
            for chunk in chunks
        ],
    }
    (directory / "partition.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return chunks


def tail(path: Path, limit: int = 30) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-limit:])


def run_one(checker: Path, backend: str, chunk: Chunk, log_dir: Path) -> tuple[Chunk, int]:
    stdout = log_dir / f"{backend}_{chunk.index:04d}.out"
    stderr = log_dir / f"{backend}_{chunk.index:04d}.err"
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(
            [str(checker), str(chunk.path)],
            stdout=out,
            stderr=err,
            check=False,
        )
    return chunk, proc.returncode


def parse_pass(path: Path, expected_count: int) -> dict[str, int | str]:
    matches = [
        match
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines()
        if (match := PASS_RE.fullmatch(line))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one aggregate PASS line in {path}, found {len(matches)}")
    match = matches[0]
    values: dict[str, int | str] = {
        "certs": int(match.group(1)),
        "fixed": int(match.group(2)),
        "adaptive": int(match.group(3)),
        "saturation_skips": int(match.group(4)),
        "open_floor_checks": int(match.group(5)),
        "exact_price_LPs": int(match.group(6)),
        "min_singleton_or_sum": match.group(7),
        "min_exact_price": match.group(8),
    }
    if values["certs"] != expected_count:
        raise ValueError(
            f"checker reported {values['certs']} records in {path}, expected {expected_count}"
        )
    if int(values["fixed"]) + int(values["adaptive"]) != expected_count:
        raise ValueError(f"fixed/adaptive census does not sum in {path}")
    return values


def minimum_fraction(current: Fraction | None, text: str) -> Fraction | None:
    if text == "NA":
        return current
    value = Fraction(text)
    return value if current is None or value < current else current


def format_fraction(value: Fraction | None) -> str:
    if value is None:
        return "NA"
    return f"{value.numerator}/{value.denominator}"


def aggregate_backend(
    checker: Path,
    backend: str,
    chunks: list[Chunk],
    jobs: int,
    work: Path,
) -> tuple[dict[str, int | str], str]:
    log_dir = work / backend
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)

    failures: list[tuple[Chunk, int]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(run_one, checker, backend, chunk, log_dir): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            chunk, returncode = future.result()
            completed += 1
            eprint(
                f"\r{backend} exact replay chunks {completed}/{len(chunks)}",
                end="",
            )
            if returncode != 0:
                failures.append((chunk, returncode))
    eprint()
    if failures:
        details = []
        for chunk, returncode in failures[:5]:
            err = log_dir / f"{backend}_{chunk.index:04d}.err"
            details.append(
                f"chunk {chunk.index} [{chunk.first_record},"
                f"{chunk.first_record + chunk.count}) exited {returncode}\n{tail(err)}"
            )
        raise RuntimeError(f"{backend} exact replay failed:\n" + "\n---\n".join(details))

    sums = {
        "certs": 0,
        "fixed": 0,
        "adaptive": 0,
        "saturation_skips": 0,
        "open_floor_checks": 0,
        "exact_price_LPs": 0,
    }
    min_singleton: Fraction | None = None
    min_price: Fraction | None = None
    for chunk in chunks:
        out = log_dir / f"{backend}_{chunk.index:04d}.out"
        values = parse_pass(out, chunk.count)
        for field in sums:
            sums[field] += int(values[field])
        min_singleton = minimum_fraction(
            min_singleton, str(values["min_singleton_or_sum"])
        )
        min_price = minimum_fraction(min_price, str(values["min_exact_price"]))

    result: dict[str, int | str] = {
        **sums,
        "min_singleton_or_sum": format_fraction(min_singleton),
        "min_exact_price": format_fraction(min_price),
    }
    line = (
        f"PASS certs={result['certs']} fixed={result['fixed']} "
        f"adaptive={result['adaptive']} "
        f"saturation_skips={result['saturation_skips']} "
        f"open_floor_checks={result['open_floor_checks']} "
        f"exact_price_LPs={result['exact_price_LPs']} "
        f"min_singleton_or_sum={result['min_singleton_or_sum']} "
        f"min_exact_price={result['min_exact_price']}"
    )
    (work / f"exact_{backend}.log").write_text(line + "\n", encoding="utf-8")
    return result, line


def validate_final(result: dict[str, int | str], expected_total: int) -> None:
    certs = int(result["certs"])
    fixed = int(result["fixed"])
    adaptive = int(result["adaptive"])
    if certs != expected_total or fixed + adaptive != certs:
        raise ValueError(
            f"invalid aggregate certificate census: certs={certs}, fixed={fixed}, adaptive={adaptive}"
        )
    singleton = str(result["min_singleton_or_sum"])
    if singleton == "NA" or Fraction(singleton) <= 0:
        raise ValueError(f"nonpositive or missing singleton/adaptive margin: {singleton}")
    price = str(result["min_exact_price"])
    if int(result["exact_price_LPs"]) > 0 and (price == "NA" or Fraction(price) <= 0):
        raise ValueError(f"nonpositive or missing exact price margin: {price}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate_file", type=Path)
    parser.add_argument("signed_rational_checker", type=Path)
    parser.add_argument("gmp_checker", type=Path)
    parser.add_argument("work_directory", type=Path)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--chunks-per-job",
        type=int,
        default=20,
        help="make enough exact contiguous pieces for dynamic load balancing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs < 1 or args.chunks_per_job < 1:
        raise SystemExit("--jobs and --chunks-per-job must be positive")
    for checker in (args.signed_rational_checker, args.gmp_checker):
        if not checker.is_file() or not os.access(checker, os.X_OK):
            raise SystemExit(f"checker is missing or not executable: {checker}")

    args.work_directory.mkdir(parents=True, exist_ok=True)
    total = read_count(args.certificate_file)
    chunks = split_stream(
        args.certificate_file,
        args.work_directory / "certificate_chunks",
        args.jobs * args.chunks_per_job,
    )
    eprint(
        f"exact certificate partition: records={total} chunks={len(chunks)} jobs={args.jobs}"
    )

    ll, ll_line = aggregate_backend(
        args.signed_rational_checker,
        "ll",
        chunks,
        args.jobs,
        args.work_directory,
    )
    gmp, gmp_line = aggregate_backend(
        args.gmp_checker,
        "gmp",
        chunks,
        args.jobs,
        args.work_directory,
    )
    if ll_line != gmp_line or any(ll[field] != gmp[field] for field in PROOF_FIELDS):
        raise RuntimeError(
            "signed-rational and GMP aggregate outputs disagree\n"
            f"signed-rational: {ll_line}\nGMP: {gmp_line}"
        )
    validate_final(gmp, total)

    summary = {
        "status": "PASS",
        "records": total,
        "jobs": args.jobs,
        "chunks": len(chunks),
        "signed_rational": ll,
        "gmp": gmp,
        "aggregate_lines_identical": True,
    }
    (args.work_directory / "parallel_exact_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"PASS m=8 signed-rational/GMP chunked replays agree "
        f"chunks={len(chunks)} jobs={args.jobs}"
    )
    print(
        "PASS m=8 exact validity "
        f"certs={gmp['certs']} fixed={gmp['fixed']} adaptive={gmp['adaptive']} "
        f"saturation_skips={gmp['saturation_skips']} "
        f"open_floor_checks={gmp['open_floor_checks']} "
        f"exact_price_LPs={gmp['exact_price_LPs']} "
        f"min_singleton_or_sum={gmp['min_singleton_or_sum']} "
        f"min_exact_price={gmp['min_exact_price']}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        eprint(f"ERROR: {exc}")
        raise SystemExit(1)
