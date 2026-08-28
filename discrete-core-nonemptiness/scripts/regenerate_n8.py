#!/usr/bin/env python3
"""From-scratch regeneration pipeline for the eight-voter computation.

The pipeline is deliberately resumable: every expensive stage validates an existing
output before reusing it.  It enumerates the canonical positive-dual hierarchy,
scans all floor cells, proposes the m=7 and m=8 certificates, and independently
replays the final certificates with exact arithmetic.

The default run performs m=4,...,8.  Use --max-m 5 for the fast smoke test.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

HARD_MAGIC = 0x3843454C4C533031
HARD_RECORD_SIZE = 24
CERT_RECORD_SIZE = 64

EXPECTED = {
    4: dict(full=5060, kernels=4779, cases=4779, feasible=22, hard=0),
    5: dict(full=69814, kernels=56479, cases=112958, feasible=84, hard=0),
    6: dict(kernels=561445, cases=1684335, feasible=5766, hard=168,
            fixed=163, adaptive=5),
    7: dict(kernels=3541727, cases=14166908, feasible=89286, hard=36128,
            fixed=36119, adaptive=9),
    8: dict(kernels=9105190, cases=45525950, feasible=1081420, hard=1049187,
            fixed=1049177, adaptive=10),
}
M8_HASHES = {
    "n8m8_pos.bin": "4bff2f6e4af42bb2ff8517e08fd7ceff36767f8a6a7bd4b95f972799a7f597d0",
    "n8m8_hard.bin": "f22c6d2ee7f8359e99ac84030a27f17588ae46c069831b57cd7f0f5f97d77ff4",
    "cps8_all.bin": "e529d6ca82525073c90d374dff789f3fdc750555cc480ec55fc9304f37046c12",
    "cps8_fail.bin": "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc",
}

PASS_M6 = re.compile(
    r"^PASS hard_records=(\d+) certs=(\d+) fixed=(\d+) adaptive=(\d+) "
    r"empty_saturation_domains=(\d+) exact_price_domains=(\d+) "
    r"min_singleton_or_sum=(\S+) min_exact_price=(\S+)$"
)
PASS_M7 = re.compile(r"^PASS certs=(\d+) fixed=(\d+) adaptive=(\d+)\b")


@dataclass(frozen=True)
class Chunk:
    offset: int
    count: int
    binary: Path
    stdout: Path
    stderr: Path


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


PROVENANCE_VERSION = 1


def provenance_path(output: Path) -> Path:
    """Return the hash-binding sidecar path for a stage output."""
    return output.with_name(output.name + ".provenance.json")


def file_descriptor(path: Path, digest: str | None = None) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest if digest is not None else sha256(resolved),
    }


def stage_identity(
    stage: str,
    tool: Path,
    argv: Sequence[str | Path],
    inputs: dict[str, Path],
    parameters: dict[str, object],
    *,
    tool_digest: str | None = None,
    input_digests: dict[str, str] | None = None,
) -> dict[str, object]:
    input_digests = input_digests or {}
    return {
        "version": PROVENANCE_VERSION,
        "stage": stage,
        "tool": file_descriptor(tool, tool_digest),
        "argv": [str(x) for x in argv],
        "inputs": {
            name: file_descriptor(path, input_digests.get(name))
            for name, path in sorted(inputs.items())
        },
        "parameters": parameters,
    }


def write_provenance(
    sidecar: Path,
    identity: dict[str, object],
    outputs: dict[str, Path],
) -> None:
    payload = {
        "identity": identity,
        "outputs": {
            name: file_descriptor(path)
            for name, path in sorted(outputs.items())
        },
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, sidecar)


def provenance_matches(
    sidecar: Path,
    identity: dict[str, object],
    outputs: dict[str, Path],
) -> bool:
    if not sidecar.exists() or any(not path.exists() for path in outputs.values()):
        return False
    try:
        payload = json.loads(sidecar.read_text(errors="strict"))
        if payload.get("identity") != identity:
            return False
        recorded = payload.get("outputs")
        if not isinstance(recorded, dict) or set(recorded) != set(outputs):
            return False
        for name, path in outputs.items():
            descriptor = recorded.get(name)
            if not isinstance(descriptor, dict):
                return False
            if descriptor.get("path") != str(path.resolve()):
                return False
            if descriptor.get("bytes") != path.stat().st_size:
                return False
            if descriptor.get("sha256") != sha256(path):
                return False
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def read_u64_header(path: Path) -> int:
    with path.open("rb") as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError(f"truncated 64-bit header: {path}")
    return struct.unpack("<Q", raw)[0]


def key_count(path: Path) -> int:
    n = read_u64_header(path)
    expected = 8 + 8 * n
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"bad key-file size for {path}: {actual}, expected {expected}")
    return n


def hard_count(path: Path) -> int:
    raw = path.read_bytes()[:16]
    if len(raw) != 16:
        raise ValueError(f"truncated hard header: {path}")
    magic, n = struct.unpack("<QQ", raw)
    if magic != HARD_MAGIC:
        raise ValueError(f"bad hard magic in {path}: {magic:#x}")
    expected = 16 + HARD_RECORD_SIZE * n
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"bad hard-file size for {path}: {actual}, expected {expected}")
    return n


def cert_count(path: Path) -> int:
    n = read_u64_header(path)
    expected = 8 + CERT_RECORD_SIZE * n
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"bad certificate-file size for {path}: {actual}, expected {expected}")
    return n


def m6_cert_count(path: Path) -> int:
    raw = path.read_bytes()[:16]
    if len(raw) != 16 or raw[:8] != b"M6CERT01":
        raise ValueError(f"bad m=6 certificate header: {path}")
    n = struct.unpack_from("<Q", raw, 8)[0]
    # The explicit m=6 record is 36 bytes.
    expected = 16 + 36 * n
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"bad m=6 certificate size for {path}: {actual}, expected {expected}")
    return n


def write_zero_count(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", 0))


def ensure_count(path: Path, expected: int, kind: str) -> None:
    readers = {"keys": key_count, "hard": hard_count, "cert": cert_count, "m6cert": m6_cert_count}
    actual = readers[kind](path)
    if actual != expected:
        raise RuntimeError(f"{path}: {kind} count {actual}, expected {expected}")


def command_text(cmd: Sequence[str]) -> str:
    import shlex
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run(
    cmd: Sequence[str | Path],
    *,
    stdout: Path | None = None,
    stderr: Path | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    argv = [str(x) for x in cmd]
    eprint("+", command_text(argv))
    if stdout:
        stdout.parent.mkdir(parents=True, exist_ok=True)
    if stderr:
        stderr.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (stdout.open("w", encoding="utf-8") if stdout else open(os.devnull, "w")) as out_f, \
         (stderr.open("w", encoding="utf-8") if stderr else open(os.devnull, "w")) as err_f:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=(os.environ | env) if env else None,
            stdout=out_f if stdout else None,
            stderr=err_f if stderr else None,
            check=False,
        )
    if proc.returncode != 0:
        tail = ""
        if stderr and stderr.exists():
            lines = stderr.read_text(errors="replace").splitlines()
            tail = "\n".join(lines[-20:])
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {command_text(argv)}"
            + (f"\nlast stderr lines:\n{tail}" if tail else "")
        )
    eprint(f"  completed in {time.monotonic() - start:.2f}s")


def run_capture(cmd: Sequence[str | Path]) -> str:
    argv = [str(x) for x in cmd]
    eprint("+", command_text(argv))
    proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError(f"command failed: {command_text(argv)}\n{proc.stderr}")
    return proc.stdout


def compile_one(
    cxx: str, source: Path, output: Path, flags: Sequence[str],
    link_flags: Sequence[str] = (),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [cxx, *flags, str(source), "-o", str(output), *link_flags]
    run(cmd)


def compile_tools(root: Path, work: Path, cxx: str, max_m: int) -> dict[str, Path]:
    bin_dir = work / "bin"
    log = work / "logs" / "build.json"
    tools: dict[str, Path] = {}
    common = ["-O3", "-DNDEBUG", "-std=c++20", "-pthread"]

    tools["format_check"] = bin_dir / "n8_binary_format_selftest"
    compile_one(cxx, root / "src/n8/binary_format_selftest.cpp", tools["format_check"], common)
    format_log = work / "logs" / "binary_format_selftest.out"
    run([tools["format_check"]], stdout=format_log, stderr=work / "logs" / "binary_format_selftest.err")
    if "PASS hard_record_size=24 certificate_record_size=64" not in format_log.read_text():
        raise RuntimeError("binary format self-test did not report the required layouts")

    tools["enumerate"] = bin_dir / "n8_enumerate"
    compile_one(cxx, root / "src/n8/m8_canonical_enumerator.cpp", tools["enumerate"], common)

    for m in range(4, max_m + 1):
        tools[f"scan{m}"] = bin_dir / f"n8_scan_m{m}"
        compile_one(
            cxx,
            root / "src/n8/eight_row_floor_cell_scanner_template.cpp",
            tools[f"scan{m}"],
            [*common, f"-DMM={m}"],
        )

    # Compile only stages reachable under max_m.  The fast m=4,5 smoke test
    # should not spend time compiling the much larger m=6,7,8 templates.
    if max_m == 8:
        tools["scan8_specialized"] = bin_dir / "n8_scan_m8_specialized"
        compile_one(
            cxx,
            root / "src/n8/m8_floor_cell_scanner.cpp",
            tools["scan8_specialized"],
            common,
        )

    for m in (7, 8):
        if max_m < m:
            continue
        tools[f"propose{m}"] = bin_dir / f"n8_propose_m{m}"
        compile_one(
            cxx,
            root / "src/n8/m8_certificate_proposer.cpp",
            tools[f"propose{m}"],
            [*common, f"-DMM={m}"],
        )

    tools["record_check"] = bin_dir / "n8_record_check"
    compile_one(cxx, root / "src/n8/m8_record_checker.cpp", tools["record_check"], common)

    if max_m == 8:
        tools["kernel_check8"] = bin_dir / "n8_kernel_check_m8"
        compile_one(cxx, root / "src/n8/m8_kernel_list_verifier.cpp", tools["kernel_check8"], common)

        tools["check8_ll"] = bin_dir / "n8_exact_check_m8_ll"
        compile_one(
            cxx,
            root / "src/n8/m8_exact_checker_ll.cpp",
            tools["check8_ll"],
            ["-O2", "-DNDEBUG", "-std=c++20"],
        )

        tools["check8_gmp"] = bin_dir / "n8_exact_check_m8_gmp"
        compile_one(
            cxx,
            root / "src/n8/m8_exact_checker_gmp.cpp",
            tools["check8_gmp"],
            ["-O2", "-DNDEBUG", "-std=c++20"],
            ["-lgmpxx", "-lgmp"],
        )

    if max_m >= 7:
        tools["check7"] = bin_dir / "n8_exact_check_m7"
        compile_one(
            cxx,
            root / "src/n8/exact_cps_fullsat_checker.cpp",
            tools["check7"],
            ["-O2", "-DNDEBUG", "-std=c++20", "-DMM=7"],
        )

    if max_m >= 6:
        tools["check6"] = bin_dir / "n8_exact_check_m6"
        compile_one(
            cxx,
            root / "src/n8/m6_exact_certificate_checker.cpp",
            tools["check6"],
            ["-O2", "-DNDEBUG", "-std=c++20"],
            ["-lgmpxx", "-lgmp"],
        )

    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({k: str(v) for k, v in tools.items()}, indent=2, sort_keys=True) + "\n")
    return tools

def maybe_run_key_stage(
    cmd: Sequence[str | Path],
    output: Path,
    expected: int,
    stdout: Path,
    stderr: Path,
    resume: bool,
    *,
    inputs: dict[str, Path],
) -> None:
    tool = Path(cmd[0])
    identity = stage_identity(
        "canonical_key_generation",
        tool,
        cmd,
        inputs,
        {"expected_keys": expected},
    )
    sidecar = provenance_path(output)
    if resume and output.exists():
        try:
            ensure_count(output, expected, "keys")
            if not provenance_matches(sidecar, identity, {"keys": output, "stdout": stdout}):
                raise ValueError("missing or mismatched hash-binding provenance")
            eprint(f"reuse {output} ({expected} keys; hash-bound)")
            return
        except Exception as exc:
            eprint(f"cannot reuse {output}: {exc}; regenerating")
    output.parent.mkdir(parents=True, exist_ok=True)
    run(cmd, stdout=stdout, stderr=stderr)
    ensure_count(output, expected, "keys")
    write_provenance(sidecar, identity, {"keys": output, "stdout": stdout})

def enumerate_hierarchy(
    root: Path,
    work: Path,
    tools: dict[str, Path],
    jobs: int,
    max_m: int,
    resume: bool,
    check_reference_hashes: bool,
) -> dict[int, Path]:
    enum = tools["enumerate"]
    logs = work / "logs" / "enumeration"
    data = work / "data"
    pos: dict[int, Path] = {}

    m4 = data / "m4"
    all4, pos4 = m4 / "n8m4_all.bin", m4 / "n8m4_pos.bin"
    maybe_run_key_stage(
        [enum, "direct", "8", "4", str(jobs), all4], all4, EXPECTED[4]["full"],
        logs / "m4_direct.out", logs / "m4_direct.err", resume,
        inputs={},
    )
    maybe_run_key_stage(
        [enum, "positive", "8", "4", str(jobs), all4, pos4], pos4, EXPECTED[4]["kernels"],
        logs / "m4_positive.out", logs / "m4_positive.err", resume,
        inputs={"all_keys": all4},
    )
    pos[4] = pos4
    if max_m == 4:
        return pos

    m5 = data / "m5"
    all5 = m5 / "n8m5_all_direct.bin"
    pos5_direct = m5 / "n8m5_pos_direct.bin"
    pos5_aug = m5 / "n8m5_pos_augmented.bin"
    maybe_run_key_stage(
        [enum, "direct", "8", "5", str(jobs), all5], all5, EXPECTED[5]["full"],
        logs / "m5_direct.out", logs / "m5_direct.err", resume,
        inputs={},
    )
    maybe_run_key_stage(
        [enum, "positive", "8", "5", str(jobs), all5, pos5_direct],
        pos5_direct, EXPECTED[5]["kernels"], logs / "m5_positive_direct.out",
        logs / "m5_positive_direct.err", resume,
        inputs={"all_keys": all5},
    )
    maybe_run_key_stage(
        [enum, "extendpos", "8", "4", str(jobs), pos4, pos5_aug],
        pos5_aug, EXPECTED[5]["kernels"], logs / "m5_positive_augmented.out",
        logs / "m5_positive_augmented.err", resume,
        inputs={"positive_parent": pos4},
    )
    if pos5_direct.read_bytes() != pos5_aug.read_bytes():
        raise RuntimeError("direct and canonical-augmentation m=5 positive lists differ")
    canonical5 = m5 / "n8m5_pos.bin"
    if not canonical5.exists() or canonical5.read_bytes() != pos5_aug.read_bytes():
        shutil.copyfile(pos5_aug, canonical5)
    pos[5] = canonical5

    for m in range(6, max_m + 1):
        out = data / f"m{m}" / f"n8m{m}_pos.bin"
        maybe_run_key_stage(
            [enum, "extendpos", "8", str(m - 1), str(jobs), pos[m - 1], out],
            out, EXPECTED[m]["kernels"], logs / f"m{m}_positive_augmented.out",
            logs / f"m{m}_positive_augmented.err", resume,
            inputs={"positive_parent": pos[m - 1]},
        )
        pos[m] = out
        if m == 6:
            reference = root / "data/n8/m6/n8m6_pos.bin"
            if reference.exists() and out.read_bytes() != reference.read_bytes():
                raise RuntimeError("regenerated m=6 positive list differs from supplied exact list")
        if m == 8 and check_reference_hashes:
            actual = sha256(out)
            expected = M8_HASHES[out.name]
            if actual != expected:
                raise RuntimeError(f"m=8 positive-list hash mismatch: {actual} != {expected}")
    return pos


def scan_chunks(total: int, chunk_size: int, directory: Path) -> list[Chunk]:
    out: list[Chunk] = []
    for offset in range(0, total, chunk_size):
        count = min(chunk_size, total - offset)
        tag = f"{offset:07d}"
        out.append(Chunk(
            offset=offset,
            count=count,
            binary=directory / f"chunk_{tag}.bin",
            stdout=directory / f"chunk_{tag}.out",
            stderr=directory / f"chunk_{tag}.err",
        ))
    return out


def scan_log_matches(path: Path, offset: int, count: int) -> bool:
    if not path.exists():
        return False
    lines = path.read_text(errors="replace").splitlines()
    if not lines:
        return False
    m = re.match(r"^input=.*? offset=(\d+) matrices=(\d+) records=(\d+)$", lines[0])
    return bool(m and int(m.group(1)) == offset and int(m.group(2)) == count)


def scan_level(
    root: Path,
    work: Path,
    tools: dict[str, Path],
    m: int,
    pos: Path,
    jobs: int,
    chunk_size: int,
    resume: bool,
    check_reference_hashes: bool,
) -> tuple[Path, Path, list[Chunk]]:
    total = EXPECTED[m]["kernels"]
    scanner = tools[f"scan{m}"]
    pos_digest = sha256(pos)
    scanner_digest = sha256(scanner)
    chunk_dir = work / "chunks" / f"m{m}" / "scan"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = scan_chunks(total, chunk_size, chunk_dir)
    layout = {
        "m": m,
        "total": total,
        "chunk_size": chunk_size,
        "chunks": [{"offset": ch.offset, "count": ch.count} for ch in chunks],
    }
    plan = {
        **layout,
        "version": PROVENANCE_VERSION,
        "positive_kernel_sha256": pos_digest,
        "scanner_sha256": scanner_digest,
    }
    plan_path = chunk_dir / "scan_plan.json"

    def clear_chunk_outputs(reason: str) -> None:
        stale = [path for path in chunk_dir.glob("chunk_*") if path.is_file()]
        if stale:
            eprint(f"invalidate {len(stale)} scan files for m={m}: {reason}")
        for path in stale:
            path.unlink()

    if plan_path.exists():
        old_plan = json.loads(plan_path.read_text(errors="strict"))
        old_layout = {key: old_plan.get(key) for key in layout}
        if old_layout != layout:
            raise RuntimeError(
                f"scan plan mismatch for m={m}; resume with the original CHUNK_SIZE "
                f"or remove {chunk_dir}"
            )
        if (
            old_plan.get("version") != PROVENANCE_VERSION
            or old_plan.get("positive_kernel_sha256") != pos_digest
            or old_plan.get("scanner_sha256") != scanner_digest
        ):
            clear_chunk_outputs("positive-kernel or scanner fingerprint changed")
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        else:
            allowed: set[Path] = set()
            for ch in chunks:
                allowed.update((ch.binary, ch.stdout, ch.stderr, provenance_path(ch.binary)))
            actual = {path for path in chunk_dir.glob("chunk_*") if path.is_file()}
            extra = sorted(str(path) for path in actual - allowed)
            if extra:
                raise RuntimeError(f"scan directory contains files outside the recorded plan: {extra}")
    else:
        # Legacy chunks have no cryptographic binding to their kernel input.
        # They cannot safely be adopted; discard them and establish a new plan.
        clear_chunk_outputs("legacy or unplanned chunks are not hash-bound")
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    for ch in chunks:
        cmd: list[str | Path] = [
            scanner, pos, ch.binary, str(jobs), str(ch.count), str(ch.offset)
        ]
        identity = stage_identity(
            "floor_cell_scan_chunk",
            scanner,
            cmd,
            {"positive_kernels": pos},
            {
                "m": m,
                "offset": ch.offset,
                "count": ch.count,
                "total": total,
                "jobs": jobs,
            },
            tool_digest=scanner_digest,
            input_digests={"positive_kernels": pos_digest},
        )
        sidecar = provenance_path(ch.binary)
        reusable = False
        if resume and ch.binary.exists() and scan_log_matches(ch.stdout, ch.offset, ch.count):
            try:
                hard_count(ch.binary)
                reusable = provenance_matches(
                    sidecar,
                    identity,
                    {"hard_records": ch.binary, "stdout": ch.stdout},
                )
            except Exception:
                reusable = False
        if reusable:
            eprint(f"reuse scan m={m} offset={ch.offset} count={ch.count} (hash-bound)")
            continue
        run(cmd, stdout=ch.stdout, stderr=ch.stderr)
        hard_count(ch.binary)
        if not scan_log_matches(ch.stdout, ch.offset, ch.count):
            raise RuntimeError(f"scanner emitted an invalid coverage header: {ch.stdout}")
        write_provenance(
            sidecar,
            identity,
            {"hard_records": ch.binary, "stdout": ch.stdout},
        )

    out_dir = work / "data" / f"m{m}"
    hard = out_dir / f"n8m{m}_hard.bin"
    summary = work / "summaries" / f"m{m}_scan_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    merge_log = work / "logs" / "scan" / f"m{m}_merge.out"
    run(
        [sys.executable, root / "src/n8/merge_and_summarize.py", chunk_dir, hard, summary,
         "--total", str(total)],
        stdout=merge_log,
        stderr=work / "logs" / "scan" / f"m{m}_merge.err",
    )

    summary_data = json.loads(summary.read_text())
    feasible = sum(
        int(value.get("cell_feasible", 0))
        for value in summary_data["by_residual_budget"].values()
    )
    cases = sum(
        int(value.get("cases", 0))
        for value in summary_data["by_residual_budget"].values()
    )
    if cases != EXPECTED[m]["cases"]:
        raise RuntimeError(f"m={m} matrix-budget cases {cases}, expected {EXPECTED[m]['cases']}")
    if feasible != EXPECTED[m]["feasible"]:
        raise RuntimeError(f"m={m} feasible cells {feasible}, expected {EXPECTED[m]['feasible']}")
    ensure_count(hard, EXPECTED[m]["hard"], "hard")

    if m == 6:
        reference = root / "data/n8/m6/n8m6_hard_exact.bin"
        if reference.exists() and hard.read_bytes() != reference.read_bytes():
            raise RuntimeError("regenerated m=6 hard-cell file differs from supplied exact file")
    if m == 7:
        reference = root / "data/n8/m7/n8m7_unresolved_exact_ll.bin"
        if reference.exists() and hard.read_bytes() != reference.read_bytes():
            raise RuntimeError("regenerated m=7 hard-cell file differs from supplied exact file")
    if m == 8 and check_reference_hashes:
        actual = sha256(hard)
        expected = M8_HASHES[hard.name]
        if actual != expected:
            raise RuntimeError(f"m=8 hard-cell hash mismatch: {actual} != {expected}")
    return hard, summary, chunks

def run_parallel(tasks: Iterable[tuple[Sequence[str | Path], Path, Path]], workers: int) -> None:
    task_list = list(tasks)
    if not task_list:
        return

    def one(task: tuple[Sequence[str | Path], Path, Path]) -> None:
        cmd, stdout, stderr = task
        run(cmd, stdout=stdout, stderr=stderr)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, task) for task in task_list]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def parse_single_pass(path: Path, pattern: re.Pattern[str]) -> re.Match[str]:
    hits = [m for line in path.read_text(errors="strict").splitlines() if (m := pattern.match(line))]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one matching PASS line in {path}, found {len(hits)}")
    return hits[0]


def certify_m4_m5(work: Path, tools: dict[str, Path], m: int, hard: Path) -> None:
    out = work / "data" / f"m{m}"
    cert = out / f"cps{m}_all.bin"
    fail = out / f"cps{m}_fail.bin"
    write_zero_count(cert)
    write_zero_count(fail)
    log = work / "logs" / "certificates" / f"m{m}_record_check.out"
    run([tools["record_check"], hard, cert, fail], stdout=log,
        stderr=work / "logs" / "certificates" / f"m{m}_record_check.err")


def certify_m6(root: Path, work: Path, tools: dict[str, Path], hard: Path) -> dict[str, object]:
    out = work / "data/m6"
    cert = out / "m6_certificates_exact.bin"
    make_log = work / "logs/certificates/m6_make.out"
    run(
        [sys.executable, root / "src/n8/m6_make_certificates.py", hard,
         root / "data/n8/m6/m6_unresolved_certificate_summary.csv", cert],
        stdout=make_log,
        stderr=work / "logs/certificates/m6_make.err",
    )
    ensure_count(cert, EXPECTED[6]["hard"], "m6cert")
    check_log = work / "logs/certificates/m6_exact.out"
    run([tools["check6"], hard, cert], stdout=check_log,
        stderr=work / "logs/certificates/m6_exact.err")
    text = check_log.read_text(errors="strict")
    if "PASS" not in text or "fixed=163" not in text or "adaptive=5" not in text:
        raise RuntimeError(f"unexpected m=6 exact replay output:\n{text}")
    return {"certificate_file": str(cert), "sha256": sha256(cert), "log": str(check_log)}


def certify_m7(work: Path, tools: dict[str, Path], hard: Path) -> dict[str, object]:
    out = work / "data/m7"
    cert = out / "cps7_all.bin"
    fail = out / "cps7_fail.bin"
    propose_log = work / "logs/certificates/m7_propose.out"
    run([tools["propose7"], hard, cert, fail], stdout=propose_log,
        stderr=work / "logs/certificates/m7_propose.err")
    ensure_count(cert, EXPECTED[7]["hard"], "cert")
    ensure_count(fail, 0, "cert")
    run([tools["record_check"], hard, cert, fail],
        stdout=work / "logs/certificates/m7_record_check.out",
        stderr=work / "logs/certificates/m7_record_check.err")
    exact_log = work / "logs/certificates/m7_exact.out"
    run([tools["check7"], cert], stdout=exact_log,
        stderr=work / "logs/certificates/m7_exact.err")
    match = parse_single_pass(exact_log, PASS_M7)
    certs, fixed, adaptive = map(int, match.groups())
    if certs != EXPECTED[7]["hard"] or fixed + adaptive != certs:
        raise RuntimeError(f"unexpected m=7 replay census: {(certs, fixed, adaptive)}")
    return {
        "certificate_file": str(cert), "failure_file": str(fail),
        "certificate_sha256": sha256(cert), "failure_sha256": sha256(fail),
        "fixed": fixed, "adaptive": adaptive, "log": str(exact_log),
    }


def certificate_chunk_paths(work: Path, scan_chunks_: list[Chunk]) -> list[tuple[Chunk, Path, Path, Path, Path]]:
    directory = work / "chunks/m8/certificates"
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for ch in scan_chunks_:
        tag = f"{ch.offset:07d}"
        result.append((
            ch,
            directory / f"cert_{tag}.bin",
            directory / f"fail_{tag}.bin",
            directory / f"propose_{tag}.out",
            directory / f"propose_{tag}.err",
        ))
    return result


def certify_m8(
    root: Path,
    work: Path,
    tools: dict[str, Path],
    hard: Path,
    scan_summary: Path,
    chunks: list[Chunk],
    jobs: int,
    resume: bool,
    skip_gmp: bool,
    check_reference_hashes: bool,
) -> dict[str, object]:
    cert_chunks = certificate_chunk_paths(work, chunks)
    proposer = tools["propose8"]
    proposer_digest = sha256(proposer)
    proposal_tasks: list[tuple[Sequence[str | Path], Path, Path]] = []
    proposal_meta: list[
        tuple[Chunk, Path, Path, Path, dict[str, object], Path]
    ] = []

    for ch, cert_chunk, fail_chunk, stdout, stderr in cert_chunks:
        hard_digest = sha256(ch.binary)
        cmd: list[str | Path] = [proposer, ch.binary, cert_chunk, fail_chunk]
        identity = stage_identity(
            "m8_certificate_proposal_chunk",
            proposer,
            cmd,
            {"hard_records": ch.binary},
            {"offset": ch.offset, "hard_records": hard_count(ch.binary)},
            tool_digest=proposer_digest,
            input_digests={"hard_records": hard_digest},
        )
        sidecar = provenance_path(cert_chunk)
        proposal_meta.append((ch, cert_chunk, fail_chunk, stdout, identity, sidecar))
        reusable = False
        if resume and cert_chunk.exists() and fail_chunk.exists() and stdout.exists():
            try:
                reusable = (
                    cert_count(cert_chunk) + cert_count(fail_chunk) == hard_count(ch.binary)
                    and provenance_matches(
                        sidecar,
                        identity,
                        {
                            "certificates": cert_chunk,
                            "failures": fail_chunk,
                            "stdout": stdout,
                        },
                    )
                )
            except Exception:
                reusable = False
        if reusable:
            eprint(f"reuse certificates offset={ch.offset} (hash-bound)")
        else:
            proposal_tasks.append((cmd, stdout, stderr))
    run_parallel(proposal_tasks, jobs)

    # Bind every proposal output to the precise hard-record chunk, executable,
    # and command parameters before it can be reused.
    for ch, cert_chunk, fail_chunk, stdout, identity, sidecar in proposal_meta:
        total = cert_count(cert_chunk) + cert_count(fail_chunk)
        expected_total = hard_count(ch.binary)
        if total != expected_total:
            raise RuntimeError(
                f"certificate proposal offset={ch.offset} covers {total} records, "
                f"expected {expected_total}"
            )
        if not stdout.exists():
            raise RuntimeError(f"missing proposer log for offset={ch.offset}: {stdout}")
        write_provenance(
            sidecar,
            identity,
            {
                "certificates": cert_chunk,
                "failures": fail_chunk,
                "stdout": stdout,
            },
        )

    out_dir = work / "data/m8"
    cert = out_dir / "cps8_all.bin"
    fail = out_dir / "cps8_fail.bin"
    cert_summary = work / "summaries/m8_certificate_summary.json"
    run(
        [sys.executable, root / "src/n8/merge_certificates.py", scan_summary,
         work / "chunks/m8/certificates", cert, fail, cert_summary],
        stdout=work / "logs/certificates/m8_merge.out",
        stderr=work / "logs/certificates/m8_merge.err",
    )
    ensure_count(cert, EXPECTED[8]["hard"], "cert")
    ensure_count(fail, 0, "cert")
    run([tools["record_check"], hard, cert, fail],
        stdout=work / "logs/certificates/m8_record_check.out",
        stderr=work / "logs/certificates/m8_record_check.err")

    if check_reference_hashes:
        # Search is proposal-only.  A platform may choose different committees;
        # complete coverage plus independent exact replay is authoritative.
        for path in (cert, fail):
            actual, expected = sha256(path), M8_HASHES[path.name]
            if actual != expected:
                eprint(
                    f"NOTE: {path.name} differs from reference hash ({actual}); "
                    "exact replay, not proposal identity, decides validity"
                )

    check_dir = work / "chunks/m8/checks"
    check_dir.mkdir(parents=True, exist_ok=True)
    cert_hashes = {ch.offset: sha256(cfile) for ch, cfile, *_ in cert_chunks}

    def run_exact_backend(name: str, checker: Path) -> tuple[Path, dict[str, object]]:
        checker_digest = sha256(checker)
        tasks: list[tuple[Sequence[str | Path], Path, Path]] = []
        metadata: list[tuple[Path, Path, dict[str, object]]] = []
        expected_logs: set[str] = set()
        for ch, cfile, _ffile, _proposal_out, _proposal_err in cert_chunks:
            tag = f"{ch.offset:07d}"
            stdout = check_dir / f"check_{name}_{tag}.out"
            stderr = check_dir / f"check_{name}_{tag}.err"
            sidecar = provenance_path(stdout)
            cmd: list[str | Path] = [checker, cfile]
            identity = stage_identity(
                f"m8_exact_replay_{name}",
                checker,
                cmd,
                {"certificates": cfile},
                {"offset": ch.offset, "certificate_records": cert_count(cfile)},
                tool_digest=checker_digest,
                input_digests={"certificates": cert_hashes[ch.offset]},
            )
            expected_logs.add(stdout.name)
            metadata.append((stdout, sidecar, identity))
            reusable = (
                resume
                and stdout.exists()
                and stdout.read_text(errors="replace").count("PASS certs=") == 1
                and provenance_matches(sidecar, identity, {"stdout": stdout})
            )
            if not reusable:
                tasks.append((cmd, stdout, stderr))
            else:
                eprint(f"reuse {name} exact replay offset={ch.offset} (hash-bound)")
        run_parallel(tasks, jobs)
        for stdout, sidecar, identity in metadata:
            if stdout.read_text(errors="replace").count("PASS certs=") != 1:
                raise RuntimeError(f"missing unique {name} PASS line in {stdout}")
            write_provenance(sidecar, identity, {"stdout": stdout})
        actual_logs = {path.name for path in check_dir.glob(f"check_{name}_*.out")}
        if actual_logs != expected_logs:
            raise RuntimeError(
                f"unexpected {name} replay logs: {sorted(actual_logs ^ expected_logs)}"
            )
        summary = work / f"summaries/m8_exact_{name}_summary.json"
        run(
            [sys.executable, root / "src/n8/aggregate_checker_logs.py", check_dir,
             f"check_{name}_*.out", summary],
            stdout=work / f"logs/certificates/m8_{name}_aggregate.out",
            stderr=work / f"logs/certificates/m8_{name}_aggregate.err",
        )
        return summary, json.loads(summary.read_text())

    ll_summary, ll = run_exact_backend("ll", tools["check8_ll"])
    certs = int(ll.get("certs", -1))
    fixed = int(ll.get("fixed", -1))
    adaptive = int(ll.get("adaptive", -1))
    if certs != EXPECTED[8]["hard"] or fixed + adaptive != certs:
        raise RuntimeError(
            f"signed-rational replay has invalid census: certs={certs}, "
            f"fixed={fixed}, adaptive={adaptive}"
        )
    if ll.get("min_singleton_or_sum") in (None, "NA"):
        raise RuntimeError("signed-rational replay did not report a positive singleton/adaptive margin")
    if int(ll.get("exact_price_LPs", 0)) > 0 and ll.get("min_exact_price") in (None, "NA"):
        raise RuntimeError("signed-rational replay did not report a positive exact price margin")

    gmp_summary: Path | None = None
    if not skip_gmp:
        gmp_summary, gmp = run_exact_backend("gmp", tools["check8_gmp"])
        proof_fields = [
            "certs", "fixed", "adaptive", "saturation_skips", "open_floor_checks",
            "exact_price_LPs", "min_singleton_or_sum", "min_exact_price",
        ]
        if any(ll.get(key) != gmp.get(key) for key in proof_fields):
            raise RuntimeError("signed-rational and GMP replay summaries disagree")

    return {
        "certificate_file": str(cert),
        "failure_file": str(fail),
        "certificate_sha256": sha256(cert),
        "failure_sha256": sha256(fail),
        "certificate_summary": str(cert_summary),
        "ll_summary": str(ll_summary),
        "gmp_summary": str(gmp_summary) if gmp_summary else None,
        "fixed": fixed,
        "adaptive": adaptive,
    }

def kernel_audit_m8(work: Path, tools: dict[str, Path], pos: Path) -> str:
    log = work / "logs/verification/m8_kernel_audit.out"
    run([tools["kernel_check8"], pos], stdout=log,
        stderr=work / "logs/verification/m8_kernel_audit.err")
    expected = (
        "PASS kernels=9105190 sorted_unique=1 antichain=1 full_rank=1 positive_dual=1 "
        "max_abs_det=56 max_abs_cramer=32 min_abs_det=1"
    )
    if expected not in log.read_text(errors="strict"):
        raise RuntimeError(f"unexpected m=8 kernel audit output: {log.read_text(errors='replace')}")
    return str(log)


def write_manifest(work: Path) -> Path:
    manifest = work / "SHA256SUMS"
    lines = []
    for path in sorted(work.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        rel = path.relative_to(work)
        # Skip compiler binaries and transient stderr progress logs from the proof-data manifest.
        if rel.parts and rel.parts[0] == "bin":
            continue
        lines.append(f"{sha256(path)}  {rel.as_posix()}")
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def smoke_compare_m8_scanners(work: Path, tools: dict[str, Path], pos8: Path, jobs: int) -> None:
    # A small prefix catches divergence between the generic and copied m=8 scanner.
    directory = work / "logs/verification/scanner_crosscheck"
    directory.mkdir(parents=True, exist_ok=True)
    limit = min(1000, key_count(pos8))
    a, b = directory / "template.bin", directory / "specialized.bin"
    out_a, out_b = directory / "template.out", directory / "specialized.out"
    run([tools["scan8"], pos8, a, str(jobs), str(limit), "0"],
        stdout=out_a, stderr=directory / "template.err")
    run([tools["scan8_specialized"], pos8, b, str(jobs), str(limit), "0"],
        stdout=out_b, stderr=directory / "specialized.err")
    if a.read_bytes() != b.read_bytes():
        raise RuntimeError("generic and specialized m=8 scanners disagree on the first 1,000 kernels")
    stable_a = [line for line in out_a.read_text(errors="strict").splitlines()
                if not line.startswith("seconds=")]
    stable_b = [line for line in out_b.read_text(errors="strict").splitlines()
                if not line.startswith("seconds=")]
    if stable_a != stable_b:
        raise RuntimeError("generic and specialized m=8 scanner census logs disagree")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path, help="new output directory (or existing with --resume)")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", max(1, os.cpu_count() or 1))))
    ap.add_argument("--chunk-size", type=int, default=int(os.environ.get("CHUNK_SIZE", "100000")))
    ap.add_argument("--max-m", type=int, choices=range(4, 9), default=int(os.environ.get("MAX_M", "8")))
    ap.add_argument("--resume", action="store_true", default=os.environ.get("RESUME") == "1")
    ap.add_argument("--skip-gmp", action="store_true", default=os.environ.get("SKIP_GMP") == "1")
    ap.add_argument("--no-reference-hashes", action="store_true",
                    help="do not require the reference m=8 positive/hard hashes")
    ap.add_argument("--stop-after", choices=("build", "enumerate", "scan", "certify", "verify"))
    args = ap.parse_args()

    if args.jobs < 1 or args.chunk_size < 1:
        ap.error("--jobs and --chunk-size must be positive")

    root = Path(__file__).resolve().parent.parent
    work = args.output.resolve()
    if work.exists() and not args.resume:
        raise SystemExit(f"ERROR: destination already exists: {work}; pass --resume to continue")
    work.mkdir(parents=True, exist_ok=True)
    for d in ("data", "logs", "summaries", "chunks", "bin"):
        (work / d).mkdir(exist_ok=True)

    cxx = os.environ.get("CXX", "g++")
    if shutil.which(cxx) is None:
        raise SystemExit(f"ERROR: C++ compiler not found: {cxx}")
    if shutil.which("python3") is None:
        raise SystemExit("ERROR: python3 not found")

    start = time.monotonic()
    tools = compile_tools(root, work, cxx, args.max_m)
    if args.stop_after == "build":
        print(f"PASS n8_build tools={len(tools)}")
        return

    pos = enumerate_hierarchy(
        root, work, tools, args.jobs, args.max_m, args.resume, not args.no_reference_hashes
    )
    if args.stop_after == "enumerate":
        print(f"PASS n8_enumeration max_m={args.max_m}")
        return

    # Fail early on a malformed square-kernel universe or divergence between
    # the generic and specialized m=8 scanners, before launching the expensive
    # full floor-cell scan.
    audits: dict[str, str] = {}
    if args.max_m == 8:
        audits["m8_kernel"] = kernel_audit_m8(work, tools, pos[8])
        smoke_compare_m8_scanners(work, tools, pos[8], args.jobs)

    scan_outputs: dict[int, tuple[Path, Path, list[Chunk]]] = {}
    for m in range(4, args.max_m + 1):
        scan_outputs[m] = scan_level(
            root, work, tools, m, pos[m], args.jobs, args.chunk_size, args.resume,
            not args.no_reference_hashes,
        )
    if args.stop_after == "scan":
        print(f"PASS n8_scan max_m={args.max_m}")
        return

    cert_results: dict[int, dict[str, object]] = {}
    for m in range(4, args.max_m + 1):
        hard, summary, chunks = scan_outputs[m]
        if m <= 5:
            certify_m4_m5(work, tools, m, hard)
            cert_results[m] = {"hard": 0, "certificate_records": 0}
        elif m == 6:
            cert_results[m] = certify_m6(root, work, tools, hard)
        elif m == 7:
            cert_results[m] = certify_m7(work, tools, hard)
        elif m == 8:
            cert_results[m] = certify_m8(
                root, work, tools, hard, summary, chunks, args.jobs, args.resume,
                args.skip_gmp, not args.no_reference_hashes,
            )
    if args.stop_after == "certify":
        print(f"PASS n8_certificates max_m={args.max_m}")
        return

    result = {
        "status": "PASS",
        "max_m": args.max_m,
        "jobs": args.jobs,
        "chunk_size": args.chunk_size,
        "gmp_replay": not args.skip_gmp,
        "elapsed_seconds": time.monotonic() - start,
        "levels": {},
        "certificate_results": cert_results,
        "audits": audits,
    }
    for m in range(4, args.max_m + 1):
        hard, summary, _ = scan_outputs[m]
        result["levels"][str(m)] = {
            "positive_kernels": key_count(pos[m]),
            "positive_sha256": sha256(pos[m]),
            "hard_cells": hard_count(hard),
            "hard_sha256": sha256(hard),
            "scan_summary": str(summary),
        }
    out_summary = work / "summaries/eight_voter_regeneration.json"
    out_summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    manifest = write_manifest(work)

    if args.max_m == 8 and args.skip_gmp:
        print("PASS n=8 regeneration and signed-rational replay; GMP replay was explicitly skipped")
    elif args.max_m == 8:
        print("PASS n=8 m4,m5,m6,m7,m8 regeneration, record coverage, and dual exact replay")
    else:
        print(f"PASS n=8 bounded smoke/regeneration through m={args.max_m}")
    print(f"summary={out_summary}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        eprint(f"ERROR: {exc}")
        raise SystemExit(1)
