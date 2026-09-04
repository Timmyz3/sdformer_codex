#!/usr/bin/env python3
"""Run and seal one parameterized Local5 identity/phase RTL canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260810


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_to_file(argv: list[str], stdout_path: Path, cwd: Path | None = None) -> None:
    with stdout_path.open("wb") as stdout:
        subprocess.run(argv, cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT, check=True)


def release_build(release: Path, heads: int, hold_cycles: int) -> dict[str, Path]:
    manifest_path = release / "release_manifest.json"
    manifest = read_json(manifest_path)
    build = manifest.get("builds", {}).get(str(heads))
    if not isinstance(build, dict) or build.get("service_mode") != "identity_derived":
        raise ValueError(f"release lacks identity-derived H{heads} build: {release}")
    compile_argv_path = release / build["compile_argv_path"]
    executable_path = release / build["executable_path"]
    compile_argv = json.loads(compile_argv_path.read_text(encoding="utf-8"))
    if not isinstance(compile_argv, list) or not all(
        isinstance(item, str) for item in compile_argv
    ):
        raise ValueError("compile argv is not an exact string list")
    hold_flags = [
        item for item in compile_argv
        if item.startswith("-GFORCE_WEIGHT_RESPONSE_HOLD_CYCLES=")
    ]
    observed_hold = int(hold_flags[0].split("=", 1)[1]) if hold_flags else 0
    if len(hold_flags) > 1 or observed_hold != hold_cycles:
        raise ValueError(
            f"release H{heads} hold mismatch: expected {hold_cycles}, got {observed_hold}"
        )
    if sha256(executable_path) != build["executable_sha256"]:
        raise ValueError("release executable SHA differs")
    if sha256(compile_argv_path) != build["compile_argv_sha256"]:
        raise ValueError("release compile argv SHA differs")
    return {
        "manifest": manifest_path,
        "release_script": release / "source/scripts/local5_erep_numeric_release.py",
        "executable": executable_path,
        "compile_argv": compile_argv_path,
    }


def run_release_verify(release: Path, build: dict[str, Path], output: Path) -> None:
    run_to_file(
        [
            sys.executable, str(build["release_script"]), "verify",
            "--release-dir", str(release),
        ],
        output,
        cwd=Path("/tmp"),
    )


def run_rtl(
    name: str,
    build: dict[str, Path],
    identity: dict[str, int],
    table_dir: Path,
    vector_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    manifest_sha = sha256(table_dir / "manifest.json")
    receipt_sha = sha256(table_dir / "verification_receipt.json")
    trace = output_dir / f"{name}_trace.csv"
    actual = output_dir / f"{name}_actual.memh"
    log = output_dir / f"{name}_verilator.log"
    timing = output_dir / f"{name}_verilator_time.txt"
    argv_path = output_dir / f"{name}_run_argv.json"
    argv = [
        str(build["executable"]),
        f"+INPUTS={vector_dir / 'vectors/combined_head_inputs.txt'}",
        f"+WEIGHTS={vector_dir / 'vectors/projection_weights.txt'}",
        f"+STAGE_ID={identity['stage']}",
        f"+BLOCK_ID={identity['block']}",
        f"+WINDOW_ID={identity['window']}",
        "+NO_ACC_CHECK",
        f"+SERVICE_SEED={SEED}",
        f"+RELATION_DELAY_MEMH={table_dir / 'relation_delay.memh'}",
        f"+WEIGHT_DELAY_MEMH={table_dir / 'weight_delay.memh'}",
        f"+FINAL_DELAY_MEMH={table_dir / 'final_delay.memh'}",
        f"+IDENTITY_MANIFEST_SHA={manifest_sha}",
        f"+IDENTITY_RECEIPT_SHA={receipt_sha}",
        f"+IDENTITY_TRACE={trace}",
        f"+ACTUAL_ACC_FILE={actual}",
    ]
    write_json(argv_path, argv)
    run_to_file(
        [
            "/usr/bin/time", "-f", "wall_seconds=%e\nmax_rss_kb=%M",
            "-o", str(timing), *argv,
        ],
        log,
    )
    return {
        "trace": trace, "actual": actual, "log": log,
        "timing": timing, "run_argv": argv_path,
    }


def chmod_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    mode = root.stat().st_mode
    root.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-release", type=Path, required=True)
    parser.add_argument("--candidate-release", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-hold-cycles", type=int, default=2)
    args = parser.parse_args()
    baseline_release = args.baseline_release.resolve()
    candidate_release = args.candidate_release.resolve()
    table_dir = args.table_dir.resolve()
    vector_dir = args.vector_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory exists: {output_dir}")
    output_dir.mkdir(parents=True)

    table_manifest = read_json(table_dir / "manifest.json")
    identity = table_manifest.get("identity")
    if (
        not isinstance(identity, dict)
        or set(identity) != {"sample", "stage", "block", "window", "heads"}
        or table_manifest.get("formal_g0") != "DENY"
    ):
        raise ValueError("table identity/formal boundary differs")
    task_plan = vector_dir / "software_expected/task_plan.json"
    if sha256(task_plan) != table_manifest.get("task_plan_sha256"):
        raise ValueError("vector task plan differs from identity table task plan")
    heads = int(identity["heads"])

    baseline_build = release_build(baseline_release, heads, 0)
    candidate_build = release_build(
        candidate_release, heads, args.candidate_hold_cycles
    )
    run_release_verify(
        baseline_release, baseline_build, output_dir / "baseline_release_preverify.json"
    )
    run_release_verify(
        candidate_release, candidate_build, output_dir / "candidate_release_preverify.json"
    )
    table_verifier = table_dir / "source/verify_local5_identity_service_tables_v4.py"
    run_to_file(
        [sys.executable, str(table_verifier), "--package-dir", str(table_dir), "verify"],
        output_dir / "table_preverify.json",
        cwd=Path("/tmp"),
    )

    baseline = run_rtl(
        "baseline", baseline_build, identity, table_dir, vector_dir, output_dir
    )
    candidate = run_rtl(
        "candidate", candidate_build, identity, table_dir, vector_dir, output_dir
    )

    trace_verifier = ROOT / "scripts/verify_local5_identity_service_rtl_trace_v2.py"
    expected = vector_dir / "software_expected/software_expected.npz"
    for name, rows, hold in (
        ("baseline", baseline, 0),
        ("candidate", candidate, args.candidate_hold_cycles),
    ):
        run_to_file(
            [
                sys.executable, str(trace_verifier),
                "--trace", str(rows["trace"]),
                "--package-dir", str(table_dir),
                "--expected-weight-hold-cycles", str(hold),
                "--actual", str(rows["actual"]),
                "--expected", str(expected),
                "--verilator-log", str(rows["log"]),
                "--output", str(output_dir / f"{name}_trace_verification.json"),
            ],
            output_dir / f"{name}_trace_verification_stdout.json",
        )

    generator = ROOT / "scripts/generate_local5_h3_phase_template_patch_v1.py"
    phase_verifier = ROOT / "scripts/verify_local5_h3_phase_template_patch_v1.py"
    template_dir = output_dir / "template_patch"
    run_to_file(
        [
            sys.executable, str(generator),
            "--trace", str(candidate["trace"]),
            "--heads", str(heads),
            "--sample", str(identity["sample"]),
            "--stage", str(identity["stage"]),
            "--block", str(identity["block"]),
            "--window", str(identity["window"]),
            "--output-dir", str(template_dir),
        ],
        output_dir / "template_generator_stdout.json",
    )
    run_to_file(
        [
            sys.executable, str(phase_verifier),
            "--archive", str(template_dir / "phase_template_patch.npz"),
            "--manifest", str(template_dir / "manifest.json"),
            "--candidate-trace", str(candidate["trace"]),
            "--baseline-trace", str(baseline["trace"]),
            "--candidate-actual", str(candidate["actual"]),
            "--baseline-actual", str(baseline["actual"]),
            "--expected", str(expected),
            "--inputs", str(vector_dir / "vectors/combined_head_inputs.txt"),
            "--weights", str(vector_dir / "vectors/projection_weights.txt"),
            "--identity-manifest", str(table_dir / "manifest.json"),
            "--identity-receipt", str(table_dir / "verification_receipt.json"),
            "--verilator-log", str(candidate["log"]),
            "--output", str(output_dir / "phase_verification.json"),
        ],
        output_dir / "phase_verification_stdout.json",
    )

    run_release_verify(
        baseline_release, baseline_build, output_dir / "baseline_release_postverify.json"
    )
    run_release_verify(
        candidate_release, candidate_build, output_dir / "candidate_release_postverify.json"
    )
    baseline_report = read_json(output_dir / "baseline_trace_verification.json")
    candidate_report = read_json(output_dir / "candidate_trace_verification.json")
    phase_report = read_json(output_dir / "phase_verification.json")
    expected_scalars = heads * 450 * 32
    if any(
        report.get("acc32", {}).get("mismatch") != 0
        or report.get("acc32", {}).get("scalars") != expected_scalars
        for report in (baseline_report, candidate_report)
    ):
        raise ValueError("baseline/candidate Acc32 verification differs")
    if (
        phase_report.get("status") != "PASS_PHASE_TEMPLATE_TILE_PATCH_NOT_G0"
        or phase_report.get("identity") != identity
        or phase_report.get("acc32", {}).get("mismatch") != 0
    ):
        raise ValueError("phase template verification differs")

    external_paths = {
        "runner": Path(__file__).resolve(),
        "trace_verifier": trace_verifier,
        "template_generator": generator,
        "template_verifier": phase_verifier,
        "table_verifier": table_verifier,
        "table_manifest": table_dir / "manifest.json",
        "table_receipt": table_dir / "verification_receipt.json",
        "task_plan": task_plan,
        "vector_manifest": vector_dir / "vectors/manifest.json",
        "combined_inputs": vector_dir / "vectors/combined_head_inputs.txt",
        "projection_weights": vector_dir / "vectors/projection_weights.txt",
        "software_expected": expected,
        "baseline_release_manifest": baseline_build["manifest"],
        "baseline_executable": baseline_build["executable"],
        "baseline_compile_argv": baseline_build["compile_argv"],
        "candidate_release_manifest": candidate_build["manifest"],
        "candidate_executable": candidate_build["executable"],
        "candidate_compile_argv": candidate_build["compile_argv"],
    }
    internal_paths = {
        path.relative_to(output_dir).as_posix(): path
        for path in output_dir.rglob("*") if path.is_file()
    }
    complete = {
        "schema": "local5_identity_phase_canary_complete_v2",
        "status": "PASS_SEALED_PARAMETERIZED_IDENTITY_PHASE_CANARY_NOT_G0",
        "evidence": "[rtl]+[软件整数金参考]+[rtl-build-provenance]",
        "formal_g0": "DENY",
        "identity": identity,
        "candidate_weight_hold_cycles": args.candidate_hold_cycles,
        "verified_metrics": {
            "baseline_trace_rows": baseline_report["trace_rows"],
            "candidate_trace_rows": candidate_report["trace_rows"],
            "acc32_scalars": expected_scalars,
            "acc32_mismatch": 0,
            "candidate_weight_held_valid_pairs": candidate_report[
                "payload_stability"
            ]["weight_held_valid_pairs"],
            "candidate_weight_valid1_ready0_cycles": candidate_report[
                "payload_stability"
            ]["weight_valid1_ready0_cycles"],
            "phase_expanded_rows": phase_report["expansion"]["rows"],
            "phase_archive_file_size_reduction": phase_report["archive"][
                "file_size_reduction"
            ],
            "phase_base_event_reuse_factor": phase_report["archive"][
                "base_event_reuse_factor"
            ],
        },
        "external_bindings": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in external_paths.items()
        },
        "internal_bindings": {
            name: sha256(path) for name, path in sorted(internal_paths.items())
        },
        "boundary": [
            "single parameterized Local5 window; not formal G0 or full encoder",
            "validation cycles and archive ratios are not architecture performance or ASIC PPA",
            "held-valid coverage is at the service-side producer/hold-adapter boundary",
        ],
    }
    write_json(output_dir / "complete.json", complete)
    chmod_read_only(output_dir)
    print(json.dumps({
        "status": complete["status"], "identity": identity,
        "verified_metrics": complete["verified_metrics"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
