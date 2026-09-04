#!/usr/bin/env python3
"""运行并密封一个 H24 Local5 identity/phase source-only RTL canary。"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import shutil
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


def input_inventory(table_dir: Path, vector_dir: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for prefix, root in (
        ("table", table_dir),
        ("vector/vectors", vector_dir / "vectors"),
        ("vector/software_expected", vector_dir / "software_expected"),
    ):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                inventory[f"{prefix}/{path.relative_to(root).as_posix()}"] = sha256(path)
    return inventory


def validate_window_inputs(
    table_dir: Path, vector_dir: Path, identity: dict[str, int]
) -> None:
    table_manifest = read_json(table_dir / "manifest.json")
    vector_manifest = read_json(vector_dir / "vectors/manifest.json")
    expected_receipt = read_json(
        vector_dir / "software_expected/software_expected_receipt.json"
    )
    task_plan = vector_dir / "software_expected/task_plan.json"
    expected = vector_dir / "software_expected/software_expected.npz"
    vector_identity = {
        name: vector_manifest.get("identity", {}).get(name) for name in identity
    }
    if (
        table_manifest.get("identity") != identity
        or table_manifest.get("formal_g0") != "DENY"
        or vector_manifest.get("status") != "PASS_CANARY_INPUTS_NOT_G0"
        or vector_manifest.get("formal_g0") != "DENY"
        or vector_identity != identity
        or vector_manifest.get("identity", {}).get("tokens") != 450
        or vector_manifest.get("identity", {}).get("out_dim") != 32
        or expected_receipt.get("status") != "PASS_NUMERIC_WINDOW_EXPECTED_NOT_G0"
        or expected_receipt.get("formal_g0") != "DENY"
        or expected_receipt.get("identity") != identity
        or expected_receipt.get("task_plan_sha256") != sha256(task_plan)
        or expected_receipt.get("software_expected_sha256") != sha256(expected)
        or table_manifest.get("task_plan_sha256") != sha256(task_plan)
    ):
        raise ValueError("table/vector/software expected identity or status differs")
    files = vector_manifest.get("files")
    if not isinstance(files, dict) or len(files) != identity["heads"] + 2:
        raise ValueError("vector manifest file set differs")
    for entry in files.values():
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("file"), str)
            or sha256(vector_dir / "vectors" / entry["file"]) != entry.get("sha256")
        ):
            raise ValueError("vector manifest file SHA differs")


def load_structure_contract(path: Path, identity: dict[str, int]) -> dict[str, Any]:
    contract = read_json(path)
    if (
        contract.get("schema") != "local5_h24_phase_structure_contract_v1"
        or contract.get("status")
        != "FROZEN_H24_EVENT_COUNTS_FROM_H3_H6_H12_NOT_G0"
        or contract.get("formal_g0") != "DENY"
        or contract.get("identity") != identity
        or contract.get("expected", {}).get("candidate_hold2", {}).get("trace_rows")
        != 47_941_735
    ):
        raise ValueError("H24 phase structure contract differs")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-release", type=Path, required=True)
    parser.add_argument("--candidate-release", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--structure-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-hold-cycles", type=int, default=2)
    args = parser.parse_args()
    if args.candidate_hold_cycles != 2:
        raise ValueError("H24 v3 runner freezes candidate hold cycles at 2")
    baseline_release = args.baseline_release.resolve()
    candidate_release = args.candidate_release.resolve()
    external_table_dir = args.table_dir.resolve()
    external_vector_dir = args.vector_dir.resolve()
    external_structure_contract = args.structure_contract.resolve()
    final_output = args.output_dir.resolve()
    if final_output.exists():
        raise FileExistsError(f"output directory exists: {final_output}")
    output_dir = final_output.with_name(f"{final_output.name}.staging.{os.getpid()}")
    if output_dir.exists():
        raise FileExistsError(f"staging directory exists: {output_dir}")
    output_dir.mkdir(parents=True)

    def preserve_failed_staging() -> None:
        if not output_dir.exists() or final_output.exists():
            return
        failed = final_output.with_name(f"{final_output.name}.failed.{os.getpid()}")
        if failed.exists():
            return
        write_json(output_dir / "failure_receipt.json", {
            "schema": "local5_h24_identity_phase_failure_v1",
            "status": "FAILED_PRESERVED_NOT_G0",
            "formal_g0": "DENY",
            "boundary": "失败归档不是 PASS 证据；可用于人工恢复昂贵 trace",
        })
        os.replace(output_dir, failed)

    atexit.register(preserve_failed_staging)
    external_input_pre = input_inventory(external_table_dir, external_vector_dir)
    external_contract_pre = sha256(external_structure_contract)
    snapshot_root = output_dir / "input_snapshot"
    table_dir = snapshot_root / "table"
    vector_dir = snapshot_root / "vector_window"
    shutil.copytree(external_table_dir, table_dir, copy_function=shutil.copy2)
    vector_dir.mkdir(parents=True)
    shutil.copytree(
        external_vector_dir / "vectors", vector_dir / "vectors",
        copy_function=shutil.copy2,
    )
    shutil.copytree(
        external_vector_dir / "software_expected",
        vector_dir / "software_expected",
        copy_function=shutil.copy2,
    )
    structure_contract_path = snapshot_root / "phase_structure_contract.json"
    shutil.copy2(external_structure_contract, structure_contract_path)
    if input_inventory(table_dir, vector_dir) != external_input_pre:
        raise ValueError("input snapshot differs from external source")
    if sha256(structure_contract_path) != external_contract_pre:
        raise ValueError("structure contract snapshot differs")
    chmod_read_only(table_dir)
    chmod_read_only(vector_dir)

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
    if heads != 24:
        raise ValueError(f"H24 runner requires heads=24, got {heads}")
    validate_window_inputs(table_dir, vector_dir, identity)
    structure_contract = load_structure_contract(structure_contract_path, identity)

    source_root = output_dir / "source_snapshot"
    source_dir = source_root / "scripts"
    source_dir.mkdir(parents=True)
    live_sources = {
        "runner": Path(__file__).resolve(),
        "trace_verifier": (
            ROOT / "scripts/verify_local5_identity_service_rtl_trace_v2.py"
        ).resolve(),
        "phase_store_runner": (
            ROOT / "scripts/run_local5_phase_array_store_canary_v3.py"
        ).resolve(),
        "phase_store_generator": (
            ROOT / "scripts/generate_local5_phase_array_store_v2.py"
        ).resolve(),
        "phase_store_verifier": (
            ROOT / "scripts/verify_local5_phase_array_store_v2.py"
        ).resolve(),
        "phase_store_tamper": (
            ROOT / "scripts/run_local5_phase_array_store_tamper_v2.py"
        ).resolve(),
    }
    for source in live_sources.values():
        shutil.copy2(source, source_dir / source.name)
    chmod_read_only(source_root)

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

    trace_verifier = source_dir / "verify_local5_identity_service_rtl_trace_v2.py"
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

    phase_store_runner = source_dir / "run_local5_phase_array_store_canary_v3.py"
    phase_store_dir = output_dir / "phase_array_store"
    run_to_file(
        [
            sys.executable, str(phase_store_runner),
            "--trace", str(candidate["trace"]),
            "--source-trace-only",
            "--sample", str(identity["sample"]),
            "--stage", str(identity["stage"]),
            "--block", str(identity["block"]),
            "--window", str(identity["window"]),
            "--heads", str(heads),
            "--expected-identity-manifest", str(table_dir / "manifest.json"),
            "--expected-identity-receipt",
            str(table_dir / "verification_receipt.json"),
            "--output-dir", str(phase_store_dir),
        ],
        output_dir / "phase_array_store_stdout.json",
    )

    run_release_verify(
        baseline_release, baseline_build, output_dir / "baseline_release_postverify.json"
    )
    run_release_verify(
        candidate_release, candidate_build, output_dir / "candidate_release_postverify.json"
    )
    run_to_file(
        [sys.executable, str(table_verifier), "--package-dir", str(table_dir), "verify"],
        output_dir / "table_postverify.json",
        cwd=Path("/tmp"),
    )
    validate_window_inputs(table_dir, vector_dir, identity)
    if (
        input_inventory(external_table_dir, external_vector_dir) != external_input_pre
        or input_inventory(table_dir, vector_dir) != external_input_pre
        or sha256(external_structure_contract) != external_contract_pre
        or sha256(structure_contract_path) != external_contract_pre
    ):
        raise ValueError("external or snapshotted input changed during H24 run")
    baseline_report = read_json(output_dir / "baseline_trace_verification.json")
    candidate_report = read_json(output_dir / "candidate_trace_verification.json")
    phase_report = read_json(phase_store_dir / "complete.json")
    expected_scalars = heads * 450 * 32
    expected_baseline = structure_contract["expected"]["baseline_hold0"]
    expected_candidate = structure_contract["expected"]["candidate_hold2"]
    if any(
        report.get("acc32", {}).get("mismatch") != 0
        or report.get("acc32", {}).get("scalars") != expected_scalars
        for report in (baseline_report, candidate_report)
    ):
        raise ValueError("baseline/candidate Acc32 verification differs")
    if (
        baseline_report.get("event_counts") != expected_baseline["event_counts"]
        or baseline_report.get("trace_rows") != expected_baseline["trace_rows"]
        or candidate_report.get("event_counts") != expected_candidate["event_counts"]
        or candidate_report.get("trace_rows") != expected_candidate["trace_rows"]
        or candidate_report.get("payload_stability", {}).get(
            "weight_held_valid_pairs"
        ) != heads * heads * 32 * 32
        or candidate_report.get("payload_stability", {}).get(
            "weight_valid1_ready0_cycles"
        ) != heads * heads * 32 * 32 * args.candidate_hold_cycles
    ):
        raise ValueError("H24 frozen event-count or held-valid contract differs")
    if (
        phase_report.get("schema") != "local5_phase_array_store_canary_complete_v3"
        or phase_report.get("status") != "PASS_SEALED_STREAMING_MMAP_CANARY_NOT_G0"
        or phase_report.get("formal_g0") != "DENY"
        or phase_report.get("identity") != identity
        or phase_report.get("verified_metrics", {}).get("expanded_rows")
        != candidate_report.get("trace_rows")
        or phase_report.get("verified_metrics", {}).get("store_arrays") != 27
        or phase_report.get("verified_metrics", {}).get("negative_cases_passed") != 10
        or phase_report.get("verified_metrics", {}).get("source_trace_only_pass") is not True
        or phase_report.get("verified_metrics", {}).get("arrays_compared") != 0
        or phase_report.get("verified_metrics", {}).get("array_mismatch") is not None
    ):
        raise ValueError("H24 phase array store verification differs")

    external_paths = {
        "table_manifest": external_table_dir / "manifest.json",
        "table_receipt": external_table_dir / "verification_receipt.json",
        "task_plan": external_vector_dir / "software_expected/task_plan.json",
        "vector_manifest": external_vector_dir / "vectors/manifest.json",
        "combined_inputs": external_vector_dir / "vectors/combined_head_inputs.txt",
        "projection_weights": external_vector_dir / "vectors/projection_weights.txt",
        "software_expected": (
            external_vector_dir / "software_expected/software_expected.npz"
        ),
        "structure_contract": external_structure_contract,
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
        "schema": "local5_h24_identity_phase_canary_complete_v3",
        "status": "PASS_SEALED_H24_IDENTITY_PHASE_ARRAY_CANARY_NOT_G0",
        "evidence": (
            "[rtl]+[软件整数金参考]+[rtl-build-provenance]+"
            "[独立软件逐行展开验证]+[资源实测]"
        ),
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
            "phase_expanded_rows": phase_report["verified_metrics"]["expanded_rows"],
            "phase_store_arrays": phase_report["verified_metrics"]["store_arrays"],
            "phase_negative_cases_passed": phase_report[
                "verified_metrics"
            ]["negative_cases_passed"],
            "phase_generator_max_rss_kb": phase_report[
                "verified_metrics"
            ]["generator_max_rss_kb"],
            "phase_verifier_max_rss_kb": phase_report[
                "verified_metrics"
            ]["verifier_max_rss_kb"],
            "frozen_candidate_trace_rows": expected_candidate["trace_rows"],
        },
        "source_bindings": {
            name: {
                "path": (source_dir / source.name).relative_to(output_dir).as_posix(),
                "sha256": sha256(source_dir / source.name),
            }
            for name, source in live_sources.items()
        },
        "input_provenance": {
            "frozen_inventory": external_input_pre,
            "structure_contract_sha256": external_contract_pre,
            "pre_post_and_snapshot_match": True,
        },
        "external_bindings": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in external_paths.items()
        },
        "internal_bindings": {
            name: sha256(path) for name, path in sorted(internal_paths.items())
        },
        "boundary": [
            "单个真实 H24 Local5 窗口；不是 formal G0 或 full encoder",
            "validation cycles、trace/store 大小与脚本 RSS 不是架构性能或 ASIC PPA",
            "held-valid 覆盖位于服务侧 producer/hold-adapter 边界",
            "Phase Array Store 是验证基础设施，不是 DATE 架构贡献",
            "H24 event counts 在运行前由 H3/H6/H12 RTL 校准合同冻结；未声称 formal proof",
            "v8/v10 release 不作为单变量性能对照，validation cycles 不进入加速比",
        ],
    }
    write_json(output_dir / "complete.json", complete)
    os.replace(output_dir, final_output)
    chmod_read_only(final_output)
    print(json.dumps({
        "status": complete["status"], "identity": identity,
        "verified_metrics": complete["verified_metrics"],
        "output_dir": str(final_output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
