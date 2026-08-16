#!/usr/bin/env python3
"""运行并封存 Local5 H24 compact phase-summary v2 单窗口验证包。"""

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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local5_cross_protocol_oracle_fast_v1 as fast_cross
import verify_local5_phase_summary_contract_v2 as contract


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {"sample": 2, "stage": 3, "block": 0, "window": 1, "heads": 24}
SERVICE_SEED = 20260810
MAIN_INSTANCE = "TOP.tb_qfit_local5_memo_multitile_cross_head"
CROSS_INSTANCE = (
    MAIN_INSTANCE
    + ".u_executor.g_scalar_cross_head_acc.u_cross_head_accumulator"
)
TCFM_INSTANCE = (
    MAIN_INSTANCE
    + ".u_executor.g_baseline_head_engine.u_head_engine.u_tile"
    + ".g_tcfm5_backend.u_projection"
)
REFERENCE_SCHEMA = "local5_h24_identity_phase_canary_complete_v3"
REFERENCE_STATUS = "PASS_SEALED_H24_IDENTITY_PHASE_ARRAY_CANARY_NOT_G0"
RELEASE_STATUS = "SEALED_RTL_RELEASE_NOT_G0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层必须是对象")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_to_file(
    argv: list[str],
    output: Path,
    *,
    cwd: Path | None = None,
) -> None:
    with output.open("wb") as handle:
        subprocess.run(
            argv,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def chmod_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    mode = root.stat().st_mode
    root.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def snapshot_files(paths: dict[str, Path], destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True)
    report: dict[str, Any] = {}
    for name, source in paths.items():
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"待封存源码缺失或为符号链接: {source}")
        target = destination / source.name
        shutil.copy2(source, target)
        report[name] = {
            "live_path": str(source),
            "snapshot_path": target.relative_to(destination.parent).as_posix(),
            "sha256": sha256_file(target),
        }
        if sha256_file(source) != report[name]["sha256"]:
            raise ValueError(f"源码封存前后不一致: {source}")
    chmod_read_only(destination)
    return report


def verify_snapshot_unchanged(report: dict[str, Any], package_root: Path) -> None:
    for name, row in report.items():
        expected = row["sha256"]
        if (
            sha256_file(Path(row["live_path"])) != expected
            or sha256_file(package_root / row["snapshot_path"]) != expected
        ):
            raise ValueError(f"运行期间源码发生变化: {name}")


def verify_input_identity(table_dir: Path, vector_dir: Path) -> dict[str, Any]:
    table = read_json(table_dir / "manifest.json")
    table_receipt = read_json(table_dir / "verification_receipt.json")
    vectors = read_json(vector_dir / "vectors/manifest.json")
    expected_receipt = read_json(
        vector_dir / "software_expected/software_expected_receipt.json"
    )
    vector_identity = {
        key: vectors.get("identity", {}).get(key) for key in IDENTITY
    }
    if (
        table.get("identity") != IDENTITY
        or table.get("formal_g0") != "DENY"
        or table_receipt.get("formal_g0") != "DENY"
        or vectors.get("status") != "PASS_CANARY_INPUTS_NOT_G0"
        or vectors.get("formal_g0") != "DENY"
        or vector_identity != IDENTITY
        or vectors.get("identity", {}).get("tokens") != 450
        or vectors.get("identity", {}).get("out_dim") != 32
        or expected_receipt.get("status")
        != "PASS_NUMERIC_WINDOW_EXPECTED_NOT_G0"
        or expected_receipt.get("formal_g0") != "DENY"
        or expected_receipt.get("identity") != IDENTITY
    ):
        raise ValueError("H24 table/vector/software-expected 身份或状态不匹配")
    required = [
        table_dir / "relation_delay.memh",
        table_dir / "weight_delay.memh",
        table_dir / "final_delay.memh",
        vector_dir / "vectors/combined_head_inputs.txt",
        vector_dir / "vectors/projection_weights.txt",
        vector_dir / "software_expected/software_expected.npz",
    ]
    if any(not path.is_file() or path.is_symlink() for path in required):
        raise ValueError("H24 输入文件缺失或包含符号链接")
    return {
        "table_manifest": sha256_file(table_dir / "manifest.json"),
        "table_receipt": sha256_file(table_dir / "verification_receipt.json"),
        "vector_manifest": sha256_file(vector_dir / "vectors/manifest.json"),
        "software_expected_receipt": sha256_file(
            vector_dir / "software_expected/software_expected_receipt.json"
        ),
        "combined_inputs": sha256_file(required[3]),
        "projection_weights": sha256_file(required[4]),
        "software_expected": sha256_file(required[5]),
    }


def verify_reference_package(reference: Path) -> dict[str, Any]:
    complete_path = reference / "complete.json"
    complete = read_json(complete_path)
    if (
        complete.get("schema") != REFERENCE_SCHEMA
        or complete.get("status") != REFERENCE_STATUS
        or complete.get("formal_g0") != "DENY"
        or complete.get("identity") != IDENTITY
        or complete.get("verified_metrics", {}).get("acc32_mismatch") != 0
        or complete.get("verified_metrics", {}).get("acc32_scalars") != 345_600
    ):
        raise ValueError("既有 H24 reference package 未被正向准入")
    paths = {
        "trace": reference / "candidate_trace.csv",
        "actual": reference / "candidate_actual.memh",
        "trace_verification": reference / "candidate_trace_verification.json",
    }
    internal = complete.get("internal_bindings")
    if not isinstance(internal, dict):
        raise ValueError("reference package 缺少 internal bindings")
    for name, path in paths.items():
        key = path.relative_to(reference).as_posix()
        if (
            not path.is_file()
            or path.is_symlink()
            or internal.get(key) != sha256_file(path)
        ):
            raise ValueError(f"reference package 文件绑定失败: {name}")
    trace_report = read_json(paths["trace_verification"])
    if (
        trace_report.get("identity") != IDENTITY
        or trace_report.get("acc32", {}).get("mismatch") != 0
        or trace_report.get("acc32", {}).get("scalars") != 345_600
        or trace_report.get("trace_rows") != 47_941_735
    ):
        raise ValueError("reference trace verification 口径不匹配")
    return {
        "complete": complete_path,
        "complete_sha256": sha256_file(complete_path),
        "trace": paths["trace"],
        "trace_sha256": sha256_file(paths["trace"]),
        "trace_rows": trace_report["trace_rows"],
        "trace_bytes": paths["trace"].stat().st_size,
        "actual": paths["actual"],
        "actual_sha256": sha256_file(paths["actual"]),
        "trace_verification": paths["trace_verification"],
        "trace_verification_sha256": sha256_file(paths["trace_verification"]),
    }


def load_release(release: Path) -> tuple[dict[str, Any], list[str]]:
    manifest_path = release / "release_manifest.json"
    manifest = read_json(manifest_path)
    build = manifest.get("builds", {}).get("24")
    if (
        manifest.get("status") != RELEASE_STATUS
        or not isinstance(build, dict)
        or build.get("service_mode") != "identity_derived"
        or build.get("heads") != 24
    ):
        raise ValueError("v10 release H24 未被正向准入")
    compile_path = release / str(build["compile_argv_path"])
    compile_argv = json.loads(compile_path.read_text(encoding="utf-8"))
    if (
        not isinstance(compile_argv, list)
        or not all(isinstance(item, str) for item in compile_argv)
        or sha256_file(compile_path) != build.get("compile_argv_sha256")
        or "-GHEADS=24" not in compile_argv
        or "-GOUTPUT_TILES=24" not in compile_argv
        or "-GIDENTITY_DERIVED_SERVICE=1" not in compile_argv
        or "-GUSE_MEMO=0" not in compile_argv
        or "-GFORCE_WEIGHT_RESPONSE_HOLD_CYCLES=2" not in compile_argv
    ):
        raise ValueError("v10 H24 compile argv 不满足冻结配置")
    return manifest, compile_argv


def make_compile_argv(
    release: Path,
    baseline: list[str],
    build_dir: Path,
    source_snapshot: Path,
) -> list[str]:
    output = [
        str((release / item).resolve()) if item.startswith("source/") else item
        for item in baseline
    ]
    mdir = output.index("--Mdir") + 1
    output[mdir] = str(build_dir)
    output.extend(
        str(source_snapshot / name)
        for name in (
            "local5_phase_summary_monitor_v2.sv",
            "local5_cross_acc_summary_monitor_v2.sv",
            "local5_tcfm5_summary_monitor_v2.sv",
            "bind_local5_phase_summary_monitors_v2.sv",
        )
    )
    return output


def expect_failure(name: str, action: Callable[[], None]) -> dict[str, str]:
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - tamper gate accepts any fail-closed error
        return {"case": name, "status": "PASS_REJECTED", "reason": str(exc)}
    raise ValueError(f"tamper case 未被拒绝: {name}")


def mutate_csv(source: Path, target: Path, mutate: Callable[[list[list[str]]], None]) -> None:
    rows = [line.split(",") for line in source.read_text(encoding="ascii").splitlines()]
    mutate(rows)
    target.write_text("\n".join(",".join(row) for row in rows) + "\n", encoding="ascii")


def run_tamper_regression(
    output: Path,
    main: contract.OrderedSummary,
    cross: contract.OrderedSummary,
    tcfm: contract.OrderedSummary,
    phase: contract.PhaseLedger,
    state_contract: dict[str, Any],
) -> list[dict[str, str]]:
    tamper = output / "tamper_evidence"
    tamper.mkdir()
    cases: list[dict[str, str]] = []
    expected_identity = contract.PhaseIdentity(3, 0, 1)

    phase_identity = tamper / "phase_identity.csv"
    def mutate_phase_identity(rows: list[list[str]]) -> None:
        row = next(item for item in rows if item[0] == "P")
        row[2] = "2"
    mutate_csv(phase.path, phase_identity, mutate_phase_identity)
    cases.append(expect_failure(
        "identity_tamper",
        lambda: contract.validate_phase_interval_ledger(
            contract.parse_phase_interval_ledger(phase_identity),
            expected_identity,
            contract.head_phase_roles_from_state_contract(state_contract),
        ),
    ))

    digest_path = tamper / "main_digest.csv"
    def mutate_digest(rows: list[list[str]]) -> None:
        row = next(item for item in rows if item[:2] == ["S", "RELATION_REQ_ACCEPT"])
        row[3] = ("0" if row[3][0] != "0" else "1") + row[3][1:]
    mutate_csv(main.path, digest_path, mutate_digest)
    cases.append(expect_failure(
        "digest_tamper",
        lambda: contract.compare_summary_resources(
            main,
            contract.parse_ordered_summary(digest_path),
            ["RELATION_REQ_ACCEPT"],
        ),
    ))

    delete_path = tamper / "main_event_delete.csv"
    def mutate_delete(rows: list[list[str]]) -> None:
        index = next(i for i, item in enumerate(rows) if item[:2] == ["S", "WEIGHT_RSP_ACCEPT"])
        rows.pop(index)
    mutate_csv(main.path, delete_path, mutate_delete)
    cases.append(expect_failure(
        "event_delete",
        lambda: contract.parse_ordered_summary(delete_path),
    ))

    swap_path = tamper / "main_resource_metadata_swap.csv"
    def mutate_swap(rows: list[list[str]]) -> None:
        left = next(i for i, item in enumerate(rows) if item[:2] == ["R", "RELATION_REQ_ACCEPT"])
        rows[left], rows[left + 1] = rows[left + 1], rows[left]
    mutate_csv(main.path, swap_path, mutate_swap)
    cases.append(expect_failure(
        "resource_metadata_swap",
        lambda: contract.parse_ordered_summary(swap_path),
    ))

    relation = main.resources["RELATION_REQ_ACCEPT"]
    if relation.first_anchor is None or relation.last_anchor is None:
        raise ValueError("relation summary 缺少 event-order 负例锚点")
    ordered_events = [
        contract.SummaryEvent(0, 10, tuple(relation.first_anchor[2:])),
        contract.SummaryEvent(1, 11, tuple(relation.last_anchor[2:])),
    ]
    ordered_digest = contract.summarize_events(
        "RELATION_REQ_ACCEPT",
        relation.instance_path,
        ordered_events,
        field_names=relation.field_names,
    )
    swapped_digest = contract.summarize_events(
        "RELATION_REQ_ACCEPT",
        relation.instance_path,
        [
            contract.SummaryEvent(0, 10, ordered_events[1].payload),
            contract.SummaryEvent(1, 11, ordered_events[0].payload),
        ],
        field_names=relation.field_names,
    )
    write_json(tamper / "event_order_swap_fixture.json", {
        "schema": "local5_ordered_digest_swap_fixture_v2",
        "ordered": {
            "digest0": f"{ordered_digest.digest0:016x}",
            "digest1": f"{ordered_digest.digest1:016x}",
        },
        "swapped": {
            "digest0": f"{swapped_digest.digest0:016x}",
            "digest1": f"{swapped_digest.digest1:016x}",
        },
    })

    def reject_event_order_swap() -> None:
        if ordered_digest == swapped_digest:
            return
        raise contract.ContractError("ordered digest rejects an event-order swap")

    cases.append(expect_failure(
        "event_order_swap",
        reject_event_order_swap,
    ))

    rebind_path = tamper / "cross_instance_rebind.csv"
    def mutate_rebind(rows: list[list[str]]) -> None:
        row = next(item for item in rows if item[0] == "TARGET_INSTANCE")
        row[1] += ".rebound"
    mutate_csv(cross.path, rebind_path, mutate_rebind)
    cases.append(expect_failure(
        "instance_rebind",
        lambda: contract.validate_observer_summary_binding(
            contract.parse_ordered_summary(rebind_path),
            expected_schema=contract.CROSS_SUMMARY_SCHEMA,
            expected_target_instance=CROSS_INSTANCE,
        ),
    ))

    mask_path = tamper / "tcfm_bank_mask.csv"
    def mutate_mask(rows: list[list[str]]) -> None:
        row = next(item for item in rows if item[:3] == ["A", "TCFM5_TERM_COMMIT", "FIRST"])
        row[8] = str(int(row[8]) ^ 1)
    mutate_csv(tcfm.path, mask_path, mutate_mask)
    cases.append(expect_failure(
        "bank_mask_tamper",
        lambda: contract.compare_summary_resources(
            main,
            contract.parse_ordered_summary(mask_path),
            ["TCFM5_TERM_COMMIT"],
        ),
    ))
    write_json(tamper / "tamper_regression.json", {
        "schema": "local5_phase_summary_tamper_regression_v2",
        "status": "PASS_ALL_NEGATIVE_CASES_REJECTED",
        "formal_g0": "DENY",
        "cases": cases,
    })
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    release = args.release.resolve()
    table_dir = args.table_dir.resolve()
    vector_dir = args.vector_dir.resolve()
    reference = args.reference_package.resolve()
    final_output = args.output_dir.resolve()
    if final_output.exists():
        raise FileExistsError(f"输出目录已存在: {final_output}")
    output = final_output.with_name(f"{final_output.name}.staging.{os.getpid()}")
    output.mkdir(parents=True)

    def preserve_failure() -> None:
        if not output.exists() or final_output.exists():
            return
        failed = final_output.with_name(f"{final_output.name}.failed.{os.getpid()}")
        if failed.exists():
            return
        write_json(output / "failure_receipt.json", {
            "schema": "local5_h24_phase_summary_failure_v2",
            "status": "FAILED_PRESERVED_NOT_G0",
            "formal_g0": "DENY",
            "说明": "失败目录不是 PASS 证据。",
        })
        os.replace(output, failed)

    atexit.register(preserve_failure)

    input_binding = verify_input_identity(table_dir, vector_dir)
    reference_binding = verify_reference_package(reference)
    release_manifest, baseline_compile = load_release(release)
    release_manifest_path = release / "release_manifest.json"
    release_sha_pre = sha256_file(release_manifest_path)
    input_sha_pre = dict(input_binding)

    source_paths = {
        "runner": Path(__file__).resolve(),
        "verifier": ROOT / "scripts/verify_local5_phase_summary_contract_v2.py",
        "verifier_test": ROOT / "scripts/test_verify_local5_phase_summary_contract_v2.py",
        "fast_cross_oracle_wrapper": ROOT / "scripts/local5_cross_protocol_oracle_fast_v1.py",
        "fast_cross_oracle_c": ROOT / "scripts/local5_cross_protocol_oracle_fast_v1.c",
        "fast_cross_oracle_test": ROOT / "scripts/test_local5_cross_protocol_oracle_fast_v1.py",
        "numeric_verifier": ROOT / "scripts/verify_local5_identity_service_rtl_trace_v2.py",
        "state_contract": ROOT / "contracts/local5_phase_state_roles_v2.json",
        "main_monitor": ROOT / "verif_qfit/local5_phase_summary_monitor_v2.sv",
        "cross_monitor": ROOT / "verif_qfit/local5_cross_acc_summary_monitor_v2.sv",
        "tcfm_monitor": ROOT / "verif_qfit/local5_tcfm5_summary_monitor_v2.sv",
        "bind": ROOT / "verif_qfit/bind_local5_phase_summary_monitors_v2.sv",
        "preflight": ROOT / "sim_qfit/run_local5_phase_summary_preflight_v2.sh",
    }
    source_binding = snapshot_files(source_paths, output / "source")
    build_dir = output / "build"
    run_dir = output / "run"
    build_dir.mkdir()
    run_dir.mkdir()
    fast_cross_binary = build_dir / "local5_cross_protocol_oracle_fast_v1"
    fast_cross_compile = fast_cross.compile_oracle(
        output / "source/local5_cross_protocol_oracle_fast_v1.c",
        fast_cross_binary,
    )
    write_json(output / "fast_cross_oracle_compile.json", fast_cross_compile)
    compile_argv = make_compile_argv(
        release,
        baseline_compile,
        build_dir,
        output / "source",
    )
    write_json(output / "compile_argv.json", compile_argv)

    plan = {
        "schema": "local5_h24_phase_summary_run_plan_v2",
        "status": "FROZEN_BEFORE_RUN_NOT_G0",
        "formal_g0": "DENY",
        "requested_identity": IDENTITY,
        "actual_identity": IDENTITY,
        "identity_status": "MATCH",
        "instances": {
            "main": MAIN_INSTANCE,
            "cross": CROSS_INSTANCE,
            "tcfm5": TCFM_INSTANCE,
        },
        "release_manifest_sha256": release_sha_pre,
        "baseline_compile_argv_sha256": hashlib.sha256(
            json.dumps(baseline_compile, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "telemetry_compile_argv_sha256": sha256_file(output / "compile_argv.json"),
        "input_bindings": input_binding,
        "reference_bindings": {
            key: value for key, value in reference_binding.items()
            if key.endswith("sha256") or key.endswith("bytes")
        },
        "source_bindings": source_binding,
        "evidence_limit_bytes": contract.EVIDENCE_LIMIT_BYTES,
        "full_identity_trace_output": "/dev/null",
    }
    write_json(output / "run_plan.json", plan)
    run_to_file(
        ["bash", str(source_paths["preflight"])],
        output / "static_preflight.log",
        cwd=ROOT,
    )
    static_preflight = contract.run_static_preflight(ROOT)
    if static_preflight.get("status") != "STATIC_PREFLIGHT_PASS_NOT_RUN_ADMISSION":
        raise ValueError("phase-summary v2 静态 preflight 未通过")
    write_json(output / "static_preflight.json", static_preflight)
    pre_admission_receipt = {
        "schema": "local5_h24_phase_summary_admission_v2",
        "status": "ADMIT_RUN_NOT_G0",
        "requested_identity": IDENTITY,
        "actual_identity": IDENTITY,
        "identity_status": "MATCH",
        "receipts": {
            "release": {"status": release_manifest["status"]},
            "reference": {"status": REFERENCE_STATUS},
            "static_preflight": {
                "status": "STATIC_PREFLIGHT_PASS_NOT_RUN_ADMISSION"
            },
        },
    }
    pre_admission = contract.verify_receipt_admission(
        pre_admission_receipt,
        allowed_schemas=["local5_h24_phase_summary_admission_v2"],
        allowed_statuses=["ADMIT_RUN_NOT_G0"],
        required_receipts={
            "release": [RELEASE_STATUS],
            "reference": [REFERENCE_STATUS],
            "static_preflight": ["STATIC_PREFLIGHT_PASS_NOT_RUN_ADMISSION"],
        },
        expected_identity=IDENTITY,
        package_digest=sha256_file(output / "run_plan.json"),
        denylist={"entries": []},
    )
    write_json(output / "pre_run_admission.json", pre_admission)
    run_to_file(
        [
            sys.executable,
            str(release / "source/scripts/local5_erep_numeric_release.py"),
            "verify",
            "--release-dir",
            str(release),
        ],
        output / "release_preverify.json",
        cwd=Path("/tmp"),
    )
    run_to_file(
        [
            "/usr/bin/time", "-f", "wall_seconds=%e\nmax_rss_kb=%M",
            "-o", str(output / "compile_time.txt"),
            *compile_argv,
        ],
        output / "compile.log",
        cwd=output,
    )

    manifest_sha = sha256_file(table_dir / "manifest.json")
    receipt_sha = sha256_file(table_dir / "verification_receipt.json")
    executable = build_dir / "Vtb_qfit_local5_memo_multitile_cross_head"
    runtime_argv = [
        str(executable),
        f"+INPUTS={vector_dir / 'vectors/combined_head_inputs.txt'}",
        f"+WEIGHTS={vector_dir / 'vectors/projection_weights.txt'}",
        "+STAGE_ID=3", "+BLOCK_ID=0", "+WINDOW_ID=1", "+NO_ACC_CHECK",
        f"+SERVICE_SEED={SERVICE_SEED}",
        f"+RELATION_DELAY_MEMH={table_dir / 'relation_delay.memh'}",
        f"+WEIGHT_DELAY_MEMH={table_dir / 'weight_delay.memh'}",
        f"+FINAL_DELAY_MEMH={table_dir / 'final_delay.memh'}",
        f"+IDENTITY_MANIFEST_SHA={manifest_sha}",
        f"+IDENTITY_RECEIPT_SHA={receipt_sha}",
        "+IDENTITY_TRACE=/dev/null",
        f"+ACTUAL_ACC_FILE={run_dir / 'actual.memh'}",
        f"+PHASE_INTERVALS_V2={run_dir / 'phase_intervals.csv'}",
        f"+MAIN_SUMMARY_V2={run_dir / 'main_summary.csv'}",
        "+TELEMETRY_STAGE=3", "+TELEMETRY_BLOCK=0", "+TELEMETRY_WINDOW=1",
        f"+MAIN_RESOURCE_INSTANCE={MAIN_INSTANCE}",
        f"+CROSS_ACC_TARGET_INSTANCE={CROSS_INSTANCE}",
        f"+TCFM5_TARGET_INSTANCE={TCFM_INSTANCE}",
        f"+CROSS_SUMMARY_PREFIX_V2={run_dir / 'cross_summary'}",
        f"+TCFM5_SUMMARY_PREFIX_V2={run_dir / 'tcfm_summary'}",
    ]
    write_json(output / "run_argv.json", runtime_argv)
    run_to_file(
        [
            "/usr/bin/time", "-f", "wall_seconds=%e\nmax_rss_kb=%M",
            "-o", str(output / "run_time.txt"),
            *runtime_argv,
        ],
        output / "verilator.log",
    )

    numeric_verifier = output / "source/verify_local5_identity_service_rtl_trace_v2.py"
    run_to_file(
        [
            sys.executable,
            str(numeric_verifier),
            "--trace", str(reference_binding["trace"]),
            "--package-dir", str(table_dir),
            "--expected-weight-hold-cycles", "2",
            "--actual", str(run_dir / "actual.memh"),
            "--expected", str(vector_dir / "software_expected/software_expected.npz"),
            "--verilator-log", str(output / "verilator.log"),
            "--output", str(output / "numeric_verification.json"),
        ],
        output / "numeric_verification_stdout.json",
    )
    numeric = read_json(output / "numeric_verification.json")
    if (
        numeric.get("identity") != IDENTITY
        or numeric.get("acc32", {}).get("mismatch") != 0
        or numeric.get("acc32", {}).get("scalars") != 345_600
        or sha256_file(run_dir / "actual.memh")
        != reference_binding["actual_sha256"]
    ):
        raise ValueError("新 H24 Acc32 未与软件整数金参考和既有 canary 同时一致")

    main_summary = contract.parse_ordered_summary(run_dir / "main_summary.csv")
    cross_summary = contract.parse_single_observer_summary_glob(
        run_dir / "cross_summary.*.csv",
        expected_schema=contract.CROSS_SUMMARY_SCHEMA,
        expected_target_instance=CROSS_INSTANCE,
    )
    tcfm_summary = contract.parse_single_observer_summary_glob(
        run_dir / "tcfm_summary.*.csv",
        expected_schema=contract.TCFM5_SUMMARY_SCHEMA,
        expected_target_instance=TCFM_INSTANCE,
    )
    phase_ledger = contract.parse_phase_interval_ledger(
        run_dir / "phase_intervals.csv"
    )
    state_contract, _, state_report = contract.load_and_verify_state_role_contract(ROOT)
    phase_identity = contract.PhaseIdentity(3, 0, 1)
    phase_audit = contract.stream_compare_phase_ledger_to_identity_trace(
        phase_ledger,
        reference_binding["trace"],
        state_contract,
        phase_identity,
    )
    aligned_audit = contract.verify_main_summary_against_identity_trace(
        main_summary,
        reference_binding["trace"],
        heads=24,
    )
    cross_oracle, fast_cross_run = fast_cross.verify_cross_summary_pair_fast(
        main_summary,
        cross_summary,
        binary=fast_cross_binary,
        heads=24,
    )
    write_json(output / "fast_cross_oracle_run.json", fast_cross_run)
    tcfm_ledger = contract.verify_tcfm5_summary_pair(
        main_summary,
        tcfm_summary,
    )
    counts = contract.workload_counts(24)
    if (
        len(phase_ledger.intervals) != counts.phase
        or aligned_audit.rows_read != reference_binding["trace_rows"]
        or cross_oracle.count != counts.cross_total
        or tcfm_ledger.mismatch_count != 0
    ):
        raise ValueError("H24 closed-form 或流式 reference 计数不匹配")

    tamper_cases = run_tamper_regression(
        output,
        main_summary,
        cross_summary,
        tcfm_summary,
        phase_ledger,
        state_contract,
    )

    run_to_file(
        [
            sys.executable,
            str(release / "source/scripts/local5_erep_numeric_release.py"),
            "verify",
            "--release-dir",
            str(release),
        ],
        output / "release_postverify.json",
        cwd=Path("/tmp"),
    )
    verify_snapshot_unchanged(source_binding, output)
    if (
        sha256_file(release_manifest_path) != release_sha_pre
        or verify_input_identity(table_dir, vector_dir) != input_sha_pre
        or sha256_file(reference_binding["trace"])
        != reference_binding["trace_sha256"]
        or sha256_file(reference_binding["complete"])
        != reference_binding["complete_sha256"]
    ):
        raise ValueError("运行期间 release/input/reference 发生变化")

    verification = {
        "schema": "local5_h24_phase_summary_verification_v2",
        "status": "PASS_H24_COMPACT_TELEMETRY_NOT_G0",
        "formal_g0": "DENY",
        "identity": IDENTITY,
        "closed_form": asdict(counts),
        "phase": asdict(phase_audit),
        "aligned_trace": {
            "path": aligned_audit.path,
            "rows_read": aligned_audit.rows_read,
            "resources": {
                name: {
                    "count": resource.count,
                    "digest0": f"{resource.digest0:016x}",
                    "digest1": f"{resource.digest1:016x}",
                }
                for name, resource in aligned_audit.resources.items()
            },
        },
        "cross_protocol": {
            "count": cross_oracle.count,
            "read_count": cross_oracle.read_count,
            "write_count": cross_oracle.write_count,
            "digest0": f"{cross_oracle.digest0:016x}",
            "digest1": f"{cross_oracle.digest1:016x}",
            "oracle_backend": "FROZEN_C_ORACLE_MITERED_TO_PYTHON",
            "oracle_wall_seconds": fast_cross_run["wall_seconds"],
        },
        "tcfm5": asdict(tcfm_ledger),
        "acc32": numeric["acc32"],
        "state_contract": state_report,
        "observer_glob_cardinality": {"cross": 1, "tcfm5": 1},
        "tamper_cases": tamper_cases,
        "boundary": [
            "单个 H24 Local5 窗口，不是 formal G0 或 full encoder。",
            "验证 cycle 和摘要耗时不是架构性能或 ASIC PPA。",
            "rolling64 是有序错误检测，不是密码学承诺；文件另用 SHA256 封存。",
            "phase telemetry 与 observer 属于验证基础设施，不是 DATE 架构贡献。",
        ],
    }
    write_json(output / "verification.json", verification)
    (output / "verification.md").write_text(
        "# Local5 H24 紧凑遥测验证结果\n\n"
        "> 状态：PASS（非 formal G0）  \n"
        "> 证据等级：`[rtl]+[软件整数金参考]+[流式独立oracle]`\n\n"
        f"- phase：`{counts.phase}` 条，与既有 H24 trace 流式逐项一致。\n"
        f"- aligned accepted：`{counts.aligned_total}` 条，五类摘要全部一致。\n"
        f"- cross-Acc：`{cross_oracle.count}` 条，读写各 "
        f"`{cross_oracle.read_count}`，闭式地址相序摘要一致。\n"
        f"- TCFM5：`{tcfm_ledger.term_count}` 个 term，"
        f"`{tcfm_ledger.update_count}` 次 bank update，mask mismatch=0。\n"
        "- Acc32：345,600 个标量，软件整数金参考 mismatch=0。\n"
        f"- 负例：`{len(tamper_cases)}` 类全部被拒绝。\n\n"
        "## 边界\n\n"
        "这是单个 H24 窗口的验证可扩展性证据，不是 formal G0、全 encoder "
        "性能或 ASIC PPA；验证基础设施不计入 DATE 架构创新。\n",
        encoding="utf-8",
    )

    internal_files = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in output.rglob("*")
        if path.is_file()
        and "build" not in path.relative_to(output).parts[:1]
        and "source" not in path.relative_to(output).parts[:1]
    }
    write_json(output / "evidence_manifest.json", {
        "schema": "local5_h24_phase_summary_evidence_manifest_v2",
        "identity": IDENTITY,
        "files": internal_files,
    })
    final_admission_receipt = dict(pre_admission_receipt)
    final_admission_receipt["status"] = "ADMIT_EVIDENCE_NOT_G0"
    final_admission_receipt["receipts"] = {
        **pre_admission_receipt["receipts"],
        "verification": {"status": verification["status"]},
    }
    final_admission = contract.verify_receipt_admission(
        final_admission_receipt,
        allowed_schemas=["local5_h24_phase_summary_admission_v2"],
        allowed_statuses=["ADMIT_EVIDENCE_NOT_G0"],
        required_receipts={
            "release": [RELEASE_STATUS],
            "reference": [REFERENCE_STATUS],
            "static_preflight": ["STATIC_PREFLIGHT_PASS_NOT_RUN_ADMISSION"],
            "verification": ["PASS_H24_COMPACT_TELEMETRY_NOT_G0"],
        },
        expected_identity=IDENTITY,
        package_digest=sha256_file(output / "evidence_manifest.json"),
        denylist={"entries": []},
    )
    write_json(output / "admission_receipt.json", final_admission)
    payload = contract.audit_evidence_payload(output)
    write_json(output / "payload_audit.json", asdict(payload))

    def make_complete(payload_bytes: int) -> dict[str, Any]:
        return {
            "schema": "local5_h24_phase_summary_complete_v2",
            "status": "PASS_SEALED_H24_PHASE_SUMMARY_PILOT_NOT_G0",
            "formal_g0": "DENY",
            "identity": IDENTITY,
            "evidence": "[rtl]+[软件整数金参考]+[流式独立oracle]",
            "verified_metrics": {
                "phase_intervals": counts.phase,
                "aligned_events": counts.aligned_total,
                "cross_commands": counts.cross_total,
                "cross_reads": counts.cross_read,
                "cross_writes": counts.cross_write,
                "tcfm5_terms": tcfm_ledger.term_count,
                "tcfm5_updates": tcfm_ledger.update_count,
                "tcfm5_mask_mismatch": tcfm_ledger.mismatch_count,
                "acc32_scalars": 345_600,
                "acc32_mismatch": 0,
                "negative_cases_passed": len(tamper_cases),
                "reference_trace_rows": 47_941_735,
                "reference_trace_bytes": reference_binding["trace_bytes"],
                "new_full_trace_bytes": 0,
                "evidence_payload_bytes": payload_bytes,
            },
            "external_bindings": {
                "release_manifest": {
                    "path": str(release_manifest_path),
                    "sha256": release_sha_pre,
                },
                "reference_complete": {
                    "path": str(reference_binding["complete"]),
                    "sha256": reference_binding["complete_sha256"],
                },
                "reference_trace": {
                    "path": str(reference_binding["trace"]),
                    "sha256": reference_binding["trace_sha256"],
                },
                "table_manifest": {
                    "path": str(table_dir / "manifest.json"),
                    "sha256": input_binding["table_manifest"],
                },
                "vector_manifest": {
                    "path": str(vector_dir / "vectors/manifest.json"),
                    "sha256": input_binding["vector_manifest"],
                },
            },
            "source_bindings": source_binding,
            "internal_bindings": {
                path.relative_to(output).as_posix(): sha256_file(path)
                for path in output.rglob("*")
                if path.is_file() and path.name != "complete.json"
            },
            "boundary": verification["boundary"],
        }

    complete = make_complete(payload.bytes)
    write_json(output / "complete.json", complete)
    for _iteration in range(4):
        observed_payload = contract.audit_evidence_payload(output)
        write_json(output / "payload_audit.json", asdict(observed_payload))
        complete = make_complete(observed_payload.bytes)
        write_json(output / "complete.json", complete)
        stable_payload = contract.audit_evidence_payload(output)
        if stable_payload.bytes == observed_payload.bytes:
            payload = stable_payload
            break
    else:
        raise ValueError("最终 evidence payload 大小未在四轮内收敛")
    if (
        read_json(output / "payload_audit.json").get("bytes") != payload.bytes
        or complete["verified_metrics"]["evidence_payload_bytes"] != payload.bytes
    ):
        raise ValueError("最终 payload_audit/complete 字节口径不一致")
    os.replace(output, final_output)
    chmod_read_only(final_output)
    print(json.dumps({
        "status": complete["status"],
        "identity": IDENTITY,
        "verified_metrics": complete["verified_metrics"],
        "output_dir": str(final_output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
