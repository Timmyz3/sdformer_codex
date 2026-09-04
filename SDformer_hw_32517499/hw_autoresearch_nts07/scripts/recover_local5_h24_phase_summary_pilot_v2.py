#!/usr/bin/env python3
"""从已完成 RTL/Acc32 的中断现场恢复 Local5 H24 phase-summary 后验证包。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {"sample": 2, "stage": 3, "block": 0, "window": 1, "heads": 24}
RECOVERY_SCHEMA = "local5_h24_phase_summary_recovery_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层不是 object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_frozen_runner(source_dir: Path) -> Any:
    sys.path.insert(0, str(source_dir))
    path = source_dir / "run_local5_h24_phase_summary_pilot_v2.py"
    spec = importlib.util.spec_from_file_location("frozen_h24_runner", path)
    if spec is None or spec.loader is None:
        raise ValueError("无法加载冻结 H24 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def run_bound_tests(output: Path, source_binding: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for binding_name in ("verifier_test", "fast_cross_oracle_test"):
        binding = source_binding[binding_name]
        live_path = Path(binding["live_path"])
        snapshot_path = output / binding["snapshot_path"]
        expected_sha = binding["sha256"]
        if (
            not live_path.is_file()
            or not snapshot_path.is_file()
            or sha256_file(live_path) != expected_sha
            or sha256_file(snapshot_path) != expected_sha
        ):
            raise ValueError(f"冻结测试 live/snapshot SHA 不一致: {binding_name}")
        completed = subprocess.run(
            [sys.executable, str(live_path)],
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
            capture_output=True,
            text=True,
        )
        row = {
            "name": live_path.name,
            "binding_name": binding_name,
            "live_path": str(live_path),
            "snapshot_path": str(snapshot_path),
            "live_sha256": sha256_file(live_path),
            "snapshot_sha256": sha256_file(snapshot_path),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        rows.append(row)
        if completed.returncode != 0:
            raise ValueError(f"冻结绑定测试失败: {binding_name}")
    receipt = {
        "schema": "local5_h24_phase_summary_recovery_test_receipt_v1",
        "status": "PASS",
        "tests": rows,
    }
    write_json(output / "recovery_test_receipt.json", receipt)
    return receipt


def verify_recovery_inputs(
    runner: Any,
    output: Path,
    release: Path,
    table_dir: Path,
    vector_dir: Path,
    reference: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = read_json(output / "run_plan.json")
    if (
        plan.get("schema") != "local5_h24_phase_summary_run_plan_v2"
        or plan.get("status") != "FROZEN_BEFORE_RUN_NOT_G0"
        or plan.get("formal_g0") != "DENY"
        or plan.get("actual_identity") != IDENTITY
        or plan.get("identity_status") != "MATCH"
    ):
        raise ValueError("中断现场 run plan 不可恢复")
    source_binding = plan.get("source_bindings")
    if not isinstance(source_binding, dict):
        raise ValueError("中断现场缺少 source bindings")
    for name, row in source_binding.items():
        if not isinstance(row, dict):
            raise ValueError(f"source binding 非 object: {name}")
        snapshot = output / str(row.get("snapshot_path", ""))
        live = Path(str(row.get("live_path", "")))
        expected = row.get("sha256")
        if (
            not isinstance(expected, str)
            or not snapshot.is_file()
            or snapshot.is_symlink()
            or sha256_file(snapshot) != expected
            or not live.is_file()
            or sha256_file(live) != expected
        ):
            raise ValueError(f"source binding 发生变化: {name}")
    input_binding = runner.verify_input_identity(table_dir, vector_dir)
    reference_binding = runner.verify_reference_package(reference)
    runner.load_release(release)
    release_manifest = release / "release_manifest.json"
    if (
        input_binding != plan.get("input_bindings")
        or {
            key: value
            for key, value in reference_binding.items()
            if key.endswith("sha256") or key in {"trace_bytes"}
        }
        != plan.get("reference_bindings")
        or sha256_file(release_manifest) != plan.get("release_manifest_sha256")
        or sha256_file(output / "compile_argv.json")
        != plan.get("telemetry_compile_argv_sha256")
    ):
        raise ValueError("中断现场 input/reference/release/compile 绑定变化")
    return plan, source_binding, input_binding, reference_binding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interrupted-staging", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--reference-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    interrupted = args.interrupted_staging.resolve()
    final_output = args.output_dir.resolve()
    recovery_staging = final_output.with_name(
        final_output.name + f".recovery_staging.{os.getpid()}"
    )
    if not interrupted.is_dir() or interrupted.is_symlink():
        raise ValueError("中断 staging 缺失或为符号链接")
    if final_output.exists() or recovery_staging.exists():
        raise ValueError("恢复输出已存在")
    shutil.copytree(interrupted, recovery_staging, symlinks=False)
    output = recovery_staging
    recovery_dir = output / "recovery"
    recovery_dir.mkdir()
    shutil.copy2(Path(__file__).resolve(), recovery_dir / Path(__file__).name)

    runner = load_frozen_runner(output / "source")
    contract = runner.contract
    fast_cross = runner.fast_cross
    release = args.release.resolve()
    table_dir = args.table_dir.resolve()
    vector_dir = args.vector_dir.resolve()
    reference = args.reference_package.resolve()
    plan, source_binding, input_binding, reference_binding = verify_recovery_inputs(
        runner, output, release, table_dir, vector_dir, reference
    )
    run_bound_tests(output, source_binding)

    run_dir = output / "run"
    required_run_files = (
        "actual.memh",
        "main_summary.csv",
        "phase_intervals.csv",
    )
    if any(not (run_dir / name).is_file() for name in required_run_files):
        raise ValueError("中断现场缺少 RTL 输出")
    numeric = read_json(output / "numeric_verification.json")
    if (
        numeric.get("status") != "PASS_IDENTITY_SERVICE_RTL_TRACE_V2_NOT_G0"
        or numeric.get("formal_g0") != "DENY"
        or numeric.get("identity") != IDENTITY
        or numeric.get("trace_sha256") != reference_binding["trace_sha256"]
        or numeric.get("trace_rows") != reference_binding["trace_rows"]
        or numeric.get("acc32", {}).get("mismatch") != 0
        or numeric.get("acc32", {}).get("scalars") != 345_600
        or numeric.get("acc32", {}).get("actual_acc32_sha256")
        != sha256_file(run_dir / "actual.memh")
        or numeric.get("acc32", {}).get("actual_acc32_sha256")
        != reference_binding["actual_sha256"]
        or numeric.get("acc32", {}).get("expected_npz_sha256")
        != input_binding["software_expected"]
        or numeric.get("verilator_log_sha256")
        != sha256_file(output / "verilator.log")
    ):
        raise ValueError("已完成 numeric verifier receipt 不可恢复")

    main_summary = contract.parse_ordered_summary(run_dir / "main_summary.csv")
    cross_summary = contract.parse_single_observer_summary_glob(
        run_dir / "cross_summary.*.csv",
        expected_schema=contract.CROSS_SUMMARY_SCHEMA,
        expected_target_instance=runner.CROSS_INSTANCE,
    )
    tcfm_summary = contract.parse_single_observer_summary_glob(
        run_dir / "tcfm_summary.*.csv",
        expected_schema=contract.TCFM5_SUMMARY_SCHEMA,
        expected_target_instance=runner.TCFM_INSTANCE,
    )
    phase_ledger = contract.parse_phase_interval_ledger(run_dir / "phase_intervals.csv")
    state_contract, _, state_report = contract.load_and_verify_state_role_contract(ROOT)
    phase_audit = contract.stream_compare_phase_ledger_to_identity_trace(
        phase_ledger,
        reference_binding["trace"],
        state_contract,
        contract.PhaseIdentity(3, 0, 1),
    )
    aligned_audit = contract.verify_main_summary_against_identity_trace(
        main_summary, reference_binding["trace"], heads=24
    )
    recovery_build = output / "recovery_build"
    fast_binary = recovery_build / "local5_cross_protocol_oracle_fast_v1"
    fast_compile = fast_cross.compile_oracle(
        output / "source/local5_cross_protocol_oracle_fast_v1.c", fast_binary
    )
    write_json(output / "fast_cross_oracle_recovery_compile.json", fast_compile)
    cross_oracle, fast_run = fast_cross.verify_cross_summary_pair_fast(
        main_summary, cross_summary, binary=fast_binary, heads=24
    )
    write_json(output / "fast_cross_oracle_run.json", fast_run)
    tcfm_ledger = contract.verify_tcfm5_summary_pair(main_summary, tcfm_summary)
    counts = contract.workload_counts(24)
    if (
        len(phase_ledger.intervals) != counts.phase
        or aligned_audit.rows_read != reference_binding["trace_rows"]
        or cross_oracle.count != counts.cross_total
        or tcfm_ledger.mismatch_count != 0
    ):
        raise ValueError("H24 恢复后 closed-form/trace 计数不匹配")
    tamper_cases = runner.run_tamper_regression(
        output,
        main_summary,
        cross_summary,
        tcfm_summary,
        phase_ledger,
        state_contract,
    )

    runner.run_to_file(
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
    runner.verify_snapshot_unchanged(source_binding, output)
    if (
        runner.verify_input_identity(table_dir, vector_dir) != input_binding
        or sha256_file(reference_binding["trace"]) != reference_binding["trace_sha256"]
        or sha256_file(reference_binding["complete"])
        != reference_binding["complete_sha256"]
        or sha256_file(release / "release_manifest.json")
        != plan["release_manifest_sha256"]
    ):
        raise ValueError("恢复后外部输入发生变化")

    recovery_binding = {
        "schema": RECOVERY_SCHEMA,
        "status": "PASS_POSTVERIFY_RECOVERY_NOT_G0",
        "formal_g0": "DENY",
        "interrupted_staging": str(interrupted),
        "interrupted_run_plan_sha256": sha256_file(interrupted / "run_plan.json"),
        "interrupted_numeric_verification_sha256": sha256_file(
            interrupted / "numeric_verification.json"
        ),
        "recovery_source_sha256": sha256_file(
            recovery_dir / Path(__file__).name
        ),
        "boundary": "复用已完成 RTL 输出和已完成 numeric verifier；重新执行全部 phase/resource/oracle/tamper 后验证。",
    }
    write_json(output / "recovery_receipt.json", recovery_binding)
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
            "oracle_wall_seconds": fast_run["wall_seconds"],
        },
        "tcfm5": asdict(tcfm_ledger),
        "acc32": numeric["acc32"],
        "state_contract": state_report,
        "observer_glob_cardinality": {"cross": 1, "tcfm5": 1},
        "tamper_cases": tamper_cases,
        "recovery": recovery_binding,
        "boundary": [
            "单个 H24 Local5 窗口，不是 formal G0 或 full encoder。",
            "验证 cycle 和摘要耗时不是架构性能或 ASIC PPA。",
            "rolling64 是有序错误检测，不是密码学承诺；文件另用 SHA256 封存。",
            "phase telemetry、observer 与恢复器属于验证基础设施，不是 DATE 架构贡献。",
        ],
    }
    write_json(output / "verification.json", verification)
    (output / "verification.md").write_text(
        "# Local5 H24 紧凑遥测恢复验证结果\n\n"
        "> 状态：PASS（非 formal G0）  \n"
        "> 证据等级：`[rtl]+[软件整数金参考]+[流式独立oracle]`\n\n"
        f"- phase：`{counts.phase}` 条，与 H24 frozen trace 流式一致。\n"
        f"- aligned accepted：`{counts.aligned_total}` 条，五类摘要一致。\n"
        f"- cross-Acc：`{cross_oracle.count}` 条，读写各 `{cross_oracle.read_count}`。\n"
        f"- TCFM5：`{tcfm_ledger.term_count}` term，`{tcfm_ledger.update_count}` update，mask mismatch=0。\n"
        "- Acc32：345,600 标量，软件整数金参考 mismatch=0。\n"
        f"- 负例：`{len(tamper_cases)}` 类全部被拒绝。\n\n"
        "该包从中断现场恢复后验证；RTL 与 numeric verifier 未重跑，phase/resource/oracle/tamper 已全部重跑。"
        "恢复器是验证基础设施，不是架构贡献。\n",
        encoding="utf-8",
    )

    internal_files = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in output.rglob("*")
        if path.is_file()
        and "build" not in path.relative_to(output).parts[:1]
        and "source" not in path.relative_to(output).parts[:1]
    }
    write_json(
        output / "evidence_manifest.json",
        {
            "schema": "local5_h24_phase_summary_evidence_manifest_v2",
            "identity": IDENTITY,
            "files": internal_files,
        },
    )
    final_receipt = {
        "schema": "local5_h24_phase_summary_admission_v2",
        "status": "ADMIT_EVIDENCE_NOT_G0",
        "requested_identity": IDENTITY,
        "actual_identity": IDENTITY,
        "identity_status": "MATCH",
        "receipts": {
        "release": {"status": runner.RELEASE_STATUS},
        "reference": {"status": runner.REFERENCE_STATUS},
        "static_preflight": {"status": "STATIC_PREFLIGHT_PASS_NOT_RUN_ADMISSION"},
        "verification": {"status": verification["status"]},
        },
    }
    final_admission = contract.verify_receipt_admission(
        final_receipt,
        allowed_schemas=["local5_h24_phase_summary_admission_v2"],
        allowed_statuses=["ADMIT_EVIDENCE_NOT_G0"],
        required_receipts={
            "release": [runner.RELEASE_STATUS],
            "reference": [runner.REFERENCE_STATUS],
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
            "recovery": recovery_binding,
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
                "reference_trace_rows": reference_binding["trace_rows"],
                "reference_trace_bytes": reference_binding["trace_bytes"],
                "new_full_trace_bytes": 0,
                "evidence_payload_bytes": payload_bytes,
            },
            "external_bindings": {
                "release_manifest": {
                    "path": str(release / "release_manifest.json"),
                    "sha256": plan["release_manifest_sha256"],
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
        observed = contract.audit_evidence_payload(output)
        write_json(output / "payload_audit.json", asdict(observed))
        complete = make_complete(observed.bytes)
        write_json(output / "complete.json", complete)
        stable = contract.audit_evidence_payload(output)
        if stable.bytes == observed.bytes:
            payload = stable
            break
    else:
        raise ValueError("恢复包 payload 大小未收敛")
    if (
        read_json(output / "payload_audit.json").get("bytes") != payload.bytes
        or complete["verified_metrics"]["evidence_payload_bytes"] != payload.bytes
    ):
        raise ValueError("恢复包 payload/complete 口径不一致")
    os.replace(output, final_output)
    runner.chmod_read_only(final_output)
    print(
        json.dumps(
            {
                "status": complete["status"],
                "verified_metrics": complete["verified_metrics"],
                "output_dir": str(final_output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
