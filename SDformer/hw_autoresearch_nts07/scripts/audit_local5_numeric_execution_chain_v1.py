#!/usr/bin/env python3
"""只读重建 Local5 数值 shard 的可证执行 receipt 链，不推测缺失历史。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXECUTIONS = {"RUN", "RESUME_INCOMPLETE_SHARD", "SKIP_ALREADY_SEALED"}
LAUNCHER_POLICY = "每个 sample 执行前后均 fail-closed 校验 live SHA"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层不是 JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_source_snapshots(batch: Path, plan: dict[str, Any]) -> dict[str, Any]:
    snapshots = plan.get("source_snapshots")
    if not isinstance(snapshots, list):
        return {"tier": "NO_SOURCE_SNAPSHOT", "files": []}
    rows = []
    names: set[str] = set()
    for row in snapshots:
        if not isinstance(row, dict):
            raise ValueError(f"{batch.name} source snapshot 非 object")
        path = Path(str(row.get("path", ""))).resolve()
        name = str(row.get("name", ""))
        if (
            not path.is_file()
            or not path.is_relative_to(batch / "source")
            or path.name != name
            or name in names
            or row.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"{batch.name} source snapshot 不一致")
        names.add(name)
        rows.append({"name": name, "sha256": row["sha256"]})
    launcher = plan.get("invocation_launcher")
    runtime = plan.get("runtime_environment")
    if isinstance(runtime, dict):
        tier = "KEY_RUNTIME_COMPONENTS_AND_SOURCE_SNAPSHOTS"
    elif isinstance(launcher, dict):
        snapshot = Path(str(launcher.get("snapshot_path", ""))).resolve()
        if (
            not snapshot.is_file()
            or not snapshot.is_relative_to(batch / "source")
            or launcher.get("sha256") != sha256_file(snapshot)
            or launcher.get("policy") != LAUNCHER_POLICY
            or not any(
                row["name"] == snapshot.name
                and row["sha256"] == launcher.get("sha256")
                for row in rows
            )
        ):
            raise ValueError(f"{batch.name} launcher snapshot/policy 不一致")
        tier = "RUNNER_TEST_LAUNCHER_SNAPSHOTS"
    else:
        tier = "RUNNER_SNAPSHOT_ONLY"
    return {"tier": tier, "files": rows}


def collect_receipts(
    sample: int,
    sample_root: Path,
    expected: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for receipt_path in RESULTS.glob(
        f"local5_numeric_samples*_batch*/sample{sample}.receipt.json"
    ):
        batch = receipt_path.parent.resolve()
        if ".failed" in batch.name or ".staging" in batch.name:
            continue
        plan_path = batch / "plan.json"
        if not plan_path.is_file():
            continue
        receipt = read_json(receipt_path)
        plan = read_json(plan_path)
        outputs = plan.get("outputs")
        if (
            plan.get("schema") not in {
                "local5_numeric_sample_batch_plan_v1",
                "local5_numeric_sample_batch_plan_v2",
            }
            or plan.get("status") != "FROZEN_BEFORE_RUN_NOT_G0"
            or plan.get("formal_g0") != "DENY"
            or sample not in plan.get("samples", [])
            or not isinstance(outputs, dict)
            or Path(str(outputs.get(str(sample), ""))).resolve()
            != sample_root.resolve()
            or
            receipt.get("sample") != sample
            or receipt.get("status") != "PASS"
            or receipt.get("execution") not in EXECUTIONS
            or Path(str(receipt.get("output", ""))).resolve()
            != sample_root.resolve()
            or receipt.get("complete_sha256") != expected["complete"]
            or receipt.get("report_sha256") != expected["report"]
            or receipt.get("acc32_archive_sha256") != expected["archive"]
            or receipt.get("mismatch") != 0
            or receipt.get("windows") != 12
            or receipt.get("acc32_scalars") != 1_987_200
        ):
            continue
        complete_path = batch / "complete.json"
        batch_status = "PARTIAL_OR_INTERRUPTED_BATCH_RECEIPT"
        complete_sha = None
        if complete_path.is_file():
            complete = read_json(complete_path)
            complete_row = next(
                (
                    row for row in complete.get("rows", [])
                    if isinstance(row, dict) and row.get("sample") == sample
                ),
                None,
            )
            if (
                complete.get("status") == "PASS_NUMERIC_SAMPLE_BATCH_NOT_G0"
                and complete.get("samples_failed") == []
                and complete.get("plan_sha256") == sha256_file(plan_path)
                and complete_row == receipt
            ):
                batch_status = "PASS_BATCH_RECEIPT"
            else:
                batch_status = "FAILED_OR_INCOMPLETE_BATCH_RECEIPT"
            complete_sha = sha256_file(complete_path)
        execution = receipt["execution"]
        execution_artifacts = None
        if execution in {"RUN", "RESUME_INCOMPLETE_SHARD"}:
            log_path = batch / f"sample{sample}.log"
            timing_path = batch / f"sample{sample}.time.json"
            timing = read_json(timing_path)
            if (
                not log_path.is_file()
                or timing.get("schema") != "local5_numeric_sample_batch_timing_v1"
                or timing.get("sample") != sample
                or timing.get("returncode") != 0
                or timing.get("log_sha256") != sha256_file(log_path)
                or receipt.get("log_sha256") != sha256_file(log_path)
                or receipt.get("timing_sha256") != sha256_file(timing_path)
            ):
                raise ValueError(f"{batch.name} sample{sample} RUN/RESUME 日志不一致")
            execution_artifacts = {
                "log_sha256": sha256_file(log_path),
                "timing_sha256": sha256_file(timing_path),
                "returncode": timing["returncode"],
            }
        rows.append({
            "batch_root": str(batch),
            "batch_name": batch.name,
            "batch_status": batch_status,
            "execution": receipt["execution"],
            "plan_sha256": sha256_file(plan_path),
            "batch_complete_sha256": complete_sha,
            "sample_receipt_sha256": sha256_file(receipt_path),
            "execution_artifacts": execution_artifacts,
            "source_provenance": validate_source_snapshots(batch, plan),
            "runtime_scope": (
                "KEY_COMPONENTS_FOR_THIS_BATCH_ONLY"
                if isinstance(plan.get("runtime_environment"), dict)
                else "NOT_RECORDED_IN_BATCH_PLAN"
            ),
        })
    return sorted(rows, key=lambda row: row["batch_name"])


def select_proven_execution(
    receipts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for mode in ("RUN", "RESUME_INCOMPLETE_SHARD"):
        candidates = [row for row in receipts if row["execution"] == mode]
        if candidates:
            return candidates[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    coverage_root = args.coverage.resolve()
    coverage_path = coverage_root / "coverage_audit.json"
    coverage = read_json(coverage_path)
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + f".staging.{os.getpid()}")
    if output.exists() or staging.exists():
        raise FileExistsError(f"输出已存在: {output} 或 {staging}")
    staging.mkdir(parents=True)
    source = staging / "source"
    source.mkdir()
    script_snapshot = source / Path(__file__).name
    test_live = Path(__file__).with_name(
        "test_audit_local5_numeric_execution_chain_v1.py"
    )
    test_snapshot = source / test_live.name
    shutil.copyfile(Path(__file__).resolve(), script_snapshot)
    shutil.copyfile(test_live, test_snapshot)
    try:
        if (
            coverage.get("status") != "PASS_CONTIGUOUS_NUMERIC_COVERAGE_NOT_G0"
            or coverage.get("formal_g0") != "DENY"
            or not isinstance(coverage.get("rows"), list)
        ):
            raise ValueError("coverage package 未正向准入")
        rows = []
        for coverage_row in coverage["rows"]:
            sample = int(coverage_row["sample"])
            sample_root = Path(coverage_row["sample_root"]).resolve()
            expected = {
                "complete": coverage_row["complete_sha256"],
                "report": coverage_row["report_sha256"],
                "archive": coverage_row["archive_sha256"],
            }
            if sample < 3:
                receipts: list[dict[str, Any]] = []
                origin = None
                origin_class = "STANDALONE_PRE_BATCH_PROVENANCE"
            else:
                receipts = collect_receipts(sample, sample_root, expected)
                origin = select_proven_execution(receipts)
                origin_class = (
                    "PROVEN_RUN_OR_RESUME_RECEIPT"
                    if origin is not None else "LEGACY_GAP_NO_RUN_OR_RESUME_RECEIPT"
                )
            rows.append({
                "sample": sample,
                "sample_root": str(sample_root),
                "origin_class": origin_class,
                "selected_proven_execution": origin,
                "validated_receipt_chain": receipts,
            })
        counts: dict[str, int] = {}
        mode_counts: dict[str, int] = {}
        for row in rows:
            counts[row["origin_class"]] = counts.get(row["origin_class"], 0) + 1
            origin = row["selected_proven_execution"]
            if origin is not None:
                mode = origin["execution"]
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
        report = {
            "schema": "local5_numeric_execution_chain_audit_v2",
            "status": "PASS_EXECUTION_CHAIN_WITH_EXPLICIT_GAPS_NOT_G0",
            "formal_g0": "DENY",
            "coverage_root": str(coverage_root),
            "coverage_audit_sha256": sha256_file(coverage_path),
            "samples": len(rows),
            "origin_class_counts": counts,
            "proven_execution_mode_counts": mode_counts,
            "rows": rows,
            "correction_precedence": {
                "supersedes_boundary_in": str(coverage_path),
                "superseded_text": "env-sealed 复验批次绑定首次执行批次",
                "replacement": (
                    "env-sealed 复验批次只绑定直接父验证批次；本 execution-chain v2 "
                    "另行搜索与最终 shard SHA 相同的 RUN/RESUME receipt，并显式保留缺口。"
                ),
                "publication_rule": "论文与后续看板以 execution-chain v2 为准。",
            },
            "boundary": [
                "receipt 链只证明与最终 shard SHA 相同的 RUN/RESUME/SKIP 记录，不重建没有 receipt 的历史。",
                "PARTIAL batch 中的 sample receipt 可证明该 sample 已验封，但不把整个 batch 倒签为 PASS。",
                "runtime 记录只覆盖相应批次的关键组件；SKIP 批次不证明原始窗口生成环境。",
                "执行 provenance、SHA 和审计器是验证证据，不是 DATE 架构创新。",
            ],
        }
        write_json(staging / "execution_chain_audit.json", report)
        (staging / "execution_chain_audit.md").write_text(
            "# Local5 数值执行来源链只读审计\n\n"
            "> 状态：PASS（显式保留历史缺口，非 formal G0）\n\n"
            f"- sample：`{len(rows)}`。\n"
            f"- 可证 `RUN/RESUME`：`{counts.get('PROVEN_RUN_OR_RESUME_RECEIPT', 0)}`。\n"
            f"- 独立 pre-batch：`{counts.get('STANDALONE_PRE_BATCH_PROVENANCE', 0)}`。\n"
            f"- 无 `RUN/RESUME` receipt 的 legacy gap："
            f"`{counts.get('LEGACY_GAP_NO_RUN_OR_RESUME_RECEIPT', 0)}`。\n\n"
            "该结果不会把直接父批次误称为首次执行，也不会把 SKIP 复验环境外推为原始生成环境。\n",
            encoding="utf-8",
        )
        manifest = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in staging.rglob("*") if path.is_file()
        }
        write_json(staging / "manifest.json", {
            "schema": "local5_numeric_execution_chain_manifest_v2",
            "files": manifest,
        })
        write_json(staging / "complete.json", {
            "schema": "local5_numeric_execution_chain_complete_v2",
            "status": "PASS_SEALED_EXECUTION_CHAIN_WITH_GAPS_NOT_G0",
            "formal_g0": "DENY",
            "report_sha256": sha256_file(staging / "execution_chain_audit.json"),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "source_sha256": sha256_file(script_snapshot),
            "test_source_sha256": sha256_file(test_snapshot),
        })
        os.replace(staging, output)
    except BaseException as error:
        write_json(staging / "failure_receipt.json", {
            "schema": "local5_numeric_execution_chain_failure_v2",
            "status": "FAIL_CLOSED_NOT_G0",
            "formal_g0": "DENY",
            "exception_type": type(error).__name__,
            "exception": str(error),
        })
        raise
    print(json.dumps({
        "status": report["status"],
        "samples": report["samples"],
        "origin_class_counts": report["origin_class_counts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
