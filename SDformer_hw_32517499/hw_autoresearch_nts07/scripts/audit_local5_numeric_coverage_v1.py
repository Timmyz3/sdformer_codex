#!/usr/bin/env python3
"""只读审计 Local5 连续 sample 的数值 RTL 覆盖与 provenance。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WINDOWS_PER_SAMPLE = 12
SCALARS_PER_SAMPLE = 1_987_200
EXPECTED_TOPOLOGY = (
    (0, 0, 3),
    (0, 1, 3),
    (1, 0, 6),
    (1, 1, 6),
    (2, 0, 12),
    (2, 1, 12),
    (2, 2, 12),
    (2, 3, 12),
    (2, 4, 12),
    (2, 5, 12),
    (3, 0, 24),
    (3, 1, 24),
)
V5_RELEASE_SHA256 = "c620cf6a33f1c9bbdb1c7d85ba0fa485580f8f578287850d08b7c6ee52939bf9"
LAUNCHER_NAME = "run_local5_erep_numeric_sample_shard.sh"
BATCH_RUNNER_NAME = "run_local5_numeric_sample_batch_v1.py"
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
        raise ValueError(f"{path} 不是 JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sample_directory(sample: int) -> tuple[Path, str]:
    if sample == 0:
        return (
            RESULTS / "local5_erep_numeric_sample0_shard_v1_reviewfix_20260811",
            "LEGACY_STANDALONE_WITH_SOURCE_BUNDLE",
        )
    if sample == 1:
        return (
            RESULTS / "local5_erep_numeric_sample1_shard_v2_release_20260811",
            "SEALED_RELEASE_V2",
        )
    if sample == 2:
        return (
            RESULTS / "local5_erep_numeric_sample2_shard_v5_release_20260811",
            "SEALED_RELEASE_V5",
        )
    return (
        RESULTS / f"local5_erep_numeric_sample{sample}_shard_v5_batch_20260812",
        "SEALED_RELEASE_V5",
    )


def validate_sha_list(path: Path, root: Path) -> int:
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError(f"空 SHA 清单：{path}")
    root = root.resolve()
    for row in rows:
        digest, separator, target_text = row.partition("  ")
        target = Path(target_text).resolve()
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not target.is_relative_to(root)
            or not target.is_file()
            or sha256_file(target) != digest
        ):
            raise ValueError(f"SHA 清单条目无效：{row}")
    return len(rows)


def validate_archive(
    archive_path: Path,
    sample: int,
    report_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "window_sample",
        "window_stage",
        "window_block",
        "window_token",
        "window_weight",
        "window_heads",
        "window_value_offsets",
        "expected_acc32",
        "actual_acc32",
    }
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"sample{sample} Acc32 schema 不一致")
        schema = archive["schema_version"]
        samples = archive["window_sample"]
        stages = archive["window_stage"]
        blocks = archive["window_block"]
        tokens = archive["window_token"]
        weights = archive["window_weight"]
        heads = archive["window_heads"]
        offsets = archive["window_value_offsets"]
        expected = archive["expected_acc32"]
        actual = archive["actual_acc32"]
        expected_offsets = np.zeros(WINDOWS_PER_SAMPLE + 1, dtype=np.int64)
        expected_offsets[1:] = np.cumsum(heads.astype(np.int64) * 450 * 32)
        archive_sequence = tuple(zip(
            stages.tolist(),
            blocks.tolist(),
            heads.tolist(),
            tokens.tolist(),
            weights.tolist(),
        ))
        report_sequence = tuple(
            (
                row.get("stage"),
                row.get("block"),
                row.get("heads"),
                row.get("window"),
                row.get("weight"),
            )
            for row in report_windows
        )
        if (
            schema.dtype != np.uint16
            or schema.shape != (1,)
            or int(schema[0]) != 4
            or samples.dtype != np.uint16
            or samples.shape != (WINDOWS_PER_SAMPLE,)
            or not np.all(samples == sample)
            or stages.dtype != np.uint8
            or blocks.dtype != np.uint8
            or heads.dtype != np.uint8
            or tokens.dtype != np.uint16
            or weights.dtype != np.uint16
            or tuple(zip(stages.tolist(), blocks.tolist(), heads.tolist()))
            != EXPECTED_TOPOLOGY
            or archive_sequence != report_sequence
            or offsets.dtype != np.int64
            or offsets.shape != (WINDOWS_PER_SAMPLE + 1,)
            or not np.array_equal(offsets, expected_offsets)
            or int(offsets[-1]) != SCALARS_PER_SAMPLE
            or expected.dtype != np.int32
            or actual.dtype != np.int32
            or expected.shape != (SCALARS_PER_SAMPLE,)
            or actual.shape != expected.shape
            or not np.array_equal(expected, actual)
        ):
            raise ValueError(f"sample{sample} Acc32 内容或拓扑不一致")
        return {
            "schema_version": int(schema[0]),
            "scalars": int(expected.size),
            "mismatch": int(np.count_nonzero(expected != actual)),
            "expected_digest": hashlib.sha256(expected.tobytes()).hexdigest(),
            "actual_digest": hashlib.sha256(actual.tobytes()).hexdigest(),
        }


def validate_release_binding(
    sample_root: Path,
    tier: str,
    complete: dict[str, Any],
) -> dict[str, Any]:
    binding_path = sample_root / "release_binding.json"
    if tier == "LEGACY_STANDALONE_WITH_SOURCE_BUNDLE":
        source_bundle = sample_root / "source_bundle.tar"
        if binding_path.exists() or not source_bundle.is_file():
            raise ValueError("sample0 legacy provenance 边界不一致")
        return {
            "tier": tier,
            "release_manifest_sha256": None,
            "source_bundle_sha256": sha256_file(source_bundle),
        }
    binding = read_json(binding_path)
    manifest = Path(str(binding.get("release_manifest", ""))).resolve()
    release_complete = Path(str(binding.get("release_complete", ""))).resolve()
    if (
        binding.get("schema") != "local5_erep_numeric_sample_release_binding_v1"
        or binding.get("status") != "PASS_RELEASE_BOUND_NOT_G0"
        or binding.get("formal_g0") != "DENY"
        or complete.get("release_binding_sha256") != sha256_file(binding_path)
        or not manifest.is_file()
        or binding.get("release_manifest_sha256") != sha256_file(manifest)
        or not release_complete.is_file()
        or binding.get("release_complete_sha256") != sha256_file(release_complete)
    ):
        raise ValueError(f"{sample_root.name} release binding 不一致")
    release_sha = str(binding["release_manifest_sha256"])
    if tier == "SEALED_RELEASE_V5" and release_sha != V5_RELEASE_SHA256:
        raise ValueError(f"{sample_root.name} 未绑定冻结 v5 release")
    return {
        "tier": tier,
        "release_manifest": str(manifest),
        "release_manifest_sha256": release_sha,
        "release_complete_sha256": str(binding["release_complete_sha256"]),
    }


def validate_batch_binding(
    sample: int,
    sample_root: Path,
    complete_sha256: str,
    report_sha256: str,
    archive_sha256: str,
) -> dict[str, Any] | None:
    """寻找并校验包含该 sample 的最强成功 batch provenance。"""
    if sample < 3:
        return None
    candidates: list[tuple[int, Path, dict[str, Any], dict[str, Any]]] = []
    for complete_path in RESULTS.glob("local5_numeric_samples*_batch*_20260812/complete.json"):
        batch_root = complete_path.parent.resolve()
        complete = read_json(complete_path)
        plan_path = batch_root / "plan.json"
        if not plan_path.is_file():
            continue
        plan = read_json(plan_path)
        plan_samples = plan.get("samples")
        if (
            complete.get("schema") != "local5_numeric_sample_batch_complete_v1"
            or complete.get("status") != "PASS_NUMERIC_SAMPLE_BATCH_NOT_G0"
            or complete.get("formal_g0") != "DENY"
            or sample not in complete.get("samples_passed", [])
            or complete.get("samples_requested") != plan_samples
            or complete.get("samples_failed") != []
        ):
            continue
        outputs = plan.get("outputs")
        if (
            plan.get("schema") != "local5_numeric_sample_batch_plan_v1"
            or plan.get("status") != "FROZEN_BEFORE_RUN_NOT_G0"
            or sample not in plan.get("samples", [])
            or complete.get("plan_sha256") != sha256_file(plan_path)
            or not isinstance(outputs, dict)
            or Path(str(outputs.get(str(sample), ""))).resolve()
            != sample_root.resolve()
        ):
            continue
        snapshots = plan.get("source_snapshots")
        launcher = plan.get("invocation_launcher")
        score = 1
        if isinstance(snapshots, list) and snapshots:
            score = 2
        if isinstance(launcher, dict):
            score = 3
        candidates.append((score, batch_root, complete, plan))
    if not candidates:
        raise ValueError(f"sample{sample} 找不到成功 batch binding")
    candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    score, batch_root, complete, plan = candidates[0]
    receipt_path = batch_root / f"sample{sample}.receipt.json"
    receipt = read_json(receipt_path)
    if (
        receipt.get("sample") != sample
        or receipt.get("status") != "PASS"
        or Path(str(receipt.get("output", ""))).resolve() != sample_root.resolve()
        or receipt.get("complete_sha256") != complete_sha256
        or receipt.get("report_sha256") != report_sha256
        or receipt.get("acc32_archive_sha256") != archive_sha256
        or receipt.get("windows") != WINDOWS_PER_SAMPLE
        or receipt.get("acc32_scalars") != SCALARS_PER_SAMPLE
        or receipt.get("mismatch") != 0
        or complete.get("release_manifest_sha256") != V5_RELEASE_SHA256
    ):
        raise ValueError(f"sample{sample} batch receipt 与 shard 不一致")
    snapshots = plan.get("source_snapshots")
    snapshot_rows = []
    snapshot_names: set[str] = set()
    snapshot_paths: set[Path] = set()
    if isinstance(snapshots, list):
        for row in snapshots:
            if not isinstance(row, dict):
                raise ValueError(f"sample{sample} batch source snapshot 非 object")
            path = Path(str(row.get("path", ""))).resolve()
            name = str(row.get("name", ""))
            if (
                not path.is_file()
                or row.get("sha256") != sha256_file(path)
                or not path.is_relative_to(batch_root / "source")
                or path.name != name
                or name in snapshot_names
                or path in snapshot_paths
            ):
                raise ValueError(f"sample{sample} batch source snapshot 不一致")
            snapshot_names.add(name)
            snapshot_paths.add(path)
            snapshot_rows.append({
                "name": name,
                "path": str(path),
                "sha256": row.get("sha256"),
            })
    launcher = plan.get("invocation_launcher")
    launcher_row = None
    if isinstance(launcher, dict):
        snapshot_path = Path(str(launcher.get("snapshot_path", ""))).resolve()
        if (
            set(snapshot_names) != {
                BATCH_RUNNER_NAME,
                "test_run_local5_numeric_sample_batch_v1.py",
                LAUNCHER_NAME,
            }
            or not snapshot_path.is_file()
            or snapshot_path.name != LAUNCHER_NAME
            or not snapshot_path.is_relative_to(batch_root / "source")
            or launcher.get("sha256") != sha256_file(snapshot_path)
            or launcher.get("policy") != LAUNCHER_POLICY
            or not any(
                row["path"] == str(snapshot_path)
                and row["sha256"] == launcher.get("sha256")
                for row in snapshot_rows
            )
        ):
            raise ValueError(f"sample{sample} batch launcher snapshot 不一致")
        launcher_row = {
            "snapshot_path": str(snapshot_path),
            "sha256": launcher.get("sha256"),
            "policy": launcher.get("policy"),
        }
        complete_snapshots = complete.get("source_snapshots")
        if complete_snapshots != snapshots:
            raise ValueError(f"sample{sample} batch complete/plan source snapshots 不一致")
        complete_row = next(
            (
                row for row in complete.get("rows", [])
                if isinstance(row, dict) and row.get("sample") == sample
            ),
            None,
        )
        if complete_row != receipt:
            raise ValueError(f"sample{sample} batch complete row/receipt 不一致")
    tier = {
        1: "LEGACY_BATCH_NO_SOURCE_SNAPSHOT",
        2: "BATCH_RUNNER_SNAPSHOT_ONLY",
        3: "BATCH_RUNNER_LAUNCHER_SNAPSHOT_SHA_FAILCLOSED",
    }[score]
    return {
        "tier": tier,
        "batch_root": str(batch_root),
        "batch_complete_sha256": sha256_file(batch_root / "complete.json"),
        "batch_plan_sha256": sha256_file(batch_root / "plan.json"),
        "sample_receipt_sha256": sha256_file(receipt_path),
        "source_snapshots": snapshot_rows,
        "invocation_launcher": launcher_row,
    }


def audit_sample(sample: int) -> dict[str, Any]:
    sample_root, tier = sample_directory(sample)
    complete_path = sample_root / "complete.json"
    report_path = sample_root / "shard/numeric_shard_report.json"
    archive_path = sample_root / "shard/acc32_miter_shard.npz"
    if not all(path.is_file() for path in (complete_path, report_path, archive_path)):
        raise FileNotFoundError(f"sample{sample} shard 尚未完整封存：{sample_root}")
    complete = read_json(complete_path)
    report = read_json(report_path)
    windows = report.get("windows")
    if (
        complete.get("schema") != "local5_erep_numeric_sample_shard_complete_v1"
        or complete.get("status") != "PASS_NUMERIC_SAMPLE_SHARD_NOT_G0"
        or complete.get("formal_g0") != "DENY"
        or complete.get("sample") != sample
        or report.get("schema") != "local5_erep_numeric_sample_shard_v1"
        or report.get("status") != "PASS_NUMERIC_SAMPLE_SHARD_NOT_G0"
        or report.get("formal_g0") != "DENY"
        or report.get("sample") != sample
        or report.get("window_count") != WINDOWS_PER_SAMPLE
        or report.get("final_acc32_scalar_count") != SCALARS_PER_SAMPLE
        or report.get("mismatch_count") != 0
        or report.get("max_abs_error") != 0
        or not isinstance(windows, list)
        or len(windows) != WINDOWS_PER_SAMPLE
        or Path(str(report.get("archive", ""))).resolve() != archive_path.resolve()
        or report.get("archive_sha256") != sha256_file(archive_path)
    ):
        raise ValueError(f"sample{sample} complete/report 合同不一致")
    release = validate_release_binding(sample_root, tier, complete)
    archive = validate_archive(archive_path, sample, windows)
    complete_sha = sha256_file(complete_path)
    report_sha = sha256_file(report_path)
    archive_sha = sha256_file(archive_path)
    batch = validate_batch_binding(
        sample,
        sample_root,
        complete_sha,
        report_sha,
        archive_sha,
    )
    sha_lists = {}
    for name in ("result_sha256.txt", "receipt_sha256.txt", "window_receipt_sha256.txt"):
        path = sample_root / name
        if path.is_file():
            sha_lists[name] = validate_sha_list(path, sample_root)
    return {
        "sample": sample,
        "status": "PASS",
        "sample_root": str(sample_root),
        "provenance": release,
        "batch_provenance": batch,
        "complete_sha256": complete_sha,
        "report_sha256": report_sha,
        "archive_sha256": archive_sha,
        "archive": archive,
        "sha_list_rows": sha_lists,
        "windows": WINDOWS_PER_SAMPLE,
        "acc32_scalars": SCALARS_PER_SAMPLE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.end >= 100 or args.start > args.end:
        raise ValueError("sample 范围必须位于 [0,99] 且连续")
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + f".staging.{os.getpid()}")
    if output.exists() or staging.exists():
        raise FileExistsError(f"输出目录已存在：{output} 或 {staging}")
    staging.mkdir(parents=True)
    source = staging / "source"
    source.mkdir()
    source_snapshot = source / Path(__file__).name
    shutil.copyfile(Path(__file__).resolve(), source_snapshot)
    source_snapshot.chmod(0o444)
    try:
        rows = [audit_sample(sample) for sample in range(args.start, args.end + 1)]
        samples = [row["sample"] for row in rows]
        expected_samples = list(range(args.start, args.end + 1))
        if samples != expected_samples or len(set(samples)) != len(samples):
            raise ValueError("sample 集合不连续或重复")
        tier_counts: dict[str, int] = {}
        batch_tier_counts: dict[str, int] = {}
        for row in rows:
            tier = str(row["provenance"]["tier"])
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if row["batch_provenance"] is not None:
                batch_tier = str(row["batch_provenance"]["tier"])
                batch_tier_counts[batch_tier] = batch_tier_counts.get(batch_tier, 0) + 1
        report = {
            "schema": "local5_numeric_coverage_audit_v1",
            "status": "PASS_CONTIGUOUS_NUMERIC_COVERAGE_NOT_G0",
            "evidence": "[rtl]+[软件整数金参考]+[只读累计审计]",
            "formal_g0": "DENY",
            "sample_start": args.start,
            "sample_end": args.end,
            "samples": len(rows),
            "windows": sum(row["windows"] for row in rows),
            "acc32_scalars": sum(row["acc32_scalars"] for row in rows),
            "mismatch": sum(row["archive"]["mismatch"] for row in rows),
            "provenance_tier_counts": tier_counts,
            "batch_provenance_tier_counts": batch_tier_counts,
            "same_sealed_v5": {
                "sample_start": max(args.start, 2),
                "sample_end": args.end,
                "samples": sum(
                    row["provenance"]["tier"] == "SEALED_RELEASE_V5"
                    for row in rows
                ),
                "release_manifest_sha256": V5_RELEASE_SHA256,
            },
            "rows": rows,
            "dependency_policy": {
                "mode": "WORKSPACE_BOUND_SHA256",
                "description": (
                    "累计包保存外部 shard/release/batch 的绝对路径与 SHA256；"
                    "当前工作区可复算，但未复制大体积依赖，不能声称为可迁移独立 artifact。"
                ),
            },
            "source": {
                "path": "source/audit_local5_numeric_coverage_v1.py",
                "sha256": sha256_file(source_snapshot),
            },
            "boundary": [
                "sample0 为早期独立 source bundle，sample1 为 sealed v2，不能倒签为 v5。",
                "sample2 起才属于同一 sealed v5 release；连续覆盖与同 release 覆盖分账。",
                "sample3 起额外绑定成功 batch plan/receipt；旧批次的 launcher 缺口保留分级。",
                "每个 sample 仅含 12 个 canonical block-window，不是全部空间窗口。",
                "该审计不生成 phase ledger，不改变 formal G0=DENY。",
                "验证回归数据不是部署性能、能耗或 ASIC PPA。",
                "累计审计只复核冻结 expected/actual；不独立重跑软件模型生成 expected。",
            ],
        }
        write_json(staging / "coverage_audit.json", report)
        (staging / "coverage_audit.md").write_text(
            "# Local5 数值 RTL 连续覆盖只读审计\n\n"
            "> 状态：PASS（非 formal G0）  \n"
            "> 证据：`[rtl]+[软件整数金参考]+[只读累计审计]`\n\n"
            f"- 连续 sample：`{args.start}-{args.end}`，共 `{len(rows)}` 个。\n"
            f"- canonical block-window：`{report['windows']}` 个。\n"
            f"- Acc32：`{report['acc32_scalars']}` 个，mismatch=`0`。\n"
            f"- 同一 sealed v5：sample2-{args.end}，共 "
            f"`{report['same_sealed_v5']['samples']}` 个 sample。\n\n"
            "## 证据边界\n\n"
            "sample0/1 保留其原始 provenance 等级，不倒签为 v5；本结果只提高数值 "
            "RTL 覆盖，不等价于正式 phase ledger、full encoder、性能或 ASIC PPA。"
            "累计审计逐元素复核冻结 expected/actual，但不重新运行软件模型生成 expected；"
            "证据依赖以工作区绝对路径和 SHA256 绑定，尚不是可迁移独立 artifact。\n",
            encoding="utf-8",
        )
        manifest = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in staging.rglob("*")
            if path.is_file()
        }
        write_json(staging / "evidence_manifest.json", {
            "schema": "local5_numeric_coverage_evidence_manifest_v1",
            "files": manifest,
        })
        write_json(staging / "complete.json", {
            "schema": "local5_numeric_coverage_complete_v1",
            "status": "PASS_SEALED_CONTIGUOUS_NUMERIC_COVERAGE_NOT_G0",
            "formal_g0": "DENY",
            "coverage_audit_sha256": sha256_file(staging / "coverage_audit.json"),
            "evidence_manifest_sha256": sha256_file(staging / "evidence_manifest.json"),
            "source_sha256": sha256_file(source_snapshot),
        })
        os.replace(staging, output)
    except BaseException as error:
        write_json(staging / "failure_receipt.json", {
            "schema": "local5_numeric_coverage_failure_v1",
            "status": "FAIL_CLOSED_NOT_G0",
            "formal_g0": "DENY",
            "exception_type": type(error).__name__,
            "exception": str(error),
        })
        raise
    print(json.dumps({
        "status": report["status"],
        "samples": report["samples"],
        "windows": report["windows"],
        "acc32_scalars": report["acc32_scalars"],
        "same_sealed_v5_samples": report["same_sealed_v5"]["samples"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
