#!/usr/bin/env python3
"""只读合并一个 Local5 sample 的 12 窗口 numeric shard。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .local5_erep_archive_replay_v4 import (
        ARCHIVE_SCHEMA_VERSION,
        MITER_ARRAY_SPECS,
        _validate_npz_container,
        parse_miter_archive,
    )
    from .local5_erep_integrated_cross_head_actual import parse_acc32, sha256
    from .local5_erep_integrated_cross_head_merge import (
        receipt_matches,
        validate_execution_binding,
    )
else:
    from local5_erep_archive_replay_v4 import (
        ARCHIVE_SCHEMA_VERSION,
        MITER_ARRAY_SPECS,
        _validate_npz_container,
        parse_miter_archive,
    )
    from local5_erep_integrated_cross_head_actual import parse_acc32, sha256
    from local5_erep_integrated_cross_head_merge import (
        receipt_matches,
        validate_execution_binding,
    )


TOKENS = 450
OUT_DIM = 32
BLOCKS = (
    (0, 0, 3, 440),
    (0, 1, 3, 440),
    (1, 0, 6, 120),
    (1, 1, 6, 120),
    (2, 0, 12, 30),
    (2, 1, 12, 30),
    (2, 2, 12, 30),
    (2, 3, 12, 30),
    (2, 4, 12, 30),
    (2, 5, 12, 30),
    (3, 0, 24, 10),
    (3, 1, 24, 10),
)


def canonical_json_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 不是 JSON object")
    return value


def _validate_complete(
    path: Path, expected_identity: dict[str, int], bindings: dict[str, Path]
) -> None:
    receipt = _read_json(path)
    if (
        receipt.get("schema") != "local5_erep_numeric_window_complete_v1"
        or receipt.get("status") != "SEALED_READY_FOR_MITER_NOT_G0"
        or receipt.get("formal_g0") != "DENY"
        or receipt.get("identity") != expected_identity
    ):
        raise ValueError("窗口完成标志或坐标不合法")
    observed = receipt.get("artifact_sha256")
    if not isinstance(observed, dict) or set(observed) != set(bindings):
        raise ValueError("窗口完成标志 artifact 集合不精确")
    for name, artifact in bindings.items():
        if not artifact.is_file() or observed[name] != sha256(artifact):
            raise ValueError(f"窗口完成标志 SHA 失配: {name}")


def _load_expected(
    directory: Path, identity: dict[str, int]
) -> tuple[dict[str, Any], np.ndarray]:
    plan_path = directory / "software_expected/task_plan.json"
    expected_path = directory / "software_expected/software_expected.npz"
    receipt_path = directory / "software_expected/software_expected_receipt.json"
    plan = _read_json(plan_path)
    receipt = _read_json(receipt_path)
    if (
        plan.get("schema") != "local5_projection_task_plan_v1"
        or plan.get("scope") != "formal_numeric_sample_shard_not_g0"
        or any(int(plan.get(key, -1)) != identity[key] for key in identity)
        or int(plan.get("out_dim", -1)) != OUT_DIM
        or len(plan.get("tasks", [])) != identity["heads"] ** 2
        or receipt.get("schema") != "local5_erep_numeric_window_expected_v1"
        or receipt.get("status") != "PASS_NUMERIC_WINDOW_EXPECTED_NOT_G0"
        or receipt.get("formal_g0") != "DENY"
        or receipt.get("task_plan_sha256") != sha256(plan_path)
        or receipt.get("software_expected_sha256") != sha256(expected_path)
        or receipt.get("numpy_version") != np.__version__
    ):
        raise ValueError("软件 expected 合同或来源绑定失效")
    sources = receipt.get("source_bindings")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("软件 expected 源码闭包不完整")
    for binding in sources:
        source = Path(str(binding.get("file", "")))
        if not source.is_file() or binding.get("sha256") != sha256(source):
            raise ValueError("软件 expected 源码 SHA 失配")
    with np.load(expected_path, allow_pickle=False) as archive:
        if set(archive.files) != {"schema_version", "expected_acc32"}:
            raise ValueError("软件 expected NPZ member 集合不精确")
        version = archive["schema_version"]
        expected = archive["expected_acc32"]
        if (
            version.dtype != np.uint16
            or version.shape != (1,)
            or int(version[0]) != 1
            or expected.dtype != np.int32
            or expected.ndim != 1
        ):
            raise ValueError("软件 expected NPZ dtype/shape 不合法")
        expected = expected.copy()
    count = identity["heads"] * TOKENS * OUT_DIM
    if expected.size != count or int(receipt.get("expected_scalar_count", -1)) != count:
        raise ValueError("软件 expected 标量数不等于 H*450*32")
    return plan, expected


def _load_actual(
    directory: Path,
    task_plan: Path,
    expected_count: int,
    release_manifest: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    actual_path = directory / "actual.memh"
    receipt_path = directory / "actual_receipt.json"
    receipt = _read_json(receipt_path)
    receipt_matches(receipt, actual_path, task_plan)
    if receipt.get("simulator") != "verilator":
        raise ValueError("正式 numeric shard actual 必须来自 Verilator")
    validate_execution_binding(receipt, "verilator")
    if (
        receipt.get("formal_g0") != "DENY"
        or receipt.get("provenance_level") != "exact_argv_sealed_release"
        or receipt.get("release_manifest_sha256") != sha256(release_manifest)
        or int(receipt.get("actual_scalar_count", -1)) != expected_count
        or not receipt.get("filelist")
        or not receipt.get("vector_file_bindings")
    ):
        raise ValueError("DUT actual 计数或来源闭包不完整")
    raw_log = Path(str(receipt.get("raw_log", "")))
    if not raw_log.is_file() or receipt.get("raw_log_sha256") != sha256(raw_log):
        raise ValueError("DUT actual 日志来源绑定失效")
    for binding in receipt["filelist"]:
        source = Path(str(binding["file"]))
        if not source.is_file() or binding["sha256"] != sha256(source):
            raise ValueError("DUT RTL/TB filelist 来源绑定失效")
    for binding in receipt["vector_file_bindings"]:
        vector = Path(str(binding["path"]))
        if (
            not vector.is_file()
            or binding["sha256"] != sha256(vector)
            or int(binding["entries"])
            != len(vector.read_text(encoding="ascii").splitlines())
        ):
            raise ValueError("DUT vector 来源绑定失效")
    actual = np.asarray(parse_acc32(actual_path), dtype=np.int32)
    if actual.size != expected_count:
        raise ValueError("DUT actual 标量数不精确")
    return receipt, actual


def merge_window(
    directory: Path,
    sample: int,
    stage: int,
    block: int,
    heads: int,
    release_manifest: Path,
) -> dict[str, Any]:
    identity = {
        "sample": sample,
        "stage": stage,
        "block": block,
        "heads": heads,
    }
    plan, expected = _load_expected(directory, identity)
    identity["window"] = int(plan["window"])
    expected_receipt = _read_json(
        directory / "software_expected/software_expected_receipt.json"
    )
    if expected_receipt.get("identity") != identity:
        raise ValueError("软件 expected window 坐标不一致")
    actual_receipt, actual = _load_actual(
        directory,
        directory / "software_expected/task_plan.json",
        expected.size,
        release_manifest,
    )
    if actual_receipt.get("identity") != identity:
        raise ValueError("DUT actual stage/block/window 与 expected 不一致")
    bindings = {
        "task_plan": directory / "software_expected/task_plan.json",
        "software_expected": directory / "software_expected/software_expected.npz",
        "software_expected_receipt": directory
        / "software_expected/software_expected_receipt.json",
        "vector_manifest": directory / "vectors/manifest.json",
        "actual": directory / "actual.memh",
        "actual_receipt": directory / "actual_receipt.json",
        "raw_log": directory / "verilator.log",
        "run_argv": directory / "run_argv.json",
        "release_manifest": release_manifest,
    }
    _validate_complete(directory / "window_complete.json", identity, bindings)
    delta = actual.astype(np.int64) - expected.astype(np.int64)
    mismatch = int(np.count_nonzero(delta))
    if mismatch:
        first = int(np.flatnonzero(delta)[0])
        raise ValueError(
            f"numeric shard Acc32 不一致 index={first} "
            f"actual={int(actual[first])} expected={int(expected[first])}"
        )
    return {
        "identity": identity,
        "cycles": int(actual_receipt["cycles"]),
        "expected": expected,
        "actual": actual,
        "mismatch_count": mismatch,
        "max_abs_error": int(np.max(np.abs(delta), initial=0)),
        "window_complete_sha256": sha256(directory / "window_complete.json"),
    }


def write_single_window_miter(
    root: Path,
    out: Path,
    sample: int,
    stage: int,
    block: int,
    heads: int,
    release_manifest: Path,
) -> dict[str, Any]:
    row = merge_window(
        root / f"s{stage}_b{block}",
        sample,
        stage,
        block,
        heads,
        release_manifest,
    )
    expected = row.pop("expected")
    actual = row.pop("actual")
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "acc32_window_miter.npz"
    np.savez(
        archive_path,
        schema_version=np.asarray([1], dtype=np.uint16),
        expected_acc32=expected.astype(np.int32, copy=False),
        actual_acc32=actual.astype(np.int32, copy=False),
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        if (
            set(archive.files)
            != {"schema_version", "expected_acc32", "actual_acc32"}
            or archive["schema_version"].dtype != np.uint16
            or archive["schema_version"].shape != (1,)
            or int(archive["schema_version"][0]) != 1
            or archive["expected_acc32"].dtype != np.int32
            or archive["actual_acc32"].dtype != np.int32
            or not np.array_equal(archive["expected_acc32"], expected)
            or not np.array_equal(archive["actual_acc32"], actual)
        ):
            raise ValueError("单窗 Acc32 miter archive 落盘复核失败")
    report = {
        "schema": "local5_erep_numeric_window_miter_v1",
        "status": "PASS_NUMERIC_WINDOW_MITER_NOT_G0",
        "evidence": "[rtl]+[软件整数金参考]",
        "formal_g0": "DENY",
        **row,
        "scalar_count": int(expected.size),
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "release_manifest": str(release_manifest),
        "release_manifest_sha256": sha256(release_manifest),
        "boundary": "单个真实窗口；pre-bias/pre-BN/pre-requant/pre-residual Acc32",
    }
    report_path = out / "numeric_window_miter_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--window-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--single-stage", type=int)
    parser.add_argument("--single-block", type=int)
    parser.add_argument("--single-heads", type=int)
    args = parser.parse_args()
    if not 0 <= args.sample < 100:
        raise ValueError("formal sample 必须位于 0..99")
    root = args.window_root.resolve()
    out = args.output_dir.resolve()
    release_manifest = args.release_manifest.resolve()
    if not release_manifest.is_file():
        raise ValueError("numeric shard release manifest 不存在")
    out.mkdir(parents=True, exist_ok=True)
    single = (args.single_stage, args.single_block, args.single_heads)
    if any(value is not None for value in single):
        if any(value is None for value in single):
            raise ValueError("single-stage/block/heads 必须同时提供")
        if not any(
            (stage, block, heads)
            == (args.single_stage, args.single_block, args.single_heads)
            for stage, block, heads, _ in BLOCKS
        ):
            raise ValueError("single window 坐标不在冻结 12-block 拓扑")
        report = write_single_window_miter(
            root,
            out,
            args.sample,
            args.single_stage,
            args.single_block,
            args.single_heads,
            release_manifest,
        )
        print(json.dumps(report, ensure_ascii=False))
        return 0

    rows = []
    expected_parts = []
    actual_parts = []
    offsets = [0]
    for stage, block, heads, weight in BLOCKS:
        row = merge_window(
            root / f"s{stage}_b{block}",
            args.sample,
            stage,
            block,
            heads,
            release_manifest,
        )
        row["weight"] = weight
        rows.append(row)
        expected_parts.append(row.pop("expected"))
        actual_parts.append(row.pop("actual"))
        offsets.append(offsets[-1] + heads * TOKENS * OUT_DIM)

    expected = np.concatenate(expected_parts).astype(np.int32, copy=False)
    actual = np.concatenate(actual_parts).astype(np.int32, copy=False)
    metadata = [row["identity"] for row in rows]
    payload = {
        "schema_version": np.asarray([ARCHIVE_SCHEMA_VERSION], dtype=np.uint16),
        "window_sample": np.asarray([row["sample"] for row in metadata], dtype=np.uint16),
        "window_stage": np.asarray([row["stage"] for row in metadata], dtype=np.uint8),
        "window_block": np.asarray([row["block"] for row in metadata], dtype=np.uint8),
        "window_token": np.asarray([row["window"] for row in metadata], dtype=np.uint16),
        "window_weight": np.asarray([row["weight"] for row in rows], dtype=np.uint16),
        "window_heads": np.asarray([row["heads"] for row in metadata], dtype=np.uint8),
        "window_value_offsets": np.asarray(offsets, dtype=np.int64),
        "expected_acc32": expected,
        "actual_acc32": actual,
    }
    parsed = parse_miter_archive(payload, formal=False)
    if len(parsed) != len(BLOCKS) or expected.size != 1_987_200:
        raise ValueError("numeric sample shard 规模不等于冻结合同")
    archive_path = out / "acc32_miter_shard.npz"
    np.savez(archive_path, **payload)
    _validate_npz_container(
        archive_path, MITER_ARRAY_SPECS, "Acc32 miter shard archive"
    )
    with np.load(archive_path, allow_pickle=False) as persisted:
        persisted_rows = parse_miter_archive(
            {name: persisted[name] for name in persisted.files}, formal=False
        )
    if [row["acc32_miter_sha256"] for row in persisted_rows] != [
        row["acc32_miter_sha256"] for row in parsed
    ]:
        raise ValueError("落盘后的 Acc32 miter shard 与内存 payload 不一致")
    report_rows = []
    for row, parsed_row in zip(rows, parsed, strict=True):
        report_rows.append(
            {
                **row["identity"],
                "weight": row["weight"],
                "cycles": row["cycles"],
                "mismatch_count": row["mismatch_count"],
                "max_abs_error": row["max_abs_error"],
                "acc32_miter_sha256": parsed_row["acc32_miter_sha256"],
                "window_complete_sha256": row["window_complete_sha256"],
            }
        )
    report = {
        "schema": "local5_erep_numeric_sample_shard_v1",
        "status": "PASS_NUMERIC_SAMPLE_SHARD_NOT_G0",
        "evidence": "[rtl]+[软件整数金参考]",
        "formal_g0": "DENY",
        "sample": args.sample,
        "window_count": len(report_rows),
        "final_acc32_scalar_count": int(expected.size),
        "mismatch_count": 0,
        "max_abs_error": 0,
        "total_regression_cycles": sum(row["cycles"] for row in report_rows),
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "release_manifest": str(release_manifest),
        "release_manifest_sha256": sha256(release_manifest),
        "window_rows_sha256": canonical_json_sha(report_rows),
        "windows": report_rows,
        "boundary": [
            "只闭合一个 sample 的 12 个 attention-to-projection 窗口",
            "尚无正式 phase trace/ledger，因此不是 formal G0",
            "cycle 是验证 service 条件下的回归延迟，不是部署性能",
        ],
    }
    report_path = out / "numeric_shard_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
