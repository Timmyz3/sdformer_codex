#!/usr/bin/env python3
"""只读合并 Local5 canary 的独立软件 expected 与 DUT actual。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TOKENS = 450
OUT_DIM = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_actual(path: Path, task_count: int) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    expected_count = task_count * TOKENS * OUT_DIM
    if len(lines) != expected_count or any(
        len(line) != 8 or any(ch not in "0123456789abcdefABCDEF" for ch in line)
        for line in lines
    ):
        raise ValueError("actual Acc32文本shape/编码不合法")
    unsigned = np.fromiter((int(line, 16) for line in lines), dtype=np.uint32)
    return unsigned.view(np.int32).reshape(task_count, TOKENS, OUT_DIM)


def merge_actual(tasks: list[dict[str, int]], actual: np.ndarray, heads: int) -> np.ndarray:
    merged = np.zeros((heads, TOKENS, OUT_DIM), dtype=np.int64)
    seen: set[tuple[int, int]] = set()
    for index, task in enumerate(tasks):
        tile = task["output_tile"]
        group = task["input_group_index"]
        identity = (group, tile)
        if identity in seen or not 0 <= tile < heads:
            raise ValueError("task plan重复或output tile越界")
        seen.add(identity)
        merged[tile] += actual[index].astype(np.int64)
    if len(seen) != heads * heads:
        raise ValueError("canary不是完整HxH任务集合")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--expected-receipt", type=Path, required=True)
    parser.add_argument("--actual", type=Path, action="append", required=True)
    parser.add_argument("--actual-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.actual) != 2 or len(args.actual_receipt) != 2:
        raise ValueError("必须恰好提供Icarus与Verilator两份actual/receipt")
    plan = json.loads(args.task_plan.read_text(encoding="utf-8"))
    expected_receipt = json.loads(args.expected_receipt.read_text(encoding="utf-8"))
    tasks = plan.get("tasks") or []
    heads = int(plan.get("heads", 0))
    if (
        plan.get("schema") != "local5_projection_task_plan_v1"
        or expected_receipt.get("task_plan_sha256") != sha256(args.task_plan)
        or expected_receipt.get("software_expected_sha256") != sha256(args.expected)
    ):
        raise ValueError("软件expected来源绑定失败")
    with np.load(args.expected, allow_pickle=False) as payload:
        if set(payload.files) != {"schema_version", "expected_acc32"}:
            raise ValueError("software expected成员集合不合法")
        if payload["schema_version"].dtype != np.uint16 or not np.array_equal(
            payload["schema_version"], np.asarray([1], dtype=np.uint16)
        ):
            raise ValueError("software expected schema不合法")
        expected = payload["expected_acc32"]
        if expected.dtype != np.int32 or expected.shape != (heads * TOKENS * OUT_DIM,):
            raise ValueError("software expected shape/dtype不合法")
        expected = expected.reshape(heads, TOKENS, OUT_DIM).astype(np.int64)
    simulator_results = []
    reference_actual = None
    observed_simulators: set[str] = set()
    for actual_path, receipt_path in zip(
        args.actual, args.actual_receipt, strict=True
    ):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != "local5_erep_formal_canary_rtl_actual_v1"
            or receipt.get("status") != "PASS_CANARY_NOT_G0"
            or receipt.get("task_plan_sha256") != sha256(args.task_plan)
            or receipt.get("actual_acc32_sha256") != sha256(actual_path)
            or receipt.get("actual_scalar_count")
            != len(tasks) * TOKENS * OUT_DIM
            or not receipt.get("dut_file_bindings")
            or not receipt.get("vector_artifact_bindings")
        ):
            raise ValueError("DUT actual来源绑定失败")
        simulator = str(receipt.get("simulator", ""))
        if simulator in observed_simulators:
            raise ValueError("DUT actual simulator重复")
        observed_simulators.add(simulator)
        raw_log = Path(str(receipt.get("raw_log", "")))
        if not raw_log.is_file() or receipt.get("raw_log_sha256") != sha256(raw_log):
            raise ValueError("DUT raw log来源绑定失败")
        for binding in receipt["dut_file_bindings"]:
            source = Path(str(binding["path"]))
            if not source.is_file() or binding["sha256"] != sha256(source):
                raise ValueError("DUT filelist来源绑定失败")
        for binding in receipt["vector_artifact_bindings"]:
            vector = Path(str(binding["path"]))
            if (
                not vector.is_file()
                or binding["sha256"] != sha256(vector)
                or int(binding["entries"])
                != len(vector.read_text(encoding="ascii").splitlines())
            ):
                raise ValueError("DUT vector artifact来源绑定失败")
        raw_actual = read_actual(actual_path, len(tasks))
        if reference_actual is not None and not np.array_equal(raw_actual, reference_actual):
            raise ValueError("不同模拟器DUT actual不一致")
        reference_actual = raw_actual.copy()
        merged = merge_actual(tasks, raw_actual, heads)
        mismatch = int(np.count_nonzero(merged != expected))
        max_abs = int(np.max(np.abs(merged - expected)))
        simulator_results.append(
            {
                "simulator": simulator,
                "actual_acc32_sha256": sha256(actual_path),
                "actual_receipt_sha256": sha256(receipt_path),
                "mismatch": mismatch,
                "max_abs_error": max_abs,
                "total_cycles": receipt["total_cycles"],
            }
        )
    if observed_simulators != {"icarus", "verilator"}:
        raise ValueError("simulator集合必须唯一等于{icarus, verilator}")
    total_mismatch = sum(row["mismatch"] for row in simulator_results)
    report = {
        "schema": "local5_erep_formal_canary_readonly_merge_v1",
        "status": "PASS_CANARY_NOT_G0" if total_mismatch == 0 else "FAIL",
        "evidence": "[rtl]+[软件整数金参考]",
        "scope": "one formal stage0 joint window; source-isolated canary, not formal G0",
        "coordinate": {
            key: plan[key] for key in ("sample", "stage", "block", "window")
        },
        "heads": heads,
        "hxh_tasks": len(tasks),
        "final_acc32_scalars_per_simulator": heads * TOKENS * OUT_DIM,
        "simulators": simulator_results,
        "cross_simulator_exact": reference_actual is not None,
        "total_mismatch": total_mismatch,
        "task_plan_sha256": sha256(args.task_plan),
        "software_expected_sha256": sha256(args.expected),
        "merge_script_sha256": sha256(Path(__file__).resolve()),
        "formal_g0": "DENY",
        "remaining": [
            "尚未生成1200-window/13800-head正式phase与Acc32 archive",
            "canary使用单head投影DUT逐任务回放后只读跨head合并，不替代集成cross-head DUT",
            "尚未生成admission_receipt，禁止EREP candidate RTL",
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if total_mismatch:
        raise SystemExit("Local5 formal canary mismatch")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
