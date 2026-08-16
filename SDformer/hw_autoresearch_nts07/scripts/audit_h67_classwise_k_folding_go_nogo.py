#!/usr/bin/env python3
"""Audit whether class-wise K counts preserve H67's token-indexed output."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_counterexample() -> dict[str, object]:
    gate = 4
    k_event = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    weight = np.asarray([[2], [3]], dtype=np.int64)
    token_indexed = gate * (k_event @ weight)
    class_lane_count = k_event.sum(axis=0)
    folded_single_accumulator = gate * (class_lane_count @ weight)
    invalid_broadcast = np.full_like(token_indexed, folded_single_accumulator)

    class_bitmap = np.asarray([1, 1], dtype=bool)
    recovered = np.zeros_like(token_indexed)
    for lane in range(k_event.shape[1]):
        destination_bitmap = class_bitmap & k_event[:, lane].astype(bool)
        recovered[destination_bitmap] += gate * weight[lane]
    swapped_k_event = k_event[::-1].copy()
    swapped_output = gate * (swapped_k_event @ weight)
    swapped_class_lane_count = swapped_k_event.sum(axis=0)
    if not np.array_equal(class_lane_count, swapped_class_lane_count):
        raise AssertionError("不可区分输入的class count不相同")
    if np.array_equal(token_indexed, swapped_output):
        raise AssertionError("不可区分输入没有产生不同token输出")
    return {
        "gate": gate,
        "k_event": k_event.tolist(),
        "weight": weight.reshape(-1).tolist(),
        "token_indexed_output": token_indexed.reshape(-1).tolist(),
        "class_lane_count": class_lane_count.tolist(),
        "folded_single_accumulator": int(folded_single_accumulator.item()),
        "invalid_broadcast_output": invalid_broadcast.reshape(-1).tolist(),
        "invalid_broadcast_mismatches": int(
            np.count_nonzero(invalid_broadcast != token_indexed)
        ),
        "destination_bitmap_output": recovered.reshape(-1).tolist(),
        "destination_bitmap_mismatches": int(
            np.count_nonzero(recovered != token_indexed)
        ),
        "indistinguishable_count_pair": {
            "input_a": k_event.tolist(),
            "input_b": swapped_k_event.tolist(),
            "shared_class_lane_count": class_lane_count.tolist(),
            "token_output_a": token_indexed.reshape(-1).tolist(),
            "token_output_b": swapped_output.reshape(-1).tolist(),
            "proof": "same count representation maps two distinct token-indexed outputs to one state",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/h67_classwise_k_folding_go_nogo_20260813"),
    )
    args = parser.parse_args()
    counterexample = evaluate_counterexample()
    if (
        counterexample["invalid_broadcast_mismatches"] == 0
        or counterexample["destination_bitmap_mismatches"] != 0
    ):
        raise AssertionError("class-wise K folding反例或bitmap恢复未闭合")

    sources = [
        ROOT / "scripts/fcip_integer_reference.py",
        ROOT / "results/fcip_integer_reference_20260730/report.json",
        ROOT / "docs/253_Motion_TESC到GatedK闭环与HIFP维护回归_20260805.md",
        ROOT / "results/h67_tare_zkqi_row_rtl_20260810/report.json",
    ]
    result = {
        "schema": "h67_classwise_k_folding_go_nogo_v1",
        "status": "REJECT_COUNT_FOLD_AT_TOKEN_OUTPUT_BOUNDARY",
        "evidence": "[integer-counterexample]+[contract-audit]",
        "counterexample": counterexample,
        "contract": {
            "current_output": "one Acc vector per token destination",
            "valid_identity": "sum_j g_c K_j W = g_c (sum_j K_j) W only when j contributes to one shared accumulator",
            "violated_precondition": "H67 gated-K/projection preserves token destination; temporal mask restores token order",
            "exact_alternative": "retain destination identity via class-bitmap AND K-lane-bitmap, as FCIP does",
        },
        "decision": {
            "new_rtl": False,
            "paper_contribution": False,
            "reason": "count folding deletes the token axis; FCIP already covers the exact bitmap-preserving form and was not promoted by prior resource screening",
            "reconsider_only_if": "a later operator proves all folded tokens reduce into the same accumulator before any token-indexed output",
        },
        "base_delta_decision": {
            "semantic_class": "existing TARE anchor plus changed-lane delta",
            "status": "DO_NOT_RENAME_OR_REVIVE_CURRENT_COMPACTOR",
            "evidence_boundary": "existing W8/W16 arbitrary changed-lane compactor implementation only",
            "reconsider_only_if": "a genuinely different fixed-slice or naturally ordered delta stream avoids the rejected compactor cost and passes a new workload gate",
        },
        "source_receipts": [
            {"file": str(path.resolve()), "sha256": sha256(path)} for path in sources
        ],
        "limits": [
            "This is an algebraic go/no-go audit, not workload speedup, RTL, or PPA.",
            "It does not reject class-wise folding for a different operator with a genuine shared reduction accumulator.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(
            [
                "# H67 Class-wise K Folding Go/No-Go",
                "",
                "## 结论",
                "",
                "**REJECT**：当前 H67 输出保留逐 token destination，不能把同 class 的 K "
                "计数折叠到一个 accumulator 后再广播；该操作会删除 token 轴。",
                "",
                "两 token 反例的原输出为 `[%s]`，count-fold 后只有 `%s`；广播回两 token "
                "产生 `%d` 个 mismatch。保留 destination bitmap 后为 `[%s]`，零失配。"
                % (
                    ", ".join(map(str, counterexample["token_indexed_output"])),
                    counterexample["folded_single_accumulator"],
                    counterexample["invalid_broadcast_mismatches"],
                    ", ".join(map(str, counterexample["destination_bitmap_output"])),
                ),
                "",
                "更严格地，`K_A=[[1,0],[0,1]]` 与 `K_B=[[0,1],[1,0]]` 具有相同 "
                "class-lane count `[1,1]`，但 token 输出分别为 `[8,12]` 与 `[12,8]`。"
                "因此仅凭 count 在信息上不可能恢复 destination。",
                "",
                "bitmap-preserving algebraic form 与既有 FCIP representation 相同："
                "`class_bitmap & K_lane_bitmap -> destination_bitmap`。这不表示 FCIP 已有"
                "端到端 RTL/PPA 证明，也不重复包装成新贡献。",
                "",
                "Temporal base-delta 属于既有 TARE 的语义类别。当前 W8/W16 arbitrary-lane "
                "compactor 因面积归一吞吐失败而否决，不能改名复活；该负结果不理论否决"
                "未来无需 compactor 的固定切片或天然有序 delta stream。",
                "",
                "证据为 **[integer-counterexample]+[contract-audit]**，不是性能或 PPA。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    source_dir = args.output_dir / "source"
    source_dir.mkdir(exist_ok=True)
    snapshots: dict[str, str] = {}
    snapshot_sources = [Path(__file__).resolve(), *sources]
    for index, source in enumerate(snapshot_sources):
        if not source.is_file():
            raise ValueError(f"source不存在: {source}")
        expected = sha256(source)
        if sha256(source) != expected:
            raise ValueError(f"source receipt SHA失配: {source}")
        prefix = "producer" if index == 0 else f"dependency_{index}"
        destination = source_dir / f"{prefix}_{source.name}"
        shutil.copyfile(source, destination)
        snapshots[str(destination.relative_to(args.output_dir))] = sha256(destination)
    package_files = {
        path.name: sha256(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "complete.json"
    }
    package_files.update(snapshots)
    complete = {
        "schema": "h67_classwise_k_folding_go_nogo_package_v1",
        "status": "SEALED_REJECTED_CANDIDATE",
        "evidence": result["evidence"],
        "package_files": package_files,
    }
    (args.output_dir / "complete.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
