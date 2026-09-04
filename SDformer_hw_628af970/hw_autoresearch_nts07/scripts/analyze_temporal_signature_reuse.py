#!/usr/bin/env python3
"""从旧profile直方图恢复K0/K1交集并评估精确投影复用。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
CASES = {
    "H67": REPO / "neuron_experiments/H9_bipolar_self_attention/results/h67_ep19_true_ttb_profile100_20260712/nts11_hardware_p0_profile.json",
    "H68": REPO / "neuron_experiments/H9_bipolar_self_attention/results/h68_ep19_true_ttb_profile100_20260713/nts11_hardware_p0_profile.json",
}


def histogram_sum(histogram: list[int]) -> int:
    return sum(index * int(count) for index, count in enumerate(histogram))


def recover_row(row: dict[str, Any]) -> dict[str, int | float]:
    if "k_count_histogram" in row and "motion_histogram" in row:
        baseline = histogram_sum(row["k_count_histogram"])
        xor = histogram_sum(row["motion_histogram"])
        method = "direct_histogram"
    else:
        required = (
            "k_active_density", "batch_windows", "num_heads", "tokens", "head_dim",
            "k_temporal_toggle_elements",
        )
        if any(field not in row for field in required):
            raise KeyError("旧schema恢复缺少K密度、形状或精确toggle字段")
        elements = (
            int(row["batch_windows"]) * int(row["num_heads"]) *
            int(row["tokens"]) * int(row["head_dim"])
        )
        baseline = round(float(row["k_active_density"]) * elements)
        xor = int(row["k_temporal_toggle_elements"])
        method = "density_shape_plus_exact_xor"
    difference = baseline - xor
    if difference < 0 or difference % 2:
        raise ValueError("K时间交集恢复失败：baseline-xor必须为非负偶数")
    intersection = difference // 2
    union = baseline - intersection
    if baseline != union + intersection:
        raise ValueError("K时间读取守恒失败")
    return {
        "baseline_weight_row_reads": baseline,
        "union_weight_row_reads": union,
        "intersection_reused_reads": intersection,
        "exact_reuse_ratio": intersection / baseline if baseline else 0.0,
        "recovery_method": method,
    }


def analyze_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["summary"]["h60_records"]
    grouped: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"baseline": 0, "union": 0, "intersection": 0}
    )
    total = {"baseline": 0, "union": 0, "intersection": 0}
    recovery_methods: set[str] = set()
    for row in rows:
        recovered = recover_row(row)
        key = (int(row["stage"]), int(row["block"]))
        recovery_methods.add(str(recovered["recovery_method"]))
        for short, long in (
            ("baseline", "baseline_weight_row_reads"),
            ("union", "union_weight_row_reads"),
            ("intersection", "intersection_reused_reads"),
        ):
            grouped[key][short] += int(recovered[long])
            total[short] += int(recovered[long])
    blocks = []
    for (stage, block), values in sorted(grouped.items()):
        baseline = values["baseline"]
        blocks.append({
            "stage": stage,
            "block": block,
            "baseline_weight_row_reads": baseline,
            "union_weight_row_reads": values["union"],
            "intersection_reused_reads": values["intersection"],
            "exact_reuse_ratio": values["intersection"] / baseline if baseline else 0.0,
        })
    return {
        "profile": str(path),
        "samples": int(data["samples"]),
        "records": len(rows),
        "baseline_weight_row_reads": total["baseline"],
        "union_weight_row_reads": total["union"],
        "intersection_reused_reads": total["intersection"],
        "exact_reuse_ratio": total["intersection"] / total["baseline"] if total["baseline"] else 0.0,
        "recovery_methods": sorted(recovery_methods),
        "blocks": blocks,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# H67/H68时间签名精确投影复用统计",
        "",
        "交集由恒等式`|K0∩K1|=(|K0|+|K1|-|K0 XOR K1|)/2`从旧profile100逐记录恢复。旧schema用`round(K密度×张量元素数)`恢复K事件总数，并与精确XOR计数做非负偶数硬校验；只有stage归属，没有block归属。它只给出权重行读取减少上界，不含分类网络、队列和partial-sum开销。",
        "",
        "## 全局结果",
        "",
        "| 模型 | 样本 | 原始读取 | union读取 | 可复用交集 | 精确读取下降 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in result["models"].items():
        lines.append(
            f"| {model} | {row['samples']} | {row['baseline_weight_row_reads']:,} | "
            f"{row['union_weight_row_reads']:,} | {row['intersection_reused_reads']:,} | "
            f"{row['exact_reuse_ratio']:.2%} |"
        )
    lines += ["", "## 逐stage", "", "| 模型 | stage | 原始读取 | 精确读取下降 |", "|---|---|---:|---:|"]
    for model, row in result["models"].items():
        for block in row["blocks"]:
            lines.append(
                f"| {model} | S{block['stage']} | "
                f"{block['baseline_weight_row_reads']:,} | {block['exact_reuse_ratio']:.2%} |"
            )
    lines += [
        "",
        "## 决策口径",
        "",
        "- `<10%`：不实现时间签名复用硬件，只保留普通active-lane扫描；",
        "- `10%~20%`：只实现三bitmap顺序扫描，与prefix基线做DC；",
        "- `>=20%`且多数block稳定：允许实现三队列压紧；蝶形网络仍需相对prefix净EDP改善至少10%才晋级；",
        "- 任何百分比都不能直接写成系统能耗下降，必须扣除分类、索引、partial-sum和不均衡成本。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = {"models": {model: analyze_profile(path) for model, path in CASES.items()}}
    json_path = ROOT / "results/h67_h68_temporal_signature_reuse.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, json_path.with_suffix(".md"))
    print(json_path.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
