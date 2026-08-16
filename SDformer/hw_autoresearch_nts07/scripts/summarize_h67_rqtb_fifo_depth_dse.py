#!/usr/bin/env python3
"""汇总 Motion RQTB 双-slot FIFO 深度的真实 T450 RTL DSE。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PASS_RE = re.compile(
    r"PASS H67 RQTB 2S physical flow rows=(?P<rows>\d+) checked=(?P<checked>\d+) "
    r"fixed_cycles=(?P<fixed_cycles>\d+) rqtb_cycles=(?P<rqtb_cycles>\d+) "
    r"fixed_slots=(?P<fixed_slots>\d+) rqtb_slots=(?P<rqtb_slots>\d+) "
    r"fixed_exp=(?P<fixed_exp>\d+) rqtb_exp=(?P<rqtb_exp>\d+) "
    r"acc32_mismatch=(?P<acc32_mismatch>\d+)"
)
COVER_RE = re.compile(
    r"RQTB_2S_COVER cross_pair=(?P<cross_pair>\d+) "
    r"same_class=(?P<same_class>\d+) double_active=(?P<double_active>\d+) "
    r"fifo_both=(?P<fifo_both>\d+) dual_k=(?P<dual_k>\d+)"
)
ROW_RE = re.compile(
    r"RQTB_ROW .*?fixed_fifo_max=(?P<fixed_fifo_max>\d+) "
    r"rqtb_fifo_max=(?P<rqtb_fifo_max>\d+)"
)


def parse_log(path: Path, depth: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pass_matches = list(PASS_RE.finditer(text))
    cover_matches = list(COVER_RE.finditer(text))
    rows = list(ROW_RE.finditer(text))
    if len(pass_matches) != 1 or len(cover_matches) != 1:
        raise ValueError(f"depth={depth} 日志缺少唯一 PASS/coverage receipt")
    result = {key: int(value) for key, value in pass_matches[0].groupdict().items()}
    coverage = {key: int(value) for key, value in cover_matches[0].groupdict().items()}
    if result["rows"] != len(rows):
        raise ValueError(f"depth={depth} row 计数不一致")
    if result["acc32_mismatch"] != 0 or min(coverage.values()) <= 0:
        raise ValueError(f"depth={depth} 未通过 exact/coverage 门槛")
    return {
        "depth": depth,
        "slot_storage_bits": depth * 16,
        **result,
        "fixed_fifo_max": max(int(row.group("fixed_fifo_max")) for row in rows),
        "rqtb_fifo_max": max(int(row.group("rqtb_fifo_max")) for row in rows),
        "coverage": coverage,
    }


def build_report(log_dir: Path, depths: list[int]) -> dict[str, Any]:
    points = [parse_log(log_dir / f"depth_{depth}.log", depth) for depth in depths]
    baseline = next((point for point in points if point["depth"] == 32), None)
    if baseline is None:
        raise ValueError("DSE 必须包含 depth=32 基线")
    for point in points:
        point["same_depth_speedup"] = point["fixed_cycles"] / point["rqtb_cycles"]
        point["rqtb_cycle_change_vs_depth32"] = (
            point["rqtb_cycles"] / baseline["rqtb_cycles"] - 1.0
        )
        point["fixed32_to_rqtb_speedup"] = (
            baseline["fixed_cycles"] / point["rqtb_cycles"]
        )
        point["storage_reduction_vs_depth32"] = 1.0 - point["depth"] / 32.0

    admissible = [
        point for point in points if point["rqtb_cycle_change_vs_depth32"] <= 0.01
    ]
    selected = min(admissible, key=lambda point: point["depth"])
    return {
        "schema": "h67_rqtb_fifo_depth_dse_v1",
        "status": "PASS",
        "evidence_level": "[rtl]",
        "scope": "H67 fullres epoch30 sample0/window0 all12，双-slot真实T450回放",
        "points": points,
        "selection": {
            "rule": "选择相对RQTB depth32周期退化不超过1%的最小深度",
            "depth": selected["depth"],
            "slot_storage_bits": selected["slot_storage_bits"],
            "rqtb_cycle_change_vs_depth32": selected["rqtb_cycle_change_vs_depth32"],
            "storage_reduction_vs_depth32": selected["storage_reduction_vs_depth32"],
            "fixed32_to_selected_rqtb_speedup": selected["fixed32_to_rqtb_speedup"],
        },
        "claim_boundary": [
            "该DSE只证明FIFO容量与RTL周期的关系，不包含SRAM宏面积、功耗或多上下文收益。",
            "缩小FIFO不是独立创新；只有把释放容量复用于exact跨窗口重叠并得到同约束收益后，才形成新架构机制。",
            "当前只有sample0/window0，选择结果必须用多样本trace复验。",
        ],
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Motion RQTB双-slot FIFO深度DSE",
        "",
        "## 结论",
        "",
        f"- 状态：**{result['status']}**；证据等级：**{result['evidence_level']}**。",
        f"- 1%周期门槛选择深度：`{result['selection']['depth']}`；slot存储为`{result['selection']['slot_storage_bits']} bit`。",
        f"- 相对depth32的RQTB周期变化：{result['selection']['rqtb_cycle_change_vs_depth32']:+.2%}；FIFO容量变化：-{result['selection']['storage_reduction_vs_depth32']:.2%}。",
        f"- Fixed32到所选RQTB的周期加速：{result['selection']['fixed32_to_selected_rqtb_speedup']:.3f}x。",
        "",
        "## 同深度公平结果",
        "",
        "| depth | FIFO bit | Fixed周期 | RQTB周期 | 同深度加速 | RQTB相对depth32 | Fixed32到RQTB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in result["points"]:
        lines.append(
            f"| {point['depth']} | {point['slot_storage_bits']} | {point['fixed_cycles']} | "
            f"{point['rqtb_cycles']} | {point['same_depth_speedup']:.3f}x | "
            f"{point['rqtb_cycle_change_vs_depth32']:+.2%} | "
            f"{point['fixed32_to_rqtb_speedup']:.3f}x |"
        )
    lines += ["", "## 边界", ""]
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    args = parser.parse_args()
    result = build_report(args.log_dir, args.depths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args.output_dir / "report.md", result)
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
