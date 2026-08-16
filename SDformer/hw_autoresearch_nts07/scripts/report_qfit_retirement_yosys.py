#!/usr/bin/env python3
"""汇总 QFIT 关系退休控制面的开放综合代理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/qfit_retirement_yosys_20260730"
VARIANTS = {
    "fcsr": "FCSR 闭式逐源退休",
    "dynamic_frontier": "Dynamic Frontier 逐源计数",
    "nonblocking_stripe": "Nonblocking Stripe 双行上下文",
}


def summarize(name: str) -> dict[str, Any]:
    path = RESULTS / f"{name}_stat.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    top = data["modules"]["\\qfit_retirement_scheduler"]
    cells = top["num_cells_by_type"]
    return {
        "cells": int(top["num_cells"]),
        "wire_bits": int(top["num_wire_bits"]),
        "register_cells": sum(
            int(count)
            for cell_type, count in cells.items()
            if "DFF" in cell_type
        ),
        "mux_cells": int(cells.get("$_MUX_", 0)),
        "and_cells": int(cells.get("$_AND_", 0)),
        "or_cells": int(cells.get("$_OR_", 0)),
    }


def delta(value: int, baseline: int) -> float:
    return value / baseline - 1.0


def main() -> None:
    rows = {name: summarize(name) for name in VARIANTS}
    fcsr = rows["fcsr"]
    dynamic = rows["dynamic_frontier"]
    stripe = rows["nonblocking_stripe"]
    report = {
        "schema": "qfit_retirement_open_synthesis_proxy_v1",
        "evidence": "[开放综合代理]，非 DC/STA/SAIF",
        "parameters": {"height": 15, "width": 15, "time_planes": 2},
        "variants": rows,
        "comparisons": {
            "fcsr_cells_vs_dynamic": delta(
                fcsr["cells"], dynamic["cells"]
            ),
            "fcsr_registers_vs_dynamic": delta(
                fcsr["register_cells"], dynamic["register_cells"]
            ),
            "fcsr_cells_vs_stripe": delta(
                fcsr["cells"], stripe["cells"]
            ),
            "fcsr_registers_vs_stripe": delta(
                fcsr["register_cells"], stripe["register_cells"]
            ),
        },
        "cycle_evidence": {
            "source": "15x15 单实例饱和流 RTL 回归",
            "no_backpressure": {
                "fcsr": 241,
                "dynamic_frontier": 241,
                "nonblocking_stripe": 256,
            },
        },
    }
    (RESULTS / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# QFIT 关系退休控制面 RTL 与开放综合代理",
        "",
        "## 结论",
        "",
        "- FCSR 与 Dynamic Frontier 在单退休端口下具有相同饱和流周期，但 FCSR 不保存每源运行时引用计数。",
        "- Dynamic 基线已改为三行循环计数状态 `O(W)`；不再使用全窗口 `O(HW)` 计数器。",
        "- Nonblocking Stripe 能降低生产者局部停顿，却因整行粒度释放增加尾部排空；当前 `15x15` 端到端周期劣于逐源退休。",
        "- 因此 C2 的有效架构主张是“固定 stencil 的闭式逐源生命周期”，不是“双行缓冲更快”。",
        "",
        "## 同约束结构结果",
        "",
        "| 变体 | cells | wire bits | register cells | mux | and | or | 无反压周期 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cycles = report["cycle_evidence"]["no_backpressure"]
    for name, label in VARIANTS.items():
        row = rows[name]
        lines.append(
            f"| {label} | {row['cells']} | {row['wire_bits']} | "
            f"{row['register_cells']} | {row['mux_cells']} | "
            f"{row['and_cells']} | {row['or_cells']} | "
            f"{cycles[name]} |"
        )
    comparisons = report["comparisons"]
    lines.extend(
        [
            "",
            "## 关键差分",
            "",
            f"- FCSR vs Dynamic：cells `{comparisons['fcsr_cells_vs_dynamic']:+.2%}`，register cells `{comparisons['fcsr_registers_vs_dynamic']:+.2%}`；",
            f"- FCSR vs Stripe：cells `{comparisons['fcsr_cells_vs_stripe']:+.2%}`，register cells `{comparisons['fcsr_registers_vs_stripe']:+.2%}`；",
            "- FCSR 与 Dynamic 的逐事件退休序列完全一致；三种模式均对每个 source 恰好退休一次。",
            "",
            "## 证据边界",
            "",
            "- cell 数来自同脚本 Yosys generic mapping，仅能支持结构趋势；",
            "- 周期来自规则 raster 输入和一个退休端口，不代表完整 Local5 tile；",
            "- 本报告只比较退休控制面；共同 gate/K 行环与 atomic snapshot 见 `results/qfit_relation_transpose_yosys_20260730/report.md`；",
            "- Dynamic 的五候选变量索引会在 generic mapping 中展开大选择网络，因此 cell 差分不能直接当作物理面积收益；",
            "- 随机反压已通过，但跨样本 p95/p99 需等待 fullres trace；",
            "- 目标工艺 PPA、SRAM macro 和 SAIF 尚未完成。",
            "",
        ]
    )
    (RESULTS / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
