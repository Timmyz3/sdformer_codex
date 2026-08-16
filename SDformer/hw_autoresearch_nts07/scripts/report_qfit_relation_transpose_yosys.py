#!/usr/bin/env python3
"""汇总 QFIT 在线关系转置叶的宏状态与打平结构代理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/qfit_relation_transpose_yosys_20260730"
TOP = "\\qfit_relation_transpose_leaf"
VARIANTS = {
    "fcsr": "FCSR-RX",
    "dynamic_frontier": "Dynamic Frontier-RX",
    "stripe3": "Safe Stripe-3-RX",
    "stripe4": "Early-fill Stripe-4-RX",
}


def top_stat(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["modules"][TOP]


def summarize(name: str) -> dict[str, int]:
    memory = top_stat(RESULTS / f"{name}_memory_stat.json")
    flat = top_stat(RESULTS / f"{name}_flat_stat.json")
    memory_types = memory["num_cells_by_type"]
    flat_types = flat["num_cells_by_type"]
    ring_rows = 4 if name == "stripe4" else 3
    return {
        "memory_count": int(memory_types.get("$mem_v2", 0)),
        "memory_bits": (32 + 5 * (9 + 1)) * ring_rows * 15,
        "flat_cells": int(flat["num_cells"]),
        "flat_register_cells": sum(
            int(count)
            for cell_type, count in flat_types.items()
            if "DFF" in cell_type
        ),
        "flat_mux_cells": int(flat_types.get("$_MUX_", 0)),
    }


def ratio(value: int, baseline: int) -> float:
    return value / baseline - 1.0


def main() -> None:
    rows = {name: summarize(name) for name in VARIANTS}
    fcsr = rows["fcsr"]
    dynamic = rows["dynamic_frontier"]
    stripe3 = rows["stripe3"]
    stripe4 = rows["stripe4"]
    comparisons = {
        "fcsr_flat_cells_vs_dynamic": ratio(
            fcsr["flat_cells"], dynamic["flat_cells"]
        ),
        "fcsr_flat_regs_vs_dynamic": ratio(
            fcsr["flat_register_cells"],
            dynamic["flat_register_cells"],
        ),
        "fcsr_flat_cells_vs_stripe3": ratio(
            fcsr["flat_cells"], stripe3["flat_cells"]
        ),
        "fcsr_flat_regs_vs_stripe3": ratio(
            fcsr["flat_register_cells"],
            stripe3["flat_register_cells"],
        ),
        "fcsr_flat_cells_vs_stripe4": ratio(
            fcsr["flat_cells"], stripe4["flat_cells"]
        ),
        "fcsr_flat_regs_vs_stripe4": ratio(
            fcsr["flat_register_cells"],
            stripe4["flat_register_cells"],
        ),
    }
    report = {
        "schema": "qfit_relation_transpose_open_synthesis_proxy_v1",
        "evidence": "[RTL/开放综合代理]，非 DC/STA/SAIF",
        "parameters": {
            "height": 15,
            "width": 15,
            "time_planes": 2,
            "k_width": 32,
            "gate_width": 9,
        },
        "variants": rows,
        "comparisons": comparisons,
        "functional": {
            "candidates": 4,
            "simulators": ["Icarus", "Verilator"],
            "descriptors_per_mode": 40,
            "k_gate_mask_mismatch": 0,
            "long_backpressure_cycles": 20,
            "same_address_read_first": "PASS",
            "random_backpressure": "PASS",
        },
        "cycle_evidence": {
            "source": "15x15 同步六bank relation leaf，无输出反压",
            "fcsr": 244,
            "dynamic_frontier": 244,
            "stripe3": 451,
            "stripe4": 265,
        },
    }
    (RESULTS / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# QFIT 有界生命周期在线关系转置 RTL 结果",
        "",
        "## 结论",
        "",
        "- 四候选共用五方向 gate/K bank 接口和 source descriptor 合同；Stripe-3/4 均以 row-distance watermark 保证长反压安全。",
        "- FCSR-RX 以静态 frontier 直接生成逐源退休，不维护逐 source 引用计数，也不等待整行结束。",
        "- 该结果把 C2 从控制 FSM 推进为可传递真实 K/gate/mask 的在线关系转置叶，但仍不是完整 projection tile。",
        "",
        "## 同约束结构结果",
        "",
        "| 变体 | `$mem_v2` 宏数 | 解析 memory bits | flat cells | flat register cells | flat mux |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in VARIANTS.items():
        row = rows[name]
        lines.append(
            f"| {label} | {row['memory_count']} | "
            f"{row['memory_bits']} | {row['flat_cells']} | "
            f"{row['flat_register_cells']} | "
            f"{row['flat_mux_cells']} |"
        )
    lines.extend(
        [
            "",
            "## 同步数据面周期",
            "",
            "| FCSR-RX | Dynamic-RX | Stripe-3-RX | Stripe-4-RX |",
            "|---:|---:|---:|---:|",
            "| 244 | 244 | 451 | 265 |",
            "",
            "- 口径为 `15x15`、一拍同步六 bank、两项 descriptor FIFO、无输出反压；",
            "- FCSR 与 Dynamic 同周期；FCSR 相对 Stripe-3/4 分别减少 207/21 拍。",
            "",
            "## 关键差分",
            "",
            f"- FCSR-RX vs Dynamic-RX：flat cells `{comparisons['fcsr_flat_cells_vs_dynamic']:+.2%}`，flat registers `{comparisons['fcsr_flat_regs_vs_dynamic']:+.2%}`；",
            f"- FCSR-RX vs Stripe-3-RX：flat cells `{comparisons['fcsr_flat_cells_vs_stripe3']:+.2%}`，flat registers `{comparisons['fcsr_flat_regs_vs_stripe3']:+.2%}`；",
            f"- FCSR-RX vs Stripe-4-RX：flat cells `{comparisons['fcsr_flat_cells_vs_stripe4']:+.2%}`，flat registers `{comparisons['fcsr_flat_regs_vs_stripe4']:+.2%}`。",
            "",
            "## 功能合同",
            "",
            "- source K 与 source id/坐标一致；",
            "- destination-major 五方向 gate 按逆方向地址转置为 source-major descriptor；",
            "- 几何边界与运行时 partial-valid 均随 role bank 在线转置，不把 invalid 当普通 gate=0；",
            "- descriptor 反压时冻结退休和输入写入；",
            "- `plane_active + HxW` 接收计数阻止平面中途误切；FCSR/Dynamic/Stripe-3 使用三行，Stripe-4 使用四行。",
            "",
            "## 证据边界",
            "",
            "- `$mem_v2` 来自 Yosys memory collect；gate bank 每项含 `9-bit gate + 1-bit valid`。三行候选为 `(32+5x10)x3x15=3690 bit`，Stripe-4 为 4920 bit；",
            "- flat cells 把 memory 映射为寄存器/逻辑，并以 `setundef -zero` 处理未复位 SRAM 的初态 X，只用于同流程结构趋势；",
            "- `check -assert` 在 memory collect 宏阶段通过；打平后不对未复位 SRAM 的未写地址执行该检查；",
            "- Dynamic 的五候选动态索引形成多端口选择网络，其 generic cell 数不能直接当作公平 DC 面积；",
            "- Dynamic 已缩为三行循环计数状态 `O(W)`，但其五候选并行访问仍需在目标 SRAM/寄存器映射下重做公平 PPA；",
            "- 当前实现一拍同步读、单项 read-inflight 和两项 atomic descriptor FIFO；尚无 projection consumer 的目标 PPA 或 post-G0 trace；",
            "- 同址行为冻结为 read-first，并已通过定向 TB；具体工艺 SRAM macro 仍需匹配该模式或加冲突 stall；",
            "- 论文主表仍需相同 SRAM macro、相同 SDC 下的 DC/STA/SAIF。",
            "",
        ]
    )
    (RESULTS / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
