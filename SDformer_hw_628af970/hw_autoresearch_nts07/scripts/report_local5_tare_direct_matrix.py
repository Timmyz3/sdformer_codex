#!/usr/bin/env python3
"""汇总 Local5 TARE/Direct 同顶层周期与开放映射矩阵。"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results/local5_tare_direct_fullflow_20260729.log"
SYNTH = ROOT / "results/local5_tare_yosys_20260729"
OUT = ROOT / "results/local5_tare_direct_arch_eval_20260729"


def parse_cycles(text: str) -> dict[str, dict[str, int]]:
    patterns = {
        "window4": (
            r"PASS tb_local5_window_attention mode=(TARE|DIRECT) "
            r"dests=\d+ cmds=(\d+) cycles=(\d+)"
        ),
        "window16": (
            r"PASS tb_local5_window16 mode=(TARE|DIRECT) "
            r"dests=\d+ cmds=(\d+) cycles=(\d+)"
        ),
        "linebuf8x3": (
            r"PASS tb_local5_linebuf_window mode=(TARE|DIRECT) "
            r"windows=\d+ mean_cycles=(\d+)"
        ),
    }
    result: dict[str, dict[str, int]] = {}
    for name, pattern in patterns.items():
        rows: dict[str, int] = {}
        for match in re.finditer(pattern, text):
            mode = match.group(1).lower()
            rows[mode] = int(match.group(match.lastindex))
        if set(rows) != {"tare", "direct"}:
            raise RuntimeError(f"{name} 缺少 TARE/Direct 成对周期")
        result[name] = rows
    return result


def load_cells(name: str) -> int:
    path = SYNTH / f"{name}_stat.json"
    data = json.loads(path.read_text())
    design = data["design"]
    if design["num_processes"] != 0:
        raise RuntimeError(f"{path} 尚未完成 proc 展开")
    return int(design["num_cells"])


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    cycles = parse_cycles(LOG.read_text())
    cells = {
        mode: {dw: load_cells(f"{mode}_dw{dw}") for dw in (8, 9)}
        for mode in ("tare", "direct")
    }
    comparison = {}
    for name, row in cycles.items():
        comparison[name] = {
            "tare_cycle_overhead_vs_direct": row["tare"] / row["direct"] - 1.0,
            "tare_area_normalized_throughput_proxy_vs_direct_dw8": (
                cells["direct"][8] * row["direct"]
                / (cells["tare"][8] * row["tare"])
            ),
        }
    result = {
        "schema": "local5_tare_direct_arch_eval_v1",
        "evidence": {
            "cycle": "rtl",
            "cells": "yosys_abc_fast_generic_mapping_proxy",
            "power": "missing_saif_dc",
        },
        "cycles": cycles,
        "generic_cells": cells,
        "cell_reduction_tare_vs_direct_dw8": (
            1.0 - cells["tare"][8] / cells["direct"][8]
        ),
        "dest_w9_cell_overhead": {
            mode: cells[mode][9] / cells[mode][8] - 1.0
            for mode in ("tare", "direct")
        },
        "comparison": comparison,
        "decision": {
            "default_throughput_mode": "direct",
            "conditional_energy_mode": "tare",
            "tare_promotion_gate": (
                "fullres post-gate0 topology profile + SAIF/DC EDP "
                "improvement >= 15%"
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        "# Local5 TARE/Direct 同顶层架构消融",
        "",
        "## 证据边界",
        "",
        "- 周期来自同一 RTL 顶层、同一刺激的 Verilator 回归，记为 `[rtl]`。",
        "- 单元数来自同一 Yosys `abc -fast` 通用门映射，只记为 `[开放映射]`，"
        "不是 DC 面积。",
        "- 当前没有目标工艺、STA、SAIF 或功耗，不能声称真实 PPA/EDP。",
        "",
        "## 周期结果",
        "",
        "| 层级 | Direct | TARE | TARE 周期开销 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in cycles.items():
        overhead = comparison[name]["tare_cycle_overhead_vs_direct"]
        lines.append(
            f"| {name} | {row['direct']} | {row['tare']} | {pct(overhead)} |"
        )
    lines.extend(
        [
            "",
            "## 开放映射",
            "",
            "| 配置 | DEST_W=8 | DEST_W=9 | 9-bit 开销 |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in ("direct", "tare"):
        lines.append(
            f"| {mode.upper()} | {cells[mode][8]} | {cells[mode][9]} | "
            f"{pct(result['dest_w9_cell_overhead'][mode])} |"
        )
    lines.extend(
        [
            "",
            f"TARE 相对 Direct 的通用单元减少为 "
            f"{pct(result['cell_reduction_tare_vs_direct_dw8'])}。",
            "",
            "## 面积归一吞吐代理",
            "",
            "| 层级 | TARE / Direct |",
            "|---|---:|",
        ]
    )
    for name, row in comparison.items():
        lines.append(
            f"| {name} | "
            f"{row['tare_area_normalized_throughput_proxy_vs_direct_dw8']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "该指标把不同逻辑门等权处理，只能用于候选淘汰，不能进入论文主 PPA 表。",
            "",
            "## 架构决策",
            "",
            "1. 默认吞吐模式冻结为 Direct；TARE 当前不是周期加速方案。",
            "2. TARE 保留为精确低功耗候选，其价值必须由 fullres post-gate0 "
            "拓扑分布和 SAIF/DC EDP 共同证明。",
            "3. 只有同约束 EDP 改善至少 15%，TARE 才晋级为 DATE 主文机制；"
            "否则作为负结果或附录消融。",
            "4. DEST_W=9 是 T=450 正式配置，8-bit 结果仅用于 crop/T=162 对照。",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
