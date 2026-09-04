#!/usr/bin/env python3
"""汇总同一 Yosys 流程下的 Local5 投影后端结构代理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIGS = {
    "TCFM-5": {
        "file": "tcfm5_stat.json",
        "banks": 5,
        "slots": 450,
        "mapping": "(x+2y) mod 5",
    },
    "Affine-4": {
        "file": "affine4_stat.json",
        "banks": 4,
        "slots": 480,
        "mapping": "(x+2y) mod 4",
    },
    "Linear-5": {
        "file": "linear5_stat.json",
        "banks": 5,
        "slots": 450,
        "mapping": "raster-id mod 5",
    },
    "Role-Sharded": {
        "file": "role_sharded_stat.json",
        "banks": 5,
        "slots": 2250,
        "mapping": "five role-local rasters",
    },
}


def module_stats(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    design = payload["design"]
    cell_types = design["num_cells_by_type"]
    return {
        "cells": design["num_cells"],
        "wire_bits": design["num_wire_bits"],
        "memories": cell_types.get("$mem_v2", 0),
        "multipliers": cell_types.get("$mul", 0),
        "muxes": cell_types.get("$mux", 0),
        "dividers": cell_types.get("$div", 0),
        "modulos": cell_types.get("$mod", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycle-evidence", type=Path, required=True)
    args = parser.parse_args()

    cycle_evidence = json.loads(args.cycle_evidence.read_text())
    cycle_by_name = cycle_evidence["cycles"]
    rows: list[dict[str, object]] = []
    for name, config in CONFIGS.items():
        row = {"name": name, **config}
        row["directed_cycles"] = int(cycle_by_name[name])
        row.update(module_stats(args.input_dir / str(config["file"])))
        if row["dividers"] or row["modulos"]:
            raise SystemExit(f"{name} 仍含运行时除法或取模")
        rows.append(row)

    baseline_cycles = int(rows[0]["directed_cycles"])
    for row in rows:
        row["cycle_overhead_pct"] = (
            100.0 * (int(row["directed_cycles"]) / baseline_cycles - 1.0)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "evidence": "统一 Yosys 展平结构代理；非 PPA",
        "cycle_evidence": cycle_evidence,
        "rows": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )

    lines = [
        "# Local5 投影强基线统一结构比较",
        "",
        "四个候选使用同一版本 Yosys、同一 `flatten/proc/opt/"
        "memory_collect/memory_dff` 流程、同一默认参数和相同同步 1R1W Acc "
        "RTL。结果用于消除历史脚本口径差异，**不是目标工艺 PPA**。",
        "",
        "周期来自同一个端到端 TB 和同一个 producer 分别连接四个真实后端 "
        "RTL 的 `projection_busy` 计数，包含 Acc clear、producer 反压与 "
        "close/drain，不含最终 readback。",
        "",
        "| 后端 | bank/槽 | 映射 | W6周期 | 周期开销 | cells | wire bits | "
        "mem | mul | mux |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['banks']}/{row['slots']} | "
            f"`{row['mapping']}` | {row['directed_cycles']} | "
            f"{row['cycle_overhead_pct']:.2f}% | {row['cells']} | "
            f"{row['wire_bits']} | {row['memories']} | "
            f"{row['multipliers']} | {row['muxes']} |"
        )
    lines.extend(
        [
            "",
            "## 可用结论",
            "",
            "- 四者均为 4 路输出乘法、同步 1R1W Acc，且展平网表中无"
            " `$div/$mod`；",
            "- TCFM-5 的价值应由零 replay、450 槽和同物理约束 EDP共同判断；",
            "- Affine-4 少一个 bank，却因不均衡需要 480 槽并承担 south/north "
            "replay；它是必须保留的面积/周期 Pareto 基线；",
            "- Linear-5 证明普通线性 banking 即使 bank 数相同，仍需动态冲突"
            "检测与 replay；",
            "- Role-Sharded 与 TCFM-5 同为零 replay，但前者复制五份"
            " partial-Acc 并在读回阶段归约；",
            "- cells/mux 数受 RTL 写法和开放映射影响，只能用于发现异常，不能"
            "写入论文主 PPA 表。",
            "",
            "## 证据边界",
            "",
            "- 周期来源为 `qfit_local5_projection_tile` 的"
            " `cycle_evidence.json`，报告脚本不再硬编码；",
            "- W6 定向 term 流不代表 full-resolution Local5；",
            "- 未映射目标 SRAM macro，未执行 DC、STA、SAIF 或布局布线；",
            "- post-G0 trace 到达后必须重算 mean/p95/p99 replay、stall 与 EDP。",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
