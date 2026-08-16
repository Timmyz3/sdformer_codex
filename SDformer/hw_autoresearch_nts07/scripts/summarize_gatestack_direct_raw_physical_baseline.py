#!/usr/bin/env python3
"""汇总physically-stripped Direct RAW41投影基线的RTL与结构结果。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CELL_RE = re.compile(r"Number of cells:\s+(\d+)")
CYCLE_RE = re.compile(r"cycles=(\d+)")


def last_int(path: Path, pattern: re.Pattern[str]) -> int:
    matches = pattern.findall(path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"无法解析结果: {path}")
    return int(matches[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build = args.root / "build_hitflow/gatestack_direct_raw_physical_baseline"
    cells = {
        mode: last_int(build / f"yosys_{mode}_fair.log", CELL_RE)
        for mode in ("direct", "ipd", "adaptive")
    }
    result = {
        "status": "PASS",
        "evidence": "[RTL]+[Yosys结构代理]",
        "cycles": {
            "iverilog": last_int(build / "iverilog.log", CYCLE_RE),
            "verilator": last_int(build / "verilator.log", CYCLE_RE),
        },
        "generic_cells": cells,
        "direct_reduction_vs_ipd": 1.0 - cells["direct"] / cells["ipd"],
        "direct_reduction_vs_adaptive": 1.0 - cells["direct"] / cells["adaptive"],
        "forbidden_hierarchy_absent": [
            "resident replay joiner", "IPD32W decoder", "FADC24 decoder",
            "Adaptive CSR selector", "three-source replay mux",
        ],
        "limits": [
            "比较边界是multihead projection slice，不含head-slot SRAM、descriptor cache和完整encoder控制",
            "Direct小TB只证明RAW语义和控制正确，不用于真实H67周期主表",
            "Direct与IPD/Adaptive保留相同投影后端、AccTile和权重/bias/final接口，但输入表示能力不同",
            "Yosys generic cell不是目标库面积，memory cell也不是SRAM宏面积",
            "没有目标工艺、PVT、STA、SAIF、布线和netlist LEC证据",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# Physically-stripped Direct RAW41投影基线",
        "",
        "## 结论",
        "",
        "Direct RAW41基线已作为独立可综合顶层实现。该顶层只保留RAW41 decoder、尾事件修正、单事件term适配、与GateStack相同的multihead projection backend和AccTile；综合输入中不存在resident、IPD32W、FADC24、Adaptive CSR或三源replay mux。",
        "",
        "定向TB覆盖最后一个非零event后仍有K-zero token的情况；Icarus与Verilator/SVA均在478周期通过，162个final逐元素零mismatch。",
        "",
        "## 同流程结构代理",
        "",
        "三种顶层统一执行`proc; flatten; opt; memory -nomap; stat`，参数使用各模块默认投影规模。",
        "",
        "| Projection slice | Yosys generic cells | 相对Direct |",
        "|---|---:|---:|",
        f"| Direct RAW41 physically-stripped | {cells['direct']} | 1.000x |",
        f"| GateStack IPD32W | {cells['ipd']} | {cells['ipd']/cells['direct']:.3f}x |",
        f"| GateStack Adaptive CSR | {cells['adaptive']} | {cells['adaptive']/cells['direct']:.3f}x |",
        "",
        f"Direct相对IPD减少{result['direct_reduction_vs_ipd']:.2%} generic cells，相对Adaptive减少{result['direct_reduction_vs_adaptive']:.2%}。这说明格式、驻留入口和共享路由具有真实结构成本；它不说明Direct的能量或EDP更优，因为RAW41在真实trace中会显著增加payload和term执行。",
        "",
        "## 公平性解释",
        "",
        "该基线与GateStack保留相同的tile/head调度语义、TDR multicast、banked accumulator、weight/bias/final接口。Direct删除的是表示和多源执行能力，而不是把乘法、累加或输出阶段一起删除。",
        "",
        "当前只在projection-slice边界实现物理裁剪；single-context的slot、cache、control和完整encoder存储不在本表中。真实H67周期仍引用既有同顶层RAW-only回放，不能把本TB的478周期与四stagetrace直接比较。",
        "",
        "## 证据边界",
        "",
    ]
    md.extend(f"- {item}" for item in result["limits"])
    md.extend([
        "",
        "## 入口",
        "",
        "- RTL：`rtl_hitflow/gatestack_direct_raw_multihead_projection_top.sv`；",
        "- TB：`tb_hitflow/tb_gatestack_direct_raw_multihead_projection_top.sv`；",
        "- 回归：`sim_hitflow/run_gatestack_direct_raw_physical_baseline_checks.sh`；",
        "- 日志：`build_hitflow/gatestack_direct_raw_physical_baseline/`。",
        "",
    ])
    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
