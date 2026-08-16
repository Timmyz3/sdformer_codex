#!/usr/bin/env python3
"""汇总DCTF96 bank-local完整projection开放库映射代理。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


AREA_RE = re.compile(r"Chip area for module .*: ([0-9]+(?:\.[0-9]+)?)")
CELLS_RE = re.compile(r"Number of cells:\s+([0-9]+)")
PROCESS_RE = re.compile(r"Number of processes:\s+([0-9]+)")
CELL_ROW_RE = re.compile(r"^\s+(\S+)\s+([0-9]+)\s*$", re.MULTILINE)


def parse_mapping(log_path: Path, netlist_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8")
    areas = AREA_RE.findall(text)
    cells = CELLS_RE.findall(text)
    processes = PROCESS_RE.findall(text)
    if not areas or not cells or not processes:
        raise RuntimeError("映射日志缺少最终统计")
    cell_section = text.rsplit("Number of cells:", 1)[1]
    cell_section = cell_section.split("Chip area for module", 1)[0]
    cell_types = {
        name: int(count) for name, count in CELL_ROW_RE.findall(cell_section)
    }
    unmapped = {
        name: count
        for name, count in cell_types.items()
        if name.startswith("$") and name != "$mem_v2"
    }
    result = {
        "logic_area": float(areas[-1]),
        "cells": int(cells[-1]),
        "processes": int(processes[-1]),
        "mem_v2": cell_types.get("$mem_v2", 0),
        "unmapped_dollar_cells": unmapped,
        "netlist_bytes": netlist_path.stat().st_size,
        "cell_types": cell_types,
    }
    if result["logic_area"] <= 0 or result["cells"] <= 0:
        raise RuntimeError("面积或cell统计非法")
    if result["processes"] != 0 or unmapped:
        raise RuntimeError(f"网表未完成映射: {result}")
    return result


def build_report(output_dir: Path) -> dict:
    mapping = output_dir / "mapping"
    run = "top_q2_tokens162_out32"
    netlist = mapping / f"{run}_mapped.v"
    if not netlist.is_file() or netlist.stat().st_size == 0:
        raise RuntimeError("映射网表缺失")
    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "evidence": "开放Nangate45无约束完整flatten逻辑映射代理",
        "top": "gatestack_dctf96_banklocal_projection_top",
        "parameters": {"Q": 2, "TOKENS": 162, "OUT_TILE": 32},
        "mapping": parse_mapping(mapping / f"{run}.log", netlist),
        "limits": [
            "无SDC与STA，不形成频率或时序结论",
            "无SAIF，不形成功耗或EDP结论",
            "未替换真实SRAM宏，$mem_v2单列且不计入逻辑库面积",
            "未运行DC、Formality或布局布线，不是ASIC签核",
            "本顶层从term/event开始，不含decoder，不能直接与含decoder的wrapper比较",
        ],
    }


def render_markdown(report: dict) -> str:
    row = report["mapping"]
    return "\n".join(
        [
            "# DCTF96 Bank-Local完整Projection开放映射代理",
            "",
            "本报告只记录Nangate45 typical Liberty上的无约束逻辑映射代理，不是DC/PPA或ASIC签核。",
            "",
            "## 结果",
            "",
            "| 顶层 | Q | TOKENS | 每bank lane | 逻辑库面积值 | cell | `$mem_v2` | 网表字节 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| `{report['top']}` | 2 | 162 | 32 | {row['logic_area']:.3f} | {row['cells']} | {row['mem_v2']} | {row['netlist_bytes']} |",
            "",
            f"最终process为{row['processes']}，除`$mem_v2`外未映射`$`单元为{len(row['unmapped_dollar_cells'])}。",
            "",
            "## 证据边界",
            "",
            *[f"- {item}；" for item in report["limits"]],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.output_dir)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["mapping"], ensure_ascii=False))


if __name__ == "__main__":
    main()
