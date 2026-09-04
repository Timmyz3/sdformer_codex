#!/usr/bin/env python3
"""汇总DCTF96完整flatten开放库映射，并与既有叶模块算术和比较。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TOP_NAME = "gatestack_dctf96_term_datapath_top"
RUN_NAME = "top_q2_tokens162_out32"
AREA_RE = re.compile(r"Chip area for module .*: ([0-9]+(?:\.[0-9]+)?)")
CELLS_RE = re.compile(r"Number of cells:\s+([0-9]+)")
PROCESSES_RE = re.compile(r"Number of processes:\s+([0-9]+)")
CELL_ROW_RE = re.compile(r"^\s+(\S+)\s+([0-9]+)\s*$", re.MULTILINE)
SHA_RE = re.compile(r"^([0-9a-f]{64})\s+(.+)$")
TOP_STAT_RE = re.compile(
    rf"^=== .*{re.escape(TOP_NAME)}.* ===$", re.MULTILINE
)


def _last(pattern: re.Pattern[str], text: str) -> int | float:
    values = pattern.findall(text)
    if not values:
        raise RuntimeError(f"无法匹配最终统计字段：{pattern.pattern}")
    value = values[-1]
    return float(value) if "." in value else int(value)


def parse_final_stat(log_path: Path) -> dict:
    text = log_path.read_text()
    matches = list(TOP_STAT_RE.finditer(text))
    if not matches:
        raise RuntimeError(f"映射日志缺少最终top统计：{log_path}")
    block = text[matches[-1].end() :]
    area = _last(AREA_RE, block)
    cells = _last(CELLS_RE, block)
    processes = _last(PROCESSES_RE, block)

    cell_section = block.rsplit("Number of cells:", 1)[1]
    cell_section = cell_section.split("Chip area for module", 1)[0]
    cell_types = {
        cell_type: int(count)
        for cell_type, count in CELL_ROW_RE.findall(cell_section)
    }
    mem_v2 = cell_types.get("$mem_v2", 0)
    unmapped_dollar_cells = {
        name: count
        for name, count in cell_types.items()
        if name.startswith("$") and name != "$mem_v2"
    }

    if area <= 0 or cells <= 0:
        raise RuntimeError("最终映射统计必须具有非零面积与cell数")
    if processes != 0:
        raise RuntimeError(f"最终映射仍有{processes}个process")
    if unmapped_dollar_cells:
        raise RuntimeError(f"最终映射仍有未映射$单元：{unmapped_dollar_cells}")

    return {
        "logic_area": area,
        "cells": cells,
        "processes": processes,
        "mem_v2": mem_v2,
        "unmapped_dollar_cells": unmapped_dollar_cells,
        "cell_types": cell_types,
    }


def _row_by_name(report: dict, name: str) -> dict:
    for row in report.get("rows", []):
        if row.get("name") == name:
            return row
    raise RuntimeError(f"既有报告缺少{name}行")


def _read_provenance(output_dir: Path) -> dict:
    version_path = output_dir / "yosys_version.txt"
    sha_path = output_dir / "input_sha256.txt"
    if not version_path.is_file() or not sha_path.is_file():
        raise RuntimeError("缺少Yosys版本或输入SHA256记录")
    inputs = []
    for line in sha_path.read_text().splitlines():
        match = SHA_RE.match(line)
        if not match:
            raise RuntimeError(f"非法SHA256记录：{line}")
        inputs.append({"sha256": match.group(1), "path": match.group(2)})
    if not inputs:
        raise RuntimeError("输入SHA256记录为空")
    return {
        "yosys_version": version_path.read_text().strip(),
        "inputs": inputs,
    }


def build_report(
    mapping_dir: Path,
    executor_report_path: Path,
    frontend_report_path: Path,
    output_dir: Path,
) -> dict:
    log_path = mapping_dir / f"{RUN_NAME}.log"
    netlist_path = mapping_dir / f"{RUN_NAME}_mapped.v"
    if not netlist_path.is_file() or netlist_path.stat().st_size == 0:
        raise RuntimeError("最终映射网表缺失或为空")

    top = parse_final_stat(log_path)
    top["name"] = RUN_NAME
    top["label"] = "DCTF96完整flatten top"
    top["netlist_bytes"] = netlist_path.stat().st_size

    executor_report = json.loads(executor_report_path.read_text())
    frontend_report = json.loads(frontend_report_path.read_text())
    executor = _row_by_name(executor_report, "executor_32")
    adapter = _row_by_name(frontend_report, "adapter")
    fabric = _row_by_name(frontend_report, "fabric_q2")
    if executor.get("out_tile") != 32:
        raise RuntimeError("既有executor报告不是OUT_TILE=32")

    terms = [
        {"name": "executor_32", "count": 3, **executor},
        {"name": "adapter", "count": 1, **adapter},
        {"name": "fabric_q2", "count": 1, **fabric},
    ]
    arithmetic_sum = {
        "formula": "3 * executor_32 + adapter + fabric_q2",
        "logic_area": round(
            sum(row["count"] * row["logic_area"] for row in terms), 6
        ),
        "cells": sum(row["count"] * row["cells"] for row in terms),
        "mem_v2": sum(row["count"] * row.get("mem_v2", 0) for row in terms),
        "terms": terms,
    }
    comparison = {
        "logic_area_delta_top_minus_leaf_sum": round(
            top["logic_area"] - arithmetic_sum["logic_area"], 6
        ),
        "logic_area_ratio_top_over_leaf_sum": (
            top["logic_area"] / arithmetic_sum["logic_area"]
        ),
        "logic_area_change": (
            top["logic_area"] / arithmetic_sum["logic_area"] - 1.0
        ),
        "cell_delta_top_minus_leaf_sum": top["cells"] - arithmetic_sum["cells"],
        "cell_ratio_top_over_leaf_sum": top["cells"] / arithmetic_sum["cells"],
        "mem_v2_delta": top["mem_v2"] - arithmetic_sum["mem_v2"],
        "interpretation": (
            "差值混合top完成tracker、地址合同、跨层优化与可观察输出裁剪；"
            "功能边界和优化上下文不同，不能称为纯协调器面积"
        ),
    }

    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "evidence": "开放Nangate45无约束完整flatten逻辑映射代理",
        "top": TOP_NAME,
        "parameters": {"Q": 2, "TOKENS": 162, "OUT_TILE": 32},
        "flow": (
            "proc; flatten; opt; memory -nomap; opt; techmap; opt; "
            "dfflibmap; abc; clean; check -assert; stat"
        ),
        "library": "NangateOpenCellLibrary_typical.lib",
        "provenance": _read_provenance(output_dir),
        "quality_checks": {
            "netlist_nonempty": True,
            "processes_zero": top["processes"] == 0,
            "unmapped_dollar_cells_zero_excluding_mem_v2": not top[
                "unmapped_dollar_cells"
            ],
            "mem_v2_reported_separately_and_area_excluded": True,
        },
        "flatten_top": top,
        "leaf_arithmetic_sum": arithmetic_sum,
        "comparison": comparison,
        "rtl_verification_facts": {
            "icarus": "PASS，synthetic TB为91周期",
            "verilator_dynamic_sva": "PASS，synthetic TB为89周期",
            "yosys": "PASS",
            "erie": "PASS，RTL/TB均为0 error / 0 warning",
            "scenarios": [
                "非法term零副作用",
                "cross-supertile overlap保持物理tile正确",
                "三bank旧epoch响应均stale drop且不污染Acc或完成路径",
            ],
        },
        "limits": [
            "无SDC与STA，未形成时序结论",
            "无SAIF，未形成功耗结论",
            "无SRAM宏表征，$mem_v2数量单列且不计面积",
            "未使用DC，也无布局布线结果",
            "不得称为ASIC PPA、论文最终面积或签核面积",
        ],
    }


def render_markdown(report: dict) -> str:
    top = report["flatten_top"]
    leaf = report["leaf_arithmetic_sum"]
    comparison = report["comparison"]
    lines = [
        "# DCTF96完整Flatten开放映射代理",
        "",
        "本报告是同一Nangate45 typical Liberty下的**无约束逻辑映射代理**。它不是ASIC PPA，也不是论文最终面积。",
        "",
        "## 映射与完整性",
        "",
        f"- 顶层：`{report['top']}`；参数：`Q=2`、`TOKENS=162`、`OUT_TILE=32`；",
        f"- Yosys：`{report['provenance']['yosys_version']}`；",
        "- 流程：`proc; flatten; opt; memory -nomap; opt; techmap; opt; dfflibmap; abc; clean; check -assert; stat`；",
        f"- 最终网表：{top['netlist_bytes']}字节，process为{top['processes']}，除`$mem_v2`外未映射`$`单元为{len(top['unmapped_dollar_cells'])}；",
        f"- `$mem_v2`为{top['mem_v2']}个，单独报告，未计入库面积值。",
        "",
        "## 谨慎开销比较",
        "",
        "| 口径 | 逻辑库面积值 | cell | `$mem_v2` |",
        "|---|---:|---:|---:|",
        f"| 三个executor + adapter + Q2 fabric算术和 | {leaf['logic_area']:.3f} | {leaf['cells']} | {leaf['mem_v2']} |",
        f"| 完整flatten top | {top['logic_area']:.3f} | {top['cells']} | {top['mem_v2']} |",
        f"| top减算术和 | {comparison['logic_area_delta_top_minus_leaf_sum']:.3f} | {comparison['cell_delta_top_minus_leaf_sum']} | {comparison['mem_v2_delta']} |",
        "",
        f"面积比`top/算术和={comparison['logic_area_ratio_top_over_leaf_sum']:.6f}`，变化为`{comparison['logic_area_change'] * 100:.3f}%`。",
        "",
        "这个差值混合了top完成tracker、地址合同、跨层优化和可观察输出裁剪。叶模块分别映射与完整flatten映射具有不同的可观察边界，ABC也可以跨原层次重写逻辑，因此差值不能称为纯协调器面积、纯tracker面积或物理互连面积。",
        "",
        "## 证据边界",
        "",
        "没有SDC、STA、SAIF、SRAM宏、DC或布局布线结果；没有时钟树、互连寄生、PVT收敛和功耗分析。以上库面积值不得称为ASIC PPA、论文最终面积或签核面积。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--executor-report", type=Path, required=True)
    parser.add_argument("--frontend-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.mapping_dir,
        args.executor_report,
        args.frontend_report,
        args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps(report["comparison"], ensure_ascii=False))


if __name__ == "__main__":
    main()
