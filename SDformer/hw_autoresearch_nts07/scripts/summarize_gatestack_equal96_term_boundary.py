#!/usr/bin/env python3
"""汇总三种96-lane term/event边界结构的开放逻辑映射。"""

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
        raise RuntimeError(f"映射日志缺少最终统计: {log_path}")
    cell_section = text.rsplit("Number of cells:", 1)[1]
    cell_section = cell_section.split("Chip area for module", 1)[0]
    cell_types = {
        name: int(count) for name, count in CELL_ROW_RE.findall(cell_section)
    }
    unmapped = {
        name: count for name, count in cell_types.items()
        if name.startswith("$") and name != "$mem_v2"
    }
    row = {
        "logic_area": float(areas[-1]),
        "cells": int(cells[-1]),
        "processes": int(processes[-1]),
        "mem_v2": cell_types.get("$mem_v2", 0),
        "unmapped_dollar_cells": unmapped,
        "netlist_bytes": netlist_path.stat().st_size,
    }
    if row["logic_area"] <= 0 or row["cells"] <= 0:
        raise RuntimeError(f"面积或cell统计非法: {row}")
    if row["processes"] != 0 or unmapped:
        raise RuntimeError(f"网表未完成映射: {row}")
    return row


def build_report(output_dir: Path) -> dict:
    mapping_dir = output_dir / "mapping"
    definitions = (
        ("central96_term", "Central96", 1, "中央96-lane"),
        ("independent32x3_term", "3xIndependent32", 3, "三套独立32-lane"),
        ("dctf96_term", "DCTF96", 1, "共享命令、三路bank-local"),
    )
    rows = []
    for name, label, term_clients, organization in definitions:
        row = parse_mapping(
            mapping_dir / f"{name}.log",
            mapping_dir / f"{name}_mapped.v",
        )
        row.update({
            "name": name,
            "label": label,
            "term_clients": term_clients,
            "product_lanes": 96,
            "physical_weight_banks": 3,
            "physical_acc_banks": 6,
            "organization": organization,
        })
        rows.append(row)

    central, independent, dctf = rows
    for baseline_name, baseline in (
        ("central96", central), ("independent32x3", independent)
    ):
        dctf[f"logic_area_reduction_vs_{baseline_name}"] = (
            1.0 - dctf["logic_area"] / baseline["logic_area"]
        )
        dctf[f"cell_reduction_vs_{baseline_name}"] = (
            1.0 - dctf["cells"] / baseline["cells"]
        )

    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "evidence": "[开放Nangate45无约束逻辑映射代理]",
        "boundary": "term/event到weight/product/Acc/bias/final",
        "same_product_lanes": 96,
        "rows": rows,
        "limits": [
            "无SDC与STA，不形成频率或时序结论",
            "无SAIF，不形成功耗或EDP结论",
            "$mem_v2未替换为相同SRAM宏且不计入logic area",
            "Central行为RTL仍是2x96-wide Acc，物理签核必须拆为6x32-lane相同宏",
            "Independent假设三个独立term client；slot三读口代价不在本projection边界",
            "DCTF额外包含term验证缓存与Q2命令fabric，mem_v2数量不能直接换算面积",
            "未运行DC、Formality、P&R或目标工艺SRAM编译器",
        ],
    }


def render_markdown(report: dict) -> str:
    dctf = report["rows"][2]
    lines = [
        "# 三种96-Lane Term边界开放逻辑映射对照",
        "",
        "三种结构均从term/event输入开始，固定96个INT8 product lane、三个物理weight bank和六个物理Acc bank的架构合同。表中面积仅为Nangate45无约束逻辑映射代理。",
        "",
        "| 结构 | term client | 组织方式 | 逻辑面积值 | cells | `$mem_v2` |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['label']} | {row['term_clients']} | {row['organization']} | "
            f"{row['logic_area']:.3f} | {row['cells']} | {row['mem_v2']} |"
        )
    lines += [
        "",
        "## 有限结论",
        "",
        f"- DCTF96相对Central96的逻辑面积变化为{dctf['logic_area_reduction_vs_central96'] * 100:.3f}%；",
        f"- DCTF96相对3xIndependent32的逻辑面积变化为{dctf['logic_area_reduction_vs_independent32x3'] * 100:.3f}%；",
        "- 正值表示DCTF更小，负值表示DCTF更大；该结论不含SRAM面积、频率和功耗；",
        "- 只有结合真实四阶段周期，才能讨论面积归一吞吐；在SAIF与相同SRAM宏完成前仍不能讨论EDP。",
        "",
        "## 证据边界",
        "",
        *[f"- {item}；" for item in report["limits"]],
        "",
    ]
    return "\n".join(lines)


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
    print(json.dumps(report["rows"], ensure_ascii=False))


if __name__ == "__main__":
    main()
