#!/usr/bin/env python3
"""汇总 Central96 与 3xIndependent32 的同库开放逻辑映射。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


AREA_RE = re.compile(r"Chip area for module .*: ([0-9.]+)")
CELLS_RE = re.compile(r"Number of cells:\s+([0-9]+)")
MEM_RE = re.compile(r"\$mem_v2\s+([0-9]+)")


def last(pattern: re.Pattern[str], text: str) -> int | float:
    values = pattern.findall(text)
    if not values:
        raise RuntimeError(f"无法匹配 {pattern.pattern}")
    value = values[-1]
    return float(value) if "." in value else int(value)


def build_report(mapping_dir: Path) -> dict:
    rows = []
    for name, label, decoders in (
        ("central96", "HATF96-Central", 1),
        ("three_independent32", "3xIndependent32", 3),
    ):
        text = (mapping_dir / f"{name}.log").read_text()
        rows.append({
            "name": name,
            "label": label,
            "product_lanes": 96,
            "decoder_instances": decoders,
            "logic_area": last(AREA_RE, text),
            "cells": last(CELLS_RE, text),
            "mem_v2": last(MEM_RE, text),
        })
    central, independent = rows
    central["logic_area_reduction_vs_independent"] = (
        1.0 - central["logic_area"] / independent["logic_area"])
    central["cell_reduction_vs_independent"] = (
        1.0 - central["cells"] / independent["cells"])
    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "library": "NangateOpenCellLibrary_typical.lib",
        "evidence": "[开放目标库逻辑映射代理]",
        "same_product_lanes": 96,
        "rows": rows,
        "limits": [
            "未使用SDC、STA、SAIF或SRAM macro",
            "$mem_v2不计入logic area，且不同设计的memory宽度不同",
            "3xIndependent32仅完成小规模结构/无串扰测试，未完成H67真实四阶段wall-time",
        ],
    }


def render_markdown(report: dict) -> str:
    central, independent = report["rows"]
    lines = [
        "# 等并行度 96-Lane 开放逻辑面积对照",
        "",
        "两种结构均固定96个product lane，并使用同一RTL集合和Nangate45开放库。面积不含未映射memory，不能替代DC、STA、SRAM macro或SAIF功耗结果。",
        "",
        "| 结构 | decoder实例 | product lane | logic area | cells | `$mem_v2` |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['label']} | {row['decoder_instances']} | "
            f"{row['product_lanes']} | {row['logic_area']:.3f} | "
            f"{row['cells']} | {row['mem_v2']} |"
        )
    lines += [
        "",
        "## 有限结论",
        "",
        f"- Central96 的开放逻辑面积相对三路独立减少 {central['logic_area_reduction_vs_independent'] * 100:.3f}%；",
        f"- 标准单元数减少 {central['cell_reduction_vs_independent'] * 100:.3f}%；",
        "- 该差异支持共享 replay/decoder/control 具有逻辑面积价值，但不证明目标总面积、功耗或EDP；",
        "- `$mem_v2=3/9` 只是未映射memory cell数量，宽度和用途不同，不能直接写成存储面积降低3倍；",
        "- 在真实三读口slot、相同六个32-lane Acc SRAM、相同权重/bias bank和H67 S0-S3 wall-time完成前，不能宣称HATF96已赢得公平架构对照。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.mapping_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps(report["rows"], ensure_ascii=False))


if __name__ == "__main__":
    main()
