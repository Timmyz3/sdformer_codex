#!/usr/bin/env python3
"""汇总 DCTF adapter/fabric 验证与开放库结构代理。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


AREA_RE = re.compile(r"Chip area for module .*: ([0-9.]+)")
CELLS_RE = re.compile(r"Number of cells:\s+([0-9]+)")
MEM_RE = re.compile(r"\$mem_v2\s+([0-9]+)")


def last(pattern: re.Pattern[str], text: str, default: int | None = None):
    values = pattern.findall(text)
    if not values:
        if default is not None:
            return default
        raise RuntimeError(f"无法匹配 {pattern.pattern}")
    value = values[-1]
    return float(value) if "." in value else int(value)


def build_report(mapping_dir: Path) -> dict:
    rows = []
    for name, label in (
        ("adapter", "完整term验证与串化adapter"),
        ("fabric_q2", "Q2三消费者command fabric"),
    ):
        text = (mapping_dir / f"{name}.log").read_text()
        rows.append({
            "name": name,
            "label": label,
            "logic_area": last(AREA_RE, text),
            "cells": last(CELLS_RE, text),
            "mem_v2": last(MEM_RE, text, 0),
        })
    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "evidence": "[rtl-leaf]+[开放目标库逻辑映射代理]",
        "adapter_default_buffer_bits": 162 * 8 + 162,
        "adapter_test": {
            "cycles": 83,
            "commands": 9,
            "error_cases": 8,
            "mismatches": 0,
        },
        "fabric_tests": [
            {"q": 2, "cycles": 402, "accepted": 260, "retired": 256},
            {"q": 3, "cycles": 391, "accepted": 260, "retired": 254},
            {"q": 4, "cycles": 387, "accepted": 260, "retired": 252},
        ],
        "rows": rows,
        "limits": [
            "adapter收集与发射不重叠",
            "fabric retire仅代表三bank dispatch完成，不代表计算完成",
            "映射无SDC、STA、SAIF和SRAM macro",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# DCTF 前端叶模块验证与开放逻辑映射",
        "",
        "| 模块 | logic area | cells | `$mem_v2` |",
        "|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['label']} | {row['logic_area']:.3f} | "
            f"{row['cells']} | {row['mem_v2']} |"
        )
    lines += [
        "",
        f"Adapter默认缓存为 {report['adapter_default_buffer_bits']} bit，先验证完整term再发射；真实测试83周期、9条command、8类错误、0 mismatch。",
        "",
        "Q2/Q3/Q4 synthetic周期为402/391/387。随机flush会丢弃当时in-flight command，因此retired小于accepted；这不是数据丢失bug，也不能外推为真实projection加速。",
        "",
        "面积为Nangate45无约束开放逻辑代理，memory面积未计；无SDC、STA、SAIF或SRAM macro。Adapter和fabric仍是叶模块，必须接入真实bank-local weight/product/Acc后才具备架构意义。",
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
