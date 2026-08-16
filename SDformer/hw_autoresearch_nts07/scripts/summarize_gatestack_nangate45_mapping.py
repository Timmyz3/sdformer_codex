#!/usr/bin/env python3
"""汇总 GateStack C0/C1 Nangate45 开放目标库逻辑映射。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def parse_log(path: Path) -> dict[str, int | float]:
    text = path.read_text(encoding="utf-8")
    hierarchy = text.split("Printing statistics.")[-1]
    return {
        "cells": int(re.search(r"Number of cells:\s+(\d+)", hierarchy).group(1)),
        "mem_v2": int(re.search(r"\$mem_v2\s+(\d+)", hierarchy).group(1)),
        "dff_x1": int(re.search(r"DFF_X1\s+(\d+)", hierarchy).group(1)),
        "logic_area": float(
            re.search(r"Chip area for module .*:\s+([0-9.]+)", hierarchy).group(1)
        ),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-log", type=Path, required=True)
    parser.add_argument("--c1-log", type=Path, required=True)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    c0 = parse_log(args.c0_log)
    c1 = parse_log(args.c1_log)
    builder_speedup = 14078 / 10035
    area_ratio = c1["logic_area"] / c0["logic_area"]
    result = {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "evidence": "[开放目标库逻辑映射代理]+[rtl周期]",
        "library": {
            "path": str(args.lib.resolve()),
            "sha256": sha256(args.lib),
            "corner_name": "NangateOpenCellLibrary_typical",
        },
        "mapping_contract": {
            "tool": "Yosys dfflibmap + ABC",
            "constraint": "未提供时钟约束；面积导向组合映射",
            "memory": "$mem_v2 保持未映射，面积不计入 logic_area",
            "physical": "无布线、时钟树、拥塞、PVT sweep或SRAM macro",
        },
        "c0": c0,
        "c1": c1,
        "comparison": {
            "builder_speedup": builder_speedup,
            "logic_area_ratio": area_ratio,
            "logic_area_increase": area_ratio - 1.0,
            "area_normalized_throughput_ratio": builder_speedup / area_ratio,
        },
        "decision": "C1仅保留为吞吐模式；系统级EDP未证明前，C0+BPB是面积效率默认候选",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cmp = result["comparison"]
    lines = [
        "# GateStack C0/C1 Nangate45 开放目标库映射",
        "",
        "## 1. 结论",
        "",
        f"同一 Nangate45 typical Liberty 下，C0 逻辑面积代理为 {c0['logic_area']:.3f}，"
        f"C1 为 {c1['logic_area']:.3f}，增加 {cmp['logic_area_increase']:.2%}。"
        f"结合 45-head Builder 的 {builder_speedup:.3f}x 吞吐，面积归一吞吐为 C0 的 "
        f"{cmp['area_normalized_throughput_ratio']:.3f}x。",
        "",
        "该结果否决了“C1天然具有更好面积效率”的假设。C1 暂时只保留为吞吐模式；若完整 projection 的能量/EDP 不能抵消面积代价，默认架构应回退 C0+BPB。",
        "",
        "## 2. 映射结果",
        "",
        "| 指标 | C0 | C1 | 比值 |",
        "|---|---:|---:|---:|",
        f"| logic area | {c0['logic_area']:.3f} | {c1['logic_area']:.3f} | {area_ratio:.3f}x |",
        f"| mapped+memory cells | {c0['cells']:,} | {c1['cells']:,} | {c1['cells']/c0['cells']:.3f}x |",
        f"| DFF_X1 | {c0['dff_x1']:,} | {c1['dff_x1']:,} | {c1['dff_x1']/c0['dff_x1']:.3f}x |",
        f"| 未映射 `$mem_v2` | {c0['mem_v2']} | {c1['mem_v2']} | 不计面积 |",
        "",
        "## 3. 口径与限制",
        "",
        "- 工具：Yosys `dfflibmap` + ABC；Liberty SHA256 已写入 JSON；",
        "- 映射未施加时钟约束，不能报告 WNS、Fmax 或时序闭合；",
        "- `$mem_v2` 保持未映射，logic area 不含 workspace、RAW scratch、slot SRAM 等 memory；",
        "- 无布线、时钟树、拥塞、PVT sweep、SAIF 动态功耗和 SRAM macro；",
        "- 该证据是公平逻辑面积消融，不是 DC、STA、功耗或流片 PPA；",
        "- 目标库映射后尚未完成 LEC，mapped netlist 只作结构产物。",
    ]
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
