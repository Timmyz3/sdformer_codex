#!/usr/bin/env python3
"""联合汇总 supertile 真实 RTL 周期与开放目标库逻辑映射。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from summarize_gatestack_builder_projection_allstages import parse_result


AREA_RE = re.compile(r"Chip area for module .*: ([0-9.]+)")
CELLS_RE = re.compile(r"Number of cells:\s+([0-9]+)")
MEM_RE = re.compile(r"\$mem_v2\s+([0-9]+)")


def last_match(pattern: re.Pattern[str], text: str) -> int | float:
    values = pattern.findall(text)
    if not values:
        raise RuntimeError(f"无法匹配 {pattern.pattern}")
    value = values[-1]
    return float(value) if "." in value else int(value)


def build_report(mapping_dir: Path, cycle_build_dir: Path) -> dict:
    rows = []
    for width in (32, 64, 96, 128):
        log = (mapping_dir / f"w{width}.log").read_text()
        total_cycles = 0
        stage_cycles = []
        for stage in range(4):
            sim_log = cycle_build_dir / f"w{width}_s{stage}" / "iverilog.log"
            lines = [line for line in sim_log.read_text().splitlines()
                     if line.startswith("RESULT stage=")]
            if len(lines) != 1:
                raise RuntimeError(f"{sim_log} RESULT数量错误")
            result = parse_result(lines[0])
            if result["status"] != "PASS" or result["mismatches"] != "0":
                raise RuntimeError(f"{sim_log} 未通过")
            cycles = int(result["total_cycles"])
            total_cycles += cycles
            stage_cycles.append(cycles)
        rows.append({
            "out_tile": width,
            "stage_cycles": stage_cycles,
            "total_cycles": total_cycles,
            "logic_area": last_match(AREA_RE, log),
            "cells": last_match(CELLS_RE, log),
            "mem_v2": last_match(MEM_RE, log),
        })
    baseline = rows[0]
    for row in rows:
        row["speedup"] = baseline["total_cycles"] / row["total_cycles"]
        row["logic_area_ratio"] = row["logic_area"] / baseline["logic_area"]
        row["area_normalized_throughput"] = (
            row["speedup"] / row["logic_area_ratio"]
        )
    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "cycle_evidence": "[rtl] sample0/B0/window0 S0-S3",
        "area_evidence": "[开放目标库逻辑映射代理]，memory面积未计",
        "rows": rows,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Projection Supertile 周期与开放库逻辑面积联合消融",
        "",
        "周期为真实四 stage RTL；面积为 Nangate45 未约束逻辑映射代理，`$mem_v2` 面积未计，不能替代 DC/PPA。",
        "",
        "| OUT_TILE | S0 | S1 | S2 | S3 | 总周期 | 加速 | logic area | 面积比 | 面积归一吞吐 | `$mem_v2` |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        stages = " | ".join(str(value) for value in row["stage_cycles"])
        lines.append(
            f"| {row['out_tile']} | {stages} | {row['total_cycles']} | "
            f"{row['speedup']:.3f}x | {row['logic_area']:.3f} | "
            f"{row['logic_area_ratio']:.3f}x | "
            f"{row['area_normalized_throughput']:.3f}x | {row['mem_v2']} |"
        )
    lines += [
        "",
        "## 判定规则",
        "",
        "- 宽 supertile 通过减少相同 head payload 在 output tile 间的重复 replay/decode，保持逐元素 exact；",
        "- 若面积归一吞吐不高于 1，或后续 SRAM+SAIF 的 EDP 改善低于 15%，则不得晋级默认配置；",
        "- 当前 mapping 未计 AccTile/decoder memory 宏面积、时序和功耗，只能筛选候选。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--cycle-build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.mapping_dir, args.cycle_build_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps(report["rows"], ensure_ascii=False))


if __name__ == "__main__":
    main()
