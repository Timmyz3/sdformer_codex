#!/usr/bin/env python3
"""汇总 HATF96 的 BSF 开关消融、真实 RTL 周期和开放库逻辑映射。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from summarize_gatestack_builder_projection_allstages import parse_result


AREA_RE = re.compile(r"Chip area for module .*: ([0-9.]+)")
CELLS_RE = re.compile(r"Number of cells:\s+([0-9]+)")
MEM_RE = re.compile(r"\$mem_v2\s+([0-9]+)")
OUT_TILE = 96
ACC_W = 32


def last_match(pattern: re.Pattern[str], text: str) -> int | float:
    values = pattern.findall(text)
    if not values:
        raise RuntimeError(f"无法匹配 {pattern.pattern}")
    value = values[-1]
    return float(value) if "." in value else int(value)


def result_from(path: Path) -> dict[str, str]:
    lines = [line for line in path.read_text().splitlines()
             if line.startswith("RESULT stage=")]
    if len(lines) != 1:
        raise RuntimeError(f"{path} RESULT 数量错误")
    result = parse_result(lines[0])
    if result["status"] != "PASS" or result["mismatches"] != "0":
        raise RuntimeError(f"{path} 未通过 exact 检查")
    return result


def build_report(mapping_dir: Path, baseline_dir: Path, bsf_dir: Path) -> dict:
    rows = []
    for mode in ("baseline", "bsf"):
        stage_cycles = []
        bias_requests = 0
        for stage in range(4):
            if mode == "baseline":
                log = baseline_dir / f"w96_s{stage}" / "iverilog.log"
            else:
                log = bsf_dir / f"hatf96_s{stage}" / "iverilog.log"
            result = result_from(log)
            stage_cycles.append(int(result["total_cycles"]))
            bias_requests += int(result["bias_req_hs"])
        mapping = (mapping_dir / f"{mode}.log").read_text()
        rows.append({
            "mode": mode,
            "stage_cycles": stage_cycles,
            "total_cycles": sum(stage_cycles),
            "bias_requests": bias_requests,
            "external_bias_payload_bits": bias_requests * OUT_TILE * ACC_W,
            "logic_area": last_match(AREA_RE, mapping),
            "cells": last_match(CELLS_RE, mapping),
            "mem_v2": last_match(MEM_RE, mapping),
        })

    baseline, bsf = rows
    bsf["cycle_reduction"] = 1.0 - bsf["total_cycles"] / baseline["total_cycles"]
    bsf["speedup"] = baseline["total_cycles"] / bsf["total_cycles"]
    bsf["request_reduction"] = baseline["bias_requests"] / bsf["bias_requests"]
    bsf["resident_bits_per_supertile"] = OUT_TILE * ACC_W
    bsf["logic_area_delta"] = bsf["logic_area"] / baseline["logic_area"] - 1.0
    bsf["area_normalized_throughput"] = (
        bsf["speedup"] / (bsf["logic_area"] / baseline["logic_area"])
    )
    return {
        "schema_version": 1,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
        "cycle_evidence": "[rtl] sample0/B0/window0 S0-S3",
        "area_evidence": "[开放目标库逻辑映射代理]，memory面积未计",
        "rows": rows,
    }


def render_markdown(report: dict) -> str:
    baseline, bsf = report["rows"]
    lines = [
        "# BSF 偏置驻留终结器联合消融",
        "",
        "周期来自真实四阶段 RTL；面积来自 Nangate45 未约束逻辑映射。映射保留未映射 memory，不能替代 DC、STA、SRAM 宏和 SAIF 功耗结果。",
        "",
        "| 模式 | S0 | S1 | S2 | S3 | 总周期 | bias请求 | 外部bias载荷(bit) | logic area | cells | `$mem_v2` |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        stages = " | ".join(str(value) for value in row["stage_cycles"])
        lines.append(
            f"| {row['mode']} | {stages} | {row['total_cycles']} | "
            f"{row['bias_requests']} | {row['external_bias_payload_bits']} | "
            f"{row['logic_area']:.3f} | "
            f"{row['cells']} | {row['mem_v2']} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        f"- BSF 周期降低 {bsf['cycle_reduction'] * 100:.3f}%，加速 {bsf['speedup']:.3f}x；",
        f"- 偏置请求降低 {bsf['request_reduction']:.1f}x；",
        f"- 外部 bias SRAM payload 从 {baseline['external_bias_payload_bits']} bit 降到 {bsf['external_bias_payload_bits']} bit；每个 supertile 需驻留 {bsf['resident_bits_per_supertile']} bit；",
        f"- 开放库逻辑面积变化 {bsf['logic_area_delta'] * 100:+.3f}%，面积归一吞吐为 {bsf['area_normalized_throughput']:.3f}x；",
        "- 162 个 token 仍需在本地读取/广播驻留 bias，不能把外部请求降幅直接当作总能量降幅；",
        "- 若后续 SRAM+SAIF 不能证明偏置端口和读能量下降，则该机制只保留为微结构优化，不单列主贡献。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--baseline-cycle-dir", type=Path, required=True)
    parser.add_argument("--bsf-cycle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.mapping_dir, args.baseline_cycle_dir, args.bsf_cycle_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(render_markdown(report))
    print(json.dumps(report["rows"], ensure_ascii=False))


if __name__ == "__main__":
    main()
