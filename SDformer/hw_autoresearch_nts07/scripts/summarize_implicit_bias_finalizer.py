#!/usr/bin/env python3
"""汇总IBF叶映射与Motion ordered周期，生成中文证据报告。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = ROOT / "build_hitflow/implicit_bias_finalizer"
DEFAULT_MOTION = ROOT / "results/motion_ecgb_ordered_profile100_20260801/report.json"
DEFAULT_OUT = ROOT / "results/implicit_bias_finalizer_20260801"


def parse_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    area_matches = re.findall(r"Chip area for module .*?:\s*([0-9.]+)", text)
    cells_matches = re.findall(r"Number of cells:\s+(\d+)", text)
    dff_matches = re.findall(r"^\s+DFF_X1\s+(\d+)\s*$", text, re.MULTILINE)
    mem_matches = re.findall(r"^\s+\$mem_v2\s+(\d+)\s*$", text, re.MULTILINE)
    read_ports = len(re.findall(r"Checking read port .*acc_mem", text))
    if not (area_matches and cells_matches and dff_matches and mem_matches):
        raise ValueError(f"无法解析映射日志: {path}")
    return {
        "path": str(path.resolve()),
        "logic_area": float(area_matches[-1]),
        "cells": int(cells_matches[-1]),
        "dff_x1": int(dff_matches[-1]),
        "mem_v2": int(mem_matches[-1]),
        "acc_memory_read_ports_total": read_ports,
    }


def build_report(build_dir: Path, motion_path: Path) -> dict[str, Any]:
    current = parse_mapping(build_dir / "current_nangate45.log")
    ibf = parse_mapping(build_dir / "ibf_nangate45.log")
    motion = json.loads(motion_path.read_text(encoding="utf-8"))
    area_ratio = ibf["logic_area"] / current["logic_area"]
    current_tail = 162 + 2
    ibf_tail = (162 + 1) // 2 + 3
    tail_speedup = current_tail / ibf_tail
    selected = []
    for group in motion["groups"]:
        if group["group_windows"] in (1, 4, 8, 16):
            selected.append(
                {
                    "group_windows": group["group_windows"],
                    "ibf_cycles": group["finite_cycles_by_finalizer"][
                        "ibf_pipelined"
                    ],
                    "speedup_vs_current_g1": group[
                        "ibf_speedup_vs_current_g1"
                    ],
                    "stage0_payload_p99_bits": group["stage_payload_bits"][0][
                        "p99"
                    ],
                }
            )
    return {
        "schema": "implicit_bias_finalizer_summary_v1",
        "evidence": {
            "function": "Icarus + Verilator动态SVA",
            "structure": "Yosys单读口memory检查",
            "area": "Nangate45无约束逻辑映射代理，memory面积未计",
            "cycles": "H67 crop/W9 ordered profile100 + finite pipeline model",
        },
        "mapping": {"current_rmw": current, "ibf": ibf},
        "area_ratio": area_ratio,
        "area_delta": area_ratio - 1.0,
        "tail_cycles": {"current_rmw": current_tail, "ibf": ibf_tail},
        "tail_speedup": tail_speedup,
        "tail_area_normalized_throughput": tail_speedup / area_ratio,
        "motion_points": selected,
        "freeze": {
            "implemented": "G1 + IBF叶RTL（尚未接完整projection top）",
            "capacity_point": "G4 + IBF",
            "performance_point": "G8 + IBF",
            "eliminated": "G16：相对G8周期收益不足0.1%，S0 p99目录约2.08倍",
        },
    }


def markdown(report: dict[str, Any]) -> str:
    current = report["mapping"]["current_rmw"]
    ibf = report["mapping"]["ibf"]
    lines = [
        "# IBF隐式偏置终结器：单读口RTL与Motion瓶颈解锁",
        "",
        "> 日期：2026-08-01  ",
        "> 证据等级：功能为`[rtl]`；面积为`[开放逻辑映射代理]`；完整周期为",
        "> `[prof-ordered]+[bounded-model]`。不是DC、STA、SAIF或fullres结果。",
        "",
        "## 1. 结论",
        "",
        "IBF把accumulator状态从`product+bias已物化`改成`product-only`，bias只在",
        "final退休路径相加，不再写回acc SRAM。两个bank各自流水读出，保留逐token",
        "final语义，但把T=162尾部从模型164拍降为84拍。",
        "",
        f"优化后的IBF逻辑面积代理相对现有RMW叶增加{report['area_delta']:.2%}，"
        f"tail吞吐为{report['tail_speedup']:.3f}x，叶级面积归一tail吞吐为"
        f"{report['tail_area_normalized_throughput']:.3f}x。",
        "",
        "## 2. 结构映射",
        "",
        "| 方案 | logic area | cells | DFF_X1 | `$mem_v2` | acc memory总读口 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 现有bias RMW | {current['logic_area']:.3f} | {current['cells']} | "
        f"{current['dff_x1']} | {current['mem_v2']} | "
        f"{current['acc_memory_read_ports_total']} |",
        f"| IBF单读口 | {ibf['logic_area']:.3f} | {ibf['cells']} | "
        f"{ibf['dff_x1']} | {ibf['mem_v2']} | "
        f"{ibf['acc_memory_read_ports_total']} |",
        "",
        "首版IBF曾因重复的`shared_read_data/final_value`全宽寄存器使面积增加约40.8%；",
        "该版本已淘汰。当前版本直接以共享读口寄存器驱动final valid/ready，且Yosys",
        "确认每个bank只推断一个acc memory读口。",
        "",
        "## 3. Motion ordered profile100",
        "",
        "| G | IBF有限周期 | 相对现有G1 | S0目录p99(bit，单buffer) |",
        "|---:|---:|---:|---:|",
    ]
    for row in report["motion_points"]:
        lines.append(
            f"| {row['group_windows']} | {row['ibf_cycles']} | "
            f"{row['speedup_vs_current_g1']:.4f}x | "
            f"{row['stage0_payload_p99_bits']:.0f} |"
        )
    lines += [
        "",
        "冻结结论：",
        "",
        "1. `G1+IBF`是已具备叶RTL的最小候选，但尚未接入完整projection top；",
        "2. `G4+IBF`是容量保守点，`G8+IBF`是性能点；",
        "3. G8相对G4只多约2.2%端到端模型收益，却让S0 p99目录约翻倍，是否选择G8",
        "必须由fullres T450容量和目标SRAM宏决定；",
        "4. G16相对G8收益不足0.1%，目录再约翻倍，正式淘汰；",
        "5. 本结果支持IBF作为ECGB/DVCO的瓶颈解锁机制，不支持把IBF单独包装成",
        "DATE主创新。",
        "",
        "## 4. exact合同与验证",
        "",
        "- acc SRAM只保存INT32 product sum；",
        "- 未触达token以0作为product sum，final仍输出bias；",
        "- bias加法位宽、二补码溢出判断与现有RMW保持一致；",
        "- final-add溢出时抑制该token输出、置sticky overflow，但不阻塞组完成；",
        "- final反压冻结共享读口数据、token和tag；",
        "- T=8幂次边界、双bank同拍退休、反压和溢出路径均已通过动态测试。",
        "",
        "## 5. DATE阶段评审",
        "",
        "本轮把Motion中被固定bias尾部遮蔽的ECGB收益转化为可实现的单读口数据流，",
        "并用负映射结果驱动了40.8%到12.9%的结构收敛。它提升了架构完整度和证据",
        "可信度，但新颖性仍来自`gate vocabulary批处理 + product-only context + exact",
        "retirement`的组合，IBF自身属于常见的延迟物化思想。",
        "",
        "当前仍不能达到DATE accept口径：缺完整top周期对照、fullres T450 profile、",
        "SRAM宏、DC/STA/SAIF和多trace p95/p99。下一门槛是把IBF接入同一projection",
        "top，与现有RMW在完全相同term流和反压下做逐token等价及周期对照。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = build_report(args.build_dir, args.motion)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
