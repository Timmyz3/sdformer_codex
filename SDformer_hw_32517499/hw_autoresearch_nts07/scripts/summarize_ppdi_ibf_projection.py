#!/usr/bin/env python3
"""汇总PPDI与IBF投影顶层的等边界RTL和开放库代理结果。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PASS_RE = re.compile(r"^PASS DCTF96 BANKLOCAL PROJECTION cycles=(\d+)", re.MULTILINE)
AREA_RE = re.compile(
    r"Chip area for module '\\gatestack_dctf96_banklocal_projection_top': ([0-9.]+)"
)
CELLS_RE = re.compile(r"Number of cells:\s+(\d+)")


def parse_cycles(path: Path) -> int:
    match = PASS_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"未在{path}中找到PASS周期")
    return int(match.group(1))


def parse_mapping(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    area = AREA_RE.search(text)
    cells = CELLS_RE.findall(text)
    if not area or not cells:
        raise ValueError(f"未在{path}中找到完整映射统计")
    return {"logic_area_proxy": float(area.group(1)), "mapped_cells": int(cells[-1])}


def build_report(build_dir: Path, ppdi_profile: Path) -> dict:
    modes = {
        "scalar_rmw": parse_cycles(build_dir / "scalar_rmw_iverilog.log"),
        "ppdi_rmw": parse_cycles(build_dir / "ppdi_rmw_iverilog.log"),
        "scalar_ibf": parse_cycles(build_dir / "scalar_ibf_iverilog.log"),
        "ppdi_ibf": parse_cycles(build_dir / "ppdi_ibf_iverilog.log"),
    }
    mappings = {
        name: parse_mapping(build_dir / f"map_{name}.log") for name in modes
    }
    ppdi = json.loads(ppdi_profile.read_text(encoding="utf-8"))["sample0_window0"]

    baseline_cycles = modes["scalar_rmw"]
    area_ratios = {
        name: row["logic_area_proxy"]
        / mappings["scalar_rmw"]["logic_area_proxy"]
        for name, row in mappings.items()
    }
    area_normalized_throughput = {
        name: (baseline_cycles / modes[name]) / area_ratios[name]
        for name in modes
    }
    return {
        "schema_version": 1,
        "architecture": "HIHP：分层不变量提升投影数据流",
        "scope": "T6定向投影顶层RTL与T162开放Nangate45逻辑映射代理",
        "status": "RTL_PASS_OPEN_LIB_PROXY_NOT_ASIC_PPA",
        "cycles": modes,
        "cycle_speedup": {
            "ppdi_only": baseline_cycles / modes["ppdi_rmw"],
            "ibf_only": baseline_cycles / modes["scalar_ibf"],
            "combined": baseline_cycles / modes["ppdi_ibf"],
        },
        "open_logic_mapping": {
            "modes": mappings,
            "area_ratios": area_ratios,
            "area_normalized_throughput": area_normalized_throughput,
            "library": "NangateOpenCellLibrary_typical",
            "memory_policy": "memory -nomap；不含真实SRAM宏面积",
        },
        "motion_sample0_window0_ppdi": ppdi,
        "analytical_bias_traffic": {
            str(tokens): {
                "scalar_bias_reads_per_three_bank_tile": 3 * tokens,
                "ibf_bias_reads_per_three_bank_tile": 3,
                "read_reduction": 1.0 - 1.0 / tokens,
            }
            for tokens in (6, 162, 450)
        },
        "contracts": [
            "PPDI每条term命令最多携带一个偶token和一个奇token",
            "同一term内gate、lane、weight和product不变",
            "IBF只适用于逐输出通道/输出tile固定、token间不变的线性偏置",
            "flush同拍送达adapter、fabric、executor与accumulator",
        ],
        "limits": [
            "周期来自T6定向功能向量，不能外推为真实trace端到端加速",
            "30.270%只来自Motion sample0/window0命令work，不是RTL周期",
            "旧profile100的M2已是无奇偶约束打包，不能再次乘PPDI比例",
            "开放库逻辑面积不含SRAM、布线、STA、时钟树和SAIF功耗",
            "Local5与Motion的fullres多样本精确PPDI统计仍待follower结果",
        ],
    }


def render_markdown(report: dict) -> str:
    c = report["cycles"]
    s = report["cycle_speedup"]
    m = report["open_logic_mapping"]
    p = report["motion_sample0_window0_ppdi"]
    b = report["analytical_bias_traffic"]
    return "\n".join(
        [
            "# PPDI+IBF投影顶层结果摘要",
            "",
            f"状态：`{report['status']}`。本报告严格区分定向RTL周期与开放库逻辑映射代理。",
            "",
            "## 等边界RTL消融",
            "",
            "| 配置 | 周期 | 相对标量RMW |",
            "|---|---:|---:|",
            f"| 标量+RMW | {c['scalar_rmw']} | 1.000x |",
            f"| PPDI+RMW | {c['ppdi_rmw']} | {s['ppdi_only']:.3f}x |",
            f"| 标量+IBF | {c['scalar_ibf']} | {s['ibf_only']:.3f}x |",
            f"| PPDI+IBF | {c['ppdi_ibf']} | {s['combined']:.3f}x |",
            "",
            "## 开放库逻辑映射代理",
            "",
            "| 配置 | 逻辑面积代理 | 面积倍率 | 面积归一吞吐 |",
            "|---|---:|---:|---:|",
            f"| 标量+RMW | {m['modes']['scalar_rmw']['logic_area_proxy']:.3f} | {m['area_ratios']['scalar_rmw']:.3f}x | {m['area_normalized_throughput']['scalar_rmw']:.3f}x |",
            f"| PPDI+RMW | {m['modes']['ppdi_rmw']['logic_area_proxy']:.3f} | {m['area_ratios']['ppdi_rmw']:.3f}x | {m['area_normalized_throughput']['ppdi_rmw']:.3f}x |",
            f"| 标量+IBF | {m['modes']['scalar_ibf']['logic_area_proxy']:.3f} | {m['area_ratios']['scalar_ibf']:.3f}x | {m['area_normalized_throughput']['scalar_ibf']:.3f}x |",
            f"| PPDI+IBF | {m['modes']['ppdi_ibf']['logic_area_proxy']:.3f} | {m['area_ratios']['ppdi_ibf']:.3f}x | {m['area_normalized_throughput']['ppdi_ibf']:.3f}x |",
            "",
            "面积归一吞吐混合了T6定向周期与T162开放逻辑映射，只用于趋势判断，不是ASIC PPA。",
            "",
            "## 真实workload边界",
            "",
            f"Motion sample0/window0中，标量命令{p['scalar_commands']}条，PPDI命令{p['ppdi_commands']}条，命令work降低{p['command_reduction'] * 100:.3f}%。该数字不能直接替代周期。",
            "",
            "## 偏置流量解析界限",
            "",
            "IBF把每个三bank输出tile的偏置读取从`3T`次降为3次，降低比例为`1-1/T`。偏置仍在最终输出逐token精确相加，不是删除运算。",
            "",
            "| Token数T | 标量偏置读取 | IBF偏置读取 | 降低 |",
            "|---:|---:|---:|---:|",
            *[
                f"| {tokens} | {row['scalar_bias_reads_per_three_bank_tile']} | {row['ibf_bias_reads_per_three_bank_tile']} | {row['read_reduction'] * 100:.3f}% |"
                for tokens, row in b.items()
            ],
            "",
            "## 限制",
            "",
            *[f"- {item}" for item in report["limits"]],
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--ppdi-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = build_report(args.build_dir, args.ppdi_profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
