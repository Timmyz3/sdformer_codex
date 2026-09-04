#!/usr/bin/env python3
"""汇总IBF完整单头projection A/B结果。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from .summarize_implicit_bias_finalizer import parse_mapping
except ImportError:
    from summarize_implicit_bias_finalizer import parse_mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "build_hitflow/gatestack_single_head_projection"
DEFAULT_IBF = ROOT / "build_hitflow/gatestack_single_head_ibf"
DEFAULT_MOTION = ROOT / "results/motion_ecgb_ordered_profile100_20260801/report.json"
DEFAULT_OUT = ROOT / "results/single_head_ibf_integration_20260801"


def parse_cycles(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"cycles=(\d+)", text)
    if not matches:
        raise ValueError(f"找不到cycles: {path}")
    return int(matches[-1])


def build_report(base_dir: Path, ibf_dir: Path, motion_path: Path) -> dict[str, Any]:
    current = parse_mapping(ibf_dir / "current_nangate45.log")
    ibf = parse_mapping(ibf_dir / "ibf_nangate45.log")
    cycles = {
        "icarus": {
            "current": parse_cycles(base_dir / "iverilog.log"),
            "ibf": parse_cycles(ibf_dir / "iverilog.log"),
        },
        "verilator": {
            "current": parse_cycles(base_dir / "verilator_assert.log"),
            "ibf": parse_cycles(ibf_dir / "verilator.log"),
        },
    }
    for row in cycles.values():
        row["speedup"] = row["current"] / row["ibf"]
    area_ratio = ibf["logic_area"] / current["logic_area"]
    motion = json.loads(motion_path.read_text(encoding="utf-8"))
    ordered = []
    for row in motion["groups"]:
        if row["group_windows"] in (1, 4, 8):
            speedup = row["ibf_speedup_vs_current_g1"]
            ordered.append(
                {
                    "group_windows": row["group_windows"],
                    "speedup_vs_current_g1": speedup,
                    "area_normalized_throughput_proxy": speedup / area_ratio,
                }
            )
    return {
        "schema": "single_head_ibf_integration_v1",
        "evidence": {
            "exact_and_cycles": "同一T=8定向term/event trace，Icarus+Verilator动态SVA",
            "area": "同顶层Nangate45无约束逻辑映射代理，memory面积未计",
            "ordered": "H67 crop/W9 ordered profile100有限模型",
        },
        "cycles": cycles,
        "mapping": {"current": current, "ibf": ibf},
        "area_ratio": area_ratio,
        "area_delta": area_ratio - 1.0,
        "verilator_area_normalized_throughput": (
            cycles["verilator"]["speedup"] / area_ratio
        ),
        "ordered_motion_proxy": ordered,
        "status": "PASS_WITH_EXPLICIT_LIMITS",
    }


def markdown(report: dict[str, Any]) -> str:
    current = report["mapping"]["current"]
    ibf = report["mapping"]["ibf"]
    lines = [
        "# IBF完整单头Projection集成与等边界A/B",
        "",
        "> 日期：2026-08-01  ",
        "> 功能证据：`[rtl]`；面积：`[开放逻辑映射代理]`；ordered外推：",
        "> `[prof-ordered]+[bounded-model]`。无DC、STA、SAIF与SRAM宏。",
        "",
        "## 1. 集成数据流",
        "",
        "```text",
        "term/event -> TDR backend -> product multicast -> product-only Acc SRAM",
        "                                                     |",
        "一次bias响应 -----------------------------------------+",
        "                                                     v",
        "                              双bank单读口IBF final drain -> final",
        "```",
        "",
        "term解码、weight请求、product、多播和外部final接口与现有顶层相同。IBF",
        "只替换backend drain后的bias物化方式，默认参数仍保留原RMW路径。",
        "",
        "## 2. 同trace RTL结果",
        "",
        "测试包含错误bias响应、至少一拍响应延迟、final反压、未触达token和INT32",
        "逐token金参考。错误响应也计一次请求，因此请求数是`9->2`；无错误时对应",
        "`8->1`。",
        "",
        "| 仿真器 | 现有RMW周期 | IBF周期 | 加速 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["cycles"].items():
        lines.append(
            f"| {name} | {row['current']} | {row['ibf']} | "
            f"{row['speedup']:.3f}x |"
        )
    lines += [
        "",
        "T=8定向向量中bias尾部占比很高，不能把约1.66x直接外推到真实网络。",
        "",
        "## 3. 同顶层结构映射",
        "",
        "| 方案 | logic area | cells | DFF_X1 | `$mem_v2` |",
        "|---|---:|---:|---:|---:|",
        f"| 现有RMW | {current['logic_area']:.3f} | {current['cells']} | "
        f"{current['dff_x1']} | {current['mem_v2']} |",
        f"| IBF | {ibf['logic_area']:.3f} | {ibf['cells']} | "
        f"{ibf['dff_x1']} | {ibf['mem_v2']} |",
        "",
        f"IBF逻辑面积代理增加{report['area_delta']:.2%}；在T=8 Verilator周期上，"
        f"面积归一吞吐代理为{report['verilator_area_normalized_throughput']:.3f}x。",
        "memory面积未计，因此该值只能证明结构趋势。",
        "",
        "## 4. Motion ordered口径",
        "",
        "| G | 相对现有G1加速 | 除以同顶层面积比后的吞吐代理 |",
        "|---:|---:|---:|",
    ]
    for row in report["ordered_motion_proxy"]:
        lines.append(
            f"| {row['group_windows']} | "
            f"{row['speedup_vs_current_g1']:.4f}x | "
            f"{row['area_normalized_throughput_proxy']:.4f}x |"
        )
    lines += [
        "",
        "`G1+IBF`已不再亏面积归一吞吐；G4/G8进一步受益，但其目录SRAM增长尚未",
        "计入上述面积比，所以论文不能直接使用最后一列作为PPA主结果。",
        "",
        "## 5. 双线适用性",
        "",
        "IBF位于共享projection backend，既不依赖Motion-XOR，也不依赖Local5的",
        "局部关系分类。只要两条线继续满足“同一output tile的bias对所有token相同”",
        "和INT32最终相加合同，Motion与Local5都可复用该终结器。Local5是否叠加",
        "ECGB仍需post-G0真实trace决定。",
        "",
        "## 6. DATE阶段复审",
        "",
        "内部严格复审结论仍是`Borderline Reject`，不是accept：",
        "",
        "- 加分：从叶算子推进到完整term-to-final子系统；同边界bit-exact、错误路径、",
        "反压和映射均闭环；",
        "- 加分：负面积结果驱动单读口与寄存器去重，完整顶层面积增量降至4.48%；",
        "- 扣分：IBF是延迟物化/流水退休的本土化组合，本身不是强独立创新；",
        "- 扣分：真实主收益仍是模型，fullres T450、Local5 post-G0、SRAM、DC/STA/",
        "SAIF和多trace尾延迟尚未闭合；",
        "- 扣分：G4/G8目录容量尚未进入同顶层物理面积，无法据此冻结最终G。",
        "",
        "下一轮应优先补fullres trace和目录容量，而不是继续给IBF堆控制功能。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--ibf-dir", type=Path, default=DEFAULT_IBF)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = build_report(args.base_dir, args.ibf_dir, args.motion)
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
