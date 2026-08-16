#!/usr/bin/env python3
"""统计DCTF奇偶配对destination发射的精确work上界。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_hex_lines(path: Path) -> list[int]:
    return [int(item, 16) for item in path.read_text(encoding="utf-8").split()]


def profile_stage(stage_dir: Path) -> dict:
    manifest = json.loads((stage_dir / "manifest.json").read_text(
        encoding="utf-8"
    ))
    counts = read_hex_lines(stage_dir / "term_destination_counts.memh")
    offsets = read_hex_lines(stage_dir / "term_token_offsets.memh")
    tokens = read_hex_lines(stage_dir / "term_tokens.memh")
    if len(offsets) != len(counts) + 1:
        raise RuntimeError(f"term offset数量错误: {stage_dir}")
    destinations = 0
    paired_commands = 0
    even_destinations = 0
    odd_destinations = 0
    max_parity_imbalance = 0
    for index, count in enumerate(counts):
        term_tokens = tokens[offsets[index]:offsets[index] + count]
        if len(term_tokens) != count or offsets[index + 1] != offsets[index] + count:
            raise RuntimeError(f"term token边界错误: {stage_dir} term={index}")
        even = sum((token & 1) == 0 for token in term_tokens)
        odd = count - even
        destinations += count
        even_destinations += even
        odd_destinations += odd
        paired_commands += max(even, odd)
        max_parity_imbalance = max(max_parity_imbalance, abs(even - odd))
    multiplier = manifest["logical_supertiles"]
    return {
        "stage": manifest["stage"],
        "base_terms": len(counts),
        "replayed_terms": len(counts) * multiplier,
        "destinations": destinations * multiplier,
        "even_destinations": even_destinations * multiplier,
        "odd_destinations": odd_destinations * multiplier,
        "scalar_commands": destinations * multiplier,
        "ppdi_commands": paired_commands * multiplier,
        "command_reduction": (
            0.0 if destinations == 0 else 1.0 - paired_commands / destinations
        ),
        "max_term_parity_imbalance": max_parity_imbalance,
    }


def build_report(vector_root: Path, profile100_path: Path) -> dict:
    stages = [profile_stage(vector_root / f"s{stage}") for stage in range(4)]
    total_scalar = sum(row["scalar_commands"] for row in stages)
    total_ppdi = sum(row["ppdi_commands"] for row in stages)
    profile100 = json.loads(profile100_path.read_text(encoding="utf-8"))
    h67 = profile100["models"]["H67"]
    pair = h67["binary_temporal_pairs"]
    profile100_destinations = pair["projection_gate_group_active_lanes_g1"]
    profile100_terms = pair["projection_gate_group_terms_g1"]
    profile100_m2 = pair["projection_gate_multicast_delivery_m2"]
    return {
        "schema_version": 1,
        "status": "PROFILE_ONLY_NOT_RTL",
        "mechanism": "每条命令最多携带一个偶token和一个奇token，同拍使用Acc偶/奇两路",
        "stages": stages,
        "sample0_window0": {
            "scalar_commands": total_scalar,
            "ppdi_commands": total_ppdi,
            "command_reduction": 1.0 - total_ppdi / total_scalar,
        },
        "profile100_context": {
            "samples": h67["samples"],
            "terms_g1": profile100_terms,
            "destinations": profile100_destinations,
            "destinations_per_term": profile100_destinations / profile100_terms,
            "unconstrained_m2_deliveries": profile100_m2,
            "unconstrained_m2_reduction": 1.0 - profile100_m2 / profile100_destinations,
            "warning": "profile100的M2未带奇偶约束，只是PPDI的乐观参照，不是PPDI预测",
        },
        "limits": [
            "只统计命令work，不是RTL周期、吞吐或能量",
            "sample0/window0的PPDI计数使用真实token奇偶分布",
            "profile100只有无奇偶约束M2聚合，不能替代多样本PPDI统计",
            "需要双destination fabric、bank executor双Acc同拍和bit-exact RTL后才能晋级",
            "无目标库面积、时序、SAIF或SRAM端口代价",
        ],
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# PPDI奇偶配对Destination发射Workload统计",
        "",
        "PPDI利用现有每bank偶/奇两路Acc端口，在同一term内每拍最多携带一个偶token和一个奇token。两者共享同一gate、weight和product，不改变任一token的整数累加值。",
        "",
        "| Stage | replay term | destination | 标量命令 | PPDI命令 | 命令work降低 | 最大term奇偶失衡 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["stages"]:
        lines.append(
            f"| S{row['stage']} | {row['replayed_terms']} | "
            f"{row['destinations']} | {row['scalar_commands']} | "
            f"{row['ppdi_commands']} | {row['command_reduction'] * 100:.3f}% | "
            f"{row['max_term_parity_imbalance']} |"
        )
    sample = report["sample0_window0"]
    prof = report["profile100_context"]
    lines += [
        "", "## 结果", "",
        f"- 真实sample0/window0四stage的destination命令由{sample['scalar_commands']}降至{sample['ppdi_commands']}，理论work降低{sample['command_reduction'] * 100:.3f}%；",
        f"- profile100共有{prof['terms_g1']}个G1 term和{prof['destinations']}个destination，平均fanout为{prof['destinations_per_term']:.3f}；",
        f"- profile100无奇偶约束M2 delivery为{prof['unconstrained_m2_deliveries']}，相对标量降低{prof['unconstrained_m2_reduction'] * 100:.3f}%，只作为乐观上界；",
        "- PPDI若实现，可同时减少adapter发射、fabric命令和executor destination调度，并把原先闲置的另一奇偶Acc端口用于同term product；",
        "", "## 证据边界", "",
        *[f"- {item}；" for item in report["limits"]],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--profile100", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.vector_root, args.profile100)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "统计结果.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "统计报告.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
