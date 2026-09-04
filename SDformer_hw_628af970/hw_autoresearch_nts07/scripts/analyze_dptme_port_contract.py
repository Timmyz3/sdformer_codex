#!/usr/bin/env python3
"""分析DP-TME的T10/T2计算、输入银行和event出口联合下界。"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POSITIONS = 81
LANES = 32
T10 = 10
T2 = 2
INPUT_WIDTH_PROXY = 8
OUTPUT_WIDTHS = (32, 64, 128, 256, 320)


def packet_drain_cycles(pack_groups: int, output_width: int, positions: int = POSITIONS) -> int:
    total = 0
    for base in range(0, positions, pack_groups):
        valid_groups = min(pack_groups, positions - base)
        packet_bits = valid_groups * T2 * LANES
        total += math.ceil(packet_bits / output_width)
    return total


def analyze(positions: int = POSITIONS) -> dict:
    t10_cycles = positions * T10
    candidates = []
    for groups in range(1, 6):
        packs = math.ceil(positions / groups)
        compute_cycles = packs * T2
        valid_slot_utilization = positions / (packs * groups)
        output_drain = {
            str(width): packet_drain_cycles(groups, width, positions) for width in OUTPUT_WIDTHS
        }
        candidates.append(
            {
                "pack_groups": groups,
                "input_banks": groups,
                "input_read_bits_per_compute_cycle_proxy": groups * LANES * INPUT_WIDTH_PROXY,
                "full_array_physical_macs": T10 * LANES,
                "active_t2_macs": groups * T2 * LANES,
                "trimmed_array_physical_macs": groups * T2 * LANES,
                "trimmed_array_t10_cycles": t10_cycles * math.ceil(T10 / (groups * T2)),
                "active_array_fraction_in_t2": groups / 5,
                "valid_slot_utilization_with_tail": valid_slot_utilization,
                "compute_cycles": compute_cycles,
                "output_drain_cycles": output_drain,
                "system_cycle_lower_bound": {
                    width: max(compute_cycles, drain) for width, drain in output_drain.items()
                },
            }
        )

    return {
        "geometry": {
            "positions": positions,
            "lanes": LANES,
            "T10": T10,
            "T2": T2,
            "input_width_proxy": INPUT_WIDTH_PROXY,
        },
        "T10": {
            "compute_cycles": t10_cycles,
            "input_read_bits_per_cycle_proxy": LANES * INPUT_WIDTH_PROXY,
            "average_event_bits_per_cycle": positions * T10 * LANES / t10_cycles,
        },
        "T2_candidates": candidates,
        "assumptions": [
            "每个T2打包组有一个可独立读取的32-lane输入银行。",
            "event packet先进入ping-pong缓冲，计算与出口排空可重叠；系统周期取两者下界的最大值。",
            "未计descriptor、bank冲突、跨层装载、写回和下游反压，因此结果仍是乐观下界。",
            "8-bit输入只用于端口宽度代理，不表示ATLIF量化已经冻结。",
        ],
    }


def write_markdown(result: dict, path: Path) -> None:
    positions = int(result["geometry"]["positions"])
    five_way = result["T2_candidates"][4]
    full_event_bits = positions * T2 * LANES
    lines = [
        "# DP-TME端口感知打包因子分析",
        "",
        f"**结论**：{positions}位置下五路T2的{five_way['compute_cycles']}拍结果只在五组输入可并行读取、且event出口持续带宽足够时成立。单32-bit出口下界为{five_way['system_cycle_lower_bound']['32']}拍。",
        "",
        "## T10基线",
        "",
        f"T10对{positions}个位置需要`{result['T10']['compute_cycles']}`拍；每拍只读一个32-lane输入向量，8-bit代理宽度为`{result['T10']['input_read_bits_per_cycle_proxy']}` bit，平均产生32 bit event/拍。",
        "",
        "## T2候选",
        "",
        "| pack组数 | T2活跃MAC/完整MAC | 裁剪阵列T10拍 | 8-bit输入读宽代理 | T2计算拍 | 尾组有效率 | 32b出口下界 | 64b | 128b | 256b |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["T2_candidates"]:
        lb = row["system_cycle_lower_bound"]
        lines.append(
            f"| {row['pack_groups']} | {row['active_t2_macs']}/{row['full_array_physical_macs']} | "
            f"{row['trimmed_array_t10_cycles']} | {row['input_read_bits_per_compute_cycle_proxy']}b | "
            f"{row['compute_cycles']} | {row['valid_slot_utilization_with_tail']:.2%} | "
            f"{lb['32']} | {lb['64']} | {lb['128']} | {lb['256']} |"
        )
    lines.extend(
        [
            "",
            "## 架构含义",
            "",
            f"- `G=5`需要5个独立32-lane输入银行；8-bit代理下峰值读宽为1280 bit/拍。256-bit event出口配ping-pong packet buffer时，乐观下界为{five_way['system_cycle_lower_bound']['256']}拍。",
            f"- `G=4 + 128-bit出口`在该几何下为{result['T2_candidates'][3]['system_cycle_lower_bound']['128']}拍候选，保留完整320-MAC阵列以维持T10吞吐，但T2只激活256个MAC。",
            f"- `G=3 + 128-bit出口`在该几何下为{result['T2_candidates'][2]['system_cycle_lower_bound']['128']}拍候选，T2激活192个MAC；其余128个MAC只做时钟门控。",
            f"- 若把物理阵列裁到`2G×32`个MAC，G3/G4、G2、G1的T10分别为{result['T2_candidates'][2]['trimmed_array_t10_cycles']}/{result['T2_candidates'][3]['trimmed_array_t10_cycles']}、{result['T2_candidates'][1]['trimmed_array_t10_cycles']}、{result['T2_candidates'][0]['trimmed_array_t10_cycles']}拍，必须放入全encoder DSE。",
            f"- 单32-bit Router无论G取多大都受{full_event_bits}个有效event bit排空约束，G=5下界为{five_way['system_cycle_lower_bound']['32']}拍。",
            "- Router RTL是单word正确性切片；完整系统应在其前面增加参数化packet buffer和多word/多lane适配器，而不是实例化320套独立控制器。",
            "",
            "这些数值不含SRAM冲突、跨层装载和下游反压，属于架构准入下界，不是最终吞吐结果。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = analyze()
    json_path = ROOT / "results/dptme_port_contract.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, json_path.with_suffix(".md"))
    print(json_path.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
