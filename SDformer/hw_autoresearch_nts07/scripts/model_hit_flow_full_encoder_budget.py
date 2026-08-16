#!/usr/bin/env python3
"""建立HIT-Flow全encoder的周期、存储事务和吞吐上下界模型。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CLOCK_HZ = 500_000_000
DEFAULT_TARGET_FPS = 30.0
DEFAULT_CONTROL_MARGIN = 1.25


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def find_model(profile: dict[str, Any], model: str) -> dict[str, Any]:
    for row in profile["results"]:
        if row["model"] == model:
            return row
    raise KeyError(model)


def build_model(
    storage: dict[str, Any],
    profile: dict[str, Any],
    sops: dict[str, Any],
    runtime_profile: dict[str, Any] | None = None,
    *,
    clock_hz: int = DEFAULT_CLOCK_HZ,
    target_fps: float = DEFAULT_TARGET_FPS,
    control_margin: float = DEFAULT_CONTROL_MARGIN,
) -> dict[str, Any]:
    h67_storage = storage["models"]["H67"]
    graph = h67_storage["atlif_execution_graph"]
    long_skip_elements = int(h67_storage["activation_evidence"]["long_skip_elements_s0_s2"])
    temporal_macs = int(graph["live_temporal_macs_per_frame"])
    event_elements = int(graph["live_output_elements_per_frame"])
    event_ops_proxy = int(round(float(sops["estimated_total_sops"])))
    dense_ops_proxy = int(round(float(sops["dense_ops"])))
    spatial_proxy_source = "旧全网dense_ops乘全局firing"
    if runtime_profile is not None:
        samples = int(runtime_profile.get("samples", 0))
        scope_rows = runtime_profile.get("summary", {}).get("operator_by_scope", [])
        encoder_row = next((row for row in scope_rows if row.get("scope") == "encoder"), None)
        if samples <= 0 or encoder_row is None:
            raise ValueError("runtime profile缺少samples或encoder逐算子分账")
        event_ops_proxy = int(round(float(encoder_row["activity_weighted_macs_proxy"]) / samples))
        dense_ops_proxy = int(round(float(encoder_row["dense_macs"]) / samples))
        spatial_proxy_source = "H67 ordered profile100逐算子encoder活动率加权MAC"
    budget_cycles = int(clock_hz / target_fps)
    h67_profile = find_model(profile, "H67")["whole"]["port_aware_pipeline_dse"]

    configurations: list[dict[str, Any]] = []
    for temporal_arrays in (2, 4):
        temporal_cycles = ceil_div(temporal_macs, temporal_arrays * 320)
        for spatial_lanes in (256, 512, 1024):
            spatial_cycles = ceil_div(event_ops_proxy, spatial_lanes)
            for contexts in (2, 4):
                for pccc_mode, key_part in (
                    ("关闭", "no_merge"),
                    ("理想全合并上界", "perfect_pccc"),
                ):
                    attention_key = f"fetch128_split_1w_{key_part}_contexts{contexts}"
                    attention_cycles = int(math.ceil(float(h67_profile[attention_key]["mean"])))
                    for memory_bus_bits in (256, 512):
                        skip_cycles = ceil_div(long_skip_elements * 8 * 2, memory_bus_bits)
                        for bypass_ratio in (0.0, 0.5, 0.75, 1.0):
                            materialized_event_bits = int(round(event_elements * 2 * (1.0 - bypass_ratio)))
                            event_bank_cycles = ceil_div(materialized_event_bits, memory_bus_bits)
                            serial_cycles = (
                                temporal_cycles
                                + spatial_cycles
                                + attention_cycles
                                + skip_cycles
                                + event_bank_cycles
                            )
                            overlap_lower_bound = max(
                                temporal_cycles,
                                spatial_cycles,
                                attention_cycles,
                                skip_cycles + event_bank_cycles,
                            )
                            guarded_serial_cycles = int(math.ceil(serial_cycles * control_margin))
                            configurations.append({
                                "temporal_arrays": temporal_arrays,
                                "spatial_lanes": spatial_lanes,
                                "contexts": contexts,
                                "pccc_mode": pccc_mode,
                                "memory_bus_bits": memory_bus_bits,
                                "event_bypass_ratio": bypass_ratio,
                                "temporal_cycles": temporal_cycles,
                                "spatial_proxy_cycles": spatial_cycles,
                                "attention_cycles": attention_cycles,
                                "skip_rw_cycles_8bit": skip_cycles,
                                "event_bank_rw_cycles": event_bank_cycles,
                                "serial_cycles": serial_cycles,
                                "perfect_overlap_lower_bound_cycles": overlap_lower_bound,
                                "guarded_serial_cycles": guarded_serial_cycles,
                                "guarded_serial_fps": clock_hz / guarded_serial_cycles,
                                "passes_30fps_guarded_serial": guarded_serial_cycles <= budget_cycles,
                            })

    configurations.sort(
        key=lambda row: (
            not row["passes_30fps_guarded_serial"],
            row["temporal_arrays"],
            row["spatial_lanes"],
            row["memory_bus_bits"],
            row["contexts"],
            row["pccc_mode"],
            -row["event_bypass_ratio"],
        )
    )
    return {
        "schema_version": 1,
        "scope": {
            "clock_hz": clock_hz,
            "target_fps": target_fps,
            "cycles_per_frame_budget": budget_cycles,
            "control_and_unmodeled_margin": control_margin,
            "warning": (
                "空间负载使用旧全网dense_ops乘全局firing得到的event-operation代理；"
                "它不是逐层encoder SOP，也不包含真实SRAM权重冲突。"
            ),
        },
        "inputs": {
            "live_atlif_temporal_macs_per_frame": temporal_macs,
            "live_atlif_event_elements_per_frame": event_elements,
            "long_skip_elements_s0_s2": long_skip_elements,
            "legacy_whole_network_event_ops_proxy": event_ops_proxy,
            "legacy_dense_ops_proxy": dense_ops_proxy,
            "spatial_proxy_source": spatial_proxy_source,
        },
        "traffic_at_30fps": {
            "packed_event_bank_full_materialization_GBps": event_elements * 2 * target_fps / 8 / 1e9,
            "long_skip_rw_4bit_GBps": long_skip_elements * 4 * 2 * target_fps / 8 / 1e9,
            "long_skip_rw_8bit_GBps": long_skip_elements * 8 * 2 * target_fps / 8 / 1e9,
            "long_skip_rw_16bit_GBps": long_skip_elements * 16 * 2 * target_fps / 8 / 1e9,
        },
        "configurations": configurations,
    }


def shortlist(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result["configurations"]
    selected = []
    signatures = (
        (2, 256, 2, 256, 0.75, "关闭"),
        (2, 512, 2, 256, 0.00, "关闭"),
        (2, 512, 2, 256, 0.75, "关闭"),
        (2, 512, 2, 512, 0.75, "关闭"),
        (2, 512, 4, 512, 0.75, "理想全合并上界"),
        (4, 512, 2, 256, 0.00, "关闭"),
        (4, 512, 2, 256, 0.75, "关闭"),
        (4, 512, 2, 512, 0.75, "关闭"),
        (4, 1024, 4, 512, 0.75, "理想全合并上界"),
    )
    for signature in signatures:
        for row in rows:
            candidate = (
                row["temporal_arrays"],
                row["spatial_lanes"],
                row["contexts"],
                row["memory_bus_bits"],
                row["event_bypass_ratio"],
                row["pccc_mode"],
            )
            if candidate == signature:
                selected.append(row)
                break
    return selected


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    inputs = result["inputs"]
    traffic = result["traffic_at_30fps"]
    scope = result["scope"]
    lines = [
        "# HIT-Flow全Encoder周期与事务预算模型",
        "",
        "**状态**：架构DSE代理，不是RTL或DC结果  ",
        f"**探索约束**：{scope['clock_hz']/1e6:.0f} MHz、{scope['target_fps']:.0f} FPS、每帧{scope['cycles_per_frame_budget']:,}拍、{scope['control_and_unmodeled_margin']:.2f}倍保护系数",
        "",
        "## 1. 输入口径",
        "",
        f"- 固定部署活跃ATLIF时间MAC：`{inputs['live_atlif_temporal_macs_per_frame']:,}`/帧；",
        f"- ATLIF二值输出元素：`{inputs['live_atlif_event_elements_per_frame']:,}`/帧；",
        f"- S0-S2长skip：`{inputs['long_skip_elements_s0_s2']:,}`元素；",
        f"- 空间event-operation代理：`{inputs['legacy_whole_network_event_ops_proxy']:,}`/帧；",
        f"- 空间dense代理：`{inputs['legacy_dense_ops_proxy']:,}`/帧；",
        f"- 空间负载来源：{inputs['spatial_proxy_source']}。",
        "",
        "活动率加权MAC对Linear较接近连通度代理，对带padding/stride的卷积仍不是精确SOP。它只用于空间lane数量预筛选；所有结果均标为模型预测。",
        "",
        "## 2. 30 FPS下最低事务带宽",
        "",
        "| 事务 | 带宽 |",
        "|---|---:|",
        f"| 所有ATLIF event写回再读出，1-bit打包 | {traffic['packed_event_bank_full_materialization_GBps']:.3f} GB/s |",
        f"| S0-S2 skip读写，4-bit | {traffic['long_skip_rw_4bit_GBps']:.3f} GB/s |",
        f"| S0-S2 skip读写，8-bit | {traffic['long_skip_rw_8bit_GBps']:.3f} GB/s |",
        f"| S0-S2 skip读写，16-bit | {traffic['long_skip_rw_16bit_GBps']:.3f} GB/s |",
        "",
        "event带宽看似不大，但256-bit单端口下全物化需要约411万拍/帧，和双DP-TME的约691万拍同量级。HTT局部驻留与producer-consumer bypass因此是系统架构变量，不是接口细节。",
        "",
        "## 3. 代表性配置",
        "",
        "串行模型将时间矩阵、空间代理、attention、8-bit skip读写和event-bank事务相加，再乘1.25保护系数。perfect-overlap仅给出不可达到的乐观下界。",
        "",
        "| DP阵列 | 空间lane | ctx | 总线 | PCCC | event bypass | 时间M拍 | 空间M拍 | attention M拍 | event M拍 | 保护后FPS | 30 FPS |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in shortlist(result):
        lines.append(
            f"| {row['temporal_arrays']} | {row['spatial_lanes']} | {row['contexts']} | "
            f"{row['memory_bus_bits']}b | {row['pccc_mode']} | {row['event_bypass_ratio']:.0%} | "
            f"{row['temporal_cycles']/1e6:.3f} | {row['spatial_proxy_cycles']/1e6:.3f} | "
            f"{row['attention_cycles']/1e6:.3f} | {row['event_bank_rw_cycles']/1e6:.3f} | "
            f"{row['guarded_serial_fps']:.2f} | "
            f"{'通过' if row['passes_30fps_guarded_serial'] else '不通过'} |"
        )
    lines += [
        "",
        "## 4. 架构指导",
        "",
        "1. `2×DP-TME + 256 spatial lanes`在该代理下空间计算本身已超过30 FPS预算，不进入平衡主候选。",
        "2. `2×DP-TME + 512 spatial lanes`只有在较高HTT bypass、较宽总线或更强阶段重叠下才接近30 FPS，属于面积优先边界点。",
        "3. `4×DP-TME + 512 spatial lanes`对event物化和未建模控制更稳健，作为吞吐主候选；是否值得其面积需DC确认。",
        "4. PCCC理想全合并带来的系统收益小于单独attention表中的百分比，因为时间矩阵和空间引擎占据主要周期。最终PCCC必须以真实同类率和子系统EDP晋级。",
        "5. 512-bit HTT端口或双256-bit bank能显著降低event物化代价，但也会增加SRAM外围和布线；应与高bypass窄端口配置同约束综合。",
        "6. 该模型支持把HIT-Flow主创新进一步聚焦为跨PSN-attention-projection的局部驻留与流式转发，而不是单个算子周期优化。",
        "",
        "## 5. 不能从本模型得出的结论",
        "",
        "- 不能把保护后FPS当作RTL实测吞吐；",
        "- 不能把旧全网SOPS代理当作encoder逐层操作数；",
        "- 未计权重SRAM、真实bank conflict、地址生成、BN folding、decoder和外部DRAM仲裁；",
        "- 不知道计算与访存能够重叠到何种程度；",
        "- event bypass比例需要逐算子liveness和ordered trace证明，不能预设75%。",
        "",
        "因此当前推荐保留两套RTL参数点：面积边界`2×320 + 512 spatial + 2ctx`，吞吐边界`4×320 + 512 spatial + 2ctx`；PCCC、4ctx、512-bit端口和蝶形压紧均保持可旁路。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--sops", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()
    runtime_profile = load_json(args.runtime_profile) if args.runtime_profile else None
    result = build_model(
        load_json(args.storage),
        load_json(args.profile),
        load_json(args.sops),
        runtime_profile,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.md, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
