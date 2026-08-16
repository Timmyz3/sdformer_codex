#!/usr/bin/env python3
"""以真实ordered trace扫描跨窗口gate-product驻留数据流。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_hit_flow_ordered_profiles import decode_count_trace
from model_gcmp_multicast_dse import percentile, row_cycle_estimate


TOKENS_PER_WINDOW = 162
HEAD_LANES = 32
ACCUMULATOR_BITS = 32


def state_cost_bits(
    *, group_windows: int, class_slots: int, output_lanes: int
) -> dict[str, int]:
    destination_bitmap = class_slots * HEAD_LANES * TOKENS_PER_WINDOW * group_windows
    accumulator = group_windows * TOKENS_PER_WINDOW * output_lanes * ACCUMULATOR_BITS
    return {
        "destination_bitmap_bits": destination_bitmap,
        "accumulator_tile_bits": accumulator,
        "group_state_bits": destination_bitmap + accumulator,
    }


def evaluate_configuration(
    records: list[dict[str, Any]],
    *,
    group_windows: int,
    class_slots: int,
    multicast_width: int,
    output_lanes: int,
    product_engines: int,
) -> dict[str, Any]:
    candidate_rows = []
    direct_total = 0
    candidate_total = 0
    product_total = 0
    delivery_total = 0
    overflow_rows = 0
    valid_window_slots = 0
    for record in records:
        active = decode_count_trace(
            record[
                f"projection_gate_group_active_lanes_g{group_windows}_ordered_trace"
            ]
        )
        terms = decode_count_trace(
            record[f"projection_gate_group_terms_g{group_windows}_ordered_trace"]
        )
        classes = decode_count_trace(
            record[
                f"projection_gate_group_active_classes_g{group_windows}_ordered_trace"
            ]
        )
        delivery = decode_count_trace(
            record[
                f"projection_gate_group_delivery_g{group_windows}_m{multicast_width}_ordered_trace"
            ]
        )
        window_count = decode_count_trace(
            record[
                f"projection_gate_group_window_count_g{group_windows}_ordered_trace"
            ]
        )
        if not (
            len(active) == len(terms) == len(classes) == len(delivery) == len(window_count)
        ):
            raise ValueError("跨窗口ordered trace长度不一致")
        output_channels = int(record["num_heads"]) * int(record["head_dim"])
        for counters in zip(active, terms, classes, delivery):
            result = row_cycle_estimate(
                active_lanes=counters[0],
                class_channel_terms=counters[1],
                active_classes=counters[2],
                delivery_transactions=counters[3],
                output_channels=output_channels,
                output_lanes=output_lanes,
                product_engines=product_engines,
                class_slots=class_slots,
            )
            candidate_rows.append(int(result["candidate_cycles"]))
            direct_total += int(result["direct_cycles"])
            candidate_total += int(result["candidate_cycles"])
            product_total += int(result["product_cycles"])
            delivery_total += int(result["delivery_cycles"])
            overflow_rows += int(bool(result["overflow"]))
        valid_window_slots += sum(window_count)
    contexts = len(candidate_rows)
    state = state_cost_bits(
        group_windows=group_windows,
        class_slots=class_slots,
        output_lanes=output_lanes,
    )
    return {
        "group_windows": group_windows,
        "class_slots": class_slots,
        "multicast_width": multicast_width,
        "output_lanes": output_lanes,
        "product_engines": product_engines,
        "contexts": contexts,
        "overflow_contexts": overflow_rows,
        "overflow_ratio": overflow_rows / contexts if contexts else 0.0,
        "valid_window_slots": valid_window_slots,
        "allocated_window_slots": contexts * group_windows,
        "window_slot_utilization": (
            valid_window_slots / (contexts * group_windows) if contexts else 0.0
        ),
        **state,
        "direct_cycles": direct_total,
        "candidate_cycles": candidate_total,
        "product_cycles": product_total,
        "delivery_cycles": delivery_total,
        "ideal_speedup": direct_total / candidate_total if candidate_total else 0.0,
        "candidate_p50": percentile(candidate_rows, 0.50),
        "candidate_p95": percentile(candidate_rows, 0.95),
        "candidate_p99": percentile(candidate_rows, 0.99),
        "candidate_max": percentile(candidate_rows, 1.0),
        "限制": "未计class表构建、SRAM端口、bank conflict、窗口组装等待和跨组尾块利用率。",
    }


def render(result: dict[str, Any]) -> str:
    configurations = result["configurations"]
    eligible = [
        row for row in configurations
        if row["overflow_ratio"] <= 0.01 and row["ideal_speedup"] > 1.0
    ]
    top = sorted(
        eligible,
        key=lambda row: (-row["ideal_speedup"], row["group_state_bits"]),
    )[:30]
    lines = [
        "# 跨窗口Gate-Product驻留周期DSE",
        "",
        f"输入：`{result['profile']}`。共扫描`{len(configurations)}`个配置。",
        "",
        "## 低溢出候选",
        "",
        "| G | S | M | L | P | 溢出 | 状态KiB | 理想加速 | p95周期 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| {row['group_windows']} | {row['class_slots']} | "
            f"{row['multicast_width']} | {row['output_lanes']} | "
            f"{row['product_engines']} | {row['overflow_ratio']:.4%} | "
            f"{row['group_state_bits'] / 8192:.2f} | {row['ideal_speedup']:.4f} | "
            f"{row['candidate_p95']:.1f} |"
        )
    lines += [
        "",
        "## 解释边界",
        "",
        "- G增大时，乘积可跨空间窗口复用，但目的bitmap和token-output累加状态线性增长；",
        "- 表中状态只包含一个窗口组的目的bitmap和一个output tile的32-bit累加器；",
        "- 理想加速只覆盖投影后端，且使用max(product, delivery)的完全重叠假设；",
        "- 若G大于1不能在相同RTL、相同SRAM宏下带来至少15%的子系统EDP改善，则保留G=1；",
        "- 真实冻结还必须检查窗口组装延迟、bank conflict、尾组利用率和DC/SAIF。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.profile.read_text(encoding="utf-8"))
    records = data["summary"]["h60_records"]
    configurations = []
    for group_windows in (1, 2, 4, 8, 16):
        for class_slots in (2, 4, 8):
            for multicast_width in (1, 2, 4, 8, 16):
                for output_lanes in (8, 16, 32):
                    for product_engines in (1, 2, 4):
                        configurations.append(
                            evaluate_configuration(
                                records,
                                group_windows=group_windows,
                                class_slots=class_slots,
                                multicast_width=multicast_width,
                                output_lanes=output_lanes,
                                product_engines=product_engines,
                            )
                        )
    result = {
        "schema_version": 1,
        "profile": str(args.profile),
        "configurations": configurations,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md.write_text(render(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
