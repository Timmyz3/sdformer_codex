#!/usr/bin/env python3
"""使用ordered profile重放GCM-P class-slot与多播宽度设计空间。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_hit_flow_ordered_profiles import decode_count_trace


def row_cycle_estimate(
    *,
    active_lanes: int,
    class_channel_terms: int,
    active_classes: int,
    delivery_transactions: int,
    output_channels: int,
    output_lanes: int,
    product_engines: int,
    class_slots: int,
) -> dict[str, int | bool]:
    if min(active_lanes, class_channel_terms, active_classes, delivery_transactions) < 0:
        raise ValueError("row计数不能为负")
    if output_channels <= 0 or output_lanes <= 0 or product_engines <= 0 or class_slots <= 0:
        raise ValueError("硬件参数必须为正")
    chunks = math.ceil(output_channels / output_lanes)
    direct_product_cycles = math.ceil(active_lanes / product_engines) * chunks
    overflow = active_classes > class_slots
    if overflow:
        return {
            "overflow": True,
            "chunks": chunks,
            "direct_cycles": direct_product_cycles,
            "product_cycles": direct_product_cycles,
            "delivery_cycles": direct_product_cycles,
            "candidate_cycles": direct_product_cycles,
        }
    product_cycles = math.ceil(class_channel_terms / product_engines) * chunks
    delivery_cycles = delivery_transactions * chunks
    return {
        "overflow": False,
        "chunks": chunks,
        "direct_cycles": direct_product_cycles,
        "product_cycles": product_cycles,
        "delivery_cycles": delivery_cycles,
        "candidate_cycles": max(product_cycles, delivery_cycles),
    }


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def evaluate_configuration(
    records: list[dict[str, Any]],
    *,
    variant: str,
    class_slots: int,
    multicast_width: int,
    output_lanes: int,
    product_engines: int,
) -> dict[str, Any]:
    row_results = []
    direct_total = 0
    candidate_total = 0
    product_total = 0
    delivery_total = 0
    overflow_rows = 0
    for record in records:
        active = decode_count_trace(record["projection_baseline_active_lanes_ordered_trace"])
        terms = decode_count_trace(
            record["projection_gate_class_channel_terms_deploy_ordered_trace"]
        )
        classes = decode_count_trace(
            record["projection_active_gate_classes_deploy_ordered_trace"]
        )
        delivery = decode_count_trace(
            record[f"projection_gate_multicast_delivery_m{multicast_width}_ordered_trace"]
        )
        if not (len(active) == len(terms) == len(classes) == len(delivery)):
            raise ValueError("GCM-P ordered trace长度不一致")
        output_channels = int(record["num_heads"]) * int(record["head_dim"])
        for row in zip(active, terms, classes, delivery):
            result = row_cycle_estimate(
                active_lanes=row[0],
                class_channel_terms=row[1],
                active_classes=row[2],
                delivery_transactions=row[3],
                output_channels=output_channels,
                output_lanes=output_lanes,
                product_engines=product_engines,
                class_slots=class_slots,
            )
            row_results.append(int(result["candidate_cycles"]))
            direct_total += int(result["direct_cycles"])
            candidate_total += int(result["candidate_cycles"])
            product_total += int(result["product_cycles"])
            delivery_total += int(result["delivery_cycles"])
            overflow_rows += int(bool(result["overflow"]))
    rows = len(row_results)
    tokens = max(int(record["tokens"]) for record in records) if records else 0
    destination_bitmap_bits = class_slots * 32 * tokens
    return {
        "class_slots": class_slots,
        "multicast_width": multicast_width,
        "output_lanes": output_lanes,
        "product_engines": product_engines,
        "rows": rows,
        "overflow_rows": overflow_rows,
        "overflow_ratio": overflow_rows / rows if rows else 0.0,
        "destination_bitmap_bits_per_context": destination_bitmap_bits,
        "direct_cycles": direct_total,
        "candidate_cycles": candidate_total,
        "product_cycles": product_total,
        "delivery_cycles": delivery_total,
        "ideal_speedup": direct_total / candidate_total if candidate_total else 0.0,
        "candidate_p50": percentile(row_results, 0.50),
        "candidate_p95": percentile(row_results, 0.95),
        "candidate_p99": percentile(row_results, 0.99),
        "candidate_max": percentile(row_results, 1.0),
        "限制": "周期仅覆盖projection backend，未含class表构建、SRAM延迟、accumulator bank冲突和控制开销。",
        "class_semantics": "最终Q1.7 gate码，不是量化前score类。",
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# GCM-P Ordered Trace周期DSE",
        "",
        f"模型：`{result['variant'].upper()}`；输入：`{result['profile']}`。",
        "",
        "| S | M | L | P | overflow | bitmap/context | ideal speedup | p95 cycles |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["configurations"]:
        lines.append(
            f"| {row['class_slots']} | {row['multicast_width']} | {row['output_lanes']} | "
            f"{row['product_engines']} | {row['overflow_ratio']:.4%} | "
            f"{row['destination_bitmap_bits_per_context']} bit | {row['ideal_speedup']:.4f} | "
            f"{row['candidate_p95']:.1f} |"
        )
    lines += [
        "",
        "## 使用限制",
        "",
        "- 这是projection backend的周期上界筛选，不是全encoder FPS；",
        "- 多播事务来自真实class-channel fanout，但尚未加入token accumulator bank conflict；",
        "- class slot、乘积项和多播事务均按最终Q1.7 gate码统计；",
        "- class表构建可与score扫描重叠的程度需由RTL有限FIFO重放确认；",
        "- 面积和功耗必须使用同约束RTL/DC/SAIF，不能由ideal speedup推导。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--variant", choices=("ttx", "h67"), required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.profile.read_text(encoding="utf-8"))
    records = data["summary"]["h60_records"]
    configurations = []
    for class_slots in (2, 4, 8):
        for multicast_width in (1, 2, 4, 8, 16):
            for output_lanes in (8, 16, 32):
                for product_engines in (1, 2, 4):
                    configurations.append(evaluate_configuration(
                        records,
                        variant=args.variant,
                        class_slots=class_slots,
                        multicast_width=multicast_width,
                        output_lanes=output_lanes,
                        product_engines=product_engines,
                    ))
    result = {
        "schema_version": 1,
        "profile": str(args.profile),
        "variant": args.variant,
        "configurations": configurations,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md.write_text(render(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
