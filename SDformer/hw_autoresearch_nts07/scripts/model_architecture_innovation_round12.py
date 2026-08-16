#!/usr/bin/env python3
"""第十二轮架构候选的存储账本与物理扫描上下界。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from scripts.analyze_hit_flow_ordered_profiles import decode_count_trace
except ModuleNotFoundError:
    from analyze_hit_flow_ordered_profiles import decode_count_trace


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT.parent
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_OUT = ROOT / "results/architecture_innovation_round12_20260730"


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return int(ordered[index])


def summarize(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
    }


def storage_ledger(
    tokens: int,
    *,
    lanes: int = 32,
    active_class_slots: int = 16,
) -> dict[str, int | float]:
    token_id_bits = max(1, math.ceil(math.log2(tokens)))
    count_bits = max(1, math.ceil(math.log2(tokens + 1)))
    gate_bits = 9
    classes = 163

    current_scs = (
        tokens * (16 + lanes + token_id_bits)
        + 35 * (token_id_bits + 1)
    )
    current_g1_s4 = 4 * lanes * tokens + 4 * (gate_bits + 1)
    current_total = current_scs + current_g1_s4

    class_table = classes * (count_bits + 1 + gate_bits)
    active_class_bitmap = active_class_slots * tokens
    k_lane_bitmap = lanes * tokens
    slot_metadata = active_class_slots * (8 + 1)
    segments = math.ceil(tokens / 64)
    segment_occupancy = active_class_slots * segments * lanes
    gate_alias = active_class_slots * gate_bits + (
        active_class_slots * active_class_slots
    )
    epoch_valid = (active_class_slots + lanes) * segments
    # 八项有界fragment FIFO，按{gate,lane,segment,bitmap,last}保守取88 bit。
    bounded_fragment_fifo = 8 * 88
    factorized_total = (
        class_table
        + active_class_bitmap
        + k_lane_bitmap
        + slot_metadata
        + segment_occupancy
        + gate_alias
        + epoch_valid
        + bounded_fragment_fifo
    )
    return {
        "tokens": tokens,
        "current_scs_bits": current_scs,
        "current_g1_s4_bits": current_g1_s4,
        "current_total_bits": current_total,
        "factorized_class_table_bits": class_table,
        "factorized_active_class_bitmap_bits": active_class_bitmap,
        "factorized_k_lane_bitmap_bits": k_lane_bitmap,
        "factorized_slot_metadata_bits": slot_metadata,
        "factorized_segment_occupancy_bits": segment_occupancy,
        "factorized_gate_alias_bits": gate_alias,
        "factorized_epoch_valid_bits": epoch_valid,
        "factorized_bounded_fragment_fifo_bits": bounded_fragment_fifo,
        "factorized_total_bits": factorized_total,
        "factorized_vs_current_ratio": factorized_total / current_total,
    }


def motion_bounds(profile: dict) -> dict:
    records = profile["summary"]["h60_records"]
    classes: list[int] = []
    terms: list[int] = []
    events: list[int] = []
    for record in records:
        classes.extend(
            int(value)
            for value in decode_count_trace(
                record["projection_active_classes_h67_ordered_trace"]
            )
        )
        terms.extend(
            int(value)
            for value in decode_count_trace(
                record["projection_class_channel_terms_h67_ordered_trace"]
            )
        )
        events.extend(
            int(value)
            for value in decode_count_trace(
                record["projection_baseline_active_lanes_ordered_trace"]
            )
        )
    if not (len(classes) == len(terms) == len(events)):
        raise ValueError("Motion ordered row trace 长度不一致")

    tokens = 162
    segments = math.ceil(tokens / 64)
    class_segment_lower = classes
    class_segment_upper = [
        min(active_classes * segments, max(0, class_lane_terms * segments))
        for active_classes, class_lane_terms in zip(classes, terms)
    ]
    class_lane_segment_lower = terms
    class_lane_segment_upper = [
        min(class_lane_terms * segments, active_lane_events)
        for class_lane_terms, active_lane_events in zip(terms, events)
    ]
    if any(
        lower > upper
        for lower, upper in zip(
            class_lane_segment_lower,
            class_lane_segment_upper,
        )
    ):
        raise ValueError("class-lane segment 上下界不闭合")

    overflow_rows = sum(value > 16 for value in classes)
    overflow_work = sum(max(0, value - 16) for value in classes)
    return {
        "evidence": "[ordered prof]+[bound model]，不是新增真实segment profile或PPA",
        "rows": len(classes),
        "tokens": tokens,
        "segment_tokens": 64,
        "segments_per_row": segments,
        "active_classes": summarize(classes),
        "class_lane_terms": summarize(terms),
        "active_lane_events": summarize(events),
        "class_segments_lower": summarize(class_segment_lower),
        "class_segments_upper": summarize(class_segment_upper),
        "class_lane_segments_lower": summarize(class_lane_segment_lower),
        "class_lane_segments_upper": summarize(class_lane_segment_upper),
        "s16_overflow_row_ratio": overflow_rows / len(classes),
        "s16_overflow_work_ratio": overflow_work / max(1, sum(classes)),
        "storage": {
            "t162": storage_ledger(162),
            "t450": storage_ledger(450),
        },
        "exact_fallback_contract": (
            "active class超过S16时，整行abort；保持Q/K驻留，重新生成score并走"
            "当前ordered SCS+G1。禁止部分fast-path结果与fallback混合提交。"
        ),
    }


def local5_frontier_storage(
    *,
    times: int,
    height: int,
    width: int,
    gate_bits: int = 9,
    directions: int = 5,
    live_rows: int = 3,
) -> dict[str, int | float]:
    full_bits = times * height * width * directions * gate_bits
    # 两个时间平面顺序处理，因此ring只需覆盖一个平面的活跃空间行。
    frontier_bits = live_rows * width * directions * gate_bits
    return {
        "tokens": times * height * width,
        "full_gate_plane_bits": full_bits,
        "three_row_frontier_bits": frontier_bits,
        "frontier_vs_full_ratio": frontier_bits / full_bits,
        "minimum_ready_lag_rows": 1,
        "warning": (
            "仅为无反压容量下界；projection堵塞时必须stall score前端或exact spill"
        ),
    }


def build_report(profile: dict) -> dict:
    return {
        "schema": "architecture_innovation_round12_v1",
        "motion_factorized_intersection_plane": motion_bounds(profile),
        "local5_deterministic_frontier": {
            "t162_2x9x9": local5_frontier_storage(
                times=2, height=9, width=9
            ),
            "t450_2x15x15": local5_frontier_storage(
                times=2, height=15, width=15
            ),
            "evidence": "[topology/storage model]，不是ordered cycle或RTL",
        },
    }


def render_markdown(report: dict) -> str:
    motion = report["motion_factorized_intersection_plane"]
    local = report["local5_deterministic_frontier"]
    lines = [
        "# 第十二轮架构候选模型",
        "",
        "## Motion：Class-Lane 因子化交集平面",
        "",
        "| 指标 | mean | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("active class", "active_classes"),
        ("class-lane term", "class_lane_terms"),
        ("64b class segment下界", "class_segments_lower"),
        ("64b class segment上界", "class_segments_upper"),
        ("64b class-lane segment下界", "class_lane_segments_lower"),
        ("64b class-lane segment上界", "class_lane_segments_upper"),
    ):
        row = motion[key]
        lines.append(
            f"| {label} | {row['mean']:.3f} | {row['p95']} | "
            f"{row['p99']} | {row['max']} |"
        )
    lines += [
        "",
        (
            f"- S16 overflow row：{motion['s16_overflow_row_ratio']:.6%}；"
            f"overflow work：{motion['s16_overflow_work_ratio']:.6%}。"
        ),
        "- 当前 profile 没有逐 segment identity，所以上下界不能当作预测周期。",
        "- 新 profile hook 已直接输出真实 class 与 class-lane segment trace。",
        "",
        "### 存储逻辑 bit 账本",
        "",
        "| T | 当前 SCS+G1 | 因子化平面 S16 | 比例 |",
        "|---:|---:|---:|---:|",
    ]
    for key in ("t162", "t450"):
        row = motion["storage"][key]
        lines.append(
            f"| {row['tokens']} | {row['current_total_bits']} | "
            f"{row['factorized_total_bits']} | "
            f"{row['factorized_vs_current_ratio']:.3f} |"
        )
    lines += [
        "",
        "FCIP已计segment occupancy、gate alias、epoch-valid和8项fragment FIFO；",
        "仍未计macro对齐、RMW/多读端口、fallback控制、Q/K延长驻留和双context，",
        "因此不是面积。",
        "",
        "## Local5：确定性前沿退休",
        "",
        "| 几何 | 完整gate plane | 三行frontier | 比例 |",
        "|---|---:|---:|---:|",
    ]
    for key in ("t162_2x9x9", "t450_2x15x15"):
        row = local[key]
        lines.append(
            f"| {key} | {row['full_gate_plane_bits']} | "
            f"{row['three_row_frontier_bits']} | "
            f"{row['frontier_vs_full_ratio']:.3f} |"
        )
    lines += [
        "",
        "三行ring是无反压的拓扑下界。若投影队列使前沿不能及时退休，必须停止",
        "score写入或进入exact spill；不能覆盖尚未消费的gate。",
        "",
        "## 证据边界",
        "",
        "- Motion segment结果当前是已有ordered trace推导的上下界；",
        "- Local5 frontier结果当前是固定五点stencil的容量模型；",
        "- 两者均不是周期、RTL、DC、功耗或EDP结果。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    report = build_report(profile)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "report.md").write_text(
        render_markdown(report) + "\n",
        encoding="utf-8",
    )
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
