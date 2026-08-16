#!/usr/bin/env python3
"""第十轮架构候选的 CPU-only 存储与 ordered-work 证伪模型。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

try:
    from scripts.analyze_hit_flow_ordered_profiles import decode_count_trace
except ModuleNotFoundError:
    from analyze_hit_flow_ordered_profiles import decode_count_trace


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = (
    ROOT.parent
    / "neuron_experiments/H9_bipolar_self_attention/results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_LOCAL5 = (
    ROOT
    / "results/local5_hardware_profile_preG0_profile100_20260726"
    / "local5_hardware_features.json"
)
DEFAULT_OUT = ROOT / "results/architecture_innovation_round10_20260730"


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return int(ordered[index])


def summarize(values: list[int]) -> dict:
    total = sum(values)
    return {
        "count": len(values),
        "sum": total,
        "mean": total / len(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
    }


def capacity_stats(values: list[int], capacities: Iterable[int]) -> dict:
    total_work = max(1, sum(values))
    rows = max(1, len(values))
    return {
        str(capacity): {
            "overflow_rows": sum(value > capacity for value in values),
            "overflow_row_ratio": sum(value > capacity for value in values) / rows,
            "overflow_work": sum(max(0, value - capacity) for value in values),
            "overflow_work_ratio": (
                sum(max(0, value - capacity) for value in values) / total_work
            ),
        }
        for capacity in capacities
    }


def decode_motion_field(records: list[dict], field: str) -> list[int]:
    values: list[int] = []
    for record in records:
        values.extend(int(value) for value in decode_count_trace(record[field]))
    return values


def aggregate_histogram(records: list[dict], field: str) -> list[int]:
    histogram: list[int] = []
    for record in records:
        local = [int(value) for value in record[field]]
        if len(histogram) < len(local):
            histogram.extend([0] * (len(local) - len(histogram)))
        for index, value in enumerate(local):
            histogram[index] += value
    values: list[int] = []
    for value, count in enumerate(histogram):
        values.extend([value] * count)
    return values


def motion_model(profile: dict) -> dict:
    records = profile["summary"]["h60_records"]
    active_classes = aggregate_histogram(
        records,
        "row_active_projection_classes_h67_histogram",
    )
    all_classes = aggregate_histogram(
        records,
        "row_all_occupied_classes_h67_histogram",
    )
    score_lane_terms = decode_motion_field(
        records,
        "projection_class_channel_terms_h67_ordered_trace",
    )
    final_gate_terms = decode_motion_field(
        records,
        "projection_gate_class_channel_terms_deploy_ordered_trace",
    )
    active_lane_events = decode_motion_field(
        records,
        "projection_baseline_active_lanes_ordered_trace",
    )

    tokens = 162
    lanes = 32
    token_id_w = 8
    current_active_entry_w = 16 + lanes + token_id_w
    current_scs_bits = tokens * current_active_entry_w + 35 * (token_id_w + 1)
    current_g1_s4_bits = 4 * lanes * tokens + 4 * 9
    current_combined_bits = current_scs_bits + current_g1_s4_bits

    acqn_class_state_bits = 163 * 9 + 163 + 163 * 9
    fixed_bitmap_bits = {
        str(slots): acqn_class_state_bits + slots * lanes * tokens
        for slots in (4, 8, 16)
    }

    term_capacity = 256
    event_capacity = 1024
    term_header_bits = 8 + 10 + 8 + 1
    alias_bits = term_capacity * 9 + 6 * lanes * 9
    sparse_lower_bound_bits = (
        acqn_class_state_bits
        + term_capacity * term_header_bits
        + event_capacity * token_id_w
        + alias_bits
    )

    return {
        "evidence": "[prof ordered]+[storage model]，不是 RTL/PPA",
        "rows": len(score_lane_terms),
        "active_score_classes_per_row": summarize(active_classes),
        "all_score_classes_per_row": summarize(all_classes),
        "score_class_lane_terms_per_row": summarize(score_lane_terms),
        "final_gate_lane_terms_per_row": summarize(final_gate_terms),
        "active_lane_events_per_row": summarize(active_lane_events),
        "capacity": {
            "global_active_class_upper_bound": capacity_stats(
                active_classes,
                (4, 8, 16),
            ),
            "shared_term_slots": capacity_stats(
                score_lane_terms,
                (128, 192, 256, 384, 512),
            ),
            "shared_event_ids": capacity_stats(
                active_lane_events,
                (256, 512, 768, 1024, 1536),
            ),
        },
        "storage_bits_per_context": {
            "current_scs_active_plus_hist": current_scs_bits,
            "current_g1_s4_directory": current_g1_s4_bits,
            "current_two_stage_total": current_combined_bits,
            "fixed_class_bitmap_s4_s8_s16": fixed_bitmap_bits,
            "lane_sharded_sparse_lower_bound_s16_t256_e1024": (
                sparse_lower_bound_bits
            ),
            "sparse_vs_current_reduction": (
                1.0 - sparse_lower_bound_bits / current_combined_bits
            ),
        },
        "post_normalization_build_service": {
            "current_token_scan_cycles": len(score_lane_terms) * tokens,
            "class_parallel_lower_bound_cycles": sum(active_classes),
            "lower_bound_reduction": (
                1.0 - sum(active_classes) / (len(score_lane_terms) * tokens)
            ),
            "warning": (
                "假定32个lane bank并行扫描class；未计lane局部event深度、"
                "alias建链、fallback与term delivery"
            ),
        },
        "decision": {
            "fixed_class_bitmap": (
                "REJECT：S16才把global active-class overflow压到0.1%以下，"
                "但状态显著高于当前SCS+G1"
            ),
            "lane_sharded_sparse": (
                "CONDITIONAL：全局T256/E1024时row overflow均低于1%，"
                "但缺per-lane occupancy trace和多写端口实现"
            ),
        },
    }


def local5_model(profile: dict) -> dict:
    summary = profile["summary"]
    query_lane_reads = int(summary["query_major_k_lane_reads"])
    source_lane_reads = int(summary["source_resident_k_lane_reads"])
    query_active_reads = int(summary["query_major_active_k_lane_reads"])
    source_active_reads = int(summary["source_resident_active_k_lanes"])
    candidate_edges = int(summary["candidate_edges"])
    valid_edges = int(summary["valid_gate_entries"])
    tokens = candidate_edges // 5
    context_bits_per_destination = 5 * 8 + 5 + 9 + 1

    return {
        "evidence": "[prof-preG0]+[topology model]，不是 post-G0/RTL/PPA",
        "candidate_edges": candidate_edges,
        "valid_edges": valid_edges,
        "destinations": tokens,
        "k_lane_reads": {
            "query_major": query_lane_reads,
            "source_owned": source_lane_reads,
            "reduction": 1.0 - source_lane_reads / query_lane_reads,
        },
        "active_k_lane_reads": {
            "query_major": query_active_reads,
            "source_owned": source_active_reads,
            "reduction": 1.0 - source_active_reads / query_active_reads,
        },
        "one_port_service": {
            "query_major_valid_k_reads": valid_edges,
            "source_owned_k_reads": source_lane_reads // 32,
            "reduction": 1.0 - (source_lane_reads // 32) / valid_edges,
            "fairness_warning": (
                "仅对当前单K读口gather；理想5读口gather可保持同row-rate，"
                "但端口/读事务能耗更高"
            ),
        },
        "stripe_context_bits": {
            "3x9": 3 * 9 * context_bits_per_destination,
            "3x15": 3 * 15 * context_bits_per_destination,
            "3x9_edge_gate_plane": 3 * 9 * 5 * 9,
            "3x15_edge_gate_plane": 3 * 15 * 5 * 9,
            "full_t450_edge_gate_plane_if_not_streamed": 450 * 5 * 9,
            "contract": (
                "每destination保存5个Q7 score、完成mask、9-bit id和valid；"
                "edge gate plane单列；未计FIFO和SRAM对齐"
            ),
        },
        "term_work": {
            "active_edge_products": query_active_reads,
            "mfep_terms": int(summary["mfep_multicast_terms"]),
            "mfep_term_ratio": float(summary["mfep_term_ratio"]),
            "gate_cardinality_mean": float(summary["gate_cardinality_mean"]),
            "gate_cardinality_p95": int(summary["gate_cardinality_p95"]),
        },
        "decision": {
            "source_owned_wavefront": (
                "PROMOTE_TO_ORDERED_MODEL：固定五点拓扑使K读取一次后可广播到"
                "self/N/S/E/W destination context；必须用post-G0 trace、"
                "相同score lane和SRAM端口公平比较"
            )
        },
    }


def render_markdown(result: dict) -> str:
    motion = result["motion"]
    local5 = result["local5"]
    storage = motion["storage_bits_per_context"]
    return "\n".join(
        [
            "# 第十轮架构创新预筛",
            "",
            "## Motion：class-owned term transducer",
            "",
            (
                "- active score class/row："
                f"mean {motion['active_score_classes_per_row']['mean']:.3f}，"
                f"p95 {motion['active_score_classes_per_row']['p95']}，"
                f"p99 {motion['active_score_classes_per_row']['p99']}，"
                f"max {motion['active_score_classes_per_row']['max']}"
            ),
            (
                "- score-class/lane term/row："
                f"mean {motion['score_class_lane_terms_per_row']['mean']:.3f}，"
                f"p95 {motion['score_class_lane_terms_per_row']['p95']}，"
                f"p99 {motion['score_class_lane_terms_per_row']['p99']}，"
                f"max {motion['score_class_lane_terms_per_row']['max']}"
            ),
            (
                "- final-gate/lane term/row："
                f"mean {motion['final_gate_lane_terms_per_row']['mean']:.3f}，"
                f"p95 {motion['final_gate_lane_terms_per_row']['p95']}，"
                f"p99 {motion['final_gate_lane_terms_per_row']['p99']}"
            ),
            (
                "- 当前 SCS+G1 状态："
                f"{storage['current_two_stage_total']} bit/context"
            ),
            (
                "- 固定 class bitmap S16："
                f"{storage['fixed_class_bitmap_s4_s8_s16']['16']} bit/context，"
                "淘汰"
            ),
            (
                "- lane-sharded sparse lower bound："
                f"{storage['lane_sharded_sparse_lower_bound_s16_t256_e1024']} "
                "bit/context，仍缺 per-lane occupancy 与端口证据"
            ),
            "",
            "固定大 bitmap 不是可接受方向；只保留 lane-sharded sparse",
            "class-owned term/event store，并要求 exact fallback。",
            "",
            "## Local5：source-owned stencil wavefront",
            "",
            (
                "- K lane read reduction："
                f"{100.0 * local5['k_lane_reads']['reduction']:.2f}%"
            ),
            (
                "- active K lane read reduction："
                f"{100.0 * local5['active_k_lane_reads']['reduction']:.2f}%"
            ),
            (
                "- 3x15 destination context lower bound："
                f"{local5['stripe_context_bits']['3x15']} bit"
            ),
            (
                "- 3x15 streaming edge-gate plane："
                f"{local5['stripe_context_bits']['3x15_edge_gate_plane']} bit"
            ),
            (
                "- MFEP term / active-edge product："
                f"{100.0 * local5['term_work']['mfep_term_ratio']:.2f}%"
            ),
            "",
            "这些 Local5 数字仍是 pre-G0；只能支持进入 ordered model，",
            "不能作为 DATE 结果。",
            "",
            "## 冻结结论",
            "",
            "- Motion：保留 lane-sharded class-owned term transducer；固定 bitmap 淘汰。",
            "- Local5：保留 source-owned stencil wavefront；当前 query-major gather 作为基线。",
            "- 两条线不再强行共享 score/normalizer 核，只共享 term/projection 接口。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--local5", type=Path, default=DEFAULT_LOCAL5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    motion_profile = json.loads(args.motion.read_text())
    local5_profile = json.loads(args.local5.read_text())
    result = {
        "schema": "architecture_innovation_round10_v1",
        "motion_source": str(args.motion),
        "local5_source": str(args.local5),
        "motion": motion_model(motion_profile),
        "local5": local5_model(local5_profile),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out / "report.md").write_text(render_markdown(result) + "\n")
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
