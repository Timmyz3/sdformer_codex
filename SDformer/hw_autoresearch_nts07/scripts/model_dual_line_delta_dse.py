#!/usr/bin/env python3
"""用真实 profile 比较 H67 Motion-Delta 与 Local5 RCSD 的多拍/FIFO DSE。"""

from __future__ import annotations

import argparse
import base64
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_H67 = REPO / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "h67_ep19_ttb_delta_cycle_v2_profile100_20260713/"
    "nts11_hardware_p0_profile.json"
)
DEFAULT_LOCAL5 = ROOT / (
    "results/local5_hardware_profile_preG0_profile100_20260726/"
    "local5_hardware_features.json"
)
DEFAULT_OUTPUT = ROOT / "results/dual_line_delta_dse_20260726"
WIDTHS = (1, 2, 4, 8, 16, 32)


def decode_trace(encoded: dict[str, Any]) -> np.ndarray:
    dtypes = {"int16_le": "<i2", "int32_le": "<i4"}
    dtype = encoded.get("dtype")
    if dtype not in dtypes or encoded.get("codec") != "zlib_base64":
        raise ValueError("不支持的 ordered trace 编码")
    raw = zlib.decompress(base64.b64decode(encoded["data"]))
    shape = tuple(int(value) for value in encoded["shape"])
    result = np.frombuffer(raw, dtype=dtypes[dtype])
    if result.size != math.prod(shape):
        raise ValueError("ordered trace shape 与 payload 不一致")
    return result.reshape(shape).astype(np.int32, copy=False)


def percentile(values: list[int | float], probability: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def summarize(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def reflected_backlog(increments: np.ndarray) -> tuple[int, int]:
    """返回反射随机游走的 final/max backlog。"""

    if increments.ndim != 1:
        increments = increments.reshape(-1)
    if increments.size == 0:
        return 0, 0
    walk = np.cumsum(increments.astype(np.int64), dtype=np.int64)
    prefix_min = np.minimum.accumulate(
        np.concatenate((np.zeros(1, dtype=np.int64), walk))
    )[1:]
    backlog = walk - prefix_min
    return int(backlog[-1]), int(backlog.max(initial=0))


def queue_backlog(
    service: np.ndarray, extra_drain_cycles: np.ndarray | None = None
) -> tuple[int, int]:
    """每拍到达一个 job，后端每拍消费一个 quantum；可插入额外 drain 拍。"""

    if service.ndim != 1:
        service = service.reshape(-1)
    if extra_drain_cycles is None:
        extra_drain_cycles = np.zeros_like(service)
    elif extra_drain_cycles.shape != service.shape:
        extra_drain_cycles = extra_drain_cycles.reshape(service.shape)
    return reflected_backlog(
        service.astype(np.int64)
        - 1
        - extra_drain_cycles.astype(np.int64)
    )


def rne_div16_array(raw: np.ndarray) -> np.ndarray:
    quotient = raw // 16
    remainder = raw % 16
    increment = (remainder > 8) | ((remainder == 8) & ((quotient & 1) != 0))
    return quotient + increment.astype(np.int32)


def h67_score_pair(record: dict[str, Any]) -> np.ndarray:
    q_count = decode_trace(record["pair_q_count_ordered_trace"])
    k_count = decode_trace(record["pair_k_count_ordered_trace"])
    overlap = decode_trace(record["pair_overlap_ordered_trace"])
    motion = decode_trace(record["pair_motion_ordered_trace"])
    if q_count.shape[0] != 2 or motion.shape != q_count.shape[1:]:
        raise ValueError("H67 score pair trace shape 不一致")
    same_zero = 32 - q_count - k_count + overlap
    raw = 65 * overlap + 32 - q_count - k_count + 16 * motion[None, ...]
    if np.any(same_zero < 0) or np.any(raw < 0):
        raise ValueError("H67 score trace 出现非法计数")
    return rne_div16_array(raw)


def h67_dse(profile: dict[str, Any]) -> dict[str, Any]:
    records = profile["summary"]["h60_records"]
    width_state: dict[int, dict[str, Any]] = {
        width: {
            "service": 0,
            "pipeline_cycles": 0,
            "serial_cycles": 0,
            "fifo_max_by_record": [],
            "fifo_final_by_record": [],
            "stage_pairs": defaultdict(int),
            "stage_pipeline_cycles": defaultdict(int),
            "stage_service": defaultdict(int),
            "fallback": {},
        }
        for width in WIDTHS
    }
    pairs = 0
    record_count = 0
    score_equal = 0
    score_equal_profile = 0
    motion_zero = 0
    ttb4_groups = 0
    ttb4_nonempty = 0
    ttb8_groups = 0
    ttb8_nonempty = 0

    for record in records:
        delta_tensor = decode_trace(record["delta_update_ordered_trace"])
        delta = delta_tensor.reshape(-1)
        motion = decode_trace(record["pair_motion_ordered_trace"]).reshape(-1)
        stage = int(record["stage"])
        pair_count = int(delta.size)
        pairs += pair_count
        record_count += 1
        motion_zero += int(np.count_nonzero(motion == 0))

        scores = h67_score_pair(record)
        equal = scores[0] == scores[1]
        score_equal += int(np.count_nonzero(equal))
        score_equal_profile += int(record["pair_score_equal_h67"])

        for bundle, totals in (
            (4, ("ttb4_groups", "ttb4_nonempty")),
            (8, ("ttb8_groups", "ttb8_nonempty")),
        ):
            token_pairs = int(delta_tensor.shape[-1])
            groups_per_row = (token_pairs + bundle - 1) // bundle
            padded = np.pad(
                delta_tensor > 0,
                (
                    (0, 0),
                    (0, 0),
                    (0, groups_per_row * bundle - token_pairs),
                ),
                constant_values=False,
            )
            nonempty = int(
                np.count_nonzero(
                    padded.reshape(
                        *padded.shape[:-1], groups_per_row, bundle
                    ).any(axis=-1)
                )
            )
            groups = math.prod(delta_tensor.shape[:-1]) * groups_per_row
            if totals[0] == "ttb4_groups":
                ttb4_groups += groups
                ttb4_nonempty += nonempty
            else:
                ttb8_groups += groups
                ttb8_nonempty += nonempty

        for width in WIDTHS:
            service = np.where(
                delta > 0, (delta + width - 1) // width, 0
            ).astype(np.int32)
            final_backlog, max_backlog = queue_backlog(service)
            state = width_state[width]
            service_sum = int(service.sum())
            pipeline_cycles = pair_count + final_backlog
            state["service"] += service_sum
            state["pipeline_cycles"] += pipeline_cycles
            state["serial_cycles"] += pair_count + service_sum
            state["fifo_max_by_record"].append(max_backlog)
            state["fifo_final_by_record"].append(final_backlog)
            state["stage_pairs"][stage] += pair_count
            state["stage_pipeline_cycles"][stage] += pipeline_cycles
            state["stage_service"][stage] += service_sum

            fallback_rows: dict[str, Any] = state["fallback"]
            for threshold in sorted({width, min(32, 2 * width), min(32, 4 * width)}):
                key = str(threshold)
                row = fallback_rows.setdefault(
                    key,
                    {
                        "fallback_items": 0,
                        "sparse_service": 0,
                        "producer_cycles": 0,
                        "decoupled_ordered_cycles": 0,
                        "fifo_max_by_record": [],
                        "fifo_final_by_record": [],
                    },
                )
                fallback = delta > threshold
                sparse_service = np.where(
                    (delta > 0) & ~fallback,
                    (delta + width - 1) // width,
                    0,
                )
                fallback_count = int(np.count_nonzero(fallback))
                sparse_sum = int(sparse_service.sum())
                producer_cycles = pair_count + fallback_count
                final_fallback_backlog, max_fallback_backlog = queue_backlog(
                    sparse_service.astype(np.int32),
                    fallback.astype(np.int32),
                )
                row["fallback_items"] += fallback_count
                row["sparse_service"] += sparse_sum
                row["producer_cycles"] += producer_cycles
                row["decoupled_ordered_cycles"] += (
                    producer_cycles + final_fallback_backlog
                )
                row["fifo_max_by_record"].append(max_fallback_backlog)
                row["fifo_final_by_record"].append(final_fallback_backlog)

    if score_equal != score_equal_profile:
        raise ValueError(
            f"H67 score-equal 重建不一致: {score_equal} != {score_equal_profile}"
        )
    baseline_serial_cycles = 2 * pairs
    baseline_dual_cycles = pairs
    result_widths: dict[str, Any] = {}
    for width, state in width_state.items():
        pipeline_cycles = int(state["pipeline_cycles"])
        serial_cycles = int(state["serial_cycles"])
        lane_area = 32 + width
        result_widths[str(width)] = {
            "backend_service_quanta": int(state["service"]),
            "backend_utilization": state["service"] / pairs if pairs else 0.0,
            "serial_shared_cycles": serial_cycles,
            "serial_speedup_vs_direct32": (
                baseline_serial_cycles / serial_cycles if serial_cycles else 0.0
            ),
            "decoupled_unbounded_cycles": pipeline_cycles,
            "decoupled_speedup_vs_direct32": (
                baseline_serial_cycles / pipeline_cycles
                if pipeline_cycles
                else 0.0
            ),
            "lane_area": lane_area,
            "area_efficiency_vs_direct32": (
                baseline_serial_cycles * 32 / (pipeline_cycles * lane_area)
                if pipeline_cycles
                else 0.0
            ),
            "throughput_vs_direct32x2": (
                baseline_dual_cycles / pipeline_cycles
                if pipeline_cycles
                else 0.0
            ),
            "area_efficiency_vs_direct32x2": (
                baseline_dual_cycles * 64 / (pipeline_cycles * lane_area)
                if pipeline_cycles
                else 0.0
            ),
            "fifo_work_quanta_required": summarize(
                state["fifo_max_by_record"]
            ),
            "fifo_final_work_quanta": summarize(
                state["fifo_final_by_record"]
            ),
            "by_stage": {
                f"S{stage}": {
                    "pairs": int(state["stage_pairs"][stage]),
                    "backend_utilization": (
                        state["stage_service"][stage]
                        / state["stage_pairs"][stage]
                        if state["stage_pairs"][stage]
                        else 0.0
                    ),
                    "decoupled_cycles": int(
                        state["stage_pipeline_cycles"][stage]
                    ),
                }
                for stage in sorted(state["stage_pairs"])
            },
            "fallback_lower_bounds": {
                threshold: {
                    **row,
                    "fallback_ratio": row["fallback_items"] / pairs
                    if pairs
                    else 0.0,
                    "speedup_vs_direct32": (
                        baseline_serial_cycles
                        / row["decoupled_ordered_cycles"]
                        if row["decoupled_ordered_cycles"]
                        else 0.0
                    ),
                    "throughput_vs_direct32x2": (
                        baseline_dual_cycles
                        / row["decoupled_ordered_cycles"]
                        if row["decoupled_ordered_cycles"]
                        else 0.0
                    ),
                    "area_efficiency_vs_direct32x2": (
                        baseline_dual_cycles
                        * 64
                        / (row["decoupled_ordered_cycles"] * lane_area)
                        if row["decoupled_ordered_cycles"]
                        else 0.0
                    ),
                    "fifo_work_quanta_required": summarize(
                        row["fifo_max_by_record"]
                    ),
                    "fifo_final_work_quanta": summarize(
                        row["fifo_final_by_record"]
                    ),
                }
                for threshold, row in state["fallback"].items()
            },
        }

    return {
        "records": record_count,
        "pairs": pairs,
        "baselines": {
            "direct32_serial_cycles": baseline_serial_cycles,
            "direct32_lane_area": 32,
            "direct32x2_ideal_cycles": baseline_dual_cycles,
            "direct32x2_lane_area": 64,
            "cycle_convention": (
                "两条基线和候选均不计 record 填充/排空；候选仅追加 ordered "
                "residual backlog 和 dense replay 拍。"
            ),
        },
        "score_equal_reconstructed": score_equal,
        "score_equal_profile": score_equal_profile,
        "score_equal_ratio": score_equal / pairs if pairs else 0.0,
        "scs_class_count_commits": {
            "baseline_transactions": 2 * pairs,
            "coalesced_transactions": 2 * pairs - score_equal,
            "transaction_reduction": score_equal / (2 * pairs)
            if pairs
            else 0.0,
            "single_port_cycle_upper_bound": 2 * pairs - score_equal,
            "dual_port_ideal_cycle_lower_bound": math.ceil(
                (2 * pairs - score_equal) / 2
            ),
        },
        "motion_zero_ratio": motion_zero / pairs if pairs else 0.0,
        "ttb": {
            "pair_descriptors": pairs,
            "ttb4_descriptors": ttb4_groups,
            "ttb4_nonempty_delta_descriptors": ttb4_nonempty,
            "ttb4_empty_ratio": 1.0 - ttb4_nonempty / ttb4_groups,
            "ttb8_descriptors": ttb8_groups,
            "ttb8_nonempty_delta_descriptors": ttb8_nonempty,
            "ttb8_empty_ratio": 1.0 - ttb8_nonempty / ttb8_groups,
        },
        "widths": result_widths,
    }


def merge_local5_histogram(record: dict[str, Any]) -> np.ndarray:
    histograms = [
        record[f"{direction}_delta_histogram"]
        for direction in ("up", "down", "left", "right")
    ]
    width = max(len(hist) for hist in histograms)
    result = np.zeros(width, dtype=np.int64)
    for histogram in histograms:
        result[: len(histogram)] += np.asarray(histogram, dtype=np.int64)
    return result


def local5_dse(profile: dict[str, Any]) -> dict[str, Any]:
    records = profile["records"]
    queries = 0
    valid_edges = 0
    width_state: dict[int, dict[str, Any]] = {
        width: {
            "service": 0,
            "serial_cycles": 0,
            "decoupled_lower_bound": 0,
            "utilization_by_record": [],
            "stage_queries": defaultdict(int),
            "stage_service": defaultdict(int),
            "stage_decoupled": defaultdict(int),
            "fallback": {},
        }
        for width in WIDTHS
    }

    for record in records:
        histogram = merge_local5_histogram(record)
        stage = int(record["stage"])
        self_queries = int(record["token_heads"])
        record_valid_edges = int(record["valid_edges"])
        queries += self_queries
        valid_edges += record_valid_edges
        counts = np.arange(histogram.size, dtype=np.int64)

        for width in WIDTHS:
            service_by_count = np.where(
                counts > 0, (counts + width - 1) // width, 0
            )
            service = int(np.dot(histogram, service_by_count))
            state = width_state[width]
            serial_cycles = self_queries + service
            decoupled = max(self_queries, service)
            state["service"] += service
            state["serial_cycles"] += serial_cycles
            state["decoupled_lower_bound"] += decoupled
            state["utilization_by_record"].append(
                service / self_queries if self_queries else 0.0
            )
            state["stage_queries"][stage] += self_queries
            state["stage_service"][stage] += service
            state["stage_decoupled"][stage] += decoupled

            for threshold in sorted({width, min(32, 2 * width), min(32, 4 * width)}):
                key = str(threshold)
                row = state["fallback"].setdefault(
                    key,
                    {
                        "fallback_edges": 0,
                        "sparse_service": 0,
                        "producer_cycles": 0,
                        "decoupled_conservation_lower_bound": 0,
                    },
                )
                fallback_mask = counts > threshold
                sparse_mask = (counts > 0) & ~fallback_mask
                fallback_edges = int(histogram[fallback_mask].sum())
                sparse_service = int(
                    np.dot(
                        histogram[sparse_mask],
                        service_by_count[sparse_mask],
                    )
                )
                producer_cycles = self_queries + fallback_edges
                row["fallback_edges"] += fallback_edges
                row["sparse_service"] += sparse_service
                row["producer_cycles"] += producer_cycles
                row["decoupled_conservation_lower_bound"] += max(
                    producer_cycles, sparse_service
                )

    baseline_cycles = valid_edges
    direct5_parallel_cycles = queries
    result_widths: dict[str, Any] = {}
    for width, state in width_state.items():
        decoupled = int(state["decoupled_lower_bound"])
        serial = int(state["serial_cycles"])
        lane_area = 32 + width
        result_widths[str(width)] = {
            "backend_service_quanta": int(state["service"]),
            "backend_offered_load_per_query": state["service"] / queries
            if queries
            else 0.0,
            "backend_offered_load_by_record": summarize(
                state["utilization_by_record"]
            ),
            "serial_shared_cycles": serial,
            "serial_speedup_vs_direct32": baseline_cycles / serial
            if serial
            else 0.0,
            "decoupled_conservation_lower_bound": decoupled,
            "ideal_decoupled_speedup_upper_bound": baseline_cycles / decoupled
            if decoupled
            else 0.0,
            "lane_area": lane_area,
            "ideal_area_efficiency_upper_bound_vs_direct32": (
                baseline_cycles * 32 / (decoupled * lane_area)
                if decoupled
                else 0.0
            ),
            "by_stage": {
                f"S{stage}": {
                    "queries": int(state["stage_queries"][stage]),
                    "backend_offered_load_per_query": (
                        state["stage_service"][stage]
                        / state["stage_queries"][stage]
                        if state["stage_queries"][stage]
                        else 0.0
                    ),
                    "decoupled_conservation_lower_bound": int(
                        state["stage_decoupled"][stage]
                    ),
                }
                for stage in sorted(state["stage_queries"])
            },
            "fallback_lower_bounds": {
                threshold: {
                    **row,
                    "fallback_ratio_vs_directional_edges": (
                        row["fallback_edges"]
                        / (valid_edges - queries)
                        if valid_edges > queries
                        else 0.0
                    ),
                    "ideal_speedup_upper_bound_vs_direct32": (
                        baseline_cycles
                        / row["decoupled_conservation_lower_bound"]
                        if row["decoupled_conservation_lower_bound"]
                        else 0.0
                    ),
                }
                for threshold, row in state["fallback"].items()
            },
        }

    return {
        "records": len(records),
        "queries": queries,
        "valid_edges": valid_edges,
        "average_degree": valid_edges / queries if queries else 0.0,
        "baselines": {
            "direct32_serial_cycles": baseline_cycles,
            "direct32_lane_area": 32,
            "direct5_parallel_cycles": direct5_parallel_cycles,
            "direct5_parallel_lane_area": 160,
            "direct5_area_efficiency_vs_direct32": (
                baseline_cycles * 32
                / (direct5_parallel_cycles * 160)
                if direct5_parallel_cycles
                else 0.0
            ),
        },
        "detector_bit_operations": {
            "directional_k_xor_bits": (valid_edges - queries) * 32,
            "note": "单独计数，未与 reduction cycle/energy 混合",
        },
        "ordered_boundary": (
            "当前 Local5 profile 只有 per-record directional histogram；"
            "decoupled cycles 是守恒下界，因此换算的 speedup/面积效率是"
            "理想上界；不能给出 FIFO/burst。"
        ),
        "widths": result_widths,
    }


def render_markdown(result: dict[str, Any]) -> str:
    h67 = result["h67_motion"]
    local5 = result["local5_pre_g0"]
    lines = [
        "# H67 Motion 与 Local5 RCSD 双线多拍/FIFO DSE",
        "",
        "## 0. 证据边界",
        "",
        "- H67 使用 profile100 的逐 pair ordered trace，可计算真实顺序服务量和"
        "无界 FIFO backlog；",
        "- Local5 当前只有 per-record 方向 histogram，周期是 work-conservation "
        "下界，换算的 speedup/面积效率是理想上界，不能声称 FIFO、p99 或"
        "真实 speedup；",
        "- 所有 lane area 都是 `32+W` 的 lane-equivalent，不是综合面积；",
        "- detector、SRAM、索引、RNE、控制和时序频率尚未折算为能量/PPA；",
        "- Local5 是 pre-G0，G0/G1 后必须同 cohort 重跑。",
        "",
        "## 1. H67 Motion-Delta",
        "",
        "架构模型：一个 32-lane T0 anchor engine 每拍产生一个 pair；独立 W-lane "
        "residual backend 用 `ceil(update/W)` 拍处理 T1。无更新 pair 不进入后端。",
        "",
        "| W | backend load | vs Direct32 | 吞吐/vs Direct32x2 | "
        "面积效率/vs Direct32x2 | FIFO p95/p99/max |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        row = h67["widths"][str(width)]
        fifo = row["fifo_work_quanta_required"]
        lines.append(
            f"| {width} | {row['backend_utilization']:.4f} | "
            f"{row['decoupled_speedup_vs_direct32']:.4f}x | "
            f"{row['throughput_vs_direct32x2']:.4f}x | "
            f"{row['area_efficiency_vs_direct32x2']:.4f}x | "
            f"{fifo['p95']:.0f}/{fifo['p99']:.0f}/{fifo['max']:.0f} |"
        )
    lines += [
        "",
        "### H67 dense fallback",
        "",
        "超过 threshold 的 T1 由 32-lane anchor engine 追加一拍 direct score；"
        "该拍同时给 residual backend 排空一个 work quantum。",
        "",
        "| W | threshold | fallback | vs Direct32 | 吞吐/vs Direct32x2 | "
        "面积效率/vs Direct32x2 | FIFO p95/p99/max |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for width, threshold in ((2, 4), (2, 8), (4, 4), (4, 8), (4, 16), (8, 8)):
        row = h67["widths"][str(width)]["fallback_lower_bounds"][str(threshold)]
        fifo = row["fifo_work_quanta_required"]
        lines.append(
            f"| {width} | {threshold} | {row['fallback_ratio']:.4%} | "
            f"{row['speedup_vs_direct32']:.4f}x | "
            f"{row['throughput_vs_direct32x2']:.4f}x | "
            f"{row['area_efficiency_vs_direct32x2']:.4f}x | "
            f"{fifo['p95']:.0f}/{fifo['p99']:.0f}/{fifo['max']:.0f} |"
        )
    lines += [
        "",
        "`W=T` 时每个 sparse 项最多产生一个 backend quantum，超过阈值的"
        "项转为 direct replay；因此在同拍旁路、backend 每拍稳定消费、无"
        "SRAM 等待和无下游反压的模型内不需要跨 pair residual work FIFO。"
        "RTL 仍必须提供输出寄存器或 skid buffer。",
        "",
        "### H67 辅助机制",
        "",
        f"- Motion-zero bypass 覆盖：`{h67['motion_zero_ratio']:.4%}`；",
        f"- score 相同率：`{h67['score_equal_ratio']:.4%}`；",
        f"- SCS class-count commit transaction 减少："
        f"`{h67['scs_class_count_commits']['transaction_reduction']:.4%}`；",
        f"- TTB4 empty-delta bundle：`{h67['ttb']['ttb4_empty_ratio']:.4%}`，"
        f"descriptor 从 `{h67['ttb']['pair_descriptors']}` 降到 "
        f"`{h67['ttb']['ttb4_descriptors']}`；",
        "- score-equal 只合并 SCS class-count，不合并 K 不同的 projection。",
        "",
        "### H67 分 Stage",
        "",
        "| W | Stage | backend util | decoupled cycles |",
        "|---:|---|---:|---:|",
    ]
    for width in (2, 4, 8):
        for stage, row in h67["widths"][str(width)]["by_stage"].items():
            lines.append(
                f"| {width} | {stage} | {row['backend_utilization']:.4f} | "
                f"{row['decoupled_cycles']} |"
            )
    lines += [
        "",
        "## 2. Local5 RCSD（pre-G0）",
        "",
        "架构模型：32-lane self anchor engine 每拍产生一个 query；W-lane residual "
        "backend 处理四方向差分。由于缺少 query 内四方向 ordered/joint trace，"
        "这里只报告周期守恒下界及由此换算的理想性能上界。",
        "",
        "| W | backend offered load/query | serial工作量speedup | "
        "decoupled理想speedup上界 | lane-area效益上界 | record load p95/p99 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        row = local5["widths"][str(width)]
        util = row["backend_offered_load_by_record"]
        lines.append(
            f"| {width} | {row['backend_offered_load_per_query']:.4f} | "
            f"{row['serial_speedup_vs_direct32']:.4f}x | "
            f"{row['ideal_decoupled_speedup_upper_bound']:.4f}x | "
            f"{row['ideal_area_efficiency_upper_bound_vs_direct32']:.4f}x | "
            f"{util['p95']:.4f}/{util['p99']:.4f} |"
        )
    lines += [
        "",
        f"- 平均有效 degree：`{local5['average_degree']:.6f}`；",
        f"- Direct5 全并行 lane-area efficiency 相对单 Direct32："
        f"`{local5['baselines']['direct5_area_efficiency_vs_direct32']:.4f}x`；",
        "- Local5 的 decoupled 数字不是可宣传 speedup，必须补 ordered "
        "STT trace、FIFO、边界 halo 和 G0/G1 后复跑。",
        "",
        "### Local5 fallback 理想上界",
        "",
        "| W | threshold | fallback edge | speedup理想上界 |",
        "|---:|---:|---:|---:|",
    ]
    for width, threshold in ((2, 2), (2, 4), (4, 4), (4, 8), (8, 8)):
        row = local5["widths"][str(width)]["fallback_lower_bounds"][str(threshold)]
        lines.append(
            f"| {width} | {threshold} | "
            f"{row['fallback_ratio_vs_directional_edges']:.4%} | "
            f"{row['ideal_speedup_upper_bound_vs_direct32']:.4f}x |"
        )
    lines += [
        "",
        "## 3. Fallback DSE 的解释",
        "",
        "每个 W 都评估 `threshold=W/2W/4W`：超过阈值的项回到 32-lane anchor "
        "engine，稀疏项进入 residual backend。H67 使用 ordered trace 精确加入"
        "fallback direct 拍的 backend drain；Local5 仍只有 producer/backend "
        "work-conservation 周期下界和性能上界，不能直接冻结阈值。",
        "",
        "## 4. 晋级规则",
        "",
        "1. H67 只有在 ordered FIFO p99 可接受、`32+W` 同约束综合后的"
        "面积归一吞吐优于 Direct32/Direct32x2 时，才进入集成 RTL；",
        "2. Local5 必须先补 G0/G1 和 ordered STT，当前只允许做 reference、"
        "叶 compactor 与 direct/delta 算术 DSE；",
        "3. 4/8-lane 不预设为优胜点，以 JSON sweep 和后续 PPA 选择；",
        "4. TTB、Motion bypass、SCS coalescing 分别做消融，不重复累计收益。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h67", type=Path, default=DEFAULT_H67)
    parser.add_argument("--local5", type=Path, default=DEFAULT_LOCAL5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    h67_profile = json.loads(args.h67.read_text(encoding="utf-8"))
    local5_profile = json.loads(args.local5.read_text(encoding="utf-8"))
    result = {
        "schema": "dual_line_delta_dse_v2",
        "h67_source": str(args.h67),
        "local5_source": str(args.local5),
        "h67_motion": h67_dse(h67_profile),
        "local5_pre_g0": local5_dse(local5_profile),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "dual_line_delta_dse.json"
    md_path = args.output_dir / "dual_line_delta_dse.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
