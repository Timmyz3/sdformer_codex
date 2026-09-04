#!/usr/bin/env python3
"""双线拓扑商流与裂分/融合残差阵列的 CPU-only 架构模型。

本模型只使用现有 profile，不运行网络推理。它回答三个彼此独立的问题：

1. 拓扑等价关系能精确折叠多少候选；
2. 四个残差切片在 Local5 裂分、在 Motion 融合后，能否避免串行 TARE 的吞吐损失；
3. score、候选缓冲和 projection 三段工作量分别减少多少。

周期均为架构模型。Local5 缺少逐查询 ordered trace，因此裂分队列结果是服务量下界，
不得写成 RTL 周期或正式 p99。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import statistics
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "topology_quotient_ff_dse_20260730"
MOTION_PROFILE = (
    ROOT.parent
    / "neuron_experiments"
    / "H9_bipolar_self_attention"
    / "results"
    / "h67_ep19_ttb_delta_cycle_v2_profile100_20260713"
    / "nts11_hardware_p0_profile.json"
)
LOCAL_PROFILE = (
    ROOT
    / "results"
    / "local5_hardware_profile_preG0_profile100_20260726"
    / "local5_hardware_features.json"
)


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor 必须为正数")
    return (value + divisor - 1) // divisor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def distribution(values: Iterable[float]) -> dict[str, float]:
    items = [float(value) for value in values]
    mean = statistics.fmean(items) if items else 0.0
    return {
        "mean": mean,
        "p50": percentile(items, 0.50),
        "p95": percentile(items, 0.95),
        "p99": percentile(items, 0.99),
        "max": max(items, default=0.0),
        "cv": statistics.pstdev(items) / mean if mean else 0.0,
    }


def histogram_distribution(histogram: list[int]) -> dict[str, float]:
    total = sum(int(value) for value in histogram)
    if total == 0:
        return {
            key: 0.0
            for key in ("mean", "p50", "p95", "p99", "max")
        }
    mean = sum(
        index * int(frequency)
        for index, frequency in enumerate(histogram)
    ) / total

    def quantile(q: float) -> float:
        threshold = q * total
        cumulative = 0
        for index, frequency in enumerate(histogram):
            cumulative += int(frequency)
            if cumulative >= threshold:
                return float(index)
        return float(len(histogram) - 1)

    return {
        "mean": mean,
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": float(
            max(
                index
                for index, frequency in enumerate(histogram)
                if frequency
            )
        ),
    }


def decode_count_trace(encoded: dict) -> np.ndarray:
    formats = {"int16_le": ("<i2", 2), "int32_le": ("<i4", 4)}
    dtype = encoded.get("dtype")
    if dtype not in formats or encoded.get("codec") != "zlib_base64":
        raise ValueError("不支持的 ordered trace 编码")
    numpy_dtype, item_bytes = formats[dtype]
    raw = zlib.decompress(base64.b64decode(encoded["data"]))
    if len(raw) % item_bytes:
        raise ValueError("ordered trace payload 未按元素宽度对齐")
    count = len(raw) // item_bytes
    values = np.frombuffer(raw, dtype=numpy_dtype).astype(np.int64, copy=False)
    expected = math.prod(int(value) for value in encoded["shape"])
    if count != expected:
        raise ValueError("ordered trace shape 与 payload 不一致")
    return values


def ordered_queue(trace: Iterable[int] | np.ndarray, width: int) -> tuple[int, int]:
    """一条 anchor/cycle 到达、残差端按 width 服务时的完成周期和最大 backlog。"""

    values = np.asarray(trace, dtype=np.int64)
    if values.size == 0:
        return 0, 0
    service = np.where(values > 0, (values + width - 1) // width, 0)
    cumulative = np.cumsum(service - 1, dtype=np.int64)
    prefix_floor = np.minimum.accumulate(
        np.concatenate((np.array([0], dtype=np.int64), cumulative[:-1]))
    )
    reflected = cumulative - prefix_floor
    final_backlog = max(0, int(reflected[-1]))
    max_backlog = max(0, int(reflected.max(initial=0)))
    return int(values.size) + final_backlog, max_backlog


def histogram_service(histogram: list[int], width: int) -> int:
    return sum(
        int(frequency) * (ceil_div(delta, width) if delta else 0)
        for delta, frequency in enumerate(histogram)
    )


@dataclass
class Sample:
    line: str
    sample_id: int
    expanded_candidates: int
    quotient_candidates: int
    exact_collapses: int
    direct_score_lane_work: int
    quotient_score_lane_work: int
    direct_k_bits: int
    quotient_k_bits: int
    direct_projection_products: int
    quotient_projection_terms: int
    projection_destinations: int
    source_push_commands: int
    ordered_pooled_m4_cycles: int | None
    direct_wide_cycles: int
    direct_wide_lanes: int
    direct_serial_cycles: int
    serial_cycles_by_slice: dict[int, int]
    monolithic_cycles_by_slice: dict[int, int]
    ff_cycles_by_slice: dict[int, int]
    ff_max_backlog_by_slice: dict[int, int]
    evidence: str

    def validate(self) -> None:
        if self.quotient_candidates > self.expanded_candidates:
            raise ValueError("商表示候选数不能大于展开候选数")
        if self.exact_collapses != (
            self.expanded_candidates - self.quotient_candidates
        ):
            raise ValueError("exact collapse 守恒失败")
        if self.quotient_projection_terms > self.projection_destinations:
            raise ValueError("投影 term 数不能超过 destination 数")


@lru_cache(maxsize=1)
def load_motion_samples() -> tuple[Sample, ...]:
    profile = json.loads(MOTION_PROFILE.read_text())
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in profile["summary"]["h60_records"]:
        grouped[int(record["sample_id"])].append(record)

    samples = []
    for sample_id, records in sorted(grouped.items()):
        total = lambda field: sum(int(record[field]) for record in records)
        pair_total = total("pair_total")
        zero_updates = total("delta_zero_update_token_heads")
        changed_updates = total("delta_changed_token_heads")
        if pair_total != zero_updates + changed_updates:
            raise ValueError("Motion delta zero/changed 与 pair_total 不守恒")

        serial_cycles = {}
        ff_cycles = {}
        ff_backlog = {}
        decoded_traces = [
            decode_count_trace(record["delta_update_ordered_trace"])
            for record in records
        ]
        for slice_width in (2, 4, 8):
            serial_total = 0
            fused_total = 0
            fused_max_backlog = 0
            for trace in decoded_traces:
                serial_record, _ = ordered_queue(trace, slice_width)
                fused_record, backlog = ordered_queue(trace, 4 * slice_width)
                serial_total += serial_record
                fused_total += fused_record
                fused_max_backlog = max(fused_max_backlog, backlog)
            serial_cycles[slice_width] = serial_total
            ff_cycles[slice_width] = fused_total
            ff_backlog[slice_width] = fused_max_backlog

        ordered_pooled_m4_cycles = 0
        for record in records:
            term_trace = decode_count_trace(
                record[
                    "projection_gate_class_channel_terms_deploy_ordered_trace"
                ]
            )
            destination_trace = decode_count_trace(
                record["projection_gate_multicast_delivery_m1_ordered_trace"]
            )
            m4_trace = decode_count_trace(
                record["projection_gate_multicast_delivery_m4_ordered_trace"]
            )
            if not (
                term_trace.size
                == destination_trace.size
                == m4_trace.size
            ):
                raise ValueError("Motion projection ordered trace 长度不一致")
            ordered_pooled_m4_cycles += int(
                np.maximum(
                    term_trace,
                    (destination_trace + 3) // 4,
                ).sum()
            )

        sample = Sample(
            line="Motion",
            sample_id=sample_id,
            expanded_candidates=2 * pair_total,
            quotient_candidates=pair_total + changed_updates,
            exact_collapses=zero_updates,
            direct_score_lane_work=2 * pair_total * 32,
            quotient_score_lane_work=pair_total * 32
            + total("qk_temporal_update_elements"),
            direct_k_bits=total("k_temporal_baseline_reads"),
            quotient_k_bits=total("k_temporal_union_reads"),
            direct_projection_products=total("projection_baseline_active_lanes"),
            quotient_projection_terms=total(
                "projection_gate_class_channel_terms_deploy"
            ),
            projection_destinations=total(
                "projection_gate_multicast_delivery_m1"
            ),
            source_push_commands=total(
                "projection_gate_multicast_delivery_m4"
            ),
            ordered_pooled_m4_cycles=ordered_pooled_m4_cycles,
            direct_wide_cycles=pair_total,
            direct_wide_lanes=64,
            direct_serial_cycles=2 * pair_total,
            serial_cycles_by_slice=serial_cycles,
            # Motion fuse 后与同总宽度 monolithic residual worker 服务等价。
            monolithic_cycles_by_slice=dict(ff_cycles),
            ff_cycles_by_slice=ff_cycles,
            ff_max_backlog_by_slice=ff_backlog,
            evidence="[prof-ordered]",
        )
        sample.validate()
        samples.append(sample)
    return tuple(samples)


@lru_cache(maxsize=1)
def load_local_samples() -> tuple[Sample, ...]:
    profile = json.loads(LOCAL_PROFILE.read_text())
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in profile["records"]:
        grouped[int(record["sample_id"])].append(record)

    directions = ("up", "down", "left", "right")
    samples = []
    for sample_id, records in sorted(grouped.items()):
        total = lambda field: sum(int(record[field]) for record in records)
        exact = sum(
            int(record[f"{direction}_exact_k"])
            for record in records
            for direction in directions
        )
        token_heads = total("token_heads")
        directional = total("directional_valid_edges")
        expanded = total("valid_edges")
        quotient = token_heads + directional - exact
        if expanded != token_heads + directional:
            raise ValueError("Local5 self/directional candidate 不守恒")

        serial_cycles = {}
        monolithic_cycles = {}
        ff_cycles = {}
        for slice_width in (2, 4, 8):
            serial_total = 0
            monolithic_total = 0
            ff_total = 0
            for record in records:
                anchor_cycles = int(record["token_heads"])
                direction_service = [
                    histogram_service(
                        record[f"{direction}_delta_histogram"],
                        slice_width,
                    )
                    for direction in directions
                ]
                serial_total += max(anchor_cycles, sum(direction_service))
                monolithic_service = sum(
                    histogram_service(
                        record[f"{direction}_delta_histogram"],
                        4 * slice_width,
                    )
                    for direction in directions
                )
                monolithic_total += max(anchor_cycles, monolithic_service)
                ff_total += max(anchor_cycles, max(direction_service))
            serial_cycles[slice_width] = serial_total
            monolithic_cycles[slice_width] = monolithic_total
            ff_cycles[slice_width] = ff_total

        sample = Sample(
            line="Local5",
            sample_id=sample_id,
            expanded_candidates=expanded,
            quotient_candidates=quotient,
            exact_collapses=exact,
            direct_score_lane_work=expanded * 32,
            quotient_score_lane_work=token_heads * 32
            + total("directional_delta_lane_sum"),
            direct_k_bits=total("query_major_k_lane_reads"),
            quotient_k_bits=total("source_resident_k_lane_reads")
            + total("directional_delta_lane_sum"),
            direct_projection_products=total("naive_active_edge_products"),
            quotient_projection_terms=total("mfep_multicast_terms"),
            projection_destinations=total("destination_gate_lane_groups"),
            # Local5 pre-G0 尚无端口配对/打包 ordered 证据，按一目的/命令。
            source_push_commands=total("destination_gate_lane_groups"),
            ordered_pooled_m4_cycles=None,
            direct_wide_cycles=token_heads,
            direct_wide_lanes=5 * 32,
            direct_serial_cycles=expanded,
            serial_cycles_by_slice=serial_cycles,
            monolithic_cycles_by_slice=monolithic_cycles,
            ff_cycles_by_slice=ff_cycles,
            ff_max_backlog_by_slice={2: -1, 4: -1, 8: -1},
            evidence="[prof-preG0][缺失-ordered]",
        )
        sample.validate()
        samples.append(sample)
    return tuple(samples)


def aggregate_work(
    samples: list[Sample] | tuple[Sample, ...],
    tokens_per_window: int = 162,
) -> dict:
    fields = (
        "expanded_candidates",
        "quotient_candidates",
        "exact_collapses",
        "direct_score_lane_work",
        "quotient_score_lane_work",
        "direct_k_bits",
        "quotient_k_bits",
        "direct_projection_products",
        "quotient_projection_terms",
        "projection_destinations",
        "source_push_commands",
    )
    sums = {
        field: sum(int(getattr(sample, field)) for sample in samples)
        for field in fields
    }
    token_bits = max(1, math.ceil(math.log2(tokens_per_window)))
    multiplicity_bits = 1 if samples[0].line == "Motion" else 3
    score_bits = 7
    expanded_entry_bits = score_bits + token_bits
    quotient_entry_bits = score_bits + token_bits + multiplicity_bits
    expanded_buffer_bits = sums["expanded_candidates"] * expanded_entry_bits
    quotient_buffer_bits = sums["quotient_candidates"] * quotient_entry_bits

    return {
        **sums,
        "candidate_reduction": 1.0
        - sums["quotient_candidates"] / sums["expanded_candidates"],
        "score_lane_work_reduction": 1.0
        - sums["quotient_score_lane_work"] / sums["direct_score_lane_work"],
        "k_payload_reduction": 1.0
        - sums["quotient_k_bits"] / sums["direct_k_bits"],
        "unique_product_generation_reduction": 1.0
        - sums["quotient_projection_terms"]
        / sums["direct_projection_products"],
        "expanded_candidate_buffer_bits": expanded_buffer_bits,
        "quotient_candidate_buffer_bits": quotient_buffer_bits,
        "candidate_buffer_bit_reduction": 1.0
        - quotient_buffer_bits / expanded_buffer_bits,
        "accumulator_destination_work": (
            "不在 unique-term 比例中折叠；必须按 multiplicity 守恒并在 RTL "
            "统计 destination Acc RMW"
        ),
        "candidate_record_assumption": {
            "score_bits": score_bits,
            "token_bits": token_bits,
            "multiplicity_bits": multiplicity_bits,
        },
    }


def trcf_model(
    samples: list[Sample] | tuple[Sample, ...],
    *,
    projection_banks: int = 3,
    destination_issue_width: int = 4,
    source_command_bits: int = 64,
    term_header_bits: int = 48,
    destination_bits: int = 9,
    destination_flag_bits: int = 2,
) -> dict:
    """源端 push 与 term-resident 跨 term M4 合并的流量/服务量对照。"""

    source_global_bits = []
    receiver_global_bits = []
    receiver_local_list_bits = []
    source_cycles = []
    receiver_cycles = []
    for sample in samples:
        source_global_bits.append(
            sample.source_push_commands * source_command_bits
        )
        receiver_global_bits.append(
            sample.quotient_projection_terms * term_header_bits
        )
        receiver_local_list_bits.append(
            sample.projection_destinations
            * (destination_bits + destination_flag_bits)
        )
        source_cycles.append(sample.source_push_commands)
        receiver_cycles.append(
            sample.ordered_pooled_m4_cycles
            if sample.ordered_pooled_m4_cycles is not None
            else max(
                sample.quotient_projection_terms,
                ceil_div(
                    sample.projection_destinations,
                    destination_issue_width,
                ),
            )
        )

    source_global_mean = statistics.fmean(source_global_bits)
    receiver_global_mean = statistics.fmean(receiver_global_bits)
    receiver_local_mean = statistics.fmean(receiver_local_list_bits)
    source_cycle_mean = statistics.fmean(source_cycles)
    receiver_cycle_mean = statistics.fmean(receiver_cycles)
    ordered = samples[0].line == "Motion"
    global_reduction = 1.0 - receiver_global_mean / source_global_mean
    total_reduction = 1.0 - (
        receiver_global_mean + receiver_local_mean
    ) / source_global_mean
    cycle_speedup = source_cycle_mean / receiver_cycle_mean
    passes_screen = (
        global_reduction >= 0.50
        and total_reduction >= 0.15
        and cycle_speedup >= 1.00
    )
    return {
        "projection_banks": projection_banks,
        "destination_issue_width": destination_issue_width,
        "source_command_bits": source_command_bits,
        "term_header_bits": term_header_bits,
        "destination_entry_bits": destination_bits + destination_flag_bits,
        "source_push_global_bits": distribution(source_global_bits),
        "receiver_owned_global_bits": distribution(receiver_global_bits),
        "receiver_owned_local_destination_bits": distribution(
            receiver_local_list_bits
        ),
        "global_fabric_bit_reduction": global_reduction,
        "global_plus_local_bit_reduction_vs_source_global": total_reduction,
        "source_push_cycles": distribution(source_cycles),
        "receiver_owned_service_cycles": distribution(receiver_cycles),
        "optimistic_service_upper_bound": cycle_speedup,
        "ordered_evidence": (
            "[prof-ordered-row-bound]"
            if ordered
            else "[prof-preG0][缺失-ordered]"
        ),
        "passes_model_screen": passes_screen,
        "candidate_status": (
            "conditional_requires_per_term_trace_and_local_build_rtl"
            if passes_screen and ordered
            else "blocked_by_local_ordered_destination_trace"
            if passes_screen
            else "rejected"
        ),
        "critical_assumption": (
            "destination list 在 projection bank 本地构建，且多个 resident "
            "term 的 M4 尾部可跨 term 合并；若 destination 仍逐项跨全局 "
            "fabric，或产品上下文不足以支持跨 term 合并，收益失效"
        ),
    }


def class_quotient_normalizer_profile() -> dict:
    """评估将所有同 score candidate 精确折叠到 class domain 的共享后端。"""

    motion_profile = json.loads(MOTION_PROFILE.read_text())
    motion_records = motion_profile["summary"]["h60_records"]
    motion_score_hist = [0] * 163
    motion_occupied_hist = [0] * 164
    for record in motion_records:
        for index, value in enumerate(record["h67_score_q7_histogram"]):
            motion_score_hist[index] += int(value)
        for index, value in enumerate(
            record["row_all_occupied_classes_h67_histogram"]
        ):
            motion_occupied_hist[index] += int(value)
    motion_support = [
        index for index, value in enumerate(motion_score_hist) if value
    ]
    motion_candidates = sum(
        int(record["token_total"]) for record in motion_records
    )
    motion_kzero = sum(
        int(record["token_kzero"]) for record in motion_records
    )
    motion_kzero_fold_classes = sum(
        int(record["row_kzero_fold_classes_sum_h67"])
        for record in motion_records
    )
    motion_active_members = motion_candidates - motion_kzero
    # Current SCS 在 ST_SUM_ACTIVE 和 ST_EMIT 对 active-K 各调用一次 exp，
    # zero-K folded class 只在 denominator 阶段调用一次。
    motion_current_scs_exp_evals = (
        2 * motion_active_members + motion_kzero_fold_classes
    )
    motion_rows = sum(int(record["row_total"]) for record in motion_records)
    motion_class_evals = sum(
        index * int(frequency)
        for index, frequency in enumerate(motion_occupied_hist)
    )

    local_profile = json.loads(LOCAL_PROFILE.read_text())
    local_summary = local_profile["summary"]
    local_support = [
        index - 256
        for index, value in enumerate(
            local_summary["valid_score_histogram_offset256"]
        )
        if value
    ]

    score_domain = 163
    count_bits = 9
    class_id_bits = 8
    gate_bits = 9
    return {
        "name": "All-Candidate Class-Quotient Normalizer",
        "abbreviation": "ACQN-163",
        "shared_structure": {
            "score_domain": "0..162，由 HEAD_DIM=32 的现有整数 score 公式界定",
            "score_class_entries": score_domain,
            "class_id_bits": class_id_bits,
            "count_bits_for_t450": count_bits,
            "occupied_bitmap_bits": score_domain,
            "histogram_bits": score_domain * count_bits,
            "gate_cache_bits": score_domain * gate_bits,
            "optional_exp_cache_bits": score_domain * 16,
            "class_state_bits_per_context": (
                score_domain
                + score_domain * count_bits
                + score_domain * gate_bits
            ),
            "class_state_with_exp_cache_bits_per_context": (
                score_domain
                + score_domain * count_bits
                + score_domain * gate_bits
                + score_domain * 16
            ),
            "member_score_field_saving_vs_score16_bits_at_t450": (
                450 * (16 - class_id_bits)
            ),
        },
        "Motion": {
            "evidence": "[prof-ordered]",
            "rows": motion_rows,
            "candidate_entries": motion_candidates,
            "observed_score_support_count": len(motion_support),
            "observed_score_min": min(motion_support),
            "observed_score_max": max(motion_support),
            "occupied_classes_per_row": histogram_distribution(
                motion_occupied_hist
            ),
            "naive_candidate_exp_evals": motion_candidates,
            "current_scs_exp_evals": motion_current_scs_exp_evals,
            "acqn_recompute_exp_evals": 2 * motion_class_evals,
            "acqn_cached_exp_evals": motion_class_evals,
            "recompute_exp_eval_reduction_vs_current_scs": 1.0
            - 2 * motion_class_evals / motion_current_scs_exp_evals,
            "cached_exp_eval_reduction_vs_current_scs": 1.0
            - motion_class_evals / motion_current_scs_exp_evals,
            "active_member_entries": motion_active_members,
            "class_stationary_gate_context_loads_upper_bound": (
                motion_class_evals
            ),
            "gate_context_load_reduction_upper_bound": 1.0
            - motion_class_evals / motion_active_members,
            "current_active_score_field_bits": motion_active_members * 16,
            "acqn_active_class_id_field_bits": motion_active_members * 8,
            "active_member_score_field_bit_reduction": 0.5,
            "status": "eligible_for_bit_exact_model_and_rtl_prototype",
        },
        "Local5": {
            "evidence": "[prof-preG0][缺失-row-score-class-trace]",
            "rows": int(local_summary["token_heads"]),
            "candidate_entries": int(local_summary["valid_edges"]),
            "observed_score_support_count": len(local_support),
            "observed_score_min": min(local_support),
            "observed_score_max": max(local_support),
            "classes_per_row_topology_upper_bound": 5,
            "pre_g0_gate_cardinality_mean": float(
                local_summary["gate_cardinality_mean"]
            ),
            "pre_g0_gate_cardinality_p95": int(
                local_summary["gate_cardinality_p95"]
            ),
            "status": "blocked_by_post_g0_row_score_class_trace",
        },
        "exact_contract": [
            "denominator 按 class_count * exp(score_class-row_max) 求和",
            "gate 每 occupied score class 只量化一次",
            "member list 保留 K/token/destination identity 并按 class gate 回放",
            "class gate 驻留并直接驱动 term builder，不物化逐 token gate",
            "zero-K 计入 denominator，但不进入 gated-K member replay",
            "不得把 class-eval 减少写成等比例端到端加速",
        ],
    }


def score_core_dse(samples: list[Sample] | tuple[Sample, ...]) -> dict:
    direct_cycles = [sample.direct_wide_cycles for sample in samples]
    direct_lanes = samples[0].direct_wide_lanes
    direct_mean = statistics.fmean(direct_cycles)
    records = []
    for slice_width in (2, 4, 8):
        serial_cycles = [
            sample.serial_cycles_by_slice[slice_width] for sample in samples
        ]
        ff_cycles = [
            sample.ff_cycles_by_slice[slice_width] for sample in samples
        ]
        monolithic_cycles = [
            sample.monolithic_cycles_by_slice[slice_width]
            for sample in samples
        ]
        ff_mean = statistics.fmean(ff_cycles)
        monolithic_mean = statistics.fmean(monolithic_cycles)
        ff_lanes = 32 + 4 * slice_width
        throughput_retention = direct_mean / ff_mean
        nominal_lane_throughput_index = (
            throughput_retention * direct_lanes / ff_lanes
        )
        tail = distribution(ff_cycles)
        ordered = samples[0].line == "Motion"
        record = {
            "slice_width": slice_width,
            "mode": (
                f"4x{slice_width} fused temporal"
                if ordered
                else f"4x{slice_width} split directional"
            ),
            "direct_wide_lanes": direct_lanes,
            "ff_score_lanes": ff_lanes,
            "lane_reduction_vs_direct_wide": 1.0 - ff_lanes / direct_lanes,
            "direct_wide_cycles": distribution(direct_cycles),
            "serial_tare_cycles": distribution(serial_cycles),
            "monolithic_residual_cycles": distribution(monolithic_cycles),
            "ff_cycles": tail,
            "throughput_retention_vs_direct_wide": throughput_retention,
            "nominal_lane_throughput_index": nominal_lane_throughput_index,
            "nominal_lane_warning": (
                "full-score lane 与 residual lane 不等价；该值只描述 RTL "
                "结构计数，不能替代面积或功耗"
            ),
            "ff_speedup_vs_equal_lane_monolithic": (
                monolithic_mean / ff_mean
            ),
            "p99_over_mean": tail["p99"] / tail["mean"] if tail["mean"] else 0.0,
            "max_pending_service_cycles": (
                max(
                    sample.ff_max_backlog_by_slice[slice_width]
                    for sample in samples
                )
                if ordered
                else None
            ),
            "ordered_tail_evidence": (
                "[prof-ordered]" if ordered else "[缺失-ordered]"
            ),
            "passes_throughput_retention_0p95": throughput_retention >= 0.95,
            "passes_tail_1p25": (
                tail["p99"] <= 1.25 * tail["mean"] if ordered else None
            ),
        }
        record["passes_nominal_lane_reduction_20pct"] = (
            record["lane_reduction_vs_direct_wide"] >= 0.20
        )
        record["passes_service_debt_256"] = (
            record["max_pending_service_cycles"] <= 256 if ordered else None
        )
        record["passes_model_screen"] = (
            record["passes_throughput_retention_0p95"]
            and record["passes_nominal_lane_reduction_20pct"]
            and (
                record["passes_service_debt_256"]
                if record["passes_service_debt_256"] is not None
                else True
            )
        )
        record["candidate_status"] = (
            "eligible_for_full_boundary_rtl_prototype"
            if record["passes_model_screen"] and ordered
            else "blocked_by_local_ordered_trace"
            if record["passes_model_screen"]
            else "rejected"
        )
        record["fission_fusion_novelty_status"] = (
            "rejected_no_gain_vs_monolithic"
            if record["ff_speedup_vs_equal_lane_monolithic"] < 1.05
            else "supported_by_ordered_service_model"
            if ordered
            else "blocked_by_local_ordered_trace"
        )
        records.append(record)

    passing = [record for record in records if record["passes_model_screen"]]
    passing.sort(
        key=lambda record: (
            -record["nominal_lane_throughput_index"],
            record["ff_score_lanes"],
        )
    )
    return {
        "line": samples[0].line,
        "samples": len(samples),
        "records": records,
        "recommended_by_model": passing[0] if passing else None,
    }


def render_markdown(report: dict) -> str:
    def pct(value: float) -> str:
        return f"{100.0 * value:.2f}%"

    lines = [
        "# 拓扑商流与裂分/融合残差阵列 DSE",
        "",
        "## 1. 结论",
        "",
        "本轮提出的候选不是新的编码名字，而是一项可实现的核组织：",
        "",
        "> **Topology-Quotient Fission-Fusion Engine（TQ-FFE）**：",
        "> 以一个 32-lane anchor score 核和四个小 residual slice 为基本阵列；",
        "> Local5 把四个 slice 裂分给上、下、左、右四条固定关系，Motion 将四个",
        "> slice 融合成 temporal delta 通路。完全相同的候选通过 multiplicity 精确",
        "> 折叠，multiplicity 一直保留到加权归一化和 term projection。",
        "",
        "该方案针对当前 Local5 串行 TARE 的结构性问题：让四个邻域 residual",
        "不再共用一个顺序事务通路；目前只有服务量下界，尚未证明已经修复。",
        "本轮提出 Motion/Local5 可重构物理组织假设，",
        "但尚未证明它比两个专用核面积或能耗更低；两条算法也不具有相同注意力语义。",
        "",
        "## 2. 精确工作量消融",
        "",
        "| 主线 | 候选减少 | score lane-work 减少 | K payload 减少 | unique product 生成减少 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("Motion", "Local5"):
        work = report["lines"][key]["work"]
        lines.append(
            f"| {key} | {pct(work['candidate_reduction'])} | "
            f"{pct(work['score_lane_work_reduction'])} | "
            f"{pct(work['k_payload_reduction'])} | "
            f"{pct(work['unique_product_generation_reduction'])} |"
        )

    lines.extend(
        [
            "",
            "候选减少只使用可证明的拓扑 identity：Motion 使用 delta-zero，",
            "Local5 使用 neighbor-K 与 self-K 完全一致。没有使用“score 恰好相等”",
            "的事后统计，也没有近似删除。projection term 与 destination 分开计数，",
            "避免把 multicast fanout 免费消掉。",
            "",
            "## 3. Score 核组织 DSE",
            "",
        ]
    )
    for key in ("Motion", "Local5"):
        lines.extend(
            [
                f"### 3.{1 if key == 'Motion' else 2} {key}",
                "",
                "| slice | 组织 | score lanes | nominal lane 减少 | residual 服务保持 | 相对同 lane monolithic | sample-total p99/mean | 最大待服务周期 | evidence | 原型状态 | fission/fusion 新意 |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for record in report["lines"][key]["dse"]["records"]:
            lines.append(
                f"| {record['slice_width']} | {record['mode']} | "
                f"{record['ff_score_lanes']} | "
                f"{pct(record['lane_reduction_vs_direct_wide'])} | "
                f"{record['throughput_retention_vs_direct_wide']:.3f}x | "
                f"{record['ff_speedup_vs_equal_lane_monolithic']:.3f}x | "
                f"{record['p99_over_mean']:.3f} | "
                f"{record['max_pending_service_cycles'] if record['max_pending_service_cycles'] is not None else '缺失'} | "
                f"{record['ordered_tail_evidence']} | "
                f"{record['candidate_status']} | "
                f"{record['fission_fusion_novelty_status']} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. 当前架构决策",
            "",
            "1. **Motion 只进入完整边界 RTL 校准**：有 ordered trace，可按模型",
            "   推荐 slice 宽度实现 fused temporal residual，并以 Direct64 为",
            "   同端口基线。其 fused 模式与同 lane monolithic residual 服务相同，",
            "   不能把 fission/fusion 写成 Motion 创新。",
            "2. **Local5 是 fission 架构创新候选**：W4 split 相对同 lane",
            "   monolithic16 的服务量下界提高约 13.8%，但当前 profile 不能证明",
            "   每方向 FIFO 的瞬时 overflow 与 transaction tail；必须等待",
            "   fullres post-G0 ordered trace。",
            "3. **butterfly 不是贡献本身**：它只负责把 32-bit delta mask 压紧到四个",
            "   slice。贡献是 topology-declared fission/fusion 与 multiplicity-carrying",
            "   quotient flow；必须用无 butterfly、固定串行 TARE、Direct-wide 三个",
            "   基线做消融。",
            "4. **DCTF 降级为后端实现**：bank-local projection 只承接 quotient term，",
            "   不再单独列为论文核心创新。",
            "",
            "## 5. Term-Resident 跨 term 合并 Fabric",
            "",
            "TQ-FFE 的第二个系统级候选是 **Term-Resident Cross-Term",
            "Coalescing Fabric（TRCF）**：unique term 只经过全局 fabric 一次；",
            "destination list 在 projection bank 本地建立。多个 resident term",
            "共享同一组 M4 destination 端口，使不同 term 的尾部 destination 可以",
            "合并，避免 per-term packing 碎片。3-bank 或双 context 本身不是创新。",
            "",
            "| 主线 | 全局 fabric bits 减少 | 加入本地 destination SRAM 后总 bits 减少 | 服务周期代理 | ordered 状态 | 决策 |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for key in ("Motion", "Local5"):
        fabric = report["lines"][key]["trcf"]
        lines.append(
            f"| {key} | {pct(fabric['global_fabric_bit_reduction'])} | "
            f"{pct(fabric['global_plus_local_bit_reduction_vs_source_global'])} | "
            f"{fabric['optimistic_service_upper_bound']:.3f}x | "
            f"{fabric['ordered_evidence']} | {fabric['candidate_status']} |"
        )
    lines.extend(
        [
            "",
            "Motion 的周期只是逐 row 上界：允许同一 row 内跨 term 理想合并，但",
            "现有 trace 没有逐 term destination/port class，不能证明真实无冲突。",
            "如果 destination 仍逐项通过全局 fabric，或 product context 无法驻留",
            "多个 term，TRCF 立即淘汰。Local5 等待 post-G0 ordered stream。",
            "",
            "## 6. ACQN-163 共享 Class-Quotient Normalizer",
            "",
        ]
    )
    acqn = report["acqn"]
    motion_norm = acqn["Motion"]
    local_norm = acqn["Local5"]
    lines.extend(
        [
            "ACQN-163 将所有 candidate 的 score 映射到固定 `0..162` class",
            "domain。denominator 对 class multiplicity 求和，gate 每个 occupied",
            "class 量化一次，member list 再按 class gate 回放 K/token identity。",
            "",
            "| 主线 | observed score support | occupied class/row | exp class-eval 减少 | 状态 |",
            "|---|---:|---:|---:|---|",
            (
                f"| Motion | {motion_norm['observed_score_support_count']} "
                f"({motion_norm['observed_score_min']}.."
                f"{motion_norm['observed_score_max']}) | "
                f"mean {motion_norm['occupied_classes_per_row']['mean']:.2f}, "
                f"p95 {motion_norm['occupied_classes_per_row']['p95']:.0f} | "
                f"{pct(motion_norm['recompute_exp_eval_reduction_vs_current_scs'])} "
                f"(重算) / "
                f"{pct(motion_norm['cached_exp_eval_reduction_vs_current_scs'])} "
                f"(exp cache) | "
                f"{motion_norm['status']} |"
            ),
            (
                f"| Local5 | {local_norm['observed_score_support_count']} "
                f"({local_norm['observed_score_min']}.."
                f"{local_norm['observed_score_max']}) | <=5 topology bound | "
                f"缺 post-G0 row trace | {local_norm['status']} |"
            ),
            "",
            "Motion 的 exp class-eval 减少不等于端到端加速；member replay、",
            "histogram update、max/denominator、gate cache 和 backpressure 仍需",
            "完整 RTL。Local5 当前只证明 class domain 兼容，不证明实际收益。",
            "",
            "## 7. DATE 晋级门槛",
            "",
            "- 同吞吐 Direct-wide 对照下，score 核目标工艺面积或动态功耗至少降低 20%；",
            "- 端到端 attention-to-projection EDP 至少改善 15%；",
            "- Motion ordered trace 的 mean/p95/p99 与 FIFO 最大占用全部闭环；",
            "- Local5 post-G0/fullres ordered trace 到达前，不进入主结果表；",
            "- 两条线均必须保持 hardware-order bit-exact，multiplicity 必须进入分母；",
            "- 最终使用同一 SRAM macro、SDC、DC/STA/SAIF 规则比较 Direct、串行",
            "  TARE 与 TQ-FFE。",
            "- TRCF 的全局 fabric bits 至少下降 50%，计入本地 destination SRAM",
            "  后总流量至少下降 15%，且逐 term RAW/port conflict 后端到端",
            "  EDP 仍改善 15%。",
            "",
            "## 8. 证据边界",
            "",
            "- `[prof-ordered]`：Motion profile100 ordered trace；",
            "- `[prof-preG0]`：Local5 浮点前门控 profile，只用于预筛；",
            "- `[模型]`：本文件的 residual 服务量、nominal lane 和 work 代理；",
            "- nominal lane 不是门面积；sample-total p99 不是 transaction tail；",
            "- 不包含 DC 面积、STA、SAIF、SRAM macro 或芯片功耗；",
            "- Local5 四方向 `max(service)` 是解耦 FIFO 的服务量下界，不是",
            "  cycle-accurate 结果。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report() -> dict:
    motion = load_motion_samples()
    local = load_local_samples()
    return {
        "schema": "topology_quotient_fission_fusion_v1",
        "generated_date": "2026-07-30",
        "architecture": {
            "name": "Topology-Quotient Fission-Fusion Engine",
            "abbreviation": "TQ-FFE",
            "anchor_lanes": 32,
            "residual_slices": 4,
            "modes": {
                "Motion": "四 slice 融合为 temporal residual engine",
                "Local5": "四 slice 裂分为 up/down/left/right residual engine",
            },
            "exact_contract": (
                "仅折叠 topology-declared identity；multiplicity 参与 Shiftmax "
                "denominator 与 projection destination 守恒"
            ),
        },
        "sources": {
            "motion_profile": {
                "path": str(MOTION_PROFILE.relative_to(ROOT.parent)),
                "sha256": sha256_file(MOTION_PROFILE),
            },
            "local_profile": {
                "path": str(LOCAL_PROFILE.relative_to(ROOT)),
                "sha256": sha256_file(LOCAL_PROFILE),
            },
        },
        "lines": {
            "Motion": {
                "evidence": motion[0].evidence,
                "work": aggregate_work(motion),
                "dse": score_core_dse(motion),
                "trcf": trcf_model(motion),
            },
            "Local5": {
                "evidence": local[0].evidence,
                "work": aggregate_work(local),
                "dse": score_core_dse(local),
                "trcf": trcf_model(local),
            },
        },
        "acqn": class_quotient_normalizer_profile(),
        "kill_thresholds": {
            "throughput_retention_vs_direct_wide": ">=0.95",
            "nominal_lane_count_reduction": ">=20%，只作 RTL 原型筛选",
            "ordered_p99_over_mean": "<=1.25",
            "ordered_max_pending_service_debt": (
                "<=256 cycles before full-boundary RTL prototype"
            ),
            "target_score_core_area_or_dynamic_power_reduction": ">=20%",
            "target_attention_to_projection_edp_improvement": ">=15%",
        },
        "non_claims": [
            "不把 lane 数当作目标工艺面积",
            "不把 work proxy 当作功耗",
            "不把 Local5 histogram 服务下界当作 ordered 周期",
            "不把 butterfly zero skipper 的既有思想声明为本工作发明",
            "不把当前结果写成 full encoder PPA",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out / "report.md").write_text(render_markdown(report) + "\n")
    print(args.out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
