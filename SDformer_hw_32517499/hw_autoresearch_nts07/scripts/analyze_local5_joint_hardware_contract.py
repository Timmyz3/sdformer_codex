#!/usr/bin/env python3
"""把 Local5 同窗全 head trace 收口为硬件架构筛选合同。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import analyze_local5_active_tcfm5_postg0 as tcfm5


SAMPLING_ID = "uniform_plan_window_all_heads_v1"
TOKENS = 450
OUT_LANES = 32
VECTOR_BITS = 1024
STAGE_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
STAGE_DEPTHS = (2, 2, 6, 2)
STAGE_WINDOWS = (440, 120, 30, 10)
PLAN_SEED = 20260809
REQUIRED_ARRAYS = {
    "descriptor_group_offsets",
    "descriptor_source_id",
    "descriptor_source_plane",
    "descriptor_source_y",
    "descriptor_source_x",
    "descriptor_k_bitmap",
    "descriptor_incoming_gates",
    "descriptor_valid_mask",
    "source_term_count",
    "source_gate_count",
    "source_delivery_count",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_percentile(
    values: np.ndarray, weights: np.ndarray, percentile: float
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.shape != weights.shape or values.size == 0:
        raise ValueError("加权分位数输入必须是等长非空一维数组")
    if np.any(weights < 0) or not float(weights.sum()) > 0:
        raise ValueError("加权分位数权重必须非负且总和为正")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    target = min(max(percentile, 0.0), 100.0) / 100.0 * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def weighted_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.size == 0:
        raise ValueError("统计输入必须是等长非空数组")
    return {
        "mean": float(np.average(values, weights=weights)),
        "p50": weighted_percentile(values, weights, 50),
        "p95": weighted_percentile(values, weights, 95),
        "p99": weighted_percentile(values, weights, 99),
        "sample_observed_max": float(values.max()),
    }


def vector_traffic_words(heads: int, output_tiles: int) -> dict[str, int]:
    """只计算投影后向量物化边界，不包含两侧共同的 term RMW。"""
    if heads <= 0 or output_tiles <= 0:
        raise ValueError("head 与 output tile 必须为正")
    b0v = 3 * heads * TOKENS * output_tiles
    b2v = TOKENS * output_tiles
    return {
        "b0v_1rw_vector_accesses": b0v,
        "b2v_1rw_vector_accesses": b2v,
        "eliminated_1rw_vector_accesses": b0v - b2v,
        "b0v_payload_bytes": b0v * VECTOR_BITS // 8,
        "b2v_payload_bytes": b2v * VECTOR_BITS // 8,
        "shared_scalar_results": TOKENS * OUT_LANES * output_tiles,
    }


def relation_pair_metrics(
    left: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float]:
    left_k, left_gates, left_valid, left_terms = left
    right_k, right_gates, right_valid, right_terms = right
    if left_k.shape != right_k.shape or left_k.size != TOKENS:
        raise ValueError("跨 head 关系比较必须覆盖同窗 450 个 source")
    exact = (
        (left_k == right_k)
        & np.all(left_gates == right_gates, axis=1)
        & (left_valid == right_valid)
    )
    gate_equal = np.all(left_gates == right_gates, axis=1) & (
        left_valid == right_valid
    )
    left_active = left_terms > 0
    right_active = right_terms > 0
    union = int(np.count_nonzero(left_active | right_active))
    jaccard = (
        int(np.count_nonzero(left_active & right_active)) / union
        if union
        else 1.0
    )
    return float(exact.mean()), float(gate_equal.mean()), float(jaccard)


def relation_pair_decomposition(
    left: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    """拆开静默重合与非空 descriptor 的真实精确复用机会。"""
    left_k, left_gates, left_valid, left_terms = left
    right_k, right_gates, right_valid, right_terms = right
    if left_k.shape != right_k.shape or left_k.size != TOKENS:
        raise ValueError("跨 head 关系分账必须覆盖同窗 450 个 source")
    exact = (
        (left_k == right_k)
        & np.all(left_gates == right_gates, axis=1)
        & (left_valid == right_valid)
    )
    left_active = left_terms > 0
    right_active = right_terms > 0
    both_empty = ~left_active & ~right_active
    both_active = left_active & right_active
    nonempty_union = left_active | right_active
    nonempty_count = int(np.count_nonzero(nonempty_union))
    return {
        "both_empty_source_fraction": float(both_empty.mean()),
        "both_active_source_fraction": float(both_active.mean()),
        "exact_nonempty_source_fraction": (
            float(np.count_nonzero(exact & nonempty_union) / nonempty_count)
            if nonempty_count
            else 1.0
        ),
    }


def relation_equivalence_metrics(
    k_bitmaps: np.ndarray,
    gates: np.ndarray,
    valid_masks: np.ndarray,
    term_counts: np.ndarray,
) -> dict[str, float]:
    """按window/source统计head间完整descriptor等价类。"""
    if (
        k_bitmaps.ndim != 2
        or gates.shape != (*k_bitmaps.shape, 5)
        or valid_masks.shape != k_bitmaps.shape
        or term_counts.shape != k_bitmaps.shape
        or k_bitmaps.shape[1] != TOKENS
    ):
        raise ValueError("等价类统计shape失效")
    heads = k_bitmaps.shape[0]
    total_classes = 0
    identical_sources = 0
    empty_sources = 0
    largest_cluster_sum = 0
    for source in range(TOKENS):
        counts: dict[tuple[int, tuple[int, ...], int], int] = defaultdict(int)
        for head in range(heads):
            signature = (
                int(k_bitmaps[head, source]),
                tuple(int(value) for value in gates[head, source]),
                int(valid_masks[head, source]),
            )
            counts[signature] += 1
        total_classes += len(counts)
        identical_sources += int(len(counts) == 1)
        empty_sources += int(bool(np.all(term_counts[:, source] == 0)))
        largest_cluster_sum += max(counts.values())
    return {
        "exact_descriptor_dedup_fraction": 1.0 - total_classes / (heads * TOKENS),
        "all_head_identical_source_fraction": identical_sources / TOKENS,
        "all_head_empty_source_fraction": empty_sources / TOKENS,
        "largest_exact_cluster_fraction": largest_cluster_sum / (heads * TOKENS),
        "equivalence_classes_per_source": total_classes / TOKENS,
    }


def load_plan(path: Path) -> tuple[dict[tuple[int, int, int], dict], dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "local5_uniform_joint_window_plan_v1"
        or value.get("sampling_id") != SAMPLING_ID
        or value.get("seed") != PLAN_SEED
    ):
        raise ValueError("selection plan schema/sampling/seed 错误")
    rows: dict[tuple[int, int, int], dict] = {}
    for row in value.get("records") or []:
        key = (int(row["sample"]), int(row["stage"]), int(row["block"]))
        sample, stage, block = key
        if (
            key in rows
            or not 0 <= sample < 100
            or not 0 <= stage < 4
            or not 0 <= block < STAGE_DEPTHS[stage]
            or int(row["heads"]) != STAGE_HEADS[stage]
            or int(row["batch_windows"]) != STAGE_WINDOWS[stage]
            or not 0 <= int(row["window"]) < STAGE_WINDOWS[stage]
            or float(row["inclusion_probability"]) != 1.0 / STAGE_WINDOWS[stage]
            or float(row["analysis_weight"]) != float(STAGE_WINDOWS[stage])
        ):
            raise ValueError(f"selection plan 记录非法: {key}")
        rows[key] = row
    expected = {
        (sample, stage, block)
        for sample in range(100)
        for stage, depth in enumerate(STAGE_DEPTHS)
        for block in range(depth)
    }
    if set(rows) != expected:
        raise ValueError("正式 selection plan 未覆盖100x12完整key set")
    return rows, value


def analyze(manifest_path: Path, plan_path: Path, chunk_size: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qualification = manifest.get("qualification") or {}
    if (
        manifest.get("schema") != "et3_ordered_term_trace_v2"
        or manifest.get("sampling", {}).get("method") != SAMPLING_ID
        or qualification.get("qualified") is not True
        or qualification.get("processed_samples") != 100
        or qualification.get("attached_blocks") != 12
        or qualification.get("captured_groups") != 13800
    ):
        raise ValueError("joint-head manifest 未通过正式 100-sample/all12/all-head 合同")
    if manifest.get("sampling", {}).get("selection_plan_sha256") != sha256(plan_path):
        raise ValueError("manifest 与 selection plan SHA 不一致")
    plan, plan_value = load_plan(plan_path)
    if manifest.get("cohort_sha256") != plan_value.get("cohort_sha256"):
        raise ValueError("manifest 与 selection plan cohort 不一致")

    payload_path = manifest_path.parent / str(manifest["payload_file"])
    if manifest.get("payload_sha256") != sha256(payload_path):
        raise ValueError("payload SHA 不一致")

    identity_path = Path(str(manifest.get("run_identity_file", ""))).resolve()
    cohort_path = manifest_path.parent / str(manifest.get("cohort_file", ""))
    gpu_audit_path = manifest_path.parent / "gpu_exclusivity_audit.json"
    if (
        not identity_path.is_file()
        or manifest.get("run_identity_file_sha256") != sha256(identity_path)
        or not cohort_path.is_file()
        or manifest.get("cohort_file_sha256") != sha256(cohort_path)
        or not gpu_audit_path.is_file()
    ):
        raise ValueError("joint profile 身份/cohort/GPU审计产物缺失或SHA失效")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    gpu_audit = json.loads(gpu_audit_path.read_text(encoding="utf-8"))
    if (
        identity.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")
        or identity.get("config_sha256") != manifest.get("config_sha256")
        or identity.get("cohort_sha256") != manifest.get("cohort_sha256")
        or identity.get("selection_plan_sha256") != sha256(plan_path)
        or gpu_audit.get("schema") != "local5_joint_gpu_exclusivity_audit_v1"
        or gpu_audit.get("status") != "PASS"
        or gpu_audit.get("identity_sha256") != sha256(identity_path)
        or gpu_audit.get("manifest_sha256") != sha256(manifest_path)
        or gpu_audit.get("payload_sha256") != sha256(payload_path)
        or gpu_audit.get("foreign_compute_pids") != []
    ):
        raise ValueError("joint profile 身份或GPU独占审计未通过")
    payload = np.load(payload_path, mmap_mode="r")
    if not REQUIRED_ARRAYS.issubset(payload.files):
        missing = sorted(REQUIRED_ARRAYS - set(payload.files))
        raise ValueError(f"payload 缺少硬件合同数组: {missing}")

    groups = manifest.get("groups") or []
    offsets = np.asarray(payload["descriptor_group_offsets"], dtype=np.int64)
    if len(groups) != 13800 or len(offsets) != len(groups) + 1:
        raise ValueError("group 与 descriptor offset 数量不一致")
    if offsets[0] != 0 or np.any(np.diff(offsets) != TOKENS):
        raise ValueError("每个 joint-head group 必须恰有 450 个 source")

    descriptor_count = int(offsets[-1])
    expected_source_ids = np.tile(np.arange(TOKENS, dtype=np.int64), len(groups))
    source_ids = np.asarray(payload["descriptor_source_id"], dtype=np.int64)
    planes = np.asarray(payload["descriptor_source_plane"], dtype=np.int64)
    ys = np.asarray(payload["descriptor_source_y"], dtype=np.int64)
    xs = np.asarray(payload["descriptor_source_x"], dtype=np.int64)
    k_bitmaps = np.asarray(payload["descriptor_k_bitmap"])
    incoming_gates = np.asarray(payload["descriptor_incoming_gates"])
    valid_masks = np.asarray(payload["descriptor_valid_mask"])
    expected_spatial = expected_source_ids % 225
    if (
        source_ids.shape != (descriptor_count,)
        or planes.shape != (descriptor_count,)
        or ys.shape != (descriptor_count,)
        or xs.shape != (descriptor_count,)
        or k_bitmaps.shape != (descriptor_count,)
        or incoming_gates.shape != (descriptor_count, 5)
        or valid_masks.shape != (descriptor_count,)
        or not np.array_equal(source_ids, expected_source_ids)
        or not np.array_equal(planes, expected_source_ids // 225)
        or not np.array_equal(ys, expected_spatial // 15)
        or not np.array_equal(xs, expected_spatial % 15)
    ):
        raise ValueError("descriptor source-id/坐标/shape合同失效")

    group_weights = np.zeros(len(groups), dtype=np.float64)
    rows_by_window: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(groups):
        key = (int(row["sample"]), int(row["stage"]), int(row["block"]))
        plan_row = plan.get(key)
        stage = key[1]
        if (
            plan_row is None
            or int(row["window"]) != int(plan_row["window"])
            or int(row["heads"]) != STAGE_HEADS[stage]
        ):
            raise ValueError(f"group 未绑定预提交同窗计划: {index}")
        group_weights[index] = float(plan_row["analysis_weight"])
        rows_by_window[(*key, int(row["window"]))].append(index)
    if len(rows_by_window) != 1200:
        raise ValueError("joint window 数量不是 100x12")
    for key, indices in rows_by_window.items():
        heads = STAGE_HEADS[key[1]]
        if len(indices) != heads or [int(groups[i]["head"]) for i in indices] != list(range(heads)):
            raise ValueError(f"同窗 head 顺序/覆盖错误: {key}")

    metric_names = (
        "active_sources",
        "product_terms",
        "destination_updates",
        "linear5_cycles",
        "tcfm5_cycles",
    )
    descriptor_metrics = {
        name: np.zeros(descriptor_count, dtype=np.int32) for name in metric_names
    }
    fanout_hist = np.zeros(6, dtype=np.float64)
    source_weights = np.repeat(group_weights, TOKENS)
    for start in range(0, descriptor_count, chunk_size):
        stop = min(start + chunk_size, descriptor_count)
        gates = np.asarray(payload["descriptor_incoming_gates"][start:stop])
        valid = np.asarray(payload["descriptor_valid_mask"][start:stop])
        k_bitmap = np.asarray(payload["descriptor_k_bitmap"][start:stop])
        chunk = tcfm5.analyze_descriptor_chunk(
            gates,
            valid,
            k_bitmap,
            np.asarray(payload["descriptor_source_plane"][start:stop]),
            np.asarray(payload["descriptor_source_y"][start:stop]),
            np.asarray(payload["descriptor_source_x"][start:stop]),
            height=15,
            width=15,
        )
        for name in metric_names:
            descriptor_metrics[name][start:stop] = chunk[name]

        lanes = np.fromiter(
            (int(value).bit_count() for value in k_bitmap),
            dtype=np.int32,
            count=len(k_bitmap),
        )
        role_valid = np.stack(
            [(((valid >> role) & 1) != 0) & (gates[:, role] != 0) for role in range(5)],
            axis=1,
        )
        for representative in range(5):
            unique = role_valid[:, representative].copy()
            for previous in range(representative):
                unique &= ~(
                    role_valid[:, previous]
                    & (gates[:, previous] == gates[:, representative])
                )
            fanout = (
                role_valid
                & (gates == gates[:, representative : representative + 1])
            ).sum(axis=1)
            for value in range(1, 6):
                mask = unique & (fanout == value)
                fanout_hist[value] += float(
                    np.sum(lanes[mask] * source_weights[start:stop][mask])
                )

    group_metrics = {
        name: np.add.reduceat(values, offsets[:-1]).astype(np.int64)
        for name, values in descriptor_metrics.items()
    }
    source_terms = np.asarray(payload["source_term_count"], dtype=np.int64)
    source_gates = np.asarray(payload["source_gate_count"], dtype=np.int64)
    deliveries = np.asarray(payload["source_delivery_count"], dtype=np.int64)
    if (
        not np.array_equal(descriptor_metrics["product_terms"], source_terms)
        or not np.array_equal(descriptor_metrics["destination_updates"], deliveries)
        or np.any(group_metrics["tcfm5_cycles"] != group_metrics["product_terms"])
    ):
        raise ValueError("descriptor 重建与 producer 工作量不一致")

    active_source = source_terms > 0
    gate_stats = weighted_summary(source_gates[active_source], source_weights[active_source])
    group_stats = {
        name: weighted_summary(values, group_weights)
        for name, values in group_metrics.items()
    }
    legal_1rw_cycles = 2 * group_metrics["tcfm5_cycles"]
    group_stats["legal_1rw_two_phase_cycle_proxy"] = weighted_summary(
        legal_1rw_cycles, group_weights
    )
    group_stats["active_scan_plus_1rw_proxy"] = weighted_summary(
        15 + np.maximum(group_metrics["active_sources"], legal_1rw_cycles),
        group_weights,
    )

    overlap_exact: list[float] = []
    overlap_gate: list[float] = []
    overlap_active: list[float] = []
    overlap_both_empty: list[float] = []
    overlap_both_active: list[float] = []
    overlap_exact_nonempty: list[float] = []
    overlap_weights: list[float] = []
    descriptor_k = payload["descriptor_k_bitmap"]
    descriptor_gates = payload["descriptor_incoming_gates"]
    descriptor_valid = payload["descriptor_valid_mask"]
    for key, indices in rows_by_window.items():
        weight = float(plan[key[:3]]["analysis_weight"])
        for left_pos, left_index in enumerate(indices):
            left_slice = slice(int(offsets[left_index]), int(offsets[left_index + 1]))
            left = (
                np.asarray(descriptor_k[left_slice]),
                np.asarray(descriptor_gates[left_slice]),
                np.asarray(descriptor_valid[left_slice]),
                source_terms[left_slice],
            )
            for right_index in indices[left_pos + 1 :]:
                right_slice = slice(int(offsets[right_index]), int(offsets[right_index + 1]))
                right = (
                    np.asarray(descriptor_k[right_slice]),
                    np.asarray(descriptor_gates[right_slice]),
                    np.asarray(descriptor_valid[right_slice]),
                    source_terms[right_slice],
                )
                exact, gate_equal, active_jaccard = relation_pair_metrics(left, right)
                decomposition = relation_pair_decomposition(left, right)
                overlap_exact.append(exact)
                overlap_gate.append(gate_equal)
                overlap_active.append(active_jaccard)
                overlap_both_empty.append(decomposition["both_empty_source_fraction"])
                overlap_both_active.append(decomposition["both_active_source_fraction"])
                overlap_exact_nonempty.append(
                    decomposition["exact_nonempty_source_fraction"]
                )
                overlap_weights.append(weight)
    pair_weights = np.asarray(overlap_weights, dtype=np.float64)
    relation_overlap = {
        "head_pairs": len(pair_weights),
        "exact_descriptor_source_fraction": weighted_summary(
            np.asarray(overlap_exact), pair_weights
        ),
        "gate_valid_source_fraction": weighted_summary(
            np.asarray(overlap_gate), pair_weights
        ),
        "active_source_jaccard": weighted_summary(
            np.asarray(overlap_active), pair_weights
        ),
        "both_empty_source_fraction": weighted_summary(
            np.asarray(overlap_both_empty), pair_weights
        ),
        "both_active_source_fraction": weighted_summary(
            np.asarray(overlap_both_active), pair_weights
        ),
        "exact_nonempty_source_fraction": weighted_summary(
            np.asarray(overlap_exact_nonempty), pair_weights
        ),
    }

    equivalence_rows: dict[str, list[float]] = defaultdict(list)
    equivalence_weights: list[float] = []
    equivalence_stages: list[int] = []
    for key, indices in rows_by_window.items():
        head_k = []
        head_gates = []
        head_valid = []
        head_terms = []
        for index in indices:
            begin = int(offsets[index])
            end = int(offsets[index + 1])
            head_k.append(k_bitmaps[begin:end])
            head_gates.append(incoming_gates[begin:end])
            head_valid.append(valid_masks[begin:end])
            head_terms.append(source_terms[begin:end])
        metrics = relation_equivalence_metrics(
            np.asarray(head_k),
            np.asarray(head_gates),
            np.asarray(head_valid),
            np.asarray(head_terms),
        )
        for name, value in metrics.items():
            equivalence_rows[name].append(value)
        equivalence_weights.append(float(plan[key[:3]]["analysis_weight"]))
        equivalence_stages.append(key[1])
    equivalence_weight_array = np.asarray(equivalence_weights, dtype=np.float64)
    equivalence_stage_array = np.asarray(equivalence_stages, dtype=np.int64)
    equivalence_summary = {
        name: {
            "overall": weighted_summary(np.asarray(values), equivalence_weight_array),
            "per_stage": [
                {
                    "stage": stage,
                    **weighted_summary(
                        np.asarray(values)[equivalence_stage_array == stage],
                        equivalence_weight_array[equivalence_stage_array == stage],
                    ),
                }
                for stage in range(4)
            ],
        }
        for name, values in equivalence_rows.items()
    }

    stage_rows = []
    for stage in range(4):
        mask = np.asarray([int(row["stage"]) == stage for row in groups])
        stage_rows.append(
            {
                "stage": stage,
                "groups": int(mask.sum()),
                "product_terms": weighted_summary(
                    group_metrics["product_terms"][mask], group_weights[mask]
                ),
                "destination_updates": weighted_summary(
                    group_metrics["destination_updates"][mask], group_weights[mask]
                ),
                "legal_1rw_two_phase_cycle_proxy": weighted_summary(
                    legal_1rw_cycles[mask], group_weights[mask]
                ),
            }
        )

    traffic_by_sample = {
        name: np.zeros(100, dtype=np.float64)
        for name in vector_traffic_words(1, 1)
    }
    for key in rows_by_window:
        sample, stage, block, _ = key
        weight = float(plan[(sample, stage, block)]["analysis_weight"])
        heads = STAGE_HEADS[stage]
        for name, value in vector_traffic_words(heads, heads).items():
            traffic_by_sample[name][sample] += value * weight

    total_weighted_terms = float(fanout_hist.sum())
    fanout = {
        "weighted_term_histogram_1_to_5": [float(value) for value in fanout_hist[1:]],
        "weighted_mean_active_banks_per_term": (
            sum(index * fanout_hist[index] for index in range(1, 6))
            / total_weighted_terms
            if total_weighted_terms
            else 0.0
        ),
        "same_term_same_bank_conflicts": 0,
        "proof_check": "TCFM5 cycles == product terms for every captured group",
    }

    return {
        "schema": "local5_joint_hardware_contract_v1",
        "status": "PROFILE_CONTRACT_COMPLETE_NOT_RTL",
        "evidence": "[prof]+[模型]",
        "input": {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "payload": str(payload_path.resolve()),
            "payload_sha256": sha256(payload_path),
            "selection_plan": str(plan_path.resolve()),
            "selection_plan_sha256": sha256(plan_path),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "config_sha256": manifest.get("config_sha256"),
            "cohort_sha256": manifest.get("cohort_sha256"),
            "run_identity_file": str(identity_path),
            "run_identity_file_sha256": sha256(identity_path),
            "cohort_file": str(cohort_path.resolve()),
            "cohort_file_sha256": sha256(cohort_path),
            "gpu_exclusivity_audit": str(gpu_audit_path.resolve()),
            "gpu_exclusivity_audit_sha256": sha256(gpu_audit_path),
            "samples": 100,
            "joint_windows": 1200,
            "head_groups": len(groups),
            "descriptors": descriptor_count,
        },
        "term_and_cycle_statistics": group_stats,
        "active_source_gate_cardinality": gate_stats,
        "cross_head_relation_overlap": relation_overlap,
        "window_normalized_relation_equivalence": equivalence_summary,
        "five_bank_term_fanout": fanout,
        "vector_boundary_frame_statistics": {
            name: weighted_summary(values, np.ones_like(values))
            for name, values in traffic_by_sample.items()
        },
        "per_stage": stage_rows,
        "promotion_contract": {
            "new_candidate_weighted_mean_cycle_improvement_min": 0.20,
            "p95_must_not_regress": True,
            "requires_legal_1rw": True,
            "requires_acc32_bit_exact_before_rtl_promotion": True,
        },
        "limits": [
            "mean按逆包含概率加权；分位数为归一化逆概率加权经验CDF；sample_observed_max不是总体max。",
            "1RW cycle保守地对每个五色term计一拍READ加一拍WRITE；未利用first-touch直写，不是RTL顶层周期。",
            "向量流量只覆盖 projection 后物化边界，不含两候选共同的 term RMW。",
            "跨 head exact overlap 比较 K bitmap、五路 gate 和 valid mask；不代表跨 block 复用。",
            "本报告不是 OpenROAD、DC、STA、SAIF 或 PTPX 结果。",
        ],
        "source_bindings": {
            "analyzer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "tcfm5_model": {"path": str(Path(tcfm5.__file__).resolve()), "sha256": sha256(Path(tcfm5.__file__).resolve())},
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    term = report["term_and_cycle_statistics"]
    overlap = report["cross_head_relation_overlap"]
    traffic = report["vector_boundary_frame_statistics"]
    fanout = report["five_bank_term_fanout"]
    lines = [
        "# Local5 同窗全 Head 硬件统计合同",
        "",
        "## 结论",
        "",
        "本报告是 **[prof]+[模型]**，覆盖 100 个 sample、12 个 attention block、",
        "1200 个预提交同窗和 13800 个 head group。所有分布按 window inclusion probability",
        "加权；它不是 RTL 顶层周期或 ASIC PPA。",
        "",
        "## 工作量分布",
        "",
        "| 指标 | mean | p50 | p95 | p99 | sample observed max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("active_sources", "active source / head-window"),
        ("product_terms", "gate-lane term / head-window"),
        ("destination_updates", "Acc update / head-window"),
        ("legal_1rw_two_phase_cycle_proxy", "合法1RW两相周期代理"),
        ("active_scan_plus_1rw_proxy", "active-scan+1RW代理"),
    ):
        row = term[key]
        lines.append(
            f"| {label} | {row['mean']:.3f} | {row['p50']:.0f} | {row['p95']:.0f} | "
            f"{row['p99']:.0f} | {row['sample_observed_max']:.0f} |"
        )
    gate = report["active_source_gate_cardinality"]
    lines += [
        "",
        "## 关系与五色拓扑",
        "",
        f"active source 的 gate 基数 mean/p95/p99 为 `{gate['mean']:.3f}/"
        f"{gate['p95']:.0f}/{gate['p99']:.0f}`。",
        f"TCFM5 每 term 活跃 bank 均值为 `{fanout['weighted_mean_active_banks_per_term']:.3f}`；"
        "同一 term 同 bank 冲突为 `0`，并逐 group 检查 `TCFM5 cycles == product terms`。",
        "",
        "| 跨 head 指标 | mean | p50 | p95 | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("exact_descriptor_source_fraction", "完整 relation descriptor 相同比例"),
        ("gate_valid_source_fraction", "gate+valid 相同比例"),
        ("active_source_jaccard", "active-source Jaccard"),
        ("both_empty_source_fraction", "双方静默 source 比例"),
        ("both_active_source_fraction", "双方活跃 source 比例"),
        ("exact_nonempty_source_fraction", "非空并集内完整 descriptor 精确比例"),
    ):
        row = overlap[key]
        lines.append(
            f"| {label} | {row['mean']:.4f} | {row['p50']:.4f} | "
            f"{row['p95']:.4f} | {row['p99']:.4f} |"
        )
    equivalence = report["window_normalized_relation_equivalence"]
    lines += [
        "",
        "## Window归一的Relation等价类",
        "",
        "以每个window/source为单位，不按head pair数重复加权：",
        "",
        "| 指标 | mean | p50 | p95 | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("exact_descriptor_dedup_fraction", "可精确消除descriptor比例"),
        ("all_head_identical_source_fraction", "全head相同source比例"),
        ("all_head_empty_source_fraction", "全head空source比例"),
        ("largest_exact_cluster_fraction", "最大精确共享簇比例"),
    ):
        row = equivalence[key]["overall"]
        lines.append(
            f"| {label} | {row['mean']:.4f} | {row['p50']:.4f} | "
            f"{row['p95']:.4f} | {row['p99']:.4f} |"
        )
    lines += [
        "",
        "## 向量边界流量",
        "",
        "| 每帧代理 | mean | p95 | p99 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("b0v_1rw_vector_accesses", "B0v 1024-bit 1RW访问"),
        ("b2v_1rw_vector_accesses", "B2v 1024-bit 1RW访问"),
        ("eliminated_1rw_vector_accesses", "B2v消除访问"),
        ("shared_scalar_results", "共同最终Acc32结果"),
    ):
        row = traffic[key]
        lines.append(
            f"| {label} | {row['mean']:.0f} | {row['p95']:.0f} | {row['p99']:.0f} |"
        )
    lines += [
        "",
        "## 晋级规则",
        "",
        "新 Local5 候选只有在同一合法 1RW 模型下预测加权平均周期改善 `>=20%`、",
        "p95 不退化，并在实现后保持 Acc32 bit-exact，才允许新增 RTL。否则停止 Local5",
        "微机制扩展，下一轮转向 Motion 多样本证据。",
        "",
        "## 边界",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limits"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("chunk-size 必须为正数")
    report = analyze(args.manifest, args.selection_plan, args.chunk_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.output_dir / "report.json"
    report_md = args.output_dir / "report.md"
    commit_path = args.output_dir / "commit.json"
    json_temporary = report_json.with_suffix(".json.tmp")
    md_temporary = report_md.with_suffix(".md.tmp")
    commit_temporary = commit_path.with_suffix(".json.tmp")
    json_temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_temporary.write_text(render_markdown(report), encoding="utf-8")
    os.replace(json_temporary, report_json)
    os.replace(md_temporary, report_md)
    commit = {
        "schema": "local5_joint_hardware_contract_commit_v1",
        "status": "COMMITTED",
        "report_json_sha256": sha256(report_json),
        "report_md_sha256": sha256(report_md),
        "manifest_sha256": report["input"]["manifest_sha256"],
        "payload_sha256": report["input"]["payload_sha256"],
        "selection_plan_sha256": report["input"]["selection_plan_sha256"],
        "identity_sha256": report["input"]["run_identity_file_sha256"],
        "cohort_file_sha256": report["input"]["cohort_file_sha256"],
        "gpu_exclusivity_audit_sha256": report["input"]["gpu_exclusivity_audit_sha256"],
    }
    commit_temporary.write_text(
        json.dumps(commit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(commit_temporary, commit_path)
    print("PASS Local5 joint-head hardware contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
