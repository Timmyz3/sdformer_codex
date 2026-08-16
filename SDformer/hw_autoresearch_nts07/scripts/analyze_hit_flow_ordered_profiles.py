#!/usr/bin/env python3
"""审计H67/H68 ordered profile并生成HIT-Flow架构冻结建议。"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any


VALUE_FIELDS = (
    "near_integer_ratio",
    "binary01_ratio",
    "ternary_ratio",
)
DELTA_FIELDS = (
    "cross_sample_exact_equal_ratio",
    "cross_sample_active_xor_ratio",
    "cross_sample_sign_class_change_ratio",
    "cross_sample_mean_abs_delta",
    "cross_sample_normalized_mean_abs_delta",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_count_trace(encoded: dict[str, Any]) -> list[int]:
    dtype = encoded.get("dtype")
    formats = {
        "int16_le": ("<h", 2),
        "int32_le": ("<i", 4),
    }
    if dtype not in formats or encoded.get("codec") != "zlib_base64":
        raise ValueError("不支持的ordered count trace编码")
    raw = zlib.decompress(base64.b64decode(encoded["data"]))
    count = 1
    for dim in encoded.get("shape", []):
        count *= int(dim)
    fmt, item_bytes = formats[dtype]
    if len(raw) != item_bytes * count:
        raise ValueError("ordered count trace字节数与shape不一致")
    return [item[0] for item in struct.iter_unpack(fmt, raw)]


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def csd_nonzero_digits(value: int) -> int:
    """返回非负整数的规范有符号数字非零项数。"""

    if value < 0:
        raise ValueError("CSD输入必须非负")
    digits = 0
    current = value
    while current:
        if current & 1:
            digit = -1 if current & 3 == 3 else 1
            current -= digit
            digits += 1
        current >>= 1
    return digits


def weighted_ratio(rows: list[dict[str, Any]], field: str, weight: str) -> float:
    denominator = sum(int(row.get(weight, 0)) for row in rows if row.get(field) is not None)
    numerator = sum(
        float(row[field]) * int(row.get(weight, 0))
        for row in rows
        if row.get(field) is not None
    )
    return numerator / denominator if denominator else 0.0


def validate_profile(data: dict[str, Any], model: str) -> None:
    if int(data.get("samples", 0)) != 100:
        raise ValueError(f"{model}: profile样本数不是100")
    if not data.get("ordered_trace", False):
        raise ValueError(f"{model}: ordered_trace未开启")
    summary = data.get("summary", {})
    if len(summary.get("h60_records", [])) != 1200:
        raise ValueError(f"{model}: H60记录数不是1200")
    if not summary.get("operator_by_scope") or not summary.get("operator_rows"):
        raise ValueError(f"{model}: 缺少逐算子运行时分账")
    if not summary.get("cross_sample_by_stage"):
        raise ValueError(f"{model}: 缺少同sequence跨样本统计")
    projection_trace_fields = (
        "projection_baseline_active_lanes_ordered_trace",
        "projection_class_channel_terms_ttx_ordered_trace",
        "projection_class_channel_terms_h67_ordered_trace",
        "projection_class_channel_max_fanout_ttx_ordered_trace",
        "projection_class_channel_max_fanout_h67_ordered_trace",
        "projection_active_classes_ttx_ordered_trace",
        "projection_active_classes_h67_ordered_trace",
        "projection_gate_class_channel_terms_deploy_ordered_trace",
        "projection_gate_class_channel_max_fanout_deploy_ordered_trace",
        "projection_active_gate_classes_deploy_ordered_trace",
    )
    for index, row in enumerate(summary.get("h60_records", [])):
        for field in projection_trace_fields:
            if field not in row:
                raise ValueError(f"{model}: H60记录{index}缺少{field}")
        for width in (1, 2, 4, 8, 16):
            for variant in ("ttx", "h67"):
                field = f"projection_multicast_delivery_{variant}_m{width}_ordered_trace"
                if field not in row:
                    raise ValueError(f"{model}: H60记录{index}缺少{field}")
            field = f"projection_gate_multicast_delivery_m{width}_ordered_trace"
            if field not in row:
                raise ValueError(f"{model}: H60记录{index}缺少{field}")
        for group_windows in (1, 2, 4, 8, 16):
            for metric in (
                "terms", "active_lanes", "active_classes", "max_fanout", "window_count"
            ):
                field = f"projection_gate_group_{metric}_g{group_windows}_ordered_trace"
                if field not in row:
                    raise ValueError(f"{model}: H60记录{index}缺少{field}")
            for width in (1, 2, 4, 8, 16):
                field = (
                    f"projection_gate_group_delivery_g{group_windows}_m{width}_ordered_trace"
                )
                if field not in row:
                    raise ValueError(f"{model}: H60记录{index}缺少{field}")
    pair = summary.get("binary_temporal_pairs", {})
    for field in (
        "k_temporal_baseline_reads", "k_temporal_union_reads",
        "k_temporal_intersection_reuse", "k_temporal_union_read_ratio",
        "k_temporal_exact_reuse_ratio",
        "projection_baseline_active_lanes", "projection_class_channel_terms_ttx",
        "projection_class_channel_terms_h67", "projection_class_channel_ratio_ttx",
        "projection_class_channel_ratio_h67",
        "projection_gate_class_channel_terms_deploy",
        "projection_gate_class_channel_ratio_deploy",
        "projection_gate_class_channel_term_histogram",
        "projection_active_lane_gate_q17_histogram",
    ):
        if field not in pair:
            raise ValueError(f"{model}: binary_temporal_pairs缺少{field}")
    if int(pair["k_temporal_baseline_reads"]) != (
        int(pair["k_temporal_union_reads"]) +
        int(pair["k_temporal_intersection_reuse"])
    ):
        raise ValueError(f"{model}: K时间复用守恒失败")
    for variant in ("ttx", "h67"):
        if int(pair[f"projection_class_channel_terms_{variant}"]) > int(
            pair["projection_baseline_active_lanes"]
        ):
            raise ValueError(f"{model}: {variant.upper()}类通道投影项超过活动K lane基线")
    if int(pair["projection_gate_class_channel_terms_deploy"]) > int(
        pair["projection_baseline_active_lanes"]
    ):
        raise ValueError(f"{model}: 最终gate类通道投影项超过活动K lane基线")
    stage_rows = [row for row in summary.get("activation_records", []) if str(row.get("kind", "")).startswith("stage_")]
    if len(stage_rows) != 800:
        raise ValueError(f"{model}: stage边界记录数不是800")
    for row in stage_rows:
        for field in ("finite_ratio", "value_min", "value_max", *VALUE_FIELDS):
            if field not in row:
                raise ValueError(f"{model}: {row.get('name')}缺少{field}")
    for row in summary.get("atlif_rows", []):
        for field in (
            "deployment_dead_result", "quant_sample_events", "recomputed_reference_mismatch",
            "parameter_q4_event_mismatch", "parameter_q6_event_mismatch", "parameter_q8_event_mismatch",
        ):
            if field not in row:
                raise ValueError(f"{model}: ATLIF {row.get('name')}缺少{field}")


def stage_value_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("kind", "")).startswith("stage_"):
            grouped[str(row["name"])].append(row)
    result = []
    for name, group in sorted(grouped.items()):
        result.append({
            "name": name,
            "records": len(group),
            "elements_per_frame": int(group[0]["elements"]),
            "value_min": min(float(row["value_min"]) for row in group),
            "value_max": max(float(row["value_max"]) for row in group),
            "value_absmax": max(float(row["value_absmax"]) for row in group),
            **{field: weighted_ratio(group, field, "finite_count") for field in VALUE_FIELDS},
        })
    return result


def operator_summary(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = int(data["samples"])
    scopes = []
    for row in data["summary"]["operator_by_scope"]:
        scopes.append({
            **row,
            "dense_macs_per_frame": float(row["dense_macs"]) / samples,
            "activity_weighted_macs_per_frame": float(row["activity_weighted_macs_proxy"]) / samples,
        })
    operators = []
    for row in data["summary"]["operator_rows"]:
        operators.append({
            "name": row["name"],
            "operator": row["operator"],
            "scope": row["scope"],
            "input_activity": row["input_activity"],
            "dense_macs_per_frame": float(row["dense_macs"]) / samples,
            "activity_weighted_macs_per_frame": float(row["activity_weighted_macs_proxy"]) / samples,
            "input_sample_binary01_ratio": row.get("input_sample_binary01_ratio"),
            "input_sample_ternary_ratio": row.get("input_sample_ternary_ratio"),
        })
    operators.sort(key=lambda row: row["activity_weighted_macs_per_frame"], reverse=True)
    return scopes, operators[:12]


def atlif_quant_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    live = [row for row in rows if not row.get("deployment_dead_result", False)]
    events = sum(int(row["quant_sample_events"]) for row in live)
    result: dict[str, Any] = {
        "live_modules": len(live),
        "sample_events": events,
        "reference_mismatches": sum(int(row["recomputed_reference_mismatch"]) for row in live),
    }
    for bits in (4, 6, 8):
        mismatch = sum(int(row[f"parameter_q{bits}_event_mismatch"]) for row in live)
        result[f"q{bits}_event_mismatches"] = mismatch
        result[f"q{bits}_event_mismatch_ratio"] = mismatch / events if events else 0.0
    for denominator, suffix in ((128, "1_128"), (64, "1_64"), (32, "1_32"), (16, "1_16")):
        count = sum(int(row[f"margin_abs_le_{suffix}"]) for row in live)
        result[f"margin_abs_le_{suffix}_ratio"] = count / events if events else 0.0
    return result


def pair_and_bank_summary(data: dict[str, Any]) -> dict[str, Any]:
    pair = data["summary"]["binary_temporal_pairs"]
    both = int(pair["pair_kzero_both"])
    same_h67 = int(pair["pair_kzero_same_class_h67"])
    result = {
        "pair_empty_ratio": float(pair["pair_empty_ratio"]),
        "both_kzero_ratio": float(pair["pair_kzero_both_ratio"]),
        "both_kzero_same_class_h67_ratio_all_pairs": float(pair["pair_kzero_same_class_h67_ratio"]),
        "both_kzero_same_class_h67_ratio_conditional": same_h67 / both if both else 0.0,
        "spatial_persistence_ratio": float(pair["spatial_persistence_ratio"]),
        "spatial_change_ratio": float(pair["spatial_change_ratio"]),
        "k_temporal_union_read_ratio": float(pair["k_temporal_union_read_ratio"]),
        "k_temporal_exact_reuse_ratio": float(pair["k_temporal_exact_reuse_ratio"]),
        "projection_class_channel_ratio_ttx": float(pair["projection_class_channel_ratio_ttx"]),
        "projection_class_channel_ratio_h67": float(pair["projection_class_channel_ratio_h67"]),
        "projection_gate_class_channel_ratio_deploy": float(
            pair["projection_gate_class_channel_ratio_deploy"]
        ),
    }
    for direction in ("horizontal", "vertical", "diag_down", "diag_up"):
        result[f"spatial_{direction}_adjacent_ratio"] = float(pair[f"spatial_{direction}_adjacent_ratio"])
    bank_rows = []
    for banks in (4, 8):
        for mapping in ("rowmajor", "diagonal", "xor"):
            key = f"spatial_bank{banks}_{mapping}_cycles_mean"
            bank_rows.append({"banks": banks, "mapping": mapping, "cycles_mean": float(pair[key])})
    result["bank_mappings"] = bank_rows
    return result


def projection_multicast_summary(data: dict[str, Any], model: str) -> dict[str, Any]:
    rows = data["summary"]["h60_records"]
    fields = {
        "baseline": "projection_baseline_active_lanes_ordered_trace",
        "ttx": "projection_class_channel_terms_ttx_ordered_trace",
        "h67": "projection_class_channel_terms_h67_ordered_trace",
        "ttx_fanout": "projection_class_channel_max_fanout_ttx_ordered_trace",
        "h67_fanout": "projection_class_channel_max_fanout_h67_ordered_trace",
        "ttx_classes": "projection_active_classes_ttx_ordered_trace",
        "h67_classes": "projection_active_classes_h67_ordered_trace",
        "deploy": "projection_gate_class_channel_terms_deploy_ordered_trace",
        "deploy_fanout": "projection_gate_class_channel_max_fanout_deploy_ordered_trace",
        "deploy_classes": "projection_active_gate_classes_deploy_ordered_trace",
    }
    for width in (1, 2, 4, 8, 16):
        fields[f"h67_delivery_m{width}"] = (
            f"projection_multicast_delivery_h67_m{width}_ordered_trace"
        )
        fields[f"deploy_delivery_m{width}"] = (
            f"projection_gate_multicast_delivery_m{width}_ordered_trace"
        )
    for group_windows in (1, 2, 4, 8, 16):
        for metric in (
            "terms", "active_lanes", "active_classes", "max_fanout", "window_count"
        ):
            fields[f"deploy_group_{metric}_g{group_windows}"] = (
                f"projection_gate_group_{metric}_g{group_windows}_ordered_trace"
            )
        for width in (1, 2, 4, 8, 16):
            fields[f"deploy_group_delivery_g{group_windows}_m{width}"] = (
                f"projection_gate_group_delivery_g{group_windows}_m{width}_ordered_trace"
            )
    decoded = {
        name: [value for row in rows for value in decode_count_trace(row[field])]
        for name, field in fields.items()
    }
    pair = data["summary"]["binary_temporal_pairs"]
    if sum(decoded["baseline"]) != int(pair["projection_baseline_active_lanes"]):
        raise ValueError("类通道多播baseline ordered trace与聚合值不一致")
    if sum(decoded["deploy"]) != int(pair["projection_gate_class_channel_terms_deploy"]):
        raise ValueError("最终gate类通道ordered trace与聚合值不一致")
    if sum(decoded["deploy_group_terms_g1"]) != sum(decoded["deploy"]):
        raise ValueError("G=1窗口组必须等于逐row最终gate类通道项")
    result: dict[str, Any] = {"rows": len(decoded["baseline"])}
    for name in fields:
        values = decoded[name]
        result[f"{name}_sum"] = sum(values)
        result[f"{name}_mean"] = sum(values) / len(values) if values else 0.0
        for label, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99), ("max", 1.0)):
            result[f"{name}_{label}"] = percentile(values, quantile)
    baseline = result["baseline_sum"]
    for name in ("ttx", "h67"):
        result[f"{name}_weighted_ratio"] = result[f"{name}_sum"] / baseline if baseline else 0.0
        result[f"{name}_product_reduction"] = 1.0 - result[f"{name}_weighted_ratio"] if baseline else 0.0
    deploy = result["deploy_sum"]
    score_variant = "h67" if model == "H67" else "ttx"
    score_terms = result[f"{score_variant}_sum"]
    result["deploy_weighted_ratio"] = deploy / baseline if baseline else 0.0
    result["deploy_product_reduction"] = 1.0 - deploy / baseline if baseline else 0.0
    result["score_to_gate_term_reduction"] = (
        1.0 - deploy / score_terms if score_terms else 0.0
    )
    for group_windows in (1, 2, 4, 8, 16):
        grouped = result[f"deploy_group_terms_g{group_windows}_sum"]
        result[f"deploy_group_g{group_windows}_reduction_vs_row"] = (
            1.0 - grouped / deploy if deploy else 0.0
        )
        valid_windows = result[f"deploy_group_window_count_g{group_windows}_sum"]
        group_contexts = len(decoded[f"deploy_group_window_count_g{group_windows}"])
        result[f"deploy_group_g{group_windows}_slot_utilization"] = (
            valid_windows / (group_contexts * group_windows)
            if group_contexts else 0.0
        )
        if result[f"deploy_group_active_lanes_g{group_windows}_sum"] != baseline:
            raise ValueError(f"G={group_windows}窗口组活动lane总数不守恒")
        if valid_windows != result["rows"]:
            raise ValueError(f"G={group_windows}窗口组有效窗口数不守恒")
    term_histogram = [int(value) for value in pair["projection_gate_class_channel_term_histogram"]]
    lane_histogram = [int(value) for value in pair["projection_active_lane_gate_q17_histogram"]]
    if sum(term_histogram) != deploy:
        raise ValueError("gate码类通道项直方图与deploy项数不一致")
    if sum(lane_histogram) != baseline:
        raise ValueError("活动lane gate码直方图与baseline不一致")
    result["active_gate_codes"] = sum(value > 0 for value in term_histogram)
    for name, histogram in (("term", term_histogram), ("active_lane", lane_histogram)):
        total = sum(histogram)
        popcount_sum = sum(index.bit_count() * count for index, count in enumerate(histogram))
        csd_sum = sum(csd_nonzero_digits(index) * count for index, count in enumerate(histogram))
        result[f"gate_{name}_binary_digits_mean"] = popcount_sum / total if total else 0.0
        result[f"gate_{name}_csd_digits_mean"] = csd_sum / total if total else 0.0
    return result


def analyze_case(model: str, profile_path: Path) -> dict[str, Any]:
    data = load_json(profile_path)
    validate_profile(data, model)
    scopes, top_operators = operator_summary(data)
    pair_dse_path = profile_path.parent / "binary_temporal_pair_arch_dse.json"
    pair_dse = load_json(pair_dse_path).get("model_summary", {}) if pair_dse_path.exists() else {}
    stage_values = stage_value_summary(data["summary"]["activation_records"])
    cross_sample = data["summary"]["cross_sample_by_stage"]
    high_traffic = [row for row in cross_sample if row["name"] in {
        "S0.skip", "S0.x_out", "S1.skip", "S1.x_out", "S2.skip", "S2.x_out",
    }]
    persistent_candidate = any(
        float(row["cross_sample_exact_equal_ratio"]) >= 0.70
        or float(row["cross_sample_active_xor_ratio"]) <= 0.10
        for row in high_traffic
    )
    rpi_binary = all(
        float(row["binary01_ratio"]) == 1.0
        for row in stage_values
        if row["name"] in {"S0.skip", "S1.skip", "S2.skip"}
    )
    quant = atlif_quant_summary(data["summary"]["atlif_rows"])
    pair_bank = pair_and_bank_summary(data)
    projection_multicast = projection_multicast_summary(data, model)
    return {
        "model": model,
        "profile": str(profile_path),
        "samples": int(data["samples"]),
        "stage_values": stage_values,
        "cross_sample_by_stage": cross_sample,
        "operator_by_scope": scopes,
        "top_operators": top_operators,
        "atlif_quant": quant,
        "pair_and_bank": pair_bank,
        "projection_multicast": projection_multicast,
        "pair_dse": pair_dse,
        "decisions": {
            "persistent_htt_profile_gate": persistent_candidate,
            "rpi_all_long_skips_binary": rpi_binary,
            "atlif_q8_sample_zero_mismatch": (
                quant["reference_mismatches"] == 0 and quant["q8_event_mismatches"] == 0
            ),
            "pccc_coverage_gate_70pct": (
                pair_bank["both_kzero_same_class_h67_ratio_conditional"] >= 0.70
            ),
            "bmrf_traffic_gate_8pct": (
                float(pair_dse.get("adaptive_traffic_reduction_vs_dense", 0.0)) >= 0.08
            ),
        },
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        "# HIT-Flow Ordered Profile架构决策报告",
        "",
        "**状态**：真实profile100统计；PPA与full825量化仍需单独签核",
        "",
    ]
    for case in result["models"]:
        lines += [
            f"## {case['model']}",
            "",
            "### 逐算子操作分账",
            "",
            "| 范围 | dense MAC/帧 | 活动率加权MAC/帧 | 输入活动率 |",
            "|---|---:|---:|---:|",
        ]
        for row in case["operator_by_scope"]:
            lines.append(
                f"| {row['scope']} | {row['dense_macs_per_frame']:.0f} | "
                f"{row['activity_weighted_macs_per_frame']:.0f} | {float(row['input_activity']):.6f} |"
            )
        lines += [
            "",
            "### Stage值域与表示",
            "",
            "| 边界 | min | max | absmax | integer | binary01 | ternary |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in case["stage_values"]:
            lines.append(
                f"| {row['name']} | {row['value_min']:.6f} | {row['value_max']:.6f} | "
                f"{row['value_absmax']:.6f} | {row['near_integer_ratio']:.6f} | "
                f"{row['binary01_ratio']:.6f} | {row['ternary_ratio']:.6f} |"
            )
        lines += [
            "",
            "### 同sequence相邻样本变化",
            "",
            "| 边界 | 对数 | 精确相等 | active翻转 | 符号类变化 | 归一化绝对变化 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in case["cross_sample_by_stage"]:
            lines.append(
                f"| {row['name']} | {int(row['comparable_pairs'])} | "
                f"{float(row['cross_sample_exact_equal_ratio']):.6f} | "
                f"{float(row['cross_sample_active_xor_ratio']):.6f} | "
                f"{float(row['cross_sample_sign_class_change_ratio']):.6f} | "
                f"{float(row['cross_sample_normalized_mean_abs_delta']):.6f} |"
            )
        quant = case["atlif_quant"]
        pair = case["pair_and_bank"]
        multicast = case["projection_multicast"]
        decisions = case["decisions"]
        lines += [
            "",
            "### 量化、PCCC与候选门槛",
            "",
            f"- 活跃ATLIF采样事件：`{quant['sample_events']}`，reference mismatch：`{quant['reference_mismatches']}`；",
            f"- Q4/Q6/Q8参数量化事件翻转率：`{quant['q4_event_mismatch_ratio']:.6%}` / `{quant['q6_event_mismatch_ratio']:.6%}` / `{quant['q8_event_mismatch_ratio']:.6%}`；",
            f"- 双K-zero：`{pair['both_kzero_ratio']:.6%}`，其中H67同class条件比例：`{pair['both_kzero_same_class_h67_ratio_conditional']:.6%}`；",
            f"- 类通道投影项/活动K lane：TTX `{pair['projection_class_channel_ratio_ttx']:.6%}`，H67 `{pair['projection_class_channel_ratio_h67']:.6%}`；",
            f"- 最终gate码类通道项/活动K lane：`{pair['projection_gate_class_channel_ratio_deploy']:.6%}`，相对对应score类额外合并：`{multicast['score_to_gate_term_reduction']:.6%}`；",
            f"- 最终gate类门控乘积减少：`{multicast['deploy_product_reduction']:.6%}`；窗口组G=2/4/8/16相对逐row再减少：`{multicast['deploy_group_g2_reduction_vs_row']:.6%}` / `{multicast['deploy_group_g4_reduction_vs_row']:.6%}` / `{multicast['deploy_group_g8_reduction_vs_row']:.6%}` / `{multicast['deploy_group_g16_reduction_vs_row']:.6%}`；",
            f"- 窗口组slot利用率G=2/4/8/16：`{multicast['deploy_group_g2_slot_utilization']:.6%}` / `{multicast['deploy_group_g4_slot_utilization']:.6%}` / `{multicast['deploy_group_g8_slot_utilization']:.6%}` / `{multicast['deploy_group_g16_slot_utilization']:.6%}`；",
            f"- gate类通道项的二进制/CSD平均非零数字：`{multicast['gate_term_binary_digits_mean']:.3f}` / `{multicast['gate_term_csd_digits_mean']:.3f}`；",
            f"- H67类门控乘积减少：`{multicast['h67_product_reduction']:.6%}`；row活动lane p50/p95/p99/max：`{multicast['baseline_p50']:.1f}` / `{multicast['baseline_p95']:.1f}` / `{multicast['baseline_p99']:.1f}` / `{multicast['baseline_max']:.1f}`；",
            f"- H67类通道项 p50/p95/p99/max：`{multicast['h67_p50']:.1f}` / `{multicast['h67_p95']:.1f}` / `{multicast['h67_p99']:.1f}` / `{multicast['h67_max']:.1f}`；",
            f"- H67单个类通道项最大token fanout p50/p95/p99/max：`{multicast['h67_fanout_p50']:.1f}` / `{multicast['h67_fanout_p95']:.1f}` / `{multicast['h67_fanout_p99']:.1f}` / `{multicast['h67_fanout_max']:.1f}`；",
            f"- H67活跃投影class p50/p95/p99/max：`{multicast['h67_classes_p50']:.1f}` / `{multicast['h67_classes_p95']:.1f}` / `{multicast['h67_classes_p99']:.1f}` / `{multicast['h67_classes_max']:.1f}`；",
            f"- 最终gate活跃class p50/p95/p99/max：`{multicast['deploy_classes_p50']:.1f}` / `{multicast['deploy_classes_p95']:.1f}` / `{multicast['deploy_classes_p99']:.1f}` / `{multicast['deploy_classes_max']:.1f}`；",
            "- H67多播交付事务p95（不含输出通道分块，M=1/2/4/8/16）："
            f"`{multicast['h67_delivery_m1_p95']:.1f}` / `{multicast['h67_delivery_m2_p95']:.1f}` / "
            f"`{multicast['h67_delivery_m4_p95']:.1f}` / `{multicast['h67_delivery_m8_p95']:.1f}` / "
            f"`{multicast['h67_delivery_m16_p95']:.1f}`；",
            f"- persistent-HTT profile门槛：`{'通过' if decisions['persistent_htt_profile_gate'] else '不通过'}`；",
            f"- 三条长skip全binary：`{'是' if decisions['rpi_all_long_skips_binary'] else '否'}`；",
            f"- Q8采样零翻转：`{'是' if decisions['atlif_q8_sample_zero_mismatch'] else '否'}`；",
            f"- PCCC 70%覆盖门槛：`{'通过' if decisions['pccc_coverage_gate_70pct'] else '不通过'}`；",
            f"- BMRF流量8%门槛：`{'通过' if decisions['bmrf_traffic_gate_8pct'] else '不通过或缺数据'}`。",
            "",
        ]
    lines += [
        "## 使用限制",
        "",
        "- 活动率加权卷积MAC不是边界精确SOP；",
        "- ATLIF量化只是在每模块首次调用的确定性采样，不替代valid825；",
        "- persistent-HTT通过profile门槛后仍需比较检测开销和系统EDP；",
        "- PCCC和BMRF通过覆盖门槛后仍必须通过同约束RTL/DC净收益门槛。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h67", type=Path, required=True)
    parser.add_argument("--h68", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "models": [analyze_case("H67", args.h67), analyze_case("H68", args.h68)],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md.write_text(render(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
