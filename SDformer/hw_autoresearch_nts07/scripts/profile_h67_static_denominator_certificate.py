#!/usr/bin/env python3
"""Profile an exact load-time denominator certificate for H67 T450 rows."""

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
DEFAULT_PROFILE = ROOT / (
    "results/h67_fullres_ep35_postconvergence_t450_20260805_profile100/"
    "nts11_hardware_p0_profile.json"
)
DEFAULT_VECTORS = ROOT / (
    "tb_h67/vectors/h67_fullres_ep35_postconvergence_t450_20260805/"
    "h67_checkpoint_rows.txt"
)
DEFAULT_OUTPUT = ROOT / "results/h67_static_denominator_certificate_20260814"
EXP_LUT = np.asarray(
    [256, 245, 234, 224, 215, 205, 196, 188, 181, 173, 165, 158, 152, 145, 139, 133],
    dtype=np.int32,
)


def decode_trace(encoded: dict[str, Any]) -> np.ndarray:
    dtypes = {"int16_le": "<i2", "int32_le": "<i4"}
    dtype = encoded.get("dtype")
    if encoded.get("codec") != "zlib_base64" or dtype not in dtypes:
        raise ValueError("unsupported ordered-trace encoding")
    raw = zlib.decompress(base64.b64decode(encoded["data"]))
    shape = tuple(int(value) for value in encoded["shape"])
    result = np.frombuffer(raw, dtype=dtypes[dtype])
    if result.size != math.prod(shape):
        raise ValueError("ordered-trace shape/payload mismatch")
    return result.reshape(shape).astype(np.int32, copy=False)


def rne_div16_array(raw: np.ndarray) -> np.ndarray:
    quotient = raw // 16
    remainder = raw % 16
    increment = (remainder > 8) | ((remainder == 8) & ((quotient & 1) != 0))
    return quotient + increment.astype(np.int32)


def rne_fraction16(integer_value: int, numerator: int) -> int:
    quotient, remainder = divmod(numerator, 16)
    value = integer_value + quotient
    if remainder > 8 or (remainder == 8 and (value & 1)):
        value += 1
    return value


def max_score_from_qcount(q_count: int) -> int:
    if not 0 <= q_count <= 32:
        raise ValueError("Q popcount must be in 0..32")
    return rne_fraction16(32 + 4 * q_count, 32 - q_count)


def max_score_from_qkm(q_count: int, k_count: int, motion_count: int) -> int:
    if not 0 <= q_count <= 32 or not 0 <= k_count <= 32:
        raise ValueError("Q/K popcount must be in 0..32")
    if not 0 <= motion_count <= 32:
        raise ValueError("motion popcount must be in 0..32")
    overlap_upper = min(q_count, k_count)
    same_zero_upper = 32 - q_count - k_count + overlap_upper
    return rne_fraction16(
        4 * overlap_upper + motion_count, same_zero_upper
    )


def h67_qkm_upper_bound(record: dict[str, Any]) -> np.ndarray:
    q_count = decode_trace(record["pair_q_count_ordered_trace"])
    k_count = decode_trace(record["pair_k_count_ordered_trace"])
    motion = decode_trace(record["pair_motion_ordered_trace"])
    overlap_upper = np.minimum(q_count, k_count)
    same_zero_upper = 32 - q_count - k_count + overlap_upper
    raw_upper = 64 * overlap_upper + same_zero_upper + 16 * motion[None, ...]
    if np.any(same_zero_upper < 0):
        raise ValueError("illegal Q/K count upper bound")
    return rne_div16_array(raw_upper)


def h67_score_pair(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    q_count = decode_trace(record["pair_q_count_ordered_trace"])
    k_count = decode_trace(record["pair_k_count_ordered_trace"])
    overlap = decode_trace(record["pair_overlap_ordered_trace"])
    motion = decode_trace(record["pair_motion_ordered_trace"])
    if q_count.shape[0] != 2 or motion.shape != q_count.shape[1:]:
        raise ValueError("H67 pair trace shape mismatch")
    same_zero = 32 - q_count - k_count + overlap
    raw = 64 * overlap + same_zero + 16 * motion[None, ...]
    if np.any(same_zero < 0) or np.any(raw < 0):
        raise ValueError("illegal H67 score counts")
    return q_count, rne_div16_array(raw)


def exp_q8(delta: np.ndarray) -> np.ndarray:
    absolute = -delta
    integer_shift = absolute >> 7
    fraction_index = (absolute & 127) >> 3
    fraction_index += ((absolute & 7) != 0).astype(np.int32)
    fraction_index = np.minimum(fraction_index, 15)
    return EXP_LUT[fraction_index] >> np.minimum(integer_shift, 8)


def rne_shift(value: int, shift: int) -> int:
    if shift == 0:
        return value
    quotient = value >> shift
    remainder = value - (quotient << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return quotient


def score_scalar(q: int, k: int, peer: int) -> int:
    overlap = (q & k).bit_count()
    same_zero = ((~q & ~k) & 0xFFFFFFFF).bit_count()
    motion = (k ^ peer).bit_count()
    raw = 64 * overlap + same_zero + 16 * motion
    return int(rne_div16_array(np.asarray(raw, dtype=np.int32)))


def parse_fair_vectors(path: Path) -> dict[str, Any]:
    words = iter(path.read_text(encoding="utf-8").split())
    rows = int(next(words))
    tokens = int(next(words))
    if tokens != 450:
        raise ValueError("certificate is frozen to T450")
    row_records = []
    baseline_gate_mismatch = 0
    forced17_gate_mismatch = 0
    for _ in range(rows):
        header = [int(next(words)) for _ in range(6)]
        q_values = []
        k_values = []
        expected_gates = []
        for _token in range(tokens):
            q_values.append(int(next(words), 16))
            k_values.append(int(next(words), 16))
            next(words)  # historical peer column; frozen RTL uses the temporal K peer.
            expected_gates.append(int(next(words)))
        scores = []
        qkm_upper = []
        for pair in range(tokens // 2):
            k0 = k_values[pair]
            k1 = k_values[pair + tokens // 2]
            motion_count = (k0 ^ k1).bit_count()
            scores.append(score_scalar(q_values[pair], k0, k1))
            scores.append(score_scalar(q_values[pair + tokens // 2], k1, k0))
            qkm_upper.append(
                max_score_from_qkm(
                    q_values[pair].bit_count(), k0.bit_count(), motion_count
                )
            )
            qkm_upper.append(
                max_score_from_qkm(
                    q_values[pair + tokens // 2].bit_count(),
                    k1.bit_count(),
                    motion_count,
                )
            )
        row_max = max(scores)
        deltas = np.asarray(scores, dtype=np.int32) - row_max
        exps = exp_q8(deltas)
        row_sum = int(exps.sum())
        denominator_shift = (row_sum - 1).bit_length() if row_sum else 0
        baseline_gates = [min(256, rne_shift(int(exp) * 450 * 128, denominator_shift)) for exp in exps]
        forced17_gates = [min(256, rne_shift(int(exp) * 450 * 128, 17)) for exp in exps]
        expected_interleaved = []
        for pair in range(tokens // 2):
            expected_interleaved.append(expected_gates[pair])
            expected_interleaved.append(expected_gates[pair + tokens // 2])
        baseline_gate_mismatch += sum(
            a != b for a, b in zip(baseline_gates, expected_interleaved)
        )
        forced17_gate_mismatch += sum(
            a != b for a, b in zip(forced17_gates, expected_interleaved)
        )
        q_max = max(value.bit_count() for value in q_values)
        pair_scores = np.asarray(scores, dtype=np.int32).reshape(tokens // 2, 2)
        row_records.append(
            {
                "row_tag": header[0],
                "stage": header[1],
                "q_popcount_max": q_max,
                "static_certificate": q_max <= 15,
                "qkm_upper_bound": max(qkm_upper),
                "qkm_certificate": max(qkm_upper) <= 96,
                "row_max": row_max,
                "actual_certificate": row_max <= 96,
                "denominator_shift": denominator_shift,
                "classes": len(set(scores)),
                "slots": int(np.sum(1 + (pair_scores[:, 0] != pair_scores[:, 1]))),
            }
        )
    try:
        next(words)
        raise ValueError("trailing fair-vector payload")
    except StopIteration:
        pass
    return {
        "rows": rows,
        "tokens": tokens,
        "records": row_records,
        "baseline_gate_mismatch": baseline_gate_mismatch,
        "forced17_gate_mismatch": forced17_gate_mismatch,
    }


def profile_population(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    totals = defaultdict(int)
    q_max_values = []
    qkm_bound_values = []
    stage = defaultdict(lambda: defaultdict(int))
    max_fail_run = 0
    for record in profile["summary"]["h60_records"]:
        q_count, scores = h67_score_pair(record)
        qkm_upper = h67_qkm_upper_bound(record)
        if scores.shape[-1] != 225:
            raise ValueError("profile record is not T450")
        row_q_max = q_count.max(axis=(0, 3))
        row_scores = np.transpose(scores, (1, 2, 0, 3)).reshape(-1, 450)
        row_max = row_scores.max(axis=1)
        static = row_q_max.reshape(-1) <= 15
        row_qkm_bound = qkm_upper.max(axis=(0, 3)).reshape(-1)
        qkm_static = row_qkm_bound <= 96
        actual = row_max <= 96
        sorted_scores = np.sort(row_scores, axis=1)
        classes = 1 + np.count_nonzero(np.diff(sorted_scores, axis=1), axis=1)
        equal = scores[0] == scores[1]
        slots = np.sum(1 + (~equal), axis=-1).reshape(-1)
        if np.any(static & ~actual):
            raise ValueError("static certificate false-accepted a row")
        if np.any(qkm_static & ~actual):
            raise ValueError("QKM certificate false-accepted a row")
        if np.any(row_qkm_bound < row_max):
            raise ValueError("QKM score upper bound fell below actual row max")
        totals["rows"] += int(static.size)
        totals["static_pass_rows"] += int(np.count_nonzero(static))
        totals["qkm_pass_rows"] += int(np.count_nonzero(qkm_static))
        totals["actual_pass_rows"] += int(np.count_nonzero(actual))
        totals["slots"] += int(slots.sum())
        totals["saved_hist_updates"] += int(slots[static].sum())
        totals["qkm_saved_hist_updates"] += int(slots[qkm_static].sum())
        totals["class_scans"] += int(classes.sum())
        totals["saved_class_scans"] += int(classes[static].sum())
        totals["qkm_saved_class_scans"] += int(classes[qkm_static].sum())
        totals["max_row_score"] = max(totals["max_row_score"], int(row_max.max()))
        totals["max_qkm_upper_bound"] = max(
            totals["max_qkm_upper_bound"], int(row_qkm_bound.max())
        )
        q_max_values.extend(int(value) for value in row_q_max.reshape(-1))
        qkm_bound_values.extend(int(value) for value in row_qkm_bound)
        stage_id = int(record["stage"])
        stage[stage_id]["rows"] += int(static.size)
        stage[stage_id]["static_pass"] += int(np.count_nonzero(static))
        stage[stage_id]["qkm_pass"] += int(np.count_nonzero(qkm_static))
        fail = ~static
        adjacent_fail = fail[:-1] & fail[1:]
        totals["adjacent_static_fail_transitions"] += int(
            np.count_nonzero(adjacent_fail)
        )
        totals["shared_fallback_serialization_class_cycles"] += int(
            classes[:-1][adjacent_fail].sum()
        )
        run = 0
        for value in fail:
            run = run + 1 if value else 0
            max_fail_run = max(max_fail_run, run)
    q_array = np.asarray(q_max_values, dtype=np.int32)
    qkm_array = np.asarray(qkm_bound_values, dtype=np.int32)
    return {
        **dict(totals),
        "static_pass_fraction": totals["static_pass_rows"] / totals["rows"],
        "qkm_pass_fraction": totals["qkm_pass_rows"] / totals["rows"],
        "actual_pass_fraction": totals["actual_pass_rows"] / totals["rows"],
        "hist_update_reduction": totals["saved_hist_updates"] / totals["slots"],
        "qkm_hist_update_reduction": (
            totals["qkm_saved_hist_updates"] / totals["slots"]
        ),
        "class_scan_reduction": totals["saved_class_scans"] / totals["class_scans"],
        "qkm_class_scan_reduction": (
            totals["qkm_saved_class_scans"] / totals["class_scans"]
        ),
        "q_popcount_max": {
            "p50": float(np.quantile(q_array, 0.50)),
            "p95": float(np.quantile(q_array, 0.95)),
            "p99": float(np.quantile(q_array, 0.99)),
            "max": int(q_array.max()),
        },
        "qkm_score_upper_bound": {
            "p50": float(np.quantile(qkm_array, 0.50)),
            "p95": float(np.quantile(qkm_array, 0.95)),
            "p99": float(np.quantile(qkm_array, 0.99)),
            "max": int(qkm_array.max()),
        },
        "max_consecutive_static_fail_rows_within_record": max_fail_run,
        "by_stage": {
            str(key): {
                **dict(value),
                "static_pass_fraction": value["static_pass"] / value["rows"],
                "qkm_pass_fraction": value["qkm_pass"] / value["rows"],
            }
            for key, value in sorted(stage.items())
        },
    }


def build_report(profile_path: Path, vector_path: Path) -> dict[str, Any]:
    fair = parse_fair_vectors(vector_path)
    population = profile_population(profile_path)
    fair_static = sum(row["static_certificate"] for row in fair["records"])
    fair_qkm = sum(row["qkm_certificate"] for row in fair["records"])
    fair_actual = sum(row["actual_certificate"] for row in fair["records"])
    fair_classes = sum(row["classes"] for row in fair["records"])
    fair_slots = sum(row["slots"] for row in fair["records"])
    fair_fail = [not row["static_certificate"] for row in fair["records"]]
    fair_conflict_cycles = sum(
        fair["records"][index]["classes"]
        for index in range(len(fair_fail) - 1)
        if fair_fail[index] and fair_fail[index + 1]
    )
    fair_pass_class_scans = sum(
        row["classes"] for row in fair["records"] if row["static_certificate"]
    )
    current_hist_bits = 2 * 163 * (9 + 1)
    candidate_bits = 163 * (9 + 1) + 2 * 6
    qkm_candidate_bits = 163 * (9 + 1) + 2 * 8
    modeled_cycles = 94891 - fair_pass_class_scans + fair_conflict_cycles
    qkm_modeled_cycles = 94891 - fair_classes
    comparison = {
        "current_dual_histogram_bits": current_hist_bits,
        "candidate_shared_fallback_plus_two_qmax_bits": candidate_bits,
        "state_reduction": current_hist_bits / candidate_bits,
        "candidate_shared_fallback_plus_two_qkm_bound_bits": qkm_candidate_bits,
        "qkm_state_reduction": current_hist_bits / qkm_candidate_bits,
        "fair_class_scan_only_cycle_model": {
            "baseline": 94891,
            "candidate": modeled_cycles,
            "speedup": 94891 / modeled_cycles,
            "removed_pass_class_scans": fair_pass_class_scans,
            "shared_fallback_conflict_cycles": fair_conflict_cycles,
            "evidence": "[模型]",
        },
        "fair_qkm_class_scan_only_cycle_model": {
            "baseline": 94891,
            "candidate": qkm_modeled_cycles,
            "speedup": 94891 / qkm_modeled_cycles,
            "removed_pass_class_scans": fair_classes,
            "shared_fallback_conflict_cycles": 0,
            "evidence": "[模型]",
        },
    }
    gates = {
        "fair_baseline_gate_mismatch_zero": fair["baseline_gate_mismatch"] == 0,
        "fair_forced17_gate_mismatch_zero": fair["forced17_gate_mismatch"] == 0,
        "profile_static_false_accept_zero": True,
        "profile_static_pass_ge_90pct": population["static_pass_fraction"] >= 0.90,
        "fair_qkm_false_accept_zero": all(
            not row["qkm_certificate"] or row["actual_certificate"]
            for row in fair["records"]
        ),
        "profile_qkm_false_accept_zero": True,
        "profile_qkm_pass_all_rows": population["qkm_pass_fraction"] == 1.0,
        "state_reduction_ge_1p5": comparison["state_reduction"] >= 1.5,
    }
    return {
        "schema": "h67_static_denominator_certificate_v1",
        "status": (
            "FROZEN_LEAF_ONLY_NO_DIRECTORY"
            if all(gates.values())
            else "NO_GO_STATIC_DENOMINATOR_CERTIFICATE"
        ),
        "evidence": ["[prof]", "[模型]"],
        "scope": (
            "H67 ep35 full-resolution T450 profile; separate leaf RTL evidence "
            "is in results/h67_denominator_certificate_rtl_20260814"
        ),
        "theorem": {
            "score_upper_bound_from_q_popcount": "32 + 4q + RNE((32-q)/16)",
            "load_time_condition": "max Q popcount across the T450 row <= 15",
            "implied_score_max": max_score_from_qcount(15),
            "normalization_condition": "n>=431 and row_max<=96 implies ceil_log2(sum_exp_q8)=17",
            "tight_counterexample": {
                "row_max": 97,
                "lower_sum": 256 + 449 * 145,
            },
            "qkm_score_upper_bound": (
                "4*min(q,k)+motion+RNE((32-q-k+min(q,k))/16)"
            ),
        },
        "fair_ep35_sample0_window0": {
            "rows": fair["rows"],
            "static_pass_rows": fair_static,
            "qkm_pass_rows": fair_qkm,
            "qkm_upper_bound_max": max(
                row["qkm_upper_bound"] for row in fair["records"]
            ),
            "actual_pass_rows": fair_actual,
            "q_popcount_max": max(row["q_popcount_max"] for row in fair["records"]),
            "row_score_max": max(row["row_max"] for row in fair["records"]),
            "class_scans": fair_classes,
            "shared_fallback_conflict_cycles": fair_conflict_cycles,
            "rqtb_slots_recomputed": fair_slots,
            "baseline_gate_mismatch": fair["baseline_gate_mismatch"],
            "forced17_gate_mismatch": fair["forced17_gate_mismatch"],
        },
        "profile100": population,
        "comparison": comparison,
        "gates": gates,
        "claim_boundary": [
            "the load-time certificate requires a row-loader Q-popcount summary before score build",
            "the stronger QKM certificate requires Q/K popcounts plus K-motion count; only Q popcounts are already present in the existing metadata builder",
            "one shared fallback histogram is sufficient only because LAWS permits at most one build; consecutive fail rows can stall behind fallback class scan",
            "the cycle model removes class scans only and does not model histogram-write energy",
            "this profile artifact contains no RTL, SAIF, DC, or full-encoder result; leaf RTL is a separate evidence package",
            "does not modify docs/359 frozen main-table columns",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    fair = report["fair_ep35_sample0_window0"]
    population = report["profile100"]
    comparison = report["comparison"]
    cycle = comparison["fair_class_scan_only_cycle_model"]
    qkm_cycle = comparison["fair_qkm_class_scan_only_cycle_model"]
    return f"""# Motion H67 加载期 denominator 严格证书

## 裁决

`{report['status']}`。本包只提供 `[prof]+[模型]`，独立 leaf RTL 证据位于 `results/h67_denominator_certificate_rtl_20260814/`；两者均不改 `docs/359`。

## Exact 合同

给定一个 32-bit query 的 popcount `q`，冻结 Motion-XOR 对任意当前/相邻 K 的 Q7 score 上界为：

`32 + 4q + RNE((32-q)/16)`。

因此整行 450 个 token 的 `max(Q-popcount)<=15` 时，score 上界为 `{report['theorem']['implied_score_max']}`。对 T450，若 `row_max<=96`，Shiftmax LUT 的最小 exp 为 152，故 `256+449*152=68504>2^16` 且 `450*256<2^17`，严格得到 denominator shift 17。97 已不能证明：`256+449*145=65361<2^16`。

## 真实账本

- 公平 ep35 sample0/window0：静态证书 `{fair['static_pass_rows']}/{fair['rows']}` 行，实际 row-max 证书 `{fair['actual_pass_rows']}/{fair['rows']}` 行；最大 Q-popcount `{fair['q_popcount_max']}`、最大 score `{fair['row_score_max']}`；baseline/fixed17 gate mismatch 均为 `{fair['baseline_gate_mismatch']}/{fair['forced17_gate_mismatch']}`。
- 更紧的 Q/K/motion 严格上界证书命中公平包 `{fair['qkm_pass_rows']}/{fair['rows']}` 行，最大上界 `{fair['qkm_upper_bound_max']}`。
- 公平包重新计算 RQTB slot `{fair['rqtb_slots_recomputed']}`，class scan `{fair['class_scans']}`；封存主列仍使用 34099 slot 和 94891 cycles，不由本报告覆盖。
- profile100：静态证书命中 `{population['static_pass_rows']}/{population['rows']} = {population['static_pass_fraction']:.2%}`；实际 row-max 证书命中 `{population['actual_pass_fraction']:.2%}`；最大 score `{population['max_row_score']}`。
- profile100 的 Q/K/motion 上界证书命中 `{population['qkm_pass_rows']}/{population['rows']} = {population['qkm_pass_fraction']:.2%}`；上界 p50/p95/p99/max 为 `{population['qkm_score_upper_bound']['p50']:.0f}/{population['qkm_score_upper_bound']['p95']:.0f}/{population['qkm_score_upper_bound']['p99']:.0f}/{population['qkm_score_upper_bound']['max']}`。
- 静态证书可避免 `{population['hist_update_reduction']:.2%}` 的 histogram update，并避免 `{population['class_scan_reduction']:.2%}` 的 class scan；这两个仍是 profile 事务计数，不是能量。
- 单 fallback histogram 在 record 内观察到 `{population['adjacent_static_fail_transitions']}` 次相邻失败，保守增加 `{population['shared_fallback_serialization_class_cycles']}` 个 class-cycle；最长连续失败为 `{population['max_consecutive_static_fail_rows_within_record']}` 行。
- Q-popcount row-max：p50 `{population['q_popcount_max']['p50']:.0f}`、p95 `{population['q_popcount_max']['p95']:.0f}`、p99 `{population['q_popcount_max']['p99']:.0f}`、max `{population['q_popcount_max']['max']}`。

## 存储与周期模型

- 双常驻 histogram：`{comparison['current_dual_histogram_bits']}` bit。
- 一份共享 fallback histogram + 两份 6-bit row Q-max：`{comparison['candidate_shared_fallback_plus_two_qmax_bits']}` bit，状态比 `{comparison['state_reduction']:.3f}x`。
- 一份共享 fallback histogram + 两份 8-bit Q/K/motion score 上界：`{comparison['candidate_shared_fallback_plus_two_qkm_bound_bits']}` bit，状态比 `{comparison['qkm_state_reduction']:.3f}x`。
- 删除命中行 class scan、并加回共享 fallback 相邻失败冲突后：`{cycle['baseline']} -> {cycle['candidate']} = {cycle['speedup']:.4f}x [模型]`；其中移除 `{cycle['removed_pass_class_scans']}` cycle，加回 `{cycle['shared_fallback_conflict_cycles']}` cycle。该数字不含 histogram 写能量，因此不能进性能主表。
- Q/K/motion 证书在公平包全命中时，class-scan-only 上界为 `{qkm_cycle['baseline']} -> {qkm_cycle['candidate']} = {qkm_cycle['speedup']:.4f}x [模型]`，同样不能作为性能主张。

## 边界

候选原本试图改变 normalization 存储对象，但共享 fallback 会破坏 LAWS 的双 workspace 解耦，且性能模型仅约 1.014x。方向已冻结为 leaf-only：不接入共享 histogram，不修改 `h67_laws_shared_backend_2s_top`，只作为 RQTB 的严格 denominator 注脚，不能称独立 DATE 贡献。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTORS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.profile.resolve(), args.vectors.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
