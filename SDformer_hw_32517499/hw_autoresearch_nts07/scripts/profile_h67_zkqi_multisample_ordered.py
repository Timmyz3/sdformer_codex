#!/usr/bin/env python3
"""用全量ordered count trace重建并校准Motion ZKQI周期与工作事件。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.profile_h67_exact_metadata_cascade import load_rows, score_q7
    from scripts.report_h67_zkqi_row_miter import parse_log
except ModuleNotFoundError:
    from profile_h67_exact_metadata_cascade import load_rows, score_q7
    from report_h67_zkqi_row_miter import parse_log


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = (
    ROOT / "results/h67_fullres_ep30_t450_profile100_20260805"
    / "nts11_hardware_p0_profile.json"
)
DEFAULT_VECTOR = (
    ROOT / "tb_h67/vectors/h67_ep30_fullres_t450_all12_20260805"
    / "h67_checkpoint_rows.txt"
)
DEFAULT_PAIR_LOG = (
    ROOT / "results/h67_zkqi_threeway_20260809/logs"
    / "iverilog_pairbitmap_mode0.log"
)
DEFAULT_TTB_LOG = (
    ROOT / "results/h67_zkqi_threeway_20260809/logs"
    / "iverilog_ttb8_mode0.log"
)
DEFAULT_OUT = ROOT / "results/h67_zkqi_multisample_ordered_20260809"
BLOCK_RE = re.compile(r"^S(?P<stage>\d+)\.B(?P<block>\d+)\.attn$")
EXPECTED_DEPTHS = {0: 2, 1: 2, 2: 6, 3: 2}
EXPECTED_HEADS = {0: 3, 1: 6, 2: 12, 3: 24}
EXPECTED_WINDOWS = {0: 440, 1: 120, 2: 30, 3: 10}
EXPECTED_NAMES = tuple(
    f"S{stage}.B{block}.attn"
    for stage, depth in EXPECTED_DEPTHS.items()
    for block in range(depth)
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def decode_trace(encoded: dict[str, Any]) -> np.ndarray:
    dtypes = {
        "int16_le": np.dtype("<i2"),
        "int32_le": np.dtype("<i4"),
    }
    dtype = dtypes.get(str(encoded.get("dtype")))
    if encoded.get("codec") != "zlib_base64" or dtype is None:
        raise ValueError("只支持zlib_base64 int16/int32 ordered trace")
    raw = zlib.decompress(base64.b64decode(encoded["data"]))
    shape = tuple(int(value) for value in encoded["shape"])
    values = np.frombuffer(raw, dtype=dtype)
    if values.size != math.prod(shape):
        raise ValueError("ordered trace payload与shape不守恒")
    return values.reshape(shape).astype(np.int32, copy=False)


def rne_div16(raw: np.ndarray) -> np.ndarray:
    quotient = raw // 16
    remainder = raw % 16
    increment = (remainder > 8) | (
        (remainder == 8) & ((quotient & 1) != 0)
    )
    return quotient + increment.astype(np.int32)


def h67_score_from_counts(
    q_count: np.ndarray,
    k_count: np.ndarray,
    overlap: np.ndarray,
    motion: np.ndarray,
    *,
    head_dim: int = 32,
) -> np.ndarray:
    if q_count.shape != k_count.shape or q_count.shape != overlap.shape:
        raise ValueError("Q/K/overlap ordered trace shape不一致")
    if q_count.ndim != 4 or q_count.shape[0] != 2:
        raise ValueError("pair count trace必须是[2,B,H,N]")
    if motion.shape != q_count.shape[1:]:
        raise ValueError("motion ordered trace必须是[B,H,N]")
    same_zero = head_dim - q_count - k_count + overlap
    raw = 65 * overlap + head_dim - q_count - k_count + 16 * motion[None, ...]
    if (
        np.any(q_count < 0)
        or np.any(q_count > head_dim)
        or np.any(k_count < 0)
        or np.any(k_count > head_dim)
        or np.any(overlap < 0)
        or np.any(overlap > q_count)
        or np.any(overlap > k_count)
        or np.any(q_count - overlap > head_dim - k_count)
        or np.any(motion < 0)
        or np.any(motion > head_dim)
        or np.any(same_zero < 0)
        or np.any(raw < 0)
    ):
        raise ValueError("ordered count trace出现非法计数")
    k_intersection_twice = k_count[0] + k_count[1] - motion
    k_intersection = k_intersection_twice // 2
    if (
        np.any((k_intersection_twice & 1) != 0)
        or np.any(k_intersection < 0)
        or np.any(k_intersection > np.minimum(k_count[0], k_count[1]))
        or np.any(k_count[0] + k_count[1] - k_intersection > head_dim)
    ):
        raise ValueError("K0/K1 count与motion XOR不可由真实bit向量实现")
    score = rne_div16(raw)
    if np.any(score > 5 * head_dim + 2):
        raise ValueError("H67 score超出离散类空间")
    return score


def ttb_depth1_front_cycles(active_counts: np.ndarray) -> np.ndarray:
    """单槽bundle descriptor的精确无反压周期。

    空bundle可在旧descriptor消费期间穿越；非空bundle只能在槽空或旧descriptor
    最后一项同拍退休时接收。新descriptor最早从下一拍开始消费。
    """

    counts = np.asarray(active_counts, dtype=np.int64)
    if counts.ndim != 2 or np.any(counts < 0):
        raise ValueError("active_counts必须是非负[R,G]矩阵")
    producer_cycle = np.zeros(counts.shape[0], dtype=np.int64)
    last_consume_cycle = np.full(counts.shape[0], -1, dtype=np.int64)
    for group in range(counts.shape[1]):
        count = counts[:, group]
        active = count != 0
        accept_cycle = np.where(
            active,
            np.maximum(producer_cycle, last_consume_cycle),
            producer_cycle,
        )
        last_consume_cycle = np.where(
            active, accept_cycle + count, last_consume_cycle
        )
        producer_cycle = accept_cycle + 1
    return np.maximum(producer_cycle, last_consume_cycle + 1)


def occupied_class_count(score_rows: np.ndarray, class_count: int) -> np.ndarray:
    scores = np.asarray(score_rows, dtype=np.int64)
    if scores.ndim != 2 or np.any(scores < 0) or np.any(scores >= class_count):
        raise ValueError("score row超出class目录")
    present = np.zeros((scores.shape[0], class_count), dtype=np.bool_)
    present[np.arange(scores.shape[0])[:, None], scores] = True
    return present.sum(axis=1, dtype=np.int64)


def compute_row_metrics(
    q_count: np.ndarray,
    k_count: np.ndarray,
    overlap: np.ndarray,
    motion: np.ndarray,
    *,
    head_dim: int = 32,
    bundle_size: int = 8,
) -> dict[str, np.ndarray]:
    score = h67_score_from_counts(
        q_count, k_count, overlap, motion, head_dim=head_dim
    )
    _, windows, heads, pairs = q_count.shape
    rows = windows * heads
    k_active = k_count != 0
    active_pair = k_active.any(axis=0)
    active_token_count = k_active.sum(axis=0, dtype=np.int64)
    score_equal = score[0] == score[1]
    active_descriptors = np.where(
        score_equal, active_pair, active_token_count
    ).sum(axis=2, dtype=np.int64)
    baseline_descriptors = np.where(score_equal, 1, 2).sum(
        axis=2, dtype=np.int64
    )
    candidate_descriptors = np.where(
        active_pair, np.where(score_equal, 1, 2), 0
    ).sum(axis=2, dtype=np.int64)
    outputs = k_active.sum(axis=(0, 3), dtype=np.int64)
    active_pairs = active_pair.sum(axis=2, dtype=np.int64)

    score_rows = score.transpose(1, 2, 0, 3).reshape(rows, 2 * pairs)
    classes = occupied_class_count(score_rows, 5 * head_dim + 3).reshape(
        windows, heads
    )

    groups = math.ceil(pairs / bundle_size)
    padded = np.pad(
        active_pair,
        ((0, 0), (0, 0), (0, groups * bundle_size - pairs)),
        constant_values=False,
    )
    active_counts = padded.reshape(
        windows, heads, groups, bundle_size
    ).sum(axis=3, dtype=np.int64)
    ttb_front = ttb_depth1_front_cycles(
        active_counts.reshape(rows, groups)
    ).reshape(windows, heads)

    # 由RTL状态机推导：seal/phase固定3拍；每个占用类1拍；每个active
    # descriptor需要descriptor/QK两级读；每个非零K temporal token发射1拍；
    # 存在active phase时再计1拍状态切换。
    backend = (
        3
        + classes
        + 2 * active_descriptors
        + outputs
        + (active_descriptors != 0).astype(np.int64)
    )
    baseline_cycles = pairs + backend
    ttb_cycles = ttb_front + backend
    preload = np.full((windows, heads), pairs, dtype=np.int64)
    baseline_read_bits = pairs * (4 * head_dim) + outputs * head_dim
    candidate_read_bits = active_pairs * (4 * head_dim) + outputs * head_dim

    return {
        "scores": score,
        "active_counts": active_counts,
        "active_pairs": active_pairs,
        "outputs": outputs,
        "score_equal_pairs": score_equal.sum(axis=2, dtype=np.int64),
        "baseline_descriptors": baseline_descriptors,
        "candidate_descriptors": candidate_descriptors,
        "active_descriptors": active_descriptors,
        "occupied_classes": classes,
        "ttb_front_cycles": ttb_front,
        "backend_cycles": backend,
        "baseline_cycles": baseline_cycles,
        "pair_bitmap_cycles": baseline_cycles.copy(),
        "ttb_cycles": ttb_cycles,
        "preload_cycles": preload,
        "baseline_e2e_cycles": baseline_cycles + preload,
        "pair_bitmap_e2e_cycles": baseline_cycles + preload,
        "ttb_e2e_cycles": ttb_cycles + preload,
        "baseline_read_bits": baseline_read_bits,
        "candidate_read_bits": candidate_read_bits,
    }


def decode_record(record: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    q_count = decode_trace(record["pair_q_count_ordered_trace"])
    k_count = decode_trace(record["pair_k_count_ordered_trace"])
    overlap = decode_trace(record["pair_overlap_ordered_trace"])
    motion = decode_trace(record["pair_motion_ordered_trace"])
    metrics = compute_row_metrics(q_count, k_count, overlap, motion)

    _, windows, heads, pairs = q_count.shape
    groups = math.ceil(pairs / 8)
    pad = groups * 8 - pairs
    union_count = q_count + k_count - overlap
    union_bundle = np.pad(
        union_count,
        ((0, 0), (0, 0), (0, 0), (0, pad)),
        constant_values=0,
    ).reshape(2, windows, heads, groups, 8).sum(axis=(0, 4))
    k_bundle = np.pad(
        k_count,
        ((0, 0), (0, 0), (0, 0), (0, pad)),
        constant_values=0,
    ).reshape(2, windows, heads, groups, 8).sum(axis=(0, 4))
    motion_bundle = np.pad(
        motion,
        ((0, 0), (0, 0), (0, pad)),
        constant_values=0,
    ).reshape(windows, heads, groups, 8).sum(axis=3)
    checks = {
        "ttb_active_trace_exact": bool(
            np.array_equal(
                union_bundle,
                decode_trace(record["ttb_tok8_active_ordered_trace"]),
            )
        ),
        "ttb_k_trace_exact": bool(
            np.array_equal(
                k_bundle, decode_trace(record["ttb_tok8_k_ordered_trace"])
            )
        ),
        "ttb_motion_trace_exact": bool(
            np.array_equal(
                motion_bundle,
                decode_trace(record["ttb_tok8_motion_ordered_trace"]),
            )
        ),
        "q_count": q_count,
        "k_count": k_count,
    }
    return metrics, checks


def vector_row_counts(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = row["vectors"]
    pairs = len(vectors) // 2
    q_count = np.zeros((2, pairs), dtype=np.int32)
    k_count = np.zeros((2, pairs), dtype=np.int32)
    scores = np.zeros((2, pairs), dtype=np.int32)
    for pair in range(pairs):
        q0, k0, peer0, _ = vectors[pair]
        q1, k1, peer1, _ = vectors[pairs + pair]
        if peer0 != k1 or peer1 != k0:
            raise ValueError("sample0 vector temporal peer不一致")
        q_count[:, pair] = (q0.bit_count(), q1.bit_count())
        k_count[:, pair] = (k0.bit_count(), k1.bit_count())
        scores[:, pair] = (score_q7(q0, k0, k1), score_q7(q1, k1, k0))
    return q_count, k_count, scores


def block_identity(record: dict[str, Any]) -> tuple[int, int, str]:
    name = str(record.get("name", ""))
    match = BLOCK_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"无法解析attention block名: {name}")
    stage = int(match.group("stage"))
    block = int(match.group("block"))
    if int(record.get("stage", -1)) != stage:
        raise ValueError(f"record stage与name不一致: {name}")
    return stage, block, name


def validate_profile_contract(
    profile: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    samples = int(profile.get("samples", 0))
    if samples != 100 or len(records) != samples * len(EXPECTED_NAMES):
        raise ValueError("输入必须是100 sample且每sample恰有12个attention record")
    protocol = profile.get("eval_protocol") or {}
    if (
        protocol.get("resolution") != [480, 640]
        or protocol.get("crop") is not None
        or protocol.get("window_size") != [2, 15, 15]
        or int(protocol.get("tokens_per_window", -1)) != 450
    ):
        raise ValueError("profile不是冻结的fullres T450部署协议")
    expected_pairs = 225
    checked_shapes = 0
    seen: set[tuple[int, str]] = set()
    for sample in range(samples):
        sample_records = records[
            sample * len(EXPECTED_NAMES):(sample + 1) * len(EXPECTED_NAMES)
        ]
        names = tuple(str(record.get("name", "")) for record in sample_records)
        if names != EXPECTED_NAMES:
            raise ValueError(f"sample={sample}: all12 block覆盖或顺序漂移")
        for record in sample_records:
            if int(record.get("sample_id", -1)) != sample:
                raise ValueError(f"sample={sample}: record sample_id漂移")
            stage, _, name = block_identity(record)
            key = (sample, name)
            if key in seen:
                raise ValueError(f"重复attention record: {key}")
            seen.add(key)
            windows = EXPECTED_WINDOWS[stage]
            heads = EXPECTED_HEADS[stage]
            if (
                int(record.get("tokens", -1)) != 450
                or int(record.get("batch_windows", -1)) != windows
                or int(record.get("num_heads", -1)) != heads
            ):
                raise ValueError(f"{key}: token/window/head合同漂移")
            expected_shapes = {
                "pair_q_count_ordered_trace": [2, windows, heads, expected_pairs],
                "pair_k_count_ordered_trace": [2, windows, heads, expected_pairs],
                "pair_overlap_ordered_trace": [2, windows, heads, expected_pairs],
                "pair_motion_ordered_trace": [windows, heads, expected_pairs],
                "ttb_tok8_active_ordered_trace": [windows, heads, 29],
                "ttb_tok8_k_ordered_trace": [windows, heads, 29],
                "ttb_tok8_motion_ordered_trace": [windows, heads, 29],
            }
            for field, shape in expected_shapes.items():
                encoded = record.get(field) or {}
                if (
                    encoded.get("codec") != "zlib_base64"
                    or encoded.get("dtype") not in ("int16_le", "int32_le")
                    or encoded.get("shape") != shape
                    or not encoded.get("data")
                ):
                    raise ValueError(f"{key}: {field}编码或shape漂移")
                checked_shapes += 1
    if len(seen) != samples * len(EXPECTED_NAMES):
        raise ValueError("profile record唯一性守恒失败")
    return {
        "status": "PASS",
        "samples": samples,
        "records": len(records),
        "unique_sample_blocks": len(seen),
        "ordered_trace_shapes_checked": checked_shapes,
        "pairs_per_row": expected_pairs,
    }


def calibrate(
    records: list[dict[str, Any]],
    vector_path: Path,
    pair_log_path: Path,
    ttb_log_path: Path,
) -> dict[str, Any]:
    _, vector_rows = load_rows(vector_path)
    pair_rows, pair_final = parse_log(pair_log_path)
    ttb_rows, ttb_final = parse_log(ttb_log_path)
    profile_rows: list[dict[str, Any]] = []
    trace_checks = defaultdict(int)

    for record in records:
        if int(record["sample_id"]) != 0:
            continue
        stage, block, _ = block_identity(record)
        metrics, checks = decode_record(record)
        for key in (
            "ttb_active_trace_exact", "ttb_k_trace_exact", "ttb_motion_trace_exact"
        ):
            trace_checks[key] += int(checks[key])
        for head in range(int(record["num_heads"])):
            profile_rows.append(
                {
                    "stage": stage,
                    "block": block,
                    "head": head,
                    "q_count": checks["q_count"][:, 0, head],
                    "k_count": checks["k_count"][:, 0, head],
                    "scores": metrics["scores"][:, 0, head],
                    **{
                        key: int(value[0, head])
                        for key, value in metrics.items()
                        if key not in ("scores", "active_counts")
                    },
                }
            )

    expected_rows = len(vector_rows)
    if len(profile_rows) != expected_rows or len(pair_rows) != expected_rows:
        raise ValueError("sample0/window0 profile、vector与RTL行数不一致")
    if pair_final["rows"] != expected_rows or ttb_final["rows"] != expected_rows:
        raise ValueError("RTL final receipt行数不一致")

    checked_fields = 0
    for index, (profile, vector, pair, ttb) in enumerate(
        zip(profile_rows, vector_rows, pair_rows, ttb_rows)
    ):
        identity = (profile["stage"], profile["block"], profile["head"])
        if identity != (vector["stage"], vector["block"], vector["head"]):
            raise ValueError(f"row={index}: profile/vector身份不一致")
        if identity != (pair["stage"], pair["block"], pair["head"]):
            raise ValueError(f"row={index}: profile/RTL身份不一致")
        vector_q, vector_k, vector_score = vector_row_counts(vector)
        if not np.array_equal(profile["q_count"], vector_q):
            raise ValueError(f"row={index}: Q-count与raw vector不一致")
        if not np.array_equal(profile["k_count"], vector_k):
            raise ValueError(f"row={index}: K-count与raw vector不一致")
        if not np.array_equal(profile["scores"], vector_score):
            raise ValueError(f"row={index}: score与raw vector不一致")

        expected = {
            "active_pairs": pair["active_pairs"],
            "outputs": pair["outputs"],
            "baseline_descriptors": pair["baseline_slots"],
            "candidate_descriptors": pair["zkqi_slots"],
            "baseline_cycles": pair["baseline_cycles"],
            "pair_bitmap_cycles": pair["zkqi_cycles"],
            "ttb_cycles": ttb["zkqi_cycles"],
            "preload_cycles": pair["baseline_preload"],
            "baseline_e2e_cycles": pair["baseline_e2e_cycles"],
            "pair_bitmap_e2e_cycles": pair["zkqi_e2e_cycles"],
            "ttb_e2e_cycles": ttb["zkqi_e2e_cycles"],
            "baseline_read_bits": pair["baseline_read_bits"],
            "candidate_read_bits": pair["zkqi_read_bits"],
        }
        for key, rtl_value in expected.items():
            if profile[key] != rtl_value:
                raise ValueError(
                    f"row={index}: {key}模型/RTL={profile[key]}/{rtl_value}"
                )
            checked_fields += 1
        if pair["baseline_cycles"] != ttb["baseline_cycles"]:
            raise ValueError(f"row={index}: 两次RTL baseline不一致")
        if pair["seeded"] != 2 * (225 - profile["active_pairs"]):
            raise ValueError(f"row={index}: zero-K seed守恒失败")

    if any(value != 12 for value in trace_checks.values()):
        raise ValueError(f"12 block TTB trace交叉检查失败: {dict(trace_checks)}")
    return {
        "status": "PASS",
        "profile_vector_score_mismatches": 0,
        "rtl_cycle_field_mismatches": 0,
        "rows": expected_rows,
        "checked_rtl_fields": checked_fields,
        "ttb_ordered_trace_checks": dict(trace_checks),
        "cycle_residual_min": 0,
        "cycle_residual_max": 0,
        "pair_final_cycles": {
            "baseline": pair_final["baseline_cycles"],
            "pair_bitmap": pair_final["zkqi_cycles"],
        },
        "ttb_final_cycles": {
            "baseline": ttb_final["baseline_cycles"],
            "ttb8": ttb_final["zkqi_cycles"],
        },
    }


def nearest_rank(values: np.ndarray, q: float) -> int | float:
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError("空数组没有分位数")
    ordered = np.sort(array.reshape(-1))
    index = max(0, min(ordered.size - 1, math.ceil(q * ordered.size) - 1))
    value = ordered[index]
    return int(value) if np.issubdtype(array.dtype, np.integer) else float(value)


def distribution(values: np.ndarray) -> dict[str, int | float]:
    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        raise ValueError("空数组不能统计")
    return {
        "min": int(array.min()) if np.issubdtype(array.dtype, np.integer) else float(array.min()),
        "mean": float(array.mean()),
        "p50": nearest_rank(array, 0.50),
        "p95": nearest_rank(array, 0.95),
        "p99": nearest_rank(array, 0.99),
        "max": int(array.max()) if np.issubdtype(array.dtype, np.integer) else float(array.max()),
        "sum": int(array.sum()) if np.issubdtype(array.dtype, np.integer) else float(array.sum()),
    }


def empty_totals() -> dict[str, int]:
    return defaultdict(int)


def add_totals(target: dict[str, int], metrics: dict[str, np.ndarray]) -> None:
    target["rows"] += int(metrics["active_pairs"].size)
    target["active_pairs"] += int(metrics["active_pairs"].sum())
    target["outputs"] += int(metrics["outputs"].sum())
    target["occupied_classes"] += int(metrics["occupied_classes"].sum())
    target["active_descriptors"] += int(metrics["active_descriptors"].sum())
    for key in (
        "baseline_cycles", "pair_bitmap_cycles", "ttb_cycles",
        "baseline_e2e_cycles", "pair_bitmap_e2e_cycles", "ttb_e2e_cycles",
        "baseline_read_bits", "candidate_read_bits",
    ):
        target[key] += int(metrics[key].sum())


def finalize_totals(totals: dict[str, int], *, pairs: int = 225) -> dict[str, Any]:
    row = dict(totals)
    row["active_pair_ratio"] = row["active_pairs"] / (row["rows"] * pairs)
    row["execution_speedup"] = row["baseline_cycles"] / row["ttb_cycles"]
    row["preload_inclusive_speedup"] = (
        row["baseline_e2e_cycles"] / row["ttb_e2e_cycles"]
    )
    row["score_evaluation_reduction"] = 1.0 - row["active_pairs"] / (
        row["rows"] * pairs
    )
    row["qk_read_bit_reduction"] = 1.0 - row["candidate_read_bits"] / row[
        "baseline_read_bits"
    ]
    return row


def analyze_all(records: list[dict[str, Any]]) -> dict[str, Any]:
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    window_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    samples: dict[int, dict[str, int]] = defaultdict(empty_totals)
    stages: dict[int, dict[str, int]] = defaultdict(empty_totals)
    blocks: dict[str, dict[str, int]] = defaultdict(empty_totals)
    stage_outcomes: dict[int, dict[str, int]] = defaultdict(empty_totals)
    trace_checks = defaultdict(int)
    record_count = 0

    for record in records:
        sample_id = int(record["sample_id"])
        stage, _, block_name = block_identity(record)
        metrics, checks = decode_record(record)
        record_count += 1
        for key in (
            "ttb_active_trace_exact", "ttb_k_trace_exact", "ttb_motion_trace_exact"
        ):
            trace_checks[key] += int(checks[key])
        add_totals(samples[sample_id], metrics)
        add_totals(stages[stage], metrics)
        add_totals(blocks[block_name], metrics)
        delta = metrics["ttb_e2e_cycles"] - metrics["baseline_e2e_cycles"]
        stage_outcomes[stage]["faster"] += int((delta < 0).sum())
        stage_outcomes[stage]["equal"] += int((delta == 0).sum())
        stage_outcomes[stage]["slower"] += int((delta > 0).sum())
        stage_outcomes[stage]["slowdown_cycles"] += int(delta[delta > 0].sum())

        for key in (
            "active_pairs", "outputs", "occupied_classes", "active_descriptors",
            "ttb_front_cycles", "backend_cycles", "baseline_cycles", "ttb_cycles",
            "baseline_e2e_cycles", "ttb_e2e_cycles", "baseline_read_bits",
            "candidate_read_bits",
        ):
            chunks[key].append(metrics[key].reshape(-1))
        chunks["row_execution_speedup"].append(
            (metrics["baseline_cycles"] / metrics["ttb_cycles"]).reshape(-1)
        )
        chunks["row_e2e_speedup"].append(
            (metrics["baseline_e2e_cycles"] / metrics["ttb_e2e_cycles"]).reshape(-1)
        )

        baseline_window = metrics["baseline_e2e_cycles"].sum(axis=1)
        ttb_window = metrics["ttb_e2e_cycles"].sum(axis=1)
        window_chunks["baseline_e2e_cycles"].append(baseline_window)
        window_chunks["ttb_e2e_cycles"].append(ttb_window)
        window_chunks["speedup"].append(baseline_window / ttb_window)

    if any(value != record_count for value in trace_checks.values()):
        raise ValueError(f"ordered TTB trace交叉检查失败: {dict(trace_checks)}")
    arrays = {key: np.concatenate(values) for key, values in chunks.items()}
    windows = {
        key: np.concatenate(values) for key, values in window_chunks.items()
    }
    global_totals = empty_totals()
    for totals in samples.values():
        for key, value in totals.items():
            global_totals[key] += value
    final_samples = {
        str(sample): finalize_totals(totals)
        for sample, totals in sorted(samples.items())
    }
    sample_speedups = np.asarray(
        [row["preload_inclusive_speedup"] for row in final_samples.values()],
        dtype=np.float64,
    )

    density = arrays["active_pairs"] / 225.0
    cycle_delta = arrays["ttb_e2e_cycles"] - arrays["baseline_e2e_cycles"]
    slower = cycle_delta > 0
    adaptive_e2e_cycles = int(
        np.minimum(
            arrays["baseline_e2e_cycles"], arrays["ttb_e2e_cycles"]
        ).sum()
    )
    density_bins = []
    boundaries = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0000001)
    for lower, upper in zip(boundaries, boundaries[1:]):
        mask = (density >= lower) & (density < upper)
        if not np.any(mask):
            continue
        density_bins.append(
            {
                "lower": lower,
                "upper": min(upper, 1.0),
                "rows": int(mask.sum()),
                "active_pair_ratio_mean": float(density[mask].mean()),
                "preload_inclusive_speedup": float(
                    arrays["baseline_e2e_cycles"][mask].sum()
                    / arrays["ttb_e2e_cycles"][mask].sum()
                ),
                "row_speedup": distribution(arrays["row_e2e_speedup"][mask]),
            }
        )

    return {
        "records": record_count,
        "samples": len(samples),
        "head_rows": int(arrays["active_pairs"].size),
        "block_windows": int(windows["speedup"].size),
        "ttb_ordered_trace_checks": dict(trace_checks),
        "global": finalize_totals(global_totals),
        "row_distributions": {
            key: distribution(arrays[key])
            for key in (
                "active_pairs", "outputs", "occupied_classes", "active_descriptors",
                "ttb_front_cycles", "backend_cycles", "baseline_cycles", "ttb_cycles",
                "baseline_e2e_cycles", "ttb_e2e_cycles", "row_execution_speedup",
                "row_e2e_speedup",
            )
        },
        "row_outcomes": {
            "ttb_faster": int((arrays["ttb_e2e_cycles"] < arrays["baseline_e2e_cycles"]).sum()),
            "ttb_equal": int((arrays["ttb_e2e_cycles"] == arrays["baseline_e2e_cycles"]).sum()),
            "ttb_slower": int((arrays["ttb_e2e_cycles"] > arrays["baseline_e2e_cycles"]).sum()),
            "slower_ratio": float(slower.mean()),
            "slowdown_cycles": distribution(cycle_delta[slower]),
            "slower_active_pair_ratio": distribution(density[slower]),
            "by_stage": {
                str(stage): dict(values)
                for stage, values in sorted(stage_outcomes.items())
            },
        },
        "ideal_row_adaptive_upper_bound": {
            "description": "每行零开销选择min(RQTB2S,TTB8)的不可实现理想上界",
            "ttb8_e2e_cycles": int(arrays["ttb_e2e_cycles"].sum()),
            "ideal_e2e_cycles": adaptive_e2e_cycles,
            "cycles_saved_vs_ttb8": int(
                arrays["ttb_e2e_cycles"].sum() - adaptive_e2e_cycles
            ),
            "cycle_reduction_vs_ttb8": float(
                1.0 - adaptive_e2e_cycles / arrays["ttb_e2e_cycles"].sum()
            ),
            "speedup_vs_baseline": float(
                arrays["baseline_e2e_cycles"].sum() / adaptive_e2e_cycles
            ),
        },
        "window_distributions": {
            key: distribution(windows[key])
            for key in ("baseline_e2e_cycles", "ttb_e2e_cycles", "speedup")
        },
        "sample_speedup_distribution": distribution(sample_speedups),
        "samples_detail": final_samples,
        "stages": {
            str(stage): finalize_totals(totals)
            for stage, totals in sorted(stages.items())
        },
        "blocks": {
            name: finalize_totals(totals)
            for name, totals in sorted(blocks.items())
        },
        "density_bins": density_bins,
    }


def make_saif_manifest(
    profile: dict[str, Any], records: list[dict[str, Any]], source: dict[str, Any]
) -> dict[str, Any]:
    sample_count = int(profile["samples"])
    sample_ids = sorted(set(np.linspace(0, sample_count - 1, 10, dtype=int).tolist()))
    block_contracts = []
    for record in records:
        if int(record["sample_id"]) != 0:
            continue
        stage, block, name = block_identity(record)
        windows = int(record["batch_windows"])
        window_ids = sorted({0, (windows - 1) // 2, windows - 1})
        block_contracts.append(
            {
                "name": name,
                "stage": stage,
                "block": block,
                "heads": int(record["num_heads"]),
                "available_windows": windows,
                "window_ids": window_ids,
            }
        )
    capture_rows = len(sample_ids) * sum(
        row["heads"] * len(row["window_ids"]) for row in block_contracts
    )
    return {
        "schema": "h67_zkqi_saif_capture_contract_v1",
        "status": "WAIT_RAW_QK_CAPTURE",
        "evidence_level": "[待验证]",
        "selection_policy": "10个等间隔sample；每block固定首/中/末window；覆盖全部head，禁止按收益挑样本",
        "sample_ids": sample_ids,
        "blocks": block_contracts,
        "capture_head_rows": capture_rows,
        "required_raw_fields_per_pair": ["Q0[31:0]", "Q1[31:0]", "K0[31:0]", "K1[31:0]"],
        "candidates": ["RQTB2S", "PairBitmap-ZKQI", "TTB8-ZKQI"],
        "stimulus_modes": {
            "main_power": "无反压，三方同输入、同5ns约束、同目标SRAM宏",
            "secondary_stress": "固定周期descriptor/output反压，仅作鲁棒性活动敏感性",
        },
        "required_external_flow": [
            "目标工艺DC/STA同约束综合",
            "门级SDF仿真导出分层SAIF",
            "PTPX按score、metadata、SRAM、SCS、emit分层功耗",
            "报告mean/p95/p99能量与TTB最差样本，不只报总和",
        ],
        "forbidden_claims_before_completion": [
            "不得把ordered count trace称为门级切换",
            "不得把工作事件按任意权重求和称为功耗或能量",
            "不得把开放fakeram45 32-bit代理称为目标20-bit SRAM PPA",
        ],
        "source_profile": source,
    }


def render_md(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    global_row = analysis["global"]
    rows = analysis["row_distributions"]
    windows = analysis["window_distributions"]
    samples = analysis["sample_speedup_distribution"]
    outcomes = analysis["row_outcomes"]
    adaptive = analysis["ideal_row_adaptive_upper_bound"]
    calibration = report["rtl_calibration"]
    lines = [
        "# Motion全量Ordered Trace与RTL校准周期分布",
        "",
        "## 结论",
        "",
        "- 状态：**PASS**。证据等级为`[prof]+[rtl校准模型]`，不是门级SAIF、功耗或ASIC PPA。",
        f"- 覆盖`{analysis['samples']}`个真实sample、`{analysis['records']}`个attention记录、"
        f"`{analysis['block_windows']}`个block-window和`{analysis['head_rows']}`条head-row；不再只代表sample0/window0。",
        f"- TTB8-ZKQI相对RQTB2S的全量总执行加速为`{global_row['execution_speedup']:.4f}x`；"
        f"计入共同225拍preload后为`{global_row['preload_inclusive_speedup']:.4f}x`。",
        f"- score次数减少`{global_row['score_evaluation_reduction']:.2%}`，Q/K读取bit减少"
        f"`{global_row['qk_read_bit_reduction']:.2%}`；二者是工作事件，不是能耗。",
        f"- 逐sample含preload加速分布为mean/p95/p99="
        f"`{samples['mean']:.4f}x/{samples['p95']:.4f}x/{samples['p99']:.4f}x`，"
        f"最差/最好=`{samples['min']:.4f}x/{samples['max']:.4f}x`。",
        "",
        "## 1. RTL零残差校准",
        "",
        "全量模型先在同一checkpoint的sample0/window0 raw Q/K vector上校准，再外推到count trace。",
        "",
        "| 校准项 | 结果 |",
        "|---|---:|",
        f"| raw vector/profile Q-K-count与H67 score | {calibration['profile_vector_score_mismatches']} mismatch |",
        f"| 138行RTL字段 | {calibration['checked_rtl_fields']}个字段，{calibration['rtl_cycle_field_mismatches']} mismatch |",
        f"| RQTB2S/PairBitmap/TTB8 cycle残差 | {calibration['cycle_residual_min']}..{calibration['cycle_residual_max']} cycle |",
        f"| 12 block的TTB active/K/motion ordered trace交叉检查 | {calibration['ttb_ordered_trace_checks']} |",
        "",
        "无反压逐行周期公式由RTL状态机直接推导：",
        "",
        "```text",
        "backend = 3 + occupied_classes + 2*active_descriptors",
        "          + emitted_K_tokens + I(active_descriptors > 0)",
        "RQTB2S = 225 + backend",
        "PairBitmap-ZKQI = 225 + backend",
        "TTB8-ZKQI = depth1_ordered_bundle_cycles + backend",
        "```",
        "",
        "其中单槽TTB允许空bundle在旧descriptor消费期间穿越，但非空bundle必须等待槽空或与旧descriptor最后一项同拍交接。",
        "",
        "## 2. 全量总账",
        "",
        "| 指标 | RQTB2S | PairBitmap-ZKQI | TTB8-ZKQI | TTB8相对基线 |",
        "|---|---:|---:|---:|---:|",
        f"| 执行周期 | {global_row['baseline_cycles']} | {global_row['pair_bitmap_cycles']} | {global_row['ttb_cycles']} | {global_row['execution_speedup']:.4f}x |",
        f"| 含preload周期 | {global_row['baseline_e2e_cycles']} | {global_row['pair_bitmap_e2e_cycles']} | {global_row['ttb_e2e_cycles']} | {global_row['preload_inclusive_speedup']:.4f}x |",
        f"| score次数 | {analysis['head_rows'] * 225} | {global_row['active_pairs']} | {global_row['active_pairs']} | -{global_row['score_evaluation_reduction']:.2%} |",
        f"| Q/K读bit | {global_row['baseline_read_bits']} | {global_row['candidate_read_bits']} | {global_row['candidate_read_bits']} | -{global_row['qk_read_bit_reduction']:.2%} |",
        "",
        "PairBitmap和TTB8共享exact zero-K work gating；只有TTB8把225项逐pair issue改成29个header与活动pair的有序流水。因此PairBitmap是因果消融，不是被弱化的基线。",
        "",
        "## 3. 多窗口与尾部分布",
        "",
        "| 粒度 | speedup mean | p50 | p95 | p99 | min | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| head-row（执行） | {rows['row_execution_speedup']['mean']:.4f} | {rows['row_execution_speedup']['p50']:.4f} | {rows['row_execution_speedup']['p95']:.4f} | {rows['row_execution_speedup']['p99']:.4f} | {rows['row_execution_speedup']['min']:.4f} | {rows['row_execution_speedup']['max']:.4f} |",
        f"| head-row（含preload） | {rows['row_e2e_speedup']['mean']:.4f} | {rows['row_e2e_speedup']['p50']:.4f} | {rows['row_e2e_speedup']['p95']:.4f} | {rows['row_e2e_speedup']['p99']:.4f} | {rows['row_e2e_speedup']['min']:.4f} | {rows['row_e2e_speedup']['max']:.4f} |",
        f"| block-window（含preload） | {windows['speedup']['mean']:.4f} | {windows['speedup']['p50']:.4f} | {windows['speedup']['p95']:.4f} | {windows['speedup']['p99']:.4f} | {windows['speedup']['min']:.4f} | {windows['speedup']['max']:.4f} |",
        f"| sample（含preload） | {samples['mean']:.4f} | {samples['p50']:.4f} | {samples['p95']:.4f} | {samples['p99']:.4f} | {samples['min']:.4f} | {samples['max']:.4f} |",
        "",
        f"逐head-row faster/equal/slower=`{outcomes['ttb_faster']}/{outcomes['ttb_equal']}/{outcomes['ttb_slower']}`；"
        f"slower占`{outcomes['slower_ratio']:.2%}`，慢行损失p50/p99/max="
        f"`{outcomes['slowdown_cycles']['p50']}/{outcomes['slowdown_cycles']['p99']}/{outcomes['slowdown_cycles']['max']}`拍。",
        f"即使假设每行零开销选择RQTB2S或TTB8，理想选择器也只再省`{adaptive['cycles_saved_vs_ttb8']}`拍，"
        f"占TTB8总周期`{adaptive['cycle_reduction_vs_ttb8']:.4%}`；当前不值得为它增加运行时双模式控制。",
        "",
        "## 4. 分stage结果",
        "",
        "| Stage | rows | active pair | score减少 | Q/K读bit减少 | 执行加速 | 含preload加速 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, row in analysis["stages"].items():
        lines.append(
            f"| {stage} | {row['rows']} | {row['active_pair_ratio']:.2%} | "
            f"{row['score_evaluation_reduction']:.2%} | {row['qk_read_bit_reduction']:.2%} | "
            f"{row['execution_speedup']:.4f}x | {row['preload_inclusive_speedup']:.4f}x |"
        )
    lines += [
        "",
        "## 5. 分block结果",
        "",
        "| Block | rows | active pair | 含preload加速 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in analysis["blocks"].items():
        lines.append(
            f"| {name} | {row['rows']} | {row['active_pair_ratio']:.2%} | "
            f"{row['preload_inclusive_speedup']:.4f}x |"
        )
    lines += [
        "",
        "## 6. 稠密度敏感性",
        "",
        "| active-pair区间 | rows | 平均密度 | 聚合含preload加速 | row p50/p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in analysis["density_bins"]:
        lines.append(
            f"| [{row['lower']:.2f}, {row['upper']:.2f}) | {row['rows']} | "
            f"{row['active_pair_ratio_mean']:.2%} | {row['preload_inclusive_speedup']:.4f}x | "
            f"{row['row_speedup']['p50']:.4f}/{row['row_speedup']['p99']:.4f} |"
        )
    lines += [
        "",
        "## 7. 对架构的直接指导",
        "",
        "1. PairBitmap只负责证明zero-K work gating；它不减少无反压周期，不能单独作为主架构贡献。",
        "2. TTB8的收益来自有序层次issue，而不是score近似；慢行只损失1拍，理想双模式上界仍为万分级收益。",
        "3. 因此否决按row自适应的RQTB2S/TTB8选择器；下一轮应评估更粗TTB粒度能否在不复制bitmap的条件下减少header扫描，而不是增加旁路控制。",
        "4. 本profile没有原始Q/K bit身份，无法生成真实门级切换。已冻结`saif_capture_manifest.json`，后续GPU空闲时按预提交sample/window抓取，禁止事后挑高收益样本。",
        "",
        "## 8. 证据边界",
        "",
        "- `[prof]`：100个sample的ordered count trace、stage/block/window/head覆盖与工作事件；",
        "- `[rtl校准模型]`：周期公式在138条raw-vector RTL行上零残差，但其余样本没有逐bit RTL replay；",
        "- `[待验证]`：门级SAIF、目标20-bit SRAM、DC/STA/PTPX、功耗、能量和EDP；",
        "- 不得把本报告称为门级功耗或ASIC PPA，也不得从开放宏代理推导目标工艺结论。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--vector", type=Path, default=DEFAULT_VECTOR)
    parser.add_argument("--pair-log", type=Path, default=DEFAULT_PAIR_LOG)
    parser.add_argument("--ttb-log", type=Path, default=DEFAULT_TTB_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    records = profile.get("summary", {}).get("h60_records") or []
    if not profile.get("ordered_trace"):
        raise ValueError("输入profile没有ordered trace")
    profile_contract = validate_profile_contract(profile, records)

    calibration = calibrate(records, args.vector, args.pair_log, args.ttb_log)
    analysis = analyze_all(records)
    source_profile = receipt(args.profile)
    manifest = make_saif_manifest(profile, records, source_profile)
    report = {
        "schema": "h67_zkqi_multisample_ordered_rtl_calibrated_v1",
        "status": "PASS",
        "evidence_level": "[prof]+[rtl校准模型]",
        "scope": "H67 ep30 fullres T450，100 sample、12 attention block、全部window/head",
        "rtl_calibration": calibration,
        "profile_contract": profile_contract,
        "analysis": analysis,
        "saif_status": manifest["status"],
        "source_receipts": {
            "profile": source_profile,
            "sample0_vector": receipt(args.vector),
            "pair_bitmap_rtl_log": receipt(args.pair_log),
            "ttb8_rtl_log": receipt(args.ttb_log),
        },
        "artifact_identity": profile.get("artifact_identity"),
        "eval_protocol": profile.get("eval_protocol"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_md(report), encoding="utf-8")
    (args.output_dir / "saif_capture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"PASS rows={analysis['head_rows']} windows={analysis['block_windows']} "
        f"speedup={analysis['global']['preload_inclusive_speedup']:.6f} "
        f"sample_min={analysis['sample_speedup_distribution']['min']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
