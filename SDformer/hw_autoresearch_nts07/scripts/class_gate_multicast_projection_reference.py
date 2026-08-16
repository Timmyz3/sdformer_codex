#!/usr/bin/env python3
"""类门控多播投影（CGMP）的整数等价参考。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def dense_selector_projection(
    k_event: np.ndarray,
    score_class: np.ndarray,
    gate_by_class: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    token_gate = gate_by_class[score_class].astype(np.int64)
    gated_k = k_event.astype(np.int64) * token_gate[:, None]
    return gated_k @ weight.astype(np.int64)


def class_gate_multicast_projection(
    k_event: np.ndarray,
    score_class: np.ndarray,
    gate_by_class: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """每个类通道只计算一次gate乘权重，再按K bitmap多播。"""

    tokens, channels = k_event.shape
    outputs = int(weight.shape[1])
    result = np.zeros((tokens, outputs), dtype=np.int64)
    class_channel_terms = 0
    max_fanout = 0
    for class_id in np.unique(score_class):
        token_ids = np.flatnonzero(score_class == class_id)
        class_k = k_event[token_ids]
        active_channels = np.flatnonzero(class_k.any(axis=0))
        gate = np.int64(gate_by_class[class_id])
        for channel in active_channels:
            destinations = token_ids[np.flatnonzero(class_k[:, channel])]
            product = gate * weight[channel].astype(np.int64)
            result[destinations] += product
            class_channel_terms += 1
            max_fanout = max(max_fanout, int(destinations.size))

    active_lanes = int(k_event.sum())
    if class_channel_terms > active_lanes:
        raise AssertionError("类通道项不能超过活动K lane基线")
    return result, {
        "baseline_active_lanes": active_lanes,
        "class_channel_terms": class_channel_terms,
        "max_token_fanout": max_fanout,
        "baseline_scalar_multiplies": active_lanes * outputs,
        "cgmp_scalar_multiplies": class_channel_terms * outputs,
        "multicast_accumulations": active_lanes * outputs,
        "output_elements": tokens * outputs,
        "input_channels": channels,
    }


def run_trials(seed: int = 20260713, trials: int = 200) -> dict:
    rng = np.random.default_rng(seed)
    mismatches = 0
    compared_outputs = 0
    baseline_multiplies = 0
    cgmp_multiplies = 0
    multicast_accumulations = 0
    max_fanout = 0
    for _ in range(trials):
        tokens = 162
        channels = 32
        outputs = 32
        occupied_classes = int(rng.integers(1, 9))
        score_class = rng.integers(0, occupied_classes, size=tokens, dtype=np.int16)
        k_event = rng.random((tokens, channels)) < rng.uniform(0.01, 0.25)
        gate_by_class = rng.integers(0, 256, size=occupied_classes, dtype=np.int16)
        weight = rng.integers(-128, 128, size=(channels, outputs), dtype=np.int16)
        expected = dense_selector_projection(k_event, score_class, gate_by_class, weight)
        actual, counters = class_gate_multicast_projection(
            k_event, score_class, gate_by_class, weight
        )
        mismatches += int(np.count_nonzero(expected != actual))
        compared_outputs += int(expected.size)
        baseline_multiplies += counters["baseline_scalar_multiplies"]
        cgmp_multiplies += counters["cgmp_scalar_multiplies"]
        multicast_accumulations += counters["multicast_accumulations"]
        max_fanout = max(max_fanout, counters["max_token_fanout"])
    return {
        "seed": seed,
        "trials": trials,
        "compared_outputs": compared_outputs,
        "mismatches": mismatches,
        "synthetic_baseline_scalar_multiplies": baseline_multiplies,
        "synthetic_cgmp_scalar_multiplies": cgmp_multiplies,
        "synthetic_multicast_accumulations": multicast_accumulations,
        "synthetic_product_reduction": (
            1.0 - cgmp_multiplies / baseline_multiplies if baseline_multiplies else 0.0
        ),
        "synthetic_max_token_fanout": max_fanout,
        "限制": "随机class和稀疏率仅验证代数、计数器与多播语义，不可作为H67真实收益。",
    }


def main() -> int:
    result = run_trials()
    path = ROOT / "results/class_gate_multicast_projection_reference.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    return int(result["mismatches"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
