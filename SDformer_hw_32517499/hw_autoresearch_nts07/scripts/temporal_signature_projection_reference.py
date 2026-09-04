#!/usr/bin/env python3
"""时间签名复用投影的整数等价参考。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def dense_projection(
    k0: np.ndarray, k1: np.ndarray, weight: np.ndarray, gate0: np.ndarray, gate1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p0 = k0.astype(np.int64) @ weight.astype(np.int64)
    p1 = k1.astype(np.int64) @ weight.astype(np.int64)
    return gate0[:, None].astype(np.int64) * p0, gate1[:, None].astype(np.int64) * p1


def temporal_signature_projection(
    k0: np.ndarray, k1: np.ndarray, weight: np.ndarray, gate0: np.ndarray, gate1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    only0 = k0 & ~k1
    only1 = k1 & ~k0
    common = k0 & k1
    p_only0 = only0.astype(np.int64) @ weight.astype(np.int64)
    p_only1 = only1.astype(np.int64) @ weight.astype(np.int64)
    p_common = common.astype(np.int64) @ weight.astype(np.int64)
    y0 = gate0[:, None].astype(np.int64) * (p_only0 + p_common)
    y1 = gate1[:, None].astype(np.int64) * (p_only1 + p_common)
    baseline_reads = int(k0.sum() + k1.sum())
    union_reads = int((k0 | k1).sum())
    intersection_reuse = int(common.sum())
    if baseline_reads != union_reads + intersection_reuse:
        raise AssertionError("时间签名权重读取守恒失败")
    return y0, y1, {
        "baseline_weight_row_reads": baseline_reads,
        "union_weight_row_reads": union_reads,
        "intersection_reused_reads": intersection_reuse,
    }


def run_trials(seed: int = 20260713, trials: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    mismatches = 0
    baseline_reads = 0
    union_reads = 0
    reused_reads = 0
    compared = 0
    for _ in range(trials):
        k0 = rng.random((81, 32)) < rng.uniform(0.0, 0.35)
        k1 = rng.random((81, 32)) < rng.uniform(0.0, 0.35)
        weight = rng.integers(-128, 128, size=(32, 64), dtype=np.int16)
        gate0 = rng.integers(0, 256, size=(81,), dtype=np.int16)
        gate1 = rng.integers(0, 256, size=(81,), dtype=np.int16)
        dense0, dense1 = dense_projection(k0, k1, weight, gate0, gate1)
        factored0, factored1, traffic = temporal_signature_projection(
            k0, k1, weight, gate0, gate1
        )
        mismatches += int(np.count_nonzero(dense0 != factored0))
        mismatches += int(np.count_nonzero(dense1 != factored1))
        compared += dense0.size + dense1.size
        baseline_reads += traffic["baseline_weight_row_reads"]
        union_reads += traffic["union_weight_row_reads"]
        reused_reads += traffic["intersection_reused_reads"]
    return {
        "seed": seed,
        "trials": trials,
        "compared_outputs": compared,
        "mismatches": mismatches,
        "synthetic_baseline_weight_row_reads": baseline_reads,
        "synthetic_union_weight_row_reads": union_reads,
        "synthetic_intersection_reused_reads": reused_reads,
        "synthetic_exact_reuse_ratio": reused_reads / baseline_reads if baseline_reads else 0.0,
        "限制": "随机稀疏率只验证代数和计数器，不可作为H67真实复用率；真实值等待ordered profile100。",
    }


def main() -> int:
    result = run_trials()
    path = ROOT / "results/temporal_signature_projection_reference.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)
    return int(result["mismatches"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
