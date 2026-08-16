#!/usr/bin/env python3
"""Deterministic, data-only statistics for the Local5 EREP G0 gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATES = ("c0", "c1", "c2", "c3", "c4")
STAGES = (0, 1, 2, 3)
EXPECTED_SEQUENCE_CLUSTERS = 18
BOOTSTRAP_TRIALS = 20_000
BOOTSTRAP_SEED = 20260810
PERCENTILE_METHOD = "inverted_cdf"
STAGE_P95_QUANTILE = 0.95
STAGE_FAMILYWISE_ALPHA = 0.05
STAGE_TESTS = 4
STAGE_BONFERRONI_ALPHA = STAGE_FAMILYWISE_ALPHA / STAGE_TESTS
STAGE_UPPER_QUANTILE = 1.0 - STAGE_BONFERRONI_ALPHA

G0_THRESHOLDS = {
    "primary_speedup_c0_over_c3": (">=", 1.25),
    "primary_bootstrap_ci95_lower": (">", 1.0),
    "all_stage_p95_delta_upper_bounds": ("<=", 0.0),
    "synergy_speedup": (">=", 1.05),
    "synergy_bootstrap_ci95_lower": (">", 1.0),
    "capacity_matched_speedup": (">=", 1.05),
    "capacity_matched_bootstrap_ci95_lower": (">", 1.0),
}


def _float_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional vector")
    return result


def _validated_weights(
    weights: Sequence[float] | np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    result = _float_vector(weights, "weights")
    if result.shape != shape or np.any(result <= 0.0):
        raise ValueError("weights must match the values and be strictly positive")
    if not np.isfinite(result.sum(dtype=np.float64)):
        raise ValueError("the total weight must be finite")
    return result


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    quantile: float,
) -> float:
    """Return inf{x: weighted empirical CDF(x) >= quantile}.

    Exact cumulative-weight boundaries select the value on the left.  This is
    deliberately an inverse CDF implemented with ``searchsorted(..., left)``;
    no interpolation is permitted.
    """

    array = _float_vector(values, "values")
    weight_array = _validated_weights(weights, array.shape)
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be finite and in [0, 1]")

    order = np.argsort(array, kind="stable")
    cumulative = np.cumsum(weight_array[order], dtype=np.float64)
    if not np.isfinite(cumulative[-1]):
        raise ValueError("the cumulative weight must be finite")
    target = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(array[order[min(index, array.size - 1)]])


def weighted_percentile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    percentile: float,
) -> float:
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be finite and in [0, 100]")
    return weighted_quantile(values, weights, float(percentile) / 100.0)


def inverted_cdf_quantile(
    values: Sequence[float] | np.ndarray, quantile: float
) -> float:
    """Return an unweighted empirical quantile with no interpolation."""

    array = _float_vector(values, "values")
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be finite and in [0, 1]")
    return float(np.quantile(array, quantile, method=PERCENTILE_METHOD))


def inverted_cdf_percentile(
    values: Sequence[float] | np.ndarray, percentile: float
) -> float:
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be finite and in [0, 100]")
    return inverted_cdf_quantile(values, float(percentile) / 100.0)


def ratio_of_weighted_sums(
    numerator: Sequence[float] | np.ndarray,
    denominator: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> float:
    """Compute sum(w * numerator) / sum(w * denominator)."""

    numerator_array = _float_vector(numerator, "numerator")
    denominator_array = _float_vector(denominator, "denominator")
    if denominator_array.shape != numerator_array.shape:
        raise ValueError("numerator and denominator shapes must match")
    weight_array = _validated_weights(weights, numerator_array.shape)
    numerator_total = float(np.dot(weight_array, numerator_array))
    denominator_total = float(np.dot(weight_array, denominator_array))
    if not np.isfinite(numerator_total) or not np.isfinite(denominator_total):
        raise ValueError("weighted totals must be finite")
    if denominator_total <= 0.0:
        raise ValueError("the weighted denominator total must be positive")
    return numerator_total / denominator_total


weighted_sum_ratio = ratio_of_weighted_sums


def _validated_sequence_keys(
    sequence_keys: Sequence[str] | np.ndarray,
    row_count: int,
    expected_clusters: int,
) -> tuple[np.ndarray, tuple[str, ...], tuple[np.ndarray, ...]]:
    keys = np.asarray(sequence_keys)
    if keys.ndim != 1 or keys.size != row_count:
        raise ValueError("sequence_keys must be one-dimensional and match the rows")
    if any(not isinstance(value, (str, np.str_)) or not str(value) for value in keys):
        raise ValueError("every sequence key must be a nonempty string")
    normalized = np.asarray([str(value) for value in keys], dtype=np.str_)
    unique = tuple(sorted(set(normalized.tolist())))
    if len(unique) != expected_clusters:
        raise ValueError(
            f"expected exactly {expected_clusters} sequence clusters, got {len(unique)}"
        )
    members = tuple(np.flatnonzero(normalized == key) for key in unique)
    return normalized, unique, members


def _validated_bootstrap_parameters(trials: int, seed: int) -> tuple[int, int]:
    if isinstance(trials, bool) or not isinstance(trials, (int, np.integer)):
        raise ValueError("trials must be a positive integer")
    if trials <= 0:
        raise ValueError("trials must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(trials), int(seed)


def sequence_cluster_expansions(
    sequence_keys: Sequence[str] | np.ndarray,
    *,
    trials: int = BOOTSTRAP_TRIALS,
    seed: int = BOOTSTRAP_SEED,
    expected_clusters: int = EXPECTED_SEQUENCE_CLUSTERS,
) -> Iterable[np.ndarray]:
    """Yield paired cluster-bootstrap row indices in deterministic draw order.

    Each replicate draws ``expected_clusters`` clusters with replacement.  A
    selected cluster contributes every member row on every occurrence, so a
    cluster selected twice is expanded twice rather than deduplicated.
    """

    trials, seed = _validated_bootstrap_parameters(trials, seed)
    if (
        isinstance(expected_clusters, bool)
        or not isinstance(expected_clusters, (int, np.integer))
        or expected_clusters <= 0
    ):
        raise ValueError("expected_clusters must be a positive integer")

    _, _, members = _validated_sequence_keys(
        sequence_keys, len(sequence_keys), int(expected_clusters)
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(trials):
        selected = rng.integers(
            0, int(expected_clusters), size=int(expected_clusters), dtype=np.int64
        )
        yield np.concatenate(tuple(members[int(cluster)] for cluster in selected))


def _validated_inputs(
    cycles: Mapping[str, Sequence[float] | np.ndarray],
    weights: Sequence[float] | np.ndarray,
    stages: Sequence[int] | np.ndarray,
    sequence_keys: Sequence[str] | np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    if set(cycles) != set(CANDIDATES):
        raise ValueError(f"cycles must contain exactly {CANDIDATES}")
    cycle_arrays = {
        name: _float_vector(cycles[name], f"cycles[{name}]") for name in CANDIDATES
    }
    shape = cycle_arrays["c0"].shape
    if any(array.shape != shape for array in cycle_arrays.values()):
        raise ValueError("all candidate cycle vectors must have the same shape")
    if any(np.any(array < 0.0) for array in cycle_arrays.values()):
        raise ValueError("cycle counts must be nonnegative")
    weight_array = _validated_weights(weights, shape)

    stage_array = np.asarray(stages)
    if stage_array.ndim != 1 or stage_array.shape != shape:
        raise ValueError("stages must be one-dimensional and match the rows")
    if not np.issubdtype(stage_array.dtype, np.integer):
        raise ValueError("stages must contain integers")
    stage_array = stage_array.astype(np.int64, copy=False)
    if set(stage_array.tolist()) != set(STAGES):
        raise ValueError(f"stages must contain exactly {STAGES}")

    normalized_keys, unique_keys, _ = _validated_sequence_keys(
        sequence_keys, shape[0], EXPECTED_SEQUENCE_CLUSTERS
    )
    for key in unique_keys:
        cluster_stages = set(stage_array[normalized_keys == key].tolist())
        if cluster_stages != set(STAGES):
            raise ValueError(f"sequence cluster {key!r} must contain every stage")

    for denominator in ("c1", "c2", "c3", "c4"):
        if float(np.dot(weight_array, cycle_arrays[denominator])) <= 0.0:
            raise ValueError(f"cycles[{denominator}] has a nonpositive weighted total")
    return cycle_arrays, weight_array, stage_array, normalized_keys


def sequence_cluster_bootstrap_replicates(
    cycles: Mapping[str, Sequence[float] | np.ndarray],
    weights: Sequence[float] | np.ndarray,
    stages: Sequence[int] | np.ndarray,
    sequence_keys: Sequence[str] | np.ndarray,
    *,
    trials: int = BOOTSTRAP_TRIALS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Compute all EREP estimands from the same paired cluster replicates."""

    trials, seed = _validated_bootstrap_parameters(trials, seed)
    cycle_arrays, weight_array, stage_array, key_array = _validated_inputs(
        cycles, weights, stages, sequence_keys
    )
    ratio_names = tuple(f"c0_over_{name}" for name in CANDIDATES[1:])
    result = {
        name: np.empty(trials, dtype=np.float64)
        for name in (*ratio_names, "synergy", "capacity_matched")
    }
    stage_deltas = np.empty((trials, len(STAGES)), dtype=np.float64)

    expansions = sequence_cluster_expansions(
        key_array,
        trials=trials,
        seed=seed,
        expected_clusters=EXPECTED_SEQUENCE_CLUSTERS,
    )
    for replicate, indices in enumerate(expansions):
        replicate_weights = weight_array[indices]
        totals = {
            name: float(np.dot(replicate_weights, values[indices]))
            for name, values in cycle_arrays.items()
        }
        if any(not np.isfinite(total) for total in totals.values()):
            raise ValueError("a bootstrap weighted total is nonfinite")
        if any(totals[name] <= 0.0 for name in CANDIDATES[1:]):
            raise ValueError("a bootstrap denominator is nonpositive")

        for name in CANDIDATES[1:]:
            result[f"c0_over_{name}"][replicate] = totals["c0"] / totals[name]
        # The best single mechanism is selected again inside every replicate.
        result["synergy"][replicate] = (
            min(totals["c1"], totals["c2"]) / totals["c3"]
        )
        # The capacity-matched comparison is also recomputed per replicate.
        result["capacity_matched"][replicate] = totals["c4"] / totals["c3"]

        replicate_stages = stage_array[indices]
        for column, stage in enumerate(STAGES):
            mask = replicate_stages == stage
            c0_p95 = weighted_quantile(
                cycle_arrays["c0"][indices][mask],
                replicate_weights[mask],
                STAGE_P95_QUANTILE,
            )
            c3_p95 = weighted_quantile(
                cycle_arrays["c3"][indices][mask],
                replicate_weights[mask],
                STAGE_P95_QUANTILE,
            )
            stage_deltas[replicate, column] = c3_p95 - c0_p95

    result["stage_p95_delta_c3_minus_c0"] = stage_deltas
    return result


def percentile_interval(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    array = _float_vector(values, "bootstrap values")
    return {
        "method": PERCENTILE_METHOD,
        "confidence": 0.95,
        "lower_quantile": 0.025,
        "upper_quantile": 0.975,
        "lower": inverted_cdf_quantile(array, 0.025),
        "upper": inverted_cdf_quantile(array, 0.975),
    }


def stage_p95_upper_bounds(stage_deltas: np.ndarray) -> np.ndarray:
    values = np.asarray(stage_deltas, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != len(STAGES)
        or values.shape[0] == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("stage deltas must have shape (replicates, 4) and be finite")
    return np.asarray(
        [inverted_cdf_quantile(values[:, column], STAGE_UPPER_QUANTILE) for column in STAGES],
        dtype=np.float64,
    )


def passes_comparison(value: float, comparison: str, threshold: float) -> bool:
    """Apply an exact preregistered comparison without tolerances."""

    if not np.isfinite(value) or not np.isfinite(threshold):
        raise ValueError("comparison values must be finite")
    if comparison == ">=":
        return bool(value >= threshold)
    if comparison == ">":
        return bool(value > threshold)
    if comparison == "<=":
        return bool(value <= threshold)
    raise ValueError(f"unsupported comparison: {comparison!r}")


def _gate(name: str, value: float) -> dict[str, Any]:
    comparison, threshold = G0_THRESHOLDS[name]
    return {
        "name": name,
        "value": float(value),
        "comparison": comparison,
        "threshold": threshold,
        "passed": passes_comparison(float(value), comparison, threshold),
    }


def evaluate_g0(
    cycles: Mapping[str, Sequence[float] | np.ndarray],
    weights: Sequence[float] | np.ndarray,
    stages: Sequence[int] | np.ndarray,
    sequence_keys: Sequence[str] | np.ndarray,
    *,
    trials: int = BOOTSTRAP_TRIALS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate the frozen EREP G0 estimands and exact threshold gates."""

    cycle_arrays, weight_array, stage_array, key_array = _validated_inputs(
        cycles, weights, stages, sequence_keys
    )
    replicates = sequence_cluster_bootstrap_replicates(
        cycle_arrays,
        weight_array,
        stage_array,
        key_array,
        trials=trials,
        seed=seed,
    )
    totals = {
        name: float(np.dot(weight_array, values))
        for name, values in cycle_arrays.items()
    }

    ratios: dict[str, Any] = {}
    for name in CANDIDATES[1:]:
        key = f"c0_over_{name}"
        ratios[key] = {
            "formula": f"sum(w*C0)/sum(w*{name.upper()})",
            "estimate": totals["c0"] / totals[name],
            "bootstrap_ci95": percentile_interval(replicates[key]),
        }

    synergy_estimate = min(totals["c1"], totals["c2"]) / totals["c3"]
    capacity_estimate = totals["c4"] / totals["c3"]
    synergy = {
        "formula": "min(sum(w*C1),sum(w*C2))/sum(w*C3)",
        "replicate_rule": "recompute min(total_C1,total_C2)/total_C3",
        "estimate": synergy_estimate,
        "bootstrap_ci95": percentile_interval(replicates["synergy"]),
    }
    capacity_matched = {
        "formula": "sum(w*C4)/sum(w*C3)",
        "replicate_rule": "recompute total_C4/total_C3",
        "estimate": capacity_estimate,
        "bootstrap_ci95": percentile_interval(replicates["capacity_matched"]),
    }

    upper_bounds = stage_p95_upper_bounds(
        replicates["stage_p95_delta_c3_minus_c0"]
    )
    stage_rows = []
    for column, stage in enumerate(STAGES):
        mask = stage_array == stage
        c0_p95 = weighted_quantile(
            cycle_arrays["c0"][mask], weight_array[mask], STAGE_P95_QUANTILE
        )
        c3_p95 = weighted_quantile(
            cycle_arrays["c3"][mask], weight_array[mask], STAGE_P95_QUANTILE
        )
        upper_bound = float(upper_bounds[column])
        stage_rows.append(
            {
                "stage": stage,
                "weighted_p95_method": "left_continuous_inverse_cdf_searchsorted_left",
                "c0_p95": c0_p95,
                "c3_p95": c3_p95,
                "delta_c3_minus_c0": c3_p95 - c0_p95,
                "bootstrap_upper_bound": upper_bound,
                "upper_bound_method": PERCENTILE_METHOD,
                "one_sided_confidence": STAGE_UPPER_QUANTILE,
                "bonferroni_alpha": STAGE_BONFERRONI_ALPHA,
                "comparison": "<=",
                "threshold": 0.0,
                "passed": passes_comparison(upper_bound, "<=", 0.0),
            }
        )

    gates = [
        _gate("primary_speedup_c0_over_c3", ratios["c0_over_c3"]["estimate"]),
        _gate(
            "primary_bootstrap_ci95_lower",
            ratios["c0_over_c3"]["bootstrap_ci95"]["lower"],
        ),
        {
            "name": "all_stage_p95_delta_upper_bounds",
            "values": upper_bounds.tolist(),
            "comparison": "<=",
            "threshold": 0.0,
            "passed": all(row["passed"] for row in stage_rows),
        },
        _gate("synergy_speedup", synergy_estimate),
        _gate(
            "synergy_bootstrap_ci95_lower", synergy["bootstrap_ci95"]["lower"]
        ),
        _gate("capacity_matched_speedup", capacity_estimate),
        _gate(
            "capacity_matched_bootstrap_ci95_lower",
            capacity_matched["bootstrap_ci95"]["lower"],
        ),
    ]

    return {
        "schema": "local5_erep_g0_statistics_v3",
        "rows": int(weight_array.size),
        "sequence_clusters": EXPECTED_SEQUENCE_CLUSTERS,
        "weighted_totals": totals,
        "determinism": {
            "bootstrap_method": "paired_sequence_cluster_complete_expansion",
            "cluster_order": "lexicographic_sequence_key",
            "bit_generator": "numpy.PCG64",
            "seed": int(seed),
            "trials": int(trials),
            "percentile_method": PERCENTILE_METHOD,
        },
        "ratios": ratios,
        "synergy": synergy,
        "capacity_matched": capacity_matched,
        "stage_p95": stage_rows,
        "g0_gates": gates,
        "g0_passed": all(gate["passed"] for gate in gates),
    }


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    trials: int = BOOTSTRAP_TRIALS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must be nonempty")
    required = {"weight", "stage", "sequence_key", *CANDIDATES}
    for index, row in enumerate(rows):
        if set(row) != required:
            raise ValueError(f"row {index} must contain exactly {sorted(required)}")
    cycles = {name: [row[name] for row in rows] for name in CANDIDATES}
    return evaluate_g0(
        cycles,
        [row["weight"] for row in rows],
        [row["stage"] for row in rows],
        [row["sequence_key"] for row in rows],
        trials=trials,
        seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON list, or object with a rows list")
    parser.add_argument("--trials", type=int, default=BOOTSTRAP_TRIALS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    rows = value.get("rows") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("input JSON must be a row list or contain a rows list")
    print(json.dumps(evaluate_rows(rows, trials=args.trials, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
