from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import local5_erep_statistics_v3 as statistics


def constant_fixture(
    *,
    c0: float = 125.0,
    c1: float = 105.0,
    c2: float = 106.0,
    c3: float = 100.0,
    c4: float = 105.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    stages = np.tile(np.arange(4, dtype=np.int64), 18)
    sequence_keys = [f"seq-{cluster:02d}" for cluster in range(18) for _ in range(4)]
    weights = np.tile(np.asarray([440.0, 120.0, 30.0, 10.0]), 18)
    cycles = {
        name: np.full(stages.size, value, dtype=np.float64)
        for name, value in {
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4,
        }.items()
    }
    return cycles, weights, stages, sequence_keys


class Local5ErepStatisticsV3Test(unittest.TestCase):
    def test_weighted_quantile_is_left_inverse_at_exact_boundary(self) -> None:
        values = [10.0, 20.0, 30.0]
        weights = [1.0, 1.0, 2.0]
        self.assertEqual(statistics.weighted_quantile(values, weights, 0.0), 10.0)
        self.assertEqual(statistics.weighted_quantile(values, weights, 0.25), 10.0)
        self.assertEqual(
            statistics.weighted_quantile(values, weights, np.nextafter(0.25, 1.0)),
            20.0,
        )
        self.assertEqual(statistics.weighted_quantile(values, weights, 1.0), 30.0)

    def test_inverted_cdf_uses_fixed_order_statistics(self) -> None:
        values = np.arange(80, dtype=np.float64)
        self.assertEqual(statistics.inverted_cdf_quantile(values, 0.0), 0.0)
        self.assertEqual(statistics.inverted_cdf_quantile(values, 0.5), 39.0)
        self.assertEqual(statistics.inverted_cdf_quantile(values, 0.9875), 78.0)
        self.assertEqual(statistics.inverted_cdf_quantile(values, 1.0), 79.0)

    def test_ratio_is_ratio_of_weighted_sums_not_mean_of_ratios(self) -> None:
        ratio = statistics.ratio_of_weighted_sums(
            [100.0, 1.0], [50.0, 1.0], [1.0, 100.0]
        )
        self.assertEqual(ratio, 200.0 / 150.0)
        self.assertNotEqual(ratio, np.average([2.0, 1.0], weights=[1.0, 100.0]))

    def test_cluster_draw_repeats_every_member_of_repeated_cluster(self) -> None:
        keys = [f"seq-{cluster:02d}" for cluster in range(18) for _ in range(cluster + 1)]
        expansion = next(
            iter(statistics.sequence_cluster_expansions(keys, trials=1))
        )
        selected = np.random.Generator(np.random.PCG64(20260810)).integers(
            0, 18, size=18, dtype=np.int64
        )
        members = [
            np.flatnonzero(np.asarray(keys) == f"seq-{cluster:02d}")
            for cluster in range(18)
        ]
        expected = np.concatenate([members[int(cluster)] for cluster in selected])
        np.testing.assert_array_equal(expansion, expected)
        repeated = int(next(cluster for cluster in selected if (selected == cluster).sum() > 1))
        repeated_members = members[repeated]
        for member in repeated_members:
            self.assertEqual(
                int((expansion == member).sum()), int((selected == repeated).sum())
            )

    def test_golden_constant_fixture_passes_exact_threshold_boundaries(self) -> None:
        cycles, weights, stages, sequence_keys = constant_fixture()
        report = statistics.evaluate_g0(
            cycles, weights, stages, sequence_keys, trials=32
        )

        self.assertTrue(report["g0_passed"])
        self.assertEqual(report["ratios"]["c0_over_c3"]["estimate"], 1.25)
        self.assertEqual(report["synergy"]["estimate"], 1.05)
        self.assertEqual(report["capacity_matched"]["estimate"], 1.05)
        self.assertEqual(
            report["ratios"]["c0_over_c3"]["bootstrap_ci95"]["lower"], 1.25
        )
        self.assertEqual(
            [row["bootstrap_upper_bound"] for row in report["stage_p95"]],
            [-25.0, -25.0, -25.0, -25.0],
        )
        self.assertEqual(
            [gate["comparison"] for gate in report["g0_gates"]],
            [">=", ">", "<=", ">=", ">", ">=", ">"],
        )

    def test_synergy_and_c4_are_recomputed_inside_each_replicate(self) -> None:
        cycles, weights, stages, sequence_keys = constant_fixture()
        for cluster in range(18):
            mask = np.asarray(sequence_keys) == f"seq-{cluster:02d}"
            cycles["c1"][mask] = 80.0 if cluster < 9 else 140.0
            cycles["c2"][mask] = 130.0 if cluster < 9 else 85.0
            cycles["c3"][mask] = 100.0 + cluster
            cycles["c4"][mask] = 110.0 + 2.0 * cluster

        replicates = statistics.sequence_cluster_bootstrap_replicates(
            cycles, weights, stages, sequence_keys, trials=8
        )
        expansions = list(
            statistics.sequence_cluster_expansions(sequence_keys, trials=8)
        )
        selected_best = set()
        for replicate, indices in enumerate(expansions):
            replicate_weights = weights[indices]
            totals = {
                name: float(np.dot(replicate_weights, values[indices]))
                for name, values in cycles.items()
            }
            selected_best.add("c1" if totals["c1"] <= totals["c2"] else "c2")
            self.assertEqual(
                replicates["synergy"][replicate],
                min(totals["c1"], totals["c2"]) / totals["c3"],
            )
            self.assertEqual(
                replicates["capacity_matched"][replicate],
                totals["c4"] / totals["c3"],
            )
        self.assertEqual(selected_best, {"c1", "c2"})

    def test_stage_upper_bound_is_bonferroni_9875_order_statistic(self) -> None:
        values = np.column_stack(
            [np.arange(80, dtype=np.float64) + 100.0 * stage for stage in range(4)]
        )
        np.testing.assert_array_equal(
            statistics.stage_p95_upper_bounds(values),
            np.asarray([78.0, 178.0, 278.0, 378.0]),
        )
        self.assertEqual(statistics.STAGE_BONFERRONI_ALPHA, 0.0125)
        self.assertEqual(statistics.STAGE_UPPER_QUANTILE, 0.9875)

    def test_exact_comparison_boundaries_have_no_tolerance(self) -> None:
        self.assertTrue(statistics.passes_comparison(1.25, ">=", 1.25))
        self.assertFalse(
            statistics.passes_comparison(np.nextafter(1.25, -np.inf), ">=", 1.25)
        )
        self.assertTrue(statistics.passes_comparison(0.0, "<=", 0.0))
        self.assertFalse(
            statistics.passes_comparison(np.nextafter(0.0, np.inf), "<=", 0.0)
        )
        self.assertFalse(statistics.passes_comparison(1.0, ">", 1.0))
        self.assertTrue(
            statistics.passes_comparison(np.nextafter(1.0, np.inf), ">", 1.0)
        )

    def test_bootstrap_is_reproducible_and_declares_pcg64(self) -> None:
        cycles, weights, stages, sequence_keys = constant_fixture(c0=130.0)
        first = statistics.evaluate_g0(
            cycles, weights, stages, sequence_keys, trials=16
        )
        second = statistics.evaluate_g0(
            cycles, weights, stages, sequence_keys, trials=16
        )
        self.assertEqual(first, second)
        self.assertEqual(first["determinism"]["bit_generator"], "numpy.PCG64")
        self.assertEqual(first["determinism"]["seed"], 20260810)

    def test_requires_exactly_18_complete_sequence_clusters(self) -> None:
        cycles, weights, stages, sequence_keys = constant_fixture()
        with self.assertRaisesRegex(ValueError, "exactly 18 sequence clusters"):
            statistics.evaluate_g0(
                {name: values[:-4] for name, values in cycles.items()},
                weights[:-4],
                stages[:-4],
                sequence_keys[:-4],
                trials=1,
            )

        broken_stages = stages.copy()
        broken_stages[0] = 1
        with self.assertRaisesRegex(ValueError, "must contain every stage"):
            statistics.evaluate_g0(
                cycles, weights, broken_stages, sequence_keys, trials=1
            )

    def test_rejects_nonpositive_bootstrap_denominator(self) -> None:
        cycles, weights, stages, sequence_keys = constant_fixture(c3=0.0)
        with self.assertRaisesRegex(ValueError, "nonpositive weighted total"):
            statistics.evaluate_g0(
                cycles, weights, stages, sequence_keys, trials=1
            )

    def test_rejects_invalid_bootstrap_controls_before_allocation(self) -> None:
        cycles, weights, stages, sequence_keys = constant_fixture()
        for trials in (0, -1, 1.5, True):
            with self.subTest(trials=trials):
                with self.assertRaisesRegex(ValueError, "trials"):
                    statistics.sequence_cluster_bootstrap_replicates(
                        cycles,
                        weights,
                        stages,
                        sequence_keys,
                        trials=trials,
                    )
        with self.assertRaisesRegex(ValueError, "seed"):
            statistics.sequence_cluster_bootstrap_replicates(
                cycles, weights, stages, sequence_keys, trials=1, seed=-1
            )


if __name__ == "__main__":
    unittest.main()
