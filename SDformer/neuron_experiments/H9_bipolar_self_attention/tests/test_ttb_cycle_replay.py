from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ENTRYPOINTS = Path(__file__).resolve().parents[1] / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))


class TTBCycleReplayTest(unittest.TestCase):
    def test_empty_route_runs_at_metadata_rate(self):
        from replay_ttb_dual_path_cycles import simulate_dual_record

        result = simulate_dual_record(
            np.zeros((16,), dtype=np.int16),
            capacity=256,
            kappa=8,
            sparse_lanes=8,
            dense_lanes=32,
            fifo_depth=4,
        )
        self.assertEqual(result["cycles"], 16)
        self.assertEqual(result["input_stalls"], 0)

    def test_finite_fifo_backpressure_and_dual_overlap(self):
        from replay_ttb_dual_path_cycles import analytical_record, simulate_dual_record

        counts = np.array([1, 256, 1, 256, 1, 256, 1, 256], dtype=np.int16)
        analytical = analytical_record(
            counts,
            capacity=256,
            kappa=8,
            sparse_lanes=2,
            dense_lanes=32,
        )
        replay = simulate_dual_record(
            counts,
            capacity=256,
            kappa=8,
            sparse_lanes=2,
            dense_lanes=32,
            fifo_depth=1,
        )
        self.assertEqual(analytical["sparse_jobs"], 4)
        self.assertEqual(analytical["dense_jobs"], 4)
        self.assertGreater(analytical["e0_traffic_bits"], analytical["dual_bitmap_traffic_bits"])
        self.assertGreater(analytical["e0_traffic_bits"], analytical["dual_index_traffic_bits"])
        self.assertGreaterEqual(analytical["e0_transactions64"], analytical["dual_bitmap_transactions64"])
        self.assertGreaterEqual(analytical["e0_transactions64"], analytical["dual_index_transactions64"])
        self.assertEqual(analytical["rows"], 1)
        self.assertGreaterEqual(analytical["dual_lower_bound_b16"], analytical["dual_lower_bound_b1"])
        self.assertGreaterEqual(replay["cycles"], analytical["dual_lower_bound"])
        self.assertLess(replay["cycles"], analytical["e0_work"])
        self.assertGreater(replay["input_stalls"], 0)

    def test_replay_aggregation_sums_cycles_but_takes_fifo_peak(self):
        from replay_ttb_dual_path_cycles import aggregate_replay

        result = aggregate_replay([
            {"cycles": 10, "max_sparse_fifo": 2, "max_dense_fifo": 4},
            {"cycles": 20, "max_sparse_fifo": 7, "max_dense_fifo": 3},
        ])
        self.assertEqual(result["cycles"], 30)
        self.assertEqual(result["max_sparse_fifo"], 7)
        self.assertEqual(result["max_dense_fifo"], 4)

    def test_sweep_and_finite_replay_emit_cycle_and_traffic_fields(self):
        from replay_ttb_dual_path_cycles import analytical_sweep, finite_replay

        prepared = [
            (np.array([0, 1, 4, 32, 0, 2], dtype=np.int16), 32, 0),
            (np.array([0, 8, 16, 32], dtype=np.int16), 32, 1),
        ]
        sweep = analytical_sweep(prepared, "delta1", 32)
        self.assertEqual(len(sweep), 15)
        self.assertEqual(set(sweep[0]["backend_lower_bound_reduction"]), {"1", "4", "8", "16"})
        candidate = min(sweep, key=lambda row: row["dual_lower_bound"])
        replay = finite_replay(prepared, "delta1", candidate, fifo_depth=4)
        self.assertIn("bitmap_traffic_reduction_vs_e0", replay)
        self.assertIn("index_traffic_reduction_vs_e0", replay)
        self.assertIn("bitmap_transaction_reduction_vs_e0", replay)
        self.assertIn("index_transaction_reduction_vs_e0", replay)
        self.assertEqual([row["stage"] for row in replay["by_stage"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
