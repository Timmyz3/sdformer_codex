#!/usr/bin/env python3
"""Author CPU-only tests for M722 LB-FUSE."""

import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "system_simulator/scripts/analyze_m722_lb_fuse_decoder_cpu_fastkill.py"
SPEC = importlib.util.spec_from_file_location("m722", SCRIPT)
M722 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M722)


class M722Tests(unittest.TestCase):
    def test_source_and_destination_contributors_conserve(self):
        bits = np.asarray([
            [[1, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 1]],
            [[1, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [1, 1, 1]],
            [[1, 0, 0], [0, 1, 0]],
        ], dtype=np.uint8)
        counts = M722.group_counts(bits, 1)
        self.assertGreater(counts["contributors"], 0)
        self.assertGreaterEqual(counts["lb_direct_groups"],
                                counts["osg_groups"])

    def test_d3_fair_a1_stripes_without_spill(self):
        plan = M722.a1_storage_plan(M722.MODULES[3])
        self.assertEqual(plan["stripe_width"], 256)
        self.assertEqual(plan["stripe_count"], 2)
        self.assertEqual(plan["summed_source_columns"], 161)
        self.assertEqual(plan["source_column_overlap"], 1)
        self.assertEqual(plan["offchip_psum_spill_bytes"], 0)
        self.assertLessEqual(plan["total_bytes"], M722.BUDGET_BYTES)

    def test_three_row_capacities(self):
        self.assertEqual(M722.line_capacity(M722.MODULES[0], 3), 34560)
        self.assertEqual(M722.line_capacity(M722.MODULES[1], 3), 69120)
        self.assertEqual(M722.line_capacity(M722.MODULES[2], 3), 138240)
        self.assertEqual(M722.line_capacity(M722.MODULES[3], 3), 276480)
        self.assertEqual(M722.line_capacity(M722.MODULES[3], 2), 184320)
        self.assertEqual(M722.line_capacity(M722.MODULES[3], 3, 48), 138240)

    def test_numeric_miter_exact(self):
        rng = np.random.default_rng(722)
        bits = rng.integers(0, 2, size=(2, 5, 2, 3), dtype=np.uint8)
        weight = rng.integers(-7, 8, size=(5, 4, 3, 3), dtype=np.int8)
        replay = M722.numeric_replay(bits, weight)
        self.assertEqual(replay["mismatches"], 0)
        self.assertTrue(replay["integer_exact"])
        self.assertTrue(all(value >= 0 for value in
                            replay["order_independent_abs_prefix_bounds"]))

    def test_same_port_cycle_row_has_no_hidden_parallel_rmw(self):
        bits = np.zeros((16, 2, 3), dtype=np.uint8)
        bits[0:8] = 1
        spec = ("T", 16, 96, 2, 3, 4, 6, 1)
        counts = M722.group_counts(bits, 1)
        row = M722.cycle_row(bits, spec, counts,
                             M722.a1_storage_plan(spec), "acc24_full96")
        self.assertEqual(row["port_model"]["lb_port_conflict_events"], 0)
        self.assertTrue(row["port_model"]
                        ["serialized_group_service_covers_all_rmw"])
        self.assertEqual(row["traffic"]["dense_commit_bytes_a1"],
                         row["traffic"]["dense_commit_bytes_lb"])


if __name__ == "__main__":
    unittest.main()
