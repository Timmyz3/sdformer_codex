#!/usr/bin/env python3

import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("compare_local5_h67_mechanisms.py")
SPEC = importlib.util.spec_from_file_location("dual_profile", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DualProfileDecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with MODULE.DEFAULT_LOCAL5.open("r", encoding="utf-8") as handle:
            cls.local5 = MODULE.local5_metrics(json.load(handle))
        with MODULE.DEFAULT_H67.open("r", encoding="utf-8") as handle:
            cls.h67 = MODULE.h67_metrics(json.load(handle))

    def test_local5_work_accounting(self) -> None:
        self.assertEqual(self.local5["channels"], 32)
        self.assertEqual(
            self.local5["valid_edges"],
            self.local5["self_edges"] + self.local5["directional_edges"],
        )
        self.assertLess(
            self.local5["selected_delta_lanes"],
            self.local5["direct_all_score_lane_work"],
        )
        self.assertAlmostEqual(
            self.local5["selected_lane_reduction_ideal"],
            0.7657243553099593,
        )

    def test_compactor_coverage_is_monotonic(self) -> None:
        self.assertLessEqual(
            self.local5["changed_edge_coverage_le2"],
            self.local5["changed_edge_coverage_le4"],
        )
        self.assertLessEqual(
            self.local5["changed_edge_coverage_le4"],
            self.local5["changed_edge_coverage_le8"],
        )
        self.assertLessEqual(
            self.h67["changed_pair_coverage_le2"],
            self.h67["changed_pair_coverage_le4"],
        )
        self.assertLessEqual(
            self.h67["changed_pair_coverage_le4"],
            self.h67["changed_pair_coverage_le8"],
        )

    def test_h67_profile_consistency(self) -> None:
        self.assertAlmostEqual(
            self.h67["full_t2_compare_reduction"],
            (1.0 - self.h67["qk_update_lane_density"]) / 2.0,
        )
        self.assertGreater(self.h67["ttb4_empty_ratio"], 0.60)
        self.assertGreater(
            self.h67["final_gate_term_count_reduction"], 0.82
        )


if __name__ == "__main__":
    unittest.main()
