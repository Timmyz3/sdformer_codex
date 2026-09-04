#!/usr/bin/env python3
"""Recursive publication-redaction attacks for M1034 r3."""

import copy
import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "scripts/analyze_m1034_decoder_stratified_block_reset_windows_source_r3.py"
SPEC = importlib.util.spec_from_file_location("m1034_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def high_result():
    return M.estimate_paired_totals([{
        "stratum": "COMPUTE_REGULAR", "population_blocks": 1000,
        "candidate_cycles": [1, 100, 1, 100, 1, 100, 1, 100],
        "baseline_cycles": [100, 1, 100, 1, 100, 1, 100, 1],
    }])


class M1034RecursiveRedactionTest(unittest.TestCase):
    def test_m1024_original_stratum_means_absent(self):
        result = high_result()
        self.assertIsNone(result["point_estimates"])
        serialized = str(result)
        self.assertNotIn("candidate_mean_cycles", serialized)
        self.assertNotIn("baseline_mean_cycles", serialized)
        self.assertNotIn("candidate_total_cycles_estimate", serialized)
        self.assertNotIn("paired_speedup_estimate", serialized)

    def test_recursive_numeric_walk_has_only_bounds_widths_coverage(self):
        result = high_result()
        allowed = ("bounds.", "uncertainty.", "coverage.strata.")
        paths = [path for path, _ in M._walk_numeric_paths(result)]
        self.assertTrue(paths)
        self.assertTrue(all(path.startswith(allowed) for path in paths))
        forbidden = ("mean", "sum", "estimate", "throughput", "fps")
        self.assertFalse(any(any(word in path for word in forbidden)
                             for path in paths))
        self.assertFalse(any("speedup" in path and "ci95" not in path and
                             "halfwidth" not in path for path in paths))

    def test_nested_cycle_mean_in_coverage_rejected(self):
        result = high_result()
        attack = copy.deepcopy(result)
        attack["coverage"]["strata"][0]["candidate_mean_cycles"] = 50.5
        with self.assertRaisesRegex(RuntimeError, "coverage row schema"):
            M.validate_publication_envelope(attack)

    def test_nested_speedup_and_throughput_containers_rejected(self):
        for key in ("speedup", "FPS", "throughput"):
            attack = copy.deepcopy(high_result())
            attack["identity"][key] = {"point": 2.0}
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, "identity schema"):
                    M.validate_publication_envelope(attack)

    def test_hard_stop_point_container_must_be_null(self):
        attack = copy.deepcopy(high_result())
        attack["point_estimates"] = {
            "candidate_total_cycles": 1.0,
            "baseline_total_cycles": 2.0,
            "paired_speedup": 2.0,
        }
        with self.assertRaisesRegex(RuntimeError, "must be null"):
            M.validate_publication_envelope(attack)

    def test_diagnostic_is_explicitly_not_admitted(self):
        raw = {
            "candidate_total_cycles_estimate": 100.0,
            "candidate_ci95": [94.0, 106.0],
            "baseline_total_cycles_estimate": 120.0,
            "baseline_ci95": [114.0, 126.0],
            "paired_speedup_estimate": 1.2,
            "paired_speedup_ci95": [1.128, 1.272],
            "t_critical": 2.365,
            "metric": "block-reset executable schedule cycles",
            "strata": [{"stratum": "COMPUTE_REGULAR",
                         "population_blocks": 8, "sample_blocks": 4,
                         "finite_population_fraction": 0.5,
                         "candidate_mean_cycles": 25,
                         "baseline_mean_cycles": 30}],
        }
        result = M.publication_projection(raw)
        self.assertEqual(result["state"], "DIAGNOSTIC_5_TO_10_PERCENT")
        self.assertFalse(result["admission"]["point_estimate_admitted"])
        self.assertFalse(result["admission"]["paper_citable"])
        self.assertNotIn("candidate_mean_cycles", str(result))

    def test_r2_selector_and_reset_semantics_remain_pinned(self):
        with self.assertRaisesRegex(RuntimeError, "semantic field forbidden"):
            M.deterministic_select([{
                "block_id": "bad", "compute_count": 1,
                "nested": {"TOTAL_CYCLES": 1}}], "COMPUTE_REGULAR", 1)
        body = M.M890.synthetic_transactions(448)
        spec = M.WindowSpec("r3-normal", "D0", "COMMIT_TAIL", 1)
        pair = M.paired_replay(body, body, spec)
        self.assertTrue(pair["paired_reset_exact_equal"])
        self.assertEqual(pair["candidate_cycles"], pair["baseline_cycles"])

    def test_source_self_test_only(self):
        result = M.self_test()
        self.assertEqual(result["status"],
                         "PASS_M1034_R3_RECURSIVE_REDACTION_SYNTHETIC_SELFTEST")
        self.assertFalse(result["real_payload_opened"])
        self.assertFalse(result["real_window_execution"])


if __name__ == "__main__":
    unittest.main()
