#!/usr/bin/env python3
"""Small synthetic/unit tests for M1014; never opens decoder payloads."""

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "scripts/analyze_m1014_decoder_stratified_block_reset_windows_source.py"
SPEC = importlib.util.spec_from_file_location("m1014_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1014SourceOnlyTest(unittest.TestCase):
    def test_layer_routes_and_d1_rejection(self):
        self.assertEqual(M.frozen_route("D0")["module_index"], 0)
        self.assertEqual(M.frozen_route("D2")["module_index"], 2)
        self.assertEqual(M.frozen_route("D3")["module_index"], 3)
        with self.assertRaisesRegex(RuntimeError, "STRICT_COMMON_CHARGE"):
            M.frozen_route("D1")

    def test_stratum_priority(self):
        self.assertEqual(M.classify_stratum({
            "source_init": True, "commit_count": 1}),
            "SOURCE_INIT_CENSUS")
        self.assertEqual(M.classify_stratum({
            "commit_count": 1, "max_dependency_fan_in": 8}),
            "COMMIT_TAIL")
        self.assertEqual(M.classify_stratum({
            "compute_count": 1, "psum_external_move_count": 1}),
            "DEPENDENCY_STRESS")
        self.assertEqual(M.classify_stratum({"compute_count": 1}),
                         "COMPUTE_REGULAR")

    def test_selection_is_cycle_blind_and_deterministic(self):
        rows = [{"block_id": "b{:02d}".format(i), "compute_count": 1}
                for i in range(12)]
        lhs = M.deterministic_select(rows, "COMPUTE_REGULAR", 8)
        rhs = M.deterministic_select(list(reversed(rows)),
                                     "COMPUTE_REGULAR", 8)
        self.assertEqual([x["block_id"] for x in lhs],
                         [x["block_id"] for x in rhs])
        with self.assertRaisesRegex(RuntimeError, "cycle-derived"):
            M.deterministic_select([
                {"block_id": "bad", "compute_count": 1, "cycles": 4}],
                "COMPUTE_REGULAR", 1)

    def test_window_cap_includes_three_reset_requests(self):
        tx = M.M890.synthetic_transactions(9998)
        spec = M.WindowSpec("too-large", "D0", "COMPUTE_REGULAR", 1)
        with self.assertRaisesRegex(RuntimeError, "exceeds 10K"):
            M.block_reset_transactions(tx, spec, "candidate")

    def test_commit_positive_gate(self):
        body = M.M890.synthetic_transactions(64)
        spec = M.WindowSpec("no-commit", "D0", "COMMIT_TAIL", 1)
        wrapped, _ = M.block_reset_transactions(body, spec, "candidate")
        with self.assertRaisesRegex(RuntimeError, "zero commit"):
            M.exact_replay(wrapped, spec)

    def test_small_synthetic_four_way_miter_and_pairing(self):
        result = M.self_test()
        self.assertEqual(result["status"],
                         "PASS_M1014_SMALL_SYNTHETIC_BLOCK_RESET_SELFTEST")
        self.assertEqual(result["paired_replay"]["candidate_cycles"],
                         result["paired_replay"]["baseline_cycles"])
        self.assertFalse(result["real_payload_opened"])
        self.assertFalse(result["launch_now"])

    def test_fpc_paired_covariance(self):
        result = M.estimate_paired_totals([
            {"stratum": "COMPUTE_REGULAR", "population_blocks": 8,
             "candidate_cycles": [8, 9, 10, 11],
             "baseline_cycles": [16, 18, 20, 22]},
        ])
        self.assertEqual(result["paired_speedup_estimate"], 2.0)
        self.assertEqual(result["paired_speedup_ci95"], [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
