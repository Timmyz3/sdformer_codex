#!/usr/bin/env python3
"""Author regression for the source-only M1645 actual-prefix runner."""
from __future__ import print_function

import ast
import importlib.util
import inspect
import json
from pathlib import Path
import unittest


TESTS = Path(__file__).resolve().parent
SOURCE = (TESTS.parent / "scripts" /
          "build_m1645_decoder_compact_actual_prefix_runner_source.py")


def load_source():
    spec = importlib.util.spec_from_file_location("m1645_author_test", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M1645Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_source()
        cls.result = cls.m.static_self_test()

    def rejects(self, function):
        with self.assertRaises(self.m.M1645Error):
            function()

    def test_01_status_population_and_claim_boundary(self):
        row = self.m.describe()
        self.assertEqual(row["status"],
            "SOURCE_ONLY__ACTUAL_PREFIX_RUNNER_AUTHORED__NO_PAYLOAD_NO_EXECUTION")
        population = row["fixed_population"]
        self.assertEqual(population["decoder_stage"], "D0")
        self.assertEqual(population["call_ordinal"], 0)
        self.assertEqual(population["module_ordinal"], 0)
        self.assertEqual(population["timestep"], 0)
        self.assertEqual(population["destinations"], list(range(42)))
        self.assertEqual(population["output_blocks"], [0, 1, 2, 3])
        self.assertEqual(population["configuration_order"], list(self.m.CONFIGS))
        for field in ("actual_payload", "actual_execution", "l2_execution",
                      "l3", "full_decoder", "production", "cycles",
                      "traffic", "energy", "speedup", "system_speedup",
                      "gpu", "rtl", "eda", "paper_result"):
            self.assertFalse(row["authorization"][field])

    def test_02_exact_sources_and_review_are_bound(self):
        row = self.m.validate_authorities()
        self.assertEqual(row["m1539_source_sha256"],
                         self.m.M1539_SOURCE_SHA256)
        self.assertEqual(row["m1610_source_sha256"],
                         self.m.M1610_SOURCE_SHA256)
        self.assertEqual(row["m1638_source_sha256"],
                         self.m.M1638_SOURCE_SHA256)
        self.assertEqual(row["m1639"]["review_sha256"],
                         self.m.M1639_REVIEW_SHA256)
        self.assertFalse(row["actual_payload"])
        self.assertFalse(row["actual_execution"])

    def test_03_static_three_session_bundle_is_exact_and_nonpaper(self):
        row = self.result
        self.assertEqual(row["status"],
            "PASS_M1645_ACTUAL_PREFIX_RUNNER_SOURCE_STATIC_ONLY")
        self.assertEqual(row["configurations"], list(self.m.CONFIGS))
        self.assertEqual(row["distinct_sessions"], 3)
        self.assertEqual(row["attacks_rejected"], 3)
        self.assertFalse(row["actual_payload"])
        self.assertFalse(row["actual_execution"])
        self.assertFalse(row["cycles_admitted"])
        self.assertFalse(row["bytes_admitted"])
        self.assertFalse(row["paper_result"])

    def test_04_every_synthetic_config_has_complete_prefix_metrics(self):
        metrics = self.result["metrics"]
        self.assertEqual([row["configuration"] for row in metrics],
                         list(self.m.CONFIGS))
        for row in metrics:
            self.assertGreater(row["total_cycles"], 0)
            self.assertGreater(row["request_count"], 42 * 4)
            self.assertEqual(row["kind_counts"]["commit"], 42 * 4)
            self.assertTrue(row["independent_hammer_pending"])
            self.assertFalse(row["paper_result"])

    def test_05_rss_measures_baseline_current_hwm_and_both_limits(self):
        rss = self.result["rss"]
        self.assertEqual(rss["absolute_limit_kib"], 2 * 1024 * 1024)
        self.assertEqual(rss["increment_limit_kib"], 512 * 1024)
        self.assertGreaterEqual(rss["gate_calls"], 1 + 3 * 42)
        self.assertLess(rss["max_current_rss_kib"], rss["absolute_limit_kib"])
        self.assertLess(rss["max_hwm_rss_kib"], rss["absolute_limit_kib"])
        self.assertLess(rss["max_hwm_rss_kib"] -
                        rss["baseline_current_rss_kib"],
                        rss["increment_limit_kib"])

    def test_06_d0_coordinate_encoder_repairs_module3_adapter_mismatch(self):
        config = self.m.CONFIGS[0]
        commit = self.m.R.request(config + ":commit:7:2", config, "commit",
                                  [0], [0], 384)
        coordinate = self.m.actual_coordinate(config, commit, 11, 7, 2)
        self.assertEqual(coordinate[1], 0)
        self.assertEqual(coordinate[2], 0)
        self.assertEqual(coordinate[4:6], (7, 2))
        self.assertEqual(coordinate[3], self.m.C.FLAG_COMMIT)
        legacy = self.m.C.parse_synthetic_identifier(config, commit["id"], 11)
        self.assertEqual(legacy[1], 3)
        self.assertNotEqual(legacy[1], coordinate[1])

    def test_07_coordinate_and_product_attacks_fail_closed(self):
        config = self.m.CONFIGS[0]
        wrong = self.m.R.request(config + ":commit:0:0", config, "commit",
                                 [0], [0], 384)
        self.rejects(lambda: self.m.actual_coordinate(config, wrong, 0, 1, 0))
        self.rejects(lambda: self.m.actual_coordinate(
            self.m.FORBIDDEN_CONFIG, {}, 0, 0, 0))

    def test_08_public_release_and_cli_do_not_execute_private_runner(self):
        self.rejects(self.m.actual_prefix_release)
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "_run_bound_actual_prefix":
                    calls.append(node.lineno)
        self.assertEqual(calls, [])
        source = inspect.getsource(self.m.main)
        self.assertNotIn("actual_prefix_release(", source)
        self.assertNotIn("_run_bound_actual_prefix(", source)

    def test_09_future_result_is_prefix_only_and_pending_hammer(self):
        row = self.m.describe()["future_result"]
        self.assertTrue(row["one_shot_prefix_only"])
        self.assertTrue(row["cycles_pending_independent_hammer"])
        self.assertTrue(row["bytes_pending_independent_hammer"])
        self.assertFalse(row["product_capture"])
        self.assertFalse(row["l3"])
        self.assertFalse(row["full_decoder"])
        self.assertFalse(row["production"])
        self.assertFalse(row["paper_result"])

    def test_10_describe_is_json_stable_and_future_gates_absent(self):
        first = json.dumps(self.m.describe(), sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
        second = json.dumps(self.m.describe(), sort_keys=True,
                            separators=(",", ":"), allow_nan=False)
        self.assertEqual(first, second)
        self.assertFalse(self.m.FUTURE_REVIEW.exists())
        self.assertFalse(self.m.FUTURE_RELEASE.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
