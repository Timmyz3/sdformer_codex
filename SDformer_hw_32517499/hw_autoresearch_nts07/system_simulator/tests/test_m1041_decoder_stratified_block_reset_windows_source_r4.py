#!/usr/bin/env python3
"""Strong value-shape and recursive-key attacks for M1041 r4."""

import copy
import importlib.util
import math
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "scripts/analyze_m1041_decoder_stratified_block_reset_windows_source_r4.py"
SPEC = importlib.util.spec_from_file_location("m1041_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1041StrongTypedEnvelopeTest(unittest.TestCase):
    def test_canonical_hard_stop_and_recursive_public_json_walk(self):
        result = M._high_result()
        self.assertEqual(result["state"], "HARD_STOP_ABOVE_10_PERCENT")
        self.assertIsNone(result["point_estimates"])
        self.assertTrue(M.validate_publication_envelope(result))
        paths = [path for path, _ in M._walk_public_json(result)]
        self.assertIn("bounds.candidate_total_cycles_ci95.0", paths)
        self.assertIn("coverage.strata.0.sample_blocks", paths)

    def test_all_eleven_m1035_escaping_attacks_rejected(self):
        attacks = M._m1035_attacks()
        self.assertEqual(len(attacks), 11)
        for name, attack in attacks:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError,
                                            "semantic point key forbidden"):
                    M.validate_publication_envelope(attack)

    def test_semantic_aliases_rejected_at_arbitrary_depth(self):
        aliases = ("runtimeEstimate", "cycle_sums", "latencies",
                   "THROUGHPUT", "meanValues", "fps")
        for alias in aliases:
            attack = copy.deepcopy(M._high_result())
            attack["bounds"]["candidate_total_cycles_ci95"] = {
                "nested": {"deeper": {alias: 1.0}}}
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(RuntimeError,
                                            "semantic point key forbidden"):
                    M.validate_publication_envelope(attack)

    def test_bound_must_be_flat_length_two(self):
        bad_values = ([1.0], [1.0, 2.0, 3.0],
                      [[1.0], 2.0], {"lo": 1.0, "hi": 2.0})
        for value in bad_values:
            attack = copy.deepcopy(M._high_result())
            attack["bounds"]["candidate_total_cycles_ci95"] = value
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    M.validate_publication_envelope(attack)

    def test_bound_must_be_finite_nonbool_and_ordered(self):
        bad_values = ([False, 2.0], [1.0, math.nan],
                      [1.0, math.inf], [2.0, 1.0])
        for value in bad_values:
            attack = copy.deepcopy(M._high_result())
            attack["bounds"]["baseline_total_cycles_ci95"] = value
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    M.validate_publication_envelope(attack)

    def test_uncertainty_exact_finite_scalar_types(self):
        for value in (False, [2.365], {"value": 2.365}, math.nan, math.inf):
            attack = copy.deepcopy(M._high_result())
            attack["uncertainty"]["t_critical"] = value
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    M.validate_publication_envelope(attack)

    def test_uncertainty_ranges(self):
        attack = copy.deepcopy(M._high_result())
        attack["uncertainty"]["maximum_relative_halfwidth"] = -0.1
        with self.assertRaisesRegex(RuntimeError, "relative uncertainty range"):
            M.validate_publication_envelope(attack)
        attack = copy.deepcopy(M._high_result())
        attack["uncertainty"]["t_critical"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "t-critical range"):
            M.validate_publication_envelope(attack)
        attack = copy.deepcopy(M._high_result())
        attack["uncertainty"]["maximum_relative_halfwidth"] += 0.01
        with self.assertRaisesRegex(RuntimeError, "maximum uncertainty identity"):
            M.validate_publication_envelope(attack)

    def test_coverage_exact_int_types_not_bool_float_or_container(self):
        for key, value in (("population_blocks", True),
                           ("population_blocks", 1000.0),
                           ("sample_blocks", False),
                           ("sample_blocks", 8.0),
                           ("sample_blocks", [8])):
            attack = copy.deepcopy(M._high_result())
            attack["coverage"]["strata"][0][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(RuntimeError):
                    M.validate_publication_envelope(attack)

    def test_coverage_ranges_and_fraction_identity(self):
        for key, value in (("population_blocks", 0),
                           ("sample_blocks", 0),
                           ("sample_blocks", 1001),
                           ("finite_population_fraction", 0.0),
                           ("finite_population_fraction", 1.1),
                           ("finite_population_fraction", 0.5)):
            attack = copy.deepcopy(M._high_result())
            attack["coverage"]["strata"][0][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(RuntimeError):
                    M.validate_publication_envelope(attack)

    def test_coverage_stratum_identity_and_uniqueness(self):
        attack = copy.deepcopy(M._high_result())
        attack["coverage"]["strata"][0]["stratum"] = "UNKNOWN"
        with self.assertRaisesRegex(RuntimeError, "stratum identity"):
            M.validate_publication_envelope(attack)
        attack = copy.deepcopy(M._high_result())
        attack["coverage"]["strata"].append(
            copy.deepcopy(attack["coverage"]["strata"][0]))
        with self.assertRaisesRegex(RuntimeError, "stratum identity"):
            M.validate_publication_envelope(attack)

    def test_nonhard_point_values_are_positive_finite_scalars(self):
        precise = M.estimate_paired_totals([{
            "stratum": "COMPUTE_REGULAR", "population_blocks": 8,
            "candidate_cycles": [10] * 8,
            "baseline_cycles": [20] * 8,
        }])
        self.assertEqual(precise["state"], "CANDIDATE_AT_MOST_5_PERCENT")
        for value in (False, [80.0], math.nan, 0.0, -1.0):
            attack = copy.deepcopy(precise)
            attack["point_estimates"]["candidate_total_cycles"] = value
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError):
                    M.validate_publication_envelope(attack)

    def test_state_status_action_binding(self):
        for key, value in (("status", "POINT_CANDIDATE_FOR_LATER_INDEPENDENT_RELEASE"),
                           ("adaptive_action", "NONE")):
            attack = copy.deepcopy(M._high_result())
            if key == "status":
                attack[key] = value
            else:
                attack["admission"][key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError,
                                            "hard-stop status/action"):
                    M.validate_publication_envelope(attack)

    def test_r3_selector_reset_semantics_remain_pinned(self):
        self.assertIs(M.deterministic_select, M.BASE.deterministic_select)
        self.assertIs(M.block_reset_transactions,
                      M.BASE.block_reset_transactions)
        self.assertIs(M.paired_replay, M.BASE.paired_replay)
        with self.assertRaisesRegex(RuntimeError, "frozen bound"):
            M.deterministic_select([{"block_id": "x"}],
                                   "COMPUTE_REGULAR", 33)
        body = M.M890.synthetic_transactions(448)
        spec = M.WindowSpec("r4-normal", "D0", "COMMIT_TAIL", 1)
        pair = M.paired_replay(body, body, spec)
        self.assertTrue(pair["paired_reset_exact_equal"])

    def test_source_self_test_only(self):
        result = M.self_test()
        self.assertEqual(result["m1035_attack_count"], 11)
        self.assertFalse(result["real_payload_opened"])
        self.assertFalse(result["real_window_execution"])


if __name__ == "__main__":
    unittest.main()
