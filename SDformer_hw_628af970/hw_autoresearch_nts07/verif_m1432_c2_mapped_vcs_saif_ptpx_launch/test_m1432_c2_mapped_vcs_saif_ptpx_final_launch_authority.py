#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-EDA source tests for the exact M1432 release authority."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "static_check_m1432_c2_mapped_vcs_saif_ptpx_final_launch_authority.py"
SPEC = importlib.util.spec_from_file_location("m1432_release", CHECKER)
M = importlib.util.module_from_spec(SPEC); assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def changed(value):
    if type(value) is bool: return not value
    if type(value) is int: return value + 1
    if type(value) is float: return value + 0.25
    if type(value) is str: return "M1432_MUTATED"
    if type(value) is list: return list(value) + ["M1432_MUTATED"]
    if type(value) is dict:
        output = dict(value); output["m1432_extra"] = True; return output
    raise TypeError(type(value))


class Tests(unittest.TestCase):
    def test_01_exact_contract_passes(self):
        self.assertEqual(M.validate_contract(skip_author=True), M.expected_contract())

    def test_02_upstream_chain_and_artifacts_pass(self):
        result = M.verify_upstream()
        self.assertTrue(result["m1361_source"])
        self.assertTrue(result["m1362_zero_false_negative"])
        self.assertEqual(result["mapped_axes"], 2)

    def test_02b_additive_executor_is_reachable(self):
        result = M.validate_runner_source()
        self.assertEqual(result["stages_reachable"], ["mapped_vcs", "production_saif", "ptpx"])
        self.assertIs(result["two_collision_gates_before_license"], True)
        self.assertIs(result["frozen_m1344_future_paths"], False)

    def test_03_every_top_level_value_is_exact(self):
        base = M.expected_contract()
        for key in base:
            candidate = copy.deepcopy(base); candidate[key] = changed(base[key])
            with self.subTest(key=key), mock.patch.object(M, "strict_json", return_value=candidate):
                with self.assertRaisesRegex(AssertionError, "exact-set/value"):
                    M.validate_contract(skip_author=True)

    def test_04_every_required_object_leaf_is_exact(self):
        base = M.expected_contract()
        objects = ("m1361_source", "m1362_blind", "executor_reachability",
                   "one_shot", "execution_budget",
                   "resource_fail_close", "measurement_gates", "receipt_contract",
                   "final_gate", "authorization", "claim_boundary", "protected_files")
        for name in objects:
            for key, value in base[name].items():
                candidate = copy.deepcopy(base); candidate[name][key] = changed(value)
                with self.subTest(obj=name, key=key), mock.patch.object(
                        M, "strict_json", return_value=candidate):
                    with self.assertRaisesRegex(AssertionError, "exact-set/value"):
                        M.validate_contract(skip_author=True)

    def test_05_duplicate_json_key_rejected(self):
        text = M.CONTRACT.read_text(encoding="utf-8").replace(
            '"status":', '"status":"DUPLICATE","status":', 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"; path.write_text(text)
            with self.assertRaisesRegex(AssertionError, "duplicate JSON key"):
                M.strict_json(path)

    def test_06_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"; path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(AssertionError, "nonfinite"):
                M.strict_json(path)

    def test_07_all_fresh_namespaces_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = {key: Path(temporary) / key for key in M.NAMESPACES}
            self.assertEqual(M.validate_absence(paths)["fresh_namespaces"], 4)

    def test_08_any_namespace_residue_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); paths = {key: root / key for key in M.NAMESPACES}
            for key in paths:
                paths[key].write_text("residue")
                with self.subTest(key=key), self.assertRaisesRegex(AssertionError, "residue"):
                    M.validate_absence(paths)
                paths[key].unlink()

    def test_09_future_hammer_residue_rejected(self):
        original = M.FUTURE_HAMMER
        with tempfile.TemporaryDirectory() as temporary:
            M.FUTURE_HAMMER = Path(temporary) / "m1440"; M.FUTURE_HAMMER.mkdir()
            try:
                paths = {key: Path(temporary) / key for key in M.NAMESPACES}
                with self.assertRaisesRegex(AssertionError, "M1440"):
                    M.validate_absence(paths)
            finally:
                M.FUTURE_HAMMER = original

    def test_10_one_campaign_and_no_retry(self):
        value = M.expected_contract()
        self.assertEqual(value["one_shot"]["campaigns"], 1)
        self.assertIs(value["one_shot"]["automatic_retry"], False)
        self.assertEqual(value["execution_budget"], {
            "ordered_stages": ["mapped_vcs", "production_saif", "ptpx"],
            "mapped_vcs_compiles": 2, "simv_runs": 10,
            "production_saif_files": 10, "ptpx_runs": 10,
            "retry_attempts": 0, "partial_axis_publication": False,
            "ptpx_only_after_all_mapped_correctness_and_saif_gates": True})

    def test_11_both_collision_gates_precede_any_license_or_tool(self):
        gate = M.expected_contract()["resource_fail_close"]
        self.assertIs(gate["collision_gate_1_before_any_license_or_tool"], True)
        self.assertIs(gate["collision_gate_2_under_lease_before_any_license_or_tool"], True)
        self.assertEqual(gate["same_uid_blocked_processes"], M.BLOCKED)

    def test_12_receipt_set_and_values_are_exact(self):
        receipt = M.expected_contract()["receipt_contract"]
        self.assertEqual(receipt["identity_sha_keys_each"], M.RECEIPT_KEYS)
        self.assertEqual(receipt["attempt"]["budget"], {
            "vcs_compiles": 2, "simv_runs": 10, "saif_files": 10, "ptpx_runs": 10})
        self.assertIs(receipt["failure"]["automatic_retry"], False)
        self.assertIs(receipt["success"]["claim_boundary_exact"], True)

    def test_13_authoring_executes_no_tool(self):
        combined = CHECKER.read_text(encoding="utf-8") + Path(__file__).read_text(encoding="utf-8")
        self.assertNotIn("sub" + "process", combined)
        self.assertNotIn("os" + ".system", combined)
        self.assertNotIn("Po" + "pen(", combined)
        self.assertNotIn("lm" + "stat", combined)

    def test_14_all_claims_false_and_launch_now_false(self):
        value = M.expected_contract()
        self.assertEqual(value["claim_boundary"], M.CLAIMS)
        self.assertTrue(all(item is False for item in value["claim_boundary"].values()))
        self.assertIs(value["authorization"]["launch_now"], False)
        self.assertIs(value["final_gate"]["actual_launch_ready"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
