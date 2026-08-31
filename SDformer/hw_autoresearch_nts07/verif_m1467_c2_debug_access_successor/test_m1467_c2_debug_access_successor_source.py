#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-EDA mutation tests for the M1467 additive C2 successor."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1467_c2_debug_access_successor_source.py"
SPEC = importlib.util.spec_from_file_location("m1467_source", CHECKER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1467_MUTATED"
    if type(value) is list:
        return list(value) + ["M1467_MUTATED"]
    if type(value) is dict:
        output = dict(value)
        output["m1467_mutated"] = True
        return output
    raise TypeError(type(value))


class Tests(unittest.TestCase):
    def test_01_predecessor_failure_exact(self):
        self.assertEqual(M.check_predecessor_failure(), {
            "phase": "SIM_k8_0", "vcs_compiles": 1, "simv_runs": 1,
            "saif_files": 0, "ptpx_runs": 0, "attempt_consumed": True,
            "private_build_read": False, "automatic_retry": False})

    def test_02_runner_has_one_minimal_delta(self):
        result = M.check_runner_static()
        self.assertEqual(result["sole_delta"], "vcs_compile_add_debug_access_r")
        self.assertEqual(result["vcs_compiles"], 2)
        self.assertEqual(result["simv_runs"], 10)
        self.assertEqual(result["saif_files"], 10)
        self.assertEqual(result["ptpx_runs"], 10)
        self.assertFalse(result["partial_axis_citable"])

    def test_03_exact_contract_passes(self):
        self.assertEqual(M.check_contract(), M.expected_contract())

    def test_04_all_top_level_mutations_rejected(self):
        base = M.expected_contract()
        for key, value in base.items():
            candidate = copy.deepcopy(base)
            candidate[key] = changed(value)
            with self.subTest(key=key), mock.patch.object(
                    M, "strict_json", return_value=candidate):
                with self.assertRaisesRegex(RuntimeError, "exact-set/value"):
                    M.check_contract()

    def test_05_all_object_leaf_mutations_rejected(self):
        base = M.expected_contract()
        for section in ("identity", "predecessor_failure", "root_cause",
                        "sole_repair", "preserved_execution", "future_authority",
                        "author_execution", "claim_boundary", "protected"):
            for key, value in base[section].items():
                candidate = copy.deepcopy(base)
                candidate[section][key] = changed(value)
                with self.subTest(section=section, key=key), mock.patch.object(
                        M, "strict_json", return_value=candidate):
                    with self.assertRaisesRegex(RuntimeError, "exact-set/value"):
                        M.check_contract()

    def test_06_duplicate_json_rejected(self):
        text = M.CONTRACT.read_text().replace(
            '"status":', '"status":"DUPLICATE","status":', 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            path.write_text(text)
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
                M.strict_json(path)

    def test_07_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                M.strict_json(path)

    def test_08_debug_flag_deletion_rejected(self):
        text = M.RUNNER.read_text().replace('"-debug_access+r",', '', 1)
        with self.assertRaisesRegex(RuntimeError, "debug flag"):
            M.check_minimal_delta_text(text)

    def test_09_future_residue_rejected(self):
        original = M.M1468
        contract = M.expected_contract()
        with tempfile.TemporaryDirectory() as temporary:
            M.M1468 = Path(temporary) / "m1468"
            M.M1468.mkdir()
            try:
                with mock.patch.object(M, "check_contract", return_value=contract), \
                        mock.patch.object(M, "check_predecessor_failure", return_value={}), \
                        mock.patch.object(M, "check_runner_static", return_value={}):
                    with self.assertRaisesRegex(RuntimeError, "future authority"):
                        M.check_source(require_future_absent=True)
            finally:
                M.M1468 = original

    def test_10_new_namespace_residue_rejected(self):
        original = M.NEW_NAMESPACES
        with tempfile.TemporaryDirectory(dir=M.HW / "results") as temporary:
            residue = Path(temporary) / "attempt"
            residue.write_text("x")
            M.NEW_NAMESPACES = {"attempt": residue.relative_to(M.HW).as_posix()}
            try:
                with mock.patch.object(M, "check_contract", return_value=M.expected_contract()), \
                        mock.patch.object(M, "check_predecessor_failure", return_value={}), \
                        mock.patch.object(M, "check_runner_static", return_value={}):
                    with self.assertRaisesRegex(RuntimeError, "namespace residue"):
                        M.check_source(require_future_absent=False)
            finally:
                M.NEW_NAMESPACES = original

    def test_11_authoring_paths_invoke_no_tool(self):
        combined = CHECKER.read_text() + Path(__file__).read_text()
        self.assertNotIn("sub" + "process", combined)
        self.assertNotIn("os" + ".system", combined)
        self.assertNotIn("Po" + "pen(", combined)
        self.assertNotIn("lm" + "stat", combined)

    def test_12_no_claim_promoted(self):
        self.assertTrue(all(value is False for value in M.CLAIMS.values()))
        contract = M.expected_contract()
        self.assertFalse(contract["future_authority"]["launch_authorized"])
        self.assertFalse(contract["author_execution"]["eda"])
        self.assertFalse(contract["author_execution"]["private_build_read"])

    def test_13_preserved_fair_campaign(self):
        campaign = M.expected_contract()["preserved_execution"]
        self.assertEqual(campaign["axes"], ["k8", "k1x8"])
        self.assertEqual(campaign["cases"], [0, 1, 2, 3, 4])
        self.assertEqual(campaign["vcs_compiles"], 2)
        self.assertEqual(campaign["simv_runs"], 10)
        self.assertEqual(campaign["production_saif_files"], 10)
        self.assertEqual(campaign["ptpx_runs"], 10)
        self.assertFalse(campaign["partial_axis_publication"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
