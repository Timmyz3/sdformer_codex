#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-EDA author tests for the M1502 C2 successor source."""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1502_c2_source_chain_successor_source.py"
SPEC = importlib.util.spec_from_file_location("m1502_source", CHECKER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1502_MUTATED"
    if type(value) is list:
        return list(value) + ["M1502_MUTATED"]
    if type(value) is dict:
        result = copy.deepcopy(value)
        result["m1502_mutated"] = True
        return result
    raise TypeError(type(value))


class Tests(unittest.TestCase):
    def test_01_predecessor_exact(self):
        value = M.check_predecessor()
        self.assertEqual(value["phase"], "SOURCE_CHAIN")
        self.assertEqual(value["error"], "AttributeError")
        self.assertFalse(value["attempt_consumed"])
        self.assertEqual((value["vcs_compiles"], value["simv_runs"],
                          value["saif_files"], value["ptpx_runs"]),
                         (0, 0, 0, 0))

    def test_02_real_corrected_callpath(self):
        value = M.check_corrected_callpath()
        self.assertFalse(value["attribute_error"])
        self.assertEqual(value["terminal"], "M1502_FUTURE_AUTHORITY_ONLY")

    def test_03_execution_text(self):
        M.check_execution_text(M.RUNNER.read_text())

    def test_04_invalid_call_injection_rejected(self):
        text = M.RUNNER.read_text().replace(
            "    verify_new_authority()",
            "    EXEC.verify_predecessor_failure()\n    verify_new_authority()", 1)
        with self.assertRaisesRegex(RuntimeError, "invalid predecessor"):
            M.check_execution_text(text)

    def test_05_debug_flag_mutation_rejected(self):
        original = M.R.COMPILE_PREFIX
        M.R.COMPILE_PREFIX = [item for item in original
                              if item != "-debug_access+r"]
        try:
            with self.assertRaisesRegex(RuntimeError, "compile prefix"):
                M.check_execution_text(M.RUNNER.read_text())
        finally:
            M.R.COMPILE_PREFIX = original

    def test_06_lca_flag_mutation_rejected(self):
        original = M.R.COMPILE_PREFIX
        M.R.COMPILE_PREFIX = [item for item in original if item != "-lca"]
        try:
            with self.assertRaisesRegex(RuntimeError, "compile prefix"):
                M.check_execution_text(M.RUNNER.read_text())
        finally:
            M.R.COMPILE_PREFIX = original

    def test_07_axis_mutation_rejected(self):
        text = M.RUNNER.read_text().replace(
            'for axis in ("k8", "k1x8"):', 'for axis in ("k8",):', 1)
        with self.assertRaisesRegex(RuntimeError, "axis loops"):
            M.check_execution_text(text)

    def test_08_case_mutation_rejected(self):
        text = M.RUNNER.read_text().replace(
            "for case in range(5):", "for case in range(4):", 1)
        with self.assertRaisesRegex(RuntimeError, "case loops"):
            M.check_execution_text(text)

    def test_09_counter_mutation_rejected(self):
        text = M.RUNNER.read_text().replace(
            'state["saif_files"] += 1', "pass", 1)
        with self.assertRaisesRegex(RuntimeError, "counter site"):
            M.check_execution_text(text)

    def test_10_contract_exact(self):
        self.assertEqual(M.check_contract(), M.expected_contract())

    def test_11_top_level_contract_mutations_rejected(self):
        base = M.expected_contract()
        for key, value in base.items():
            candidate = copy.deepcopy(base)
            candidate[key] = changed(value)
            with self.subTest(key=key), mock.patch.object(
                    M, "strict_json", return_value=candidate):
                with self.assertRaisesRegex(RuntimeError, "exact-set/value"):
                    M.check_contract()

    def test_12_section_contract_mutations_rejected(self):
        base = M.expected_contract()
        for section in ("identity", "predecessor_failure", "sole_repair",
                        "corrected_callpath_test", "preserved_execution",
                        "future_authority", "author_execution",
                        "claim_boundary", "protected"):
            for key, value in base[section].items():
                candidate = copy.deepcopy(base)
                candidate[section][key] = changed(value)
                with self.subTest(section=section, key=key), mock.patch.object(
                        M, "strict_json", return_value=candidate):
                    with self.assertRaisesRegex(RuntimeError,
                                                "exact-set/value"):
                        M.check_contract()

    def test_13_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x.json"
            path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON"):
                M.strict_json(path)

    def test_14_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x.json"
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                M.strict_json(path)

    def test_15_namespace_residue_rejected(self):
        original = M.NEW_NAMESPACES
        with tempfile.TemporaryDirectory(dir=M.HW / "results") as temporary:
            residue = Path(temporary) / "x"
            residue.write_text("x")
            M.NEW_NAMESPACES = {"attempt": residue.relative_to(M.HW).as_posix()}
            try:
                with mock.patch.object(M, "check_predecessor", return_value={}), \
                        mock.patch.object(M, "check_corrected_callpath",
                                          return_value={}), \
                        mock.patch.object(M, "check_execution_text"), \
                        mock.patch.object(M, "check_contract",
                                          return_value=M.expected_contract()):
                    with self.assertRaisesRegex(RuntimeError,
                                                "namespace residue"):
                        M.check_source(False)
            finally:
                M.NEW_NAMESPACES = original

    def test_16_future_residue_rejected(self):
        original = M.FUTURE
        contract = M.expected_contract()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authority"
            path.mkdir()
            M.FUTURE = (path,)
            try:
                with mock.patch.object(M, "check_predecessor", return_value={}), \
                        mock.patch.object(M, "check_corrected_callpath",
                                          return_value={}), \
                        mock.patch.object(M, "check_execution_text"), \
                        mock.patch.object(M, "check_contract",
                                          return_value=contract):
                    with self.assertRaisesRegex(RuntimeError,
                                                "future authority"):
                        M.check_source(True)
            finally:
                M.FUTURE = original

    def test_17_no_claim_promotion_or_execution(self):
        self.assertTrue(all(value is False for value in M.CLAIMS.values()))
        expected = M.expected_contract()
        self.assertFalse(expected["future_authority"]["launch_authorized"])
        self.assertFalse(expected["author_execution"]["eda"])
        self.assertFalse(expected["author_execution"]["attempt_consumed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
