#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Lightweight no-EDA tests for M1493 source."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1493_c2_lca_successor_source.py"
SPEC = importlib.util.spec_from_file_location("m1493_source", CHECKER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1493_MUTATED"
    if type(value) is list:
        return list(value) + ["M1493_MUTATED"]
    if type(value) is dict:
        result = dict(value)
        result["m1493_mutated"] = True
        return result
    raise TypeError(type(value))


class Tests(unittest.TestCase):
    def test_01_predecessor_exact(self):
        value = M.check_predecessor()
        self.assertEqual(value["required_option"], "-lca")
        self.assertEqual(value["vcs_compiles"], 1)
        self.assertEqual(value["simv_runs"], 1)
        self.assertFalse(value["automatic_retry"])

    def test_02_minimal_delta(self):
        M.check_minimal_delta_text(M.RUNNER.read_text())

    def test_03_lca_deletion_rejected(self):
        text = M.RUNNER.read_text().replace('"-lca",', "", 1)
        with self.assertRaisesRegex(RuntimeError, "lca cardinality"):
            M.check_minimal_delta_text(text)

    def test_04_debug_deletion_rejected(self):
        text = M.RUNNER.read_text().replace('"-debug_access+r",', "", 1)
        with self.assertRaisesRegex(RuntimeError, "debug_access cardinality"):
            M.check_minimal_delta_text(text)

    def test_05_lca_duplication_rejected(self):
        text = M.RUNNER.read_text().replace('"-lca",', '"-lca", "-lca",', 1)
        with self.assertRaisesRegex(RuntimeError, "lca cardinality"):
            M.check_minimal_delta_text(text)

    def test_06_contract_exact(self):
        self.assertEqual(M.check_contract(), M.expected_contract())

    def test_07_top_level_mutations_rejected(self):
        base = M.expected_contract()
        for key, value in base.items():
            candidate = copy.deepcopy(base)
            candidate[key] = changed(value)
            with self.subTest(key=key), mock.patch.object(M, "strict_json",
                                                          return_value=candidate):
                with self.assertRaisesRegex(RuntimeError, "exact-set/value"):
                    M.check_contract()

    def test_08_section_leaf_mutations_rejected(self):
        base = M.expected_contract()
        for section in ("identity", "predecessor_failure", "sole_repair",
                        "preserved_execution", "future_authority",
                        "author_execution", "claim_boundary", "protected"):
            for key, value in base[section].items():
                candidate = copy.deepcopy(base)
                candidate[section][key] = changed(value)
                with self.subTest(section=section, key=key), mock.patch.object(
                        M, "strict_json", return_value=candidate):
                    with self.assertRaisesRegex(RuntimeError, "exact-set/value"):
                        M.check_contract()

    def test_09_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x.json"
            path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON"):
                M.strict_json(path)

    def test_10_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x.json"
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                M.strict_json(path)

    def test_11_namespace_residue_rejected(self):
        original = M.NEW_NAMESPACES
        with tempfile.TemporaryDirectory(dir=M.HW / "results") as temporary:
            residue = Path(temporary) / "x"
            residue.write_text("x")
            M.NEW_NAMESPACES = {"attempt": residue.relative_to(M.HW).as_posix()}
            try:
                with mock.patch.object(M, "check_predecessor", return_value={}), \
                        mock.patch.object(M, "check_minimal_delta_text"), \
                        mock.patch.object(M, "check_contract",
                                          return_value=M.expected_contract()):
                    with self.assertRaisesRegex(RuntimeError, "namespace residue"):
                        M.check_source(False)
            finally:
                M.NEW_NAMESPACES = original

    def test_12_future_residue_rejected(self):
        original = M.FUTURE
        contract = M.expected_contract()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authority"
            path.mkdir()
            M.FUTURE = (path,)
            try:
                with mock.patch.object(M, "check_predecessor", return_value={}), \
                        mock.patch.object(M, "check_minimal_delta_text"), \
                        mock.patch.object(M, "check_contract",
                                          return_value=contract):
                    with self.assertRaisesRegex(RuntimeError, "future authority"):
                        M.check_source(True)
            finally:
                M.FUTURE = original

    def test_13_no_claim_promotion(self):
        self.assertTrue(all(value is False for value in M.CLAIMS.values()))
        self.assertFalse(M.expected_contract()["future_authority"]["launch_authorized"])
        self.assertFalse(M.expected_contract()["author_execution"]["eda"])

    def test_14_campaign_frozen(self):
        value = M.expected_contract()["preserved_execution"]
        self.assertEqual(value["axes"], ["k8", "k1x8"])
        self.assertEqual(value["cases"], [0, 1, 2, 3, 4])
        self.assertEqual((value["vcs_compiles"], value["simv_runs"],
                          value["production_saif_files"], value["ptpx_runs"]),
                         (2, 10, 10, 10))
        self.assertFalse(value["automatic_retry"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
