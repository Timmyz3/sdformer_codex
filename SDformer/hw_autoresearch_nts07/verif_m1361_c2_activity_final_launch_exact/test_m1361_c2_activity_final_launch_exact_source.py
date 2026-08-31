#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-EDA regressions for all thirty M1357 contract false negatives."""
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
CHECKER = HERE / "static_check_m1361_c2_activity_final_launch_exact_source.py"
SPEC = importlib.util.spec_from_file_location("m1361_exact_source", CHECKER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return "M1361_MUTATED"
    if type(value) is list:
        return value + ["M1361_MUTATED"]
    if type(value) is dict:
        result = dict(value); result["m1361_extra"] = True; return result
    raise TypeError(type(value))


def mutation_cases():
    cases = [
        ("contract_extra_top_level", lambda d: d.__setitem__("m1361_extra", True)),
        ("contract_date_changed", lambda d: d.__setitem__("date", "2099-01-01")),
        ("contract_purpose_changed", lambda d: d.__setitem__("purpose", "M1361_MUTATED")),
        ("one_shot_removed", lambda d: d.pop("one_shot")),
    ]
    base = M.expected_contract()
    cases.extend(("one_shot_" + key, lambda d, key=key:
                  d["one_shot"].__setitem__(key, changed(d["one_shot"][key])))
                 for key in base["one_shot"])
    cases.append(("resource_fail_close_removed", lambda d: d.pop("resource_fail_close")))
    cases.extend(("resource_fail_close_" + key, lambda d, key=key:
                  d["resource_fail_close"].__setitem__(
                      key, changed(d["resource_fail_close"][key])))
                 for key in base["resource_fail_close"])
    cases.append(("receipt_contract_removed", lambda d: d.pop("receipt_contract")))
    cases.extend(("receipt_contract_" + key, lambda d, key=key:
                  d["receipt_contract"].__setitem__(
                      key, changed(d["receipt_contract"][key])))
                 for key in base["receipt_contract"])
    cases.extend([
        ("authorization_automatic_retry_true", lambda d:
         d["authorization"].__setitem__("automatic_retry", True)),
        ("authorization_source_only_tests_false", lambda d:
         d["authorization"].__setitem__("source_only_tests", False)),
        ("future_blind_zero_false_negatives_false", lambda d:
         d["future_blind"].__setitem__("zero_false_negatives_required", False)),
        ("future_blind_fresh_different_author_false", lambda d:
         d["future_blind"].__setitem__("fresh_different_author", False)),
        ("protected_files_removed", lambda d: d.pop("protected_files")),
    ])
    assert len(cases) == 30, len(cases)
    return cases


class Tests(unittest.TestCase):
    def test_01_canonical_exact_contract_passes(self):
        self.assertEqual(M.validate_contract(skip_author=True), M.expected_contract())

    def test_02_m1356_and_m1357_chain_passes(self):
        common = M.validate_common(skip_author=True)
        self.assertEqual(common["m1357_false_negatives_repaired"], 30)
        self.assertIs(common["launch_authorized"], False)

    def test_33_duplicate_json_key_rejected(self):
        text = M.CONTRACT.read_text(encoding="utf-8").replace(
            '"status":', '"status":"DUPLICATE","status":', 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "duplicate JSON key"):
                M.strict_json(path)

    def test_34_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            path.write_text('{"x":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "nonfinite"):
                M.strict_json(path)

    def test_35_future_blind_absent_and_residue_rejected(self):
        self.assertTrue(M.validate_future("source_absent")["future_blind_absent"])
        original = M.FUTURE_BLIND
        with tempfile.TemporaryDirectory() as temporary:
            M.FUTURE_BLIND = Path(temporary) / "residue"; M.FUTURE_BLIND.mkdir()
            try:
                with self.assertRaisesRegex(AssertionError, "residue"):
                    M.validate_future("source_absent")
            finally:
                M.FUTURE_BLIND = original

    def test_36_claim_identity_and_no_execution_boundary(self):
        base = M.expected_contract()
        self.assertEqual(base["claim_boundary"], M.EXACT_CLAIMS)
        self.assertIs(base["authorization"]["launch_authorized"], False)
        text = CHECKER.read_text(encoding="utf-8")
        self.assertNotIn("lmstat -a", text)
        self.assertNotIn("subprocess.run", text)


def make_mutation_test(name, mutate):
    def test(self):
        candidate = copy.deepcopy(M.expected_contract())
        mutate(candidate)
        with mock.patch.object(M, "strict_json", return_value=candidate):
            with self.assertRaisesRegex(AssertionError, "exact-set/value"):
                M.validate_contract(skip_author=True)
    test.__name__ = "test_m1357_regression_" + name
    return test


for index, (name, mutate) in enumerate(mutation_cases(), start=3):
    setattr(Tests, "test_{:02d}_m1357_regression_{}".format(index, name),
            make_mutation_test(name, mutate))


if __name__ == "__main__":
    unittest.main(verbosity=2)
