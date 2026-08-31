#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Source-only regressions for the fresh M1363 C1/R16 runner and contract."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1363_c1_r16_vcs_release_exact_source.py"
SPEC = importlib.util.spec_from_file_location("m1363_source", CHECKER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def changed(value):
    if type(value) is bool: return not value
    if type(value) is int: return value + 1
    if type(value) is str: return "M1363_MUTATED"
    if type(value) is dict:
        result = dict(value); result["m1363_extra"] = True; return result
    raise TypeError(type(value))


def m1355_cases():
    base = M.expected_contract()
    cases = [
        ("contract_extra_top_level", lambda d: d.__setitem__("m1363_extra", True)),
        ("contract_date_changed", lambda d: d.__setitem__("date", "2099-01-01")),
        ("contract_future_execution_removed", lambda d: d.pop("future_execution")),
        ("contract_future_execution_extra", lambda d:
         d["future_execution"].__setitem__("m1363_extra", True)),
    ]
    cases.extend(("future_execution_" + key, lambda d, key=key:
                  d["future_execution"].__setitem__(key, changed(d["future_execution"][key])))
                 for key in base["future_execution"])
    cases.extend([
        ("author_execution_extra", lambda d: d["author_execution"].__setitem__("m1363_extra", False)),
        ("claim_boundary_extra", lambda d: d["claim_boundary"].__setitem__("m1363_extra", False)),
    ])
    assert len(cases) == 16, len(cases)
    return cases


class Tests(unittest.TestCase):
    def test_01_canonical_common_and_contract(self):
        common = M.validate_common(skip_author=True)
        self.assertEqual(common["m1355_false_negatives_bound"], 16)
        self.assertEqual(M.strict_json(M.CONTRACT), M.expected_contract())

    def test_02_runner_exact_byte_mutations_rejected(self):
        source = M.RUNNER.read_bytes()
        for mutant in (source + b"\n", source.replace(b"one-shot C1/R16", b"two-shot C1/R16", 1),
                       source.replace(b"./simv -no_save", b"./simv", 1)):
            self.assertNotEqual(__import__("hashlib").sha256(mutant).hexdigest(), M.RUNNER_SHA256)

    def test_03_runner_protocol_audit(self):
        audit = M.audit_runner(M.RUNNER.read_text())
        self.assertTrue(audit["attempt_before_tool"])
        self.assertEqual(audit["collision_gates"], 2)
        self.assertTrue(audit["failure_quarantine_recursive_seal"])

    def test_20_env_pins_fail_closed(self):
        names = (
            "M1363_EXPECTED_RUNNER_SHA256", "M1363_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256",
            "M1363_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256", "M1363_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256",
            "M1363_EXPECTED_LAUNCH_RELEASE_SHA256", "M1363_EXPECTED_FINAL_HAMMER_REVIEW_SHA256",
            "M1363_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256", "M1363_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256")
        good = {name: "a" * 64 for name in names}; self.assertTrue(M.env_gate(good))
        for name in names:
            bad = dict(good); bad.pop(name); self.assertFalse(M.env_gate(bad))
            bad = dict(good); bad[name] = "B" * 64; self.assertFalse(M.env_gate(bad))

    def test_21_duplicate_and_nonfinite_json_rejected(self):
        text = M.CONTRACT.read_text().replace('"status":', '"status":"DUP","status":', 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"; path.write_text(text)
            with self.assertRaisesRegex(AssertionError, "duplicate JSON key"): M.strict_json(path)
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(AssertionError, "nonfinite"): M.strict_json(path)

    def test_22_source_absent_and_residue(self):
        self.assertTrue(M.validate_future("source_absent")["future_absent"])
        original = M.FUTURE_HAMMER
        with tempfile.TemporaryDirectory() as temporary:
            M.FUTURE_HAMMER = Path(temporary) / "residue"; M.FUTURE_HAMMER.mkdir()
            try:
                with self.assertRaisesRegex(AssertionError, "residue"): M.validate_future("source_absent")
            finally: M.FUTURE_HAMMER = original

    def test_23_author_boundary_and_no_execution(self):
        contract = M.expected_contract()
        self.assertEqual(contract["claim_boundary"], M.EXACT_CLAIMS)
        self.assertIs(contract["future_release"]["launch_authorized"], False)
        text = CHECKER.read_text()
        self.assertNotIn("lmstat -a", text)
        self.assertNotIn("subprocess.run([str(M.RUNNER)", text)


def make_regression(name, mutate):
    def test(self):
        candidate = copy.deepcopy(M.expected_contract()); mutate(candidate)
        with self.assertRaisesRegex(AssertionError, "exact-set/value"):
            M.check_contract_dict(candidate)
    test.__name__ = "test_m1355_regression_" + name
    return test


for index, (name, mutate) in enumerate(m1355_cases(), start=4):
    setattr(Tests, "test_{:02d}_m1355_regression_{}".format(index, name), make_regression(name, mutate))


if __name__ == "__main__":
    unittest.main(verbosity=2)
