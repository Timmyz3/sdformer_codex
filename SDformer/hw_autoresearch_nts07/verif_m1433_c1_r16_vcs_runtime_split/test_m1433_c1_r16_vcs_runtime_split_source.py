#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Author/source-only regressions for the additive M1433 runtime split."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1433_c1_r16_vcs_runtime_split_source.py"
RUNTIME = HERE / "test_m1433_c1_r16_vcs_runtime_present.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1433_source_checker", CHECKER)
R = load("m1433_runtime_tests", RUNTIME)


class Tests(unittest.TestCase):
    def test_01_canonical_common_and_contract(self):
        common = M.validate_common(skip_author=True)
        self.assertTrue(common["m1364_p0_bound"])
        self.assertEqual(M.strict_json(M.CONTRACT), M.expected_contract())

    def test_02_runner_exact_identity(self):
        contract = M.strict_json(M.CONTRACT)
        self.assertEqual(contract["identity"]["runner_sha256"], M.sha(M.RUNNER))
        self.assertEqual(contract["identity"]["runtime_tests_sha256"], M.sha(M.RUNTIME_TESTS))

    def test_03_runner_protocol_audit(self):
        audit = M.audit_runner(M.RUNNER.read_text())
        self.assertTrue(audit["attempt_before_tool"])
        self.assertEqual(audit["collision_gates"], 2)
        self.assertTrue(audit["failure_quarantine_recursive_seal"])
        self.assertTrue(audit["runtime_suite_only"])

    def test_20_env_pins_fail_closed(self):
        names = (
            "M1433_EXPECTED_RUNNER_SHA256", "M1433_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256",
            "M1433_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256",
            "M1433_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256",
            "M1433_EXPECTED_LAUNCH_RELEASE_SHA256", "M1433_EXPECTED_FINAL_HAMMER_REVIEW_SHA256",
            "M1433_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256",
            "M1433_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256")
        good = {name: "a" * 64 for name in names}; self.assertTrue(M.env_gate(good))
        for name in names:
            for value in (None, "a" * 63, "A" * 64):
                bad = dict(good)
                if value is None: bad.pop(name)
                else: bad[name] = value
                self.assertFalse(M.env_gate(bad))

    def test_21_duplicate_and_nonfinite_json_rejected(self):
        text = M.CONTRACT.read_text().replace('"status":', '"status":"DUP","status":', 1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"; path.write_text(text)
            with self.assertRaisesRegex(AssertionError, "duplicate JSON key"): M.strict_json(path)
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(AssertionError, "nonfinite"): M.strict_json(path)

    def test_22_source_author_gate_requires_future_absent(self):
        self.assertTrue(M.validate_future("source_absent")["future_absent"])

    def test_23_runtime_suite_is_separate_and_launch_safe(self):
        text = RUNTIME.read_text()
        self.assertIn('validate_future("runtime_present")', text)
        self.assertNotIn('validate_future("source_absent")', text)
        self.assertEqual(R.validate_contract_regressions()["rejected"], 16)
        self.assertNotIn("lmstat", text)
        self.assertNotIn("subprocess", text)


def make_regression(name, mutate):
    def test(self):
        candidate = copy.deepcopy(M.expected_contract()); mutate(candidate)
        with self.assertRaisesRegex(AssertionError, "exact-set/value"):
            M.check_contract_dict(candidate)
    test.__name__ = "test_predecessor_regression_" + name
    return test


for index, (name, mutate) in enumerate(R.predecessor_regression_cases(), start=4):
    setattr(Tests, "test_{:02d}_predecessor_regression_{}".format(index, name),
            make_regression(name, mutate))


if __name__ == "__main__":
    unittest.main(verbosity=2)
