#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-EDA regressions for M1350 strict JSON and three receipt writers."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "static_check_m1350_c2_activity_vcs_release_source.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = load("m1350_checker_test", CHECKER)
T = load("m1350_bound_m1344_tests", M.M1344_TEST)
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")


def update_source_expected(fixture) -> None:
    source = fixture.paths["source_hammer"]
    T.seal_dir(source)
    fixture.expected["M1344_EXPECTED_SOURCE_HAMMER_REVIEW_SHA256"] = M.sha(source / "review.json")
    fixture.expected["M1344_EXPECTED_SOURCE_HAMMER_MANIFEST_SHA256"] = M.sha(source / "SHA256SUMS")
    fixture.expected["M1344_EXPECTED_SOURCE_HAMMER_OUTER_FILE_SHA256"] = M.sha(source / "SHA256SUMS.seal.sha256")


def update_release_expected(fixture) -> None:
    release = fixture.paths["launch_release"]
    T.sidecar(release)
    fixture.expected["M1344_EXPECTED_LAUNCH_RELEASE_SHA256"] = M.sha(release)


def update_final_expected(fixture) -> None:
    final = fixture.paths["final_hammer"]
    T.seal_dir(final)
    fixture.expected["M1344_EXPECTED_FINAL_HAMMER_REVIEW_SHA256"] = M.sha(final / "review.json")
    fixture.expected["M1344_EXPECTED_FINAL_HAMMER_MANIFEST_SHA256"] = M.sha(final / "SHA256SUMS")
    fixture.expected["M1344_EXPECTED_FINAL_HAMMER_OUTER_FILE_SHA256"] = M.sha(final / "SHA256SUMS.seal.sha256")


def delete_identity_with_comment(runner: str, receipt: str, key: str) -> str:
    if receipt in ("failure", "attempt"):
        status = "FAILED_OR_INCOMPLETE" if receipt == "failure" else "M1344_ATTEMPT_CONSUMED"
        marker = "printf 'status=" + status
        start = runner.index(marker)
        end = runner.index("' \\", start)
        segment = runner[start:end]
        token = key + r"=%s\n"
        assert segment.count(token) == 1
        segment = segment.replace(token, "", 1)
        mutant = runner[:start] + segment + runner[end:]
    else:
        code = M.extract_success_python(runner)
        expression = M.SUCCESS_EXPRESSIONS[key]
        if key == M.IDENTITY_KEYS[-1]:
            token = ",'%s':%s" % (key, expression)
        else:
            token = "'%s':%s," % (key, expression)
        assert code.count(token) == 1, (key, token)
        mutated_code = code.replace(token, "", 1)
        mutant = runner.replace(code, mutated_code, 1)
    return mutant + "\n# inactive comment residue: %s appears here only\n" % key


def run_inherited(path: Path, count: int) -> None:
    env = dict(os.environ); env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run([str(PYTHON), "-B", str(path)], cwd=M.HW.parent, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False)
    if result.returncode != 0 or "Ran %d tests" % count not in result.stdout or "OK" not in result.stdout:
        raise AssertionError(result.stdout)


class Tests(unittest.TestCase):
    def test_01_canonical_three_receipt_parser_positive(self):
        result = M.validate_runner_receipts(M.RUNNER.read_text(encoding="utf-8"))
        self.assertEqual(result["receipts"], 3)
        self.assertEqual(result["identities_per_receipt"], 9)

    def test_02_canonical_future_chain_strict_positive(self):
        fixture = T.RuntimeFixture()
        try:
            result = M.validate_future_strict("runtime_present", fixture.paths, fixture.expected)
            self.assertTrue(result["strict_json"])
            self.assertTrue(result["exact_claim_boundaries"])
        finally:
            fixture.close()

    def test_03_duplicate_status_key_rejected(self):
        fixture = T.RuntimeFixture()
        try:
            path = fixture.paths["final_hammer"] / "review.json"
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace('"status":', '"status":"FORGED_DUPLICATE","status":', 1),
                            encoding="utf-8")
            update_final_expected(fixture)
            with self.assertRaisesRegex(AssertionError, "duplicate JSON key: status"):
                M.validate_future_strict("runtime_present", fixture.paths, fixture.expected)
        finally:
            fixture.close()

    def _extra_claim(self, document: str):
        fixture = T.RuntimeFixture()
        try:
            if document == "source":
                path = fixture.paths["source_hammer"] / "review.json"
            elif document == "release":
                path = fixture.paths["launch_release"]
            else:
                path = fixture.paths["final_hammer"] / "review.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["claim_boundary"]["launch_authorized"] = True
            path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
            if document == "source": update_source_expected(fixture)
            elif document == "release": update_release_expected(fixture)
            else: update_final_expected(fixture)
            with self.assertRaisesRegex(AssertionError, "exact nine-key"):
                M.validate_future_strict("runtime_present", fixture.paths, fixture.expected)
        finally:
            fixture.close()

    def test_04_extra_claim_source_rejected(self): self._extra_claim("source")
    def test_05_extra_claim_launch_rejected(self): self._extra_claim("release")
    def test_06_extra_claim_final_rejected(self): self._extra_claim("final")

    def test_07_inherited_m1344_12_of_12(self): run_inherited(M.M1344_TEST, 12)
    def test_08_inherited_m1336_10_of_10(self): run_inherited(M.M1336_TEST, 10)
    def test_09_inherited_m1334_12_of_12(self): run_inherited(M.M1334_TEST, 12)


def make_receipt_regression(receipt: str, key: str):
    def test(self):
        mutant = delete_identity_with_comment(M.RUNNER.read_text(encoding="utf-8"), receipt, key)
        with self.assertRaises(AssertionError):
            M.validate_runner_receipts(mutant)
    test.__name__ = "test_delete_%s_%s_with_comment_fill" % (receipt, key)
    return test


counter = 10
for receipt_name in ("failure", "attempt", "success"):
    for identity_key in M.IDENTITY_KEYS:
        setattr(Tests, "test_%02d_delete_%s_%s_with_comment_fill" %
                (counter, receipt_name, identity_key),
                make_receipt_regression(receipt_name, identity_key))
        counter += 1


if __name__ == "__main__":
    unittest.main(verbosity=2)
