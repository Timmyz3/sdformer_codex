#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Source-only regressions for M1459; never invokes VCS/simv/EDA."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1459_c1_generic_seal_successor_source.py"
RUNNER = HERE.parent / "dc_handoff/scripts/run_vcs_m1459_m1433_c1_runtime_split_generic_seal_successor.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.argv)
    try:
        sys.argv = [str(path)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


C = load("m1459_checker", CHECKER)
R = load("m1459_runner", RUNNER)


class M1459SourceTests(unittest.TestCase):
    def test_01_full_source_gate(self):
        result = C.check_source(require_future_absent=True)
        self.assertEqual(result["status"],
                         "PASS_M1459_C1_GENERIC_SEAL_SUCCESSOR_SOURCE__NO_VCS_NO_EDA")

    def test_02_predecessor_failure_is_pretool(self):
        failure = C.check_failure_evidence()
        self.assertEqual((failure["phase"], failure["compile_count"],
                          failure["sim_count"]), ("ATTEMPT_CONSUME", 0, 0))

    def test_03_predecessor_canonical_absent(self):
        self.assertFalse(any(path.exists() for path in
                             (C.M1433_ATTEMPT, C.M1433_RESULT, C.M1433_QUARANTINE)))

    def test_04_successor_canonical_absent(self):
        self.assertFalse(any(path.exists() for path in
                             (C.M1459_ATTEMPT, C.M1459_RESULT, C.M1459_QUARANTINE)))

    def test_05_frozen_runtime_suite(self):
        self.assertEqual(C.sha(C.RUNTIME_TESTS), C.RUNTIME_TESTS_SHA)

    def test_06_frozen_m1433_runner(self):
        self.assertEqual(C.sha(C.OLD_RUNNER), C.OLD_RUNNER_SHA)

    def test_07_generic_stage_without_review_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "attempt"
            root.mkdir()
            (root / "attempt.json").write_text('{"status":"TEST"}\n')
            R.seal_dir_generic(root)
            R.verify_recursive_seal_generic(root)
            self.assertFalse((root / "review.json").exists())

    def test_08_authority_without_review_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "attempt"
            root.mkdir()
            (root / "attempt.json").write_text('{"status":"TEST"}\n')
            R.seal_dir_generic(root)
            with self.assertRaises((FileNotFoundError, RuntimeError)):
                R.verify_authority(root)

    def test_09_authority_with_review_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "authority"
            root.mkdir()
            (root / "review.json").write_text('{"status":"PASS"}\n')
            R.seal_dir_generic(root)
            self.assertEqual(R.verify_authority(root)["status"], "PASS")

    def test_10_generic_payload_mutation_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "attempt"
            root.mkdir()
            payload = root / "attempt.json"
            payload.write_text('{"status":"TEST"}\n')
            R.seal_dir_generic(root)
            payload.write_text('{"status":"MUTATED"}\n')
            with self.assertRaises(RuntimeError):
                R.verify_recursive_seal_generic(root)

    def test_11_unlisted_member_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "attempt"
            root.mkdir()
            (root / "attempt.json").write_text('{"status":"TEST"}\n')
            R.seal_dir_generic(root)
            (root / "extra.txt").write_text("attack\n")
            with self.assertRaises(RuntimeError):
                R.verify_recursive_seal_generic(root)

    def test_12_symlink_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "attempt"
            root.mkdir()
            target = root / "attempt.json"
            target.write_text('{"status":"TEST"}\n')
            (root / "alias").symlink_to(target)
            with self.assertRaises(RuntimeError):
                R.seal_dir_generic(root)

    def test_13_contract_claims_are_exact(self):
        contract = C.check_contract()
        self.assertEqual(contract["claim_boundary"], C.CLAIMS)

    def test_14_c1_authority_namespaces_present(self):
        text = RUNNER.read_text()
        self.assertTrue(all(token in text for token in ("M1464", "M1465", "M1466")))

    def test_15_source_suite_unreachable(self):
        self.assertNotIn("BASE.SOURCE_TESTS", RUNNER.read_text())

    def test_16_failure_seal_is_generic(self):
        text = RUNNER.read_text()
        self.assertIn("seal_dir_generic(FAILURE_STAGE)", text)

    def test_17_attempt_seal_is_generic(self):
        text = RUNNER.read_text()
        self.assertIn("seal_dir_generic(ATTEMPT_STAGE)", text)

    def test_18_no_tool_execution_markers(self):
        contract = json.loads(C.CONTRACT.read_text())
        self.assertEqual(contract["author_execution"]["vcs"], False)
        self.assertEqual(contract["author_execution"]["simv"], False)
        self.assertEqual(contract["author_execution"]["eda"], False)


if __name__ == "__main__":
    unittest.main()
