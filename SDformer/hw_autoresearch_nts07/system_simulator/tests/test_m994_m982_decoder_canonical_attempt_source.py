#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "scripts/execute_m994_m982_decoder_canonical_attempt_source_r1.py"
RUNNER = HERE.parent / "scripts/run_m998_m994_decoder_canonical_attempt_one_shot.sh"
CONTRACT = HERE.parent.parent / "contracts/m994_m982_decoder_canonical_attempt_source_contract_r1_20260829.json"
SPEC = importlib.util.spec_from_file_location("m994_test_driver", DRIVER)
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
AUTH = {"release_sha256": "a" * 64, "release_hammer_review_sha256": "b" * 64}


def names(parent):
    return parent / M.ATTEMPT.name, parent / M.RESULT.name


class M994Test(unittest.TestCase):
    def test_m982_stop_is_exactly_frozen(self):
        sealed = M.verify_m982()
        self.assertEqual(sealed["manifest_sha256"], M.M982_ID[1])

    def test_canonical_mkdir_is_irreversible_consumption(self):
        with tempfile.TemporaryDirectory() as t:
            parent = Path(t); attempt, result = names(parent)
            with self.assertRaisesRegex(RuntimeError, "after canonical"):
                M.consume_attempt(AUTH, attempt, result, parent, "after_canonical_mkdir")
            self.assertTrue(attempt.is_dir())
            self.assertFalse((attempt / "attempt.json").exists())
            with self.assertRaisesRegex(RuntimeError, "already consumed"):
                M.consume_attempt(AUTH, attempt, result, parent)
            self.assertEqual(list(parent.glob(M.ATTEMPT.name + ".stage.*")), [])

    def test_receipt_interruption_also_blocks_retry(self):
        with tempfile.TemporaryDirectory() as t:
            parent = Path(t); attempt, result = names(parent)
            with self.assertRaisesRegex(RuntimeError, "after attempt receipt"):
                M.consume_attempt(AUTH, attempt, result, parent, "after_attempt_receipt")
            self.assertTrue((attempt / "attempt.json").is_file())
            self.assertFalse((attempt / M.B.SEAL_DIR).exists())
            with self.assertRaisesRegex(RuntimeError, "already consumed"):
                M.consume_attempt(AUTH, attempt, result, parent)

    def test_sealed_canonical_attempt_validates(self):
        with tempfile.TemporaryDirectory() as t:
            parent = Path(t); attempt, result = names(parent)
            value = M.consume_attempt(AUTH, attempt, result, parent)
            self.assertEqual(value, M.validate_attempt(AUTH, attempt))
            self.assertTrue((attempt / M.B.SEAL_DIR / M.B.SEAL_OUTER).is_file())

    def test_postseal_interruption_still_blocks_retry(self):
        with tempfile.TemporaryDirectory() as t:
            parent = Path(t); attempt, result = names(parent)
            with self.assertRaisesRegex(RuntimeError, "after attempt seal"):
                M.consume_attempt(AUTH, attempt, result, parent, "after_attempt_seal")
            M.validate_attempt(AUTH, attempt)
            with self.assertRaisesRegex(RuntimeError, "already consumed"):
                M.consume_attempt(AUTH, attempt, result, parent)

    def test_source_contract_and_chain_are_additive(self):
        value = M.validate_source_contract(CONTRACT, RUNNER)
        self.assertEqual(value["status"], "PASS_M994_SOURCE__NO_REAL_10K")
        joined = " ".join(M.canonical_paths().values())
        for token in ("m994_", "m995_", "m996_", "m997_", "m998_"):
            self.assertIn(token, joined)
        self.assertNotIn("attempt.stage", RUNNER.read_text())

    def test_source_selftest_executes_no_prefix(self):
        value = M.source_self_test()
        self.assertFalse(value["real_10k_executed"])
        self.assertTrue(value["interrupted_canonical_attempt_blocks_retry"])


if __name__ == "__main__":
    unittest.main()
