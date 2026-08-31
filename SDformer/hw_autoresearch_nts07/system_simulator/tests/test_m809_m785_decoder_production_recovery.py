#!/usr/bin/env python3
"""Source-only tests for the M809 decoder production recovery boundary."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "scripts/execute_m809_m785_decoder_physical_residency_production.py"
RUNNER = HERE.parent / "scripts/run_m809_m785_decoder_physical_residency_one_shot.sh"


def load_driver():
    spec = importlib.util.spec_from_file_location("m809_test_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M809 driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M809 = load_driver()


class M809RecoveryTests(unittest.TestCase):
    def test_synthetic_self_test(self):
        result = M809.self_test()
        self.assertEqual(
            result["status"],
            "PASS_M809_REPAIRED_DRIVER_SYNTHETIC_SELF_TEST")
        self.assertTrue(result["duplicate_json_rejected"])
        self.assertTrue(result["d1_headline_perturbation_invariant"])
        self.assertTrue(result["atomic_destination_race_rejected"])
        self.assertTrue(result["flat_attempt_population_passed"])
        self.assertTrue(result["old_hierarchical_attempt_precise_failure"])
        self.assertTrue(result[
            "prestage_failure_receipt_four_member_sealed"])
        self.assertTrue(result["failure_quarantine_collision_rejected"])
        self.assertIsNone(result["production_cycles"])

    def test_duplicate_authorization_key_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m809_dup_") as directory:
            path = Path(directory) / "release.json"
            path.write_text(
                '{"launch_now":true,"launch_now":false,"release":true}\n',
                encoding="utf-8")
            with self.assertRaises(M809.Failure):
                M809.strict_json(path)

    def test_duplicate_canonical_path_key_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m809_dup_path_") as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(
                '{"canonical":{"result":"a","result":"b"}}\n',
                encoding="utf-8")
            with self.assertRaises(M809.Failure):
                M809.strict_json(path)

    def test_d1_does_not_change_headline_ratio(self):
        baseline = {
            M809.HEADLINE_DENOMINATOR: {
                "headline_total_cycles": 900,
                "total_cycles": 1000,
            },
            M809.HEADLINE_NUMERATOR: {
                "headline_total_cycles": 600,
                "total_cycles": 800,
            },
        }
        expected = M809.headline_ratio(baseline)
        baseline[M809.HEADLINE_DENOMINATOR]["total_cycles"] += 99999999
        baseline[M809.HEADLINE_NUMERATOR]["total_cycles"] += 1
        self.assertEqual(M809.headline_ratio(baseline), expected)
        self.assertEqual(expected, 1.5)

    def test_atomic_destination_race_is_no_replace(self):
        with tempfile.TemporaryDirectory(prefix="m809_race_") as directory:
            parent = Path(directory)
            stage = parent / "result.stage.1"
            result = parent / "result"
            stage.mkdir()
            (stage / "sentinel").write_text("stage\n", encoding="utf-8")
            result.mkdir()
            (result / "attacker").write_text("collision\n", encoding="utf-8")
            with self.assertRaises(M809.Failure):
                M809._rename_noreplace(stage, result)
            self.assertEqual((stage / "sentinel").read_text(), "stage\n")
            self.assertEqual((result / "attacker").read_text(), "collision\n")
            self.assertFalse((result / stage.name).exists())

    def test_flat_attempt_passes_and_old_hierarchy_fails_precisely(self):
        with tempfile.TemporaryDirectory(prefix="m809_flat_") as directory:
            root = Path(directory)
            flat = root / "flat"
            M809.write_flat_attempt(flat, {
                "schema": "m809_source_test_attempt_v1",
                "status": "SOURCE_TEST_ONLY",
            })
            identity = M809.verify_sealed(flat)
            self.assertEqual(identity["manifest_sha256"],
                             M809.sha256(flat / "SHA256SUMS"))
            self.assertEqual({entry.name for entry in flat.iterdir()}, {
                "attempt.json", "SHA256SUMS",
                "SHA256SUMS.seal.sha256"})

            old = root / "old"
            initial = old / "initial"
            initial.mkdir(parents=True)
            M809._write_json(initial / "attempt.json", {"old": True})
            M809.seal_exact_members(initial, ("attempt.json",))
            (old / "SHA256SUMS").write_text(
                M809.sha256(initial / "SHA256SUMS.seal.sha256") +
                "  initial/SHA256SUMS.seal.sha256\n", encoding="utf-8")
            (old / "SHA256SUMS.seal.sha256").write_text(
                M809.sha256(old / "SHA256SUMS") +
                "  SHA256SUMS\n", encoding="utf-8")
            with self.assertRaisesRegex(M809.Failure,
                                        "^sealed population mismatch$"):
                M809.verify_sealed(old)

    def test_prestage_failure_receipt_is_four_member_double_sealed(self):
        with tempfile.TemporaryDirectory(prefix="m809_receipt_") as directory:
            root = Path(directory)
            stdout = root / "stdout"
            stderr = root / "stderr"
            stdout.write_text("preflight stdout\n", encoding="utf-8")
            stderr.write_text("pre-stage failure\n", encoding="utf-8")
            quarantine = root / "result.failed_or_incomplete.test"
            M809._write_failure_receipt(quarantine, stdout, stderr, {
                "schema": "m809_source_test_failure_v1",
                "status": "FAILED_BEFORE_STAGE__SOURCE_TEST_ONLY",
                "return_code": 19,
                "production_replay": False,
            })
            self.assertEqual({entry.name for entry in quarantine.iterdir()}, {
                "failure.json", "driver.log", "SHA256SUMS",
                "SHA256SUMS.seal.sha256"})
            M809.verify_sealed(quarantine)
            failure = M809.strict_json(quarantine / "failure.json")
            self.assertEqual(failure["return_code"], 19)
            self.assertEqual(failure["driver_log_sha256"],
                             M809.sha256(quarantine / "driver.log"))

            before = {entry.name: M809.sha256(entry) for entry in
                      quarantine.iterdir() if entry.is_file()}
            with self.assertRaisesRegex(
                    M809.Failure, "failure quarantine destination collision"):
                M809._write_failure_receipt(
                    quarantine, stdout, stderr, {"status": "ATTACK"})
            after = {entry.name: M809.sha256(entry) for entry in
                     quarantine.iterdir() if entry.is_file()}
            self.assertEqual(before, after)

    def test_wrong_sha_gate_is_before_attempt_and_has_zero_formal_side_effect(self):
        text = RUNNER.read_text(encoding="utf-8")
        sha_gate = text.index('"${M809_EXPECTED_RUNNER_SHA256}" ]] || {')
        attempt_mkdir = text.index('mkdir "${m809_attempt_stage}"')
        self.assertLess(sha_gate, attempt_mkdir)
        with tempfile.TemporaryDirectory(prefix="m809_wrong_sha_") as directory:
            root = Path(directory)
            source = root / "source"
            attempt = root / "attempt"
            result = root / "result"
            source.write_text("source\n", encoding="utf-8")
            with self.assertRaisesRegex(M809.Failure, "SHA drift"):
                M809.regular_exact(source, "0" * 64, "synthetic source")
            self.assertFalse(attempt.exists())
            self.assertFalse(result.exists())

    def test_runner_uses_explicit_atomic_publication_and_root_postcheck(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("--publish-no-replace", text)
        self.assertIn("renameat2(RENAME_NOREPLACE)", text)
        self.assertIn('"${m809_result}/result.json"', text)
        self.assertIn('"${m809_result}/detailed_rows.json"', text)
        self.assertIn('"${m809_result}/SHA256SUMS"', text)
        self.assertIn('"${m809_result}/SHA256SUMS.seal.sha256"', text)
        self.assertNotIn('mv -- "${m809_stage}" "${m809_result}"', text)
        self.assertIn('--validate-consumed-attempt', text)
        self.assertIn('--write-failure-receipt', text)
        self.assertIn('"${m809_attempt}/attempt.json"', text)
        self.assertNotIn('"${m809_attempt}/initial"', text)
        self.assertIn('mv -T --no-clobber -- "${m809_attempt_stage}" "${m809_attempt}"',
                      text)
        self.assertIn('>>"${m809_stdout_log}" 2>>"${m809_stderr_log}"',
                      text)
        self.assertIn('"${m809_quarantine}/failure.json"', text)
        self.assertIn('"${m809_quarantine}/driver.log"', text)


if __name__ == "__main__":
    unittest.main()
