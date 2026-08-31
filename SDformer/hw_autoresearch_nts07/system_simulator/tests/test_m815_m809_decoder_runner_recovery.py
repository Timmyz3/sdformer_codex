#!/usr/bin/env python3
"""Source-only tests for the additive M815 decoder runner repair."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HERE.parent / "scripts/execute_m815_m809_decoder_production_runner_recovery.py"
RUNNER = HERE.parent / "scripts/run_m815_m785_decoder_physical_residency_one_shot.sh"
CANDIDATE = HW / "contracts/m815_m785_decoder_production_runner_recovery_candidate_r1_20260829.json"


def load_driver():
    spec = importlib.util.spec_from_file_location("m815_source_test", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M815 driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M815 = load_driver()


class M815RunnerRecoveryTests(unittest.TestCase):
    def test_self_test(self):
        value = M815.self_test()
        self.assertEqual(
            value["status"],
            "PASS_M815_RUNNER_RECOVERY_SYNTHETIC_SELF_TEST")
        self.assertEqual(value["postpublish_injection"]["scheduled_rows"], 0)
        self.assertFalse(value["postpublish_injection"][
            "canonical_result_exists"])
        self.assertTrue(value["postpublish_injection"][
            "collision_no_clobber"])
        self.assertIsNone(value["production_cycles"])

    def test_runner_arms_trap_before_postcheck(self):
        text = RUNNER.read_text(encoding="utf-8")
        publish = text.index(
            'mv -T --no-clobber -- "${m815_attempt_stage}" "${m815_attempt}"')
        started = text.index("m815_started=1", publish)
        phase = text.index('m815_phase="ATTEMPT_PUBLISHED_POSTCHECK"', publish)
        postcheck = text.index('[[ -d "${m815_attempt}"', publish)
        consumed = text.index("--validate-consumed-attempt", postcheck)
        production = text.index("--run-production", consumed)
        self.assertLess(publish, started)
        self.assertLess(started, phase)
        self.assertLess(phase, postcheck)
        self.assertLess(postcheck, consumed)
        self.assertLess(consumed, production)

    def test_injected_postpublish_failure_exact_receipt(self):
        value = M815.injected_postpublish_failure_test()
        self.assertEqual(
            value["status"], "PASS_M815_INJECTED_POSTPUBLISH_FAILURE")
        self.assertEqual(value["scheduled_rows"], 0)
        self.assertTrue(value["attempt_consumed"])
        self.assertFalse(value["canonical_result_exists"])
        self.assertEqual(set(value["failure_members"]), {
            "failure.json", "driver.log", "SHA256SUMS",
            "SHA256SUMS.seal.sha256"})

    def test_failure_destination_collision_is_no_clobber(self):
        with tempfile.TemporaryDirectory(prefix="m815_collision_") as directory:
            root = Path(directory)
            stdout = root / "stdout"
            stderr = root / "stderr"
            output = root / "failure"
            stdout.write_text("stdout\n", encoding="utf-8")
            stderr.write_text("stderr\n", encoding="utf-8")
            M815._write_failure_receipt(
                output, stdout, stderr, {"status": "FIRST"})
            before = {entry.name: M815.sha256(entry) for entry in
                      output.iterdir()}
            with self.assertRaises(M815.Failure):
                M815._write_failure_receipt(
                    output, stdout, stderr, {"status": "ATTACK"})
            after = {entry.name: M815.sha256(entry) for entry in
                     output.iterdir()}
            self.assertEqual(before, after)

    def test_flat_attempt_and_strict_json_are_inherited_exactly(self):
        with tempfile.TemporaryDirectory(prefix="m815_flat_") as directory:
            root = Path(directory)
            attempt = root / "attempt"
            M815.write_flat_attempt(attempt, {
                "schema": "m815_source_test_attempt_v1",
                "status": "SOURCE_TEST_ONLY",
            })
            M815.verify_sealed(attempt)
            self.assertEqual({entry.name for entry in attempt.iterdir()}, {
                "attempt.json", "SHA256SUMS", "SHA256SUMS.seal.sha256"})
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
            with self.assertRaises(M815.Failure):
                M815.strict_json(duplicate)
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
            with self.assertRaises(M815.Failure):
                M815.strict_json(nonfinite)

    def test_candidate_validation_is_source_only(self):
        value = M815.validate_candidate(CANDIDATE)
        self.assertEqual(
            value["status"],
            "PASS_M815_RUNNER_RECOVERY_SOURCE_CANDIDATE__NO_PRODUCTION_RUN")
        self.assertIsNone(value["production_cycles"])

    def test_m811_remains_negative_provenance(self):
        candidate = M815.strict_json(CANDIDATE)
        basis = candidate["m811_no_go_basis"]
        self.assertFalse(basis["true_release_authorized"])
        self.assertTrue(basis["additive_runner_repair_required"])
        review = M815.strict_json(M815.M811_REVIEW / "review.json")
        self.assertEqual(
            review["status"],
            "NO_GO_M809_TRUE_RELEASE__P1_1__AUTHOR_ADDITIVE_RUNNER_REPAIR_REQUIRED")
        self.assertFalse(review["true_release_authorized"])

    def test_frozen_schedule_resource_and_headline_are_identical(self):
        candidate = M815.strict_json(CANDIDATE)
        parent = M815.strict_json(M815.M809_CANDIDATE)
        self.assertEqual(candidate["common_resource"],
                         parent["common_resource"])
        expected = dict(parent["production_semantics"])
        expected.pop("external_opportunity_artifact_candidate_input")
        actual = dict(candidate["production_semantics"])
        actual.pop("delegated_schedule_body")
        self.assertEqual(actual, expected)
        self.assertEqual(candidate["production_semantics"][
            "delegated_schedule_body"], "FROZEN_M809_EXACT_SHA")
        self.assertEqual(candidate["source_identity"][
            "m809_parent_driver"]["sha256"], M815.M809_SHA256)

    def test_formal_artifacts_are_absent(self):
        candidate = M815.strict_json(CANDIDATE)
        result, attempt, release = M815._canonical_paths(candidate)
        self.assertFalse(result.exists() or result.is_symlink())
        self.assertFalse(attempt.exists() or attempt.is_symlink())
        self.assertFalse(release.exists() or release.is_symlink())

    def test_runner_log_creation_refuses_symlink_clobber_pattern(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('Path(sys.argv[1]).open("x").close()', text)
        self.assertNotIn('|| : >"${m815_stdout_log}"', text)
        self.assertNotIn('|| : >"${m815_stderr_log}"', text)


if __name__ == "__main__":
    unittest.main()
