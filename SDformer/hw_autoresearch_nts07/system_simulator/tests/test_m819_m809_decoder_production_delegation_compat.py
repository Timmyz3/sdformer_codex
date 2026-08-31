#!/usr/bin/env python3
"""Source-only tests for M819 parent-compatible decoder delegation."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HERE.parent / "scripts/execute_m819_m809_decoder_production_delegation_compat.py"
RUNNER = HERE.parent / "scripts/run_m819_m785_decoder_physical_residency_one_shot.sh"
CANDIDATE = HW / "contracts/m819_m785_decoder_production_delegation_compat_candidate_r1_20260829.json"


def load_driver():
    spec = importlib.util.spec_from_file_location("m819_source_test", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M819 driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M819 = load_driver()


class M819DelegationCompatibilityTests(unittest.TestCase):
    def test_self_test(self):
        value = M819.self_test()
        self.assertEqual(value["status"],
                         "PASS_M819_DELEGATION_COMPAT_SYNTHETIC_SELF_TEST")
        self.assertEqual(value["preproduction_traversal"]["scheduled_rows"], 0)
        self.assertFalse(value["preproduction_traversal"]["output_exists"])
        self.assertIsNone(value["production_cycles"])

    def test_parent_compatible_preproduction_traversal(self):
        value = M819.preproduction_traversal_test()
        self.assertEqual(value["status"],
                         "PASS_M819_PARENT_COMPAT_PREPRODUCTION_TRAVERSAL")
        self.assertTrue(value["parent_attempt_status_accepted"])
        self.assertFalse(value["attempt_receipt_identity_drift"])
        self.assertEqual(value["stopped_at"], "M809_OUTPUT_MKDIR")
        self.assertEqual(value["scheduled_rows"], 0)
        self.assertFalse(value["output_exists"])

    def test_runner_order_and_exact_parent_token(self):
        text = RUNNER.read_text(encoding="utf-8")
        publish = text.index(
            'mv -T --no-clobber -- "${m819_attempt_stage}" "${m819_attempt}"')
        started = text.index("m819_started=1", publish)
        phase = text.index('m819_phase="ATTEMPT_PUBLISHED_POSTCHECK"', publish)
        postcheck = text.index('[[ -d "${m819_attempt}"', publish)
        consumed = text.index("--validate-consumed-attempt", postcheck)
        production = text.index("--run-production", consumed)
        self.assertLess(publish, started)
        self.assertLess(started, phase)
        self.assertLess(phase, postcheck)
        self.assertLess(postcheck, consumed)
        self.assertLess(consumed, production)
        self.assertIn('"status": "' + M819.PARENT_ATTEMPT_STATUS + '"', text)
        self.assertNotIn("CONSUMED_IMMEDIATELY_BEFORE_M819_PRODUCTION_REPLAY",
                         text)

    def test_injected_postpublish_failure_exact_receipt(self):
        value = M819.injected_postpublish_failure_test()
        self.assertEqual(value["status"],
                         "PASS_M819_INJECTED_POSTPUBLISH_FAILURE")
        self.assertEqual(value["scheduled_rows"], 0)
        self.assertTrue(value["attempt_consumed"])
        self.assertFalse(value["canonical_result_exists"])
        self.assertEqual(set(value["failure_members"]), {
            "failure.json", "driver.log", "SHA256SUMS",
            "SHA256SUMS.seal.sha256"})

    def test_failure_destination_and_log_symlink_no_clobber(self):
        with tempfile.TemporaryDirectory(prefix="m819_symlink_") as directory:
            root = Path(directory)
            sentinel = root / "sentinel"
            stdout, stderr = root / "stdout", root / "stderr"
            sentinel.write_text("KEEP", encoding="utf-8")
            stdout.write_text("stdout", encoding="utf-8")
            stderr.write_text("stderr", encoding="utf-8")
            output = root / "output"
            output.symlink_to(sentinel)
            with self.assertRaises(M819.Failure):
                M819._write_failure_receipt(output, stdout, stderr,
                                            {"status": "ATTACK"})
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "KEEP")
            output.unlink()
            stdout.unlink()
            stdout.symlink_to(sentinel)
            with self.assertRaises(M819.Failure):
                M819._write_failure_receipt(output, stdout, stderr,
                                            {"status": "ATTACK"})
            self.assertFalse(output.exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "KEEP")

    def test_flat_attempt_and_strict_json_inherited(self):
        with tempfile.TemporaryDirectory(prefix="m819_flat_") as directory:
            root = Path(directory)
            attempt = root / "attempt"
            M819.write_flat_attempt(attempt, {
                "schema": "m819_source_test_attempt_v1",
                "status": M819.PARENT_ATTEMPT_STATUS})
            M819.verify_sealed(attempt)
            self.assertEqual({entry.name for entry in attempt.iterdir()}, {
                "attempt.json", "SHA256SUMS", "SHA256SUMS.seal.sha256"})
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
            with self.assertRaises(M819.Failure):
                M819.strict_json(duplicate)
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
            with self.assertRaises(M819.Failure):
                M819.strict_json(nonfinite)

    def test_candidate_validation_is_source_only(self):
        value = M819.validate_candidate(CANDIDATE)
        self.assertEqual(value["status"],
                         "PASS_M819_DELEGATION_COMPAT_SOURCE_CANDIDATE__NO_PRODUCTION_RUN")
        self.assertIsNone(value["production_cycles"])

    def test_m817_remains_negative_provenance(self):
        candidate = M819.strict_json(CANDIDATE)
        basis = candidate["m817_no_go_basis"]
        self.assertFalse(basis["true_release_authorized"])
        self.assertTrue(basis["additive_delegation_repair_required"])
        review = M819.strict_json(M819.M817_REVIEW / "review.json")
        self.assertEqual(review["status"],
                         "NO_GO_M815_TRUE_RELEASE__P1_1__ADDITIVE_DELEGATION_ATTEMPT_STATUS_REPAIR_REQUIRED")
        self.assertFalse(review["true_release_authorized"])

    def test_frozen_schedule_resource_and_headline_are_identical(self):
        candidate = M819.strict_json(CANDIDATE)
        parent = M819.strict_json(M819.M815_CANDIDATE)
        self.assertEqual(candidate["common_resource"], parent["common_resource"])
        expected = dict(parent["production_semantics"])
        expected.pop("delegated_schedule_body")
        actual = dict(candidate["production_semantics"])
        self.assertEqual(actual.pop("delegated_schedule_body"),
                         "FROZEN_M809_EXACT_SHA")
        self.assertEqual(actual, expected)
        self.assertEqual(candidate["attempt_compatibility"]["formal_status"],
                         M819.PARENT_ATTEMPT_STATUS)
        self.assertEqual(M819.sha256(M819.M809_PATH), M819.M809_SHA256)

    def test_formal_artifacts_are_absent(self):
        candidate = M819.strict_json(CANDIDATE)
        result, attempt, release = M819._canonical_paths(candidate)
        self.assertFalse(result.exists() or result.is_symlink())
        self.assertFalse(attempt.exists() or attempt.is_symlink())
        self.assertFalse(release.exists() or release.is_symlink())

    def test_runner_log_creation_refuses_symlink_clobber_pattern(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('Path(sys.argv[1]).open("x").close()', text)
        self.assertNotIn('|| : >"${m819_stdout_log}"', text)
        self.assertNotIn('|| : >"${m819_stderr_log}"', text)

    def test_delegated_validator_restored_on_exception(self):
        candidate = M819.strict_json(CANDIDATE)
        original_local = M819.validate_true_release
        original_parent_validate = M819.M809.validate_true_release
        original_parent_run = M819.M809.run_production

        class Forced(RuntimeError):
            pass

        def forced(*args, **kwargs):
            self.assertIsNot(M819.M809.validate_true_release,
                             original_parent_validate)
            raise Forced()

        M819.validate_true_release = lambda *args, **kwargs: {
            "candidate": candidate}
        M819.M809.run_production = forced
        try:
            with self.assertRaises(Forced):
                M819.run_production(Path("/release"), CANDIDATE,
                                    Path("/attempt"), Path("/output"))
            self.assertIs(M819.M809.validate_true_release,
                          original_parent_validate)
        finally:
            M819.validate_true_release = original_local
            M819.M809.validate_true_release = original_parent_validate
            M819.M809.run_production = original_parent_run


if __name__ == "__main__":
    unittest.main()
