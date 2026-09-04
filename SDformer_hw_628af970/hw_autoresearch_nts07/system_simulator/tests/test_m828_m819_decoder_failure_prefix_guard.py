#!/usr/bin/env python3
"""Source-only tests for the M828 stable failure-prefix guard."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HERE.parent / "scripts/execute_m828_m819_decoder_failure_prefix_guard.py"
RUNNER = HERE.parent / "scripts/run_m828_m785_decoder_physical_residency_one_shot.sh"
CANDIDATE = HW / "contracts/m828_m785_decoder_failure_prefix_guard_candidate_r1_20260829.json"


def load_driver():
    spec = importlib.util.spec_from_file_location("m828_source_test", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M828 driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M828 = load_driver()


class M828FailurePrefixGuardTests(unittest.TestCase):
    def test_self_test(self):
        value = M828.self_test()
        self.assertEqual(value["status"],
                         "PASS_M828_FAILURE_PREFIX_GUARD_SYNTHETIC_SELF_TEST")
        self.assertEqual(value["scheduled_rows"], 0)
        self.assertFalse(value["formal_attempt_created"])
        self.assertIsNone(value["production_cycles"])

    def test_preexisting_regular_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m828_regular_") as directory:
            root = Path(directory)
            (root / (M828.CANONICAL_FAILURE_PREFIX + "regular")).write_text(
                "evidence", encoding="utf-8")
            with self.assertRaises(M828.Failure):
                M828.guard_failure_prefix_absence(root, M828.GUARDED_PREFIXES)

    def test_preexisting_directory_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m828_directory_") as directory:
            root = Path(directory)
            (root / (M828.INHERITED_FAILURE_PREFIX + "directory")).mkdir()
            with self.assertRaises(M828.Failure):
                M828.guard_failure_prefix_absence(root, M828.GUARDED_PREFIXES)

    def test_preexisting_symlink_and_dangling_symlink_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m828_symlink_") as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("KEEP", encoding="utf-8")
            link = root / (M828.CANONICAL_FAILURE_PREFIX + "symlink")
            link.symlink_to(target)
            with self.assertRaises(M828.Failure):
                M828.guard_failure_prefix_absence(root, M828.GUARDED_PREFIXES)
            self.assertEqual(target.read_text(encoding="utf-8"), "KEEP")
        with tempfile.TemporaryDirectory(prefix="m828_dangling_") as directory:
            root = Path(directory)
            link = root / (M828.INHERITED_FAILURE_PREFIX + "dangling")
            link.symlink_to(root / "absent")
            with self.assertRaises(M828.Failure):
                M828.guard_failure_prefix_absence(root, M828.GUARDED_PREFIXES)
            self.assertTrue(link.is_symlink())

    def test_concurrent_injection_between_samples_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m828_concurrent_") as directory:
            root = Path(directory)
            injected = root / (M828.CANONICAL_FAILURE_PREFIX + "concurrent")
            original = M828._guard_sample_yield
            fired = [False]

            def inject_once():
                if not fired[0]:
                    fired[0] = True
                    injected.symlink_to(root / "absent")

            M828._guard_sample_yield = inject_once
            try:
                with self.assertRaises(M828.Failure):
                    M828.guard_failure_prefix_absence(
                        root, M828.GUARDED_PREFIXES)
            finally:
                M828._guard_sample_yield = original
            self.assertTrue(fired[0])
            self.assertTrue(injected.is_symlink())

    def test_wrong_prefix_does_not_kill_clean_guard(self):
        with tempfile.TemporaryDirectory(prefix="m828_wrong_prefix_") as directory:
            root = Path(directory)
            names = [
                "x" + M828.CANONICAL_FAILURE_PREFIX + "not-a-prefix",
                M828.CANONICAL_FAILURE_PREFIX[:-1],
                "m828_m785_h67_decoder_physical_residency_cycles_r1_20260829.failed_or_complete.x",
            ]
            for index, name in enumerate(names):
                (root / name).write_text(str(index), encoding="utf-8")
            value = M828.guard_failure_prefix_absence(
                root, M828.GUARDED_PREFIXES)
            self.assertEqual(value["status"],
                             "PASS_M828_STABLE_FAILURE_PREFIX_ABSENCE")
            self.assertEqual(value["matches"], [])

    def test_results_directory_symlink_rejected_without_target_change(self):
        with tempfile.TemporaryDirectory(prefix="m828_dir_symlink_") as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            alias = root / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(M828.Failure):
                M828.guard_failure_prefix_absence(alias,
                                                   M828.GUARDED_PREFIXES)
            self.assertEqual(list(target.iterdir()), [])

    def test_clean_guard_then_exact_parent_zero_row_traversal(self):
        value = M828.preproduction_traversal_test()
        self.assertEqual(value["status"],
                         "PASS_M828_GUARD_CLEAN_PARENT_PREPRODUCTION_TRAVERSAL")
        self.assertTrue(value["entered_exact_frozen_m819"])
        self.assertTrue(value["entered_exact_frozen_m809"])
        self.assertEqual(value["stopped_at"], "M809_OUTPUT_MKDIR")
        self.assertEqual(value["scheduled_rows"], 0)
        self.assertFalse(value["output_exists"])
        self.assertFalse(value["attempt_receipt_identity_drift"])
        self.assertTrue(value["delegated_validators_restored"])

    def test_runner_guard_is_final_operation_before_attempt_mkdir(self):
        text = RUNNER.read_text(encoding="utf-8")
        release = text.index("--validate-release-preflight")
        resource = text.index("m828_free_kib=")
        guard = text.index("--guard-failure-prefix-absence")
        attempt_mkdir = text.index('mkdir "${m828_attempt_stage}"')
        publish = text.index(
            'mv -T --no-clobber -- "${m828_attempt_stage}" "${m828_attempt}"')
        production = text.index("--run-production")
        self.assertLess(release, resource)
        self.assertLess(resource, guard)
        self.assertLess(guard, attempt_mkdir)
        self.assertLess(attempt_mkdir, publish)
        self.assertLess(publish, production)
        between = text[guard:attempt_mkdir]
        self.assertNotIn("mkdir ", between)
        self.assertNotIn("--run-production", between)
        self.assertEqual(text.count("--guard-failure-prefix-absence"), 1)

    def test_strict_json_duplicate_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m828_json_") as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
            with self.assertRaises(M828.Failure):
                M828.strict_json(duplicate)
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
            with self.assertRaises(M828.Failure):
                M828.strict_json(nonfinite)

    def test_m825_remains_negative_and_parent_is_exact(self):
        review = M828._verify_m825_negative()
        self.assertEqual(review["status"],
                         "NO_GO_M819_TRUE_RELEASE__P1_1__ADDITIVE_FAILURE_PREFIX_PRECONSUMPTION_GATE_REQUIRED")
        self.assertFalse(review["production_replay_authorized"])
        self.assertEqual(M828.sha256(M828.M819_PATH), M828.M819_SHA256)
        self.assertEqual(M828.sha256(M828.M819_CANDIDATE),
                         M828.M819_CANDIDATE_SHA256)

    def test_candidate_validation_and_formal_absence(self):
        value = M828.validate_candidate(CANDIDATE)
        self.assertEqual(value["status"],
                         "PASS_M828_FAILURE_PREFIX_GUARD_SOURCE_CANDIDATE__NO_PRODUCTION_RUN")
        self.assertIsNone(value["production_cycles"])
        candidate = M828.strict_json(CANDIDATE)
        result, attempt, release = M828._canonical_paths(candidate)
        self.assertFalse(result.exists() or result.is_symlink())
        self.assertFalse(attempt.exists() or attempt.is_symlink())
        self.assertFalse(release.exists() or release.is_symlink())


if __name__ == "__main__":
    unittest.main()
