#!/usr/bin/env python3
"""Source-only tests for M832 directory-FD-bound attempt consumption."""

import importlib.util
import os
from pathlib import Path
import shutil
import socket
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HERE.parent / "scripts/execute_m832_m828_decoder_directory_bound_consumption.py"
RUNNER = HERE.parent / "scripts/run_m832_m785_decoder_physical_residency_one_shot.sh"
CANDIDATE = HW / "contracts/m832_m785_decoder_directory_bound_consumption_candidate_r1_20260829.json"


def load_driver():
    spec = importlib.util.spec_from_file_location("m832_source_test", str(DRIVER))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M832 driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M832 = load_driver()


def source_receipt():
    return {"schema": "source_test", "status": M832.PARENT_ATTEMPT_STATUS}


class M832DirectoryBoundConsumptionTests(unittest.TestCase):
    def test_self_test(self):
        value = M832.self_test()
        self.assertEqual(
            value["status"],
            "PASS_M832_DIRECTORY_BOUND_CONSUMPTION_SYNTHETIC_SELF_TEST")
        self.assertEqual(value["scheduled_rows"], 0)
        self.assertFalse(value["formal_attempt_created"])
        self.assertIsNone(value["production_cycles"])

    def test_matching_regular_directory_fifo_and_socket_rejected_no_clobber(self):
        makers = {
            "regular": lambda root, path: path.write_text(
                "KEEP", encoding="utf-8"),
            "directory": lambda root, path: path.mkdir(),
            "fifo": lambda root, path: os.mkfifo(str(path)),
        }
        for kind, maker in makers.items():
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory(
                        prefix="m832_type_") as directory:
                    root = Path(directory)
                    artifact = root / (M832.CANONICAL_FAILURE_PREFIX + kind)
                    maker(root, artifact)
                    before = artifact.lstat()
                    with self.assertRaises(M832.Failure):
                        M832.atomic_guard_and_consume(
                            root, M832.GUARDED_PREFIXES,
                            "attempt.stage.type", "attempt", source_receipt())
                    after = artifact.lstat()
                    self.assertEqual((before.st_dev, before.st_ino,
                                      before.st_mode),
                                     (after.st_dev, after.st_ino,
                                      after.st_mode))
        root = Path(tempfile.mkdtemp(prefix="x", dir="/tmp"))
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            artifact = root / (M832.CANONICAL_FAILURE_PREFIX + "s")
            channel.bind(str(artifact))
            before = artifact.lstat()
            with self.assertRaises(M832.Failure):
                M832.atomic_guard_and_consume(
                    root, M832.GUARDED_PREFIXES,
                    "attempt.stage.socket", "attempt", source_receipt())
            after = artifact.lstat()
            self.assertEqual((before.st_ino, before.st_mode),
                             (after.st_ino, after.st_mode))
        finally:
            channel.close()
            shutil.rmtree(str(root))

    def test_matching_symlink_and_dangling_rejected_no_clobber(self):
        for dangling in (False, True):
            with self.subTest(dangling=dangling):
                with tempfile.TemporaryDirectory(
                        prefix="m832_link_") as directory:
                    root = Path(directory)
                    target = root / "target"
                    if not dangling:
                        target.write_text("KEEP", encoding="utf-8")
                    link = root / (M832.M828_FAILURE_PREFIX +
                                   ("dangling" if dangling else "symlink"))
                    link.symlink_to(target)
                    with self.assertRaises(M832.Failure):
                        M832.atomic_guard_and_consume(
                            root, M832.GUARDED_PREFIXES,
                            "attempt.stage.link", "attempt", source_receipt())
                    self.assertTrue(link.is_symlink())
                    if not dangling:
                        self.assertEqual(target.read_text(encoding="utf-8"),
                                         "KEEP")

    def test_persistent_concurrent_injection_between_scans_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m832_inject_") as directory:
            root = Path(directory)
            artifact = root / (M832.INHERITED_FAILURE_PREFIX + "concurrent")
            original = M832._after_first_scan_hook
            fired = [False]

            def inject():
                if not fired[0]:
                    fired[0] = True
                    artifact.symlink_to(root / "absent")

            M832._after_first_scan_hook = inject
            try:
                with self.assertRaises(M832.Failure):
                    M832.atomic_guard_and_consume(
                        root, M832.GUARDED_PREFIXES,
                        "attempt.stage.inject", "attempt", source_receipt())
            finally:
                M832._after_first_scan_hook = original
            self.assertTrue(fired[0])
            self.assertTrue(artifact.is_symlink())
            self.assertFalse((root / "attempt").exists())

    def test_directory_swap_at_scan_prestage_and_poststage_rejected(self):
        hook_names = ("_after_first_scan_hook", "_before_stage_mkdir_hook",
                      "_after_stage_mkdir_hook")
        for hook_name in hook_names:
            with self.subTest(hook=hook_name):
                with tempfile.TemporaryDirectory(
                        prefix="m832_swap_") as directory:
                    top = Path(directory)
                    current = top / "results"
                    old = top / "old"
                    current.mkdir()
                    artifact = current / (
                        M832.CANONICAL_FAILURE_PREFIX + "replacement")
                    original = getattr(M832, hook_name)
                    fired = [False]

                    def swap():
                        if not fired[0]:
                            fired[0] = True
                            current.rename(old)
                            current.mkdir()
                            (current / artifact.name).write_text(
                                "KEEP", encoding="utf-8")

                    setattr(M832, hook_name, swap)
                    try:
                        with self.assertRaises(M832.Failure):
                            M832.atomic_guard_and_consume(
                                current, M832.GUARDED_PREFIXES,
                                "attempt.stage.swap", "attempt",
                                source_receipt())
                    finally:
                        setattr(M832, hook_name, original)
                    self.assertTrue(fired[0])
                    self.assertEqual((current / artifact.name).read_text(
                        encoding="utf-8"), "KEEP")
                    self.assertFalse((old / "attempt").exists())
                    self.assertFalse((old / "attempt.stage.swap").exists())

    def test_stage_and_attempt_collision_rejected_without_clobber(self):
        for collision in ("attempt.stage.collision", "attempt"):
            with self.subTest(collision=collision):
                with tempfile.TemporaryDirectory(
                        prefix="m832_collision_") as directory:
                    root = Path(directory)
                    item = root / collision
                    item.mkdir()
                    (item / "marker").write_text("KEEP", encoding="utf-8")
                    with self.assertRaises(M832.Failure):
                        M832.atomic_guard_and_consume(
                            root, M832.GUARDED_PREFIXES,
                            "attempt.stage.collision", "attempt",
                            source_receipt())
                    self.assertEqual((item / "marker").read_text(
                        encoding="utf-8"), "KEEP")

    def test_wrong_prefix_survives_clean_consumption(self):
        with tempfile.TemporaryDirectory(prefix="m832_wrong_") as directory:
            root = Path(directory)
            wrong = [
                root / ("x" + M832.CANONICAL_FAILURE_PREFIX + "not-prefix"),
                root / M832.CANONICAL_FAILURE_PREFIX[:-1],
                root / "m832_m785_h67_decoder_physical_residency_cycles_r1_20260829.failed_or_complete.x",
            ]
            for index, path in enumerate(wrong):
                path.write_text(str(index), encoding="utf-8")
            value = M832.atomic_guard_and_consume(
                root, M832.GUARDED_PREFIXES, "attempt.stage.clean",
                "attempt", source_receipt())
            self.assertEqual(value["status"],
                             "PASS_M832_DIRECTORY_FD_BOUND_ATTEMPT_CONSUMED")
            self.assertTrue((root / "attempt").is_dir())
            self.assertEqual([p.read_text(encoding="utf-8") for p in wrong],
                             ["0", "1", "2"])

    def test_runner_uses_one_atomic_helper_after_release_and_resource_gate(self):
        text = RUNNER.read_text(encoding="utf-8")
        release = text.index("--validate-release-preflight")
        resource = text.index("m832_free_kib=")
        consume = text.index("--guard-and-consume-attempt")
        postcheck = text.index('m832_started=1')
        production = text.index("--run-production")
        self.assertLess(release, resource)
        self.assertLess(resource, consume)
        self.assertLess(consume, postcheck)
        self.assertLess(postcheck, production)
        self.assertNotIn('mkdir "${m832_attempt_stage}"', text)
        self.assertNotIn('mv -T --no-clobber -- "${m832_attempt_stage}"',
                         text)
        self.assertEqual(text.count("--guard-and-consume-attempt"), 1)

    def test_strict_json_duplicate_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m832_json_") as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
            with self.assertRaises(M832.Failure):
                M832.strict_json(duplicate)
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
            with self.assertRaises(M832.Failure):
                M832.strict_json(nonfinite)

    def test_m831_remains_negative_and_parent_is_exact(self):
        review = M832._verify_m831_negative()
        self.assertEqual(
            review["status"],
            "NO_GO_M828_SOURCE_CANDIDATE__P1_1__DIRECTORY_BINDING_TOCTOU_REPAIR_REQUIRED")
        self.assertFalse(review["true_release_authorized"])
        self.assertEqual(M832.sha256(M832.M828_PATH), M832.M828_SHA256)
        self.assertEqual(M832.sha256(M832.M828_CANDIDATE),
                         M832.M828_CANDIDATE_SHA256)

    def test_candidate_validation_and_formal_absence(self):
        value = M832.validate_candidate(CANDIDATE)
        self.assertEqual(
            value["status"],
            "PASS_M832_DIRECTORY_BOUND_CONSUMPTION_SOURCE_CANDIDATE__NO_PRODUCTION_RUN")
        self.assertIsNone(value["production_cycles"])
        candidate = M832.strict_json(CANDIDATE)
        result, attempt, release = M832._canonical_paths(candidate)
        self.assertFalse(result.exists() or result.is_symlink())
        self.assertFalse(attempt.exists() or attempt.is_symlink())
        self.assertFalse(release.exists() or release.is_symlink())

    def test_clean_consumption_then_exact_parent_zero_row_traversal(self):
        value = M832.preproduction_traversal_test()
        self.assertEqual(
            value["status"],
            "PASS_M832_DIRECTORY_BOUND_CLEAN_PARENT_PREPRODUCTION_TRAVERSAL")
        self.assertTrue(value["entered_exact_frozen_m828"])
        self.assertTrue(value["entered_exact_frozen_m819"])
        self.assertTrue(value["entered_exact_frozen_m809"])
        self.assertEqual(value["stopped_at"], "M809_OUTPUT_MKDIR")
        self.assertEqual(value["scheduled_rows"], 0)
        self.assertFalse(value["output_exists"])
        self.assertFalse(value["attempt_receipt_identity_drift"])
        self.assertTrue(value["delegated_validators_restored"])


if __name__ == "__main__":
    unittest.main()
