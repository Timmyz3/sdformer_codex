#!/usr/bin/env python3
"""Source-only tests for M836 publication-boundary repair."""

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
DRIVER = HERE.parent / "scripts/execute_m836_m832_decoder_publication_boundary_repair.py"
RUNNER = HERE.parent / "scripts/run_m836_m785_decoder_physical_residency_one_shot.sh"
CANDIDATE = HW / "contracts/m836_m785_decoder_publication_boundary_repair_candidate_r1_20260829.json"


def load_driver():
    spec = importlib.util.spec_from_file_location("m836_source_test", str(DRIVER))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M836 driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M836 = load_driver()


def source_receipt(stage_name):
    return {
        "schema": "source_test",
        "status": M836.PARENT_ATTEMPT_STATUS,
        "publication_nonce":
            hashlib.sha256(stage_name.encode("utf-8")).hexdigest(),
    }


class M836PublicationBoundaryRepairTests(unittest.TestCase):
    def test_self_test(self):
        value = M836.self_test()
        self.assertEqual(
            value["status"],
            "PASS_M836_PUBLICATION_BOUNDARY_REPAIR_SYNTHETIC_SELF_TEST")
        self.assertEqual(value["scheduled_rows"], 0)
        self.assertFalse(value["formal_attempt_created"])
        self.assertIsNone(value["production_cycles"])

    def test_matching_regular_directory_fifo_socket_rejected_no_clobber(self):
        makers = {
            "regular": lambda path: path.write_text("KEEP", encoding="utf-8"),
            "directory": lambda path: path.mkdir(),
            "fifo": lambda path: os.mkfifo(str(path)),
        }
        for kind, maker in makers.items():
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory(prefix="m836_type_") as d:
                    root = Path(d)
                    artifact = root / (M836.CANONICAL_FAILURE_PREFIX + kind)
                    maker(artifact)
                    before = artifact.lstat()
                    stage = "attempt.stage.type"
                    with self.assertRaises(M836.Failure):
                        M836.atomic_guard_and_consume(
                            root, M836.GUARDED_PREFIXES, stage, "attempt",
                            source_receipt(stage))
                    after = artifact.lstat()
                    self.assertEqual((before.st_dev, before.st_ino,
                                      before.st_mode),
                                     (after.st_dev, after.st_ino,
                                      after.st_mode))
        root = Path(tempfile.mkdtemp(prefix="x", dir="/tmp"))
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            artifact = root / (M836.CANONICAL_FAILURE_PREFIX + "socket")
            channel.bind(str(artifact))
            before = artifact.lstat()
            stage = "attempt.stage.socket"
            with self.assertRaises(M836.Failure):
                M836.atomic_guard_and_consume(
                    root, M836.GUARDED_PREFIXES, stage, "attempt",
                    source_receipt(stage))
            after = artifact.lstat()
            self.assertEqual((before.st_ino, before.st_mode),
                             (after.st_ino, after.st_mode))
        finally:
            channel.close()
            shutil.rmtree(str(root))

    def test_matching_symlink_and_dangling_rejected_no_clobber(self):
        for dangling in (False, True):
            with self.subTest(dangling=dangling):
                with tempfile.TemporaryDirectory(prefix="m836_link_") as d:
                    root = Path(d)
                    target = root / "target"
                    if not dangling:
                        target.write_text("KEEP", encoding="utf-8")
                    link = root / (M836.M832_FAILURE_PREFIX +
                                   ("dangling" if dangling else "symlink"))
                    link.symlink_to(target)
                    stage = "attempt.stage.link"
                    with self.assertRaises(M836.Failure):
                        M836.atomic_guard_and_consume(
                            root, M836.GUARDED_PREFIXES, stage, "attempt",
                            source_receipt(stage))
                    self.assertTrue(link.is_symlink())
                    if not dangling:
                        self.assertEqual(target.read_text(encoding="utf-8"),
                                         "KEEP")

    def test_persistent_injections_poststage_and_prepublish_reject_cleanup(self):
        for hook_name in ("_after_first_scan_hook", "_after_stage_mkdir_hook",
                          "_before_attempt_publish_hook"):
            with self.subTest(hook=hook_name):
                with tempfile.TemporaryDirectory(prefix="m836_inject_") as d:
                    root = Path(d)
                    artifact = root / (M836.INHERITED_FAILURE_PREFIX +
                                       hook_name)
                    original = getattr(M836, hook_name)

                    def inject():
                        if not artifact.exists():
                            artifact.write_text("KEEP", encoding="utf-8")

                    setattr(M836, hook_name, inject)
                    stage = "attempt.stage.inject"
                    try:
                        with self.assertRaises(M836.Failure):
                            M836.atomic_guard_and_consume(
                                root, M836.GUARDED_PREFIXES, stage, "attempt",
                                source_receipt(stage))
                    finally:
                        setattr(M836, hook_name, original)
                    self.assertEqual(artifact.read_text(encoding="utf-8"),
                                     "KEEP")
                    self.assertFalse((root / stage).exists())
                    self.assertFalse((root / "attempt").exists())

    def test_stage_and_attempt_collisions_rejected_no_clobber(self):
        for collision in ("attempt.stage.collision", "attempt"):
            with self.subTest(collision=collision):
                with tempfile.TemporaryDirectory(prefix="m836_collision_") as d:
                    root = Path(d)
                    item = root / collision
                    item.mkdir()
                    (item / "marker").write_text("KEEP", encoding="utf-8")
                    stage = "attempt.stage.collision"
                    with self.assertRaises(M836.Failure):
                        M836.atomic_guard_and_consume(
                            root, M836.GUARDED_PREFIXES, stage, "attempt",
                            source_receipt(stage))
                    self.assertEqual((item / "marker").read_text(
                        encoding="utf-8"), "KEEP")

    def test_wrong_prefix_and_transient_boundary_absent_are_accepted(self):
        with tempfile.TemporaryDirectory(prefix="m836_wrong_") as d:
            root = Path(d)
            wrong = root / ("x" + M836.CANONICAL_FAILURE_PREFIX + "wrong")
            wrong.write_text("KEEP", encoding="utf-8")
            transient = root / (M836.M828_FAILURE_PREFIX + "transient")
            original = M836._after_first_scan_hook

            def flash():
                transient.write_text("FLASH", encoding="utf-8")
                transient.unlink()

            M836._after_first_scan_hook = flash
            stage = "attempt.stage.transient"
            try:
                value = M836.atomic_guard_and_consume(
                    root, M836.GUARDED_PREFIXES, stage, "attempt",
                    source_receipt(stage))
            finally:
                M836._after_first_scan_hook = original
            self.assertEqual(
                value["status"],
                "PASS_M836_PUBLICATION_BOUNDARY_CLOSED_ATTEMPT_CONSUMED")
            self.assertEqual(wrong.read_text(encoding="utf-8"), "KEEP")
            self.assertFalse(transient.exists())

    def test_prepublish_content_change_rejected_and_owned_stage_removed(self):
        value = M836._prepublish_content_attack()
        self.assertTrue(value["rejected"])
        self.assertTrue(value["recorded_inode_stage_removed"])
        self.assertTrue(value["canonical_attempt_absent"])

    def test_after_final_rebind_directory_swap_rejects_and_rolls_back(self):
        value = M836._directory_swap_attack("_after_final_rebind_hook")
        self.assertTrue(value["rejected"])
        self.assertTrue(value["self_publication_rolled_back"])
        self.assertTrue(value["replacement_unchanged"])

    def test_postpublish_directory_swap_rejects_and_rolls_back(self):
        value = M836._directory_swap_attack("_after_attempt_publish_hook")
        self.assertTrue(value["rejected"])
        self.assertTrue(value["self_publication_rolled_back"])
        self.assertTrue(value["replacement_unchanged"])

    def test_runner_has_one_helper_after_preflight_resource_before_production(self):
        text = RUNNER.read_text(encoding="utf-8")
        release = text.index("--validate-release-preflight")
        resource = text.index("m836_free_kib=")
        consume = text.index("--guard-and-consume-attempt")
        started = text.index("m836_started=1")
        production = text.index("--run-production")
        self.assertLess(release, resource)
        self.assertLess(resource, consume)
        self.assertLess(consume, started)
        self.assertLess(started, production)
        self.assertEqual(text.count("--guard-and-consume-attempt"), 1)
        self.assertNotIn('mkdir "${m836_attempt_stage}"', text)
        self.assertNotIn('mv -T --no-clobber -- "${m836_attempt_stage}"',
                         text)

    def test_candidate_m835_negative_strict_json_and_formal_absence(self):
        with tempfile.TemporaryDirectory(prefix="m836_json_") as d:
            root = Path(d)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
            with self.assertRaises(M836.Failure):
                M836.strict_json(duplicate)
            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"x":NaN}\n', encoding="utf-8")
            with self.assertRaises(M836.Failure):
                M836.strict_json(nonfinite)
        review = M836._verify_m835_negative()
        self.assertEqual(
            review["status"],
            "NO_GO_M832_SOURCE_CANDIDATE__P1_1__PUBLICATION_BOUNDARY_REPAIR_REQUIRED")
        value = M836.validate_candidate(CANDIDATE)
        self.assertEqual(
            value["status"],
            "PASS_M836_PUBLICATION_BOUNDARY_REPAIR_SOURCE_CANDIDATE__NO_PRODUCTION_RUN")
        candidate = M836.strict_json(CANDIDATE)
        result, attempt, release = M836._canonical_paths(candidate)
        self.assertFalse(result.exists() or result.is_symlink())
        self.assertFalse(attempt.exists() or attempt.is_symlink())
        self.assertFalse(release.exists() or release.is_symlink())

    def test_clean_exact_parent_zero_row_traversal(self):
        value = M836.preproduction_traversal_test()
        self.assertEqual(
            value["status"],
            "PASS_M836_PUBLICATION_BOUNDARY_CLEAN_PARENT_PREPRODUCTION_TRAVERSAL")
        self.assertTrue(value["entered_exact_m832"])
        self.assertTrue(value["entered_exact_m828"])
        self.assertTrue(value["entered_exact_m819"])
        self.assertTrue(value["entered_exact_m809"])
        self.assertEqual(value["stopped_at"], "M809_OUTPUT_MKDIR")
        self.assertEqual(value["scheduled_rows"], 0)
        self.assertFalse(value["output_exists"])
        self.assertTrue(value["delegated_validators_restored"])


if __name__ == "__main__":
    unittest.main()
