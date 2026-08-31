#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m851_c2_r24_recursive_seal_guard as guard  # noqa: E402


class RecursiveSealTest(unittest.TestCase):
    def populate_work(self, work):
        for index, relative in enumerate(guard.r848.WHITELIST):
            path = work / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((relative + "\n").encode("utf-8") +
                             bytes([index]))
        tool = work / "attack" / "simv.daidir"
        tool.mkdir()
        (tool / "archive.so").write_bytes(b"private tool output\n")
        os.symlink("../simv.daidir/archive.so",
                   str(work / "attack" / "tool_archive.so"))

    def sealed_stage(self, root):
        work = root / "work"
        work.mkdir()
        self.populate_work(work)
        stage = root / "stage"
        guard.r848.stage_result_whitelist(work, stage)
        guard.base.seal_directory(stage)
        return work, stage

    def test_complete_nested_stage_verify_and_publish(self):
        with tempfile.TemporaryDirectory(prefix="m851_full_pipeline.") as raw:
            root = Path(raw)
            work, stage = self.sealed_stage(root)
            before = guard.verify_recursive_sealed_directory(
                stage, guard.RESULT_MEMBERS)
            self.assertEqual(before["member_count"], 15)
            self.assertEqual(before["file_count_including_seals"], 17)
            self.assertEqual(before["directory_count"], 2)
            # The inherited API must remain strict-flat and therefore reject
            # this nested exact set; R24 must not weaken it.
            with self.assertRaises(guard.base.Failure):
                guard.base.verify_sealed_directory(stage,
                                                   set(guard.RESULT_MEMBERS))
            destination = root / "canonical"
            after = guard.publish_recursive_noreplace(
                stage, destination, guard.RESULT_MEMBERS)
            self.assertEqual(before, after)
            self.assertFalse(stage.exists())
            self.assertTrue(work.is_dir())
            self.assertTrue((work / "attack" / "tool_archive.so").is_symlink())
            actual = {p.relative_to(destination).as_posix()
                      for p in destination.rglob("*") if p.is_file()}
            self.assertEqual(actual, set(guard.RESULT_MEMBERS))
            self.assertFalse(any(p.is_symlink()
                                 for p in destination.rglob("*")))

    def test_extra_empty_directory_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m851_extra_dir.") as raw:
            root = Path(raw)
            _, stage = self.sealed_stage(root)
            (stage / "extra_empty").mkdir()
            with self.assertRaises(guard.base.Failure):
                guard.verify_recursive_sealed_directory(
                    stage, guard.RESULT_MEMBERS)

    def test_recursive_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m851_nested_link.") as raw:
            root = Path(raw)
            _, stage = self.sealed_stage(root)
            os.symlink("compile.log", str(stage / "attack" / "extra_link"))
            with self.assertRaises((guard.base.Failure, OSError)):
                guard.verify_recursive_sealed_directory(
                    stage, guard.RESULT_MEMBERS)

    def test_manifest_or_payload_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m851_mutation.") as raw:
            root = Path(raw)
            _, stage = self.sealed_stage(root)
            path = stage / "equalbw" / "sim.log"
            data = path.read_bytes()
            path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            with self.assertRaises(guard.base.Failure):
                guard.verify_recursive_sealed_directory(
                    stage, guard.RESULT_MEMBERS)

    def test_destination_collision_is_no_clobber(self):
        with tempfile.TemporaryDirectory(prefix="m851_collision.") as raw:
            root = Path(raw)
            _, stage = self.sealed_stage(root)
            destination = root / "canonical"
            destination.mkdir()
            marker = destination / "attacker"
            marker.write_text("preserve\n", encoding="utf-8")
            with self.assertRaises(guard.base.Failure):
                guard.publish_recursive_noreplace(
                    stage, destination, guard.RESULT_MEMBERS)
            self.assertEqual(marker.read_text(encoding="utf-8"),
                             "preserve\n")
            self.assertTrue(stage.is_dir())


if __name__ == "__main__":
    unittest.main()
