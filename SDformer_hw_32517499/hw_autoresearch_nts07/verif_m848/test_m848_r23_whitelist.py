#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m848_c2_r23_whitelist_guard as guard  # noqa: E402


class WhitelistTest(unittest.TestCase):
    def populate(self, root):
        for index, relative in enumerate(guard.WHITELIST):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((relative + "\n").encode("utf-8") +
                             bytes([index]))

    def test_tool_symlinks_and_extras_stay_out_of_stage(self):
        with tempfile.TemporaryDirectory(prefix="m848_whitelist.") as raw:
            root = Path(raw)
            work = root / "work"
            work.mkdir()
            self.populate(work)
            (work / "attack" / "simv.daidir").mkdir()
            target = work / "attack" / "simv.daidir" / "archive.so"
            target.write_bytes(b"tool output\n")
            os.symlink("../simv.daidir/archive.so",
                       str(work / "attack" / "tool_archive.so"))
            (work / "unlisted.log").write_text("exclude\n", encoding="utf-8")
            stage = root / "stage"
            result = guard.stage_result_whitelist(work, stage)
            self.assertEqual(result["member_count"], 15)
            self.assertEqual(result["symlinks"], 0)
            actual = {p.relative_to(stage).as_posix()
                      for p in stage.rglob("*") if p.is_file()}
            self.assertEqual(actual, set(guard.WHITELIST))
            self.assertFalse(any(p.is_symlink() for p in stage.rglob("*")))

    def test_whitelisted_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m848_nofollow.") as raw:
            root = Path(raw)
            work = root / "work"
            work.mkdir()
            self.populate(work)
            victim = work / "attack" / "compile.log"
            victim.unlink()
            (work / "real.log").write_text("not admitted\n", encoding="utf-8")
            os.symlink("../real.log", str(victim))
            with self.assertRaises((guard.base.Failure, OSError)):
                guard.stage_result_whitelist(work, root / "stage")

    def test_exact_stage_rejects_preexistence(self):
        with tempfile.TemporaryDirectory(prefix="m848_collision.") as raw:
            root = Path(raw)
            work = root / "work"
            work.mkdir()
            self.populate(work)
            stage = root / "stage"
            stage.mkdir()
            (stage / "attacker").write_text("preserve\n", encoding="utf-8")
            with self.assertRaises(guard.base.Failure):
                guard.stage_result_whitelist(work, stage)
            self.assertEqual((stage / "attacker").read_text(encoding="utf-8"),
                             "preserve\n")


if __name__ == "__main__":
    unittest.main()
