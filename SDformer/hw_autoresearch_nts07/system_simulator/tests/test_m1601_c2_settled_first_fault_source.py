#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "dc_handoff/scripts/check_m1601_c2_settled_first_fault_source.py"
SPEC = importlib.util.spec_from_file_location("m1601_source", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1601SourceTest(unittest.TestCase):
    def test_frozen_source_passes(self) -> None:
        value = M.check()
        self.assertEqual(value["status"], "PASS_SOURCE_ONLY__DIFFERENT_AUTHOR_HAMMER_REQUIRED")
        self.assertEqual(value["settle_count"], 1)
        self.assertEqual(value["vcs_compiles"], 0)
        self.assertFalse(value["claim"])

    def test_normalization_rejects_missing_or_duplicate_settle(self) -> None:
        text = M.NEW_TB.read_text(encoding="utf-8")
        with self.assertRaises(M.Failure):
            M.normalized_new_tb(text.replace("            #1ps;\n", ""))
        duplicate = M.normalized_new_tb(
            text.replace("            #1ps;\n", "            #1ps;\n            #1ps;\n"))
        self.assertNotEqual(duplicate, M.OLD_TB.read_text(encoding="utf-8"))

    def test_normalization_rejects_semantic_drift(self) -> None:
        text = M.NEW_TB.read_text(encoding="utf-8")
        normalized = M.normalized_new_tb(text.replace("header_raw_beat_count = 6'd4;",
                                                       "header_raw_beat_count = 6'd5;"))
        self.assertNotEqual(normalized, M.OLD_TB.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
