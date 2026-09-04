#!/usr/bin/env python3
"""Unit tests for source-owned Q-silent cycle model helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_local5_source_owned_qsilent_cycles import parse_log


class SourceOwnedCycleModelTest(unittest.TestCase):
    def test_parser_requires_contiguous_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtl.log"
            path.write_text(
                "GROUP group=0 cycles=100 score_rows=450 qsilent_rows=400 "
                "identk_rows=10\n"
                "GROUP group=1 cycles=120 score_rows=450 qsilent_rows=300 "
                "identk_rows=20\n",
                encoding="utf-8",
            )
            rows = parse_log(path)
            self.assertEqual([row["cycles"] for row in rows], [100, 120])

    def test_parser_rejects_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtl.log"
            path.write_text(
                "GROUP group=1 cycles=100 score_rows=450 qsilent_rows=400 "
                "identk_rows=10\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                parse_log(path)


if __name__ == "__main__":
    unittest.main()
