#!/usr/bin/env python3
"""Tests for H67 real-weight projection2 log parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import report_h67_real_weight_projection2_rtl as report


class ReportTests(unittest.TestCase):
    def test_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.log"
            path.write_text(
                "REALW_ROW row=0 stage=0 block=0 head=0 expected0=1 expected1=2 fixed0=1 fixed1=2 rqtb0=1 rqtb1=3\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                report.parse_log(path)


if __name__ == "__main__":
    unittest.main()
