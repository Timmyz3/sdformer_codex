#!/usr/bin/env python3
"""三架构真实回放汇总器核心计算测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_equal96_real_trace import destination_profile


class RealTraceSummaryTest(unittest.TestCase):
    def test_destination_profile_scales_supertile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "term_destination_counts.memh").write_text(
                "01\n04\n08\n", encoding="utf-8"
            )
            row = destination_profile({
                "vector_dir": str(root),
                "logical_supertiles": 2,
            })
            self.assertEqual(row["logical_terms"], 6)
            self.assertEqual(row["destinations"], 26)
            self.assertEqual(row["event_beats"], 8)
            self.assertEqual(row["term_one_destination"], 2)
            self.assertEqual(row["term_ge8_destinations"], 2)
            self.assertEqual(row["max_destinations_per_term"], 8)


if __name__ == "__main__":
    unittest.main()
