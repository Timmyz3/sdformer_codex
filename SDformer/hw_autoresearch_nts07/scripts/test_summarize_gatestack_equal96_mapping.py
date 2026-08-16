#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_equal96_mapping import build_report


class Equal96MappingSummaryTest(unittest.TestCase):
    def test_shared_front_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, area, cells, memories in (
                ("central96", 75.0, 750, 3),
                ("three_independent32", 100.0, 1000, 9),
            ):
                (root / f"{name}.log").write_text(
                    f"   $mem_v2 {memories}\n"
                    f"   Number of cells: {cells}\n"
                    f"   Chip area for module '\\top': {area:.3f}\n"
                )
            report = build_report(root)
            central, independent = report["rows"]
            self.assertEqual(central["product_lanes"], 96)
            self.assertEqual(independent["decoder_instances"], 3)
            self.assertAlmostEqual(
                central["logic_area_reduction_vs_independent"], 0.25)
            self.assertEqual([central["mem_v2"], independent["mem_v2"]],
                             [3, 9])


if __name__ == "__main__":
    unittest.main()
