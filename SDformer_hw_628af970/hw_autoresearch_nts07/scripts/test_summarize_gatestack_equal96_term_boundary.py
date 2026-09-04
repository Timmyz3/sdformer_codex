#!/usr/bin/env python3
"""同term边界三架构映射汇总器单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_equal96_term_boundary import build_report


class EqualTermBoundarySummaryTest(unittest.TestCase):
    def test_three_way_reductions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            mapping = out / "mapping"
            mapping.mkdir()
            for name, area, cells, memories in (
                ("central96_term", 100.0, 1000, 3),
                ("independent32x3_term", 150.0, 1500, 9),
                ("dctf96_term", 90.0, 900, 11),
            ):
                (mapping / f"{name}.log").write_text(
                    "Number of processes: 0\n"
                    f"Number of cells: {cells}\n"
                    f"  NAND2_X1 {cells - memories}\n"
                    f"  $mem_v2 {memories}\n"
                    f"Chip area for module '\\top': {area:.3f}\n",
                    encoding="utf-8",
                )
                (mapping / f"{name}_mapped.v").write_text(
                    "module top; endmodule\n", encoding="utf-8"
                )
            report = build_report(out)
            self.assertEqual(len(report["rows"]), 3)
            self.assertEqual([row["product_lanes"] for row in report["rows"]],
                             [96, 96, 96])
            dctf = report["rows"][2]
            self.assertAlmostEqual(
                dctf["logic_area_reduction_vs_central96"], 0.1
            )
            self.assertAlmostEqual(
                dctf["logic_area_reduction_vs_independent32x3"], 0.4
            )


if __name__ == "__main__":
    unittest.main()
