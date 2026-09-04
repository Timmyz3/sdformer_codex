#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_dctf32_bank_executor import build_report, render_markdown


class Dctf32BankExecutorSummaryTest(unittest.TestCase):
    def test_same_source_boundary_and_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "product_engine_32.log").write_text(
                "   Number of cells: 1000\n"
                "   Chip area for module '\\leaf': 100.000\n"
            )
            (root / "executor_32.log").write_text(
                "   Number of cells: 1450\n"
                "   Chip area for module '\\executor': 160.000\n"
            )

            report = build_report(root)
            leaf, executor = report["rows"]
            comparison = report["comparison"]

            self.assertEqual(report["evidence"], "开放库无约束logic proxy")
            self.assertEqual(leaf["out_tile"], executor["out_tile"])
            self.assertEqual(
                report["same_source_settings"]["aligned_engine_tag_w"], 36
            )
            self.assertEqual([leaf["mem_v2"], executor["mem_v2"]], [0, 0])
            self.assertEqual(comparison["logic_area_delta"], 60.0)
            self.assertAlmostEqual(comparison["logic_area_increase"], 0.6)
            self.assertEqual(comparison["cell_delta"], 450)
            self.assertEqual(report["epoch_constraint"]["epoch_states"], 16)
            self.assertIn("不是纯路由面积", comparison["interpretation"])

            markdown = render_markdown(report)
            self.assertIn("stale_rsp=1", markdown)
            self.assertIn("不得称为ASIC PPA或签核结果", markdown)

    def test_uses_last_mapping_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, final_area, final_cells, memories in (
                ("product_engine_32", 20.0, 200, 0),
                ("executor_32", 30.0, 300, 2),
            ):
                (root / f"{name}.log").write_text(
                    "   Number of cells: 9\n"
                    "   Chip area for module '\\early': 1.000\n"
                    + (f"   $mem_v2 {memories}\n" if memories else "")
                    + f"   Number of cells: {final_cells}\n"
                    + f"   Chip area for module '\\final': {final_area:.3f}\n"
                )

            report = build_report(root)
            self.assertEqual(report["rows"][0]["cells"], 200)
            self.assertEqual(report["rows"][1]["logic_area"], 30.0)
            self.assertEqual(report["rows"][1]["mem_v2"], 2)


if __name__ == "__main__":
    unittest.main()
