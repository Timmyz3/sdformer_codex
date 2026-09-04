#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_dctf96_term_datapath import build_report, render_markdown


class Dctf96TermDatapathSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mapping = self.root / "mapping"
        self.mapping.mkdir()
        (self.mapping / "top_q2_tokens162_out32.log").write_text(
            "=== $paramod\\gatestack_dctf96_term_datapath_top\\OUT_TILE=32 ===\n"
            "   Number of processes: 0\n"
            "   Number of cells: 450\n"
            "     $mem_v2 1\n"
            "     AND2_X1 449\n"
            "   Area for cell type $mem_v2 is unknown!\n"
            "   Chip area for module '\\gatestack_dctf96_term_datapath_top': 460.000\n"
        )
        (self.mapping / "top_q2_tokens162_out32_mapped.v").write_text(
            "module mapped; endmodule\n"
        )
        self.executor_report = self.root / "executor.json"
        self.executor_report.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "name": "executor_32",
                            "out_tile": 32,
                            "logic_area": 100.0,
                            "cells": 100,
                            "mem_v2": 0,
                        }
                    ]
                }
            )
        )
        self.frontend_report = self.root / "frontend.json"
        self.frontend_report.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "name": "adapter",
                            "logic_area": 50.0,
                            "cells": 50,
                            "mem_v2": 1,
                        },
                        {
                            "name": "fabric_q2",
                            "logic_area": 25.0,
                            "cells": 25,
                            "mem_v2": 0,
                        },
                    ]
                }
            )
        )
        (self.root / "yosys_version.txt").write_text("Yosys 0.33\n")
        (self.root / "input_sha256.txt").write_text(
            f"{'a' * 64}  rtl_hitflow/top.sv\n"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_arithmetic_sum_and_cautious_delta(self) -> None:
        report = build_report(
            self.mapping,
            self.executor_report,
            self.frontend_report,
            self.root,
        )
        self.assertEqual(report["leaf_arithmetic_sum"]["logic_area"], 375.0)
        self.assertEqual(report["leaf_arithmetic_sum"]["cells"], 375)
        self.assertEqual(report["leaf_arithmetic_sum"]["mem_v2"], 1)
        self.assertEqual(
            report["comparison"]["logic_area_delta_top_minus_leaf_sum"], 85.0
        )
        self.assertEqual(report["comparison"]["cell_delta_top_minus_leaf_sum"], 75)
        self.assertTrue(report["quality_checks"]["processes_zero"])
        self.assertEqual(report["flatten_top"]["mem_v2"], 1)
        self.assertIn("不能称为纯协调器面积", report["comparison"]["interpretation"])
        markdown = render_markdown(report)
        self.assertIn("无约束逻辑映射代理", markdown)
        self.assertIn("不得称为ASIC PPA", markdown)

    def test_rejects_unmapped_dollar_cell(self) -> None:
        log = self.mapping / "top_q2_tokens162_out32.log"
        log.write_text(log.read_text().replace("AND2_X1 449", "$mux 1\n     AND2_X1 448"))
        with self.assertRaisesRegex(RuntimeError, "未映射\\$单元"):
            build_report(
                self.mapping,
                self.executor_report,
                self.frontend_report,
                self.root,
            )

    def test_rejects_nonzero_process_or_empty_netlist(self) -> None:
        log = self.mapping / "top_q2_tokens162_out32.log"
        log.write_text(log.read_text().replace("Number of processes: 0", "Number of processes: 2"))
        with self.assertRaisesRegex(RuntimeError, "仍有2个process"):
            build_report(
                self.mapping,
                self.executor_report,
                self.frontend_report,
                self.root,
            )
        log.write_text(log.read_text().replace("Number of processes: 2", "Number of processes: 0"))
        (self.mapping / "top_q2_tokens162_out32_mapped.v").write_text("")
        with self.assertRaisesRegex(RuntimeError, "网表缺失或为空"):
            build_report(
                self.mapping,
                self.executor_report,
                self.frontend_report,
                self.root,
            )


if __name__ == "__main__":
    unittest.main()
