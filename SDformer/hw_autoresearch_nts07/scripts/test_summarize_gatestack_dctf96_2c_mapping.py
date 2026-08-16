#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_dctf96_2c_mapping import (
    build_report,
    dctf2c_state_bits,
    render_markdown,
)


class Dctf96TwoContextMappingSummaryTest(unittest.TestCase):
    def test_state_bit_ledger(self) -> None:
        bits = dctf2c_state_bits()
        self.assertEqual(bits["context_token_bits"], 2592)
        self.assertEqual(bits["context_seen_bits"], 324)
        self.assertEqual(bits["context_metadata_bits"], 192)
        self.assertEqual(bits["shared_control_bits"], 30)
        self.assertEqual(bits["total_architectural_state_bits"], 3138)

    def test_area_normalized_throughput_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mapping = root / "mapping"
            mapping.mkdir()
            (mapping / "dctf96_2c.log").write_text(
                "Number of processes: 0\n"
                "Number of cells: 900\n"
                "  AND2_X1 899\n"
                "  $mem_v2 1\n"
                "Chip area for module '\\\\top': 90.000\n",
                encoding="utf-8",
            )
            (mapping / "dctf96_2c_mapped.v").write_text(
                "module top; endmodule\n", encoding="utf-8"
            )
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({
                "rows": [
                    {"name": "central96_term", "logic_area": 100.0,
                     "cells": 1000, "mem_v2": 2},
                    {"name": "dctf96_term", "logic_area": 80.0,
                     "cells": 800, "mem_v2": 3},
                ]
            }), encoding="utf-8")

            report = build_report(root, baseline)
            expected = (59853 * 100.0) / (53910 * 90.0)
            self.assertAlmostEqual(
                report["dctf2c_area_normalized_throughput_vs_central"],
                expected,
            )
            self.assertIn("3138 bit", render_markdown(report))
            self.assertIn("共享控制30 bit", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
