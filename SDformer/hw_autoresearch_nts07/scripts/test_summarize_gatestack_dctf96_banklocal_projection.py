#!/usr/bin/env python3
"""DCTF96 bank-local projection映射汇总器单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_dctf96_banklocal_projection import parse_mapping


class MappingParserTest(unittest.TestCase):
    def test_parse_final_stat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "map.log"
            netlist = root / "map.v"
            log.write_text(
                """
Number of processes:              0
Number of cells:                 12
  NAND2_X1                       10
  $mem_v2                         2
Chip area for module 'top': 123.500
""",
                encoding="utf-8",
            )
            netlist.write_text("module top; endmodule\n", encoding="utf-8")
            row = parse_mapping(log, netlist)
            self.assertEqual(row["logic_area"], 123.5)
            self.assertEqual(row["cells"], 12)
            self.assertEqual(row["mem_v2"], 2)
            self.assertEqual(row["unmapped_dollar_cells"], {})

    def test_reject_unmapped_arithmetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "map.log"
            netlist = root / "map.v"
            log.write_text(
                """
Number of processes:              0
Number of cells:                  1
  $mul                             1
Chip area for module 'top': 1.000
""",
                encoding="utf-8",
            )
            netlist.write_text("module top; endmodule\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                parse_mapping(log, netlist)


if __name__ == "__main__":
    unittest.main()
