#!/usr/bin/env python3
"""IBF映射日志解析测试。"""

import tempfile
import unittest
from pathlib import Path

from scripts.summarize_implicit_bias_finalizer import parse_mapping


class TestSummarizeImplicitBiasFinalizer(unittest.TestCase):
    def test_parse_last_mapping_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.log"
            path.write_text(
                "Checking read port `x.acc_mem'[0]\n"
                "Number of cells: 123\n"
                "  DFF_X1 45\n"
                "  $mem_v2 2\n"
                "Chip area for module 'x': 678.25\n",
                encoding="utf-8",
            )
            row = parse_mapping(path)
            self.assertEqual(row["cells"], 123)
            self.assertEqual(row["dff_x1"], 45)
            self.assertEqual(row["mem_v2"], 2)
            self.assertEqual(row["acc_memory_read_ports_total"], 1)
            self.assertAlmostEqual(row["logic_area"], 678.25)


if __name__ == "__main__":
    unittest.main()
