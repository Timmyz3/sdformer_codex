#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_dctf_frontend import build_report


class DctfFrontendSummaryTest(unittest.TestCase):
    def test_parse_two_leaf_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, area, cells, memories in (
                ("adapter", 10.0, 100, 1),
                ("fabric_q2", 20.0, 200, 0),
            ):
                (root / f"{name}.log").write_text(
                    (f"   $mem_v2 {memories}\n" if memories else "") +
                    f"   Number of cells: {cells}\n"
                    f"   Chip area for module '\\top': {area:.3f}\n"
                )
            report = build_report(root)
            self.assertEqual(report["adapter_default_buffer_bits"], 1458)
            self.assertEqual(report["rows"][0]["mem_v2"], 1)
            self.assertEqual(report["rows"][1]["mem_v2"], 0)


if __name__ == "__main__":
    unittest.main()
