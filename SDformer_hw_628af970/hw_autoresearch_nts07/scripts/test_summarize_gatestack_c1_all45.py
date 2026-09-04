#!/usr/bin/env python3
"""验证 C1 汇总器不会截断浮点加速比。"""

from __future__ import annotations

import unittest

from summarize_gatestack_c1_all45 import FINAL_RE, STAGE_RE, parse_final, parse_stage


class SummarizeC1All45Test(unittest.TestCase):
    def test_stage_speedup_remains_float(self) -> None:
        match = STAGE_RE.search(
            "C1_STAGE stage=3 heads=24 cycles=6182 c0=8898 speedup=1.439340"
        )
        self.assertIsNotNone(match)
        row = parse_stage(match)
        self.assertEqual(row["cycles"], 6182)
        self.assertAlmostEqual(row["speedup"], 1.439340)

    def test_final_speedup_remains_float(self) -> None:
        match = FINAL_RE.search(
            "PASS: C1 all45 stage-bounded cycles=10035 C0=14078 "
            "speedup=1.402890 overlap=2356 blocked=2523 stalls=354"
        )
        self.assertIsNotNone(match)
        row = parse_final(match)
        self.assertEqual(row["c0"], 14078)
        self.assertIsInstance(row["speedup"], float)
        self.assertAlmostEqual(row["speedup"], 1.402890)


if __name__ == "__main__":
    unittest.main()
