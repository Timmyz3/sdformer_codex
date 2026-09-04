#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from summarize_gatestack_builder_projection_allstages import build_report, parse_result


class SummaryTest(unittest.TestCase):
    def test_parse_result(self) -> None:
        parsed = parse_result(
            "RESULT stage=S0 mode=C1 status=PASS total_cycles=10 "
            "build_cycles=4 projection_cycles=6 compared=8 mismatches=0")
        self.assertEqual(parsed["stage"], "S0")
        self.assertEqual(parsed["mode"], "C1")
        self.assertEqual(parsed["total_cycles"], "10")

    def test_build_report(self) -> None:
        numeric = (
            "total_cycles={total} build_cycles={build} projection_cycles=6 "
            "compared=8 mismatches=0 checksum=-2 replay=1 release=1 "
            "projection_heads=1 projection_terms=1 bias=1 slot_commits=1 "
            "payload_copy=0 errors=0 scan=1 stalls=0 blocked=0 overlap=0 "
            "order_wait=0 scale0=0 event_sum=1"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for stage in range(4):
                for mode, total, build in ((0, 10, 4), (1, 8, 2)):
                    path = root / f"s{stage}_c{mode}"
                    path.mkdir()
                    path.joinpath("iverilog.log").write_text(
                        f"RESULT stage=S{stage} mode=C{mode} status=PASS "
                        + numeric.format(total=total, build=build) + "\n")
            report = build_report(root)
        self.assertEqual(report["aggregate"]["c0_total_cycles"], 40)
        self.assertEqual(report["aggregate"]["c1_total_cycles"], 32)
        self.assertEqual(report["aggregate"]["system_speedup"], 1.25)


if __name__ == "__main__":
    unittest.main()
