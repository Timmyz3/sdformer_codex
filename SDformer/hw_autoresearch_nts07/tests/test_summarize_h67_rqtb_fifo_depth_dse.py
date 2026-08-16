from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.summarize_h67_rqtb_fifo_depth_dse import build_report, parse_log


def make_log(depth: int, fixed: int, rqtb: int) -> str:
    return "\n".join(
        [
            f"RQTB_ROW row=0 fixed_fifo_max={depth} rqtb_fifo_max={depth - 1}",
            "RQTB_2S_COVER cross_pair=1 same_class=2 double_active=3 fifo_both=4 dual_k=5",
            f"PASS H67 RQTB 2S physical flow rows=1 checked=7 fixed_cycles={fixed} "
            f"rqtb_cycles={rqtb} fixed_slots=10 rqtb_slots=6 fixed_exp=5 "
            "rqtb_exp=4 acc32_mismatch=0",
        ]
    )


class SummarizeH67RqtbFifoDepthDseTest(unittest.TestCase):
    def test_parse_requires_exact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth_4.log"
            path.write_text(make_log(4, 120, 100), encoding="utf-8")
            result = parse_log(path, 4)
            self.assertEqual(result["slot_storage_bits"], 64)
            self.assertEqual(result["rqtb_fifo_max"], 3)

    def test_selects_smallest_depth_within_one_percent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for depth, rqtb in ((2, 205), (4, 201), (8, 200), (16, 200), (32, 200)):
                (root / f"depth_{depth}.log").write_text(
                    make_log(depth, 240, rqtb), encoding="utf-8"
                )
            result = build_report(root, [2, 4, 8, 16, 32])
            self.assertEqual(result["selection"]["depth"], 4)


if __name__ == "__main__":
    unittest.main()
