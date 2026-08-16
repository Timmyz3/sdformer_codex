from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_local5_tcfm5_model_to_rtl import parse_group_log, ratio


class Local5ModelRtlCalibrationTest(unittest.TestCase):
    def test_parse_group_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtl.log"
            path.write_text(
                "GROUP backend=0 new1rw=0 mode=1 latency=1 group=0 cycles=10 "
                "active=2 avoided=448 memory_wait=4 terms=3 updates=5 term_stall=0\n"
                "GROUP backend=0 new1rw=0 mode=1 latency=1 group=1 cycles=12 "
                "active=3 avoided=447 memory_wait=6 terms=4 updates=7 term_stall=0\n",
                encoding="utf-8",
            )
            rows = parse_group_log(path, 2)
            self.assertEqual(rows[0]["cycles"], 10)
            self.assertEqual(rows[1]["terms"], 4)

    def test_parse_group_log_rejects_missing_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rtl.log"
            path.write_text(
                "GROUP backend=0 group=1 cycles=12 active=3 avoided=447 "
                "memory_wait=6 terms=4 updates=7\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "顺序不连续"):
                parse_group_log(path, 1)

    def test_ratio_rejects_zero_denominator(self) -> None:
        with self.assertRaisesRegex(ValueError, "denominator"):
            ratio(1, 0)


if __name__ == "__main__":
    unittest.main()
