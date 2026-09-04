from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.report_h67_zkqi_row_miter import parse_area, parse_log


class ReportH67ZkqiRowMiterTest(unittest.TestCase):
    def test_parse_single_row_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sim.log"
            path.write_text(
                "ROW_RESULT row=0 stage=0 block=0 head=0 active_pairs=3 outputs=4 "
                "baseline_cycles=10 zkqi_cycles=8 baseline_slots=5 zkqi_slots=3 "
                "seeded=444 baseline_read_bits=100 zkqi_read_bits=60 fifo_max=1\n"
                "PASS tb_h67_zkqi_row_miter rows=1 stall_mode=0 outputs=4 "
                "baseline_cycles=10 zkqi_cycles=8 baseline_read_bits=100 zkqi_read_bits=60\n",
                encoding="utf-8",
            )
            rows, final = parse_log(path, expected_rows=1)
        self.assertEqual(rows[0]["active_pairs"], 3)
        self.assertEqual(final["zkqi_cycles"], 8)

    def test_reject_duplicate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sim.log"
            row = (
                "ROW_RESULT row=0 stage=0 block=0 head=0 active_pairs=3 outputs=4 "
                "baseline_cycles=10 zkqi_cycles=8 baseline_slots=5 zkqi_slots=3 "
                "seeded=444 baseline_read_bits=100 zkqi_read_bits=60 fifo_max=1\n"
            )
            final = (
                "PASS tb_h67_zkqi_row_miter rows=1 stall_mode=0 outputs=4 "
                "baseline_cycles=10 zkqi_cycles=8 baseline_read_bits=100 zkqi_read_bits=60\n"
            )
            path.write_text(row + final + final, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "单次PASS"):
                parse_log(path, expected_rows=1)

    def test_parse_area_uses_last_top_stat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.log"
            path.write_text(
                "Chip area for module 'old': 12.5\n"
                "Chip area for module 'top': 34.75\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_area(path), 34.75)


if __name__ == "__main__":
    unittest.main()
