#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from scripts import report_h67_real_weight_projection_all_rtl as report


class ReportProjectionAllTests(unittest.TestCase):
    def test_rejects_false_pass_after_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.log"
            lines = [
                "FATAL: injected",
                *[
                    "REALWALL_ROW batch=0 row={row} stage={stage} block={block} "
                    "head={head} valid={valid}".format(**item)
                    for row, item in enumerate(report.expected_rows(0))
                ],
                "PASS H67 RQTB 2S physical flow rows=138 checked=1 "
                "fixed_cycles=112589 rqtb_cycles=94891 fixed_slots=62100 "
                "rqtb_slots=34099 fixed_exp=1 rqtb_exp=1 acc32_mismatch=0",
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                report.parse_log(path, 0)


if __name__ == "__main__":
    unittest.main()
