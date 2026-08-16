#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from scripts.report_local5_score_projection_rtl import parse_rows, stats


class Local5ScoreProjectionReportTest(unittest.TestCase):
    def test_parse_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.log"
            path.write_text(
                "GROUP backend=0 latency=1 group=0 cycles=10 score_rows=450 "
                "score_service=5 score_direct_rows=2 active=3 memory_wait=4 "
                "terms=6 updates=7\nPASS Local5 score-to-projection\n",
                encoding="utf-8",
            )
            rows = parse_rows(path)
            self.assertEqual(rows[0]["cycles"], 10)
            self.assertEqual(rows[0]["score_rows"], 450)

    def test_stats(self) -> None:
        result = stats([1.0, 2.0, 3.0])
        self.assertEqual(result["total"], 6.0)
        self.assertEqual(result["p50"], 2.0)


if __name__ == "__main__":
    unittest.main()
