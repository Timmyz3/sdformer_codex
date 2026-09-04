#!/usr/bin/env python3
"""Unit tests for the Local5 OUT32 population reporter parser."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from scripts import report_local5_out32_population_sensitivity as report
except ModuleNotFoundError:
    import report_local5_out32_population_sensitivity as report


def write_log(path: Path, cycle_offset: int = 0, failure: bool = False) -> None:
    lines = []
    total = 0
    for group in range(100):
        cycles = 1000 + group + cycle_offset
        total += cycles
        lines.append(
            "GROUP backend=0 latency=1 "
            f"group={group} cycles={cycles} score_rows=450 score_service=10 "
            "score_direct_rows=5 qsilent_rows=400 identk_rows=20 overlap=3 "
            "active=12 memory_wait=0 terms=24 updates=36"
        )
    if failure:
        lines.append("FAIL injected")
    lines.append(
        "PASS Local5 score-to-projection backend=0 latency=1 "
        f"groups=100 total_cycles={total}"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ParserTests(unittest.TestCase):
    def test_accepts_complete_population(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pass.log"
            write_log(path)
            rows = report.parse_log(path)
            self.assertEqual(len(rows), 100)
            self.assertEqual(rows[99]["cycles"], 1099)

    def test_rejects_failure_marker_and_bad_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed = root / "failed.log"
            write_log(failed, failure=True)
            with self.assertRaises(ValueError):
                report.parse_log(failed)
            bad = root / "bad.log"
            write_log(bad)
            bad.write_text(
                bad.read_text().replace("total_cycles=104950", "total_cycles=1"),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                report.parse_log(bad)


if __name__ == "__main__":
    unittest.main()
