from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.report_h67_denominator_certificate_rtl import (
    parse_macro_area,
    parse_sta,
    parse_top_area,
    require_pass,
)


class ReportH67DenominatorCertificateRtlTest(unittest.TestCase):
    def test_exact_pass_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.log"
            path.write_text(
                "PASS tb_h67_denominator_certificate rows=18 gates=16200 errors=0\n"
            )
            self.assertEqual(require_pass(path), (18, 16200))
            path.write_text("ERROR: bad\nPASS tb_h67_denominator_certificate rows=18 gates=16200 errors=0\n")
            with self.assertRaises(ValueError):
                require_pass(path)

    def test_top_area_is_unambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yosys.log"
            path.write_text("Chip area for top module '\\dut': 12.500000\n")
            self.assertEqual(parse_top_area(path, "dut"), 12.5)

    def test_sta_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sta.log"
            path.write_text(
                "  1.250000 data arrival time\n  1.650000 slack (MET)\n"
            )
            parsed = parse_sta(path)
            self.assertEqual(parsed["status"], "MET")
            self.assertAlmostEqual(parsed["data_arrival_ns"], 1.25)

    def test_macro_area_bound_to_named_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "macro.lib"
            path.write_text(
                "cell(other) { area : 1.0; }\n"
                "cell(fakeram45_256x16) { area : 3155.026; }\n"
            )
            self.assertAlmostEqual(parse_macro_area(path), 3155.026)


if __name__ == "__main__":
    unittest.main()
