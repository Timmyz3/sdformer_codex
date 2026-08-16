#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from scripts.report_local5_phase_residual_openproxy import (
    build_report,
    parse_mapping,
    parse_sta,
)


class PhaseResidualOpenProxyTest(unittest.TestCase):
    def test_mapping_parser_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.log"
            path.write_text("Number of cells: 12\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_mapping(path)

    def test_sta_parser_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sta.log"
            path.write_text("0.1 slack (MET)\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_sta(path)

    def test_report_requires_rtl_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            logs = output / "logs"
            logs.mkdir()
            mapping = (
                "Number of cells: 10\n"
                "Chip area for module '\\qfit_local5_score_leaf': 20.0\n"
                "Found and reported 0 problems\n"
            )
            sta = "1.0 data arrival time\n2.0 slack (MET)\n"
            for name in ("absolute", "phase_residual"):
                (logs / f"nangate45_{name}.log").write_text(mapping)
                (logs / f"sta_{name}.log").write_text(sta)
            (logs / "score_leaf_regression.log").write_text("PASS only\n")
            with self.assertRaises(ValueError):
                build_report(output)


if __name__ == "__main__":
    unittest.main()
