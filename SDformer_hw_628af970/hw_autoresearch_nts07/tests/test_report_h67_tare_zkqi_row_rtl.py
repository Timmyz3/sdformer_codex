import tempfile
import unittest
from pathlib import Path

from scripts.report_h67_tare_zkqi_row_rtl import (
    candidate_decision,
    parse_area_log,
    parse_leaf_log,
    parse_pass_log,
)


class TestReportH67TareZkqiRowRtl(unittest.TestCase):
    def test_parse_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sim.log"
            path.write_text(
                "PASS tb_h67_zkqi_row_miter rows=138 stall_mode=2 bundle_skip=1 "
                "outputs=1 baseline_e2e_cycles=1000 zkqi_e2e_cycles=1005 "
                "baseline_tare_dense=0 candidate_tare_dense=251\n",
                encoding="utf-8",
            )
            row = parse_pass_log(path)
            self.assertEqual(row["mode"], 2)
            self.assertEqual(row["candidate_dense"], 251)
            self.assertAlmostEqual(row["candidate_cycle_regression"], 0.005)

    def test_parse_area(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.log"
            path.write_text(
                "Number of cells: 1234\nChip area for module 'top': 5678.250000\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_area_log(path), {"area": 5678.25, "cells": 1234})

    def test_parse_leaf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaf.log"
            path.write_text(
                "PASS tb_h67_tare_score_pair W=16 received=35\n", encoding="utf-8"
            )
            self.assertEqual(parse_leaf_log(path), {"width": 16, "received": 35})

    def test_candidate_decision_checks_both_gates(self):
        self.assertEqual(candidate_decision(True, True), "ADMIT")
        self.assertEqual(candidate_decision(False, True), "REJECT")
        self.assertEqual(candidate_decision(True, False), "REJECT")


if __name__ == "__main__":
    unittest.main()
