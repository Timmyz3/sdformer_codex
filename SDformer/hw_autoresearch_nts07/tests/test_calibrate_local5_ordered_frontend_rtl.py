from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_local5_ordered_frontend_rtl as calibration


class Local5OrderedFrontendCalibrationTest(unittest.TestCase):
    def test_parse_requires_100_unique_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rtl.log"
            path.write_text(
                "\n".join(
                    f"GROUP group={index} cycles=500 active=10 terms=20 term_stall=5"
                    for index in range(100)
                )
                + "\n",
                encoding="utf-8",
            )
            rows = calibration.parse_log(path)
            self.assertEqual(len(rows), 100)
            self.assertEqual(calibration.residual(rows[0]), 465)

    def test_sequential_model_recovers_fixed_boundary(self) -> None:
        rows = [
            {
                "group": index,
                "cycles": 456 + 10 + index + 2,
                "active": 10,
                "terms": index,
                "term_stall": 2,
            }
            for index in range(100)
        ]
        sequential = calibration.model_metrics(rows, fixed=456, mode="sequential")
        overlap = calibration.model_metrics(rows, fixed=456, mode="v2_max_overlap")
        self.assertEqual(sequential["mae"], 0.0)
        self.assertGreater(overlap["mae"], 0.0)

    def test_summary_reports_observed_bounds(self) -> None:
        result = calibration.summary(np.asarray([1, 2, 3], dtype=np.float64))
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 3.0)


if __name__ == "__main__":
    unittest.main()
