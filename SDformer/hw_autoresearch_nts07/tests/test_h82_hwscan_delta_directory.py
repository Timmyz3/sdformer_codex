#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from scripts.h82_class_file_reference import SCORE_HI, SCORE_LO
from scripts.h82_hwscan_delta_directory import demo, scan_delta, summarize


class HwscanDeltaTests(unittest.TestCase):
    def test_identical_windows_have_zero_member_edits(self) -> None:
        codes = np.zeros(450, dtype=np.int64)
        codes[10:20] = 3
        codes[225:235] = 3
        delta = scan_delta(codes, codes)
        self.assertEqual(delta["member_edits"], 0)
        self.assertEqual(delta["class_insert"], 0)
        self.assertEqual(delta["member_jaccard"], 1.0)

    def test_class_set_stable_roster_churn_still_pays_members(self) -> None:
        prev = np.zeros(450, dtype=np.int64)
        curr = np.zeros(450, dtype=np.int64)
        prev[0:50] = 1
        curr[50:100] = 1
        delta = scan_delta(prev, curr)
        self.assertGreaterEqual(delta["class_jaccard"], 0.99)
        self.assertEqual(delta["member_edits"], 200)
        self.assertLess(delta["member_jaccard"], 0.6)

    def test_smooth_field_does_not_invent_frozen_0_30(self) -> None:
        report = demo(seed=138)
        self.assertGreater(report["spatial_smooth"]["n_scan_rows"], 4)
        self.assertLess(report["spatial_smooth"]["mean_member_jaccard"], 0.30)
        self.assertGreater(
            report["spatial_smooth"]["mean_member_edits"],
            report["spatial_smooth"]["mean_class_edits"],
        )


if __name__ == "__main__":
    unittest.main()
