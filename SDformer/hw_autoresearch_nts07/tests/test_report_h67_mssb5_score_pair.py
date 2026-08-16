from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.report_h67_mssb5_score_pair import evaluate, parse_mapping, parse_sta


class H67Mssb5ScorePairReportTest(unittest.TestCase):
    def test_mapping_and_sta_parsers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = root / "mapping.log"
            mapping.write_text(
                "Found and reported 0 problems.\n"
                "   Number of cells: 123\n"
                "   Chip area for module '\\demo': 456.750000\n",
                encoding="utf-8",
            )
            sta = root / "sta.log"
            sta.write_text(
                "  1.250000 data arrival time\n"
                "  3.750000 slack (MET)\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_mapping(mapping), {"cells": 123, "area": 456.75})
            self.assertEqual(parse_sta(sta), {"delay_ns": 1.25, "slack_ns": 3.75})
            sta.write_text("no path\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                parse_sta(sta)

    def test_decision_uses_cse7_and_does_not_overclaim_packed_tree(self) -> None:
        candidates = {
            "h67_cse7_score_pair": {"area": 3000.0, "delay_ns": 1.0},
            "h67_ssr5_score_pair": {"area": 2200.0, "delay_ns": 0.9},
            "h67_mssb5_score_pair": {"area": 2150.0, "delay_ns": 0.91},
        }
        result = evaluate(candidates)
        self.assertEqual(result["decision"], "ADMIT_ROW_TOP_INTEGRATION")
        self.assertFalse(result["packed_butterfly_is_independent_contribution"])

        candidates["h67_mssb5_score_pair"]["area"] = 2700.0
        self.assertEqual(evaluate(candidates)["decision"], "REJECT_MSSB5")


if __name__ == "__main__":
    unittest.main()
