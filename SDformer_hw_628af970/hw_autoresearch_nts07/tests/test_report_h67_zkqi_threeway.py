from __future__ import annotations

import unittest

from scripts.report_h67_zkqi_threeway import distribution


class ReportH67ZkqiThreewayTest(unittest.TestCase):
    def test_distribution_is_nearest_rank(self) -> None:
        result = distribution([1, 2, 3, 4, 100])
        self.assertEqual(result["p50"], 3)
        self.assertEqual(result["p95"], 100)
        self.assertEqual(result["sum"], 110)


if __name__ == "__main__":
    unittest.main()
