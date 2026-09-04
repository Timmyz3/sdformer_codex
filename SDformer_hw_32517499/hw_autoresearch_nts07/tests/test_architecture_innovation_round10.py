import unittest

from scripts.model_architecture_innovation_round10 import (
    capacity_stats,
    percentile,
    summarize,
)


class ArchitectureInnovationRound10Test(unittest.TestCase):
    def test_percentile_and_summary(self):
        values = [0, 1, 2, 3, 4]
        self.assertEqual(percentile(values, 0.50), 2)
        self.assertEqual(percentile(values, 0.95), 4)
        self.assertEqual(summarize(values)["sum"], 10)

    def test_capacity_accounts_rows_and_work(self):
        result = capacity_stats([0, 2, 5, 9], [4])["4"]
        self.assertEqual(result["overflow_rows"], 2)
        self.assertEqual(result["overflow_work"], 6)
        self.assertAlmostEqual(result["overflow_row_ratio"], 0.5)
        self.assertAlmostEqual(result["overflow_work_ratio"], 6 / 16)


if __name__ == "__main__":
    unittest.main()
