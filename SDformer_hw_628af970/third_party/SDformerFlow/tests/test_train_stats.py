import unittest

from utils.train_stats import compute_throughput_stats


class ComputeThroughputStatsTest(unittest.TestCase):
    def test_handles_zero_values(self):
        step_time, samples_per_sec = compute_throughput_stats(0, 0.0)

        self.assertEqual(step_time, 0.0)
        self.assertEqual(samples_per_sec, 0.0)

    def test_returns_expected_values(self):
        step_time, samples_per_sec = compute_throughput_stats(5, 10.0)

        self.assertEqual(step_time, 2.0)
        self.assertEqual(samples_per_sec, 0.5)


if __name__ == "__main__":
    unittest.main()
