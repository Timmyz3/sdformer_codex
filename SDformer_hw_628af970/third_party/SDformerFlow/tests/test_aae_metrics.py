import math
import unittest

import torch

from loss.flow_supervised import AAE, AAE_Benchmark, AEE, DSEC_Fl


class AngularErrorMetricsTest(unittest.TestCase):
    def test_standard_dsec_fl_uses_ground_truth_magnitude(self):
        pred = torch.tensor([[[[105.1]], [[0.0]]]])
        label = torch.tensor([[[[100.0]], [[0.0]]]])
        mask = torch.ones(1, 1, 1, 1)

        legacy_outlier = AEE(pred, label, mask, flow_scaling=1)()[4]
        standard_fl = DSEC_Fl(pred, label, mask, flow_scaling=1)()

        self.assertAlmostEqual(float(legacy_outlier), 0.0, places=6)
        self.assertAlmostEqual(float(standard_fl[0]), 100.0, places=6)

    def test_benchmark_metric_matches_barron_space_time_formula(self):
        pred = torch.tensor([[[[1.0]], [[0.0]]]])
        label = torch.tensor([[[[0.0]], [[1.0]]]])
        mask = torch.ones(1, 1, 1, 1)

        value = AAE_Benchmark(pred, label, mask, flow_scaling=1)()
        expected = math.degrees(math.acos(0.5))

        self.assertEqual(tuple(value.shape), (1,))
        self.assertAlmostEqual(value.item(), expected, places=4)

    def test_legacy_directional_metric_remains_distinct(self):
        pred = torch.tensor([[[[1.0]], [[0.0]]]])
        label = torch.tensor([[[[0.0]], [[1.0]]]])
        mask = torch.ones(1, 1, 1, 1)

        legacy = AAE(pred, label, mask, flow_scaling=1)()[0]
        benchmark = AAE_Benchmark(pred, label, mask, flow_scaling=1)()[0]

        self.assertAlmostEqual(legacy.item(), 90.0, places=3)
        self.assertAlmostEqual(benchmark.item(), 60.0, places=3)

    def test_benchmark_metric_respects_each_batch_mask(self):
        pred = torch.tensor([
            [[[1.0]], [[0.0]]],
            [[[0.0]], [[1.0]]],
        ])
        label = torch.tensor([
            [[[1.0]], [[0.0]]],
            [[[1.0]], [[0.0]]],
        ])
        mask = torch.ones(2, 1, 1, 1)

        value = AAE_Benchmark(pred, label, mask, flow_scaling=1)()

        self.assertEqual(tuple(value.shape), (2,))
        self.assertLess(value[0].item(), 0.1)
        self.assertAlmostEqual(value[1].item(), 60.0, places=3)


if __name__ == "__main__":
    unittest.main()
