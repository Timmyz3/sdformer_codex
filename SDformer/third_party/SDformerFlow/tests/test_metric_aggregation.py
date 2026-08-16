import unittest

import torch

from utils.metric_aggregation import FlowMetricAggregationAudit


class FlowMetricAggregationAuditTest(unittest.TestCase):
    def test_dsec_fl_is_gt_magnitude_percent(self):
        audit = FlowMetricAggregationAudit()
        pred = torch.tensor([[[[105.1]], [[0.0]]]])
        label = torch.tensor([[[[100.0]], [[0.0]]]])
        mask = torch.ones(1, 1, 1, 1)
        audit.update(pred, label, mask, 1.0, ["seq"])
        result = audit.summary()

        self.assertAlmostEqual(result["frame_equal_mean"]["DSEC_Fl"], 100.0, places=5)
        self.assertAlmostEqual(result["pixel_global_mean"]["DSEC_Fl"], 100.0, places=5)
        self.assertAlmostEqual(result["sequence_balanced_mean"]["DSEC_Fl"], 100.0, places=5)

    def test_frame_pixel_and_sequence_weighting_are_distinct(self):
        audit = FlowMetricAggregationAudit()
        pred = torch.tensor(
            [
                [[[0.0, 2.0]], [[0.0, 0.0]]],
                [[[2.0, 2.0]], [[0.0, 0.0]]],
            ]
        )
        label = torch.zeros_like(pred)
        mask = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 1.0]]]])
        audit.update(pred, label, mask, 1.0, ["seq_a", "seq_b"])
        result = audit.summary()

        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["valid_pixels"], 3.0)
        self.assertAlmostEqual(result["frame_equal_mean"]["AEE"], 1.0)
        self.assertAlmostEqual(result["pixel_global_mean"]["AEE"], 4.0 / 3.0)
        self.assertAlmostEqual(result["sequence_balanced_mean"]["AEE"], 1.0)

    def test_frame_equal_angles_match_production_metrics(self):
        from loss.flow_supervised import AAE, AAE_Benchmark

        pred = torch.tensor([[[[1.0, 0.5]], [[0.0, 1.0]]]])
        label = torch.tensor([[[[0.5, 1.0]], [[1.0, 0.0]]]])
        mask = torch.ones((1, 1, 1, 2))
        audit = FlowMetricAggregationAudit()
        audit.update(pred, label, mask, 1.0, ["seq"])
        result = audit.summary()["frame_equal_mean"]

        self.assertAlmostEqual(
            result["AAE"], float(AAE(pred, label, mask, flow_scaling=1)()[0]), places=5
        )
        self.assertAlmostEqual(
            result["AAE_Benchmark"],
            float(AAE_Benchmark(pred, label, mask, flow_scaling=1)()[0]),
            places=5,
        )

    def test_requires_one_sequence_id_per_batch_item(self):
        audit = FlowMetricAggregationAudit()
        value = torch.zeros((1, 2, 1, 1))
        with self.assertRaises(ValueError):
            audit.update(value, value, torch.ones((1, 1, 1, 1)), 1.0, [])


if __name__ == "__main__":
    unittest.main()
