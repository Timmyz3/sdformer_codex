"""Posthoc direction error masks use valid GT with actual144px tail supports."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'flow_story_20261013/motion_groups'))
from run_fast_runtime import direction_regions

CONTRACT = {'direction': 'forward', 'time_interval': 'same input interval', 'units': 'original_image_pixels'}


class DirectionRegionsTest(unittest.TestCase):
    def test_correct_wrong_abstain_and_ineligible_partition_with_holes_and_tails(self):
        gt = np.zeros((2, 480, 640), np.float32); gt[0] = 2
        mask = np.ones((480, 640), bool); mask[0:5, 0:7] = False
        flow = [[[2., 0.] for _ in range(5)] for _ in range(4)]
        flow[0][1] = [-2., 0.]; flow[0][2] = [0., 0.]
        gt[:, 432:, 576:] = 0
        original = gt.copy(); original_flow = [[list(v) for v in row] for row in flow]
        regions, info = direction_regions(gt, mask, flow, CONTRACT)
        self.assertEqual(int(regions['direction_wrong'].sum()), 144*144)
        self.assertEqual(int(regions['direction_abstain'].sum()), 144*144)
        self.assertEqual(int(regions['direction_GT_ineligible'].sum()), 48*64)
        self.assertTrue(np.array_equal(sum(v.astype(int) for v in regions.values()), mask.astype(int)))
        self.assertTrue(np.array_equal(gt, original)); self.assertEqual(flow, original_flow)
        self.assertFalse(info['GT_used_by_inference'])

    def test_empty_GT_tile_has_no_evaluated_pixels(self):
        gt = np.ones((2, 480, 640), np.float32)
        mask = np.zeros((480, 640), bool); mask[-1, -1] = True
        flow = [[[1., 1.] for _ in range(5)] for _ in range(4)]
        regions, _ = direction_regions(gt, mask, flow, CONTRACT)
        self.assertEqual(int(regions['direction_correct'].sum()), 1)
        self.assertEqual(sum(int(v.sum()) for v in regions.values()), 1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
