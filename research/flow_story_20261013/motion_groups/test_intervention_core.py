import unittest
import numpy as np
import torch
from intervention_core import choose_targets, comparable_pairs, pair_statistics, masked_branch


class Tests(unittest.TestCase):
    def test_metric_nonlinearity_is_not_network_coupling(self):
        z = np.zeros((2, 2, 2)); a = z.copy(); b = z.copy()
        a[0] = 1; b[1] = 1
        result = pair_statistics(z, a, b, a+b, z, {'all_valid': np.ones((2, 2), bool), 'B': np.zeros((2, 2), bool)})
        self.assertAlmostEqual(result['all_valid']['interaction'], 2**.5-2)
        self.assertEqual(result['all_valid']['nonlinear_network_AEE_excess'], 0)
        self.assertEqual(result['all_valid']['flow_superposition_residual_mean'], 0)
        self.assertIsNone(result['B']['interaction'])

    def test_nonlinear_joint_response(self):
        z = np.zeros((2, 2, 2)); a = np.ones_like(z); b = a*2
        result = pair_statistics(z, a, b, a+b+1, z, {'all_valid': np.ones((2, 2), bool)})
        self.assertGreater(result['all_valid']['nonlinear_network_AEE_excess'], 0)
        self.assertGreater(result['all_valid']['relative_superposition_residual'], 0)
        r = result['all_valid']
        self.assertAlmostEqual(r['interaction'], r['metric_only_interaction']+r['nonlinear_network_AEE_excess'])
        with self.assertRaises(ValueError):
            pair_statistics(z, a, b, a+b, np.full_like(z, np.nan), {'all_valid': np.ones((2, 2), bool)})

    def test_common_anchors_match_full_cost_and_are_stable(self):
        flow = [[(2., 0.)]*5 for _ in range(5)]
        kw = dict(target='t', layer='l', count=3, quantize=lambda x, y: (1, 0))
        pairs, info = comparable_pairs(flow, 38, 38, **kw)
        self.assertEqual((pairs, info), comparable_pairs(flow, 38, 38, **kw))
        self.assertEqual(len(pairs), 3)
        self.assertGreater(info['rejected']['tail_anchor'], 0)
        for pair in pairs:
            self.assertEqual(pair['each_member_valid_feature_points'], 81)
            for y, x in pair['endpoints_yx'].values():
                self.assertLessEqual((y+1)*9, 38); self.assertLessEqual((x+1)*9, 38)

    def test_no_eligible_pairs_reported_not_fabricated(self):
        pairs, info = comparable_pairs([[(0., 0.)]*2 for _ in range(2)], 18, 18,
                                      target='t', layer='l', count=2, quantize=lambda x, y: None)
        self.assertEqual(pairs, []); self.assertEqual(info['selected_anchors'], 0)

    def test_hook_masks_branch_only_and_restores_even_on_error(self):
        layer = torch.nn.Identity().eval(); x = torch.ones(10, 1, 10, 10, 2)
        with torch.inference_mode():
            with masked_branch(layer, [(0, 0)]) as calls:
                y = layer(x)
                self.assertEqual(y[:, :, :9, :9].count_nonzero(), 0)
                self.assertTrue(torch.equal(y[:, :, 9:], x[:, :, 9:]))
                self.assertEqual(calls[0]['logical_omitted_fc_rows_each'], 810)
                self.assertEqual(calls[0]['producer_calls_cancelled'], 0)
            self.assertTrue(torch.equal(layer(x), x))
            with self.assertRaises(RuntimeError):
                with masked_branch(layer, []):
                    raise RuntimeError('probe abort')
            self.assertEqual(len(layer._forward_hooks), 0)

    def test_round_robin_population(self):
        rows = [{'target_id': str(i), 'sequence': 'a' if i < 3 else 'b'} for i in range(6)]
        self.assertEqual([x['target_id'] for x in choose_targets(rows, 4)], ['0', '3', '1', '4'])


if __name__ == '__main__':
    unittest.main()
