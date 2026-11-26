"""Synthetic CPU checks; no actual datasets, backbone, head fitting or CUDA."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import evaluate_coarse_dense as e
from coarse_head import make_tile_targets

FLOW = {'direction': 'forward', 'time_interval': 'synthetic same target interval',
        'units': 'original_image_pixels'}


def labels(stage, gt=None, valid=None):
    hw, stride, _ = e.GEOMETRY[stage]
    if gt is None:
        gt = torch.zeros(2, 480, 640)
    if valid is None:
        valid = torch.ones(480, 640, dtype=torch.bool)
    return make_tile_targets(gt, valid, feature_hw=hw, feature_stride_pixels=(stride, stride),
                             flow_contract=FLOW, image_origin_xy=(0., 0.))


class DenseExpansionChecks(unittest.TestCase):
    def test_tail_pixels_covered_once_without_flow_unit_rescaling(self):
        # Independent pixel-coordinate lookup checks every pixel, including tails.
        yy, xx = np.indices((480, 640))
        for stage, (_, stride, _) in e.GEOMETRY.items():
            target = labels(stage)
            ny, nx = target['valid'].shape[1:]
            values = torch.zeros(1, ny, nx, 2)
            for y in range(ny):
                for x in range(nx):
                    values[0, y, x] = torch.tensor([100*y+x, -7*y-3*x])
            dense = e.expand_tiles(values, target)
            expected_y, expected_x = yy//(9*stride), xx//(9*stride)
            np.testing.assert_array_equal(dense[0], 100*expected_y+expected_x)
            np.testing.assert_array_equal(dense[1], -7*expected_y-3*expected_x)
            self.assertEqual(int(target['support_pixel_count'].sum()), 480*640)
            tail_h, tail_w = 480-(ny-1)*9*stride, 640-(nx-1)*9*stride
            self.assertEqual(int(target['support_pixel_count'][0, -1, -1]), tail_h*tail_w)

    def test_empty_GT_placeholders_never_enter_metric_or_denominator(self):
        gt = torch.full((2, 480, 640), 999.)
        valid = torch.zeros(480, 640, dtype=torch.bool)
        valid[0, 0] = True; valid[0, 1] = True
        gt[:, 0, 0] = torch.tensor([1., 0.]); gt[:, 0, 1] = torch.tensor([3., 0.])
        target = labels('2', gt, valid)
        self.assertTrue(torch.isnan(target['flow'][~target['valid']]).all())
        dense = e.expand_tiles(target['flow'], target, gt_valid=valid, diagnostic_reference=True)
        self.assertTrue(np.isfinite(dense).all())
        self.assertTrue((dense[:, 200, 200] == 0).all())
        # The actual shared metric verifies empty B remains null and all-GT count is two.
        path = Path(__file__).resolve().parents[2]/'flow_story_20261012/evaluation/flow_eval_contract.py'
        spec = importlib.util.spec_from_file_location('coarse_dense_test_metrics', path)
        metrics = importlib.util.module_from_spec(spec); spec.loader.exec_module(metrics)
        meter = metrics.MetricAccumulator()
        meter.update('synthetic_1', 'synthetic', dense, gt.numpy(),
                     {'all_valid': valid.numpy(), 'B': np.zeros((480, 640), dtype=bool)})
        result = meter.summary()['regions']
        self.assertEqual(result['all_valid']['valid_pixels'], 2)
        self.assertEqual(result['all_valid']['frame_mean_AEE'], 1.)
        self.assertIsNone(result['B']['frame_mean_AEE'])
        changed = valid.clone(); changed[200, 200] = True
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            e.expand_tiles(target['flow'], target, gt_valid=changed, diagnostic_reference=True)

    def test_shifted_support_or_wrong_grid_refused(self):
        target = labels('0'); pred = torch.zeros_like(target['flow'])
        wrong = copy.deepcopy(target); wrong['image_origin_xy'] = [1.5, 1.5]
        with self.assertRaises(ValueError): e.expand_tiles(pred, wrong)
        with self.assertRaises(ValueError): e.expand_tiles(pred[:, :-1], target)


class FoldedHeadChecks(unittest.TestCase):
    def test_folded_linear_consumes_raw_descriptor_exactly_once(self):
        raw = torch.tensor([[[[5., 11.], [7., 17.]]]])
        # This equals W*((raw-mean)/divisor)+b. The artifact already folded it.
        original_w = torch.tensor([[2., 4.], [-1., 3.]])
        mean, divisor, bias = torch.tensor([1., 5.]), torch.tensor([2., 3.]), torch.tensor([.5, -.25])
        folded_w = original_w/divisor
        folded_b = bias-folded_w@mean
        stage = {'channels': 2, 'normalizer_audit_only': {'mean': mean, 'divisor': divisor},
                 'methods': {'linear': {'input_contract': e.RAW_INPUT,
                                       'state_dict': {'weight': folded_w, 'bias': folded_b}}}}
        result = e.predict_tiles(raw, stage, 'linear')
        torch.testing.assert_close(result, F.linear((raw-mean)/divisor, original_w, bias), rtol=1e-6, atol=1e-6)
        self.assertFalse(torch.allclose(result, F.linear((raw-mean)/divisor, folded_w, folded_b)))
        stage['methods']['linear']['input_contract'] = 'normalized descriptor'
        with self.assertRaisesRegex(ValueError, 'repeat normalization'):
            e.predict_tiles(raw, stage, 'linear')

    def test_C16_and_constant_use_frozen_state_without_updates(self):
        gen = torch.Generator().manual_seed(19)
        raw = torch.randn(1, 2, 3, 4, generator=gen)
        state = {k: torch.randn(shape, generator=gen) for k, shape in {
            'fc1.weight': (16, 4), 'fc1.bias': (16,), 'fc2.weight': (2, 16), 'fc2.bias': (2,)}.items()}
        snapshot = {k: v.clone() for k, v in state.items()}
        stage = {'channels': 4, 'methods': {'C16_GELU2': {'input_contract': e.RAW_INPUT, 'state_dict': state},
                                            'constant_component_median': {'constant_dxdy': torch.tensor([2., -3.])}}}
        reference = F.linear(F.gelu(F.linear(raw, state['fc1.weight'], state['fc1.bias'])), state['fc2.weight'], state['fc2.bias'])
        torch.testing.assert_close(e.predict_tiles(raw, stage, 'C16_GELU2'), reference, rtol=0, atol=0)
        constant = e.predict_tiles(raw, stage, 'constant_component_median')
        torch.testing.assert_close(constant, torch.tensor([2., -3.]).expand(1, 2, 3, 2), rtol=0, atol=0)
        for k in state: torch.testing.assert_close(state[k], snapshot[k], rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_cache_weights_and_fit_contract_binding_reject_mutations(self):
        cache_sha, contract_sha = 'a'*64, 'b'*64
        cache = {'schema': 'flow_coarse_feature_cache_v1', 'flow_contract': FLOW,
                 'contract_sha256': contract_sha, 'BN_scope': 'MLP_CMA176',
                 'label_geometry': 'logical_cell_support_origin0_v1'}
        weights = {'schema': 'flow_coarse_folded_head_weights_v1', 'flow_contract': FLOW,
                   'provenance': {'cache_sha256': cache_sha, 'GPU_used': False, 'source_sha256': {
                       'fit_coarse_cache.py': e.FIT_SHA, 'coarse_head.py': e.HEAD_SHA, 'group_policy.py': e.GROUP_SHA}},
                   'recipe': {'development_selection': False, 'device': 'cpu', 'steps_per_trainable_head': 1000,
                              'seed': 20261013}, 'stages': {'0': {}, '2': {}}}
        e.validate_binding(cache, weights, cache_sha, contract_sha)
        with self.assertRaises(ValueError): e.validate_binding(cache, weights, 'c'*64, contract_sha)
        with self.assertRaises(ValueError): e.validate_binding(cache, weights, cache_sha, 'c'*64)
        for kind in ('dev_selected', 'short_training', 'source', 'stage', 'GPU'):
            bad = copy.deepcopy(weights)
            if kind == 'dev_selected': bad['recipe']['development_selection'] = True
            elif kind == 'short_training': bad['recipe']['steps_per_trainable_head'] = 10
            elif kind == 'source': bad['provenance']['source_sha256']['coarse_head.py'] = '0'*64
            elif kind == 'stage': del bad['stages']['2']
            else: bad['provenance']['GPU_used'] = True
            with self.assertRaises(ValueError): e.validate_binding(cache, bad, cache_sha, contract_sha)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
