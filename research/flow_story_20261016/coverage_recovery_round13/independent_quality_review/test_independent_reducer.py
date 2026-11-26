#!/usr/bin/env python3
"""Synthetic metric and decision-rule checks; no model/data/result imports."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import reduce_raw_quality as q

HERE = Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_strict_boundaries_unknown_and_voxel_cancellation(self):
        gt = np.zeros((2, 3, 3), dtype=np.float32)
        gt[:, 1, 1] = [3, 4]
        valid = np.ones((3, 3), dtype=bool)
        events = np.zeros((2, 3, 3), dtype=np.float32)
        events[:, 1, 1] = [1, -1]
        masks = q.masks_from_original(gt, valid, events, 5., 5.)
        self.assertEqual(int(masks['B'].sum()), 0)
        self.assertEqual(int(masks['L'].sum()), 0)
        self.assertEqual(int(masks['B_unknown'].sum()), 8)
        self.assertEqual(int(masks['voxel_nonzero'].sum()), 1)
        self.assertEqual(int(masks['voxel_zero'].sum()), 8)
        masks = q.masks_from_original(gt, valid, events, 4.99, 4.99)
        self.assertEqual(int(masks['B_and_L'].sum()), 1)
        valid[0, 1] = False
        masks = q.masks_from_original(gt, valid, events, 0., 0.)
        self.assertEqual(int(masks['B'].sum()), 0)
        self.assertEqual(int(masks['B_unknown'].sum()), 8)
        self.assertEqual(int(masks['all_valid'].sum()), 8)

    def test_known_endpoint_error_empty_and_weighting(self):
        gt = np.zeros((2, 3, 3), dtype=np.float32)
        pred = np.zeros_like(gt)
        pred[0], pred[1] = 3, 4
        valid = np.ones((3, 3), dtype=bool)
        masks = q.masks_from_original(gt, valid, np.zeros((2, 3, 3)), 1., 1.)
        metrics = q.measure(pred, gt, masks)
        self.assertEqual(metrics['all_valid'], {'valid_pixels': 9, 'error_sum': 45., 'AEE': 5.})
        self.assertEqual(metrics['B'], {'valid_pixels': 0, 'error_sum': 0., 'AEE': None})
        values = [{'valid_pixels': 1, 'error_sum': 10., 'AEE': 10.},
                  {'valid_pixels': 9, 'error_sum': 9., 'AEE': 1.},
                  {'valid_pixels': 0, 'error_sum': 0., 'AEE': None}]
        total = q.aggregate(values)
        self.assertEqual(total['frame_mean_AEE'], 5.5)
        self.assertEqual(total['pixel_weighted_AEE'], 1.9)
        self.assertEqual((total['frames_with_region'], total['empty_region_frames']), (2, 1))

    def test_four_families_and_sequence_gate_are_required(self):
        def table(v):
            return {scope: {r: {'frame_mean_AEE': v} for r in ('all_valid', 'B', 'L')}
                    for scope in ('overall', 'sequence_a', 'sequence_b')}
        summary = {'initial/' + a: table(1.) for a in q.ARMS}
        summary.update({a + '/' + b: table(1.) for a in q.ARMS for b in q.ARMS})
        self.assertTrue(q.four_family_gates(summary)['CC']['ordinary_base_quality_pass'])
        changes = [('native/native', 'normal_references', 1.04),
                   ('CC/CC', 'matched_recovered_references', 1.04),
                   ('CC/native', 'shared_fine_preserved', 1.04)]
        for key, family, value in changes:
            bad = copy.deepcopy(summary)
            bad[key] = table(value)
            gates = q.four_family_gates(bad)['CC']
            self.assertFalse(gates['ordinary_base_quality_pass'])
            self.assertFalse(all(v['pass'] for v in gates[family].values()))
        bad = copy.deepcopy(summary)
        bad['CC/native'] = table(.9)
        self.assertFalse(q.four_family_gates(bad)['CC']['own_checkpoint_references']['native']['pass'])
        bad = copy.deepcopy(summary)
        bad['CC/CC']['sequence_b']['L']['frame_mean_AEE'] = 1.051
        self.assertFalse(q.four_family_gates(bad)['CC']['ordinary_base_quality_pass'])
        a, b = table(1.03), table(1.)
        self.assertTrue(q.compare(a, b)['pass'])
        a['overall']['all_valid']['frame_mean_AEE'] = 1.03000001
        self.assertFalse(q.compare(a, b)['pass'])

    def test_null_and_mismatch_rejection(self):
        a = {'overall': {r: {'frame_mean_AEE': None if r == 'B' else 1.} for r in ('all_valid', 'B', 'L')}}
        result = q.compare(a, a)
        self.assertTrue(result['pass'])
        self.assertFalse(result['scopes']['overall']['B']['evidence'])
        bad = copy.deepcopy(a)
        bad['overall']['B']['frame_mean_AEE'] = 0.
        with self.assertRaises(ValueError):
            q.compare(a, bad)
        audit = q.NumericAudit()
        audit.check(1., 1., 'equal')
        audit.check(None, None, 'empty')
        with self.assertRaises(ValueError):
            audit.check(1., 1.01, 'wrong')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt = {'schema': 'r13_independent_quality_synthetic_checks_v1', 'passed': result.wasSuccessful(),
               'tests_run': result.testsRun, 'numpy': np.__version__,
               'source_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                 for name in ('reduce_raw_quality.py', 'test_independent_reducer.py')},
               'errors': [str(x) for x in result.errors], 'failures': [str(x) for x in result.failures],
               'scope': 'Synthetic arrays only. No runner/analyzer, actual check arrays, prediction payloads, GPU or SSH.'}
    out = HERE / 'synthetic_checks_v1.json'
    if out.exists():
        raise ValueError('Refuse check receipt overwrite')
    out.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
