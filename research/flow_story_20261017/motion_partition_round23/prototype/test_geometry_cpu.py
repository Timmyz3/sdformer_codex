"""Predefined synthetic geometry checks; no dataset/model imports or inputs."""
import hashlib
import json
from pathlib import Path
import time
import unittest

import numpy as np
import partition_geometry as p


class GeometryChecks(unittest.TestCase):
    def test_exact_cover_and_areas(self):
        for i in range(4):
            masks = p.coverage(i)
            self.assertTrue(np.array_equal(masks.sum(0), np.ones((3, 3))))
            self.assertEqual(sorted(masks.sum((1, 2)).tolist()), [3, 6])
            self.assertFalse(bool((masks[0] & masks[1]).any()))

    def test_complete_time_batch_channel_conservation(self):
        for seed in (0, 7, 19):
            x = np.random.default_rng(seed).normal(size=(10, 2, 3, 3, 384)).astype(np.float32)
            for i in range(4):
                result = p.pool_and_expand(x, i)
                self.assertEqual(result['representatives'].shape, (10, 2, 2, 384))
                self.assertEqual(result['expanded'].shape, x.shape)
                weighted = (result['representatives'] * result['areas'][None, None, :, None]).sum(-2)
                np.testing.assert_allclose(weighted, x.astype(np.float64).sum((2, 3)), rtol=0, atol=2e-14)
                for k, mask in enumerate(p.coverage(i)):
                    actual = result['expanded'][..., mask, :]
                    expected = np.broadcast_to(result['representatives'][..., k:k+1, :], actual.shape)
                    self.assertTrue(np.array_equal(actual, expected))

    def test_translation_fixed_tie_and_direction_boundaries(self):
        q = np.broadcast_to(np.array([4., -1.]), (3, 3, 2)).copy()
        result = p.choose_partition(q)
        self.assertEqual(result['name'], 'H1')
        self.assertTrue(np.array_equal(result['scores'], np.zeros(4)))
        for i in range(4):
            q = np.empty((3, 3, 2), np.float64)
            masks = p.coverage(i)
            q[masks[0]] = (2., 0.)
            q[masks[1]] = (0., 2.)
            # Equal flow magnitudes, different directions; the full vector is used.
            self.assertTrue(np.array_equal(np.linalg.norm(q, axis=-1), np.full((3, 3), 2.)))
            chosen = p.choose_partition(q)
            self.assertEqual(chosen['index'], i)
            self.assertEqual(chosen['scores'][i], 0.)
            self.assertEqual(np.count_nonzero(chosen['scores'] == 0), 1)

    def test_same_interface_two_and_384_channels(self):
        rng = np.random.default_rng(5)
        q = rng.normal(size=(3, 3, 2))
        generic = np.zeros((3, 3, 384), np.float64)
        generic[..., :2] = q
        a, b = p.choose_partition(q), p.choose_partition(generic)
        self.assertEqual(a['index'], b['index'])
        np.testing.assert_allclose(a['scores'] * (2 / 384), b['scores'], atol=1e-16, rtol=1e-14)
        # A diagonal discontinuity cannot be represented exactly by any allowed rectangle split.
        diagonal = np.zeros((3, 3, 2), np.float64)
        diagonal[np.eye(3, dtype=bool)] = (1., -1.)
        self.assertTrue(bool((p.distortion_scores(diagonal) > 0).all()))

    def test_independent_mask_math_and_fixed_producer_rows(self):
        x = np.arange(10 * 3 * 3 * 2, dtype=np.float64).reshape(10, 3, 3, 2)
        independent = []
        for i in range(4):
            restored = np.empty_like(x)
            for mask in p.coverage(i):
                restored[:, mask, :] = x[:, mask, :].sum(1, keepdims=True) / mask.sum()
            independent.append(float(np.square(x - restored).sum() / x.size))
        np.testing.assert_array_equal(p.distortion_scores(x), independent)
        g = p.producer_geometry(10, 2)
        self.assertEqual(g['native_rows_per_FC'], 180)
        self.assertEqual(g['selected_rows_per_FC'], 40)
        self.assertEqual(g['measured_model_FC_calls'], 0)

    def test_rejects_invalid_shapes_values_and_indices(self):
        bad = [np.zeros((3, 2, 2)), np.zeros((3, 3, 0)), np.zeros((3, 3, 2), np.int32),
               np.full((3, 3, 2), np.nan), np.full((3, 3, 2), np.inf)]
        for values in bad:
            with self.assertRaises(ValueError):
                p.choose_partition(values)
        for index in (-1, 4, 1.5):
            with self.assertRaises(ValueError):
                p.pool_and_expand(np.zeros((3, 3, 2)), index)


if __name__ == '__main__':
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryChecks))
    here = Path(__file__).resolve().parent
    receipt = {'schema': 'flow_r23_two_slot_geometry_cpu_v1',
        'status': 'PASS' if result.wasSuccessful() else 'FAIL',
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'wall_seconds': time.monotonic() - start,
        'source_sha256': {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ('partition_geometry.py', 'test_geometry_cpu.py', 'PROTOCOL_V1.md')},
        'random_seeds': [0, 7, 19, 5], 'real_data_read': False,
        'model_forward_count': 0, 'GPU_used': False, 'trained_models': 0,
        'quality_or_hardware_gain_claim': False, 'producer_geometry_is_derived_not_measured': True}
    output = here / 'CPU_GEOMETRY_V1.json'
    if output.exists():
        raise SystemExit('Refusing to overwrite previous CPU receipt')
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
