"""Geometric/support/gradient checks, independent of any real data or outcomes."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
import structure_models as m


class Checks(unittest.TestCase):
    def test_labels_include_zero_and_sparse_support(self):
        gt = np.zeros((2, 480, 640), np.float32)
        valid = np.zeros((480, 640), np.uint8)
        for y, x, a, b in ((0, 0, 2., -1.), (15, 15, 4., 3.), (479, 639, -6., 8.)):
            valid[y, x] = 1; gt[:, y, x] = [a, b]
        gt[:, 8, 9] = np.nan  # Invalid values cannot poison an offline mean.
        flow, count = m.pixel_labels(gt, valid)
        self.assertEqual(int(count.sum()), 3)
        np.testing.assert_array_equal(flow[0, 0], [3., 1.])
        np.testing.assert_array_equal(flow[29, 39], [-6., 8.])
        self.assertEqual(int(count[29, 39]), 1)
        self.assertTrue(np.isfinite(flow).all())

    def test_true_tail_pooling_and_score_input_scope(self):
        x = np.arange(5*7, dtype=np.float32).reshape(1, 5, 7, 1)
        y = m.pool_expand3(x)
        self.assertEqual(float(y[0, 4, 6, 0]), (27+34)/2)
        self.assertEqual(float(y[0, 0, 0, 0]), 8.)
        prefix = np.zeros((1, 30, 40, 384), np.float32)
        q = np.ones((1, 30, 40, 2), np.float32)*7
        inp = m.spatial_inputs(prefix, q)
        self.assertEqual(inp['raw'].shape, (12, 386, 9, 9))
        self.assertEqual(inp['motion'].shape, (12, 6, 9, 9))
        self.assertEqual(inp['ordinary'].shape, (12, 34, 9, 9))
        np.testing.assert_array_equal(inp['motion'][:, 2:4], 0.)
        np.testing.assert_array_equal(inp['context'][-1, -2:], [1., 1.])
        self.assertEqual(m.top1(np.ones(12)), 0)
        self.assertEqual(m.top1(np.arange(12)), 11)
        a = np.random.default_rng(7).normal(size=(12, 6, 9, 9)).astype(np.float32)
        a[:, 0] *= 5; a[:, 1] *= .2
        scaler = m.fit_motion_scaler(a)
        self.assertEqual(len(set(scaler['std'][0, :4, 0, 0].tolist())), 1)
        z = m.scale(a, scaler)
        np.testing.assert_allclose(z[0, :2, 0, 0]-z[0, :2, 0, 1],
                                   (a[0, :2, 0, 0]-a[0, :2, 0, 1])/scaler['std'][0, 0, 0, 0], atol=1e-6)

    def test_edge_cover_and_real_gradients(self):
        torch.set_num_threads(2); torch.manual_seed(11)
        edge = m.MotionEdges()
        self.assertEqual(len(edge.src), 144)
        pairs = list(zip(edge.src.tolist(), edge.dst.tolist()))
        self.assertEqual(len(set(pairs)), 144)
        for a, b in pairs:
            self.assertIn((b//9-a//9, b%9-a%9), ((0, 1), (1, 0)))
        np.testing.assert_allclose(edge.membership.sum(0).numpy(), np.ones(9), rtol=1e-6)
        for model, channels in ((edge, 6), (m.OrdinaryCNN(6), 6), (m.OrdinaryCNN(34), 34), (m.OrdinaryPrefixCNN(), 386)):
            x = torch.randn(12, channels, 9, 9)
            c = torch.randn(12, 34)
            before = m.state_sha(model)
            y = model(x, c)
            self.assertEqual(tuple(y.shape), (12,))
            loss = ((y-torch.linspace(-1, 1, 12))**2).mean()
            opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
            loss.backward()
            self.assertTrue(all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in model.parameters()))
            opt.step()
            self.assertNotEqual(m.state_sha(model), before)

    def test_fit_only_projection_and_folded_fp32(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(2, 30, 40, 384)).astype(np.float32)
        w = rng.normal(size=(384, 2)).astype(np.float32)/384
        truth = x@w+np.array([1., -2.], np.float32)
        counts = np.ones(x.shape[:-1], np.int64)
        counts[:, 0] = 0
        truth[:, 0] = np.nan
        fitted = m.ridge_motion(x, truth, counts)
        pred = m.project(fitted, x)
        self.assertEqual(pred.dtype, np.float32)
        self.assertLess(float(np.square(pred[counts > 0]-truth[counts > 0]).mean()), 1e-5)
        self.assertEqual(fitted['fit_cells'], 2320)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    out = Path(sys.argv[1])
    if out.exists(): raise FileExistsError(out)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out.write_text(json.dumps({'schema': 'flow_r21_structure_cpu_v1', 'pass': result.wasSuccessful(),
                              'tests': result.testsRun, 'real_data_read': False, 'CUDA_initialized': torch.cuda.is_initialized()}, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
