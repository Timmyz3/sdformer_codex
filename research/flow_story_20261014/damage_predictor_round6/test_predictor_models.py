"""Small mathematical checks for the new model interface; no network/data."""
import unittest
import numpy as np
import predictor_models as m


class PredictorChecks(unittest.TestCase):
    def test_affine_folding_and_fp32(self):
        x = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])
        model = m.ridge_fit(x, 2*x[:, 0] + 1, 1.)
        w, b = m.folded(model)
        np.testing.assert_allclose(m.predict(model, x), x @ w + b, rtol=1e-12)
        np.testing.assert_allclose(m.predict_fp32(model, x), m.predict(model, x), rtol=1e-6)
        self.assertEqual(m.predict_fp32(model, x).dtype, np.float32)

    def test_constant_and_train_only(self):
        model = m.ridge_fit(np.ones((4, 3)), np.array([1., 2., 3., 4.]), 1.)
        np.testing.assert_array_equal(model['std'], np.ones(3))
        before = model['mean'].copy()
        np.testing.assert_array_equal(m.predict(model, np.ones((2, 3))*100), np.ones(2)*2.5)
        np.testing.assert_array_equal(model['mean'], before)

    def test_geometric_invariance_and_edges(self):
        v = np.zeros((3, 3, 2), dtype=np.float32)
        v[..., 0] = np.arange(3)[None, :]
        stats = m.vector_statistics(v)
        # Six horizontal differences are unit, six vertical differences zero.
        self.assertAlmostEqual(float(stats[3]), .5)
        self.assertEqual(float(stats[4]), 1.)
        rotated = v[..., ::-1] * np.array([1, -1], dtype=np.float32)
        np.testing.assert_allclose(stats, m.vector_statistics(rotated), atol=1e-6)
        np.testing.assert_array_equal(m.vector_statistics(np.zeros_like(v)), np.zeros(6))

    def test_group_support_and_ties(self):
        x = np.tile(np.repeat(np.arange(16), 24), (3, 3, 1)).astype(np.float32)
        stats = m.richer_statistics(x)
        np.testing.assert_array_equal(stats[:16], np.arange(16))
        np.testing.assert_array_equal(stats[16:], np.zeros(16))
        self.assertEqual(m.top1(np.ones(12)), 0)
        with self.assertRaises(ValueError):
            m.top1(np.ones(11))

    def test_pca_subspace(self):
        x = np.array([[a, b, a, b] for a in (-3., -1., 1., 3.) for b in (-2., 0., 2.)])
        p = m.pca_fit(x)
        z = (x-p['mean'])/p['std']
        recon = m.predict(p, x) @ p['beta'].T
        np.testing.assert_allclose(recon, z, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
