#!/usr/bin/env python3
"""CPU-only calibration arithmetic audit. No model, dataset, pickle, or torch load.

Independent pooled per-channel sufficient statistics for a frozen-affine BN
contract. Population variance is intentional; this is not PyTorch EMA running_var.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import numpy as np


class ChannelMoments:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def update(self, x, channel_axis):
        x = np.asarray(x, dtype=np.float64)
        x = np.moveaxis(x, channel_axis, 0).reshape(x.shape[channel_axis], -1)
        if x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError('empty or nonfinite calibration tensor')
        n2 = x.shape[1]
        mu2 = x.mean(axis=1)
        m22 = ((x - mu2[:, None]) ** 2).sum(axis=1)
        if self.n == 0:
            self.n, self.mean, self.m2 = n2, mu2, m22
            return
        if self.mean.shape != mu2.shape:
            raise ValueError('channel count changed')
        delta = mu2 - self.mean
        n = self.n + n2
        self.m2 = self.m2 + m22 + delta ** 2 * (self.n * n2 / float(n))
        self.mean = self.mean + delta * (n2 / float(n))
        self.n = n

    def population_variance(self):
        if not self.n:
            raise ValueError('no observations')
        return np.maximum(self.m2 / self.n, 0.0)

    def affine(self, gamma, beta, eps):
        a = np.asarray(gamma) / np.sqrt(self.population_variance() + eps)
        return a, np.asarray(beta) - a * self.mean


def audit():
    rng = np.random.RandomState(20261011)
    # Real multi-step BN layout is T,B,C,H,W; B differs between these batches.
    parts = [rng.randn(2, 1, 3, 3, 4) - 4,
             rng.randn(2, 2, 3, 3, 4) * 2 + 7]
    stats = ChannelMoments()
    for x in parts:
        stats.update(x, 2)
    joined = np.concatenate([np.moveaxis(x, 2, 0).reshape(3, -1)
                             for x in parts], axis=1)
    mu_error = float(np.max(np.abs(stats.mean - joined.mean(axis=1))))
    var_error = float(np.max(np.abs(stats.population_variance() - joined.var(axis=1))))
    naive_variance = np.mean([np.moveaxis(x, 2, 0).reshape(3, -1).var(axis=1)
                              for x in parts], axis=0)
    naive_error = float(np.max(np.abs(naive_variance - joined.var(axis=1))))
    gamma, beta, eps = np.array([.5, -1, 2]), np.array([0., .2, -.3]), 1e-5
    a, b = stats.affine(gamma, beta, eps)
    dense = a[:, None] * joined + b[:, None]
    keep = np.arange(joined.shape[1]) % 3 != 0
    sparse = a[:, None] * joined[:, keep] + b[:, None]
    frozen_error = float(np.max(np.abs(dense[:, keep] - sparse)))
    # High-value selection deliberately exhibits conditional-statistics bias.
    selected = joined[:, joined[0] > np.median(joined[0])]
    selected_mu_error = float(np.max(np.abs(selected.mean(axis=1) - stats.mean)))
    old_buffers = {'mean': stats.mean.copy(), 'variance': stats.population_variance().copy()}
    immutable = all(np.array_equal(old_buffers[k], v) for k, v in
                    [('mean', stats.mean), ('variance', stats.population_variance())])
    assert mu_error < 1e-12 and var_error < 1e-12
    assert naive_error > 1 and frozen_error == 0 and selected_mu_error > 1 and immutable
    return {
        'scope': 'synthetic arithmetic only; no trained-model/AEE or hardware proof',
        'python': sys.version.split()[0], 'numpy': np.__version__,
        'torch_imported': False, 'GPU_used': False, 'checkpoint_loaded': False,
        'per_channel_count': stats.n,
        'pooled_mean_max_abs_error': mu_error,
        'pooled_population_variance_max_abs_error': var_error,
        'naive_average_batch_variance_max_abs_error': naive_error,
        'frozen_affine_subset_max_abs_error': frozen_error,
        'risk_selected_mean_max_abs_shift': selected_mu_error,
        'script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='bn_calibration_math_results.json')
    args = parser.parse_args()
    result = audit()
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
