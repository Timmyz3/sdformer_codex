#!/usr/bin/env python3
"""Small CPU/synthetic checks. No dataset/checkpoint/GPU/optimizer steps."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.dont_write_bytecode = True
import torch
from torch import nn
from static_methods import (build_static_candidate, factor_linear, choose_rank,
                            FactoredLinear, digest, _svs, SVS_PATH, configure_prior_root)

HERE = Path(__file__).resolve().parent
PROBE = HERE.parents[1] / 'flow_story_20261011/preflight/cpu_numerics/native_mlp_probe.py'
spec = importlib.util.spec_from_file_location('breadth_native_probe', PROBE)
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
MEASUREMENTS = []


def native():
    return probe.make_mlp(8, torch.float64)


def model():
    return nn.ModuleDict({'a': native(), 'b': native()}).eval()


def envelope(m):
    svs = _svs(); ids = ['syntheticTrain0', 'syntheticTrain1']
    provenance = {key: hashlib.sha256(key.encode()).hexdigest() for key in (
        'checkpoint_sha256', 'data_contract_sha256', 'source_sha256', 'bn_policy_sha256', 'geometry_sha256')}
    provenance.update(fit_ids_sha256=digest(ids), split='train')
    blocks = {}
    for path in ('a', 'b'):
        generator = torch.Generator().manual_seed(231)
        spikes = torch.randint(0, 2, (10, 2, 3, 4, m[path].fc1.out_features), generator=generator).double()
        accumulator = svs.SVSAccumulator(ids, 2)
        accumulator.update(spikes, ids)
        blocks[path] = accumulator.result()
    return {'schema': 'flow_breadth_svs_replay_v1', 'fit_ids': ids, 'provenance': provenance, 'blocks': blocks}, provenance


class Checks(unittest.TestCase):
    def test_relocated_dependency_still_requires_original_sha(self):
        original_root = SVS_PATH.parents[2]
        try:
            with tempfile.TemporaryDirectory(prefix='flow_static_dependency_') as temp:
                root = Path(temp); dest = root/'transfer_experiments/svs/svs_mlp.py'
                dest.parent.mkdir(parents=True); dest.write_bytes(SVS_PATH.read_bytes())
                self.assertEqual(configure_prior_root(root), str(dest))
                _, report = build_static_candidate(native(), 'magnitude', keep_ratio=.5)
                self.assertGreater(report['parameter_reduction'], 0)
                dest.write_text('different implementation\n')
                with self.assertRaises(ValueError): configure_prior_root(root)
                with self.assertRaises(ValueError): build_static_candidate(native(), 'magnitude', keep_ratio=.5)
        finally:
            configure_prior_root(original_root)

    def test_linear_fullrank_arbitrary_axes_bias_and_dtype(self):
        for dtype, tol in ((torch.float64, 1e-11), (torch.float32, 2e-5)):
            for i, o, bias in ((8, 19, True), (19, 8, False), (8, 8, True)):
                torch.manual_seed(89)
                linear = nn.Linear(i, o, bias=bias, dtype=dtype).eval()
                snapshot = copy.deepcopy(linear.state_dict())
                factor, report = factor_linear(linear, min(i, o))
                for shape in ((i,), (5, i), (10, 2, 3, 4, i)):
                    x = torch.randn(shape, dtype=dtype)
                    with torch.inference_mode():
                        old, new = linear(x), factor(x)
                    self.assertEqual(old.shape, new.shape)
                    self.assertTrue(torch.allclose(old, new, atol=tol, rtol=tol))
                for k, v in snapshot.items(): self.assertTrue(torch.equal(v, linear.state_dict()[k]))
                if bias: self.assertTrue(torch.equal(linear.bias, factor.up.bias))
                self.assertIsNone(factor.down.bias)
                self.assertEqual(report['parameters_after'], sum(p.numel() for p in factor.parameters()))
                MEASUREMENTS.append({'case': 'linear_fullrank', 'dtype': str(dtype), 'shape': [o, i],
                    'bias': bias, 'max_abs_error_last_5D': float((old-new).abs().max()), 'tolerance': tol})

    def test_truncated_svd_matches_independent_optimal_rank_matrix(self):
        torch.manual_seed(84)
        linear = nn.Linear(8, 24, dtype=torch.float64)
        factor, report = factor_linear(linear, 4)
        u, s, vh = torch.linalg.svd(linear.weight.detach(), full_matrices=False)
        expected = (u[:, :4]*s[:4]) @ vh[:4]
        self.assertTrue(torch.allclose(factor.up.weight @ factor.down.weight, expected, atol=1e-12, rtol=1e-12))
        self.assertAlmostEqual(report['weight_frobenius_error_after_dtype_cast'], float(torch.linalg.vector_norm(s[4:])), places=11)

    def test_rank_basis_is_not_parameter_ratio(self):
        linear = nn.Linear(8, 32, bias=True)
        self.assertEqual(choose_rank(linear, .75), 6)
        self.assertEqual(choose_rank(linear, .5), 4)
        r = choose_rank(linear, .5, rank_basis='parameter_ratio')
        self.assertLessEqual(r*(8+32)+32, .5*(8*32+32))
        with self.assertRaises(ValueError): choose_rank(linear, .01)
        with self.assertRaises(ValueError): choose_rank(linear, .5, rank=3)
        with self.assertRaises(ValueError): choose_rank(linear, rank=9)

    def test_allkeep_pruning_exact_and_original_unchanged(self):
        m = native(); before = copy.deepcopy(m.state_dict())
        x = torch.randn(10, 2, 3, 4, 8, dtype=torch.float64)
        for method in ('magnitude', 'random'):
            out, report = build_static_candidate(m, method, keep_ratio=1.)
            with torch.inference_mode(): self.assertTrue(torch.equal(m(x), out(x)))
            self.assertEqual(report['parameter_reduction'], 0)
            for k, v in before.items():
                self.assertTrue(torch.equal(v, m.state_dict()[k]))
                self.assertNotEqual(m.state_dict()[k].data_ptr(), out.state_dict()[k].data_ptr())

    def test_magnitude_uses_expected_rows_and_real_narrowing(self):
        m = model()
        with torch.no_grad():
            m['a'].fc1.weight.copy_(torch.arange(32).double()[:, None].expand(32, 8))
        out, report = build_static_candidate(m, 'magnitude', ['a'], keep_ratio=.5, pe_group_width=4)
        self.assertEqual(report['blocks']['a']['selection']['keep_indices'], list(range(16, 32)))
        self.assertEqual(out['a'].fc1.out_features, 16)
        self.assertEqual(out['a'].fc2.in_features, 16)
        self.assertEqual(out['a'].bn1.norm_layer.num_features, 16)
        self.assertEqual(out['b'].fc1.out_features, 32)
        self.assertEqual(out['a'].sn2.spiking_neuron.T, 10)

    def test_random_path_stability_rng_and_seed(self):
        m = model(); rng = torch.get_rng_state().clone()
        all_model, all_report = build_static_candidate(m, 'random', keep_ratio=.5, seed=17)
        _, sub = build_static_candidate(m, 'random', ['a'], keep_ratio=.5, seed=17)
        _, other = build_static_candidate(m, 'random', ['a'], keep_ratio=.5, seed=18)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        a = all_report['blocks']['a']['selection']['keep_indices']
        self.assertEqual(a, sub['blocks']['a']['selection']['keep_indices'])
        self.assertNotEqual(a, other['blocks']['a']['selection']['keep_indices'])
        self.assertEqual(all_report['parameters_after'], sum(p.numel() for p in all_model.parameters()))

    def test_svs_replay_bound_formula_and_fit_ids(self):
        m = model(); scores, expected = envelope(m)
        out, report = build_static_candidate(m, 'svs', keep_ratio=.5, svs_scores=scores, svs_expected_provenance=expected)
        self.assertEqual(out['a'].fc1.out_features, 16)
        self.assertEqual(report['svs_replay_canonical_digest'], digest(scores))
        for change in ('checkpoint', 'split', 'score', 'fit_ids', 'formula'):
            bad = copy.deepcopy(scores)
            if change == 'checkpoint': bad['provenance']['checkpoint_sha256'] = '0'*64
            if change == 'split': bad['provenance']['split'] = 'valid'
            if change == 'score': bad['blocks']['a']['scores'][0] += .01
            if change == 'fit_ids': bad['fit_ids'].reverse()
            if change == 'formula': bad['blocks']['a']['formula']['epsilon'] = 0.
            with self.assertRaises(ValueError):
                build_static_candidate(m, 'svs', keep_ratio=.5, svs_scores=bad, svs_expected_provenance=expected)
        with self.assertRaises(ValueError): build_static_candidate(m, 'svs', keep_ratio=.5)

    def test_native_fullrank_shape_BN_PSN_and_source_state(self):
        m = native(); before = copy.deepcopy(m.state_dict()); rng = torch.get_rng_state().clone()
        out, report = build_static_candidate(m, 'lowrank', keep_ratio=1.)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        x = torch.randn(10, 2, 3, 4, 8, dtype=torch.float64)
        with torch.inference_mode(): expected, actual = m(x), out(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=1e-10))
        self.assertTrue(all(torch.equal(v, m.state_dict()[k]) for k, v in before.items()))
        for k, v in before.items():
            if not k.startswith(('fc1.', 'fc2.')): self.assertTrue(torch.equal(v, out.state_dict()[k]))
        self.assertEqual(actual.shape, x.shape)
        self.assertGreater(report['parameters_after'], report['parameters_before'])
        MEASUREMENTS.append({'case': 'native_fullrank', 'shape': list(x.shape),
            'max_abs_error': float((actual-expected).abs().max()), 'tolerance': 1e-10, 'report': report})

    def test_native_lowrank_both_workpoints_and_linear_leaf_hooks(self):
        m = native(); x = torch.randn(10, 1, 3, 5, 8, dtype=torch.float64)
        for ratio in (.75, .5):
            out, report = build_static_candidate(m, 'lowrank', keep_ratio=ratio)
            rows = {}; handles = []
            for name, layer in out.named_modules():
                if isinstance(layer, nn.Linear):
                    handles.append(layer.register_forward_hook(lambda mod, inp, output, n=name:
                        rows.update({n: {'rows': inp[0].numel()//inp[0].shape[-1],
                                        'in': mod.in_features, 'out': mod.out_features}})))
            with torch.inference_mode(): result = out(x)
            for h in handles: h.remove()
            self.assertEqual(set(rows), {'fc1.down', 'fc1.up', 'fc2.down', 'fc2.up'})
            self.assertTrue(all(r['rows'] == 150 for r in rows.values()))
            self.assertTrue(bool(torch.isfinite(result).all()))
            self.assertEqual(result.shape, x.shape)
            self.assertEqual(out.bn1.norm_layer.num_features, 32)
            self.assertEqual(report['parameters_after'], sum(p.numel() for p in out.parameters()))
            MEASUREMENTS.append({'case': 'native_lowrank', 'requested_rank_fraction': ratio,
                                 'shape': list(x.shape), 'leaf_linear_observations': rows, 'report': report})

    def test_native_truncated_matches_dense_rank_reconstruction(self):
        m = native(); out, _ = build_static_candidate(m, 'lowrank', keep_ratio=.5)
        reference = copy.deepcopy(m)
        with torch.no_grad():
            for name in ('fc1', 'fc2'):
                f = getattr(out, name)
                getattr(reference, name).weight.copy_(f.up.weight @ f.down.weight)
        x = torch.randn(10, 1, 3, 4, 8, dtype=torch.float64)
        with torch.inference_mode(): a, b = out(x), reference(x)
        self.assertTrue(torch.allclose(a, b, atol=1e-10, rtol=1e-10))

    def test_single_factor_target_and_explicit_rank(self):
        m = model()
        out, report = build_static_candidate(m, 'lowrank', ['b'], ranks={'b': {'fc2': 3}}, factor_targets=('fc2',))
        self.assertIsInstance(out['b'].fc2, FactoredLinear)
        self.assertIsInstance(out['b'].fc1, nn.Linear)
        self.assertIsInstance(out['a'].fc2, nn.Linear)
        self.assertEqual(report['blocks']['b']['factors']['fc2']['rank'], 3)

    def test_factor_gradient_and_state_dict_reconstruction(self):
        m = native(); candidate, _ = build_static_candidate(m, 'lowrank', keep_ratio=.5)
        x = torch.randn(10, 1, 3, 4, 8, dtype=torch.float64, requires_grad=True)
        candidate(x).square().mean().backward()
        for name, p in candidate.named_parameters():
            if name.startswith(('fc1.', 'fc2.')):
                self.assertIsNotNone(p.grad)
                self.assertTrue(bool(torch.isfinite(p.grad).all()))
                self.assertGreater(float(p.grad.abs().sum()), 0.)
        rebuilt, _ = build_static_candidate(m, 'lowrank', keep_ratio=.5)
        rebuilt.load_state_dict(candidate.state_dict(), strict=True)
        with torch.inference_mode(): self.assertTrue(torch.equal(candidate(x), rebuilt(x)))

    def test_weight_fakequant_bias_BN_PSN_and_zero_row(self):
        m = native()
        with torch.no_grad(): m.fc1.weight[0].zero_()
        before = copy.deepcopy(m.state_dict())
        for bits in (8, 6, 4):
            candidate, report = build_static_candidate(m, 'weight_fakequant', bits=bits)
            self.assertEqual(report['parameter_reduction'], 0)
            self.assertEqual(candidate.fc1.weight.dtype, m.fc1.weight.dtype)
            self.assertTrue(bool((candidate.fc1.weight[0] == 0).all()))
            for k, v in before.items():
                self.assertTrue(torch.equal(v, m.state_dict()[k]))
                if k not in ('fc1.weight', 'fc2.weight'): self.assertTrue(torch.equal(v, candidate.state_dict()[k]))
            x = torch.randn(10, 1, 2, 3, 8, dtype=torch.float64)
            with torch.inference_mode(): y = candidate(x)
            self.assertEqual(y.shape, x.shape); self.assertTrue(bool(torch.isfinite(y).all()))
            MEASUREMENTS.append({'case': 'weight_fakequant', 'bits': bits, 'report': report})

    def test_fakequant_matches_scalar_definition_and_preserves_float32(self):
        m = native().float()
        candidate, report = build_static_candidate(m, 'weight_fakequant', bits=4, factor_targets=('fc1',))
        for c, row in enumerate(m.fc1.weight):
            scale = float(row.abs().max())/7 if bool(row.abs().max() > 0) else 1.
            expected = (row/scale).round().clamp(-7, 7)*scale
            self.assertTrue(torch.equal(expected, candidate.fc1.weight[c]))
        self.assertTrue(torch.equal(m.fc2.weight, candidate.fc2.weight))
        self.assertEqual(candidate.fc1.weight.dtype, torch.float32)

    def test_refuse_ambiguous_or_unclosed_graphs(self):
        m = model()
        bad_options = [dict(method='lowrank', keep_ratio=.5, ranks={'a': {'fc1': 3}}),
            dict(method='magnitude', keep_ratio=.01, pe_group_width=16),
            dict(method='lowrank', keep_ratio=.5, pe_group_width=16),
            dict(method='weight_fakequant', bits=3), dict(method='weight_fakequant', bits=8, keep_ratio=.5),
            dict(method='lowrank', keep_ratio=float('nan')), dict(method='magnitude', keep_ratio=.5, bits=8)]
        for options in bad_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError): build_static_candidate(m, **options)
        shared = nn.ModuleDict({'a': m['a'], 'b': m['a']})
        with self.assertRaises(ValueError): build_static_candidate(shared, 'magnitude', keep_ratio=.5)
        handle = m['a'].sn2.register_forward_hook(lambda m, x, y: None)
        with self.assertRaises(ValueError): build_static_candidate(m, 'lowrank', keep_ratio=.5)
        handle.remove()


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    artifact = {'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
        'failures': [str(item) for item in result.failures], 'errors': [str(item) for item in result.errors],
        'scope': 'CPU synthetic weights/inputs/metadata only; no real checkpoint or fit-score evidence',
        'GPU_initialized': torch.cuda.is_initialized(), 'optimizer_steps': 0,
        'torch': str(torch.__version__), 'python': sys.version.split()[0],
        'source_sha256': {str(p.relative_to(HERE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (HERE/'static_methods.py', Path(__file__), SVS_PATH, PROBE)},
        'measurements': MEASUREMENTS}
    (HERE/'static_methods_cpu_results.json').write_text(json.dumps(artifact, indent=2, allow_nan=False)+'\n')
    return 0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1


if __name__ == '__main__':
    raise SystemExit(main())
