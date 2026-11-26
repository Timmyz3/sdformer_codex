#!/usr/bin/env python3
"""CPU synthetic component checks; no dataset, checkpoint, GPU or training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.dont_write_bytecode = True
import torch
from torch import nn
import static_score_extensions as ext

HERE = Path(__file__).resolve().parent
PROBE = HERE.parents[1] / 'flow_story_20261011/preflight/cpu_numerics/native_mlp_probe.py'
spec = importlib.util.spec_from_file_location('breadth_score_native_probe', PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
torch.set_num_threads(2)
MEASUREMENTS = []


def native(dtype=torch.float64):
    return probe.make_mlp(8, dtype)


class Checks(unittest.TestCase):
    def test_fpgm_independent_scalar_reference_and_chunking(self):
        m = native()
        rows = m.fc1.weight.detach().tolist()
        ref = torch.tensor([sum(math.sqrt(sum((a-b)**2 for a, b in zip(x, y)))
                                for y in rows) for x in rows], dtype=torch.float64)
        for chunk in (1, 3, 64):
            score = ext.fpgm_score(m, distance_row_chunk=chunk)
            self.assertTrue(torch.allclose(ref, score, atol=1e-12, rtol=1e-12))
            MEASUREMENTS.append({'check': 'FPGM_scalar_reference', 'row_chunk': chunk,
                                 'max_absolute_error': float((ref-score).abs().max())})

    def test_fpgm_direction_low_distance_deleted(self):
        m = native()
        with torch.no_grad():
            m.fc1.weight.zero_()
            m.fc1.weight[-1, 0] = 10
        score = ext.fpgm_score(m)
        self.assertTrue(torch.equal(score[:-1], torch.full((31,), 10., dtype=torch.float64)))
        self.assertEqual(float(score[-1]), 310.)
        _, report = ext.build_static_score_candidate(m, 'fpgm', keep_ratio=.25)
        self.assertEqual(report['blocks']['']['selection']['keep_indices'], list(range(7)) + [31])

    def test_identical_rows_tie_and_zero(self):
        m = native()
        with torch.no_grad(): m.fc1.weight.fill_(3.)
        self.assertTrue(torch.equal(ext.fpgm_score(m), torch.zeros(32, dtype=torch.float64)))
        _, report = ext.build_static_score_candidate(m, 'fpgm', keep_ratio=.5)
        self.assertEqual(report['blocks']['']['selection']['keep_indices'], list(range(16)))

    def test_fpgm_ignores_bias_gamma_without_folding(self):
        m = native(); score = ext.fpgm_score(m)
        with torch.no_grad():
            if m.fc1.bias is not None: m.fc1.bias.add_(123.)
            m.bn1.norm_layer.weight.mul_(-12.)
            m.bn1.norm_layer.bias.add_(12.)
        self.assertTrue(torch.equal(score, ext.fpgm_score(m)))

    def test_bn_gamma_abs_and_direction(self):
        m = native()
        with torch.no_grad(): m.bn1.norm_layer.weight.copy_(torch.arange(-16, 16, dtype=torch.float64))
        expected = torch.arange(-16, 16, dtype=torch.float64).abs()
        self.assertTrue(torch.equal(ext.bn1_abs_gamma_score(m), expected))
        _, report = ext.build_static_score_candidate(m, 'bn1_abs_gamma', keep_ratio=.25)
        ranked = sorted(range(32), key=lambda i: (-float(expected[i]), i))
        self.assertEqual(report['blocks']['']['selection']['keep_indices'], sorted(ranked[:8]))

    def test_bn_all_equal_is_index_control_not_learned_sparsity(self):
        m = native()
        with torch.no_grad(): m.bn1.norm_layer.weight.fill_(1.)
        _, report = ext.build_static_score_candidate(m, 'bn1_abs_gamma', keep_ratio=.25)
        selection = report['blocks']['']['selection']
        self.assertEqual(selection['keep_indices'], list(range(8)))
        self.assertEqual(selection['channels_tied_at_cutoff'], 32)

    def test_real_narrowing_all_ratios_and_rounding(self):
        for method in ext.METHODS:
            m = native()
            for ratio, group, width in ((.75, 4, 24), (.5, 4, 16), (.25, 4, 8), (.7, 4, 20)):
                out, report = ext.build_static_score_candidate(m, method, keep_ratio=ratio, pe_group_width=group)
                self.assertEqual(out.fc1.out_features, width)
                self.assertEqual(out.fc2.in_features, width)
                self.assertEqual(out.bn1.norm_layer.num_features, width)
                self.assertEqual(out.fc2.out_features, 8)
                self.assertEqual(out.sn2.spiking_neuron.T, 10)
                self.assertEqual(report['parameters_after'], sum(p.numel() for p in out.parameters()))
                self.assertGreater(report['parameter_reduction'], 0)

    def test_allkeep_exact_full_native_branch(self):
        torch.manual_seed(42)
        for dtype in (torch.float64, torch.float32):
            m = native(dtype)
            x = torch.randn(10, 1, 11, 13, 8, dtype=dtype)
            for method in ext.METHODS:
                out, report = ext.build_static_score_candidate(m, method, keep_ratio=1.)
                with torch.inference_mode(): before, after = m(x), out(x)
                self.assertGreater(int(torch.count_nonzero(before)), 0)
                self.assertTrue(torch.equal(before, after))
                self.assertEqual(report['parameter_reduction'], 0)
                MEASUREMENTS.append({'check': 'allkeep_native', 'method': method, 'dtype': str(dtype),
                                     'shape': list(x.shape), 'max_absolute_error': float((before-after).abs().max())})

    def test_narrow_equals_dense_post_spike_channel_mask(self):
        torch.manual_seed(118)
        for method in ext.METHODS:
            for ratio in (.75, .5, .25):
                m = native()
                out, report = ext.build_static_score_candidate(m, method, keep_ratio=ratio)
                ref = copy.deepcopy(m)
                keep = report['blocks']['']['selection']['keep_indices']
                mask = torch.zeros(32, dtype=torch.float64); mask[keep] = 1.
                # Removal is after sn2: masking BN1's input would not represent
                # deletion because PSN has a nonzero temporal bias.
                handle = ref.sn2.register_forward_hook(lambda mod, args, y: y*mask)
                x = torch.randn(10, 2, 4, 5, 8, dtype=torch.float64)
                with torch.inference_mode(): expected, actual = ref(x), out(x)
                handle.remove()
                self.assertGreater(float(expected.abs().sum()), 0)
                error = float((expected-actual).abs().max())
                self.assertTrue(torch.allclose(expected, actual, atol=1e-12, rtol=1e-12))
                MEASUREMENTS.append({'check': 'narrow_vs_dense_post_sn2_mask', 'method': method,
                                     'keep_ratio': ratio, 'max_absolute_error': error})

    def test_parent_unchanged_rng_grad_and_no_shared_storage(self):
        m = native(); state = copy.deepcopy(m.state_dict()); rng = torch.get_rng_state().clone()
        before_sha = ext.model_state_sha256(m)
        for method in ext.METHODS:
            out, report = ext.build_static_score_candidate(m, method, keep_ratio=.5)
            self.assertEqual(report['source_model_state_sha256'], before_sha)
            self.assertEqual(ext.model_state_sha256(m), before_sha)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for name, value in state.items():
                self.assertTrue(torch.equal(value, m.state_dict()[name]))
                self.assertNotEqual(m.state_dict()[name].data_ptr(), out.state_dict()[name].data_ptr())
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            self.assertEqual([p.requires_grad for p in m.parameters()], [p.requires_grad for p in out.parameters()])

    def test_nested_only_selected_path_and_report_json(self):
        m = nn.ModuleDict({'a': native(), 'b': native()}).eval()
        out, report = ext.build_static_score_candidate(m, 'fpgm', ['b'], keep_ratio=.5)
        self.assertEqual(out['a'].fc1.out_features, 32)
        self.assertEqual(out['b'].fc1.out_features, 16)
        self.assertEqual(report['mlp_paths'], ['b'])
        self.assertEqual(report['collection'], 'weight_only_no_data')
        self.assertIsNone(report['fit_ids'])
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)
        self.assertEqual(report['blocks']['b']['scores_sha256'], ext.digest(report['blocks']['b']['scores']))

    def test_rebuild_and_strict_state_load(self):
        m = native(); out, a = ext.build_static_score_candidate(m, 'bn1_abs_gamma', keep_ratio=.5)
        rebuilt, b = ext.build_static_score_candidate(m, 'bn1_abs_gamma', keep_ratio=.5)
        rebuilt.load_state_dict(out.state_dict(), strict=True)
        self.assertEqual(a['width_and_score_plan_sha256'], b['width_and_score_plan_sha256'])
        self.assertEqual(ext.model_state_sha256(out), ext.model_state_sha256(rebuilt))
        with self.assertRaises(RuntimeError): m.load_state_dict(out.state_dict(), strict=True)

    def test_state_binding_changes_for_weight_and_bn_buffer(self):
        m = native(); first = ext.model_state_sha256(m)
        with torch.no_grad(): m.fc1.weight[0, 0].add_(1.)
        second = ext.model_state_sha256(m)
        self.assertNotEqual(first, second)
        with torch.no_grad(): m.bn1.norm_layer.running_mean[0].add_(1.)
        self.assertNotEqual(second, ext.model_state_sha256(m))

    def test_invalid_options(self):
        m = native()
        for ratio in (0, -1, 1.1, float('nan'), True, .01):
            with self.assertRaises(ValueError): ext.build_static_score_candidate(m, 'fpgm', keep_ratio=ratio)
        for group in (0, -1, True, 64):
            with self.assertRaises(ValueError): ext.build_static_score_candidate(m, 'fpgm', keep_ratio=.5, pe_group_width=group)
        for chunk in (0, -1, True, 1.1):
            with self.assertRaises(ValueError): ext.fpgm_score(m, distance_row_chunk=chunk)
        with self.assertRaises(ValueError): ext.build_static_score_candidate(m, 'svs', keep_ratio=.5)
        for paths in ([], ['missing'], ['', '']):
            with self.assertRaises(ValueError): ext.build_static_score_candidate(m, 'fpgm', paths, keep_ratio=.5)

    def test_nonfinite_weights_gamma_or_unselected_state_rejected(self):
        for name in ('fc1_weight', 'gamma', 'bn2_buffer'):
            m = native()
            with torch.no_grad():
                if name == 'fc1_weight': m.fc1.weight[0, 0] = float('nan')
                elif name == 'gamma': m.bn1.norm_layer.weight[0] = float('inf')
                else: m.bn2.norm_layer.running_mean[0] = float('nan')
            for method in ext.METHODS:
                with self.assertRaises(ValueError): ext.build_static_score_candidate(m, method, keep_ratio=.5)

    def test_affine_less_bn_rejected_for_gamma(self):
        m = native(); m.bn1.norm_layer = nn.BatchNorm2d(32, affine=False, dtype=torch.float64).eval()
        with self.assertRaises(ValueError): ext.bn1_abs_gamma_score(m)

    def test_shared_modules_params_and_hooks_rejected(self):
        m = native(); aliased = nn.ModuleDict({'a': m, 'b': m})
        with self.assertRaises(ValueError): ext.build_static_score_candidate(aliased, 'fpgm', keep_ratio=.5)
        owner = nn.ModuleDict({'a': native()}); owner.register_parameter('external', owner['a'].fc1.weight)
        with self.assertRaises(ValueError): ext.build_static_score_candidate(owner, 'fpgm', keep_ratio=.5)
        m = native(); hook = m.fc1.register_forward_hook(lambda mod, inp, out: None)
        try:
            with self.assertRaises(ValueError): ext.build_static_score_candidate(m, 'fpgm', keep_ratio=.5)
        finally: hook.remove()

    def test_dependency_pin_relocation_and_corruption(self):
        previous = ext._PRIOR_ROOT
        try:
            with tempfile.TemporaryDirectory(prefix='flow_score_dependencies_') as tmp:
                root = Path(tmp)
                source = ext._static().SVS_PATH
                target = root/'transfer_experiments/svs/svs_mlp.py'
                target.parent.mkdir(parents=True); target.write_bytes(source.read_bytes())
                self.assertEqual(ext.configure_prior_root(root), str(target))
                ext.build_static_score_candidate(native(), 'bn1_abs_gamma', keep_ratio=.5)
                target.write_text('changed source\n')
                with self.assertRaises(ValueError): ext.build_static_score_candidate(native(), 'fpgm', keep_ratio=.5)
        finally: ext._PRIOR_ROOT = previous


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    payload = {'schema': 'flow_breadth_static_score_cpu_checks_v1', 'passed': result.wasSuccessful(),
               'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'torch_version': str(torch.__version__), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
               'cuda_initialized': torch.cuda.is_initialized(), 'dataset_or_checkpoint_loaded': False,
               'GPU_or_training_executed': False, 'measurements': MEASUREMENTS,
               'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (HERE/'static_score_extensions.py', Path(__file__), HERE/'static_methods.py', PROBE)}}
    (HERE/'static_score_extensions_cpu_checks.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
