#!/usr/bin/env python3
"""CPU-only native PSN/MLP tests; no checkpoint, training dataset or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
import copy
import hashlib
import json
from pathlib import Path
import unittest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NATIVE = ROOT / 'SDformer/third_party/SDformerFlow'
sys.path[:0] = [str(ROOT / 'research/flow_story_20261011/preflight/cpu_numerics/deps'), str(NATIVE)]
from spikingjelly.activation_based import functional
from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp
from svs_mlp import (svs_ranks, svs_score, SVSAccumulator, magnitude_score,
                     select_hidden_channels, narrow_mlp_deepcopy, narrow_model_deepcopy, FORMULA)

MEASUREMENTS = []


def model(spike_norm='BN'):
    torch.manual_seed(219)
    m = MS_Spiking_Mlp(8, hidden_features=16, norm_layer='BN', drop=0.,
        num_steps=10, spike_norm=spike_norm, neuron_type='psn', v_th=.1,
        v_reset=None, surrogate_fun='surrogate.ATan()', tau=2., detach_reset=True).double().eval()
    functional.set_step_mode(m, 'm')
    with torch.no_grad():
        for sn in (m.sn1.spiking_neuron, m.sn2.spiking_neuron):
            sn.weight.normal_(0, .35); sn.bias.fill_(.04)
        for wrapper in (m.bn1, m.bn2):
            bn = wrapper.norm_layer
            bn.weight.uniform_(.8, 1.2); bn.bias.uniform_(-.1, .1)
            if bn.running_mean is not None:
                bn.running_mean.normal_(0, .1); bn.running_var.uniform_(.7, 1.3)
                bn.num_batches_tracked.fill_(7)
    return m


def spikes():
    # [T,B,C,H,W]: zero / rank-one / sample-dependent-rank /
    # temporally alternating rank-one whose TIME MEAN has rank two.
    s = torch.zeros(10, 2, 4, 3, 4)
    s[:, :, 1] = 1
    s[:, 0, 2, :3, :3] = torch.eye(3)
    s[::2, :, 3, 0, 0] = 1
    s[1::2, :, 3, 1, 1] = 1
    return s


class SVSTests(unittest.TestCase):
    def test_formula_time_mean_rank_then_minibatch_mean(self):
        s = spikes()
        self.assertEqual(svs_ranks(s, 'TBCHW').tolist(), [[0, 1, 3, 2], [0, 1, 0, 2]])
        self.assertEqual(svs_score(s, 'TBCHW').tolist(), [0., 1., 1.5, 2.])
        self.assertTrue(torch.equal(svs_score(s, 'TBCHW', channel_chunk=1),
                                    svs_score(s.permute(0, 1, 3, 4, 2), 'TBHWC', channel_chunk=3)))

    def test_accumulator_unequal_batches_and_identity_guards(self):
        s = spikes()
        a = SVSAccumulator(['a', 'b', 'c'], 3)
        a.update(s, ['a', 'b'], 'TBCHW')
        a.update(s[:, :1], ['c'], 'TBCHW')
        self.assertEqual(a.result()['scores'], [0., 1., 2., 2.])
        with self.assertRaises(ValueError): a.update(s[:, :1], ['c'], 'TBCHW')
        with self.assertRaises(ValueError): a.update(s[:, :1], ['invalid'], 'TBCHW')
        fresh = SVSAccumulator(['a', 'b'], 1)
        with self.assertRaises(ValueError): fresh.update(s, ['a', 'b'], 'TBCHW')

    def test_score_input_rejections(self):
        s = spikes()
        x = s.clone(); x[0, 0, 0, 0, 0] = .5
        with self.assertRaises(ValueError): svs_score(x, 'TBCHW')
        x[0, 0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError): svs_score(x, 'TBCHW')
        with self.assertRaises(ValueError): svs_score(s[:9], 'TBCHW')
        with self.assertRaises(ValueError): svs_score(s, 'BTHWC')

    def test_selection_ties_width_and_group_alignment(self):
        scores = [1., 4., 4., 4., 2., 3., 0., 1.]
        result = select_hidden_channels(scores, 4, 4)
        self.assertEqual(result['keep_indices'], [1, 2, 3, 5])
        self.assertEqual(select_hidden_channels([1.] * 8, 4, 4)['keep_indices'], [0, 1, 2, 3])
        for width, group in ((3, 4), (0, 1), (9, 1), (4, 0)):
            with self.assertRaises(ValueError): select_hidden_channels(scores, width, group)

    def test_all_keep_native_output_state_and_storage(self):
        m = model(); original = copy.deepcopy(m.state_dict())
        new, report = narrow_mlp_deepcopy(m, list(range(16)), 8)
        x = torch.randn(10, 2, 3, 4, 8, dtype=torch.float64)
        with torch.inference_mode():
            old_out, new_out = m(x), new(x)
        self.assertTrue(torch.equal(old_out, new_out))
        self.assertEqual(report['parameters_removed'], 0)
        for k, t in m.state_dict().items():
            self.assertTrue(torch.equal(t, original[k]), k)
            self.assertTrue(torch.equal(t, new.state_dict()[k]), k)
            self.assertNotEqual(t.data_ptr(), new.state_dict()[k].data_ptr(), k)
        MEASUREMENTS.append({'case': 'all_keep', 'max_abs_error': float((old_out-new_out).abs().max()), **report})

    def test_noncontiguous_pruning_dependency_and_real_parameters(self):
        m = model(); keep = [0, 2, 3, 7, 9, 10, 12, 15]
        snapshot = copy.deepcopy(m.state_dict())
        new, report = narrow_mlp_deepcopy(m, keep, 4)
        self.assertEqual(tuple(new.fc1.weight.shape), (8, 8))
        self.assertEqual(tuple(new.fc2.weight.shape), (8, 8))
        self.assertEqual(new.bn1.norm_layer.num_features, 8)
        self.assertEqual(new.sn2.spiking_neuron.T, 10)
        self.assertEqual(report['parameters_removed'], 144)
        for name, tensor in new.state_dict().items():
            if name == 'fc1.weight' or name.startswith('bn1.norm_layer.') and tensor.ndim == 1:
                self.assertTrue(torch.equal(tensor, snapshot[name][keep]), name)
            elif name == 'fc2.weight':
                self.assertTrue(torch.equal(tensor, snapshot[name][:, keep]), name)
            else:
                self.assertTrue(torch.equal(tensor, snapshot[name]), name)
        x = torch.randn(10, 2, 3, 4, 8, dtype=torch.float64)
        rows = []
        h1 = new.fc1.register_forward_hook(lambda mod, inp, out: rows.append(('fc1_out', out.shape[-1])))
        h2 = new.fc2.register_forward_pre_hook(lambda mod, inp: rows.append(('fc2_in', inp[0].shape[-1])))
        with torch.inference_mode():
            narrowed = new(x); dense = m(x)
        h1.remove(); h2.remove()
        self.assertEqual(rows, [('fc1_out', 8), ('fc2_in', 8)])
        # Independent dense masked reference: keep original dimensions, zero
        # discarded HIDDEN spikes at sn2. This verifies closure, NOT equality
        # with the unpruned network, whose output generally changes.
        mask = torch.zeros(16, dtype=torch.float64); mask[keep] = 1
        hook = m.sn2.register_forward_hook(lambda mod, inp, out: out * mask)
        with torch.inference_mode(): masked = m(x)
        hook.remove()
        self.assertTrue(torch.allclose(narrowed, masked, atol=1e-12, rtol=1e-12))
        self.assertGreater(float((dense-narrowed).abs().max()), 1e-5)
        self.assertEqual(narrowed.shape, x.shape)
        for k, t in m.state_dict().items(): self.assertTrue(torch.equal(t, snapshot[k]), k)
        MEASUREMENTS.append({'case': 'pruned', 'dense_masked_closure_max_abs': float((narrowed-masked).abs().max()),
                             'unpruned_output_change_max_abs': float((dense-narrowed).abs().max()), **report})

    def test_current_stat_bn_static_channel_closure(self):
        # Static channel slicing differs from spatial tile cancellation:
        # BN normalizes each retained channel over the same complete domain.
        m = model('BN_notrack'); keep = list(range(8))
        new, _ = narrow_mlp_deepcopy(m, keep, 8)
        self.assertIsNone(new.bn1.norm_layer.running_mean)
        x = torch.randn(10, 1, 3, 4, 8, dtype=torch.float64)
        mask = torch.zeros(16, dtype=torch.float64); mask[keep] = 1
        hook = m.sn2.register_forward_hook(lambda mod, inp, out: out * mask)
        with torch.inference_mode(): a, b = m(x), new(x)
        hook.remove()
        self.assertTrue(torch.allclose(a, b, atol=1e-12, rtol=1e-12))

    def test_magnitude_same_width_real_narrow_control(self):
        m = model(); score = magnitude_score(m)
        expected = m.fc1.weight.detach().abs().mean(dim=1)
        self.assertTrue(torch.equal(score, expected))
        selection = select_hidden_channels(score, 8, 4)
        new, report = narrow_mlp_deepcopy(m, selection['keep_indices'], 4)
        self.assertEqual(report['after']['hidden_channels'], 8)
        self.assertEqual(report['parameters_removed'], 144)
        self.assertEqual(new.fc2.in_features, 8)

    def test_gradient_and_width_aware_reload(self):
        m = model(); keep = list(range(0, 16, 2))
        new, _ = narrow_mlp_deepcopy(m, keep, 4)
        x = torch.randn(10, 1, 3, 4, 8, dtype=torch.float64, requires_grad=True)
        new(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters()))
        rebuilt, _ = narrow_mlp_deepcopy(model(), keep, 4)
        rebuilt.load_state_dict(new.state_dict(), strict=True)
        with self.assertRaises(RuntimeError): model().load_state_dict(new.state_dict(), strict=True)

    def test_invalid_closure_and_hooks_rejected(self):
        m = model()
        for indices in ([1, 0], [0, 0], [-1, 1], [0, 16], [], [True, 2]):
            with self.assertRaises(ValueError): narrow_mlp_deepcopy(m, indices)
        bad = copy.deepcopy(m); bad.sn2.spiking_neuron.channel_scale = torch.nn.Parameter(torch.ones(16))
        with self.assertRaises(ValueError): narrow_mlp_deepcopy(bad, list(range(8)), 4)
        bad = copy.deepcopy(m); bad.norm_layer = 'GN'
        with self.assertRaises(ValueError): narrow_mlp_deepcopy(bad, list(range(8)), 4)
        hook = m.sn2.register_forward_hook(lambda *args: None)
        with self.assertRaises(ValueError): narrow_mlp_deepcopy(m, list(range(8)), 4)
        hook.remove()

    def test_model_copy_residual_and_alias_rejection(self):
        class Residual(torch.nn.Module):
            def __init__(self): super().__init__(); self.mlp = model()
            def forward(self, x): return x + self.mlp(x)
        m = Residual().eval(); x = torch.randn(10, 1, 3, 4, 8, dtype=torch.float64)
        new, report = narrow_model_deepcopy(m, {'mlp': list(range(8))}, 4)
        self.assertEqual(m.mlp.fc1.out_features, 16)
        self.assertEqual(new.mlp.fc1.out_features, 8)
        self.assertEqual(new(x).shape, x.shape)
        self.assertEqual(report['parameters_before'] - report['parameters_after'], 144)
        m.alias = m.mlp
        with self.assertRaises(ValueError): narrow_model_deepcopy(m, {'mlp': list(range(8))}, 4)


if __name__ == '__main__':
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SVSTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    paths = ['models/STSwinNet_SNN/Spiking_swin_transformer3D.py',
             'models/STSwinNet_SNN/Spiking_submodules.py', 'models/STSwinNet_SNN/Spiking_modules.py']
    report = {'schema': 'flow_svs_native_cpu_checks_v1', 'passed': result.wasSuccessful(),
              'tests_run': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures),
              'GPU_initialized': torch.cuda.is_initialized(), 'torch': str(torch.__version__), 'CPU_threads': 1,
              'source_sha256': {p: hashlib.sha256((NATIVE/p).read_bytes()).hexdigest() for p in paths},
              'implementation_sha256': hashlib.sha256((HERE/'svs_mlp.py').read_bytes()).hexdigest(),
              'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'formula': FORMULA,
              'measurements': MEASUREMENTS,
              'scope': 'Synthetic binary maps and native small T10 PSN/MLP; no trained weights, AEE, recovery, quantization or PE cycle claim'}
    (HERE/'cpu_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
