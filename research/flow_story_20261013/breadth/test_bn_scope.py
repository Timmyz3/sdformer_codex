import ast
import copy
import json
from pathlib import Path
import unittest

import torch
from torch import nn
import run_bn_scope as scope

NATIVE = Path(__file__).resolve().parents[3]/'SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py'


class TemporalPSN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10)*.2)
        self.bias = nn.Parameter(torch.randn(10, 1)*.1)
    def forward(self, x):
        return (torch.addmm(self.bias, self.weight, x.flatten(1)) >= 0).to(x.dtype).view_as(x)


class Norm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm_layer = nn.BatchNorm2d(channels)
        self.norm_layer.num_batches_tracked.fill_(7)
    def forward(self, x):
        return self.norm_layer(x.flatten(0, 1)).view_as(x)


class MlpBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_layer = 'BN'
        self.fc1, self.fc2 = nn.Linear(3, 8, bias=False), nn.Linear(8, 3, bias=False)
        self.bn1, self.bn2 = Norm(8), Norm(3)
        self.sn1, self.sn2 = TemporalPSN(), TemporalPSN()
        self.drop1, self.drop2 = nn.Dropout(.1), nn.Dropout(.1)


def native_mlp_class():
    tree = ast.parse(NATIVE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MS_Spiking_Mlp')
    namespace = {'Spiking_Mlp': MlpBase}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NATIVE), 'exec'), namespace)
    return namespace['MS_Spiking_Mlp']


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.before = Norm(3)
        self.mlp = native_mlp_class()()
        self.after = Norm(3)


def artifact(model):
    all_bn, _ = scope.bn_inventory(model, torch, 1, 4)
    state_hash = scope.extended.non_bn_digest(model, all_bn)
    return {'source_non_bn_state_sha256': state_hash, 'after_non_bn_state_sha256': state_hash,
            'buffers': {name: {'running_mean': [.3]*layer.num_features, 'running_var': [1.7]*layer.num_features,
                               'num_batches_tracked': 176} for name, layer in all_bn.items()}}


class ScopeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(174)
        torch.set_num_threads(1)

    def test_four_scopes_exact_bn_partition_no_parameter_change(self):
        for name in scope.SCOPES:
            model = Toy().double()
            params = {k: v.detach().clone() for k, v in model.named_parameters()}
            record = scope.apply_scope(model, name, artifact(model), torch, 1, 4)
            self.assertEqual(len(record['frozen_BN_names']), 0 if name == 'P_paper' else 2)
            for path, module in model.named_modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    self.assertEqual(module.running_mean is not None, path in record['frozen_BN_names'])
                    if path in record['frozen_BN_names']:
                        self.assertEqual(module.num_batches_tracked.item(), 176 if name.endswith('CMA176') else 7)
            self.assertTrue(all(torch.equal(v, dict(model.named_parameters())[k]) for k, v in params.items()))
            self.assertTrue(all(not m.training for m in model.modules()))

    def test_native_mlp_local_closure_with_external_current_stats(self):
        model = Toy().double()
        scope.apply_scope(model, 'MLP_original', None, torch, 1, 4)
        x = torch.randn(10, 1, 11, 13, 3, dtype=torch.float64)
        mask = torch.zeros((11, 13), dtype=torch.bool)
        mask[:9, :9] = True
        mask[9:, 9:] = True
        norm = lambda module, x: module(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        with torch.inference_mode():
            u = norm(model.before, x)
            dense_branch = model.mlp(u) * mask[None, None, :, :, None]
            tiled_branch = torch.zeros_like(u)
            for ys, xs in ((slice(0, 9), slice(0, 9)), (slice(9, 11), slice(9, 13))):
                tiled_branch[:, :, ys, xs, :] = model.mlp(u[:, :, ys, xs, :].contiguous())
            expected = norm(model.after, u+dense_branch)
            actual = norm(model.after, u+tiled_branch)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        self.assertIsNone(model.after.norm_layer.running_mean)
        # An external current-stat BN still consumes the whole completed residual grid.
        self.assertGreater((actual[:, :, :9, :9, :] - norm(model.after, (u+tiled_branch)[:, :, :9, :9, :])).abs().max().item(), 1e-4)

    def test_missing_or_foreign_cma_rejected(self):
        model = Toy()
        with self.assertRaises(ValueError):
            scope.apply_scope(model, 'MLP_CMA176', None, torch, 1, 4)
        bad = artifact(model)
        bad['source_non_bn_state_sha256'] = 'a'*64
        with self.assertRaises(ValueError):
            scope.apply_scope(model, 'nonMLP_CMA176', bad, torch, 1, 4)

    def test_exact_frozen_loop_without_global_mutation(self):
        original = scope.base.apply_bn
        callback = lambda *args: None
        function = scope.borrowed_inference(callback)
        self.assertIs(function.__code__, scope.base.run_inference.__code__)
        self.assertIs(function.__globals__['apply_bn'], callback)
        self.assertIs(scope.base.apply_bn, original)
        self.assertIs(scope.base.run_inference.__globals__['apply_bn'], original)
        self.assertEqual(json.loads(Path(__file__).with_name('cases_bn_scope4.json').read_text())['cases'], scope.CASES)


if __name__ == '__main__':
    run = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScopeTests))
    Path(__file__).with_name('bn_scope_cpu_checks.json').write_text(json.dumps({
        'passed': run.wasSuccessful(), 'tests': run.testsRun, 'GPU_executed': False,
        'native_forward_source': str(NATIVE), 'native_forward_source_sha256': scope.base.sha(NATIVE),
        'scope': 'Actual native MS_Spiking_Mlp.forward AST on small CPU FP64 modules. Confirms dependency scope, not GPU FP32 skip equivalence.'}, indent=2)+'\n')
    raise SystemExit(0 if run.wasSuccessful() else 1)
