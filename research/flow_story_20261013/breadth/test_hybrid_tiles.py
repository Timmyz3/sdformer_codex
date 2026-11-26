import json
from pathlib import Path
import types
import unittest

import torch
from torch import nn
import test_bn_scope as fixture
import hybrid_tile_context as hybrid
import run_hybrid_tiles as runner


class PSN(fixture.TemporalPSN):
    def __init__(self):
        super().__init__()
        self.T = 10


class Spike(nn.Module):
    def __init__(self):
        super().__init__()
        self.spiking_neuron = PSN()
    def forward(self, x):
        return self.spiking_neuron(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnf, self.use_checkpoint = 'ADD', False
        self.mlp = fixture.native_mlp_class()()
        self.mlp.sn1, self.mlp.sn2 = Spike(), Spike()
    def forward(self, x):
        return x+self.mlp(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.before, self.block, self.after = fixture.Norm(3), Block(), fixture.Norm(3)
    def forward(self, x):
        norm = lambda m, y: m(y.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        return norm(self.after, self.block(norm(self.before, x)))


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = Path(__file__).resolve().parents[2]/'flow_story_20261012'
        cls.original = hybrid.base.import_file('hybrid_cpu_original_tile', cls.prior/'deployment_contract/tile_mlp.py')
        torch.set_num_threads(1)
    def model(self):
        torch.manual_seed(199)
        model = Model().double()
        fixture.scope.apply_scope(model, 'MLP_CMA176', fixture.artifact(model), torch, 1, 4)
        return model
    def context(self, model, policy):
        return hybrid.hybrid_tiled_mlp_model(model, 'actual_skip', policy, self.original, torch,
                                           expected_mlp_blocks=1, expected_total_bn=4)

    def test_actual_rows_external_full_grid_and_restoration(self):
        model = self.model()
        original_mlp = model.block.mlp
        signature = fixture.scope.base.bn_signature(model, torch)
        x = torch.randn(10, 1, 11, 13, 3, dtype=torch.float64)
        policy = lambda name, u: torch.tensor([[True, False], [False, True]], device=u.device)
        observed = []
        handles = [m.norm_layer.register_forward_pre_hook(lambda m, xs: observed.append(tuple(xs[0].shape)))
                   for m in (model.before, model.after)]
        with torch.inference_mode(), self.context(model, policy) as wrappers:
            result = model(x)
            record = wrappers[0].records[0]
            self.assertEqual(record['observed']['fc1_rows'], 10*(9*9+2*4))
            self.assertEqual(record['observed']['fc2_rows'], 10*(9*9+2*4))
            self.assertEqual(record['observed']['native_mlp_calls'], 2)
        for handle in handles:
            handle.remove()
        self.assertEqual(observed, [(10, 3, 11, 13), (10, 3, 11, 13)])
        self.assertEqual(tuple(result.shape), tuple(x.shape))
        self.assertIs(model.block.mlp, original_mlp)
        self.assertEqual(signature, fixture.scope.base.bn_signature(model, torch))
        # The original module remains unmodified and still rejects this mixed-scope graph.
        with self.assertRaises(ValueError):
            self.original.require_frozen(model)

    def test_guard_modes_and_norm_scope(self):
        policy = lambda name, u: torch.ones((2, 2), dtype=torch.bool)
        for corruption in ('training', 'external_frozen', 'MLP_current', 'residual'):
            model = self.model()
            if corruption == 'training':
                model.before.train()
            elif corruption == 'external_frozen':
                model.before.norm_layer.track_running_stats = True
            elif corruption == 'MLP_current':
                model.block.mlp.bn1.norm_layer.track_running_stats = False
            else:
                model.block.cnf = 'AND'
            with self.assertRaises(ValueError), self.context(model, policy):
                pass

    def test_exception_restores_originals(self):
        model = self.model()
        original = model.block.mlp
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with self.context(model, lambda name, x: None):
                raise RuntimeError('intentional')
        self.assertIs(model.block.mlp, original)

    def test_manifest_and_cloned_loop_only_two_overrides(self):
        cases = json.loads(Path(__file__).with_name('cases_hybrid9.json').read_text())['cases']
        self.assertEqual(cases, runner.CASES)
        args = types.SimpleNamespace(prior_root=self.prior, bn_payload=None)
        original_import = runner.base.import_file
        fn = runner.borrowed_inference(args, (None, {'arms': {}}))
        self.assertIs(fn.__code__, runner.base.run_inference.__code__)
        self.assertIs(runner.base.import_file, original_import)
        different = {k for k in fn.__globals__ if fn.__globals__[k] is not runner.base.run_inference.__globals__[k]}
        self.assertEqual(different, {'apply_bn', 'import_file'})
        with self.assertRaises(ValueError):
            fn.__globals__['import_file']('breadth_tile_executor', self.prior/'unexpected.py')


if __name__ == '__main__':
    run = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HybridTests))
    Path(__file__).with_name('hybrid_tiles_cpu_checks.json').write_text(json.dumps({
        'passed': run.wasSuccessful(), 'tests': run.testsRun, 'GPU_executed': False,
        'old_tile_sha256': hybrid.OLD_TILE_SHA,
        'scope': 'Native MLP.forward AST CPU fixture plus unchanged old TileMLP; whole external BN domain, actual FC rows, guard and restoration checks. Not full-network FP32 GPU accuracy.'}, indent=2)+'\n')
    raise SystemExit(0 if run.wasSuccessful() else 1)
