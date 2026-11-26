"""Small native CPU gradient/structure/resume checks; no dataset or GPU training."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import model_api as api
import train_recovery as train

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT/'research/flow_story_20261012'
BREADTH = HERE.parent/'breadth'
PROBE = ROOT/'research/flow_story_20261011/preflight/cpu_numerics/native_mlp_probe.py'
spec = importlib.util.spec_from_file_location('recovery_cpu_native_probe', PROBE)
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
core = api.import_pinned('recovery_cpu_oldcore', PRIOR/'baseline/train_baseline.py',
                         '9628cad1c7dc7e20416ec9465303692f1b3860aaf6c6f3413345cc243b9e11e0')
MEASUREMENTS = []


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([probe.make_mlp(16, torch.float64) for _ in range(12)])
    def forward(self, x):
        return sum(m(x) for m in self.blocks)/12


def native_factory(*args, **kwargs):
    return Tiny()


def envelope(model):
    static, svs = api.dependencies(PRIOR, BREADTH)
    ids = ['synthetic0', 'synthetic1']
    expected = {k: api.digest(k) for k in ('checkpoint_sha256', 'data_contract_sha256', 'source_sha256',
                                          'bn_policy_sha256', 'geometry_sha256')}
    expected.update(fit_ids_sha256=api.digest(ids), split='train')
    blocks = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'MS_Spiking_Mlp':
            accumulator = svs.SVSAccumulator(ids, 2)
            spikes = torch.randint(0, 2, (10, 2, 3, 4, module.fc1.out_features), generator=torch.Generator().manual_seed(75))
            accumulator.update(spikes, ids)
            blocks[name] = accumulator.result()
    return {'schema': 'flow_breadth_svs_replay_v1', 'fit_ids': ids, 'provenance': expected, 'blocks': blocks}, expected


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(902)

    def initial(self, arm):
        source = Tiny(); artifact, expected = envelope(source)
        with patch.object(api, 'instantiate_native', native_factory):
            return api.build_initial_model({}, source.state_dict(), arm, source_root=ROOT, prior_root=PRIOR,
                breadth_root=BREADTH, source_sha256={'synthetic': 'a'*64}, svs_artifact=artifact, svs_expected=expected)

    def test_all_four_structures_backward_and_exact_endpoint_rebuild(self):
        for arm in api.ARMS:
            model, architecture, _ = self.initial(arm)
            model.train(); probe.functional.set_step_mode(model, 'm')
            x = torch.randn(10, 1, 3, 4, 16, dtype=torch.float64)
            loss = model(x).square().mean(); loss.backward()
            psn = [p for m in model.modules() if m.__class__.__name__ == 'PSN' for p in m.parameters(recurse=False)]
            self.assertTrue(all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in psn))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertTrue(all(m.track_running_stats and m.training and m.running_mean is not None
                                for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)))
            state = copy.deepcopy(model.state_dict())
            with patch.object(api, 'instantiate_native', native_factory), patch('torch.linalg.svd', side_effect=AssertionError('Endpoint must not SVD')):
                rebuilt = api.construct_model({}, architecture, source_root=ROOT, prior_root=PRIOR, breadth_root=BREADTH)
            rebuilt.load_state_dict(state, strict=True)
            self.assertTrue(train.same_tree(torch, state, rebuilt.state_dict()))
            self.assertTrue(all(p.requires_grad for p in rebuilt.parameters()))
            MEASUREMENTS.append({'arm': arm, 'PSN_parameters_with_finite_gradient': len(psn),
                                 'parameter_count': architecture['parameter_count'], 'endpoint_rebuild_exact_state': True})

    def test_augmentation_isolated_and_order_fixed(self):
        random.seed(14); np.random.seed(15); torch.manual_seed(16)
        before = core.capture_rng(torch, np)
        def draw():
            with train.augmentation_rng(torch, np, 'fixed-target'):
                return random.randint(0, 192), random.randint(0, 256), torch.rand(2), np.random.rand()
        first = draw(); self.assertTrue(train.same_tree(torch, before, core.capture_rng(torch, np)))
        torch.rand(77); random.random(); np.random.rand(23)
        second = draw(); self.assertTrue(train.same_tree(torch, first, second))
        manifest = json.loads((HERE/'manifests/gatefit4573_recovery.json').read_text())
        rows = [{'target_id': t, 'protocol_split': 'train'} for t in manifest['target_ids']]
        rows += [{'target_id': 'valid-only-metadata', 'protocol_split': 'valid'},
                 {'target_id': 'other-train-role', 'protocol_split': 'train'}]
        roles = {t: 'gate_fit' for t in manifest['target_ids']}
        roles['other-train-role'] = 'gate_development'
        validated = train.validate_order(manifest, rows, roles, manifest['canonical_sha256'], manifest['roles_sha256'])
        self.assertEqual([r['target_id'] for r in validated], manifest['target_ids'])
        bad = copy.deepcopy(manifest); bad['target_ids'][0], bad['target_ids'][1] = bad['target_ids'][1], bad['target_ids'][0]
        with self.assertRaises(ValueError):
            train.validate_order(bad, rows, roles, manifest['canonical_sha256'], manifest['roles_sha256'])

    def test_dense_and_lowrank_cpu_optimizer_resume_rng(self):
        for arm in ('dense', 'lowrank_param25'):
            initial, architecture, _ = self.initial(arm)
            state = copy.deepcopy(initial.state_dict())
            data = torch.randn(10, 1, 3, 4, 16, generator=torch.Generator().manual_seed(41), dtype=torch.float64)
            def make():
                with patch.object(api, 'instantiate_native', native_factory):
                    model = api.construct_model({}, architecture, source_root=ROOT, prior_root=PRIOR, breadth_root=BREADTH)
                model.load_state_dict(state); model.train(); probe.functional.set_step_mode(model, 'm')
                return model, torch.optim.AdamW(model.parameters(), lr=.00025, weight_decay=.01)
            def step(model, optimizer):
                probe.functional.reset_net(model)
                jitter = torch.rand(())*.01
                loss = (model(data)+jitter).square().mean(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100., error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                probe.functional.reset_net(model)
            direct, opt_direct = make(); torch.manual_seed(778)
            for _ in range(4): step(direct, opt_direct)
            first, opt_first = make(); torch.manual_seed(778)
            for _ in range(2): step(first, opt_first)
            snapshot = copy.deepcopy(first.state_dict()); optimizer_state = copy.deepcopy(opt_first.state_dict()); rng = core.capture_rng(torch, np)
            resumed, opt_resumed = make(); resumed.load_state_dict(snapshot); opt_resumed.load_state_dict(optimizer_state)
            core.restore_rng(torch, np, rng)
            # DataLoader iterator construction must not consume dropout RNG.
            list(torch.utils.data.DataLoader([1, 2], generator=torch.Generator().manual_seed(91)))
            for _ in range(2): step(resumed, opt_resumed)
            self.assertTrue(train.same_tree(torch, direct.state_dict(), resumed.state_dict()))
            self.assertTrue(train.same_tree(torch, opt_direct.state_dict(), opt_resumed.state_dict()))

    def test_safe_checkpoint_guards_and_immutable_visit(self):
        model, architecture, projection = self.initial('dense')
        contract = {'arm': 'dense', 'architecture_sha256': api.digest(architecture), 'source_sha256': architecture['source_sha256'], 'effective_config': {}}
        payload = {'schema_version': api.SCHEMA, 'contract': contract, 'contract_sha256': api.digest(contract),
                   'architecture': architecture, 'architecture_sha256': api.digest(architecture), 'initial_projection': projection,
                   'model_state_dict': model.state_dict(), 'optimizer_state_dict': {}, 'scheduler_state_dict': None,
                   'rng_state': core.capture_rng(torch, np), 'effective_config': {}, 'training_order': ['t0'],
                   'training': {'visited_targets': 0, 'completed_optimizer_steps': 0, 'zero_valid_crops': 0, 'charged_wall_seconds': 1.}}
        with tempfile.TemporaryDirectory(prefix='flow_recovery_cpu_') as directory:
            path = Path(directory)
            side = train.save_checkpoint(torch, core, path, payload)
            loaded, _ = api.read_checkpoint(path/side['checkpoint_file'], side['checkpoint_sha256'], expected_contract=contract)
            self.assertTrue(train.same_tree(torch, payload['model_state_dict'], loaded['model_state_dict']))
            self.assertEqual(train.save_checkpoint(torch, core, path, payload), side)
            modified = copy.deepcopy(payload)
            key = next(k for k, v in modified['model_state_dict'].items() if v.is_floating_point())
            modified['model_state_dict'][key].add_(1)
            with self.assertRaises(ValueError): train.save_checkpoint(torch, core, path, modified)
            with self.assertRaises(ValueError): api.read_checkpoint(path/side['checkpoint_file'], 'a'*64)
            with self.assertRaises(ValueError): api.read_checkpoint(path/side['checkpoint_file'], side['checkpoint_sha256'], expected_contract={'wrong': True})
            altered = copy.deepcopy(payload); altered['effective_config'] = {'model': {'dropout': .8}}
            bad_path = path/'bad_configuration.pt'
            torch.save(altered, bad_path)
            core.atomic_json(bad_path.with_suffix('.json'), dict(side, checkpoint_file=bad_path.name, checkpoint_sha256=api.sha(bad_path)))
            with self.assertRaises(ValueError): api.read_checkpoint(bad_path, api.sha(bad_path))
            old = dict(side, schema_version='flow_a_native_v2_checkpoint_v1')
            core.atomic_json((path/side['checkpoint_file']).with_suffix('.json'), old)
            with self.assertRaises(ValueError): api.read_checkpoint(path/side['checkpoint_file'], side['checkpoint_sha256'])
        with self.assertRaises(ValueError):
            train.validate_cursor({'visited_targets': 3, 'completed_optimizer_steps': 3, 'zero_valid_crops': 1, 'charged_wall_seconds': 1}, ['a']*4)

    def test_wall_budget_same_cursor_and_unsettled_rejection(self):
        recipe = {'arm': 'dense', 'example': 'synthetic budget only'}
        with tempfile.TemporaryDirectory(prefix='flow_recovery_budget_cpu_') as directory:
            root = Path(directory)
            budget = train.begin_budget(core, root, recipe, resume_sha256=None)
            core.atomic_json(root/'latest.json', {'checkpoint_sha256': 'fixed0'})
            with self.assertRaises(ValueError):
                train.begin_budget(core, root, recipe, resume_sha256='fixed0', checkpoint_charge=1)
            train.settle_budget(core, root, budget, 12., 'SEGMENT_COMPLETE')
            # Immutable step0 still stores charge1; ledger must preserve all12 consumed seconds.
            resumed = train.begin_budget(core, root, recipe, resume_sha256='fixed0', checkpoint_charge=1)
            self.assertEqual(resumed['starting_charged_wall_seconds'], 12.)
            train.settle_budget(core, root, resumed, 8., 'FAILED')
            resumed = train.begin_budget(core, root, recipe, resume_sha256='fixed0', checkpoint_charge=1)
            self.assertEqual(resumed['starting_charged_wall_seconds'], 20.)
            train.settle_budget(core, root, resumed, 2680., 'SEGMENT_COMPLETE')
            with self.assertRaises(ValueError):
                train.begin_budget(core, root, recipe, resume_sha256='fixed0', checkpoint_charge=1)
            with self.assertRaises(ValueError):
                train.begin_budget(core, root, recipe, resume_sha256=None)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests))
    (HERE/'recovery_cpu_checks.json').write_text(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun,
        'GPU_executed': False, 'measurements': MEASUREMENTS,
        'scope': 'Small12-block native-MLP CPU fixtures; structural/gradient/RNG/checkpoint checks, not full-model GPU recovery or AEE'}, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
