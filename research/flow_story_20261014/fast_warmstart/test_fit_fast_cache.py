#!/usr/bin/env python3
"""CPU synthetic tests: borrowed topology, learning, immutable data and cache guards."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import fit_fast_cache as fit
import torch


class FastFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.fast = fit.load_fast_class()
        manifest_path = Path(__file__).resolve().parents[2] / 'flow_story_20261013/recovery/manifests/svs_fit20.json'
        manifest = json.loads(manifest_path.read_text())
        cls.contract = {'fit_manifest': str(manifest_path), 'fit_manifest_sha256': fit.FIT_MANIFEST_SHA}
        cls.cache = {'schema': 'flow_fast_branch_cache_v1', 'status': 'FAST_CACHE_COMPLETE',
                     'contract': cls.contract, 'contract_sha256': 'c' * 64,
                     'contract_canonical_sha256': fit.digest(cls.contract),
                     'checkpoint_sha256': fit.CHECKPOINT_SHA, 'BN_scope': 'MLP_CMA176', 'backbone_seen': True,
                     'fit_manifest_sha256': fit.FIT_MANIFEST_SHA, 'fit_target_ids': manifest['target_ids'],
                     'fit_target_ids_sha256': fit.FIT_IDS_SHA, 'source_pins': {'synthetic_test': 'a' * 64}, 'frames': []}
        tensors = {s: torch.zeros(10, 128, c) for s, c in fit.CHANNELS.items()}
        for target in manifest['target_ids']:
            stages = {}
            for stage, channels in fit.CHANNELS.items():
                indices = fit.expected_indices(target, stage)
                stages[stage] = {'u': tensors[stage], 'target': tensors[stage],
                                  'u_sha256': fit.tensor_sha(tensors[stage]), 'target_sha256': fit.tensor_sha(tensors[stage]),
                                  'spatial_indices': indices, 'indices_sha256': fit.digest(indices),
                                  'source_input_shape_TBHWC': fit.SHAPES[stage],
                                  'source_branch_shape_TBHWC': fit.SHAPES[stage], 'sampled_shape_TNC': [10, 128, channels]}
            cls.cache['frames'].append({'target_id': target, 'sequence': target.rsplit('_', 1)[0],
                                        'role': 'fit', 'input_evidence': {'events': 'b' * 64}, 'stages': stages})

    def test_official_topology_and_raw_rank3(self):
        model = self.fast(96)
        self.assertIsInstance(model.fc1[0], torch.nn.LayerNorm)
        self.assertIsInstance(model.fc1[1], torch.nn.Linear)
        self.assertEqual(model.drop.p, 0.)
        x = torch.randn(10, 128, 96)
        torch.testing.assert_close(model(x), model(x.reshape(-1, 96)).reshape_as(x), rtol=0, atol=0)

    def test_learning_and_immutable_input_teacher(self):
        torch.manual_seed(7)
        x = torch.randn(256, 8)
        # Independent, exactly representable synthetic LN-linear teacher.
        normalized = torch.nn.functional.layer_norm(x, (8,), eps=1e-5)
        matrix = .7 * torch.eye(8) + .2 * torch.roll(torch.eye(8), shifts=1, dims=1)
        y = normalized @ matrix.T + torch.linspace(-.2, .2, 8)
        before = (x.clone(), y.clone())
        weights, report = fit.fit_stage(x, y, ['synthetic_a', 'synthetic_b'], 128, self.fast,
                                        steps=500, batch_rows=64)
        metrics = report['metrics']
        self.assertLess(metrics['trained_fast']['pooled_MSE'], .1 * metrics['initialized_fast']['pooled_MSE'])
        self.assertLess(metrics['trained_fast']['pooled_MSE'], metrics['fit_mean_constant_branch']['pooled_MSE'])
        self.assertEqual(report['minibatch_row_draws'], 500 * 64)
        self.assertEqual([x['optimizer_steps'] for x in report['training_loss_every_100_steps']], [100, 200, 300, 400, 500])
        torch.testing.assert_close(x, before[0], rtol=0, atol=0)
        torch.testing.assert_close(y, before[1], rtol=0, atol=0)
        model = self.fast(8).eval()
        model.load_state_dict(weights['state_dict'], strict=True)
        self.assertTrue(all(torch.isfinite(v).all() for v in model.state_dict().values()))
        direct = (model(x).double() - y.double()).square().mean().item()
        self.assertAlmostEqual(direct, metrics['trained_fast']['pooled_MSE'], places=12)
        torch.testing.assert_close(weights['mean_branch'], y.double().mean(0).float(), rtol=0, atol=0)
        self.assertIsNone(x.grad)
        self.assertIsNone(y.grad)

    def test_fixed_seed_determinism_and_state_reload(self):
        generator = torch.Generator().manual_seed(22)
        x, y = (torch.randn(48, 8, generator=generator) for _ in range(2))
        first, _ = fit.fit_stage(x, y, ['a'], 48, self.fast, steps=5, batch_rows=16)
        torch.randn(100)  # Global RNG noise must not change the recipe.
        second, _ = fit.fit_stage(x, y, ['a'], 48, self.fast, steps=5, batch_rows=16)
        for key in first['state_dict']:
            torch.testing.assert_close(first['state_dict'][key], second['state_dict'][key], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            torch.save({'schema': 'flow_fast_warmstart_weights_v1', 'stages': {'0': first}}, path)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
            self.fast(8).load_state_dict(loaded['stages']['0']['state_dict'], strict=True)

    def test_full_members_and_shape_hash_sampling_guards(self):
        audit = fit.validate_cache(self.cache)
        self.assertEqual(audit['rows_per_stage'], 25600)
        self.assertEqual(len(audit['frames']), 20)
        for field, replacement in [('role', 'dev'), ('target_id', 'wrong_target')]:
            altered = copy.deepcopy(self.cache)
            altered['frames'][0][field] = replacement
            with self.assertRaises(ValueError):
                fit.validate_cache(altered)
        for field, replacement in [('u', torch.zeros(9, 128, 96)),
                                   ('u_sha256', '0' * 64),
                                   ('spatial_indices', [0] * 128),
                                   ('source_branch_shape_TBHWC', [10, 1, 1, 1, 96])]:
            altered = copy.deepcopy(self.cache)
            altered['frames'][0]['stages']['0'][field] = replacement
            with self.assertRaises(ValueError):
                fit.validate_cache(altered)

    def test_foreign_checkpoint_contract_and_nonfinite_rejected(self):
        for field, replacement in [('status', 'PARTIAL'), ('checkpoint_sha256', '0' * 64),
                                   ('contract_sha256', 'not_a_sha'), ('contract_canonical_sha256', '1' * 64),
                                   ('BN_scope', 'P_paper')]:
            altered = dict(self.cache)
            altered[field] = replacement
            with self.assertRaises(ValueError):
                fit.validate_cache(altered)
        altered = copy.deepcopy(self.cache)
        item = altered['frames'][0]['stages']['0']
        item['target'] = item['target'].clone()
        item['target'][0, 0, 0] = float('nan')
        item['target_sha256'] = fit.tensor_sha(item['target'])
        with self.assertRaises(ValueError):
            fit.validate_cache(altered)
        with self.assertRaises(ValueError):
            fit.fit_stage(torch.zeros(2, 8), torch.zeros(2, 7), ['a'], 2, self.fast, steps=1)
        with self.assertRaises(ValueError):
            fit.fit_stage(torch.full((2, 8), float('inf')), torch.zeros(2, 8), ['a'], 2, self.fast, steps=1)

    def test_cache_sha_guard_and_output_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.pt'
            torch.save({'schema': 'not_a_cache'}, path)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                fit.load_verified_cache(path, '0' * 64)
            with self.assertRaisesRegex(ValueError, 'Foreign cache'):
                fit.load_verified_cache(path, fit.sha_file(path))
            with self.assertRaisesRegex(ValueError, 'refuse overwrite'):
                fit.run(path, fit.sha_file(path), directory)


if __name__ == '__main__':
    unittest.main(verbosity=2)
