#!/usr/bin/env python3
"""CPU-only tests of receipt guards, real dataset IO, and exact state recovery."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
import train_baseline as baseline

torch.set_num_threads(1)


class ToyDataset:
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return torch.tensor([index / 5, 1.0, -0.5, 0.25]), index


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def payload(self, contract=None):
        contract = contract or {'scope': 'engineering_subset', 'seed': 0}
        return {'schema_version': baseline.SCHEMA, 'contract': contract,
                'contract_sha256': baseline.digest(contract),
                'provenance': {'created_by': baseline.CREATOR, 'random_init': True,
                               'old_checkpoint_loaded': False, 'training_scope': contract['scope']},
                'training': {'completed_optimizer_steps': 3},
                'model_state_dict': {'weight': torch.tensor([1.0, 2.0])},
                'rng_state': baseline.capture_rng(torch, np)}

    def test_weights_only_rng_roundtrip(self):
        random.seed(6); np.random.seed(6); torch.manual_seed(6)
        data = self.payload()
        baseline.save_checkpoint(torch, self.root, data)
        expected = (random.random(), np.random.random(), torch.rand(3))
        loaded, _ = baseline.load_checkpoint(torch, self.root / 'latest.json', data['contract'])
        baseline.restore_rng(torch, np, loaded['rng_state'])
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        self.assertTrue(torch.equal(torch.rand(3), expected[2]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_checkpoint_byte_corruption_rejected(self):
        p = self.payload(); baseline.save_checkpoint(torch, self.root, p)
        path = next(self.root.glob('*.pt'))
        with path.open('ab') as f:
            f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            baseline.load_checkpoint(torch, path, p['contract'])

    def test_checkpoint_same_step_is_immutable(self):
        p = self.payload(); baseline.save_checkpoint(torch, self.root, p)
        checkpoint = next(self.root.glob('*.pt')); before = baseline.sha(checkpoint)
        p['stop_reason'] = 'different_reason'
        with self.assertRaisesRegex(ValueError, 'immutable'):
            baseline.save_checkpoint(torch, self.root, p)
        self.assertEqual(baseline.sha(checkpoint), before)

    def test_engineering_cannot_be_promoted_to_formal(self):
        p = self.payload(); baseline.save_checkpoint(torch, self.root, p)
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            baseline.load_checkpoint(torch, self.root / 'latest.json', {'scope': 'canonical6000', 'seed': 0})

    def test_legacy_checkpoint_rejected_before_deserialization(self):
        path = self.root / 'legacy.pt'; path.write_bytes(b'not a torch archive')
        baseline.atomic_json(path.with_suffix('.json'), {'schema_version': 'legacy'})
        with self.assertRaisesRegex(ValueError, 'legacy imports forbidden'):
            baseline.load_checkpoint(torch, path, {})

    def test_checkpoint_retention_only_recognized_current_run(self):
        foreign = self.root / 'checkpoint_step_000000000.pt'; foreign.write_bytes(b'foreign')
        for step in range(1, 5):
            p = self.payload(); p['training']['completed_optimizer_steps'] = step
            baseline.save_checkpoint(torch, self.root, p)
        self.assertTrue(foreign.exists())
        self.assertEqual(sorted(p.name for p in self.root.glob('*.pt')),
                         ['checkpoint_step_000000000.pt', 'checkpoint_step_000000003.pt', 'checkpoint_step_000000004.pt'])

    def test_full_acceptance_requires_real_hash_equivalence(self):
        report = {'schema_version': 1, 'mode': 'fullhash', 'passed_requested_checks': True,
                  'complete_source_destination_hash_equivalence_verified': False,
                  'all_file_hash_inventory_generated': True}
        with self.assertRaisesRegex(ValueError, 'source/destination hash acceptance'):
            baseline.validate_acceptance(report, self.root, {}, {})

    def test_acceptance_wrong_root_rejected(self):
        report = {'schema_version': 1, 'mode': 'fullhash', 'passed_requested_checks': True,
                  'complete_source_destination_hash_equivalence_verified': True,
                  'all_file_hash_inventory_generated': True, 'root': str(self.root / 'other')}
        with self.assertRaisesRegex(ValueError, 'root differs'):
            baseline.validate_acceptance(report, self.root, {}, {})

    def test_canonical6000_and_manifest_tamper(self):
        path = Path(__file__).resolve().parents[2] / 'flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json'
        self.assertEqual(len(baseline.canonical_rows(path)), 6000)
        tampered = self.root / 'common.json'; tampered.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'Pinned SHA mismatch'):
            baseline.canonical_rows(tampered)

    def test_hash_inventory_duplicate_and_traversal_rejected(self):
        csv = self.root / 'hashes.csv'
        row = 'saved_flow_data/test.npy,3,' + '1' * 64 + '\n'
        csv.write_text('relative_path,bytes,sha256\n' + row + row)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            baseline.read_hashes(csv)
        csv.write_text('relative_path,bytes,sha256\n../escape,3,' + '1' * 64 + '\n')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            baseline.read_hashes(csv)

    def test_dataset_actual_read_and_changed_content_rejected(self):
        row = {'target_id': 'train_0001', 'sequence': 'train',
               'v2_target_filename': 'train_0001.npy', 'gt_filename': 'train_0001.npy', 'mask_filename': 'train_0001.npy'}
        hashes = {}
        for category, relative in baseline.target_files(row).items():
            path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.zeros(baseline.SHAPES[category], dtype=baseline.DTYPES[category]))
            hashes[relative] = {'bytes': path.stat().st_size, 'sha256': baseline.sha(path)}
        dataset = baseline.VerifiedTrainDataset(self.root, [row], hashes)
        x, m, y, identifier = dataset[0]
        self.assertEqual(tuple(x.shape), (10, 480, 640)); self.assertEqual(identifier, 'train_0001')
        path = self.root / baseline.target_files(row)['gt']
        with path.open('r+b') as f:
            f.seek(-1, 2); f.write(b'1')
        with self.assertRaisesRegex(ValueError, 'changed after acceptance'):
            dataset[0]

    def test_dataloader_adamw_scheduler_and_rng_exact_resume(self):
        # Real multi-worker DataLoader; stochastic main-process augmentation and dropout.
        # Interrupted in the middle of an epoch; continue across two scheduler events.
        def build():
            random.seed(33); np.random.seed(33); torch.manual_seed(33)
            model = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Dropout(0.2), torch.nn.Linear(7, 1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            return model, optimizer, scheduler

        def advance(model, optimizer, scheduler, state, limit):
            seen = []
            while state['completed_optimizer_steps'] < limit:
                order = baseline.epoch_order(torch, 5, 33, state['epoch'])
                loader = torch.utils.data.DataLoader(ToyDataset(), batch_size=1,
                    sampler=order[state['next_index']:], num_workers=2,
                    generator=torch.Generator().manual_seed(2000003 + 33 + state['epoch']))
                for x, index in loader:
                    if state['completed_optimizer_steps'] >= limit:
                        break
                    random_shift = random.random() + float(np.random.random())
                    x = x + random_shift
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(x) - index.float().view(1, 1) / 5).square().mean()
                    loss.backward(); optimizer.step()
                    seen.append((int(index.item()), random_shift, float(loss.item()), optimizer.param_groups[0]['lr']))
                    state['completed_optimizer_steps'] += 1; state['next_index'] += 1
                    if state['next_index'] == 5:
                        scheduler.step(); state['epoch'] += 1; state['next_index'] = 0
            return seen

        state = {'completed_optimizer_steps': 0, 'epoch': 0, 'next_index': 0}
        full = build(); expected_order = advance(*full, copy.deepcopy(state), 13)
        partial = build(); partial_state = copy.deepcopy(state)
        first = advance(*partial, partial_state, 3)
        payload = self.payload()
        payload.update(model_state_dict=partial[0].state_dict(), optimizer_state_dict=partial[1].state_dict(),
                       scheduler_state_dict=partial[2].state_dict(), training=partial_state,
                       rng_state=baseline.capture_rng(torch, np))
        baseline.save_checkpoint(torch, self.root, payload)
        loaded, _ = baseline.load_checkpoint(torch, self.root / 'latest.json', payload['contract'])
        resumed = build()
        resumed[0].load_state_dict(loaded['model_state_dict'], strict=True)
        resumed[1].load_state_dict(loaded['optimizer_state_dict']); resumed[2].load_state_dict(loaded['scheduler_state_dict'])
        baseline.restore_rng(torch, np, loaded['rng_state'])
        rest = advance(*resumed, loaded['training'], 13)
        self.assertEqual(expected_order, first + rest)
        for name, value in full[0].state_dict().items():
            self.assertTrue(torch.equal(value, resumed[0].state_dict()[name]), name)
        self.assertEqual(full[2].state_dict(), resumed[2].state_dict())
        for key, state_value in full[1].state_dict()['state'].items():
            for name, tensor in state_value.items():
                self.assertTrue(torch.equal(tensor, resumed[1].state_dict()['state'][key][name]), name)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BaselineTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {'CPU_only': True, 'GPU_initialized': torch.cuda.is_initialized(),
                'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
                'passed': result.wasSuccessful(), 'torch': str(torch.__version__),
                'test_file_sha256': baseline.sha(__file__), 'trainer_sha256': baseline.sha(baseline.__file__),
                'scope': 'Toy CPU recovery and actual temporary NPY IO guards; no native-model/GPU/AEE training'}
    baseline.atomic_json(Path(__file__).with_name('baseline_selftest_results.json'), evidence)
    raise SystemExit(0 if result.wasSuccessful() else 1)
