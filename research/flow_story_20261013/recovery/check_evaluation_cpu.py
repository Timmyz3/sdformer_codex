#!/usr/bin/env python3
"""Boundary tests for the actual recovery evaluator, with CPU fixtures only.

No native model or real event/GT arrays are loaded. The entrypoint keeps its
actual checkpoint reader and pinned member files; only dataset preflight is
stubbed. Core tests use real PyTorch BN/Dropout on tiny CPU tensors and replace
CUDA transport plus data/metric adapters, never BN math or evaluator guards.
"""
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BREADTH = HERE.parent/'breadth'
RESEARCH = HERE.parents[1]
DATA = RESEARCH/'flow_story_20261011/preflight/data_protocol/artifacts'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BREADTH))
import evaluate_breadth as e
import evaluation_core as core
import evaluate_recovery as wrapper
import model_api as api

MEASUREMENTS = []
ROLE_ROWS = list(csv.DictReader((DATA/'train_auxiliary_roles.csv').open()))
ROWS = [dict(target_id=r['target_id'], sequence=r['sequence'], protocol_split='train') for r in ROLE_ROWS]
ROLES = {r['target_id']: r['auxiliary_role'] for r in ROLE_ROWS}
DEV = json.loads((BREADTH/'gate_dev32.json').read_text())['target_ids']
FIT = json.loads((DATA/'bn_calibration_schedule.json').read_text())['stages']['initial']['target_ids']
ORDER = json.loads((HERE/'manifests/gatefit4573_recovery.json').read_text())
CHECK = json.loads((HERE/'manifests/check64.json').read_text())
ROW_MAP = {r['target_id']: r for r in ROWS}


class Metrics:
    def __init__(self): self.read_ids = []
    def canonical_rows(self, _): return ROWS
    def auxiliary_roles(self, *_): return ROLES
    def load_arrays(self, root, row, *, evidence, expected_hashes):
        self.read_ids.append(row['target_id'])
        evidence.update(target_id=row['target_id'], synthetic=True)
        return {'events': np.full((1,), 5, dtype=np.float32), 'gt': None, 'mask': None}, None
    def partitions(self, *_): return {'synthetic': True}
    class MetricAccumulator:
        def __init__(self): self.frames = []; self.values = []
        def update(self, target, seq, output, gt, masks):
            self.frames.append(target); self.values.append(float(output.mean()))
        def summary(self): return {'synthetic_mean_output': sum(self.values)/len(self.values)}


class Tiny(nn.Module):
    def __init__(self, mutate=False, scale=2.):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.bn = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(.7)
        self.mutate = mutate
        self.calls = 0
        self.observed = []
    def to(self, device):
        # Explicit CPU test transport; does not call any CUDA method.
        assert device.type == 'cuda'
        return self
    def forward(self, x):
        self.calls += 1
        self.observed.append((self.bn.training, self.dropout.training,
                              self.bn.track_running_stats, self.bn.running_mean is None))
        if self.mutate and self.calls == 1:
            with torch.no_grad(): self.scale.add_(1)
        value = self.bn(self.dropout(x*self.scale))[0]
        return {'flow': [value.view(1, 2, 1, 1).expand(1, 2, 480, 640)]}


class CPUTransport:
    """Only CUDA routing is synthetic; Tensor, BN, modes and math are real CPU."""
    cuda = types.SimpleNamespace(is_available=lambda: True, set_device=lambda _: None,
                                 empty_cache=lambda: None, get_device_name=lambda _: 'CPU TEST TRANSPORT')
    def device(self, _): return types.SimpleNamespace(type='cuda', index=0)
    def __getattr__(self, name): return getattr(torch, name)


def synthetic_payload(steps=4573):
    state = {'scale': torch.tensor(2.)}
    sources = {'synthetic_native.py': 'a'*64}
    architecture = {'schema': 'flow_recovery_architecture_v1', 'arm': 'dense',
                    'source_sha256': sources,
                    'state_layout': {'scale': {'shape': [], 'dtype': 'torch.float32'}}}
    contract = {'start_checkpoint_sha256': wrapper.START_SHA, 'start_step': 135930,
        'data_contract': {'synthetic': True}, 'source_sha256': sources,
        'source_manifest_sha256': None, 'recovery_manifest_sha256': wrapper.ORDER_SHA,
        'training_target_ids': ORDER['target_ids'], 'training_target_ids_sha256': e.digest(ORDER['target_ids']),
        'canonical_manifest_sha256': e.sha(DATA/'common_targets.json'),
        'roles_sha256': e.sha(DATA/'train_auxiliary_roles.csv'),
        'implementation_sha256': {'model_api.py': e.sha(api.__file__)},
        'architecture_sha256': api.digest(architecture), 'arm': 'dense',
        'effective_config': api.semantic_config({}),
        'one_pass': True, 'max_target_visits': 4573, 'no_validation_or_best_checkpoint_selection': True,
        'svs_binding': None}
    return {'schema_version': api.SCHEMA, 'contract': contract, 'architecture': architecture,
            'architecture_sha256': api.digest(architecture), 'model_state_dict': state,
            'training': {'visited_targets': steps, 'completed_optimizer_steps': steps, 'zero_valid_crops': 0},
            'effective_config': {}, 'provenance': {'validation_used': False}}


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix='flow_recovery_eval_cpu_')
        self.root = Path(self.temp.name)
        self.core_breadth = self.root/'core_breadth'; self.core_breadth.mkdir()
        for name, source in (('collect_bn_cma.py', BREADTH/'collect_bn_cma.py'),
                             ('bn_calibration_schedule.json', DATA/'bn_calibration_schedule.json')):
            (self.core_breadth/name).write_bytes(source.read_bytes())
    def tearDown(self): self.temp.cleanup()

    def entry(self, *, steps=4573, stage='endpoint', mutate=None, wrong_sha=False,
              wrong_manifest=False, wrong_side=False):
        source_manifest = self.root/'source.json'; source_manifest.write_text('{}\n')
        payload = synthetic_payload(steps)
        payload['contract']['source_manifest_sha256'] = e.sha(source_manifest)
        if mutate: mutate(payload)
        payload['contract_sha256'] = api.digest(payload['contract'])
        checkpoint = self.root/'candidate.pt'; torch.save(payload, checkpoint)
        sha = api.sha(checkpoint)
        side = {'schema_version': api.SCHEMA, 'checkpoint_sha256': sha, 'checkpoint_file': checkpoint.name,
                'contract_sha256': payload['contract_sha256'], 'visited_targets': steps,
                'completed_optimizer_steps': steps, 'arm': payload['contract']['arm'],
                'architecture_sha256': payload['architecture_sha256']}
        if wrong_side: side['checkpoint_file'] = 'another.pt'
        checkpoint.with_suffix('.json').write_text(json.dumps(side))
        metrics = Metrics()
        plan = {'data_contract': {'synthetic': True}, 'source_sha256': {'synthetic_native.py': 'a'*64}}
        context = (None, None, metrics, [ROW_MAP[t] for t in DEV], {}, {}, {},
                   {'completed_optimizer_steps': 135930}, {'fixed': True}, plan)
        a = types.SimpleNamespace(checkpoint_sha256=wrapper.START_SHA, geometry='full9',
            actual_pair_count=0, wall_minutes=15, source_manifest=source_manifest,
            canonical=DATA/'common_targets.json', roles=DATA/'train_auxiliary_roles.csv',
            run_inference=False, output=self.root/'entry_result.json')
        check_path = HERE/'manifests/check64.json'
        if wrong_manifest:
            check_path = self.root/'check64.json'
            altered = copy.deepcopy(CHECK); altered['target_ids'][0] = DEV[0]
            check_path.write_text(json.dumps(altered))
        args = ['--breadth-root', str(BREADTH), '--candidate-checkpoint', str(checkpoint),
                '--candidate-sha256', '0'*64 if wrong_sha else sha,
                '--model-api-sha256', e.sha(api.__file__), '--evaluation-core-sha256', e.sha(core.__file__),
                '--recovery-manifest', str(HERE/'manifests/gatefit4573_recovery.json'),
                '--check64-manifest', str(check_path), '--stage', stage]
        with patch.object(e, 'parse_args', return_value=a), patch.object(e, 'preflight', return_value=context), \
             patch.object(core, 'run_evaluation', side_effect=AssertionError('Default must never infer')):
            self.assertEqual(wrapper.main(args), 0)
        return json.loads(a.output.read_text()), metrics

    def test_entry_real_checkpoint_and_stage_boundaries(self):
        result, metrics = self.entry()
        self.assertEqual(result['status'], 'RECOVERY_EVAL_PREFLIGHT_PASS_NO_INFERENCE')
        self.assertEqual(result['populations'], {'gate_dev32': DEV, 'check64': CHECK['target_ids']})
        self.assertEqual(metrics.read_ids, [])
        result, _ = self.entry(steps=0, stage='initial')
        self.assertEqual(result['populations'], {'gate_dev32': DEV})
        for steps in (0, 4572, 4574):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, 'Only saved step0'):
                self.entry(steps=steps)
        with self.assertRaisesRegex(ValueError, 'Only saved step0'):
            self.entry(steps=4573, stage='initial')

    def test_entry_rejects_sha_sidecar_source_and_manifest(self):
        for kwargs, message in (({'wrong_sha': True}, 'SHA/schema'), ({'wrong_side': True}, 'SHA/schema'),
                ({'wrong_manifest': True}, 'SHA'),
                ({'mutate': lambda p: p['contract'].update(start_checkpoint_sha256='f'*64)}, 'native provenance'),
                ({'mutate': lambda p: p['contract'].update(training_target_ids=list(reversed(ORDER['target_ids'])))}, 'population/source')):
            with self.subTest(kwargs=str(kwargs)), self.assertRaisesRegex(ValueError, message): self.entry(**kwargs)

    def test_entry_rejects_nonfinite_state_and_inconsistent_updates(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.entry(mutate=lambda p: p['model_state_dict'].update(scale=torch.tensor(float('nan'))))
        with self.assertRaisesRegex(ValueError, 'accounting'):
            self.entry(mutate=lambda p: p['training'].update(zero_valid_crops=1))

    def core_fixture(self, *, mutate=False, shared=False, changed_fresh=False, bad_population=None, bad_input_hash=False):
        metrics = Metrics(); models = []; writes = []
        raw_path = self.root/'synthetic_input.bin'; raw_path.write_bytes(b'SYNTHETIC ONLY\n')
        raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        hashes = {t+'/input': {'bytes': raw_path.stat().st_size, 'sha256': '0'*64 if bad_input_hash else raw_hash}
                  for t in FIT+DEV+CHECK['target_ids']}
        native = types.SimpleNamespace(target_files=lambda row: {'input': row['target_id']+'/input'},
                                       checked_path=lambda *_: raw_path)
        a = types.SimpleNamespace(breadth_root=self.core_breadth, canonical=DATA/'common_targets.json',
            roles=DATA/'train_auxiliary_roles.csv', geometry='full9', stage='endpoint', device='cuda:0',
            cpu_threads=1, wall_minutes=1, data_root=self.root, output=self.root/'core.json')
        context = (native, None, metrics, [ROW_MAP[t] for t in DEV], hashes, {}, {}, {}, {'fixed': True}, {})
        def factory():
            if shared and models: return models[0]
            model = Tiny(mutate=mutate and not models, scale=3. if changed_fresh and models else 2.)
            models.append(model); return model
        def prepare(events, device, proxy):
            base = float(events.reshape(-1)[0])
            return torch.tensor([[1., 3.], [2., 7.], [4., 11.], [9., 15.]])+base
        fake_e = types.SimpleNamespace(**{k: getattr(e, k) for k in
            ('require', 'sha', 'import_file', 'read_pin', 'apply_bn', 'bn_signature')})
        fake_e.prepare_input = prepare
        fake_e.write_result = lambda path, result: writes.append(copy.deepcopy(result))
        functional = types.SimpleNamespace(set_step_mode=lambda *_: None, reset_net=lambda _: None)
        populations = {'gate_dev32': DEV, 'check64': CHECK['target_ids']}
        if bad_population is not None: populations = bad_population
        loaded = [0]
        def synthetic_load(*_, **__):
            loaded[0] += 1
            return np.broadcast_to(np.float32(loaded[0] % 7), (10, 480, 640))
        with patch.object(np, 'load', synthetic_load), contextlib.redirect_stdout(io.StringIO()):
            result = core.run_evaluation(a, context, factory, {'synthetic': True}, populations,
                                         fake_e, CPUTransport(), functional)
        return result, models, metrics, loaded[0]

    def test_core_candidate_own_CMA176_and_independent_BN_arms(self):
        result, models, metrics, reads = self.core_fixture()
        self.assertEqual(reads, 176)
        self.assertEqual(len(models), 3)
        self.assertEqual(len({id(m.bn) for m in models}), 3)
        self.assertEqual(result['status'], 'CANDIDATE_EVALUATION_COMPLETE')
        self.assertEqual(result['calibration']['count_per_BN'], 176)
        self.assertTrue(result['calibration']['non_BN_state_unchanged'])
        self.assertEqual(metrics.read_ids, (DEV+CHECK['target_ids'])*2)
        calibrated, paper, deployed = models
        self.assertEqual(int(calibrated.bn.num_batches_tracked), 176)
        # Independent arithmetic oracle: CMA is mean of per-batch unbiased stats.
        bases = torch.tensor([float(i % 7) for i in range(1, 177)])
        expected_mean = 2*(torch.tensor([4., 9.])+bases.mean())
        expected_var = 4*torch.tensor([[1.,3.],[2.,7.],[4.,11.],[9.,15.]]).var(0, unbiased=True)
        torch.testing.assert_close(calibrated.bn.running_mean, expected_mean)
        torch.testing.assert_close(calibrated.bn.running_var, expected_var)
        self.assertIsNone(paper.bn.running_mean)
        self.assertFalse(paper.bn.track_running_stats)
        self.assertTrue(deployed.bn.track_running_stats)
        torch.testing.assert_close(deployed.bn.running_mean, calibrated.bn.running_mean)
        torch.testing.assert_close(deployed.bn.running_var, calibrated.bn.running_var)
        self.assertTrue(all(not dropout for model in models for _, dropout, _, _ in model.observed))
        self.assertTrue(all(training for training, _, _, _ in calibrated.observed))
        self.assertTrue(all(not training for model in (paper, deployed) for training, _, _, _ in model.observed))
        for model in models: self.assertEqual(float(model.scale), 2.)
        p = result['evaluations']['P_paper__gate_dev32']['summary']['synthetic_mean_output']
        d = result['evaluations']['D_candidate_CMA176__gate_dev32']['summary']['synthetic_mean_output']
        self.assertNotAlmostEqual(p, d, places=4)
        MEASUREMENTS.append({'test': 'CPU_tiny_candidate_CMA_and_isolation', 'calibration_inputs': reads,
            'BN_model_objects': len(models), 'evaluation_frames_each_policy': 96,
            'synthetic_output_Ppaper': p, 'synthetic_output_D_candidate': d,
            'output_values_are_not_AEE': True, 'dropout_always_off': True})

    def test_core_refuses_nonBN_mutation_or_wrong_fresh_candidate(self):
        with self.assertRaisesRegex(ValueError, 'changed non-BN'): self.core_fixture(mutate=True)
        with self.assertRaisesRegex(ValueError, 'graph/weights'): self.core_fixture(changed_fresh=True)

    def test_core_shared_model_cannot_contaminate_Ppaper_into_D(self):
        with self.assertRaisesRegex(ValueError, 'buffer layout'): self.core_fixture(shared=True)

    def test_core_bad_input_or_evaluation_members_rejected(self):
        with self.assertRaisesRegex(ValueError, 'accepted native data'): self.core_fixture(bad_input_hash=True)
        with self.assertRaisesRegex(ValueError, 'BN fit overlap'):
            self.core_fixture(bad_population={'gate_dev32': DEV, 'check64': [FIT[0]]+CHECK['target_ids'][1:]})
        with self.assertRaisesRegex(ValueError, 'explicit dev32/check64'):
            self.core_fixture(bad_population={'gate_dev32': DEV, 'final_valid': CHECK['target_ids']})


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'schema': 'flow_recovery_evaluation_cpu_checks_v1', 'passed': result.wasSuccessful(),
        'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'measurements': MEASUREMENTS, 'torch': str(torch.__version__),
        'CUDA_initialized': torch.cuda.is_initialized(), 'real_event_GT_or_check_arrays_read': False,
        'real_native_model_forward_or_training': False,
        'limitations': 'CPU tiny models and synthetic input/metric adapters; CUDA transport mocked; no GPU throughput, AEE or native full-graph equivalence claim.',
        'source_sha256': {name: e.sha(HERE/name) for name in
            ('evaluation_core.py', 'evaluate_recovery.py', 'model_api.py', 'check_evaluation_cpu.py')}}
    (HERE/'evaluation_cpu_checks.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
