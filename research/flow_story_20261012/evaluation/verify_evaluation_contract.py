#!/usr/bin/env python3
"""CPU guards for scientific aggregation, membership, BN policy, and new weights."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
import torch
import yaml
from flow_eval_contract import *
from evaluate_native_v2 import validate_checkpoint_metadata, apply_bn_policy, SOURCE_MANIFEST_SHA, SCHEMA, CREATOR
from fit_region_thresholds import sample_values

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parents[1] / 'flow_story_20261011'
ARTIFACTS = PREVIOUS / 'preflight/data_protocol/artifacts'
SOURCE = HERE.parents[2] / 'SDformer/third_party/SDformerFlow'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Guards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.records = canonical_rows(ARTIFACTS / 'common_targets.json')
        cls.roles = auxiliary_roles(ARTIFACTS / 'train_auxiliary_roles.csv', cls.records)
        cls.fit = select_rows(cls.records, cls.roles, 'train_gate_fit')
        cls.sources = json.loads((PREVIOUS / 'server_setup_20261011/native_source_sha256.json').read_text())

    def region(self, all_fit=False):
        ids = [r['target_id'] for r in (self.fit if all_fit else self.fit[:2])]
        return {'schema': REGION_SCHEMA, 'definition': REGION_DEFINITION,
                'canonical_manifest_sha256': CANONICAL_SHA, 'auxiliary_roles_sha256': AUX_ROLES_SHA,
                'fit_target_ids': ids, 'fit_target_ids_sha256': ids_sha(ids),
                'boundary_threshold_pixels': 0.5, 'large_displacement_threshold_pixels': 2.5,
                'boundary_quantile': 0.95, 'large_quantile': 0.9}

    def metadata(self):
        cfg = yaml.safe_load((SOURCE / 'configs/train_DSEC_supervised_SDformerFlow_en4.yml').read_text())
        normalized = copy.deepcopy(cfg)
        normalized['data']['path'] = '<verified_DSEC>/saved_flow_data'
        normalized['loader']['gpu'] = '<logical-device>'
        tids = digest([r['target_id'] for r in self.records if r['protocol_split'] == 'train'])
        provenance = {'created_by': CREATOR, 'random_init': True, 'old_checkpoint_loaded': False,
                      'training_protocol_split': 'train', 'canonical_manifest_sha256': CANONICAL_SHA,
                      'BN_policy': 'native_train_tracked', 'training_scope': 'canonical6000',
                      'training_target_ids_sha256': tids, 'window_size': [2, 9, 9],
                      'source_sha256': self.sources}
        contract = {'source_sha256': self.sources, 'canonical_manifest_sha256': CANONICAL_SHA,
                    'training_target_ids_sha256': tids, 'effective_config': normalized,
                    'source_manifest_sha256': SOURCE_MANIFEST_SHA}
        ckpt = {'schema_version': SCHEMA, 'provenance': provenance, 'effective_config': cfg,
                'contract': contract, 'contract_sha256': digest(contract),
                'training': {'completed_optimizer_steps': 1}}
        side = {'contract_sha256': ckpt['contract_sha256'], 'training_scope': 'canonical6000', 'completed_optimizer_steps': 1}
        a = types.SimpleNamespace(scope='train_gate_dev', geometry='full9', source_root=SOURCE,
                                  canonical_training_ids_sha256=tids, pinned_source_sha256=self.sources)
        return ckpt, side, a

    def test_01_frame_vs_pixel_weighting(self):
        m = MetricAccumulator()
        gt = np.zeros((2, 2, 2)); mask1 = np.array([[1, 0], [0, 0]], dtype=bool)
        mask2 = np.array([[1, 1], [1, 0]], dtype=bool)
        p1, p2 = gt.copy(), gt.copy(); p1[0] = 1; p2[0] = 3
        m.update('a', 's1', p1, gt, {'all_valid': mask1})
        m.update('b', 's2', p2, gt, {'all_valid': mask2})
        r = m.summary()['regions']['all_valid']
        self.assertEqual(r['frame_mean_AEE'], 2.0); self.assertEqual(r['pixel_weighted_AEE'], 2.5)

    def test_02_empty_strata_null_no_zero_dilution(self):
        m = MetricAccumulator(); gt = np.zeros((2, 2, 2)); v = np.ones((2, 2), bool)
        m.update('a', 's', gt, gt, {'all_valid': v, 'B': ~v})
        r = m.summary()['regions']['B']
        self.assertIsNone(r['frame_mean_AEE']); self.assertEqual(r['empty_region_frames'], 1)

    def test_03_empty_all_valid_rejected(self):
        with self.assertRaises(ValueError):
            MetricAccumulator().update('x', 's', np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), {'all_valid': np.zeros((2, 2), bool)})

    def test_04_duplicate_target_rejected(self):
        m = MetricAccumulator(); x = np.zeros((2, 2, 2)); masks = {'all_valid': np.ones((2, 2), bool)}
        m.update('x', 's', x, x, masks)
        with self.assertRaises(ValueError): m.update('x', 's', x, x, masks)

    def test_05_invalid_pixels_not_evaluated(self):
        m = MetricAccumulator(); gt = np.zeros((2, 2, 2)); pred = gt.copy(); pred[:, 1, 1] = 1000
        mask = np.ones((2, 2), bool); mask[1, 1] = False
        m.update('x', 's', pred, gt, {'all_valid': mask})
        self.assertEqual(m.summary()['regions']['all_valid']['frame_mean_AEE'], 0)

    def test_06_holes_unknown_not_flat_or_boundary(self):
        gt = np.zeros((2, 7, 7)); valid = np.ones((7, 7), bool); valid[3, 3] = False
        gt[:, 3, 3] = 999
        _, score, eligible = region_features(gt, valid)
        self.assertFalse(eligible[3, 2]); self.assertFalse(eligible[0, 0])
        masks = partitions(gt, valid, np.zeros((10, 7, 7)), self.region())
        self.assertTrue(masks['B_unknown'][3, 2]); self.assertFalse(masks['B'][3, 2])

    def test_07_mutually_exclusive_regions_cover_every_GT(self):
        rng = np.random.default_rng(0); gt = rng.normal(size=(2, 7, 7)); v = np.ones((7, 7), bool); v[2, 2] = False
        p = partitions(gt, v, np.zeros((10, 7, 7)), self.region())
        total = sum(p[k].astype(int) for k in ('B_and_L', 'B_only', 'L_only_B_known', 'neither_B_known', 'B_unknown'))
        np.testing.assert_array_equal(total, v.astype(int))

    def test_08_no_fake_boundaries_on_constant_zero(self):
        r = self.region(); r['boundary_threshold_pixels'] = r['large_displacement_threshold_pixels'] = 0
        p = partitions(np.zeros((2, 5, 5)), np.ones((5, 5), bool), np.zeros((10, 5, 5)), r)
        self.assertFalse(p['B'].any()); self.assertFalse(p['L'].any())

    def test_09_valid_has_exact_members_no_subsets(self):
        self.assertEqual(len(select_rows(self.records, self.roles, 'valid_all')), 2152)
        with self.assertRaises(ValueError): select_rows(self.records, self.roles, 'valid_all', [self.fit[0]['target_id']])

    def test_10_train_dev_rejects_fit_or_valid(self):
        with self.assertRaises(ValueError): select_rows(self.records, self.roles, 'train_gate_dev', [self.fit[0]['target_id']])
        valid = next(r['target_id'] for r in self.records if r['protocol_split'] == 'valid')
        with self.assertRaises(ValueError): select_rows(self.records, self.roles, 'train_gate_dev', [valid])

    def test_11_region_fit_no_valid_leak(self):
        r = self.region(); r['fit_target_ids'] = [next(x['target_id'] for x in self.records if x['protocol_split'] == 'valid')]
        r['fit_target_ids_sha256'] = ids_sha(r['fit_target_ids'])
        with self.assertRaises(ValueError): validate_region_contract(r, self.records, self.roles)

    def test_12_subset_regions_not_formal_validation(self):
        validate_region_contract(self.region(), self.records, self.roles)
        with self.assertRaises(ValueError): validate_region_contract(self.region(), self.records, self.roles, True)
        validate_region_contract(self.region(True), self.records, self.roles, True)

    def test_13_fixed_sampling_deterministic_bounded(self):
        x = np.arange(5000, dtype=np.float64)
        a = sample_values(x, 'a', 'boundary'); b = sample_values(x, 'a', 'boundary')
        np.testing.assert_array_equal(a, b); self.assertEqual(len(a), 4096); self.assertEqual(len(set(a)), 4096)

    def test_14_center_crop_fixed_no_rescale(self):
        ys, xs = crop_slices('crop_center9')
        self.assertEqual(np.zeros((480, 640))[ys, xs].shape, (288, 384))
        self.assertEqual((ys.start, xs.start), (96, 128))

    def test_15_paper_BN_current_stats_eval_dropout_off(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(1), torch.nn.Dropout(0.9))
        model[0].num_batches_tracked.fill_(3)
        apply_bn_policy(model, 'P_paper')
        x = torch.arange(8.).reshape(1, 1, 2, 4)
        y = model(x)
        self.assertAlmostEqual(y.mean().item(), 0, places=6)
        self.assertIsNone(model[0].running_mean); self.assertFalse(any(m.training for m in model.modules()))

    def test_16_code_BN_running_buffers_unchanged_distinct_output(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(1), torch.nn.Dropout(0.9))
        model[0].num_batches_tracked.fill_(3); model[0].running_mean.fill_(20)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        paper = copy.deepcopy(model); apply_bn_policy(paper, 'P_paper'); apply_bn_policy(model, 'P_code')
        x = torch.arange(8.).reshape(1, 1, 2, 4)
        self.assertFalse(torch.equal(model(x), paper(x)))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))

    def test_17_code_BN_untrained_or_bad_variance_rejected(self):
        model = torch.nn.BatchNorm2d(1)
        with self.assertRaises(ValueError): apply_bn_policy(model, 'P_code')
        model.num_batches_tracked.fill_(1); model.running_var.fill_(-1)
        with self.assertRaises(ValueError): apply_bn_policy(model, 'P_code')

    def test_18_new_checkpoint_metadata_compatible(self):
        ckpt, side, a = self.metadata(); validate_checkpoint_metadata(ckpt, side, a)

    def test_19_old_or_zero_update_checkpoint_rejected(self):
        for field, value in [('old_checkpoint_loaded', True), ('random_init', False)]:
            ckpt, side, a = self.metadata(); ckpt['provenance'][field] = value
            with self.assertRaises(ValueError): validate_checkpoint_metadata(ckpt, side, a)
        ckpt, side, a = self.metadata(); ckpt['training']['completed_optimizer_steps'] = 0
        with self.assertRaises(ValueError): validate_checkpoint_metadata(ckpt, side, a)

    def test_20_engineering_not_formal_validation(self):
        ckpt, side, a = self.metadata(); ckpt['provenance']['training_scope'] = side['training_scope'] = 'engineering_subset'; a.scope = 'valid_all'
        with self.assertRaises(ValueError): validate_checkpoint_metadata(ckpt, side, a)

    def test_21_no_implicit_window_transition(self):
        ckpt, side, a = self.metadata(); a.geometry = 'full15'
        with self.assertRaises(ValueError): validate_checkpoint_metadata(ckpt, side, a)

    def test_22_contract_or_sidecar_drift_rejected(self):
        ckpt, side, a = self.metadata(); side['completed_optimizer_steps'] = 2
        with self.assertRaises(ValueError): validate_checkpoint_metadata(ckpt, side, a)
        ckpt, side, a = self.metadata(); ckpt['effective_config']['model']['norm_input'] = 'std'
        with self.assertRaises(ValueError): validate_checkpoint_metadata(ckpt, side, a)

    def test_23_source_drift_rejected(self):
        ckpt, side, a = self.metadata(); ckpt['provenance']['source_sha256'] = {}
        with self.assertRaises(ValueError): validate_checkpoint_metadata(ckpt, side, a)

    def test_24_loaded_bytes_hash_binding_and_corruption(self):
        row = self.fit[0]
        with tempfile.TemporaryDirectory() as directory:
            ps = paths_for(directory, row)
            arrays = {'events': np.ones((10, 480, 640), np.float32), 'gt': np.zeros((2, 480, 640), np.float32), 'mask': np.ones((480, 640), bool)}
            for key, path in ps.items():
                path.parent.mkdir(parents=True, exist_ok=True); np.save(path, arrays[key])
            expected = {k: sha(p) for k, p in ps.items()}; evidence = {}
            load_arrays(directory, row, evidence=evidence, expected_hashes=expected)
            self.assertEqual(evidence, expected)
            arrays['gt'][0, 0, 0] = 99; np.save(ps['gt'], arrays['gt'])
            with self.assertRaises(ValueError): load_arrays(directory, row, expected_hashes=expected)

    def test_25_safe_torch_load_of_producer_style_payload(self):
        ckpt, side, a = self.metadata(); ckpt['model_state_dict'] = {'toy': torch.ones(1)}
        ckpt['rng_state'] = {'numpy': {'keys': [1, 2], 'cached_gaussian': 0.0}, 'torch_cpu': torch.get_rng_state()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.pt'; torch.save(ckpt, path)
            value = torch.load(path, map_location='cpu', weights_only=True)
            validate_checkpoint_metadata(value, side, a)


if __name__ == '__main__':
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text())
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Guards)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests_run': result.testsRun,
              'failures': [name.id() for name, _ in result.failures], 'errors': [name.id() for name, _ in result.errors],
              'cuda_initialized': torch.cuda.is_initialized(), 'GPU_executed': False,
              'real_checkpoint_accuracy_evaluated': False, 'scope': 'Synthetic CPU metrics/BN/schema and pinned real membership/source guards; no trained baseline or AEE evidence',
              'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
              'source_hashes': {path.name: sha(path) for path in HERE.glob('*.py')}}
    (HERE / 'evaluation_cpu_tests.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if torch.cuda.is_initialized(): raise RuntimeError('Unexpected CUDA initialization')
    raise SystemExit(0 if result.wasSuccessful() else 1)
