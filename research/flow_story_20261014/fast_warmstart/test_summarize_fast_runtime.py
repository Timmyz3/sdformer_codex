#!/usr/bin/env python3
"""Synthetic new-schema checks, borrowing only audited old metadata geometry.

No GT/NPY/NPZ arrays or network are opened. This is not a scientific result.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('summary', HERE/'summarize_fast_runtime.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
PRIOR = HERE.parents[1]/'flow_story_20261013/motion_groups/runtime_group_cpu_v1/results.json'
PRIOR_SHA = '7c6ffabf5def6d0340bf88baa06eb7f84fea84e47cac6ed502bfcc781c73970d'


def fake_hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixture():
    assert s.sha(PRIOR) == PRIOR_SHA
    prior = json.loads(PRIOR.read_text()); d = copy.deepcopy(prior)
    d.update(schema='flow_fast_runtime_results_v1', status='FAST_WARMSTART_RUNTIME_COMPLETE',
             backbone_or_gate_recovery_performed=False)
    d['contract'].update(schema='flow_fast_runtime_contract_v1', fast_weights_sha256=fake_hash('weights'),
                         fast_cache_sha256=fake_hash('cache'))
    frames = []
    for fi, old_frame in enumerate(prior['frames']):
        target = old_frame['target_id']
        frame = {'target_id': target, 'sequence': old_frame['sequence'], 'arms': {},
            'native_to_tiled_conversion': copy.deepcopy(old_frame['native_vs_all_tiled']),
            'predictions_artifact': {'file': 'predictions_'+target+'.npz', 'sha256': fake_hash(target)},
            'direction_error_strata': {'GT_used_by_inference': False,
                'tile_labels_0ineligible_1correct_2wrong_3abstain': [[0, 1, 2, 3, 0]]*4}}
        for arm in s.ARMS:
            old_arm = arm.split('_')[0] if arm not in ('native_dense', 'tiled_all_keep') else arm
            record = copy.deepcopy(old_frame['arms'][old_arm])
            record.pop('same_shape_reference', None)
            record['final_flow_sha256'] = fake_hash((target, 'dense' if arm in ('native_dense', 'tiled_all_keep') else arm))
            if arm != 'native_dense':
                item = record['execution'][0]; branch = arm.split('_', 1)[1] if arm != 'tiled_all_keep' else 'zero'
                points, calls = item['kept_real_pixels'], item['kept_tiles']
                item.update(mode='actual', branch=branch, slow_points=points, fast_region_points=1200-points,
                    slow_tiles=calls, observed={'slow_fc1_rows': 10*points, 'slow_fc2_rows': 10*points,
                    'fast_fc_rows': 10*(1200-points) if branch == 'trained' else 0,
                    'slow_calls': calls, 'fast_calls': 20-calls if branch == 'trained' else 0,
                    'mean_fill_elements': 10*(1200-points)*384 if branch == 'mean' else 0})
                item['fast_LN_rows'] = item['observed']['fast_fc_rows']
                item['fast_real_MACs_observed_rows_times_C2'] = item['fast_LN_rows']*384*384
                if fi == 0 and arm in s.REFERENCE_ARMS:
                    reference = {'execution': [copy.deepcopy(item)], 'wall_seconds_CPU': .1,
                                 'final_flow_sha256': record['final_flow_sha256']}
                    ref = reference['execution'][0]; ref['mode'] = 'reference'
                    ref['observed'] = {'slow_fc1_rows': 12000, 'slow_fc2_rows': 12000,
                        'fast_fc_rows': 12000 if branch == 'trained' else 0, 'slow_calls': 20,
                        'fast_calls': 20 if branch == 'trained' else 0,
                        'mean_fill_elements': 12000*384 if branch == 'mean' else 0}
                    ref['fast_LN_rows'] = ref['observed']['fast_fc_rows']
                    ref['fast_real_MACs_observed_rows_times_C2'] = ref['fast_LN_rows']*384*384
                    record['same_shape_reference'] = {'final_flow_exact': True, 'final_flow_max_abs': 0, 'reference': reference}
            # Known uniform per-pixel perturbation, allowing an independent contrast assertion.
            extra = .25 if arm.endswith('trained') else 0
            regions = record['metrics']['regions']
            for value in regions.values():
                if value['valid_pixels']:
                    value['error_sum'] += extra*value['valid_pixels']
                    value['AEE'] = value['error_sum']/value['valid_pixels']
            whole = regions['all_valid']; n = whole['valid_pixels']
            counts = [n//5, n//3, 0, n-n//5-n//3]
            for name, count in zip(s.DIRECTIONS, counts):
                total = whole['error_sum']*count/n
                regions[name] = {'valid_pixels': count, 'error_sum': total, 'AEE': total/count if count else None}
            frame['arms'][arm] = record
        frames.append(frame)
    d['frames'] = frames; d['arms'] = {}
    for arm in s.ARMS:
        rows = [f['arms'][arm]['metrics'] for f in frames]
        summary = s.aggregate(rows)
        summary['per_sequence'] = {seq: s.aggregate([r for r in rows if r['sequence'] == seq]) for seq in sorted({r['sequence'] for r in rows})}
        d['arms'][arm] = {'frames': rows, 'summary': summary}
    return d, prior


class SummaryChecks(unittest.TestCase):
    def test_complete_and_known_contrast_empty_region(self):
        d, _ = fixture(); value = s.summarize(d)
        self.assertEqual(value['validation']['arm_count'], 7)
        delta = value['mean_contrasts']['independent_trained_minus_independent_zero']['all_valid']
        self.assertAlmostEqual(delta['delta_AEE'], .25, places=12)
        self.assertEqual(delta['frames'], 4)
        self.assertIsNone(value['summaries']['motion_trained']['regions']['direction_abstain']['frame_mean_AEE'])
        self.assertEqual(len(value['first_frame_references']), 4)
        self.assertEqual(len(value['cost_rows']), 20)
        self.assertIn('总成本不同', s.markdown(value))

    def test_summary_region_and_sequence_tamper_rejected(self):
        for mutation in (
            lambda d: d['arms']['independent_mean']['summary']['regions']['B'].__setitem__('frame_mean_AEE', 99),
            lambda d: d['arms']['motion_trained']['summary']['per_sequence'].pop('zurich_city_01_a'),
            lambda d: d['frames'][0]['arms']['independent_trained']['metrics']['regions']['direction_wrong'].__setitem__('error_sum', 0),
            lambda d: d.__setitem__('status', 'FAST_WARMSTART_RUNTIME_INCOMPLETE'),
        ):
            d, _ = fixture(); mutation(d)
            with self.assertRaises(ValueError): s.summarize(d)

    def test_execution_and_mask_binding_rejected(self):
        for mutation in (
            lambda d: d['frames'][2]['arms']['motion_trained']['execution'][0]['observed'].__setitem__('fast_fc_rows', 0),
            lambda d: d['frames'][0]['arms']['independent_trained']['same_shape_reference'].__setitem__('final_flow_exact', False),
            lambda d: d['frames'][1]['arms']['independent_mean']['execution'][0].__setitem__('input_sha256', '0'*64),
            lambda d: d.__setitem__('BN_buffers_unchanged', False),
        ):
            d, _ = fixture(); mutation(d)
            with self.assertRaises(ValueError): s.summarize(d)

    def test_prior_differences_are_reported_not_rejected(self):
        d, prior = fixture(); value = s.summarize(d)
        comparison = s.prior_comparison(value, prior, str(PRIOR), PRIOR_SHA)
        self.assertTrue(comparison['all_recorded_common_region_metrics_exact'])
        value_region = prior['frames'][0]['arms']['independent']['metrics']['regions']['all_valid']
        value_region['AEE'] += .01
        value_region['error_sum'] += .01*value_region['valid_pixels']
        comparison = s.prior_comparison(value, prior, str(PRIOR), PRIOR_SHA)
        self.assertFalse(comparison['all_recorded_common_region_metrics_exact'])
        row = next(r for r in comparison['rows'] if r['current_arm'] == 'independent_zero')
        self.assertAlmostEqual(row['current_minus_prior_AEE']['all_valid'], -.01)

    def test_cli_and_existing_output_refused(self):
        d, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); input_file = root/'input.json'
            input_file.write_text(json.dumps(d, allow_nan=False))
            argv = ['--input', str(input_file), '--input-sha256', s.sha(input_file), '--output-dir', str(root/'out')]
            s.main(argv)
            self.assertTrue((root/'out/summary.json').is_file())
            self.assertTrue((root/'out/SUMMARY.md').is_file())
            with self.assertRaises(ValueError): s.main(argv)


if __name__ == '__main__':
    unittest.main(verbosity=2)
