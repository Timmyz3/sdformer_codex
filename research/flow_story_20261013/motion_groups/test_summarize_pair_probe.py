"""CPU-only summary checks using the real cpu_smoke.json schema as a fixture."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import summarize_pair_probe as summary

HERE = Path(__file__).resolve().parent


def fixture():
    return json.loads((HERE/'cpu_smoke.json').read_text())


def region_numbers(base, delta, residual):
    ae = {'base': base, 'a': base+2, 'b': base+3, 'ab': base+delta, 'additive': base+4}
    return {'valid_pixels': 1, 'AEE': ae, 'delta_a': 2., 'delta_b': 3., 'delta_ab': delta,
            'interaction': delta-5, 'metric_only_interaction': -1., 'nonlinear_network_AEE_excess': delta-4,
            'flow_superposition_residual_mean': residual, 'relative_superposition_residual': .5}


def weighted_fixture():
    """Synthetic numbers only: frame0 has2 anchors and frame1 has1."""
    d = fixture(); prototype = d['frames'][0]; frames = []
    for index, deltas in enumerate(((2., 6.), (10.,))):
        f = copy.deepcopy(prototype); f['target_id'] = 'synthetic_frame'+str(index)
        f['base']['target_id'] = f['target_id']
        layer = f['layers'][0]; template = layer['pairs'][0]; pairs = []
        for k, value in enumerate(deltas):
            p = copy.deepcopy(template)
            p['anchor_yx'][0] -= k
            for end in p['endpoints_yx'].values():
                end[0] -= k
            for condition, body in p['conditions'].items():
                delta = value if condition != 'rotated90' else ((1., 3.)[k] if index == 0 else 4.)
                for r in summary.REGIONS:
                    base = f['base']['regions'][r]
                    record = region_numbers(base['AEE'], delta, value/10)
                    record['valid_pixels'] = base['valid_pixels']
                    body['regions'][r] = record
            pairs.append(p)
        layer['pairs'] = pairs
        layer['selection'].update(requested_anchors=2, selected_anchors=len(pairs), eligible_common_anchors=len(pairs))
        frames.append(f)
    d['frames'] = frames; d['targets'] = [f['target_id'] for f in frames]
    d['contract'].update(target_count=2, pairs_per_layer=2)
    d.update(complete_frames=2, pair_anchors=3)
    return d


class Checks(unittest.TestCase):
    def test_real_cpu_schema_and_duplicate_horizontal(self):
        data = fixture(); result = summary.summarize(data)
        self.assertEqual(result['evidence_label'], 'CPU_SMOKE_ONLY')
        self.assertEqual(result['status'], 'SUMMARY_COMPLETE')
        self.assertEqual(result['included_frame_count'], 1)
        raw = data['frames'][0]['layers'][0]['pairs'][0]['conditions']['motion']['regions']['all_valid']
        got = result['stages']['0']['conditions']['motion']['all_valid']['means']
        self.assertEqual(got['delta_ab'], raw['delta_ab'])
        self.assertEqual(result['stages']['0']['duplicate_horizontal_anchor_counts'], {'motion': 1, 'rotated90': 0, 'neither': 0})
        self.assertIn('current-stat', summary.markdown(result))
        self.assertFalse(result['runtime_or_hardware_benefit_claim'])

    def test_anchor_first_then_equal_frames_and_matched_pair_differences(self):
        result = summary.summarize(weighted_fixture()); stage = result['stages']['0']
        motion = stage['conditions']['motion']['all_valid']
        self.assertEqual(motion['means']['delta_ab'], 7.)  # (mean(2,6)+10)/2; pooled anchor mean would6.
        self.assertEqual(motion['frames_with_values'], 2)
        matched = stage['motion_minus_rotated90']['all_valid']
        self.assertEqual(matched['means']['delta_ab'], 4.)  # (mean(1,3)+6)/2.
        self.assertEqual(matched['matched_anchor_count'], 3)
        self.assertEqual([v['matched_anchor_count'] for v in matched['per_frame']], [2, 1])
        self.assertEqual([v['means']['delta_ab'] for v in matched['per_frame']], [2., 6.])

    def test_empty_and_missing_regions_null_without_changing_denominator(self):
        d = weighted_fixture(); f = d['frames'][0]
        f['base']['regions']['B'] = {'valid_pixels': 0, 'AEE': None, 'error_sum': 0.}
        for pair in f['layers'][0]['pairs']:
            for body in pair['conditions'].values():
                body['regions']['B'] = {'valid_pixels': 0, 'AEE': None, 'interaction': None}
                body['regions'].pop('L')
        result = summary.summarize(d); stage = result['stages']['0']
        self.assertEqual(stage['conditions']['motion']['B']['means']['delta_ab'], 10.)
        self.assertEqual(stage['conditions']['motion']['B']['frames_with_values'], 1)
        self.assertEqual(stage['conditions']['motion']['B']['region_anchor_counts']['empty'], 2)
        self.assertEqual(stage['conditions']['motion']['L']['region_anchor_counts']['missing'], 2)
        self.assertEqual(stage['motion_minus_rotated90']['B']['frames_with_values'], 1)
        for row in result['frame_statistics']:
            if row['target_id'] == f['target_id'] and row['region'] in ('B', 'L'):
                self.assertTrue(all(v is None for v in row['means'].values()))

    def test_incomplete_run_excluded_and_partial_frame_not_salvaged(self):
        d = weighted_fixture(); d['status'] = 'PAIR_DIAGNOSTIC_INCOMPLETE'
        result = summary.summarize(d)
        self.assertEqual(result['status'], 'SOURCE_INCOMPLETE_EXCLUDED')
        self.assertEqual(result['stages'], {})
        self.assertEqual(len(result['excluded_frames']), 2)
        d = weighted_fixture(); del d['frames'][0]['layers'][0]['pairs'][0]['conditions']['rotated90']
        result = summary.summarize(d)
        self.assertEqual(result['status'], 'SUMMARY_PARTIAL_FRAME_EXCLUSIONS')
        self.assertEqual(result['included_frame_count'], 1)
        self.assertEqual(result['stages']['0']['conditions']['motion']['all_valid']['means']['delta_ab'], 10.)

    def test_no_eligible_anchors_remain_null(self):
        d = fixture(); item = d['frames'][0]['layers'][0]
        item['pairs'] = []; item['selection'].update(selected_anchors=0, eligible_common_anchors=0)
        d['pair_anchors'] = 0
        result = summary.summarize(d); stage = result['stages']['0']
        self.assertEqual(stage['conditions']['motion']['all_valid']['frames_without_selected_anchors'], 1)
        self.assertIsNone(stage['conditions']['motion']['all_valid']['means']['delta_ab'])
        self.assertEqual(stage['motion_minus_rotated90']['all_valid']['matched_anchor_count'], 0)

    def test_bad_decomposition_cost_and_population_are_rejected(self):
        for mutation in (
            lambda d: d['frames'][0]['layers'][0]['pairs'][0]['conditions']['motion']['regions']['all_valid'].update(interaction=999.),
            lambda d: d['frames'][0]['layers'][0]['pairs'][0]['conditions']['motion']['ab_execution'][0].update(producer_calls_cancelled=1),
            lambda d: d['frames'][0]['layers'][0]['pairs'][0]['conditions']['motion']['regions']['all_valid'].update(valid_pixels=1),
            lambda d: d['frames'][0]['layers'][0]['pairs'][0]['conditions']['motion']['regions']['all_valid'].update(delta_ab=float('nan')),
            lambda d: d['frames'][0]['layers'][0]['pairs'][0]['conditions']['horizontal_local'].update(same_pair_as_other_control=[]),
        ):
            d = fixture(); mutation(d)
            with self.assertRaises(ValueError): summary.summarize(d)

    def test_device_and_single_source_contract(self):
        d = fixture(); d['runtime'].update(device='cuda:0', GPU_inference_executed=True)
        self.assertEqual(summary.summarize(d)['evidence_label'], 'GPU_DIAGNOSTIC')
        d['runtime']['GPU_inference_executed'] = False
        with self.assertRaises(ValueError): summary.summarize(d)
        d = fixture(); d['frames'].append(copy.deepcopy(d['frames'][0]))
        with self.assertRaises(ValueError): summary.summarize(d)

    def test_cli_preserves_input_and_emits_json_markdown(self):
        before = summary.file_sha(HERE/'cpu_smoke.json')
        with tempfile.TemporaryDirectory(prefix='flow_pair_summary_cpu_') as root:
            args = ['--input', str(HERE/'cpu_smoke.json'), '--input-sha256', before, '--output-dir', root]
            with contextlib.redirect_stdout(io.StringIO()): summary.main(args)
            record = json.loads((Path(root)/'pair_summary.json').read_text())
            self.assertEqual(record['source_sha256'], before)
            self.assertIn('CPU_SMOKE_ONLY', (Path(root)/'PAIR_SUMMARY.md').read_text())
            with self.assertRaises(ValueError): summary.main(args)
        self.assertEqual(summary.file_sha(HERE/'cpu_smoke.json'), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
