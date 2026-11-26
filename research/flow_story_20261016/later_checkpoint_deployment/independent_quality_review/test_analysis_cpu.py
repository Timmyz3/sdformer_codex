import copy
import unittest

from analysis_core import REGIONS, REFERENCES, analyze


def fixture():
    members = [{'target_id': f'seq{q}_{i:04}', 'sequence': f'seq{q}'} for q in range(2) for i in range(4)]
    frames = [{**r, 'regions': {name: {'valid_pixels': 10, 'error_sum': 100., 'AEE': 10.} for name in REGIONS}}
              for r in members]
    return members, {key: copy.deepcopy(frames) for key in ('native', 'FF')}, {key: copy.deepcopy(frames) for key in REFERENCES}


def scale(frames, factor, region=None, sequence=None):
    for frame in frames:
        if sequence is not None and frame['sequence'] != sequence:
            continue
        for name, metric in frame['regions'].items():
            if region is None or name == region:
                metric['AEE'] *= factor
                metric['error_sum'] *= factor


class AnalysisCPU(unittest.TestCase):
    def test_equal_state_go_is_only_diagnostic(self):
        members, candidates, references = fixture()
        r = analyze(candidates, references, members, True)
        self.assertEqual(r['resource_decision']['recommendation'], 'GO_ONE_ORDINARY_RECOVERY_DIAGNOSTIC')
        for name in ('automatic_execution', 'H0_H1_admitted', 'recovery_stability_proven'):
            self.assertFalse(r['resource_decision'][name])

    def test_denominator_direction_is_native(self):
        members, candidates, references = fixture()
        scale(candidates['native'], .5)
        r = analyze(candidates, references, members, True)
        self.assertEqual(r['comparisons']['FF_vs_sameW_native']['scopes']['overall']['all_valid']['relative_change'], 1.)
        self.assertIn('NO_GO_FF_CONVERSION', r['resource_decision']['reasons'])

    def test_single_sequence_large_region_fails_anchor(self):
        members, candidates, references = fixture()
        for frames in candidates.values():
            scale(frames, 1.10, 'L', 'seq1')
        r = analyze(candidates, references, members, True)
        self.assertEqual(r['resource_decision']['reasons'], ['NO_GO_STATE_QUALITY_ANCHOR'])

    def test_smallLR_reference_never_rescues_fixed_anchor(self):
        members, candidates, references = fixture()
        for frames in candidates.values():
            scale(frames, 1.1)
        for key in ('smallLR/native', 'smallLR/nativeW_FF'):
            scale(references[key], 2.)
        r = analyze(candidates, references, members, True)
        self.assertTrue(r['comparisons']['native_vs_smallLR_native']['pass'])
        self.assertIn('NO_GO_STATE_QUALITY_ANCHOR', r['resource_decision']['reasons'])

    def test_engineering_failure_blocks_resource_rule(self):
        members, candidates, references = fixture()
        r = analyze(candidates, references, members, False)
        self.assertEqual(r['resource_decision']['reasons'], ['NO_GO_EXECUTION'])

    def test_empty_gate_is_not_observed_success(self):
        members, candidates, references = fixture()
        for frames in list(candidates.values()) + list(references.values()):
            for frame in frames:
                frame['regions']['L'] = {'valid_pixels': 0, 'error_sum': 0., 'AEE': None}
        r = analyze(candidates, references, members, True)
        self.assertIn('NO_GO_UNSUPPORTED_GATE', r['resource_decision']['reasons'])

    def test_all16_regions_and_eight_members_retained(self):
        members, candidates, references = fixture()
        r = analyze(candidates, references, members, True)
        self.assertEqual(len(r['all_endpoints']), 6)
        self.assertTrue(all(len(frames) == 8 for frames in r['all_endpoints'].values()))
        self.assertTrue(all(len(scopes) == 3 and all(len(regs) == 16 for regs in scopes.values()) for scopes in r['summaries'].values()))

    def test_bad_support_identity_or_region_rejected(self):
        for mutation in ('support', 'identity', 'region'):
            members, candidates, references = fixture()
            if mutation == 'support':
                candidates['FF'][0]['regions']['all_valid']['valid_pixels'] = 20
                candidates['FF'][0]['regions']['all_valid']['error_sum'] = 200.
            elif mutation == 'identity':
                candidates['FF'].reverse()
            else:
                candidates['FF'][0]['regions'].pop('voxel_zero')
            with self.assertRaises(ValueError):
                analyze(candidates, references, members, True)


if __name__ == '__main__':
    unittest.main()
