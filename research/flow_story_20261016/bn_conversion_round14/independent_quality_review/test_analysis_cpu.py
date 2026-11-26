import copy
import unittest

from analysis_core import REGIONS, STEPS, analyze


def fixture():
    members = [{'target_id': f'seq{q}_{i:04}', 'sequence': f'seq{q}'} for q in range(2) for i in range(4)]
    rows = [{**r, 'regions': {name: {'valid_pixels': 10, 'error_sum': 100., 'AEE': 10.} for name in REGIONS}}
            for r in members]
    return members, {s: copy.deepcopy(rows) for s in STEPS}, {s: copy.deepcopy(rows) for s in STEPS}


def scale(rows, factor, region=None, sequence=None):
    for row in rows:
        if sequence is not None and row['sequence'] != sequence:
            continue
        for name, metric in row['regions'].items():
            if region is None or name == region:
                metric['error_sum'] *= factor
                metric['AEE'] *= factor


class AnalysisCPU(unittest.TestCase):
    def test_equal_conversion_is_only_consideration(self):
        members, current, deployment = fixture()
        out = analyze(current, deployment, members, True)
        self.assertEqual(out['next_step_scope']['recommendation'], 'CONSIDER_ONE_FIXED_ORDINARY_RECOVERY_DIAGNOSTIC')
        for key in ('automatic_execution', 'H0_H1_admitted', 'recovery_stability_proven', 'K_prior_resource_gate_changed'):
            self.assertFalse(out['next_step_scope'][key])

    def test_current_is_the_denominator(self):
        members, current, deployment = fixture()
        scale(current['271482'], .5)
        out = analyze(current, deployment, members, True)
        cell = out['same_W_conversion_gates']['271482']['scopes']['overall']['all_valid']
        self.assertEqual(cell['relative_change'], 1.)
        self.assertFalse(cell['pass'])

    def test_old_checkpoint_cannot_rescue_later_failure(self):
        members, current, deployment = fixture()
        scale(deployment['271482'], 1.1)
        out = analyze(current, deployment, members, True)
        self.assertTrue(out['same_W_conversion_gates']['135930']['pass'])
        self.assertIn('LATER_BN_DEPLOYMENT_QUALITY_LOSS', out['next_step_scope']['reasons'])

    def test_older_failure_does_not_reselect_or_veto_later(self):
        members, current, deployment = fixture()
        scale(deployment['135930'], 1.1)
        out = analyze(current, deployment, members, True)
        self.assertFalse(out['same_W_conversion_gates']['135930']['pass'])
        self.assertTrue(out['same_W_conversion_gates']['271482']['pass'])
        self.assertEqual(out['next_step_scope']['recommendation'], 'CONSIDER_ONE_FIXED_ORDINARY_RECOVERY_DIAGNOSTIC')

    def test_single_sequence_large_region_loss_is_retained(self):
        members, current, deployment = fixture()
        scale(deployment['271482'], 1.1, region='L', sequence='seq1')
        out = analyze(current, deployment, members, True)
        self.assertFalse(out['same_W_conversion_gates']['271482']['pass'])

    def test_engineering_or_empty_support_cannot_be_success(self):
        members, current, deployment = fixture()
        out = analyze(current, deployment, members, False)
        self.assertIn('EXECUTION_NOT_VERIFIED', out['next_step_scope']['reasons'])
        for rows in list(current.values()) + list(deployment.values()):
            for row in rows:
                row['regions']['B'] = {'valid_pixels': 0, 'error_sum': 0., 'AEE': None}
        out = analyze(current, deployment, members, True)
        self.assertIn('LATER_CONVERSION_GATE_UNSUPPORTED', out['next_step_scope']['reasons'])

    def test_four_states_all8_all16_and_descriptive_only_between_W(self):
        members, current, deployment = fixture()
        out = analyze(current, deployment, members, True)
        self.assertEqual(len(out['all_endpoints']), 4)
        self.assertEqual(len(out['same_W_conversion_gates']), 2)
        self.assertTrue(all(len(rows) == 8 and all(len(r['regions']) == 16 for r in rows) for rows in out['all_endpoints'].values()))
        self.assertTrue(all('pass' not in regions for scopes in out['between_checkpoint_description'].values() for regions in scopes.values()))

    def test_wrong_state_or_support_is_rejected(self):
        members, current, deployment = fixture()
        current['smallLR'] = current.pop('135930')
        with self.assertRaises(ValueError):
            analyze(current, deployment, members, True)
        members, current, deployment = fixture()
        current['271482'][0]['regions']['all_valid'] = {'valid_pixels': 20, 'error_sum': 200., 'AEE': 10.}
        with self.assertRaises(ValueError):
            analyze(current, deployment, members, True)


if __name__ == '__main__':
    unittest.main()
