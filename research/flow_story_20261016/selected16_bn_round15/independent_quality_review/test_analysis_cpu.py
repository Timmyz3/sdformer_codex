"""Synthetic science-rule checks; no model, array inputs, or real quality."""
import copy
import unittest
from analysis_core import analyze, REGIONS


def fixture():
    members = [{'target_id': seq + '_' + str(i), 'sequence': seq} for seq in ('a', 'b') for i in range(4)]
    rows = [{'target_id': r['target_id'], 'sequence': r['sequence'], 'regions': {k: {'valid_pixels': 10, 'error_sum': 10., 'AEE': 1.} for k in REGIONS}} for r in members]
    return members, copy.deepcopy(rows), copy.deepcopy(rows), copy.deepcopy(rows)


def scale(rows, factor, seq=None, region=None):
    for row in rows:
        if seq is None or row['sequence'] == seq:
            for key, value in row['regions'].items():
                if region is None or key == region:
                    value['AEE'] *= factor
                    value['error_sum'] *= factor


class ScientificRules(unittest.TestCase):
    def test_exact_conversion_pass(self):
        m,s,c,p = fixture()
        x = analyze(s,c,p,m,True)
        self.assertTrue(x['selected16_vs_current78_gate']['pass'])
        self.assertEqual(x['next_step_scope']['recommendation'], 'SELECTED16_CONVERSION_SCOPE_CHECK_PASSED')
        self.assertFalse(x['next_step_scope']['automatic_execution'])

    def test_private_improvement_cannot_rescue_current_failure(self):
        m,s,c,p = fixture();scale(s,1.1);scale(p,2.)
        x = analyze(s,c,p,m,True)
        self.assertFalse(x['selected16_vs_current78_gate']['pass'])
        self.assertLess(x['all_region_deltas']['private24/native']['overall']['all_valid']['delta'],0)
        self.assertEqual(x['next_step_scope']['recommendation'],'STOP_BN_SUBSET_SWEEP_REVIEW_DEPLOYMENT_CONTRACT')

    def test_private_quality_does_not_gate_current(self):
        m,s,c,p = fixture();scale(p,.5)
        x = analyze(s,c,p,m,True)
        self.assertTrue(x['selected16_vs_current78_gate']['pass'])
        self.assertGreater(x['all_region_deltas']['private24/native']['overall']['all_valid']['relative_change'],.9)

    def test_sequence_guard_cannot_be_hidden_in_mean(self):
        m,s,c,p = fixture();scale(s,1.06,'a','L');scale(s,.5,'b','L')
        x = analyze(s,c,p,m,True)
        self.assertTrue(x['selected16_vs_current78_gate']['scopes']['overall']['L']['pass'])
        self.assertFalse(x['selected16_vs_current78_gate']['pass'])

    def test_empty_main_region_is_not_positive_evidence(self):
        m,s,c,p = fixture()
        for rows in (s,c,p):
            for row in rows:row['regions']['L']={'valid_pixels':0,'error_sum':0.,'AEE':None}
        x = analyze(s,c,p,m,True)
        self.assertFalse(x['conversion_gate_positive_support'])
        self.assertIn('CURRENT_REFERENCE_GATE_UNSUPPORTED',x['next_step_scope']['reasons'])

    def test_wrong_population_support_or_nonfinite_rejected(self):
        for kind in ('order','support','nan','negative'):
            m,s,c,p = fixture()
            if kind=='order':s[0],s[1]=s[1],s[0]
            if kind=='support':s[0]['regions']['all_valid']['valid_pixels']=11
            if kind=='nan':s[0]['regions']['B']['AEE']=float('nan')
            if kind=='negative':s[0]['regions']['B'].update(AEE=-1.,error_sum=-10.)
            with self.subTest(kind=kind),self.assertRaises(ValueError):analyze(s,c,p,m,True)

    def test_execution_required_and_no_new_subset_grid(self):
        m,s,c,p=fixture();x=analyze(s,c,p,m,False)
        self.assertIn('EXECUTION_NOT_VERIFIED',x['next_step_scope']['reasons'])
        self.assertFalse(x['next_step_scope']['additional_BN_subset_grid'])
        with self.assertRaises(ValueError):analyze(s,c,p,m,1)

    def test_all_regions_reported_without_frame_gates(self):
        m,s,c,p=fixture();x=analyze(s,c,p,m,True)
        self.assertEqual(len(x['all_endpoints']),3)
        for rows in x['per_frame_region_deltas'].values():
            self.assertEqual(len(rows),8)
            self.assertTrue(all(set(r['regions'])==set(REGIONS) for r in rows))
        self.assertEqual(len(x['summaries']['selected16/native']),3)


if __name__=='__main__':unittest.main()
