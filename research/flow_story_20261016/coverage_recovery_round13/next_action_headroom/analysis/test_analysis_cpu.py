#!/usr/bin/env python3
"""Synthetic-only checks; no source results/arrays are opened."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import analysis_core as core


def metrics(error=1., count=100, overrides=None):
    out = {r:{'valid_pixels':count,'error_sum':count*error,'AEE':error} for r in core.REGIONS}
    for region, error in (overrides or {}).items():
        out[region] = {'valid_pixels':count,'error_sum':count*error,'AEE':error}
    return out


def fixture():
    members = [{'target_id':'synthetic_%d' % i,'sequence':'synthetic_sequence_%d' % i} for i in range(8)]
    h0 = [dict(m,arms={a:metrics() for a in core.ANCHORS}) for m in members]
    h1 = []
    for m in members:
        arms = {core.action((i,)):metrics(1.01) for i in range(12)}
        arms.update({core.action(p):metrics(1.01) for p in core.PAIRS})
        scores = [dict(tile=list(t),**{s:float(12-i) for s in core.SCORES}) for i,t in enumerate(core.TILES)]
        h1.append(dict(m,arms=arms,ordinary_scores=scores))
    return members,h0,h1


class SyntheticTests(unittest.TestCase):
    def test_complete_coverage_and_count(self):
        members,h0,h1 = fixture()
        report = core.analyze_h1(h0,h1,members)
        self.assertEqual(len(core.PAIRS),66)
        self.assertEqual(len(report['all_actual_endpoint_summaries']),78)
        self.assertEqual(len(report['all_actual_endpoint_summaries']['k1_r0c0']),9)
        self.assertEqual(len(report['all_actual_endpoint_summaries']['k1_r0c0']['overall']),16)
        self.assertEqual(report['label_counts'],{'empty_background':96,'ordered_single_background':1056,'independent_frames':8})
        with self.assertRaisesRegex(ValueError,'Complete H8'):
            core.analyze_h1(h0[:7],h1[:7],members[:7])
        incomplete = copy.deepcopy(h1)
        del incomplete[0]['arms'][core.action((0,1))]
        with self.assertRaisesRegex(ValueError,'Complete endpoint'):
            core.analyze_h1(h0,incomplete,members)

    def test_actual_joint_and_nonmonotonic_regions(self):
        members,h0,h1 = fixture()
        for f in h1:
            f['arms'][core.action((0,))] = metrics(.98)
            f['arms'][core.action((1,))] = metrics(.99)
            f['arms'][core.action((0,1))] = metrics(1.02)
            f['arms'][core.action((2,3))] = metrics(.8,overrides={'B':1.2})
            f['arms'][core.action((4,5))] = metrics(.9)
        report = core.analyze_h1(h0,h1,members)
        self.assertEqual(report['allGT_oracle']['selected_pairs'],[[2,3]]*8)
        self.assertFalse(report['allGT_oracle']['gate']['pass_supported_constraints'])
        self.assertEqual(report['region_feasible_oracle']['selected_pairs'],[[4,5]]*8)
        self.assertEqual(report['singleton_top2_actual_endpoint']['selected_pairs'],[[0,1]]*8)
        self.assertAlmostEqual(report['singleton_top2_minus_allGT_oracle']['overall']['all_valid'],.22)
        interaction = next(x for x in report['interaction'][0]['pairs'] if x['pair']==[0,1])
        self.assertAlmostEqual(interaction['interaction'],-.05)
        self.assertAlmostEqual(interaction['actual_joint_gain'],-.02)
        conditional = next(x for x in report['interaction'][0]['ordered_conditional'] if x['background']==0 and x['tile']==1)
        self.assertTrue(conditional['strict_sign_flip'])
        self.assertEqual(len(report['interaction'][0]['ordered_conditional']),132)
        self.assertEqual(report['investment_screen']['decision'],'STOP_THIS_W_Q_K2_HEAD_INSUFFICIENT_MARGIN')

    def test_same_method_all_frames_and_fixed_pair_scope(self):
        members,h0,h1 = fixture()
        for index,f in enumerate(h1):
            best = core.PAIRS[index+20]
            f['arms'][core.action(best)] = metrics(.9)
            f['arms'][core.action((0,1))] = metrics(.95 if index%2==0 else 1.02)
            f['arms'][core.action((2,3))] = metrics(1.02 if index%2==0 else .95)
            for i,row in enumerate(f['ordinary_scores']):
                row['amplitude_abs_mean'] = 10. if i in (0,1) else 0.
                row['feature_rms'] = 10. if i in (2,3) else 0.
        report = core.analyze_h1(h0,h1,members)
        ordinary = report['ordinary_controls']
        self.assertEqual(ordinary['amplitude_abs_mean']['selected_pairs'],[[0,1]]*8)
        self.assertEqual(ordinary['feature_rms']['selected_pairs'],[[2,3]]*8)
        chosen = ordinary[report['investment_screen']['strongest_single_ordinary_name']]
        self.assertGreater(chosen['summary']['overall']['all_valid']['frame_mean_AEE'],.95)
        self.assertEqual(report['investment_screen']['decision'],'MARGIN_FOR_FURTHER_REVIEW_ONLY')
        self.assertEqual(len(set(tuple(p) for p in report['feasible_best_fixed_pair']['result']['selected_pairs'])),1)

    def test_no_shared_feasible_pair_but_framewise_exists(self):
        members,h0,h1 = fixture()
        for index,f in enumerate(h1):
            preferred = core.PAIRS[index]
            for pair in core.PAIRS:
                f['arms'][core.action(pair)] = metrics(.9 if pair==preferred else 1.1)
            for i,row in enumerate(f['ordinary_scores']):
                for score in core.SCORES:
                    row[score] = 10. if i in preferred else 0.
        report = core.analyze_h1(h0,h1,members)
        self.assertEqual(report['feasible_fixed_pair_count'],0)
        self.assertIsNone(report['feasible_best_fixed_pair'])
        self.assertIsNotNone(report['region_feasible_oracle'])
        self.assertEqual(report['per_frame_feasible_pair_count'],[1]*8)
        for pair in core.PAIRS:
            h1[0]['arms'][core.action(pair)] = metrics(1.1)
        report = core.analyze_h1(h0,h1,members)
        self.assertIsNone(report['region_feasible_oracle'])
        self.assertEqual(report['investment_screen']['decision'],'NO_REGION_FEASIBLE_ORACLE')

    def test_ties_and_primary_controls(self):
        members,h0,h1 = fixture()
        for f in h1:
            for row in f['ordinary_scores']:
                for score in core.SCORES:
                    row[score] = 1.
        report = core.analyze_h1(h0,h1,members)
        self.assertEqual(report['allGT_oracle']['selected_pairs'],[[0,1]]*8)
        self.assertEqual(report['ordinary_controls']['fixed_uniform']['selected_pairs'],[[3,8]]*8)
        self.assertEqual(report['ordinary_controls']['psn1_activity']['selected_pairs'],[[0,1]]*8)
        self.assertEqual(report['investment_screen']['strongest_single_ordinary_name'],'fixed_uniform')
        self.assertEqual(len(report['ordinary_primary_score_names']),4)

    def test_missing_support_and_no_W0_fabrication(self):
        members,h0,h1 = fixture()
        for f in h0+h1:
            for arm in f['arms'].values():
                arm['B'] = {'valid_pixels':0,'error_sum':0.,'AEE':None}
        report = core.analyze_h0(h0,members)
        self.assertIsNone(report['normality_W0_gate'])
        self.assertTrue(report['same_W_fine_pass'])
        self.assertFalse(report['same_W_fine_all_regions_observed'])
        report = core.analyze_h1(h0,h1,members)
        self.assertEqual(report['investment_screen']['decision'],'MISSING_REGION_EVIDENCE')
        h1[0]['arms'][core.action((0,1))]['B'] = {'valid_pixels':1,'error_sum':0.,'AEE':0.}
        with self.assertRaisesRegex(ValueError,'Different endpoint region support'):
            core.analyze_h1(h0,h1,members)

    def test_same_W_fine_guard_and_relative_zero(self):
        members,h0,h1 = fixture()
        h0[0]['arms']['native'] = metrics(1.04)
        report = core.analyze_h1(h0,h1,members)
        self.assertEqual(report['investment_screen']['decision'],'SAME_W_FINE_PROTECTION_FAILED_NO_INFORMATION_ATTRIBUTION')
        self.assertIsNone(core.relative_margin(0.,0.))
        summary = core.summarize([metrics(0.)]*8,members)
        comparison = core.compare(summary,summary)
        self.assertTrue(comparison['pass_supported_constraints'])
        self.assertIsNone(comparison['scopes']['overall']['all_valid']['relative_change'])

    def test_frame_mean_vs_pixel_mean_and_exact_threshold(self):
        members,h0,h1 = fixture()
        values = [metrics(1.,1)] + [metrics(2.,100)]*7
        summary = core.summarize(values,members)['overall']['all_valid']
        self.assertAlmostEqual(summary['frame_mean_AEE'],15/8)
        self.assertAlmostEqual(summary['pixel_weighted_AEE'],1401/701)
        reference = core.summarize([metrics()]*8,members)
        boundary = core.summarize([metrics(1.03,overrides={'B':1.05,'L':1.05})]*8,members)
        self.assertTrue(core.compare(boundary,reference)['pass_supported_constraints'])
        above = core.summarize([metrics(1.03000001)]*8,members)
        self.assertFalse(core.compare(above,reference)['pass_supported_constraints'])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'schema':'flow_headroom_analysis_synthetic_cpu_v1','synthetic_only':True,
               'real_quality_or_input_arrays_read':False,'tests_run':result.testsRun,'failures':len(result.failures),
               'errors':len(result.errors),'pass':result.wasSuccessful(),
               'source_sha256':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                                 for n in ('analysis_core.py','test_analysis_cpu.py','ANALYSIS_PROTOCOL.md')}}
    print(json.dumps(receipt,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
