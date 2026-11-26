import copy
import json
from pathlib import Path
import unittest
import summarize_breadth as s


class SummaryTests(unittest.TestCase):
    def fixture(self):
        ids=['t%d'%i for i in range(32)];cases=[{'id':'dense_D','bn':'D','method':'dense'},{'id':'candidate','bn':'D','method':'tile'}]
        arms={}
        for name,error in [('dense_D',1.),('candidate',1.5)]:
            frames=[{'target_id':t,'regions':{'all_valid':{'valid_pixels':10,'error_sum':error*10,'AEE':error},
                                            'B':{'valid_pixels':0,'error_sum':0.,'AEE':None}}} for t in ids]
            arms[name]={'status':'PASS','frames':frames,'BN_buffers_unchanged':True,'summary':{'regions':s.regions_for(frames)}}
        return {'schema':'flow_breadth_exploration_v1','population':{'target_ids':ids,'target_ids_sha256':'test','population':'train_gate_development'},
                'cases':cases,'arms':arms,'checkpoint_sha256':'c'*64,'checkpoint_sidecar':{'completed_optimizer_steps':100},'geometry':'full9'}

    def test_paired_complete_and_empty_strata(self):
        result=s.summarize(self.fixture(),'a'*64)
        self.assertEqual(result['completed_cases'],2);self.assertFalse(result['evidence_errors'])
        self.assertEqual(result['cases'][1]['paired_frame_delta_AEE_vs_dense_D'],.5)
        region=[r for r in result['regions'] if r['case_id']=='candidate' and r['region']=='B'][0]
        self.assertIsNone(region['frame_mean_AEE']);self.assertEqual(region['empty_region_frames'],32)

    def test_pending_and_bad_PASS_not_promoted(self):
        value=self.fixture();value['arms']['candidate']['status']='PENDING_FIT'
        result=s.summarize(value,'a'*64);self.assertEqual(result['completed_cases'],1);self.assertEqual(len(result['incomplete_or_pending']),1)
        value=self.fixture();value['arms']['candidate']['frames'].pop()
        result=s.summarize(value,'a'*64);self.assertEqual(result['completed_cases'],1);self.assertEqual(len(result['evidence_errors']),1)

    def test_stored_summary_mismatch_rejected(self):
        value=self.fixture();value['arms']['candidate']['summary']['regions']['all_valid']['frame_mean_AEE']=0
        result=s.summarize(value,'a'*64);self.assertEqual(result['completed_cases'],1);self.assertEqual(len(result['evidence_errors']),1)

    def test_same_BN_baseline_preserves_cross_BN_D_diagnostic(self):
        value=self.fixture()
        for name,error,method in [('dense_P_paper',.7,'dense'),('static_P_paper',.9,'static')]:
            value['cases'].append({'id':name,'bn':'P_paper','method':method})
            entry=copy.deepcopy(value['arms']['dense_D'])
            for frame in entry['frames']:
                frame['regions']['all_valid'].update(error_sum=error*10,AEE=error)
            entry['summary']['regions']=s.regions_for(entry['frames']);value['arms'][name]=entry
        result=s.summarize(value,'a'*64);row=next(r for r in result['cases'] if r['case_id']=='static_P_paper')
        self.assertEqual(row['same_bn_baseline_id'],'dense_P_paper')
        self.assertAlmostEqual(row['delta_same_bn_frame_AEE'],.2)
        self.assertAlmostEqual(row['paired_frame_delta_AEE_vs_dense_D'],-.1)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SummaryTests))
    (Path(__file__).with_name('summary_cpu_checks.json')).write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,
        'scope':'Synthetic report aggregation checks; no experiments or method quality claims'},indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
