"""Synthetic measurement/gate boundary checks, no real data/model execution."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import analyze_quality as q

def frames():
    names=list(q.ANCHORS)+[q.arm(p,k) for k in q.BUDGETS for p in q.POLICIES]
    out=[]
    for i in range(8):
        f={'target_id':'synthetic_%d'%i,'sequence':'a' if i<4 else 'b','arms':{}}
        for name in names:
            value=1. if name=='native' else .99 if name.startswith('flow11_task_k') else 1.02
            f['arms'][name]={'metrics':{'regions':{r:{'valid_pixels':i+1,'error_sum':value*(i+1),'AEE':value} for r in q.REGIONS}}}
        out.append(f)
    return out

def set_region(f,name,region,value,count=None):
    v=f['arms'][name]['metrics']['regions'][region];n=v['valid_pixels'] if count is None else count
    v.update(valid_pixels=n,error_sum=value*n,AEE=value if n else None)

class QualityBoundaryTests(unittest.TestCase):
    def test_frame_and_pixel_means_and_empty_regions(self):
        f=frames()
        for i,row in enumerate(f):
            for name in row['arms']:
                set_region(row,name,'B',0,count=0)
                set_region(row,name,'all_valid',float(i+1))
        r=q.analyze(f);s=r['summaries']['native']['scopes']['overall']
        self.assertEqual(s['all_valid']['frame_mean_AEE'],4.5)
        self.assertAlmostEqual(s['all_valid']['pixel_weighted_AEE'],204/36)
        self.assertIsNone(s['B']['frame_mean_AEE'])
        self.assertFalse(r['per_budget_gates']['2']['application_overall_native']['B']['evidence'])

    def test_nonempty_zero_reference_fails_positive_candidate(self):
        f=frames()
        for row in f:
            set_region(row,'CC','B',0)
            set_region(row,q.arm(q.PRIMARY,2),'B',.001)
        r=q.analyze(f)['per_budget_gates']['2']
        self.assertFalse(r['B_L_vs_CC_guards']['overall']['B']['pass'])
        self.assertFalse(r['pilot_gate_passed'])
        self.assertIsNone(r['B_L_vs_CC_guards']['overall']['B']['relative_change'])

    def test_budgets_independent_and_tied_control_not_a_win(self):
        f=frames();r=q.analyze(f)
        self.assertTrue(r['per_budget_gates']['2']['main_quality_increment_gate_passed'])
        before=copy.deepcopy(r['per_budget_gates']['4'])
        for row in f:
            for reg in q.REGIONS:set_region(row,'B5_task_k2',reg,.99)
        r=q.analyze(f)
        self.assertFalse(r['per_budget_gates']['2']['strictly_beats_all_15_comparators'])
        self.assertEqual(before,r['per_budget_gates']['4'])

    def test_sequence_reversal_survives_good_pooled_mean(self):
        f=frames()
        for i,row in enumerate(f):
            set_region(row,'flow11_task_k2','all_valid',1.021 if i<4 else .90)
        r=q.analyze(f)['per_budget_gates']['2']
        self.assertTrue(r['strictly_beats_all_15_comparators'])
        self.assertFalse(r['sequence_allGT_nonreverse']['a']['B5_task'])
        self.assertFalse(r['pilot_gate_passed'])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(QualityBoundaryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    here=Path(__file__).resolve().parent
    receipt={'schema':'flow_round11_quality_synthetic_checks_v1','passed':result.wasSuccessful(),
        'tests_run':result.testsRun,'failures':[str(v) for v in result.failures],
        'errors':[str(v) for v in result.errors],
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (here/'analyze_quality.py',Path(__file__))},
        'scope':'Synthetic quality reductions and fixed gate boundaries only; no real member/GT, fitting, GPU, or full network.'}
    (here/'quality_cpu_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
