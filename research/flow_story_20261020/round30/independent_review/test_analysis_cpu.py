import copy
import unittest
import analysis_joint as m

def fixture():
    rows=[]
    for i in range(64):
        metric={r:dict(valid_pixels=1000,error_sum=1000.,AEE=1.) for r in m.old.REGIONS}
        rows.append(dict(target_id=str(i),sequence=str(i//8),
            states={s:copy.deepcopy(metric) for s in m.old.STATES}))
    r27=m.old.analyze(rows)
    r27.update(schema='flow_r27_independent_raw_quality_v1',status='RAW_METRICS_COMPLETE')
    r28=dict(schema='flow_r28_independent_raw_quality_v1',status='RAW_METRICS_COMPLETE',
        frames=[dict(target_id=f['target_id'],sequence=f['sequence'],
            states={s:copy.deepcopy(f['states']['original_native']) for s in ('initial_Q4','recovered_Q4')})
            for f in rows],
        summaries={s:copy.deepcopy(r27['summaries']['original_native']) for s in ('initial_Q4','recovered_Q4')})
    new=[dict(target_id=f['target_id'],sequence=f['sequence'],
        states={s:copy.deepcopy(f['states']['original_native']) for s in m.NEW_STATES}) for f in rows]
    return r27,r28,new

def set_metric(frames,state,region,aee,n=1000):
    for f in frames:f['states'][state][region].update(valid_pixels=n,error_sum=aee*n,AEE=aee if n else None)

class Test(unittest.TestCase):
    def test_new_four_states_and_no_historical_mutation(self):
        a,b,n=fixture();before=copy.deepcopy((a,b));v=m.analyze(a,b,n)
        self.assertEqual((a,b),before)
        self.assertEqual(v['new_quality_predictions'],256)
        self.assertEqual(v['new_frame_region_cells'],4096)
        self.assertEqual(v['new_summary_cells'],576)
        self.assertEqual(len(v['comparisons']),23)
        self.assertEqual(set(v['summaries']),set(m.NEW_STATES))
        self.assertTrue(v['M_native_dual_reference_pass'])
    def test_Q8_bad_does_not_veto_valid_M(self):
        a,b,n=fixture();set_metric(n,'joint_Q8','all_valid',1.8)
        v=m.analyze(a,b,n)
        self.assertFalse(v['descriptive_native_dual_reference_pass']['joint_Q8'])
        self.assertTrue(v['M_native_dual_reference_pass'])
        self.assertFalse(v['Q8_is_extra_M_veto'])
        self.assertTrue(v['comparisons']['joint_Q8__vs__original_native']['thresholds_are_descriptive_only'])
    def test_M_bad_is_not_saved_by_Q8(self):
        a,b,n=fixture();set_metric(n,'joint_M','B',1.051)
        v=m.analyze(a,b,n)
        self.assertTrue(v['descriptive_native_dual_reference_pass']['joint_Q8'])
        self.assertFalse(v['M_native_dual_reference_pass'])
        self.assertEqual(v['configuration_decision'],'FIXED_M_NOT_QUALIFIED_OTHER_MIXED_UNDECIDED')
    def test_both_native_references_are_required(self):
        a,b,n=fixture()
        for f in a['frames']:f['states']['recovered_native']['L'].update(AEE=.9,error_sum=900.)
        a=m.old.analyze(a['frames']);a.update(schema='flow_r27_independent_raw_quality_v1',status='RAW_METRICS_COMPLETE')
        v=m.analyze(a,b,n)
        self.assertTrue(v['comparisons']['joint_M__vs__original_native']['aggregate_pass'])
        self.assertFalse(v['comparisons']['joint_M__vs__recovered_native']['aggregate_pass'])
        self.assertFalse(v['M_native_dual_reference_pass'])
    def test_tail_flag_is_visible_not_an_extra_veto(self):
        a,b,n=fixture();n[0]['states']['joint_M']['all_valid'].update(AEE=1.21,error_sum=1210.)
        v=m.analyze(a,b,n);c=v['comparisons']['joint_M__vs__original_native']
        self.assertTrue(v['M_native_dual_reference_pass'])
        self.assertEqual(len(c['high_support_tail_review']),1)
        self.assertEqual(c['high_support_tail_review'][0]['target_id'],'0')
    def test_support_or_identity_corruption_rejected(self):
        for corruption in ('support','identity','old_summary'):
            a,b,n=fixture()
            if corruption=='support':n[0]['states']['joint_M']['L'].update(valid_pixels=12,error_sum=12.)
            elif corruption=='identity':n[0]['target_id']='not_member'
            else:a['summaries']['original_native']['overall']['L']['error_sum']+=1
            with self.assertRaises(ValueError):m.analyze(a,b,n)
    def test_empty_and_zero_reference_not_fake_zero_relative(self):
        self.assertFalse(m.old.one_gate(0.,None,.05,0)['pass'])
        self.assertTrue(m.old.one_gate(0.,0.,.05,1000)['pass'])
        self.assertFalse(m.old.one_gate(1e-9,0.,.05,1000)['pass'])
        self.assertIsNone(m.old.one_gate(0.,0.,.05,1000)['relative_change'])

if __name__=='__main__':unittest.main()
