import copy,unittest
import analysis_same_weight as m

def fixture():
 rows=[]
 for i in range(64):
  metric={r:dict(valid_pixels=1000,error_sum=1000.,AEE=1.) for r in m.old.REGIONS}
  rows.append(dict(target_id=str(i),sequence=str(i//8),states={s:copy.deepcopy(metric) for s in m.old.STATES}))
 r27=m.old.analyze(rows);r27.update(schema='flow_r27_independent_raw_quality_v1',status='RAW_METRICS_COMPLETE')
 r28=dict(schema='flow_r28_independent_raw_quality_v1',status='RAW_METRICS_COMPLETE',frames=[dict(target_id=f['target_id'],sequence=f['sequence'],states={s:copy.deepcopy(f['states']['original_native']) for s in ('initial_Q4','recovered_Q4')}) for f in rows],summaries={s:copy.deepcopy(r27['summaries']['original_native']) for s in ('initial_Q4','recovered_Q4')})
 new=[dict(target_id=f['target_id'],sequence=f['sequence'],states={m.STATE:copy.deepcopy(f['states']['original_native'])}) for f in rows]
 return r27,r28,new

class Test(unittest.TestCase):
 def test_new64_only_and_prior_unchanged(self):
  a,b,n=fixture();before=copy.deepcopy((a,b));v=m.analyze(a,b,n)
  self.assertEqual((a,b),before);self.assertEqual(set(v['summaries']),{m.STATE});self.assertFalse(v['qualification_gate_added']);self.assertEqual(v['new_frame_region_cells'],1024)
 def test_high_precision_may_be_worse_without_new_gate(self):
  a,b,n=fixture()
  for f in n:f['states'][m.STATE]['L'].update(AEE=1.3,error_sum=1300.)
  v=m.analyze(a,b,n);c=v['comparisons'][m.STATE+'__vs__recovered_Q4']
  self.assertFalse(c['aggregate_pass']);self.assertTrue(c['thresholds_are_descriptive_only']);self.assertFalse(c['new_qualification_gate']);self.assertNotIn('resource_decision',v)
 def test_mismatched_identity_rejected(self):
  a,b,n=fixture();n[0]['target_id']='wrong'
  with self.assertRaises(ValueError):m.analyze(a,b,n)
if __name__=='__main__':unittest.main()
