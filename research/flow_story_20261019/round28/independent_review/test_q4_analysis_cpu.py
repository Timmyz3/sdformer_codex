import copy,unittest
import q4_analysis as q

def fixture():
 frames=[]
 for i in range(64):
  regs={r:dict(valid_pixels=1000,error_sum=1000.,AEE=1.) for r in q.old.REGIONS}
  frames.append(dict(target_id=str(i),sequence='s'+str(i//8),states={s:copy.deepcopy(regs) for s in q.old.STATES}))
 ref=q.old.analyze(frames);ref.update(schema='flow_r27_independent_raw_quality_v1',status='RAW_METRICS_COMPLETE')
 new=[dict(target_id=f['target_id'],sequence=f['sequence'],states={s:copy.deepcopy(f['states']['original_native']) for s in q.NEW_STATES}) for f in frames]
 return ref,new

class Test(unittest.TestCase):
 def test_new_two_only_and_immutable(self):
  ref,new=fixture();before=copy.deepcopy(ref);v=q.analyze(ref,new)
  self.assertEqual(ref,before);self.assertTrue(v['Q4_uniform_dual_reference_pass']);self.assertEqual(set(v['summaries']),set(q.NEW_STATES));self.assertEqual(v['new_frame_region_cells'],2048)
 def test_failure_does_not_stop_mixed_family(self):
  ref,new=fixture()
  for f in new:f['states']['recovered_Q4']['L'].update(error_sum=1100.,AEE=1.1)
  v=q.analyze(ref,new);self.assertFalse(v['Q4_uniform_dual_reference_pass']);self.assertEqual(v['configuration_decision'],'UNIFORM_Q4_NOT_QUALIFIED_MIXED_FAMILY_UNDECIDED')
 def test_tail_flag_and_bad_population(self):
  ref,new=fixture();new[0]['states']['recovered_Q4']['L'].update(error_sum=1300.,AEE=1.3)
  v=q.analyze(ref,new);self.assertTrue(v['Q4_uniform_dual_reference_pass']);self.assertEqual(len(v['comparisons']['recovered_Q4__vs__recovered_native']['high_support_tail_review']),1)
  new[0]['states']['recovered_Q4']['L']['valid_pixels']=999
  with self.assertRaises(ValueError):q.analyze(ref,new)

if __name__=='__main__':unittest.main()
