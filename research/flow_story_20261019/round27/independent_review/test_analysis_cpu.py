import copy,unittest
import analysis_core as core

def fixture():
 return [dict(target_id='seq%d_%d'%(s,i),sequence='seq%d'%s,states={a:{r:dict(valid_pixels=1000,error_sum=1000.,AEE=1.) for r in core.REGIONS} for a in core.STATES}) for s in range(8) for i in range(8)]
class Checks(unittest.TestCase):
 def test_01_full_matrix_and_small_support_not_veto(self):
  f=fixture()
  for a in core.STATES:f[0]['states'][a]['L']=dict(valid_pixels=12,error_sum=12.,AEE=1.)
  f[0]['states']['recovered_Q6']['L']=dict(valid_pixels=12,error_sum=18.,AEE=1.5)
  out=core.analyze(f);self.assertTrue(out['Q6_dual_reference_pass']);self.assertEqual(out['frame_region_cells'],6144)
  c=out['comparisons']['recovered_Q6__vs__original_native'];self.assertEqual(c['support_strata']['L']['1_to_31']['legacy_threshold_exceedance_count'],1);self.assertFalse(c['high_support_tail_review'])
 def test_02_original_and_matched_both_required(self):
  f=fixture()
  for row in f:
   for region in core.REGIONS:
    row['states']['recovered_native'][region]=dict(valid_pixels=1000,error_sum=800.,AEE=.8)
    row['states']['recovered_Q6'][region]=dict(valid_pixels=1000,error_sum=1000.,AEE=1.)
  out=core.analyze(f);self.assertFalse(out['Q6_dual_reference_pass']);self.assertTrue(out['comparisons']['recovered_Q6__vs__original_native']['aggregate_pass']);self.assertFalse(out['comparisons']['recovered_Q6__vs__recovered_native']['aggregate_pass'])
 def test_03_high_support_flags_not_posthoc_gate(self):
  f=fixture();f[0]['states']['recovered_Q6']['B']=dict(valid_pixels=1000,error_sum=1500.,AEE=1.5)
  out=core.analyze(f);self.assertTrue(out['Q6_dual_reference_pass']);self.assertEqual(len(out['comparisons']['recovered_Q6__vs__original_native']['high_support_tail_review']),1)
 def test_04_empty_zero_denominator_and_rejections(self):
  f=fixture()
  for a in core.STATES:f[0]['states'][a]['B_and_L']=dict(valid_pixels=0,error_sum=0.,AEE=None)
  out=core.analyze(f);self.assertEqual(out['summaries']['original_native']['overall']['B_and_L']['empty_frames'],1)
  self.assertFalse(core.one_gate(1.,0.,.03,1)['pass']);self.assertIsNone(core.one_gate(1.,0.,.03,1)['relative_change'])
  f[1]['target_id']=f[0]['target_id']
  with self.assertRaises(ValueError):core.analyze(f)
if __name__=='__main__':unittest.main()
