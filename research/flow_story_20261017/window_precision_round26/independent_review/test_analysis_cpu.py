import copy,unittest
import analysis_core as c
def metric(v,n=2):return dict(valid_pixels=n,error_sum=v*n if n else 0.,AEE=v if n else None)
def fixture():return [dict(target_id=str(i),sequence='s'+str(i),arms={a:{r:metric(2.) for r in c.REGIONS} for a in c.ARMS}) for i in range(8)]
class Checks(unittest.TestCase):
 def test_anchor_and_primary(self):
  x=c.analyze(fixture(),serial8_exact=True);self.assertTrue(x['Q6_learning_action_candidate']);self.assertEqual(x['frame_region_cells'],640);self.assertEqual(x['summary_cells'],720)
  f=fixture();f[0]['arms']['Q8ref']['L']=metric(2.2);f[0]['arms']['Q8serial']['L']=metric(2.2)
  x=c.analyze(f,serial8_exact=True);self.assertFalse(x['Q8_conversion_anchor_pass']);self.assertFalse(x['Q6_learning_action_candidate']);self.assertEqual(x['resource_decision'],'STOP_Q8_DEPLOYMENT_ANCHOR_FAILED')
 def test_q4_cannot_rescue_q6_and_zero_support(self):
  f=fixture();f[0]['arms']['Q6']['L']=metric(2.2);x=c.analyze(f,serial8_exact=True)
  self.assertTrue(x['Q4_diagnostic_pass']);self.assertFalse(x['Q4_promoted_to_primary']);self.assertFalse(x['Q6_learning_action_candidate']);self.assertEqual(x['per_frame_two_mode_feasibility'][0]['feasible_modes'],['Q4'])
  f[0]['arms']['Q4']['L']=metric(2.2);x=c.analyze(f,serial8_exact=True);self.assertEqual(x['zero_feasible_mode_frames'],['0'])
  for a in c.ARMS:f[0]['arms'][a]['L']=metric(0.,0)
  x=c.analyze(f,serial8_exact=True);self.assertFalse(x['Q8_conversion_anchor_pass']);self.assertIsNone(x['summaries']['native']['s0']['L']['frame_mean_AEE'])
 def test_exact_checks_original_direction_and_limits(self):
  f=fixture();f[0]['arms']['Q6']['all_valid']=metric(2.06);f[0]['arms']['Q6']['B']=metric(2.1);f[0]['arms']['Q6']['L']=metric(2.1)
  x=c.analyze(f,serial8_exact=True);self.assertTrue(x['Q6_primary_pass']);self.assertAlmostEqual(x['signed_changes_vs_native'][0]['Q6']['all_valid']['signed_AEE'],.06)
  with self.assertRaises(ValueError):c.analyze(f,serial8_exact=False)
  f[0]['arms']['Q8serial']['B']=metric(2.0001)
  with self.assertRaises(ValueError):c.analyze(f,serial8_exact=True)
if __name__=='__main__':unittest.main()
