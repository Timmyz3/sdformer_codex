import copy
import unittest
import reduce_raw_v1 as io

class QualityIO(unittest.TestCase):
 def fixture(self):
  r=dict(schema='flow_r26_window_precision_v1',status='WINDOW_PRECISION42_COMPLETE',graph_id=io.GRAPH,binding={'graph_id':io.GRAPH},full_forward_count=42,full_forward_attempts=42,training_performed=False,optimizer_steps=0,calibration_forwards=0,all_states_unchanged=True,all_hooks_removed=True,original_modules_restored=True,forward_records=[{}]*42,backend_operator_smoke=[dict(passed_exact_column_sum=True,model_forward_count=0)]*4)
  c=dict(full_forwards=42,native=8,integer_reference=8,bitserial=26,quality_endpoints=40,determinism_replays=2,full_grid_FC1_rows=23436000,full_grid_FC2_rows=23436000,original_FC1_rows=23028000,original_FC2_rows=23028000,target_row_plane_FC1_work=1990560,target_row_plane_FC2_work=1990560,model_int_mm_calls=432,backend_operator_calls=4,BN_calls=3276,archives=42,head_payloads=168,float_capture_payloads=108,integer_capture_payloads=38,saved_payloads=314,quantization_archives=1,quantization_payloads=4,optimizer_steps=0,calibration_forwards=0)
  a=dict(schema='flow_r26_window_precision_independent_execution_audit_v1',status='PASS',quality_analysis_allowed=True,graph_id=io.GRAPH,counts=c,predictions=[{}]*42,numerical_checks=[{'kind':k,'pass':True} for k in ['Q8serial_vs_Q8ref_exact']*8+['Q6_exact_replay']*2])
  return r,a
 def test_plan_and_replay_separation(self):
  x=io.plan([{'target_id':str(i)} for i in range(8)])
  self.assertEqual(len(x),42);self.assertEqual(sum(v[3] for v in x),40)
  self.assertEqual([v[1] for v in x[:7]],['native','Q8ref','Q8serial','Q6','Q6_repeat1','Q6_repeat2','Q4'])
 def test_complete_metadata_and_rejections(self):
  r,a=self.fixture();io.validate_complete(r,a)
  for key in a['counts']:
   b=copy.deepcopy(a);b['counts'][key]+=1
   with self.subTest(count=key),self.assertRaises(ValueError):io.validate_complete(r,b)
  for transform in (lambda x:x.update(quality_analysis_allowed=False),lambda x:x['numerical_checks'][0].update(pass_=False),lambda x:x['numerical_checks'][0].update(kind='FF_vs_native')):
   b=copy.deepcopy(a);transform(b)
   if 'pass_' in b['numerical_checks'][0]:b['numerical_checks'][0]['pass']=False
   with self.assertRaises(ValueError):io.validate_complete(r,b)
  r['full_forward_count']=41
  with self.assertRaises(ValueError):io.validate_complete(r,a)
 def test_frozen_pure_scope(self):
  _,core=io.frozen_math();self.assertEqual(core.ARMS,io.ARMS);self.assertEqual(len(core.REGIONS),16)

if __name__=='__main__':unittest.main()
