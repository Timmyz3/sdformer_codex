import copy,unittest
import reduce_same_weight_v1 as r
class Test(unittest.TestCase):
 def fixture(self):
  ids=[str(i) for i in range(64)];f=dict(auditor=dict(sha256='a'));a=dict(schema=r.AUDIT_SCHEMA,status='PASS',quality_analysis_allowed=True,auditor_sha256='a',counts=dict(optimizer_updates=0,quality_predictions=64,formal_model_forwards=67),source_result_sha256='r',Q4_source_checkpoint_sha256='w',source_state_sha256='s',R28_execution_audit_sha256='b',predictions=[dict(state=r.core.STATE,target_id=t) for t in ids]);return ids,f,a
 def test_complete64(self):
  ids,f,a=self.fixture();self.assertEqual(len(r.validate_audit(a,f,ids,'r','w','s','b')),64)
 def test_wrong_W_incomplete_and_dup_rejected(self):
  ids,f,a=self.fixture()
  for key,value in [('status','RUNNING'),('quality_analysis_allowed',False),('Q4_source_checkpoint_sha256','wrong'),('source_state_sha256','wrong'),('R28_execution_audit_sha256','wrong')]:
   b=copy.deepcopy(a);b[key]=value
   with self.assertRaises(ValueError):r.validate_audit(b,f,ids,'r','w','s','b')
  b=copy.deepcopy(a);b['predictions'][-1]=b['predictions'][0]
  with self.assertRaises(ValueError):r.validate_audit(b,f,ids,'r','w','s','b')
if __name__=='__main__':unittest.main()
