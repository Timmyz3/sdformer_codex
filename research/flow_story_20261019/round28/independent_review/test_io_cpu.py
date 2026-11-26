import copy,unittest
import reduce_raw_v1 as r

class Test(unittest.TestCase):
 def fixture(self):
  ids=[str(i) for i in range(64)];f=dict(auditor=dict(sha256='a'),audit_freeze=dict(sha256='b'),dependencies=dict(entry_scope_correction=dict(sha256='c')))
  a=dict(schema=r.AUDIT_SCHEMA,status='PASS',quality_analysis_allowed=True,auditor_sha256='a',auditor_freeze_sha256='b',pre_result_freeze_sha256=r.ANALYSIS_SHA,entry_scope_correction_sha256='c',counts=dict(optimizer_updates=512,quality_predictions=128,quality_archives=128,formal_model_forwards=1155),source_results={'Q4':'s'},predictions=[dict(state=s,target_id=t) for s in r.core.NEW_STATES for t in ids])
  return ids,f,a
 def test_exact_population(self):
  ids,f,a=self.fixture();self.assertEqual(len(r.validate_audit(a,f,ids)),128)
 def test_incomplete_and_wrong_binding_rejected(self):
  ids,f,a=self.fixture()
  for key,val in [('status','RUNNING'),('quality_analysis_allowed',False),('auditor_freeze_sha256','wrong'),('pre_result_freeze_sha256','wrong'),('entry_scope_correction_sha256','wrong')]:
   bad=copy.deepcopy(a);bad[key]=val
   with self.assertRaises(ValueError):r.validate_audit(bad,f,ids)
  bad=copy.deepcopy(a);bad['predictions'][-1]=bad['predictions'][0]
  with self.assertRaises(ValueError):r.validate_audit(bad,f,ids)
 def test_safe_paths(self):
  for s in ['/tmp/a','../a','foo/../../a']:
   with self.assertRaises(ValueError):r.safe('/tmp/root',s)
  self.assertEqual(str(r.audit_dependency('/tmp/flowdate/round28/AUDITOR_FREEZE.json','../round27/audit.py')),'/tmp/flowdate/round27/audit.py')
  with self.assertRaises(ValueError):r.audit_dependency('/tmp/flowdate/round28/AUDITOR_FREEZE.json','../../general/audit.py')
if __name__=='__main__':unittest.main()
