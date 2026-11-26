#!/usr/bin/env python3
"""Exercise the actual frozen R27 membership through the R28 V2 input loader."""
import ast,hashlib,importlib.util,json,os,sys,tempfile,time,unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
MEMBERS=HERE.parent.parent/'round27/independent_review/membership_v1.json'
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
smoke=load(HERE/'run_smoke_v2.py','_r28_smoke_v2_membership_test')
formal=load(HERE/'run_recovery_v2.py','_r28_formal_v2_membership_test')
class Tests(unittest.TestCase):
 def test_real_frozen_input_loader_and_full_order(self):
  m=smoke.load_membership(MEMBERS)
  self.assertEqual(smoke.sha(MEMBERS),'40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b')
  self.assertEqual(m['schema'],'flow_r27_recovery_identity_v1');formal.validate_order(m)
  self.assertEqual((len(m['train_members']),len(m['dev_members']),len(m['train_order'])),(256,64,512))
 def test_V1_frozen_bytes_and_actual_schema_failure_retained(self):
  freeze=json.loads((HERE/'PREPARATION_FREEZE_V1.json').read_text())
  for name,expected in freeze['files'].items():self.assertEqual(smoke.sha(HERE/name),expected,name)
  members=json.loads(MEMBERS.read_text());conditions=[]
  for node in ast.walk(ast.parse((HERE/'run_smoke.py').read_text())):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='require' and node.args and "members['schema']" in ast.unparse(node.args[0]):conditions.append(node.args[0])
  self.assertEqual(len(conditions),1)
  self.assertFalse(eval(compile(ast.Expression(conditions[0]),'<frozen_v1_schema_condition>','eval'),{'members':members}))
 def test_modified_metadata_bytes_rejected(self):
  m=json.loads(MEMBERS.read_text());m['schema']='flow_r28_recovery_identity_v1'
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'modified.json';p.write_text(json.dumps(m))
   with self.assertRaisesRegex(ValueError,'Fixed original R27 recovery roles'):smoke.load_membership(p)
 def test_V2_dependency_and_recipe_closure(self):
  self.assertEqual(formal.SMOKE_SHA,smoke.sha(HERE/'run_smoke_v2.py'))
  self.assertEqual(formal.CORE_SHA,smoke.CORE_SHA)
  self.assertEqual(smoke.CORE_SHA,'f4e4dae18a7e59fab286f2e59f23f72d20583a2b8de22dee4c78204bbfd2fe9a')
  original=load(HERE.parent.parent/'round27/implementation/run_smoke.py','_r28_original_recipe_test')
  self.assertEqual(original.RECIPE,smoke.RECIPE)
  self.assertIn('members=load_membership(a.members)',(HERE/'run_smoke_v2.py').read_text())
  self.assertIn("HERE/'run_smoke_v2.py'",(HERE/'run_recovery_v2.py').read_text())
if __name__=='__main__':
 start=time.monotonic();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));p=Path(sys.argv[1]);assert not p.exists()
 p.write_text(json.dumps({'schema':'flow_r28_real_membership_cpu_v2','status':'PASS' if r.wasSuccessful() else 'FAIL','tests':r.testsRun,'errors':len(r.errors),'failures':len(r.failures),
  'sources':{n:smoke.sha(HERE/n) for n in ('run_smoke_v2.py','run_recovery_v2.py','test_membership_cpu_v2.py')},
  'actual_frozen_membership_sha256':smoke.sha(MEMBERS),'actual_input_loader_exercised':True,'full_remote_asset_preflight_performed':False,
  'V1_source_and_receipts_preserved':True,'GPU_used':False,'GT_arrays_read':False,'model_forwards':0,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
 sys.exit(0 if r.wasSuccessful() else 1)
