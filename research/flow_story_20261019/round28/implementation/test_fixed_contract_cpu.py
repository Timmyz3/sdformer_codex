#!/usr/bin/env python3
"""R28 only changes the fixed low-plane action; original R27 sources read only."""
import ast,hashlib,importlib.util,json,os,sys,time,unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;OLD=HERE.parent.parent/'round27/implementation'
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Tests(unittest.TestCase):
 def test_frozen_R27_sources_unchanged(self):
  self.assertEqual(sha(OLD/'recovery_integer_core.py'),'19623636229b4dc5a85132e4acc8bf217f9f62a5e51fcab0fdd7394ee186f788')
  self.assertEqual(sha(OLD/'run_smoke.py'),'48563d36f1b448a66d18278e0448deee02a7650583182a628dcb3865923c6c5c')
  self.assertEqual(sha(OLD/'run_recovery.py'),'495f7dbfdc9d54f74ed2f40ef5cfb9c6fcc58da5a03998efca733845d0b6753a')
 def test_recipe_scope_members_source_binding(self):
  old_s=load(OLD/'run_smoke.py','_r28_test_old_smoke');new_s=load(HERE/'run_smoke.py','_r28_test_new_smoke')
  old_r=load(OLD/'run_recovery.py','_r28_test_old_runner');new_r=load(HERE/'run_recovery.py','_r28_test_new_runner')
  self.assertEqual(old_s.RECIPE,new_s.RECIPE);self.assertEqual(old_r.recipe(old_s),new_r.recipe(new_s))
  self.assertEqual(old_s.MEMBERS_SHA,new_s.MEMBERS_SHA);self.assertEqual(new_s.CORE_SHA,sha(HERE/'recovery_integer_core.py'))
  self.assertEqual(new_r.SMOKE_SHA,sha(HERE/'run_smoke.py'));self.assertEqual(new_r.CORE_SHA,new_s.CORE_SHA)
  self.assertIn('flow_r28_',new_r.GRAPH);self.assertIn('_Q4_',new_r.GRAPH)
 def test_only_fixed_Q4_executable_arm(self):
  for file in ('run_smoke.py','run_recovery.py'):
   tree=ast.parse((HERE/file).read_text());options=[]
   for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value=='--arm':
     options.append(next(ast.literal_eval(x.value) for x in node.keywords if x.arg=='choices'))
   self.assertEqual(options,[('Q4',)])
 def test_unchanged_forward_adapters_and_surrogate_derivative(self):
  old=ast.parse((OLD/'recovery_integer_core.py').read_text());new=ast.parse((HERE/'recovery_integer_core.py').read_text())
  maps=[{n.name:ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))} for t in (old,new)]
  for name in ('quantize','tail_rows','kernel_mm','_LinearSTE','_ConvSTE','deploy_forward','installed','trainable_parameters'):
   self.assertEqual(maps[0][name],maps[1][name],name)
if __name__=='__main__':
 start=time.monotonic();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));p=Path(sys.argv[1]);assert not p.exists()
 p.write_text(json.dumps({'schema':'flow_r28_fixed_contract_cpu_v1','status':'PASS' if r.wasSuccessful() else 'FAIL','tests':r.testsRun,
  'failures':len(r.failures),'errors':len(r.errors),'source_sha256':{x:sha(HERE/x) for x in ('recovery_integer_core.py','run_smoke.py','run_recovery.py','test_fixed_contract_cpu.py')},
  'GPU_used':False,'GT_values_read':False,'model_forwards':0,'wall_seconds':time.monotonic()-start},indent=2)+'\n')
 sys.exit(0 if r.wasSuccessful() else 1)
