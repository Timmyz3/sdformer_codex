#!/usr/bin/env python3
"""Real R28 checkpoint/audit/metadata contracts; zero model forwards or GT reads."""
import copy,json,os,sys,unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import torch
import run_same_weight_q8 as r
HERE=Path(__file__).resolve().parent
DATE=HERE.parents[1]
R28=DATE/'round28'

class SourceContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2)
  cls.audit=r.read(R28/'actual_execution_audit_v1.json')
  cls.weights=R28/'actual_v1/formal_Q4_v2/recovered_step512.pt'
  cls.weights_sha=r.sha(cls.weights)
  cls.cp=torch.load(cls.weights,map_location='cpu',weights_only=True)
  cls.smoke=r.load(R28/'implementation/run_smoke_v2.py','_r29_cpu_smoke')
  cls.ev=r.load(R28/'implementation/run_recovery_v2.py','_r29_cpu_ev')

 def test_01_actual_frozen_sources_and_members(self):
  for name,digest in [('run_smoke_v2.py',r.SMOKE_SHA),('run_recovery_v2.py',r.EVAL_SHA),('recovery_integer_core.py',r.CORE_SHA)]:
   self.assertEqual(r.sha(R28/'implementation'/name),digest)
  self.assertEqual(r.sha(R28/'audit_actual_recovery_v1.py'),r.AUDITOR_SHA)
  m=self.smoke.load_membership(DATE/'round27/independent_review/membership_v1.json')
  self.ev.validate_order(m)
  self.assertEqual(m['schema'],'flow_r27_recovery_identity_v1')
  self.assertEqual(len(m['dev_members']),64)
  self.assertEqual(self.cp['recipe'],self.ev.recipe(self.smoke))

 def test_02_real_audited_checkpoint(self):
  state,record=r.validate_checkpoint(self.cp,self.audit,self.weights_sha,torch)
  self.assertEqual(self.weights_sha,'fefa60cc61684c204a79082173f3e10a4d8a1eff3ec5d07021935da8ee1d9196')
  self.assertEqual(len(state),477)
  self.assertEqual(record['state_sha256'],self.cp['final_state_sha256'])
  self.assertEqual(record['source_arm'],'Q4')
  self.assertFalse(torch.cuda.is_initialized())

 def test_03_wrong_audit_and_checkpoint_rejected(self):
  bad=copy.deepcopy(self.audit);bad['counts']['formal_model_forwards']=1154
  with self.assertRaises(ValueError):r.validate_audit(bad)
  bad=copy.deepcopy(self.audit);bad['auditor_sha256']='0'*64
  with self.assertRaises(ValueError):r.validate_audit(bad)
  for field,value in [('arm','Q8'),('completed_optimizer_steps',511),('final_state_sha256','0'*64),('members_sha256','0'*64)]:
   bad=dict(self.cp);bad[field]=value
   with self.assertRaises(ValueError):r.validate_checkpoint(bad,self.audit,self.weights_sha,torch)
  with self.assertRaises(ValueError):r.validate_checkpoint(self.cp,self.audit,'0'*64,torch)

 def test_04_fixed_scope_and_budget(self):
  self.assertEqual(r.STAGE,'diagnostic')
  self.assertEqual(66*(1116000+3072000)*8,2211264000)
  self.assertEqual((64+3)*78,5226);self.assertEqual((64+3)*105,7035)
  # CPU source-level execution scope: no training calls, optimizer or teacher.
  import ast
  tree=ast.parse(Path(r.__file__).read_text())
  calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
  forbidden={'backward','step','training_arrays','trainable_parameters','AdamW','run_training'}
  self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in calls))
  capture=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='capture_full']
  self.assertEqual(len(capture),3)
  self.assertEqual([n.args[3].value for n in capture],['Q8']*3)
  self.assertEqual([next(k.value.value for k in n.keywords if k.arg=='ste') for n in capture],[True,False,False])

if __name__=='__main__':unittest.main()
