#!/usr/bin/env python3
"""New formal orchestration contracts, synthetic arrays only; no CUDA or real GT."""
import contextlib,copy,hashlib,importlib.util,json,os,sys,tempfile,time,unittest
from unittest import mock
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('formal',HERE/'run_recovery.py');formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal)

class Tests(unittest.TestCase):
 def test_real_frozen_metadata_and_mutations(self):
  file=HERE.parent.parent/'round27/independent_review/membership_v1.json'
  self.assertEqual(formal.sha(file),formal.MEMBERS_SHA)
  m=json.loads(file.read_text());formal.validate_order(m)
  for mutation in ('sequence','step','pass','crop','augmentation','member','dev','order'):
   v=copy.deepcopy(m)
   if mutation=='sequence':v['train_order'][1]['sequence']='wrong_sequence'
   if mutation=='step':v['train_order'][1]['step']=1
   if mutation=='pass':v['train_order'][1]['pass_index']=2
   if mutation=='crop':v['train_order'][1]['crop']['height']=480
   if mutation=='augmentation':v['train_order'][1]['augmentation']='flip'
   if mutation=='member':v['train_order'][1]['target_id']=v['train_order'][0]['target_id']
   if mutation=='dev':v['dev_members'][0]=v['train_members'][0]
   if mutation=='order':
    a,b=v['train_order'][256],v['train_order'][257]
    for key in ('target_id','sequence'):a[key],b[key]=b[key],a[key]
   with self.assertRaises(ValueError,msg=mutation):formal.validate_order(v)

 def test_event_only_loading_and_hash_binding(self):
  import types
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);file=root/'events.npy';np.save(file,np.zeros((10,480,640),np.float32))
   evidence={'events':{'relative_path':'events.npy','sha256':formal.sha(file)},
    'gt':{'relative_path':'MUST_NOT_OPEN_GT','sha256':'x'},'mask':{'relative_path':'MUST_NOT_OPEN_MASK','sha256':'x'}}
   accesses=[]
   def checked(base,relative):accesses.append(relative);return base/relative
   p={'context':(types.SimpleNamespace(checked_path=checked),),'b':types.SimpleNamespace(data_root=root)}
   result=formal.event_only(p,{'target_id':'synthetic'},evidence,np)
   self.assertEqual(result.shape,(10,480,640));self.assertEqual(accesses,['events.npy'])
   bad=copy.deepcopy(evidence);bad['events']['sha256']='0'*64
   with self.assertRaises(ValueError):formal.event_only(p,{},bad,np)
   bad=evidence.copy();np.save(file,np.zeros((10,3,4),np.float32));bad['events']={'relative_path':'events.npy','sha256':formal.sha(file)}
   with self.assertRaises(ValueError):formal.event_only(p,{},bad,np)

 def test_actual_hook_geometry_and_finally_cleanup(self):
  import types
  class Tiny(torch.nn.Module):
   def __init__(self):
    super().__init__();self.affines=torch.nn.ModuleList([torch.nn.Linear(8,8,bias=False) for _ in range(28)])
    self.bn=torch.nn.ModuleList([torch.nn.BatchNorm1d(8,track_running_stats=False) for _ in range(78)])
   def forward(self,x):
    for module in self.affines:x=module(x)
    for module in self.bn:x=module(x)
    return x
  tiny=Tiny().eval();core=types.SimpleNamespace(TARGETS=tuple('affines.'+str(i) for i in range(28)))
  with formal.observed(tiny,core,torch) as (bn,affine):result=tiny(torch.arange(80,dtype=torch.float32).reshape(10,8))
  self.assertEqual(sum(bn.values()),78);self.assertEqual(sum(len(x) for x in affine.values()),28)
  self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in tiny.modules()))
  with self.assertRaisesRegex(RuntimeError,'synthetic interrupt'):
   with formal.observed(tiny,core,torch):raise RuntimeError('synthetic interrupt')
  self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in tiny.modules()))
  with self.assertRaises(ValueError):
   with formal.observed(tiny,core,torch):pass
  self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in tiny.modules()))

 def test_gradient_ledger_and_work_preservation(self):
  params={'used':torch.nn.Parameter(torch.tensor([2.,3.])), 'unused':torch.nn.Parameter(torch.tensor([5.]))}
  (params['used']**2).sum().backward();names,none,norms=formal.grad_stats(params,torch)
  self.assertEqual(names,['used']);self.assertEqual(none,['unused']);self.assertAlmostEqual(norms[0],52**.5,places=5)
  params['used'].grad[0]=float('nan')
  with self.assertRaises(ValueError):formal.grad_stats(params,torch)
  work={'fc':{'rows':10,'actual_integer_kernel_calls':8,'logical_row_plane_work':64,'kernel_calls':[{'rows':8}]}}
  short=formal.work_summary(work)
  self.assertNotIn('kernel_calls',short['fc']);self.assertEqual(short['fc']['logical_row_plane_work'],64)
  self.assertEqual(work['fc']['kernel_calls'],[{'rows':8}])
  self.assertEqual(512*2+64*2+2,1154);self.assertEqual(512*2+64*2+3,1155)

 def test_all64_prediction_schema_without_GT_loader(self):
  import types
  class NativeStub(torch.nn.Module):
   def __init__(self):
    super().__init__();self.affines=torch.nn.ModuleList([torch.nn.Linear(8,8,bias=False) for _ in range(28)])
    self.bn=torch.nn.ModuleList([torch.nn.BatchNorm1d(8,track_running_stats=False) for _ in range(78)])
    self.head=torch.zeros(1,2,480,640)
   def forward(self,x):
    for module in self.affines:x=module(x)
    for module in self.bn:x=module(x)
    return {'flow':[self.head]*4}
  @contextlib.contextmanager
  def installed(*args,**kwargs):yield {}
  core=types.SimpleNamespace(TARGETS=tuple('affines.'+str(i) for i in range(28)),installed=installed)
  rows=[{'target_id':'synthetic_%02d'%i,'sequence':'synthetic_sequence'} for i in range(64)]
  records=[dict(target_id=r['target_id'],source_files=[dict(category=k,relative_path=k+'_never_loaded_here',sha256='1'*64) for k in ('input','gt','mask')]) for r in rows]
  m={'dev_members':rows,'records':records};read_events=[]
  def events_only(p,row,evidence,np):read_events.append(row['target_id']);return np.zeros(1,np.float32)
  base=types.SimpleNamespace(bn_signature=lambda *a:'absent',prepare_input=lambda *a:torch.ones(10,8))
  p={'modules':[types.SimpleNamespace(state_digest=lambda model:'same_fixed_state')],'base':base}
  functional=types.SimpleNamespace(reset_net=lambda model:None);model=NativeStub().eval();result={'model_forwards':0}
  with tempfile.TemporaryDirectory() as temp:
   a=types.SimpleNamespace(output=Path(temp),arm='native')
   with mock.patch.object(formal,'event_only',events_only):
    formal.evaluate(a,p,m,None,core,model,functional,'initial',torch,np,result)
   frames=result['evaluations']['initial']['frames'];self.assertEqual(len(frames),64);self.assertEqual(result['model_forwards'],64)
   self.assertEqual(read_events,[r['target_id'] for r in rows]);self.assertFalse(result['evaluations']['initial']['GT_arrays_read'])
   for frame in frames:
    self.assertEqual(set(frame['input_evidence']),{'events','gt','mask'})
    self.assertEqual(set(frame['heads']),{'head0','head1','head2','head3'})
    file=a.output/frame['prediction_file'];self.assertEqual(formal.sha(file),frame['prediction_file_sha256'])
    with np.load(file,allow_pickle=False) as z:self.assertEqual(z['head3'].shape,(2,480,640));self.assertEqual(z['head3'].dtype,np.float32)
   self.assertEqual(3*2*len(frames),384)

if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
 output=Path(sys.argv[1]);assert not output.exists()
 formal.write(output,{'schema':'flow_r28_formal_cpu_v1','status':'PASS' if r.wasSuccessful() else 'FAIL','tests':r.testsRun,
  'failures':len(r.failures),'errors':len(r.errors),'source_sha256':{n:formal.sha(HERE/n) for n in ('run_recovery.py','test_recovery_cpu.py','run_smoke.py','recovery_integer_core.py')},
  'GPU_used':False,'real_GT_read':False,'full_model_forwards':0,'synthetic_hook_network_forwards':65,'wall_seconds':time.monotonic()-start})
 sys.exit(0 if r.wasSuccessful() else 1)
