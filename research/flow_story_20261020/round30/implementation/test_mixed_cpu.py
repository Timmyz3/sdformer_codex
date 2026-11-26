#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,collections,hashlib,importlib.util,json,sys,time,unittest
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path[:0]=[str(REPO/'research/flow_story_20261011/preflight/cpu_numerics/deps'),str(REPO/'SDformer/third_party/SDformerFlow')]
import torch
from torch.nn import functional as F
import mixed_integer_core as c
import mask_contract as mc
import run_joint_recovery as run
REPORT=[]
def selected(h=12,w=19):
 x=torch.zeros(h,w,dtype=torch.bool);x[:9,:9]=True;return x
def original():
 p=REPO/'research/flow_story_20261019/round28/implementation/recovery_integer_core.py'
 assert hashlib.sha256(p.read_bytes()).hexdigest()=='f4e4dae18a7e59fab286f2e59f23f72d20583a2b8de22dee4c78204bbfd2fe9a'
 s=importlib.util.spec_from_file_location('_r30_cpu_original_core',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class Tests(unittest.TestCase):
 def test_01_real_member_schedule_and_reference_metadata(self):
  p=REPO/'research/flow_story_20261019/round27/independent_review/membership_v1.json'
  self.assertEqual(run.sha(p),mc.MEMBERS_SHA);m=run.read(p)
  counts=collections.Counter();seqpass=collections.defaultdict(collections.Counter);ids=collections.defaultdict(list)
  for row in m['train_order']:
   mode=mc.mode_for(row,m['sequences']);counts[mode]+=1;seqpass[(row['sequence'],row['pass_index'])][mode]+=1;ids[row['target_id']].append(mode)
  self.assertEqual(counts,{'Q4':256,'Q8':128,'M':128});self.assertTrue(all(x=={'Q4':16,'Q8':8,'M':8} for x in seqpass.values()))
  self.assertTrue(all(len(v)==2 and v.count('Q4')==1 for v in ids.values()))
  refs=run.initial_references(HERE.parent/'INITIAL_REFERENCES_LOCAL_V1.json',m)
  self.assertEqual(set(refs),{'initial_Q4','initial_Q8'});self.assertTrue(all(v['new_forwards']==0 for v in refs.values()))

 def test_02_tail_mask_and_contract_identity(self):
  tail,high,a=c.row_masks(10,2,12,19,'cpu','M',selected())
  self.assertEqual(int(tail.sum()),(12*19-2*81)*20);self.assertEqual(int((high&~tail).sum()),81*20)
  self.assertEqual(a['selected_complete_tiles'],[[0,0]])
  for invalid in (selected().float(),torch.ones(12,19,dtype=torch.bool)):
   with self.assertRaises(ValueError):c.row_masks(10,1,12,19,'cpu','M',invalid)
  fragmented=selected();fragmented[0,0]=False
  with self.assertRaises(ValueError):c.row_masks(10,1,12,19,'cpu','M',fragmented)
  with self.assertRaises(ValueError):c.row_masks(10,1,12,19,'cpu','M',None)
  with self.assertRaises(ValueError):c.row_masks(10,1,12,19,'cpu','Q4',selected())
  contract=dict(schema='flow_r30_mask_contract_v1',seed=20261019,tile_size=9,tail_mode='Q8',members_sha256=mc.MEMBERS_SHA,
   training_selection='sha256_seed_step_group_row_col_topk_v1',geometries={},evaluation_tiles={})
  for geo in ('full9','crop_center9'):
   contract['geometries'][geo]={g:dict(H=h,W=w,Q8_complete_tiles=0) for g,(h,w) in mc.expected_shapes(geo).items()}
  contract['evaluation_tiles']={g:[] for g in c.GROUPS};mc.validate(contract)
  mapped,receipt=mc.materialize(contract,'crop_center9',step=1)
  self.assertTrue(all(not bool(v.any()) for v in mapped.values()));self.assertEqual(len(mapped),16)
  self.assertEqual(c.group_name(c.LINEARS[0]),c.group_name(c.LINEARS[1]))
  group=c.GROUPS[0];contract['geometries']['crop_center9'][group]['Q8_complete_tiles']=1
  a,_=mc.materialize(contract,'crop_center9',step=1);b,_=mc.materialize(contract,'crop_center9',step=1)
  self.assertTrue(torch.equal(a[group],b[group]));self.assertEqual(int(a[group].sum()),81)

 def test_03_integer_signed_planes_reference_and_old_uniform(self):
  old=original();torch.manual_seed(30);x=torch.randint(0,2,(10*12*19,8)).float()
  q=torch.tensor([[-127,-1,0,1,15,16,127,64]],dtype=torch.int8).repeat(16,1);scale=torch.linspace(.01,.16,16)
  for mode in ('Q4','Q8','M'):
   tail,high,_=c.row_masks(10,1,12,19,'cpu',mode,selected() if mode=='M' else None)
   y,_,integer,a=c.integer_affine(x,q,scale,tail,mode,high8=high)
   yr,_,ir,_=c.integer_affine(x,q,scale,tail,mode,high8=high,reference=True)
   expected=x.to(torch.int64)@q.to(torch.int64).T
   expected[~high]=x[~high].to(torch.int64)@(((q.to(torch.int64)>>4)<<4).T)
   self.assertTrue(torch.equal(integer.to(torch.int64),expected));self.assertTrue(torch.equal(ir,integer));self.assertTrue(torch.equal(y.view(torch.uint8),yr.view(torch.uint8)))
   self.assertEqual(a['logical_row_plane_work'],len(x)*4+int(high.sum())*4)
   self.assertEqual(sum(z['rows'] for z in a['kernel_calls'] if z['bit']==0),int(high.sum()))
   self.assertTrue(all(z['coefficient']==-128 for z in a['kernel_calls'] if z['bit']==7))
   if mode!='M':
    yo,_,io,_=old.integer_affine(x,q,scale,tail,mode)
    self.assertTrue(torch.equal(io,integer));self.assertTrue(torch.equal(yo.view(torch.uint8),y.view(torch.uint8)))
  bad=x.clone();bad[0,0]=.5
  with self.assertRaises(ValueError):c.integer_affine(bad,q,scale,tail,'M',high8=high)

 def test_04_actual_linear_ste_jacobian_and_bias_once(self):
  torch.manual_seed(31);source=torch.nn.Linear(8,16,bias=True);x=torch.randint(0,2,(10,1,12,19,8)).float().requires_grad_()
  op=c.STEAffine(source,'linear','M',selected=selected());op.capture_integer=True;y=op(x);g=torch.randn_like(y);y.backward(g)
  q,s=c.quantize(source.weight);full=q.float()*s[:,None];low=((q.to(torch.int16)>>4)<<4).float()*s[:,None]
  high=torch.ones(10,1,12,19,dtype=torch.bool);high[:,:,:9,9:18]=False;hh=high.reshape(-1);gg=g.reshape(-1,16)
  dx=torch.empty_like(x.reshape(-1,8));dx[hh]=gg[hh]@full;dx[~hh]=gg[~hh]@low
  self.assertTrue(torch.equal(x.grad.reshape_as(dx),dx));self.assertTrue(torch.equal(source.weight.grad,gg.T@x.detach().reshape(-1,8)))
  self.assertTrue(torch.equal(source.bias.grad,gg.sum(0)))
  with torch.no_grad():deploy=c.deploy_forward(op,x);dense=c.deploy_forward(op,x,reference=True)
  self.assertTrue(torch.equal(y.detach().view(torch.uint8),deploy.view(torch.uint8)));self.assertTrue(torch.equal(deploy.view(torch.uint8),dense.view(torch.uint8)))

 def test_05_actual_conv_ste_halo_jacobian(self):
  torch.manual_seed(32);source=torch.nn.Conv2d(8,16,3,padding=1,bias=True);x=torch.randint(0,2,(10,1,8,12,19)).float().requires_grad_()
  op=c.STEAffine(source,'conv','M',selected=selected());y=op(x);g=torch.randn_like(y);y.backward(g)
  q,s=c.quantize(source.weight);full=(q.float()*s[:,None]).reshape_as(source.weight);low=(((q.to(torch.int16)>>4)<<4).float()*s[:,None]).reshape_as(source.weight)
  high=torch.ones(10,1,12,19,dtype=torch.bool);high[:,:,:9,9:18]=False;xx=x.detach().reshape(10,8,12,19).requires_grad_();gg=g.reshape(10,16,12,19)
  z=F.conv2d(xx,full,padding=1)*high+F.conv2d(xx,low,padding=1)*(~high);(z*gg).sum().backward()
  self.assertTrue(torch.allclose(x.grad.reshape_as(xx),xx.grad,atol=2e-5,rtol=2e-5))
  ww=source.weight.detach().clone().requires_grad_();(F.conv2d(x.detach().reshape_as(xx),ww,padding=1)*gg).sum().backward()
  self.assertTrue(torch.equal(source.weight.grad,ww.grad));self.assertTrue(torch.equal(source.bias.grad,gg.sum((0,2,3))))
  with torch.no_grad():deploy=c.deploy_forward(op,x);dense=c.deploy_forward(op,x,reference=True)
  self.assertTrue(torch.equal(y.detach().contiguous().view(torch.uint8),deploy.contiguous().view(torch.uint8)));self.assertTrue(torch.equal(deploy.contiguous().view(torch.uint8),dense.contiguous().view(torch.uint8)))

 def test_06_real_native_MLP_residual_currentBN_mixed(self):
  from spikingjelly.activation_based import functional
  from models.STSwinNet_SNN.Spiking_modules import MS_ResBlock
  from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp
  torch.manual_seed(33);kw=dict(num_steps=10,spike_norm='BN',neuron_type='psn',v_th=.1,v_reset=None,surrogate_fun='surrogate.ATan()',tau=2.,detach_reset=True)
  mlp=MS_Spiking_Mlp(8,hidden_features=16,norm_layer='BN',drop=0.,**kw);block=MS_ResBlock(8,8,**kw)
  for module in (mlp,block):
   functional.set_step_mode(module,'m');module.eval()
   for bn in module.modules():
    if isinstance(bn,torch.nn.modules.batchnorm._BatchNorm):bn.track_running_stats=False;bn.running_mean=bn.running_var=bn.num_batches_tracked=None
  specs=[(mlp,'fc1','linear'),(mlp,'fc2','linear'),(block.conv1,'0','conv'),(block.conv2,'0','conv')];saved=[];shared=selected()
  for parent,name,kind in specs:
   old=getattr(parent,name);saved.append((parent,name,old));setattr(parent,name,c.STEAffine(old,kind,'M',selected=shared))
  try:
   u=torch.randn(10,1,12,19,8);v=torch.randn(10,1,8,12,19);outputs=[]
   for ste,reference in ((True,False),(False,False),(False,True)):
    for parent,name,_ in saved:getattr(parent,name).ste=ste;getattr(parent,name).reference=reference
    functional.reset_net(mlp);functional.reset_net(block)
    if ste:
     a,b=mlp(u),block(v);(a.square().mean()+b.square().mean()).backward()
     self.assertTrue(all(getattr(p,n).weight.grad is not None for p,n,_ in saved))
    else:
     with torch.no_grad():a,b=mlp(u),block(v)
    self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all());outputs.append((a.detach(),b.detach()))
    self.assertTrue(all(getattr(p,n).last_audit['binary_verified'] for p,n,_ in saved))
   self.assertTrue(all(torch.equal(x,y) for z in outputs[1:] for x,y in zip(z,outputs[0])))
   self.assertIs(mlp.fc1.selected,mlp.fc2.selected)
   self.assertTrue(all(bn.running_mean is None for mod in (mlp,block) for bn in mod.modules() if isinstance(bn,torch.nn.modules.batchnorm._BatchNorm)))
  finally:
   for p,n,old in saved:setattr(p,n,old)
  REPORT.append(dict(real_original_MLP=True,real_original_MS_ResBlock=True,micro_forwards=6,actual_backward=True,GT_read=False,model_forwards=0))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();assert not a.output.exists()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);start=time.monotonic()
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 receipt=dict(schema='flow_r30_mixed_cpu_v1',status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),records=REPORT,
  source_sha256={n:run.sha(HERE/n) for n in ('mixed_integer_core.py','mask_contract.py','run_joint_recovery.py','test_mixed_cpu.py')},GPU_used=False,GT_read=False,model_forwards=0,wall_seconds=time.monotonic()-start)
 a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));assert not torch.cuda.is_initialized();raise SystemExit(0 if r.wasSuccessful() else 1)
