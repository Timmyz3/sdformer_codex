import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,sys,time,unittest
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path[:0]=[str(REPO/'research/flow_story_20261011/preflight/cpu_numerics/deps'),str(REPO/'SDformer/third_party/SDformerFlow')]
import torch
from spikingjelly.activation_based import functional
from models.STSwinNet_SNN.Spiking_modules import MS_ResBlock
from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp
import recovery_integer_core as c
REPORT=[]

class Tests(unittest.TestCase):
 def test_integer_q6_tail_reference(self):
  torch.manual_seed(27);w=torch.randn(16,8);q,s=c.quantize(w)
  x=torch.randint(0,2,(10*12*14,8)).float();tail=c.tail_rows(10,1,12,14,'cpu')
  self.assertEqual(int((~tail).sum()),810);self.assertEqual(int(tail.sum()),870)
  for mode in ('Q8','Q6'):
   y,_,ints,a=c.integer_affine(x,q,s,tail,mode)
   yr,_,ir,ar=c.integer_affine(x,q,s,tail,mode,reference=True)
   self.assertTrue(torch.equal(ints,ir));self.assertTrue(torch.equal(y.view(torch.uint8),yr.view(torch.uint8)))
   expected=x.to(torch.int64)@q.to(torch.int64).T
   if mode=='Q6':expected[~tail]=x[~tail].to(torch.int64)@(((q.to(torch.int64)>>2)<<2).T)
   self.assertTrue(torch.equal(ints.to(torch.int64),expected))
   self.assertEqual(a['logical_row_plane_work'],len(x)*8 if mode=='Q8' else len(x)*6+int(tail.sum())*2)
 def test_ste_linear_exact_and_gradient_contract(self):
  torch.manual_seed(28);source=torch.nn.Linear(8,16,bias=True);x=torch.randint(0,2,(10,1,12,14,8)).float().requires_grad_()
  for mode in ('Q8','Q6'):
   module=c.STEAffine(source,'linear',mode);source.zero_grad();x.grad=None
   y=module(x);g=torch.randn_like(y);y.backward(g)
   q,s=c.quantize(source.weight);full,low=c.effective_weights(q,s,mode);tail=c.tail_rows(10,1,12,14,'cpu');gf=g.reshape(-1,16)
   expected_dx=torch.empty_like(x.reshape(-1,8));expected_dx[tail]=gf[tail]@full;expected_dx[~tail]=gf[~tail]@low
   self.assertTrue(torch.equal(x.grad.reshape(-1,8),expected_dx))
   self.assertTrue(torch.equal(source.weight.grad,gf.T@x.detach().reshape(-1,8)))
   self.assertTrue(torch.equal(source.bias.grad,gf.sum(0)))
   with torch.no_grad():yd=c.deploy_forward(module,x,reference=False);yr=c.deploy_forward(module,x,reference=True)
   self.assertTrue(torch.equal(y.detach().view(torch.uint8),yd.view(torch.uint8)));self.assertTrue(torch.equal(yd.view(torch.uint8),yr.view(torch.uint8)))
 def test_ste_conv_exact_and_gradient_contract(self):
  torch.manual_seed(29);source=torch.nn.Conv2d(8,16,3,padding=1,bias=True);x=torch.randint(0,2,(10,1,8,12,14)).float().requires_grad_()
  for mode in ('Q8','Q6'):
   source.zero_grad();x.grad=None;module=c.STEAffine(source,'conv',mode);y=module(x);g=torch.randn_like(y);y.backward(g)
   q,s=c.quantize(source.weight);full,low=c.effective_weights(q,s,mode);tail=c.tail_rows(10,1,12,14,'cpu').reshape(10,1,12,14)
   full,low=full.reshape(source.weight.shape),low.reshape(source.weight.shape);gg=g.reshape(10,16,12,14)
   # Independent autograd conv model defines exactly the declared surrogate Jacobian.
   xx=x.detach().reshape(10,8,12,14).requires_grad_();ww=source.weight.detach().clone().requires_grad_()
   xf=torch.nn.functional.conv2d(xx,full,padding=1);xl=torch.nn.functional.conv2d(xx,low,padding=1)
   ((xf*tail+xl*(~tail))*gg).sum().backward();self.assertTrue(torch.allclose(x.grad.reshape_as(xx),xx.grad,atol=2e-5,rtol=2e-5))
   z=torch.nn.functional.conv2d(x.detach().reshape_as(xx),ww,padding=1);(z*gg).sum().backward()
   self.assertTrue(torch.equal(source.weight.grad,ww.grad))
   with torch.no_grad():yd=c.deploy_forward(module,x);yr=c.deploy_forward(module,x,reference=True)
   self.assertTrue(torch.equal(y.detach().contiguous().view(torch.uint8),yd.contiguous().view(torch.uint8)))
   self.assertTrue(torch.equal(yd.contiguous().view(torch.uint8),yr.contiguous().view(torch.uint8)))
 def test_actual_native_MLP_and_residual_binary_operands(self):
  torch.manual_seed(30)
  kw=dict(num_steps=10,spike_norm='BN',neuron_type='psn',v_th=.1,v_reset=None,surrogate_fun='surrogate.ATan()',tau=2.,detach_reset=True)
  mlp=MS_Spiking_Mlp(8,hidden_features=16,norm_layer='BN',drop=0.,**kw)
  block=MS_ResBlock(8,8,**kw)
  for module in (mlp,block):
   functional.set_step_mode(module,'m');module.eval()
   for bn in module.modules():
    if isinstance(bn,torch.nn.modules.batchnorm._BatchNorm):bn.track_running_stats=False;bn.running_mean=bn.running_var=bn.num_batches_tracked=None
  for mode in ('Q8','Q6'):
   specs=[(mlp,'fc1','linear'),(mlp,'fc2','linear'),(block.conv1,'0','conv'),(block.conv2,'0','conv')]
   old=[]
   for parent,name,kind in specs:
    source=getattr(parent,name);old.append((parent,name,source));setattr(parent,name,c.STEAffine(source,kind,mode))
   try:
    u=torch.randn(10,1,12,14,8);v=torch.randn(10,1,8,12,14)
    functional.reset_net(mlp);functional.reset_net(block);out1=mlp(u);out2=block(v)
    loss=out1.square().mean()+out2.square().mean();loss.backward()
    self.assertTrue(torch.isfinite(loss));self.assertTrue(all(getattr(parent,name).weight.grad is not None for parent,name,_ in old))
    for parent,name,_ in old:self.assertTrue(getattr(parent,name).last_audit['binary_verified'])
    for parent,name,_ in old:getattr(parent,name).ste=False
    functional.reset_net(mlp);functional.reset_net(block)
    with torch.no_grad():ref1=mlp(u);ref2=block(v)
    self.assertTrue(torch.equal(out1.detach(),ref1));self.assertTrue(torch.equal(out2.detach(),ref2))
   finally:
    for parent,name,source in old:setattr(parent,name,source)
  REPORT.append(dict(real_original_MLP=True,real_original_MS_ResBlock=True,local_module_forwards=8,all_binary_operands=True,integer_deploy_equal=True,actual_backward=True,model_forwards=0))
 def test_invalid_inputs_and_scope(self):
  self.assertEqual(len(c.LINEARS),24);self.assertEqual(len(c.CONVS),4);self.assertEqual(len(set(c.TARGETS)),28)
  q,s=c.quantize(torch.randn(16,8));x=torch.zeros(1680,8);x[0,0]=.5
  with self.assertRaises(ValueError):c.integer_affine(x,q,s,c.tail_rows(10,1,12,14,'cpu'),'Q6')
  with self.assertRaises(ValueError):c.tail_rows(9,1,12,14,'cpu')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();assert not a.output.exists()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);start=time.monotonic()
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 d=dict(schema='flow_r27_integer_ste_cpu_v1',status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),records=REPORT,
 source_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('recovery_integer_core.py','test_integer_core_cpu.py')},
 GPU_used=False,new_model_forwards=0,GT_read=False,wall_seconds=time.monotonic()-start)
 a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d));assert not torch.cuda.is_initialized()
 raise SystemExit(0 if r.wasSuccessful() else 1)
