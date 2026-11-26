"""Independent actual integer primitive test; original network/data not loaded."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse,hashlib,importlib.util,json,time,unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'implementation/bitplane_linear.py'
INITIAL_SHA=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('_r26_independent_candidate_primitive',SOURCE);K=importlib.util.module_from_spec(s);s.loader.exec_module(K)
MEASUREMENTS=[]

def fixture(bias=False):
 linear=torch.nn.Linear(8,16,bias=bias,dtype=torch.float32)
 base=np.array([-127,-64,-1,0,1,63,126,127],dtype=np.float32)
 w=np.stack([np.roll(base,i%8)*np.float32(2.**((i%5)-2)) for i in range(16)]);w[-1]=0
 with torch.no_grad():
  linear.weight.copy_(torch.from_numpy(w))
  if bias:linear.bias.copy_(torch.arange(16,dtype=torch.float32)/32)
 rng=np.random.default_rng(20261018);x=rng.integers(0,2,size=(10,1,30,40,8),dtype=np.int8).astype(np.float32)
 return linear.eval().requires_grad_(False),w,x

def independent_quant(w):
 scale=np.max(np.abs(w),axis=1).astype(np.float32)/np.float32(127)
 denom=np.where(scale==0,np.float32(1),scale)
 q=np.clip(np.rint(w/denom[:,None]),-127,127).astype(np.int8)
 return q,scale

def expected_integer(x,q,mode):
 flat=x.reshape(-1,8).astype(np.int64);full=np.tile(np.array([y<27 and z<36 for y in range(30) for z in range(40)]),10)
 ref=flat@q.astype(np.int64).T
 if mode in ('Q6','Q4'):
  drop={'Q6':2,'Q4':4}[mode];trunc=np.floor_divide(q.astype(np.int64),2**drop)*(2**drop)
  ref[full]=flat[full]@trunc.T
 return ref.reshape(10,1,30,40,16),full

class Checks(unittest.TestCase):
 def test_source_quantization_zero_ties_and_shape(self):
  linear,w,x=fixture();q,scale=independent_quant(w);spec=K.quantize_source(linear)
  self.assertTrue(np.array_equal(spec['q8'].numpy(),q));self.assertTrue(np.array_equal(spec['scale'].numpy(),scale))
  self.assertEqual(spec['zero_output_channels'],1);self.assertEqual(spec['source_quantization_device'],'cpu');self.assertIsNone(spec['bias'])
  with torch.no_grad():linear.weight[0].copy_(torch.tensor([-127.,-.5,.5,1.5,-1.5,0.,2.5,127.]))
  spec=K.quantize_source(linear);self.assertEqual(spec['q8'][0].tolist(),[-127,0,0,2,-2,0,2,127])
 def test_all_modes_true_integer_perrow_and_observed_calls(self):
  linear,w,x=fixture();q,scale=independent_quant(w);spec=K.place_spec(K.quantize_source(linear),'cpu');native_calls=[]
  handle=linear.register_forward_hook(lambda *args:native_calls.append(1));checks=[]
  try:
   for mode in K.MODES:
    op=K.QuantLinear(linear,spec,mode);calls=[];real=K.integer_mm
    def observed(a,b):
     out=real(a,b);calls.append((tuple(a.shape),tuple(b.shape),str(out.dtype)));return out
    with mock.patch.object(K,'integer_mm',side_effect=observed):out=op(torch.from_numpy(x))
    expected,full=expected_integer(x,q,mode);self.assertTrue(np.array_equal(op.last_integer.numpy(),expected))
    want=expected.astype(np.float32)*scale
    self.assertTrue(np.array_equal(out.numpy(),want));self.assertEqual(len(calls),1 if mode=='Q8ref' else 8)
    audit=op.last_audit;self.assertEqual(audit['bias_applications_per_output_element'],0);self.assertEqual(audit['scale_applications_per_output_element'],1)
    self.assertEqual(audit['input_rows'],12000);self.assertEqual(audit['tail_rows'],2280)
    for j,(actual,rec) in enumerate(zip(calls,audit['int_mm_calls'])):
     drop={'Q8ref':0,'Q8serial':0,'Q6':2,'Q4':4}[mode]
     rows=2280 if mode!='Q8ref' and j<drop else 12000
     self.assertEqual(actual,((rows,8),(8,16),'torch.int32'));self.assertEqual(rec['rows'],rows)
     if mode!='Q8ref':self.assertEqual(rec['coefficient'],-128 if j==7 else 1<<j)
     source=x.reshape(-1,8)[~full] if rows==2280 else x.reshape(-1,8)
     self.assertEqual(rec['input_spike_ones'],int(source.sum(dtype=np.int64)))
    checks.append({'mode':mode,'observed_integer_mm_calls':len(calls),'actual_row_sizes':[v[0][0] for v in calls],'integer_reference_exact':True,'scaled_FP32_exact':True,'bias_adds':0})
   self.assertFalse(native_calls)
  finally:handle.remove()
  MEASUREMENTS.append({'primitive_full_grid_calls':4,'original_network_forwards':0,'actual_CPU_integer_operator_calls':25,'modes':checks})
 def test_signed_extremes_and_no_posttruncate_reclamp(self):
  q=np.array([-127,-1,0,1,127],dtype=np.int64)
  for drop in (0,2,4):
   expected=(q>>drop)<<drop
   actual=sum(((q>>bit)&1)*(-128 if bit==7 else 1<<bit) for bit in range(drop,8))
   self.assertTrue(np.array_equal(expected,actual))
  self.assertEqual(int((q[0]>>2)<<2),-128);self.assertEqual(int((q[1]>>4)<<4),-16)
 def test_nonbinary_nonfinite_and_bad_geometry_rejected_before_mm(self):
  linear,w,x=fixture();op=K.QuantLinear(linear,K.place_spec(K.quantize_source(linear),'cpu'),'Q6')
  with mock.patch.object(K,'integer_mm',side_effect=AssertionError('Must reject before integer contraction')):
   for v in (-1.,.5,2.,np.nan,np.inf):
    bad=x.copy();bad[0,0,0,0,0]=v
    with self.assertRaises(ValueError):op(torch.from_numpy(bad))
   with self.assertRaises(ValueError):op(torch.from_numpy(x[:2]))
 def test_bias_once_and_zero_channel_exact(self):
  linear,w,x=fixture(True);q,scale=independent_quant(w);op=K.QuantLinear(linear,K.place_spec(K.quantize_source(linear),'cpu'),'Q8ref')
  out=op(torch.from_numpy(x));integer,_=expected_integer(x,q,'Q8ref');want=integer.astype(np.float32)*scale+linear.bias.numpy()
  self.assertTrue(np.array_equal(out.numpy(),want));self.assertEqual(op.last_audit['bias_applications_per_output_element'],1)
  self.assertTrue(np.all(out.numpy()[...,-1]==linear.bias.numpy()[-1]));MEASUREMENTS.append({'synthetic_biased_primitive_calls':1,'actual_CPU_integer_operator_calls':1,'production_source_bias':False})
 def test_repeat_exact_and_model_source_unchanged(self):
  linear,w,x=fixture();before={k:v.detach().numpy().tobytes() for k,v in linear.state_dict().items()};spec=K.place_spec(K.quantize_source(linear),'cpu');outputs=[]
  for _ in range(2):
   op=K.QuantLinear(linear,spec,'Q6');out=op(torch.from_numpy(x));outputs.append((out.numpy().tobytes(),op.last_integer.numpy().tobytes()))
  self.assertEqual(outputs[0],outputs[1]);self.assertEqual(before,{k:v.detach().numpy().tobytes() for k,v in linear.state_dict().items()})
  self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(),INITIAL_SHA)
  MEASUREMENTS.append({'Q6_repeat_calls':2,'actual_CPU_integer_operator_calls':16,'bytewise_exact':True})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 torch.set_num_threads(2);torch.set_num_interop_threads(1);start=time.monotonic();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 out={'schema':'flow_r26_independent_primitive_CPU_v1','status':'PASS' if r.wasSuccessful() else 'FAIL','source_sha256':{str(SOURCE):INITIAL_SHA,str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'measurements':MEASUREMENTS,'wall_seconds':time.monotonic()-start,'GPU_used':False,'GT_read':False,'original_network_forwards':0,'limits':['CPU true int8/int32 operator with full T10/B1/30x40 and synthetic Cin8/Cout16; not full production-width model or CUDA proof.','Perrow NumPy int64 reference is independently constructed; runtime counter rows cross-checked against actually called integer operator shapes.','Original model actual target FC bias is absent; separate synthetic bias test ensures one-time bias handling.']}
 assert not torch.cuda.is_initialized();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));raise SystemExit(not r.wasSuccessful())
