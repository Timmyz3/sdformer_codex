import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,contextlib,hashlib,json,sys,time,types,unittest
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO/'SDformer/third_party/SDformerFlow'))
import numpy as np
import torch
import run_smoke as s
from loss.flow_supervised import flow_loss_supervised

class Tests(unittest.TestCase):
 def test_crop_source_evidence_and_pixel_units(self):
  events=np.broadcast_to(np.arange(480,dtype=np.float32)[None,:,None],(10,480,640)).copy()
  gt=np.stack(np.meshgrid(np.arange(640,dtype=np.float32),np.arange(480,dtype=np.float32)),0)
  mask=np.ones((480,640),dtype=bool);seen=[]
  def arrays(root,row,evidence,expected_hashes):
   self.assertEqual(expected_hashes,{'events':'input_sha','gt':'gt_sha','mask':'mask_sha'})
   evidence.update(expected_hashes);seen.append(row['target_id'])
   return {'events':events,'gt':gt,'mask':mask},{'events':Path('/unserializable-path-if-misused')}
  native=types.SimpleNamespace(target_files=lambda row:{'input':'input','gt':'gt','mask':'mask'})
  metrics=types.SimpleNamespace(load_arrays=arrays)
  base=types.SimpleNamespace(prepare_input=lambda x,*_:torch.from_numpy(x.copy()))
  p={'context':(native,None,metrics,None,{k:{'sha256':k+'_sha'} for k in ('input','gt','mask')}),
     'b':types.SimpleNamespace(data_root=Path('/synthetic-only')),'base':base}
  chunk,label,valid,full,evidence=s.training_arrays(p,{'target_id':'fixedtrain'},{'top':118,'left':128},torch,'cpu')
  self.assertEqual(chunk.shape,(10,288,384));self.assertEqual(label.shape,(1,2,288,384));self.assertEqual(valid.shape,(1,1,288,384))
  self.assertEqual(label[0,:,0,0].tolist(),[128.,118.]);self.assertEqual(label[0,:,-1,-1].tolist(),[511.,405.])
  self.assertEqual(evidence,{'events':'input_sha','gt':'gt_sha','mask':'mask_sha'});json.dumps(evidence)
  self.assertEqual(seen,['fixedtrain'])
 def test_loss_teacher_scale1_and_native_zero(self):
  config={'metrics':{'flow_scaling':1},'loss':{'lambda_mod':1,'lambda_ang':0}}
  loss=flow_loss_supervised(config,'cpu');gt=torch.tensor([[[[3.,6.]],[[4.,8.]]]])
  mask=torch.ones(1,1,1,2);pred=[torch.zeros_like(gt,requires_grad=True) for _ in range(4)]
  value=loss(pred,gt,mask,gamma=None)
  self.assertAlmostEqual(float(value),7.5,places=5)
  teacher=[v.detach().clone() for v in pred]
  kd=sum((x-y).abs().mean() for x,y in zip(pred,teacher))/4
  self.assertEqual(float(kd),0.)
  total=value+.1*kd;total.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in pred))
  self.assertEqual(s.RECIPE['all_parameters'],477);self.assertEqual(s.RECIPE['flow_scaling'],1.)
  self.assertEqual(s.RECIPE['dropout_mode'],'model.eval()')
 def test_exact_full_compare_rejects_signedzero_integer_and_missing(self):
  a={'integer::fc':torch.zeros(2,dtype=torch.int32),'PSN::one':torch.zeros(2),'head0':torch.zeros(2)}
  b={k:v.clone() for k,v in a.items()};self.assertTrue(s.compare_full(a,b,torch)['pass'])
  b['head0'][0]=-0.;self.assertFalse(s.compare_full(a,b,torch)['pass']);b['head0'][0]=0.
  b['integer::fc'][0]=1;self.assertFalse(s.compare_full(a,b,torch)['pass']);b['integer::fc'][0]=0
  b.pop('PSN::one')
  with self.assertRaises(ValueError):s.compare_full(a,b,torch)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();assert not a.output.exists()
 torch.set_num_threads(2);start=time.monotonic();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 d={'schema':'flow_r28_smoke_cpu_v1','status':'PASS' if r.wasSuccessful() else 'FAIL','tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),
 'source_sha256':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('run_smoke.py','test_smoke_cpu.py','recovery_integer_core.py')},
 'GPU_used':False,'real_GT_read':False,'real_model_forwards':0,'wall_seconds':time.monotonic()-start}
 a.output.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d));raise SystemExit(0 if r.wasSuccessful() else 1)
