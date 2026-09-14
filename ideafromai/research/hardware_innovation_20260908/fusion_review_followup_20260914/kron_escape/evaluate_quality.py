"""Fixed train-only Kronecker hybrid, complete R8 path and exact I24 consumer."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
from model_access import HERE,BASE,load_parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['diverse','valid'],default='diverse');o=ap.parse_args()
 import torch,cupy
 import torch.nn.functional as F
 f=np.load(HERE/'fitted.npz');c=np.load(HERE/'consumer_coefficients.npz')
 q1=torch.tensor(f['q1'].reshape(8,96,3,3),device='cuda',dtype=torch.float64)
 q2=torch.tensor(f['W_hybrid'].reshape(96,8,1,1),device='cuda',dtype=torch.float64)
 scale=torch.tensor(f['output_scale'][None,:,None,None],device='cuda',dtype=torch.float64)
 a=torch.tensor(c['a_q40'].astype(np.int64)[None,:,None,None],device='cuda');b=torch.tensor(c['b_q20'].astype(np.int64)[None,:,None,None],device='cuda')
 args,net=load_parent(HERE/('quality_work_'+o.split));args.split=o.split
 from run_bn_probe import read_names
 from evaluate_branch_control import evaluate_axis
 names=read_names(args.data,'valid') if o.split=='valid' else json.loads((BASE/'algorithm/samples.json').read_text())['valid'][:10]
 r0name=net.BLOCK.rsplit('.',1)[0]+'.0';r0=net.modules[r0name];conv=net.modules[r0name+'.conv2.0'];bn=net.modules[r0name+'.norm2.norm_layer'];assert not bn.training and bn.track_running_stats
 rows=[];state={};original=conv.forward
 def flat(x):return x[:,0] if x.ndim==5 else x
 def pre(m,inputs):state['id']=flat(inputs[0]).detach()
 def forward(x):
  g=flat(x);assert torch.equal(g,g.ne(0).to(g))
  z=F.conv2d(g.double(),q1,padding=1);p=F.conv2d(z,q2)
  assert torch.equal(z,z.round()) and z.abs().max()<=2592
  assert torch.equal(p,p.round()) and p.abs().max()<2**31
  state['p']=p.to(torch.int64);state['zmax']=int(z.abs().max())
  y=(p*scale).to(x.dtype);return y[:,None] if x.ndim==5 else y
 def post(m,inputs,output):
  identity=state.pop('id');p=state.pop('p');jr=(identity.double()*2**20).round();j=jr.clamp(-2**31,2**31-1).to(torch.int64)
  wide=p*a+((j+b)<<20);q=torch.div(wide,1<<26,rounding_mode='floor');r=wide-q*(1<<26)
  v=q+((r>1<<25)|((r==1<<25)&((q&1)!=0))).to(torch.int64);i=v.clamp(-2**23,2**23-1)
  rows.append(dict(index=len(rows),z_abs_max=state.pop('zmax'),p_abs_max=int(p.abs().max()),identity_saturations=int((jr!=j).sum()),I24_saturations=int((v!=i).sum())))
  state['i']=i;return (i.float()/16384)[:,None]
 def reader(m,inputs,out):assert torch.equal(net.helper.i,state.pop('i').double())
 hooks=[r0.register_forward_pre_hook(pre),r0.register_forward_hook(post),net.helper.source.register_forward_hook(reader)];conv.forward=forward
 path=HERE/f'quality_{o.split}.json';start=time.time()
 report=dict(complete=False,split=o.split,frames=names,python=sys.version,torch=torch.__version__,cupy=cupy.__version__,gpu=torch.cuda.get_device_name(0),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,calibration_frame='thun_00_a_0002.npy',calibration_windows=336,calibration_overlap=False,function='Q1 then hybrid Wh (10 Kronecker output groups + 2 exact groups) then exact signed64 I24; every T identity evaluated',fullnet_bittrue=False,per_frame_integer_checks=rows)
 path.write_text(json.dumps(report,indent=2)+'\n')
 try:
  result=evaluate_axis(args,net.model,net.current,names,'kron_escape',progress_tag='KRON_'+o.split)
  gate=1.447936665574317 if o.split=='valid' else 1.4699681144337489
  report.update(complete=True,result=result,elapsed_seconds=time.time()-start,NB0_same_set=gate,below_same_set_NB0=result['AEE_frame_mean']<gate)
  print('QUALITY_COMPLETE',result['AEE_frame_mean'],flush=True)
 except Exception as e:report['error']=repr(e);raise
 finally:
  path.write_text(json.dumps(report,indent=2)+'\n');conv.forward=original
  for h in hooks:h.remove()
  net.close()
if __name__=='__main__':main()
