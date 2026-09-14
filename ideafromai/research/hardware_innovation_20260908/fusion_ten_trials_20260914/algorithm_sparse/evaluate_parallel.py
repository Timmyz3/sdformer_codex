"""Frozen train-calibrated interfaces, FP64 integer dots and exact signed64 consumer."""
import argparse,json,sys,time
import numpy as np
from model_access import HERE,BASE,load_parent
from reference import MODES,transform

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['diverse','valid'],default='diverse');ap.add_argument('--modes',default='0,1,2,3,4,5,6');o=ap.parse_args()
 import torch,cupy
 import torch.nn.functional as F
 f=np.load(HERE/'factors.npz');c=np.load(HERE/'consumer_coefficients.npz');par=json.loads((HERE/'parameters.json').read_text())
 q1=torch.tensor(f['q1'].reshape(8,96,3,3),device='cuda',dtype=torch.float64);q2=torch.tensor(f['q2'].reshape(96,8,1,1),device='cuda',dtype=torch.float64)
 scale=torch.tensor(f['output_scale'][None,:,None,None],device='cuda',dtype=torch.float64)
 a=torch.tensor(c['a_q40'].astype(np.int64)[None,:,None,None],device='cuda');b=torch.tensor(c['b_q20'].astype(np.int64)[None,:,None,None],device='cuda')
 sh=torch.tensor(par['shifts'],device='cuda');cb=torch.tensor(par['codebook'],device='cuda');rt=torch.tensor(par['rank_tau'],device='cuda');trt=torch.tensor(par['temporal_rank_tau'],device='cuda')
 def trans(z,mode):
  # [H*W,1,T,R], no cross-spatial or cross-frame retained state.
  x=z.permute(2,3,0,1).reshape(-1,1,10,8).to(torch.int64);out=x.clone();v=x.reshape(-1,8)
  if mode==1:out[(x.abs()<<sh).sum(-1)<=par['group_tau']*x.count_nonzero(-1)]=0
  elif mode==2:out[x.abs()<=rt]=0
  elif mode in [3,4]:
   if mode==3:
    distances=((v[:,None,:]-cb[None]).abs()<<sh).sum(-1);ix=distances.argmin(-1);base=cb[ix].clone()
   else:base=torch.zeros_like(v)
   e=v-base;rr=(e.abs()<<sh).argmax(-1);ind=torch.arange(len(v),device='cuda');base[ind,rr]=v[ind,rr];out=base.reshape(x.shape)
  elif mode in [5,6]:
   for t in range(1,10):
    prev=out[:,:,t-1];diff=x[:,:,t]-prev
    hold=((diff.abs()<<sh).sum(-1)<=par['temporal_tau'])[...,None] if mode==5 else diff.abs()<=trt
    out[:,:,t]=torch.where(hold,prev,x[:,:,t])
  return out.reshape(z.shape[2],z.shape[3],10,8).permute(2,3,0,1).contiguous()
 args,net=load_parent(HERE/('quality_work_parallel_'+o.split));args.split=o.split
 from run_bn_probe import read_names
 from evaluate_branch_control import evaluate_axis
 names=read_names(args.data,'valid') if o.split=='valid' else json.loads((BASE/'algorithm/samples.json').read_text())['valid'][:10]
 r0name=net.BLOCK.rsplit('.',1)[0]+'.0';r0=net.modules[r0name];conv=net.modules[r0name+'.conv2.0'];bn=net.modules[r0name+'.norm2.norm_layer'];assert not bn.training and bn.track_running_stats
 state={};rows=[];mode=0;original=conv.forward
 def flat(x):return x[:,0] if x.ndim==5 else x
 def pre(m,inputs):state['id']=flat(inputs[0]).detach()
 def forward(x):
  g=flat(x);assert torch.equal(g,g.ne(0).to(g));z=F.conv2d(g.double(),q1,padding=1);assert torch.equal(z,z.round()) and z.abs().max()<=2592
  zh=trans(z,mode);p=F.conv2d(zh.double(),q2);assert torch.equal(p,p.round()) and p.abs().max()<=679477248
  if not rows:
   # CPU implementation on fixed grid cross-checks GPU integer transformation.
   zz=z[:,:,::16,::16].permute(2,3,0,1).reshape(-1,1,10,8).cpu().numpy().astype(np.int64)
   gold,_=transform(zz,mode,par);got=zh[:,:,::16,::16].permute(2,3,0,1).reshape(-1,1,10,8).cpu().numpy();assert np.array_equal(gold,got)
  state['p']=p.to(torch.int64);state['zmax']=int(zh.abs().max());y=(p*scale).to(x.dtype);return y[:,None] if x.ndim==5 else y
 def post(m,inputs,output):
  identity=state.pop('id');p=state.pop('p');jr=(identity.double()*2**20).round();j=jr.clamp(-2**31,2**31-1).to(torch.int64)
  wide=p*a+((j+b)<<20);q=torch.div(wide,1<<26,rounding_mode='floor');r=wide-q*(1<<26)
  v=q+((r>1<<25)|((r==1<<25)&((q&1)!=0))).to(torch.int64);i=v.clamp(-2**23,2**23-1)
  rows.append(dict(index=len(rows),z_abs_max=state.pop('zmax'),p_abs_max=int(p.abs().max()),identity_saturations=int((jr!=j).sum()),I24_saturations=int((v!=i).sum())))
  state['i']=i;return (i.float()/16384)[:,None]
 def reader(m,inputs,out):assert torch.equal(net.helper.i,state.pop('i').double())
 hooks=[r0.register_forward_pre_hook(pre),r0.register_forward_hook(post),net.helper.source.register_forward_hook(reader)];conv.forward=forward
 try:
  for mode in map(int,o.modes.split(',')):
   rows.clear();start=time.time();path=HERE/f'quality_{o.split}_{mode}_parallel.json'
   report=dict(complete=False,mode=mode,name=MODES[mode],split=o.split,frames=names,python=sys.version,torch=torch.__version__,cupy=cupy.__version__,gpu=torch.cuda.get_device_name(0),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,parameters='parameters.json',calibration_frame=par['training_frame'],calibration_overlap=False,function='completeQ1 then frozen latent transform then Q2 then exact signed64 I24; everyT identity evaluated',fullnet_bittrue=False,per_frame_integer_checks=rows)
   path.write_text(json.dumps(report,indent=2)+'\n')
   try:
    result=evaluate_axis(args,net.model,net.current,names,MODES[mode],progress_tag=f'AS{mode}_{o.split}')
    gate=1.447936665574317 if o.split=='valid' else 1.4699681144337489
    report.update(complete=True,result=result,elapsed_seconds=time.time()-start,NB0_same_set=gate,below_same_set_NB0=result['AEE_frame_mean']<gate,historical_diverse_gate=1.45460286107)
   except Exception as e:report['error']=repr(e);raise
   finally:path.write_text(json.dumps(report,indent=2)+'\n')
   print('ARM_COMPLETE',mode,result['AEE_frame_mean'],time.time()-start,flush=True)
 finally:
  conv.forward=original
  for h in hooks:h.remove()
  net.close()
if __name__=='__main__':main()
