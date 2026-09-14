"""Exact integer definitions of the three frozen latent interfaces."""
import numpy as np
MODES={0:'exact',1:'group_drop',2:'rank_drop',3:'prototype_one_residual',4:'zero_prototype_one_rank',5:'temporal_group_hold',6:'temporal_rank_deadband'}

def latent(source,q1):
 s=source.astype(np.int64)
 patches=np.stack([s[:,:,:,y:y+3,x:x+3].reshape(len(s),10,864) for y in range(2) for x in range(2)],axis=1)
 return patches@q1.astype(np.int64).T

def consumer(p,J,a,b):
 v=p.astype(np.int64)*a+((J.astype(np.int64)+b)<<20)
 q=v//2**26;r=v-q*2**26;q+=((r>2**25)|((r==2**25)&((q&1)!=0)))
 return np.clip(q,-2**23,2**23-1).astype(np.int32)

def transform(z,mode,params):
 z=np.asarray(z,dtype=np.int64);out=z.copy();shape=z.shape;flat=z.reshape(-1,8);sh=np.asarray(params['shifts'],np.int64)
 refresh=np.ones(shape[:-1],bool);code=np.zeros(shape[:-1],np.int64);rank=np.zeros(shape[:-1],np.int64);res=np.zeros(shape[:-1],np.int64)
 if mode==1:
  score=(np.abs(z)<<sh).sum(-1);nz=np.count_nonzero(z,axis=-1);out[score<=params['group_tau']*nz]=0
 elif mode==2:out[np.abs(z)<=np.asarray(params['rank_tau'])]=0
 elif mode in (3,4):
  if mode==3:
   cb=np.asarray(params['codebook'],np.int64)
   distances=(np.abs(flat[:,None,:]-cb[None])<<sh).sum(-1);ix=distances.argmin(-1);base=cb[ix].copy()
  else:ix=np.zeros(len(flat),np.int64);base=np.zeros_like(flat)
  e=flat-base;rr=(np.abs(e)<<sh).argmax(-1);rv=e[np.arange(len(e)),rr];base[np.arange(len(e)),rr]+=rv
  out=base.reshape(shape);code=ix.reshape(shape[:-1]);rank=rr.reshape(shape[:-1]);res=rv.reshape(shape[:-1])
 elif mode in (5,6):
  # z axes F,P,T,R; each P has an independent t0 and retained approximation.
  for t in range(1,10):
   prev=out[:,:,t-1];diff=z[:,:,t]-prev
   if mode==5:
    hold=(np.abs(diff)<<sh).sum(-1)<=params['temporal_tau'];out[:,:,t]=np.where(hold[...,None],prev,z[:,:,t])
   else:out[:,:,t]=np.where(np.abs(diff)<=np.asarray(params['temporal_rank_tau']),prev,z[:,:,t])
   refresh[:,:,t]=np.any(out[:,:,t]!=prev,axis=-1)
 return out,dict(refresh=refresh,code=code,rank=rank,residual=res)

def rank_threshold(values,budget=.25):
 """One nearest physical rank-threshold budget, no validation tuning."""
 a=np.abs(values.reshape(-1,8));total=np.count_nonzero(a);target=budget*total;th=np.zeros(8,np.int64);saved=0
 for value in sorted(set(a[a!=0].tolist())):
  for r in range(8):
   count=int(np.count_nonzero((a[:,r]<=value)&(a[:,r]>th[r])))
   if count and abs(saved+count-target)<=abs(saved-target):th[r]=value;saved+=count
  if saved>=target:break
 return th.tolist()
