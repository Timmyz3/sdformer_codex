#!/opt/anaconda3/bin/python3.12
from pathlib import Path
import json,struct
import numpy as np
HERE=Path(__file__).resolve().parent
src=HERE.parents[2]/'support_lut_execution_20260915'/'cases.npz'
d=np.load(src,allow_pickle=False);a=d['A'].astype(np.int64)
# Direct Y input-domain diagnosis, deliberately not claimed to come from binary FC1 S/W.
values=np.array([-2**23,2**23-1,0,1,-1,2**22,-2**22,3,-3,127,-128],dtype=np.int64)
p,s,h=np.indices((32,10,96));y=values[(p*7+s*(1+2*(h%3))+h*5)%len(values)]
# Explicit alternating endpoints in multiple channels and times.
y[:,:,0]=np.where((p[:,:,0]+s[:,:,0])%2,-2**23,2**23-1)
y[:,:,1]=np.where((p[:,:,1]+s[:,:,1])%2,2**23-1,-2**23)
u=np.matmul(a,y) # independent full signed int64 MVM; no DA, LUT, prefix, or bounds.
assert np.all((-2**23<=y)&(y<2**23)) and np.all((-2**47<=u)&(u<2**47))
assert {-2**23,2**23-1,0,1,-1}.issubset(set(np.unique(y)))
tau=d['tau'][0].astype(np.int64).copy();positive=(np.arange(96)%3!=1);constant=(np.arange(96)%17==0)
cgate=(np.indices((10,96)).sum(axis=0)%2==0)
# Exact equality for both gain signs, plus unaffected original thresholds.
tau[:,::4]=u[0][:,::4]
gold=np.where(constant[None,None,:],cgate[None,:,:],np.where(positive[None,None,:],u>=tau[None,:,:],u<=tau[None,:,:]))
name='diagnostic_signed24_fullrange'
with (HERE/'fullrange.bin').open('wb') as f:
 f.write(struct.pack('<II',0x50534e35,1));f.write(a.astype('<i2').tobytes())
 f.write(name.encode().ljust(64,b'\0'));f.write(struct.pack('<I',0))
 for arr,ty in [(y,'<i4'),(u,'<i8'),(tau,'<i8'),(positive,'u1'),(constant,'u1'),(cgate,'u1'),(gold,'u1')]:f.write(arr.astype(ty).tobytes())
# A separate exact-high-part model derives retirement, without LUT or recurrent tails.
P=np.maximum(a,0).sum(axis=1);N=np.minimum(a,0).sum(axis=1)
fullplanes=certplanes=early=0;exponents=[]
for pi in range(32):
 for hg in range(12):
  yg=y[pi,:,hg*8:hg*8+8];e=max(int(abs(x)).bit_length() for x in yg.flat);exponents.append(e)
  fullplanes+=max(e,1);lock=np.broadcast_to(constant[hg*8:hg*8+8],(10,8)).copy()
  for m in range(max(e,1)-1,-1,-1):
   high=np.matmul(a,yg>>m)<<m
   low=high+N[:,None]*((1<<m)-1);upper=high+P[:,None]*((1<<m)-1)
   th=tau[:,hg*8:hg*8+8];pos=positive[hg*8:hg*8+8]
   lock|=np.where(pos[None,:],(low>=th)|(upper<th),(low>th)|(upper<=th))
   certplanes+=1
   if lock.all():
    early+=int(m>0);break
  assert lock.all()
info={'status':'PASS','case':name,'real':False,'source':str(src),'Y_direct_input_domain':True,'Y_min':int(y.min()),'Y_max':int(y.max()),'U_min':int(u.min()),'U_max':int(u.max()),'tau_min':int(tau.min()),'tau_max':int(tau.max()),'positive_gain_channels':int(positive.sum()),'negative_gain_channels':int((~positive).sum()),'constant_channels':int(constant.sum()),'ties':int((u==tau[None]).sum()),'negative_gain_ties':int(((u==tau[None])&(~positive[None,None,:])&(~constant[None,None,:])).sum()),'exponent_min':min(exponents),'exponent_max':max(exponents),'groups_e24':exponents.count(24),'expected_full_planes':fullplanes,'expected_cert_planes':certplanes,'expected_early_groups':early,'gold_values':int(u.size)}
(HERE/'fullrange_inputs.json').write_text(json.dumps(info,separators=(',',':'))+'\n')
print(json.dumps(info,separators=(',',':')))
