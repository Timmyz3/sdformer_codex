from pathlib import Path
import json,struct
import numpy as np
P=Path(__file__).resolve().parent
raw=(P.parent/'fused_datapath/fullrange.bin').read_bytes();magic,count=struct.unpack_from('<II',raw);assert magic==0x50534e35 and count==1
at=8;a=np.frombuffer(raw,dtype='<i2',count=100,offset=at).astype(np.int64).reshape(10,10);at+=200+64+4
arrays=[]
for ty,n in [('<i4',30720),('<i8',30720),('<i8',960),('u1',96),('u1',96),('u1',960),('u1',30720)]:
 x=np.frombuffer(raw,dtype=ty,count=n,offset=at).copy();at+=x.nbytes;arrays.append(x)
y=arrays[0].reshape(32,10,96).astype(np.int64);old_u=arrays[1].reshape(32,10,96);old_tau=arrays[2].reshape(10,96)
pos=arrays[3].astype(bool);const=arrays[4].astype(bool);cg=arrays[5].reshape(10,96).astype(bool)
u=np.matmul(a,y);assert np.array_equal(u,old_u)
Psum=np.maximum(a,0).sum(1);Nsum=np.minimum(a,0).sum(1)
rows=[]
with (P/'diagnostics.bin').open('wb') as f:
 f.write(struct.pack('<II',0x50534e35,2));f.write(a.astype('<i2').tobytes())
 for variant in range(2):
  name=['diagnostic_signed24_fullrange','diagnostic_Y24_tau48_endpoints'][variant];tau=old_tau.copy()
  if variant:
   tau[:,0::8]=-(1<<47);tau[:,1::8]=(1<<47)-1;tau[:,2::8]=(1<<47)-1;tau[:,3::8]=-(1<<47)
   tau[:,4::8]=u[0][:,4::8];tau[:,5::8]=0;tau[:,6::8]=-1;tau[:,7::8]=1
  g=np.where(const[None,None,:],cg[None,:,:],np.where(pos[None,None,:],u>=tau[None,:,:],u<=tau[None,:,:]))
  qn=tau+Nsum[:,None]-pos[None,:].astype(np.int64);qp=tau+Psum[:,None]-pos[None,:].astype(np.int64)
  assert np.all((qn>=-(1<<48))&(qn<(1<<48))) and np.all((qp>=-(1<<48))&(qp<(1<<48)))
  if variant:assert qn.min()<-(1<<47) and qp.max()>=(1<<47)
  f.write(name.encode().ljust(64,b'\0'));f.write(struct.pack('<I',0))
  for arr,ty in [(y,'<i4'),(u,'<i8'),(tau,'<i8'),(pos,'u1'),(const,'u1'),(cg,'u1'),(g,'u1')]:f.write(arr.astype(ty).tobytes())
  certplanes=early=0
  for pi in range(32):
   for hg in range(12):
    yy=y[pi,:,hg*8:hg*8+8];locked=np.broadcast_to(const[hg*8:hg*8+8],(10,8)).copy()
    for m in range(23,-1,-1):
     nv=np.matmul(a,yy>>m);scale=1<<m
     lo=nv*scale+Nsum[:,None]*(scale-1);hi=nv*scale+Psum[:,None]*(scale-1)
     th=tau[:,hg*8:hg*8+8];ps=pos[hg*8:hg*8+8]
     lower=np.where(ps[None,:],lo>=th,lo>th);upper=np.where(ps[None,:],hi<th,hi<=th)
     assert np.array_equal(lower,nv+Nsum[:,None]>(qn[:,hg*8:hg*8+8]>>m))
     assert np.array_equal(upper,nv+Psum[:,None]<=(qp[:,hg*8:hg*8+8]>>m))
     locked|=lower|upper;certplanes+=1
     if locked.all():early+=int(m>0);break
  rows.append({'case':name,'Y_min':int(y.min()),'Y_max':int(y.max()),'U_min':int(u.min()),'U_max':int(u.max()),'tau_min':int(tau.min()),'tau_max':int(tau.max()),'qN_min':int(qn.min()),'qN_max':int(qn.max()),'qP_min':int(qp.min()),'qP_max':int(qp.max()),'q_values_outside_signed48':int(((qn<-(1<<47))|(qn>=(1<<47))).sum()+((qp<-(1<<47))|(qp>=(1<<47))).sum()),'ties':int((u==tau[None]).sum()),'expected_full_planes':24*384,'expected_cert_planes':certplanes,'expected_early_groups':early})
# Arithmetic identity probe: actual A row sums, all m0..23, both gains and exact predicate boundaries.
checks=0
for t in range(10):
 for m in range(24):
  for nv in [-(1<<23),-100001,-1,0,1,100001,(1<<23)-1]:
   lo=(nv+int(Nsum[t]))*(1<<m)-int(Nsum[t]);hi=(nv+int(Psum[t]))*(1<<m)-int(Psum[t])
   for tau in [-(1<<47),(1<<47)-1,-1,0,1,lo-1,lo,lo+1,hi-1,hi,hi+1]:
    if not -(1<<47)<=tau<(1<<47):continue
    for positive in [0,1]:
     assert ((nv+int(Nsum[t]))>((tau+int(Nsum[t])-positive)>>m))==(lo>=tau if positive else lo>tau)
     assert ((nv+int(Psum[t]))<=((tau+int(Psum[t])-positive)>>m))==(hi<tau if positive else hi<=tau)
     checks+=2
info={'status':'PASS','diag_cases':rows,'algebra_predicate_checks':checks,'scope':'Two Y24 direct-input diagnoses using actual A; second covers tau48 endpoints and ties. Not FC1-attainable distribution and excluded from main32real.'}
(P/'diagnostic_inputs.json').write_text(json.dumps(info,separators=(',',':'))+'\n');print(json.dumps(info,separators=(',',':')))
