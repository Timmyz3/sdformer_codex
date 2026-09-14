import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
D=H.parents[2]/'r8_consumer_fusion_20260914/data'
x=np.load(D/'consumer_first8.npz');c=np.load(D/'consumer_coefficients.npz')
q1=c['q1'];q2=c['q2'];a=c['a_q40'];b=c['b_q20'];cases=[]
def order(t):return np.asarray(t).reshape(10,12,8,2,2).transpose(1,3,4,0,2).reshape(480,8)
def rne(v,shift):
 q=v>>shift;r=v&((1<<shift)-1)
 return q+((r>(1<<(shift-1)))|((r==(1<<(shift-1)))&((q&1)!=0)))
def params(aq,bq,aa,bb):
 ew=bq.astype(np.int64)@aq.astype(np.int64)
 assert abs(ew).max()<2**20
 p={1:ew.reshape(12,8,864).transpose(0,2,1).reshape(10368,8),2:np.ones((288,8),np.int64),4:aq.T,5:bq.reshape(12,8,8).transpose(0,2,1).reshape(96,8),6:np.repeat(np.any(aq!=0,axis=0)[:,None],8,1)}
 p[7]=np.stack((aa.reshape(12,8),bb.reshape(12,8)),axis=1).reshape(24,8)
 return {k:v for k,v in p.items() if k not in [1,2]}
def emit(name,s,origin,j,aq=q1,bq=q2,aa=a,bb=b,expected=None,fp_identity=None):
 d=H/'fixtures'/name;d.mkdir(parents=True,exist_ok=True)
 s=np.asarray(s,dtype=np.int64).copy();raw_s=s.copy()
 for y in range(4):
  for xx in range(4):
   if not(0<=origin[0]+y<240 and 0<=origin[1]+xx<320):s[:,:,y,xx]=0
 patches=np.stack([s[:,:,p//2:p//2+3,p%2:p%2+3].reshape(10,864) for p in range(4)])
 z=patches@aq.astype(np.int64).T;p=(z@bq.astype(np.int64).T).transpose(1,2,0).reshape(10,96,2,2)
 if expected is not None:assert np.array_equal(p,expected)
 j=np.asarray(j,dtype=np.int64)
 fp_identity=(j.astype(np.float64)/(1<<20)).astype(np.float32) if fp_identity is None else np.asarray(fp_identity,np.float32)
 jr=np.rint(fp_identity.astype(np.float64)*(1<<20));j=np.clip(jr,-2**31,2**31-1).astype(np.int64)
 order(fp_identity).astype('<f4').tofile(d/'identity_fp32.bin')
 conversion_saturations=int(np.count_nonzero(jr!=j))
 wide=p*aa[None,:,None,None].astype(np.int64)+(j+bb[None,:,None,None].astype(np.int64))*(1<<20)
 rounded=rne(wide,26);gold=np.clip(rounded,-2**23,2**23-1)
 words=np.sum(raw_s.astype(np.uint16)*(1<<np.arange(10,dtype=np.uint16))[:,None,None,None],axis=0)
 words.astype('<u2').tofile(d/'source.bin')
 for key,v in [('raw',p),('identity',j),('gold',gold)]:order(v).astype('<i4').tofile(d/(key+'.bin'))
 for k,v in params(aq,bq,aa,bb).items():np.asarray(v,dtype='<u4').tofile(d/f'param{k}.bin')
 ty=(int(origin[0])+1)//2;tx=(int(origin[1])+1)//2;assert tuple(origin)==(2*ty-1,2*tx-1)
 case=dict(name=name,tile_id=ty*160+tx,input_origin=list(map(int,origin)),spikes=int(s.sum()),saturations=int(np.count_nonzero(gold!=rounded)),conversion_saturations=conversion_saturations)
 (d/'meta.json').write_text(json.dumps(case,indent=2)+'\n');cases.append(case)
for k in range(8):
 j=np.clip(np.rint(x['identity_fp32'][k].astype(np.float64)*(1<<20)),-2**31,2**31-1).astype(np.int64)
 emit(f'real_{k}',x['source_bits'][k],x['input_origin_yx'][k],j,expected=x['p_int'][k],fp_identity=x['identity_fp32'][k])
zeros=np.zeros((10,96,4,4),np.int64);ones=np.ones_like(zeros);j0=np.zeros((10,96,2,2),np.int64)
emit('zero',zeros,[79,119],j0)
emit('one',ones,[79,119],j0)
emit('padding_poison',ones,[-1,-1],j0)
# A zero raw p isolates exact negative ties and signed24 saturation of identity.
j=np.resize(np.array([-160,-96,-32,32,96,160,-2**31,2**31-1],np.int64),j0.shape)
emit('identity_ties_sat',zeros,[79,119],j,aa=np.zeros(96,np.int32),bb=np.zeros(96,np.int32))
# Product ties of both signs, with nonzero raw p produced by full native K service.
aq=np.zeros_like(q1);aq[0,0]=1;bq=np.zeros_like(q2);bq[:,0]=1
at=np.resize(np.array([1,3,5,-1,-3,-5],np.int64)*(1<<25),(96,)).astype(np.int32)
emit('product_ties',ones,[79,119],j0,aq,bq,at,np.zeros(96,np.int32))
# Exercise full producer extrema and signed64 multiply/add plus output clipping.
for sign in [1,-1]:
 emit('extreme_'+str(sign),ones,[79,119],j,np.full_like(q1,3*sign),np.full_like(q2,-32768),np.full(96,2**31-1,np.int32),np.full(96,-2**31,np.int32))
fpbits=np.array([0x00000000,0x80000000,0x00000001,0x80000001,0x007fffff,0x807fffff,0x7f7fffff,0xff7fffff],np.uint32)
fp=np.concatenate((fpbits.view(np.float32),np.array([.5,1.5,2.5,-.5,-1.5,-2.5,31.5,32.5,-31.5,-32.5],np.float32)/(1<<20)))
emit('fp_conversion_boundaries',zeros,[79,119],j0,aa=np.zeros(96,np.int32),bb=np.zeros(96,np.int32),fp_identity=np.resize(fp,j0.shape))
(H/'fixtures.json').write_text(json.dumps(cases,indent=2)+'\n')
print(json.dumps({'fixtures':len(cases),'real_p_array_equal':True,'a_range':[int(a.min()),int(a.max())],'b_range':[int(b.min()),int(b.max())]},indent=2))
