import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
D=H.parents[1]/'r8_consumer_fusion_20260914/data'
x=np.load(D/'consumer_first8.npz');c=np.load(D/'consumer_coefficients.npz')
f=np.load(H/'fitted.npz');q1=f['q1'];q2=f['W_hybrid'];a=c['a_q40'];b=c['b_q20'];A=f['A'];B=f['B'];escape=f['escape'];cases=[]
def order(t):return np.asarray(t).reshape(10,12,8,2,2).transpose(1,3,4,0,2).reshape(480,8)
def rne(v,shift):
 q=v>>shift;r=v&((1<<shift)-1)
 return q+((r>(1<<(shift-1)))|((r==(1<<(shift-1)))&((q&1)!=0)))
def params(aq,bq,aa,bb):
 ew=bq.astype(np.int64)@aq.astype(np.int64)
 assert abs(ew).max()<2**31
 p={1:ew.reshape(12,8,864).transpose(0,2,1).reshape(10368,8),2:np.ones((288,8),np.int64),4:aq.T,5:bq.reshape(12,8,8).transpose(0,2,1).reshape(96,8),6:np.repeat(np.any(aq!=0,axis=0)[:,None],8,1)}
 p[7]=np.stack((aa.reshape(12,8),bb.reshape(12,8)),axis=1).reshape(24,8)
 p[8]=np.pad(A,((0,0),(0,4)));p[9]=B.T
 p[10]=np.zeros((1,8),np.int64);p[10][0,0]=sum(int(v)<<i for i,v in enumerate(escape))
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
 emit(f'real_{k}',x['source_bits'][k],x['input_origin_yx'][k],j,fp_identity=x['identity_fp32'][k])
zeros=np.zeros((10,96,4,4),np.int64);ones=np.ones_like(zeros);j0=np.zeros((10,96,2,2),np.int64)
emit('zero',zeros,[79,119],j0);emit('one',ones,[79,119],j0);emit('padding_poison',ones,[-1,-1],j0)
j=np.resize(np.array([-160,-96,-32,32,96,160,-2**31,2**31-1],np.int64),j0.shape)
emit('identity_ties_sat',zeros,[79,119],j,aa=np.zeros(96,np.int32),bb=np.zeros(96,np.int32))
# Same factor chain, all-exact and all-factor masks cover transitions absent in the fit.
for name,esc in [('all_factor',np.zeros(12,bool)),('all_exact',np.ones(12,bool))]:
 escape=esc
 wh=f['W_kron'].copy();wh.reshape(12,8,8)[esc]=f['q2'].reshape(12,8,8)[esc]
 emit(name,ones,[79,119],j0,bq=wh)
# Exercise signed multiplier widths with bounded native producer outputs.
A=np.full((12,4),-4096,np.int64);B=np.full((8,2),-32,np.int64);escape=np.zeros(12,bool)
aq=np.zeros_like(q1);aq[:,0]=-4;bq=np.einsum('ok,lj->olkj',A,B).reshape(96,8)
emit('factor_negative_extreme',ones,[79,119],j0,aq,bq,np.full(96,2**31-1,np.int32),np.full(96,-2**31,np.int32))
# Explicit cancellation within each two-input contraction.
A=np.full((12,4),4095,np.int64);B=np.tile(np.array([31,-31]),(8,1));escape=np.zeros(12,bool)
aq=np.zeros_like(q1);aq[:,0]=3;bq=np.einsum('ok,lj->olkj',A,B).reshape(96,8)
emit('factor_cancellation',ones,[79,119],j0,aq,bq)
(H/'fixtures.json').write_text(json.dumps(cases,indent=2)+'\n')
print(dict(fixtures=len(cases),real_function='train-only hybrid Kronecker approximation with two exact output groups',same_function_control=True))
