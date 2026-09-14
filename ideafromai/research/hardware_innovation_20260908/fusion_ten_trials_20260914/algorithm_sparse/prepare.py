from pathlib import Path
import json,numpy as np
from reference import latent,transform,consumer,MODES
H=Path(__file__).resolve().parent;OLD=H.parents[1]/'r8_consumer_fusion_20260914'
f=np.load(H/'factors.npz');co=np.load(H/'consumer_coefficients.npz');par=json.loads((H/'parameters.json').read_text())
r=np.load(OLD/'data/consumer_integer_first8.npz');ident=np.load(OLD/'data/consumer_first8.npz')
print('keys',r.files,ident.files)
def hx(path,x):
 path.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in np.asarray(x).reshape(-1)))
def rows(x):return x.reshape(4,10,12,8).transpose(2,0,1,3)
q1=f['q1'].astype(np.int64);q2=f['q2'].astype(np.int64);pt=np.asarray(par['codebook'])@q2.T
assert np.max(np.abs(np.asarray(par['codebook'])))<=2592 and np.max(np.abs(pt))<2**31
common=H/'constants';common.mkdir(exist_ok=True)
hx(common/'q1.hex',q1.T);hx(common/'q2.hex',q2.reshape(12,8,8).transpose(0,2,1));hx(common/'k_live.hex',np.any(q1!=0,axis=0))
params=np.zeros((13,8),np.int64);params[0]=par['shifts'];params[1]=par['rank_tau'];params[2]=par['temporal_rank_tau'];params[3,0]=par['temporal_tau'];params[4:,0]=np.arange(9)*par['group_tau']
hx(common/'parameters.hex',params);hx(common/'codebook.hex',par['codebook']);hx(common/'prototypes.hex',pt.reshape(4,12,8).transpose(1,0,2))
hx(common/'consumer.hex',np.stack([co['a_q40'].reshape(12,8),co['b_q20'].reshape(12,8)],axis=1))
fixtures=[]
# independent real evaluation frame, never used by this calibration.
source=r['source_bits'];identity=ident['identity_fp32'];origins=r['input_origin_yx']
for ix in range(11):
 if ix<8:s=source[ix];id=identity[ix];origin=origins[ix];name=f'real_{ix}'
 else:
  s=np.zeros_like(source[0]) if ix==8 else np.ones_like(source[0]);id=np.zeros_like(identity[0]);origin=np.array([31,47]) if ix<10 else np.array([-1,-1]);name=['zero','ones','poison_corner'][ix-8]
 # Golden enforces RTL boundary protection, even for deliberately poisoned outside source.
 valid=(origin[0]+np.arange(4)[:,None]>=0)&(origin[0]+np.arange(4)[:,None]<240)&(origin[1]+np.arange(4)[None]>=0)&(origin[1]+np.arange(4)[None]<320)
 masked=s*valid;z=latent(masked[None],q1)[0];J=np.clip(np.rint(id.astype(np.float64)*2**20),-2**31,2**31-1).astype(np.int64).reshape(10,96,4).transpose(2,0,1)
 d=H/'fixtures'/name;d.mkdir(parents=True,exist_ok=True)
 words=np.sum(s.astype(np.uint16)<<np.arange(10,dtype=np.uint16)[:,None,None,None],axis=0,dtype=np.uint16);hx(d/'source.hex',words);hx(d/'origin.hex',origin)
 hx(d/'identity.hex',rows(id.reshape(10,96,4).transpose(2,0,1)).astype('<f4').view('<u4'));hx(d/'j.hex',rows(J))
 for mode in range(9):
  zh,meta=transform(z[None],{7:6,8:5}.get(mode,mode),par);p=zh[0]@q2.T;i=consumer(p,J,co['a_q40'],co['b_q20'])
  if mode in [3,4]:
   decoded=pt[meta['code'][0]]+meta['residual'][0,...,None]*q2.T[meta['rank'][0]];assert np.array_equal(decoded,p)
  hx(d/f'gold_{mode}.hex',rows(p));hx(d/f'i24_{mode}.hex',rows(i))
 np.savez_compressed(d/'reference.npz',source_bits=s,origin=origin,identity=id,z=z,J=J)
 fixtures.append(dict(name=name,origin=origin.tolist(),kind='heldout_actual_frame' if ix<8 else 'functional'))
(H/'definition.json').write_text(json.dumps(dict(fixtures=fixtures,modes=dict(MODES,**{'7':'temporal_rank_deadband_delta','8':'temporal_group_hold_delta'}),training_frame=par['training_frame'],evaluation_frame='zurich_city_09_a_0001.npy',completeK=864,T=10,C=96,N=96,spatial_outputs=4,output_layout='og,P,T,lane',configuration_words=1536+1+864+96+864+13+4+48+24),indent=2)+'\n')
