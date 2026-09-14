import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
old=BASE/'r0_source_retirement_20260913/data'
factor=BASE/'open_fusion_execution/major_operator_fusions_20260913/decomposition_owned/results/flat_svd_r8_w32.npz'
f=np.load(factor); inp=np.load(old/'dense_q16.npz')
u=f['first'].astype(np.float64).reshape(8,864)
v=f['second'].astype(np.float64).reshape(96,8)
s1=np.max(abs(u),axis=1)/3
q1=np.rint(u/s1[:,None]).astype(np.int8)
c=v*s1[None,:]
s2=np.max(abs(c),axis=1)/32767
q2=np.rint(c/s2[:,None]).astype(np.int16)
w=q2.astype(np.int64)@q1.astype(np.int64)
assert np.max(abs(w)) < 2**20
np.savez(HERE/'factors.npz',q1=q1,q2=q2,output_scale=s2,expanded=w,first_scale=s1,
         first_original=u,second_original=v,bias=f['bias'],theta=f['theta'])
meta={'input_factor':str(factor),'rank':8,'Q1_range':[int(q1.min()),int(q1.max())],
      'Q2_range':[int(q2.min()),int(q2.max())],'expanded_range':[int(w.min()),int(w.max())],
      'definition':'p=(Q2@Q1)g; network output=p*output_scale; no intermediate rounding',
      'bound_z':864*3,'bound_p':8*32768*864*3,
      'factor_bytes':int(q1.nbytes+q2.nbytes),'Q1_hardware_packed_bits':3*8*864,
      'weight_relL2_vs_existing_SVD8':float(np.linalg.norm(s2[:,None]*w-v@u)/np.linalg.norm(v@u)),
      'quality_record':'../data/r8_valid825.json',
      'quality_note':'Separate measured record; this preparation script does not evaluate AEE.',
      'bias_zero':bool(np.all(f['bias']==0))}
assert meta['bias_zero'] and float(f['theta'])==1
cases=[]
def write_hex(path,a):
    path.write_text(''.join(f'{int(x)&0xffffffff:08x}\n' for x in np.asarray(a).ravel()))
def emit(name,s,origin,q1_case=None,q2_case=None):
    aq=q1 if q1_case is None else q1_case
    bq=q2 if q2_case is None else q2_case
    ew=bq.astype(np.int64)@aq.astype(np.int64)
    s=s.copy().astype(np.int64)
    for y in range(4):
        for x in range(4):
            if not(0<=origin[0]+y<240 and 0<=origin[1]+x<320):s[:,:,y,x]=0
    patches=np.stack([s[:,:,p//2:p//2+3,p%2:p%2+3].reshape(10,864) for p in range(4)])
    z=patches@aq.astype(np.int64).T
    gold=z@bq.astype(np.int64).T
    assert np.array_equal(gold,patches@ew.T)
    d=HERE/'fixtures'/name;d.mkdir(parents=True,exist_ok=True)
    words=np.sum(s.astype(np.uint16)* (1<<np.arange(10,dtype=np.uint16))[:,None,None,None],axis=0)
    write_hex(d/'source.hex',words)
    write_hex(d/'weight.hex',ew.reshape(12,8,864).transpose(0,2,1))
    write_hex(d/'q1.hex',aq.T);write_hex(d/'q2.hex',bq.reshape(12,8,8).transpose(0,2,1))
    write_hex(d/'k_live.hex',np.any(aq!=0,axis=0))
    write_hex(d/'mask.hex',np.ones(288));write_hex(d/'origin.hex',origin)
    write_hex(d/'gold.hex',gold.reshape(4,10,12,8).transpose(2,0,1,3))
    cases.append({'name':name,'origin':list(map(int,origin)),'spikes':int(s.sum()),'nonzero_z':int(np.count_nonzero(z))})
for j in range(8):emit(f'real_{j}',inp['source_bits'][j],inp['input_origin_yx'][j])
emit('zero',np.zeros((10,96,4,4)),[0,0])
emit('one',np.ones((10,96,4,4)),[80,120])
emit('corner',np.ones((10,96,4,4)),[-1,-1])
emit('max_positive',np.ones((10,96,4,4)),[80,120],np.full_like(q1,-3),np.full_like(q2,-32768))
emit('max_negative',np.ones((10,96,4,4)),[80,120],np.full_like(q1,3),np.full_like(q2,-32768))
emit('zero_factor',np.ones((10,96,4,4)),[80,120],np.zeros_like(q1),q2)
meta['fixtures']=cases
(HERE/'definition.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({k:v for k,v in meta.items() if k!='fixtures'},indent=2))
