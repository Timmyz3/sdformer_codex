"""Read-only equations and expanded-matrix reference; no RTL/GPU rerun."""
from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent
D=H.parent.parent/'r8_consumer_fusion_20260914/data'
pc=np.array([i.bit_count() for i in range(1024)],np.int64)
def load_i(p):return np.fromfile(p,dtype='<i4').astype(np.int64)
def predict(name):
 f=H/'fixtures'/name;m=json.loads((f/'meta.json').read_text());oy,ox=m['input_origin']
 s=np.fromfile(f/'source.bin',dtype='<u2').reshape(96,4,4).copy()
 valid=np.array([[0<=oy+y<240 and 0<=ox+x<320 for x in range(4)] for y in range(4)])
 s*=valid;g=(s[None]>>np.arange(10)[:,None,None,None])&1
 q1=load_i(f/'param4.bin').reshape(864,8).T
 q2=load_i(f/'param5.bin').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
 k_live=np.any(q1!=0,axis=0)
 patches=np.stack([g[:,:,p//2:p//2+3,p%2:p%2+3].reshape(10,864) for p in range(4)])
 z=patches@q1.T;p=patches@(q2@q1).T
 raw=load_i(f/'raw.bin').reshape(12,4,10,8).transpose(1,2,0,3).reshape(4,10,96)
 assert np.array_equal(p,raw)
 a=patches[:,:,k_live].astype(bool)
 counts={2:int((a[0]|a[1]).sum()+(a[2]|a[3]).sum()),3:int(a.any(0).sum())}
 A=int(a.sum());K=int(k_live.sum());Q=int(a.any((0,1)).sum())
 live=z!=0;vl=np.any(q2.reshape(12,8,8)!=0,axis=1);rl=live.any((0,1))
 V=int((vl&rl).sum());M=int((live.sum((0,1))*vl.sum(0)).sum())
 return dict(U=counts,A=A,K=K,Q=Q,V=V,M=M,source=96*int(valid.sum()))
def main():
 rows=json.loads((H/'results.json').read_text());cache={};checks=0
 for r in rows:
  if r['fixture'] not in cache:cache[r['fixture']]=predict(r['fixture'])
  p=cache[r['fixture']];u=p['U'][r['mode']]
  expected=dict(core_first_issues=u,core_merged_updates=p['A']-u,core_z_vector_reads=u+40,
   core_z_scalar_reads=p['M'],core_z_writes=u+10,core_mac_issues=p['M'],core_local_source_reads=p['K'],
   core_weight_words=p['Q']+p['V'],core_second_weight_words=p['V'],core_source_words=p['source'],
   core_psum_reads=480,core_psum_writes=480)
  expected['core_cycles']=5427+p['K']+2*p['Q']+3*u+p['M']+sum(r[k] for k in ['core_source_stalls','core_weight_stalls','core_output_stalls','core_borrow_waits'])
  for k,v in expected.items():assert r[k]==v,(r['fixture'],r['mode'],k,r[k],v);checks+=1
 all_rows=rows.copy()
 for name in ['64','full']:
  f=H/f'results_{name}.json'
  if f.exists():all_rows+=json.loads(f.read_text())
 for r in all_rows:
  t=r['retired_tiles'];assert r['consumer_cycles']==3385*t+r['consumer_join_wait_cycles']+r['consumer_output_stalls']+r['consumer_wide_waits'];checks+=1
  assert r['total_cycles']==r['consumer_cycles']+r['static_words']+r['parameter_stalls']+1536*t+r['origin_words']+r['source_load_stalls']+2*t+1;checks+=1
  for k,v in [('consumer_identity_words',480*t),('consumer_coefficient_words',24*t),('consumer_add_issues',960*t),('consumer_conversion_issues',480*t),('outputs',3840*t)]:assert r[k]==v;checks+=1
  if r['mode']==2:assert r['core_borrow_waits']==0;checks+=1
 full_counts={}
 if (H/'results_full.json').exists():
  s=np.pad(np.load(D/'first_source_words.npy'),((0,0),(1,1),(1,1)))
  q1=np.load(D/'consumer_coefficients.npz')['q1'];kl=np.any(q1!=0,0).reshape(96,9)
  a=u2=u4=0
  for ky in range(3):
   for kx in range(3):
    m=kl[:,ky*3+kx,None,None]
    v=[s[:,ky+p//2:ky+p//2+240:2,kx+p%2:kx+p%2+320:2]*m for p in range(4)]
    a+=sum(int(pc[x].sum()) for x in v)
    u2+=int(pc[v[0]|v[1]].sum()+pc[v[2]|v[3]].sum());u4+=int(pc[v[0]|v[1]|v[2]|v[3]].sum())
  rr={r['mode']:r for r in json.loads((H/'results_full.json').read_text())}
  assert rr[2]['core_first_issues']==u2 and rr[3]['core_first_issues']==u4
  assert rr[2]['total_cycles']-rr[3]['total_cycles']==3*(u2-u4)
  full_counts=dict(scalar=a,pair_union=u2,quad_union=u4,additional_merge=u2-u4,saved_cycles=3*(u2-u4));checks+=3
 real={}
 for mode in [2,3]:
  a=[r for r in rows if r['mode']==mode and r['command']==0 and r['stall']==0 and r['fixture'].startswith('real_')]
  real[mode]={k:sum(r[k] for r in a) for k in ['total_cycles','core_cycles','consumer_cycles','core_first_issues','core_merged_updates','static_words']}
 out=dict(complete=True,checks=checks,command_records=len(all_rows),expanded_gold_values=len(cache)*3840,real8=real,full_source_counts=full_counts)
 (H/'checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
