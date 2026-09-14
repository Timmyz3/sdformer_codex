"""Independent native-pair packed execution and full integer function audit."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent;R=HERE.parent/'packed_rtl'
def hx(p,s=False):
 a=np.array([int(v,16) for v in p.read_text().split()],np.int64)
 return np.where(a>=2**31,a-2**32,a) if s else a

def predict(name):
 f=R/'fixtures'/name;w=hx(f/'source.hex').reshape(96,4,4)&1023;oy,ox=hx(f/'origin.hex',True)
 valid=np.array([[0<=oy+y<240 and 0<=ox+x<320 for x in range(4)] for y in range(4)]);w=w*valid
 g=((w[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
 q1=hx(f/'q1.hex',True).reshape(864,8).T;q2=hx(f/'q2.hex',True).reshape(12,8,8).transpose(0,2,1).reshape(96,8)
 kl=hx(f/'k_live.hex').astype(bool);assert np.array_equal(kl,np.any(q1!=0,axis=0))
 patches=np.stack([g[:,:,y:y+3,x:x+3].reshape(10,864) for y in range(2) for x in range(2)])
 z=patches@q1.T;p=z@q2.T
 gold=hx(f/'gold.hex',True).reshape(12,4,10,8).transpose(1,2,0,3).reshape(4,10,96)
 assert np.array_equal(p,gold) and np.array_equal(p,patches@(q2@q1).T)
 assert np.max(np.abs(z))<=2592 and np.max(np.abs(p))<=679477248
 K=int(kl.sum());Q=int(patches[:,:,kl].any((0,1)).sum());A=int(patches[:,:,kl].sum())
 G=patches[:,:,kl].astype(bool);U=int(np.logical_or(G[::2],G[1::2]).sum());D=int(np.logical_and(G[::2],G[1::2]).sum());assert A-U==D
 nonzero=z!=0;vlive=np.any(q2.reshape(12,8,8)!=0,axis=1);r_live=np.any(nonzero,axis=(0,1))
 V=int((vlive&r_live).sum());M=int((nonzero.sum((0,1))*vlive.sum(0)).sum())
 pred={}
 for mode,a in ((14,A),(15,U)):
  pred[mode]=dict(cycles=5437+K+2*Q+3*a+M,outputs=3840,source_words=96*int(valid.sum()),weight_words=Q+V,
   second_weight_words=V,local_source_reads=K,z_vector_reads=a+40,z_scalar_reads=M,z_writes=20+a,
   first_issues=a,dual_updates=D if mode==15 else 0,psum_reads=480,psum_writes=480,mac_issues=M)
 return pred,dict(fixture=name,K=K,Q=Q,A_scalar=A,A_pair_union=U,dual_overlap=D,second_active_weight_words=V,MACs=M,gold_values=p.size)

def main():
 rows=json.loads((R/'results.json').read_text());cache={};checks=0
 for r in rows:
  name=r['fixture']
  if name not in cache:cache[name]=predict(name)
  pred,d=cache[name];p=pred[r['mode']]
  for k,v in p.items():
   observed=r[k]-(r['source_stalls']+r['weight_stalls']+r['output_stalls']) if k=='cycles' else r[k]
   assert observed==v,(name,r['mode'],k,observed,v);checks+=1
  assert r['configuration_cycles']==(3361 if r['command']==0 else 0);checks+=1
  a=p['first_issues'];s=r['state_cycles'];expected={0:0,1:20,2:96,3:1536+r['source_stalls'],4:864,5:d['K'],7:a+d['Q'],8:a,9:a,10:864,11:40,13:480,14:d['MACs'],15:480,16:480,17:480+r['output_stalls'],18:1}
  for i,v in expected.items():assert s[i]==v,(name,r['mode'],i,s[i],v);checks+=1
  assert s[6]+s[12]==d['Q']+96+r['weight_stalls'];assert sum(s)==r['cycles'];checks+=2
 out=dict(complete=True,RTL_rerun=False,GPU_rerun=False,runs=len(rows),outputs_compared=sum(r['outputs'] for r in rows),independent_gold_values=sum(v[1]['gold_values'] for v in cache.values()),checks=checks,all_match=True,fixtures=[v[1] for v in cache.values()])
 (HERE/'review_packed_counts.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='fixtures'},indent=2))
if __name__=='__main__':main()
