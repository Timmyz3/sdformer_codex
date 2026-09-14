"""Independent work ledger from binary source and numpy integer functions, no RTL latent/counter as input."""
from pathlib import Path
import json,numpy as np
from reference import latent,transform
H=Path(__file__).resolve().parent;p=json.loads((H/'parameters.json').read_text());f=np.load(H/'factors.npz');q1=f['q1'].astype(np.int64);q2=f['q2'].astype(np.int64)
rows=json.loads((H/'results.json').read_text());checks=[]
def ck(row,k,expected):
 got=row[k];checks.append(dict(fixture=row['fixture'],mode=row['mode'],stall=row['stall'],command=row['command'],field=k,expected=int(expected),actual=int(got),pass_=int(expected)==int(got)))
 assert got==expected,(checks[-1])
for case in sorted(set(r['fixture'] for r in rows)):
 r=np.load(H/'fixtures'/case/'reference.npz');src=r['source_bits'].astype(bool);oy,ox=r['origin'];valid=(oy+np.arange(4)[:,None]>=0)&(oy+np.arange(4)[:,None]<240)&(ox+np.arange(4)[None]>=0)&(ox+np.arange(4)[None]<320);src&=valid
 z=latent(src[None],q1)[0];assert np.array_equal(z,r['z'])
 patch=np.stack([src[:,:,y:y+3,x:x+3].reshape(10,864) for y in range(2) for x in range(2)])
 klive=np.any(q1!=0,axis=0);patch=patch[:,:,klive];Q=np.count_nonzero(np.any(patch,axis=(0,1)));A=np.count_nonzero(patch);U=np.count_nonzero(patch[0]|patch[1])+np.count_nonzero(patch[2]|patch[3]);K=int(klive.sum())
 for mode in range(9):
  func={7:6,8:5}.get(mode,mode);zh,meta=transform(z[None],func,p);zh=zh[0];encoded=zh.copy();delta_count=0
  if mode in [7,8]:
   for spatial in range(4):
    for t in range(1,10):
     diff=zh[spatial,t]-zh[spatial,t-1]
     if np.all((diff>=-4096)&(diff<=4095)) and np.count_nonzero(diff)<np.count_nonzero(zh[spatial,t]):encoded[spatial,t]=diff;delta_count+=1
  support=encoded!=0
  if mode in [3,4]:
   support[:]=False
   for spatial in range(4):
    for t in range(10):
     if meta['residual'][0,spatial,t]!=0:support[spatial,t,meta['rank'][0,spatial,t]]=True
  v_live=np.any(q2.reshape(12,8,8)!=0,axis=1);ranklive=np.any(support,axis=(0,1));V=int(np.count_nonzero(v_live&ranklive))
  active=support.copy()
  if mode in [5,6,7,8]:active&=meta['refresh'][0,...,None]
  M=int(sum(np.count_nonzero(active&vl) for vl in v_live));R=int(12*np.count_nonzero(meta['code'])) if mode==3 else 0
  held=int(12*np.count_nonzero(~meta['refresh'])) if mode in [5,6,7,8] else 0
  E={0:0,1:320,2:160,3:1160,4:200,5:296,6:152,7:152,8:296}[mode]
  if mode in [1,2,3,4]:E-=int(np.count_nonzero(~np.any(z!=0,axis=-1)))*{1:6,2:2,3:27,4:3}[mode]
  for row in [x for x in rows if x['fixture']==case and x['mode']==mode]:
   vals=dict(source_words=96*valid.sum(),local_source_reads=K,first_issues=U,dual_updates=A-U,weight_words=Q+V+R,second_weight_words=V,z_vector_reads=U+40+(40 if mode else 0),z_writes=20+U+(40 if mode else 0),z_scalar_reads=M,mac_issues=M,encoder_cycles=E,prototype_reads=R,held_vectors=held,psum_reads=480,psum_writes=480,consumer_raw_words=480,consumer_identity_words=480,consumer_mul_issues=480,consumer_add_issues=960,consumer_round_issues=480,consumer_conversion_issues=480,consumer_output_words=480,consumer_coefficient_words=24,configuration_cycles=3450 if row['command']==0 else 0)
   for key,val in vals.items():ck(row,key,val)
   # VLOAD visits all96 rows even when permission skips physical reads.
   core=5437+K+2*Q+3*U+M+E+R+row['source_stalls']+row['weight_stalls']+row['output_stalls']
   ck(row,'cycles',core)
   ck(row,'consumer_cycles',3385+row['consumer_join_wait']+row['consumer_output_stalls'])
   ck(row,'total_cycles',row['consumer_cycles']+1)
   ck(row,'cycles',sum(row['state_cycles']))
predictions={}
for item in checks:
 if item['stall']==0 and item['command']==0:
  predictions.setdefault(item['fixture']+'/'+str(item['mode']),{})[item['field']]=item['expected']
(H/'ledger_checks.json').write_text(json.dumps(dict(complete=True,check_count=len(checks),all_pass=True,failed=[],independent_no_stall_predictions=predictions,note='All four stall/command records checked; deterministic repeated expected fields stored once. See results.json for actuals.'),indent=2)+'\n')
print('ALL_PASS',len(checks))
