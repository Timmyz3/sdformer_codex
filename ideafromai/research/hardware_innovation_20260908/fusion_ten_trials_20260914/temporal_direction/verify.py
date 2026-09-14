"""Independent equations from raw source/parameters; no RTL-exported latent/mask inputs."""
import json,csv
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;checks=0
def eq(a,b):
 global checks
 checks+=1
 assert a==b,(a,b)
def order(x):return np.asarray(x).reshape(10,12,8,2,2).transpose(1,3,4,0,2).reshape(480,8)
def encode(z,mode,rank_cost):
 out=z.copy();codes=np.zeros((4,10),int);anchors=np.zeros((4,10),bool);used=np.zeros(4,bool);trials=rejects=0
 if not mode:return out,codes,used,trials,rejects,anchors
 for p in range(4):
  anchor=None;prev=None
  for t in range(10):
   cur=z[p,t];score=int(rank_cost[cur!=0].sum());best=cur.copy();choice=0
   first=np.any(cur) and anchor is None
   candidates=[]
   if t and np.any(cur):
    if np.any(prev):candidates.append((1,prev,1,0))
    if anchor is not None:
     if not np.any(prev) or not np.array_equal(prev,anchor):candidates.append((2,anchor,1,12))
     if mode==2:candidates.extend([(3,anchor,-1,24),(4,anchor,2,12),(5,anchor,-2,24)])
   for code,ref,alpha,basecost in candidates:
    if score<=basecost:continue
    trials+=1;diff=cur-alpha*ref
    if np.any(diff<-4096) or np.any(diff>4095):rejects+=1;continue
    cost=int(rank_cost[diff!=0].sum())+basecost
    if cost<score:score=cost;best=diff;choice=code
   out[p,t]=best;codes[p,t]=choice
   if first:anchor=cur.copy();anchors[p,t]=1
   used[p]|=choice>=2;prev=cur.copy()
 return out,codes,used,trials,rejects,anchors
records=json.loads((H/'results.json').read_text());cases=json.loads((H/'fixtures.json').read_text());pred={};alpha_nonzero_residual=0;range_fallbacks=0
for case in cases:
 d=H/'fixtures'/case['name'];src=np.fromfile(d/'source.bin',dtype='<u2').reshape(96,4,4)
 s=((src[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
 oy,ox=case['input_origin'];valid=np.array([[0<=oy+y<240 and 0<=ox+x<320 for x in range(4)] for y in range(4)])
 s*=valid[None,None]
 q1=np.fromfile(d/'param4.bin',dtype='<i4').reshape(864,8).T.astype(np.int64)
 q2=np.fromfile(d/'param5.bin',dtype='<i4').reshape(12,8,8).transpose(0,2,1).reshape(96,8).astype(np.int64)
 liveK=np.any(q1!=0,axis=0);vlive=np.any(q2.reshape(12,8,8)!=0,axis=1);rankcost=vlive.sum(0)
 patches=np.stack([s[:,:,p//2:p//2+3,p%2:p%2+3].reshape(10,864) for p in range(4)])
 z=patches@q1.T;ptrue=z@q2.T;K=int(liveK.sum());Q=int(np.count_nonzero(np.any(patches!=0,axis=(0,1))&liveK))
 active=patches.astype(bool)&liveK[None,None];U=int((active[0]|active[1]).sum()+(active[2]|active[3]).sum());dual=int((active[0]&active[1]).sum()+(active[2]&active[3]).sum())
 eq(bool(np.array_equal(order(ptrue.transpose(1,2,0).reshape(10,96,2,2)),np.fromfile(d/'raw.bin',dtype='<i4').reshape(480,8))),True)
 fp=np.fromfile(d/'identity_fp32.bin',dtype='<f4').reshape(480,8);J=np.clip(np.rint(fp.astype(np.float64)*2**20),-2**31,2**31-1).astype(np.int64)
 eq(bool(np.array_equal(J,np.fromfile(d/'identity.bin',dtype='<i4').reshape(480,8))),True)
 ab=np.fromfile(d/'param7.bin',dtype='<i4').reshape(12,2,8).astype(np.int64);a=np.repeat(ab[:,0],40,axis=0);b=np.repeat(ab[:,1],40,axis=0)
 wide=order(ptrue.transpose(1,2,0).reshape(10,96,2,2))*a+((J+b)<<20);quo=wide>>26;rem=wide&((1<<26)-1)
 I=np.clip(quo+((rem>1<<25)|((rem==1<<25)&((quo&1)!=0))),-2**23,2**23-1)
 eq(bool(np.array_equal(I,np.fromfile(d/'gold.bin',dtype='<i4').reshape(480,8))),True)
 for mode in range(3):
  e,codes,used,trials,rejects,anchors=encode(z,mode,rankcost);pcalc=np.zeros_like(ptrue)
  for p in range(4):
   anchor_p=None
   for t in range(10):
    code=codes[p,t];base=np.zeros(96,np.int64)
    if code==1:base=pcalc[p,t-1]
    elif code>=2:base=anchor_p*{2:1,3:-1,4:2,5:-2}[code]
    pcalc[p,t]=base+e[p,t]@q2.T
    if anchors[p,t] and used[p]:anchor_p=pcalc[p,t].copy()
  eq(bool(np.array_equal(pcalc,ptrue)),True)
  M=int(((e!=0)*rankcost).sum());V=int((vlive&np.any(e!=0,axis=(0,1))).sum());base=int((codes>=2).sum())*12;neg=int(((codes==3)|(codes==5)).sum())*12;refs=int(used.sum())*12
  enc=80+2*trials if mode else 0
  v=dict(source_words=int(valid.sum())*96,weight_words=Q+V,second_weight_words=V,local_source_reads=K,z_vector_reads=U+40+(40 if mode else 0),z_scalar_reads=M,z_writes=20+U+(40 if mode else 0),first_issues=U,dual_updates=dual,psum_reads=480,psum_writes=480,mac_issues=M,encoder_cycles=enc,candidate_trials=trials,range_rejects=rejects,base_reads=base,base_negations=neg,reference_writes=refs,anchor_count=int(anchors.sum()))
  for code,name in [(1,'prev'),(2,'anchor'),(3,'neg1'),(4,'pos2'),(5,'neg2')]:v['choice_'+name]=int((codes==code).sum())
  v['base_cycles']=5437+K+2*Q+3*U+M+enc+base+neg+refs
  pred[case['name'],mode]=v
  if mode==2:
   alpha_nonzero_residual+=int(((codes>=3)&np.any(e!=0,axis=-1)).sum());range_fallbacks+=rejects
for r in records:
 v=pred[r['fixture'],r['mode']]
 for k,val in v.items():
  if k!='base_cycles':eq(r['core_'+k],val)
 eq(r['core_cycles'],v['base_cycles']+r['core_source_stalls']+r['core_weight_stalls']+r['core_output_stalls'])
 for k in ['outputs','raw_outputs','J_outputs']:eq(r[k],3840)
 eq(r['consumer_cycles'],3385+r['consumer_join_wait_cycles']+r['consumer_output_stalls'])
 eq(r['total_cycles'],r['consumer_cycles']+sum(r[k] for k in ['static_words','parameter_stalls','source_load_words','origin_words','source_load_stalls'])+3)
 eq(r['static_words'],0 if r['command'] else 1848);eq(r['source_load_words'],1536);eq(r['origin_words'],1)
# Original full-mode service must still match the frozen full producer/consumer.
old=json.loads((H.parent/'dataflow/d1_forward/results.json').read_text());lookup={(r['fixture'],r['stall'],r['command']):r for r in old if r['mode']==0}
for r in records:
 if r['mode']==0 and (r['fixture'],r['stall'],r['command']) in lookup:
  oldr=lookup[r['fixture'],r['stall'],r['command']]
  for k,val in oldr.items():
   if isinstance(val,int):eq(r[k],val)
eq(alpha_nonzero_residual>0,True);eq(range_fallbacks>0,True)
res=dict(complete=True,checks=checks,fixtures=len(cases),runs=len(records),values_each_checkpoint=sum(r['outputs'] for r in records),alpha_choices_with_nonzero_residual=alpha_nonzero_residual,range_rejections_per_single_mode2_set=range_fallbacks,method='independent native source/Q1/Q2 integer reconstruction and schedule equations; not another RTL run')
(H/'verification.json').write_text(json.dumps(res,indent=2)+'\n')
with (H/'benefits.csv').open('w') as f:
 fields=['fixture','mode','stall','command','total_cycles','static_words','source_load_words','core_cycles','core_encoder_cycles','core_candidate_trials','core_mac_issues','core_base_reads','core_base_negations','core_reference_writes','core_choice_prev','core_choice_anchor','core_choice_neg1','core_choice_pos2','core_choice_neg2','core_range_rejects','consumer_cycles','consumer_join_wait_cycles','consumer_output_stalls']
 w=csv.DictWriter(f,fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(records)
print(json.dumps(res,indent=2))
