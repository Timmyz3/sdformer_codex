import numpy as np,json
from pathlib import Path
H=Path(__file__).resolve().parent
B=H.parents[1]
S=np.load(B/'r8_consumer_fusion_20260914/data/first_source_words.npy',mmap_mode='r')
k_live=np.array([int(x,16)for x in (B/'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0/k_live.hex').read_text().split()],bool)
def events(ids):
 out=[]
 for tile in ids:
  oy,ox=2*(tile//160)-1,2*(tile%160)-1; a=np.zeros((864,4),np.uint16)
  for k in range(864):
   if not k_live[k]:continue
   c,tap=divmod(k,9)
   for p in range(4):
    y=oy+p//2+tap//3;x=ox+p%2+tap%3
    if 0<=y<240 and 0<=x<320:a[k,p]=S[c,y,x]
  out.append((a[:,:,None]>>np.arange(10))&1)
 return np.stack(out).astype(np.uint8)
cal=events(range(0,32)); x=cal.reshape(-1,2,2,10); states=x[:,:,0,:]+2*x[:,:,1,:]
cost=np.count_nonzero(states[...,None]!=states[...,None,:],axis=(0,1)) if False else np.array([[np.count_nonzero(states[...,i]!=states[...,j])for j in range(10)]for i in range(10)])
start=np.count_nonzero(states,axis=(0,1));dp={}
for i in range(10):dp[1<<i,i]=(int(start[i]),(i,))
for bits in range(1,1024):
 for j in range(10):
  if (bits,j)not in dp:continue
  c,p=dp[bits,j]
  for nxt in range(10):
   if bits>>nxt&1:continue
   key=bits|(1<<nxt),nxt;v=(c+int(cost[j,nxt]),p+(nxt,))
   if key not in dp or v<dp[key]:dp[key]=v
value,perm=min(dp[1023,j]for j in range(10))
held=events(range(128,192));metrics={}
for label,a in [('cal_0_31',cal),('held_128_191',held)]:
 for name,order in [('original',tuple(range(10))),('calibrated',perm)]:
  a0=a[...,order];d=a0^np.concatenate([np.zeros_like(a0[...,:1]),a0[...,:-1]],axis=-1)
  D=np.count_nonzero(a0.reshape(-1,2,2,10).any(axis=2),axis=(1,2)).reshape(a.shape[:2])
  E=np.count_nonzero(d.reshape(-1,2,2,10).any(axis=2),axis=(1,2)).reshape(a.shape[:2])
  metrics[label,name]={'direct_updates':int(D.sum()),'endpoint_updates':int(E.sum()),'mixed_updates':int(np.minimum(D,E).sum()),'chosen_columns':int(np.count_nonzero(E<D)), 'active_columns':int(np.count_nonzero(D))}
result={'permutation':perm,'calibration_tiles':[0,31],'held_tiles':[128,191],'split_note':'same frame; tile IDs disjoint but held160..191 source halo overlaps cal at y1..2; not sequence generalization','calibration_cost':'offline fixed-table derivation, not included in RTL service','metrics':{str(k):v for k,v in metrics.items()}}
(H/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
