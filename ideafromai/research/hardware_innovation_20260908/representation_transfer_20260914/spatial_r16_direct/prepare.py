from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent;F=H.parent/'spatial_r16_integer';A=H.parent/'spatial_r16_rtl'
f=np.load(F/'factors.npz');w=f['expanded_int32'].astype(np.int64)
assert np.array_equal(w,np.einsum('orx,rcy->ocyx',f['q2'].astype(np.int64),f['q1'].astype(np.int64)))
lo=np.minimum(w,0).sum((1,2,3));hi=np.maximum(w,0).sum((1,2,3))
assert lo.min()>=-2**31 and hi.max()<2**31
vec=w.reshape(12,8,864).transpose(0,2,1);live=np.any(vec!=0,axis=2)
(H/'parameters').mkdir(exist_ok=True)
(H/'parameters/weight.hex').write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in vec.reshape(-1)))
def readhex(path):return np.array([int(v,16) for v in path.read_text().split()],np.uint32)
profiles={}
for name in ['small','held','disjoint']:
 lines=(A/f'{name}.txt').read_text().splitlines();(H/f'{name}.txt').write_text('\n'.join(lines)+'\n')
 for line in lines:
  if line in profiles:continue
  d=Path(line);source=readhex(d/'source.hex').reshape(96,4,4)&1023
  origin=readhex(d/'origin.hex').view(np.int32).astype(np.int64)
  inside=0
  for y in range(4):
   for x in range(4):
    if 0<=origin[0]+y<240 and 0<=origin[1]+x<320:inside+=1
    else:source[:,y,x]=0
  ev=((source[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
  p=np.zeros((10,96,2,2),np.int64);updates=reads=visits=0
  for c in range(96):
   for y in range(3):
    for x in range(3):
     k=c*9+y*3+x;active=ev[:,c,y:y+2,x:x+2];count=int(active.sum())
     visits+=12 if count else 1
     if count:
      reads+=int(live[:,k].sum());updates+=count*int(live[:,k].sum())
     p+=active[:,None]*w[None,:,c,y,x,None,None]
  raw=p.transpose(1,2,3,0).reshape(12,8,4,10).transpose(0,2,3,1).reshape(480,8)
  assert np.array_equal(raw.astype(np.int32).view(np.uint32).ravel(),readhex(d/'gold.hex'))
  state=[0]*64
  for i,v in {1:480,2:96,3:1536,4:864,5:visits,6:updates,7:updates,8:864,9:480,10:480,11:1}.items():state[i]=v
  profiles[line]=dict(base_cycles=sum(state),source_words=96*inside,weight_words=reads,local_gathers=864,add_issues=updates,
   psum_reads=updates+480,psum_writes=updates+480,psum_clears=480,base_states=state)
(H/'profiles.json').write_text(json.dumps(profiles,separators=(',',':'))+'\n')
(H/'admission.json').write_text(json.dumps(dict(passed=True,weight_shape=list(w.shape),weight_range=[int(w.min()),int(w.max())],
 prefix_lower=lo.tolist(),prefix_upper=hi.tolist(),all_binary_prefix_abs=int(np.maximum(-lo,hi).max()),weight_vectors=int(live.size),
 nonzero_weight_vectors=int(live.sum()),weight_payload_bytes=int(w.size*4),weight_support_bytes=int(live.size//8),
 full_output_checked_fixtures=len(profiles),oracle='actual source bits x expanded integer W, checked against factor fixture full gold'),separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,fixtures=len(profiles),weight_payload_bytes=int(w.size*4),prefix_abs=int(np.maximum(-lo,hi).max()))))
