"""One directed existing-branch fixture, not a new layout or parameter sweep."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent
p=H/'fixtures/q2_zero_leader';p.mkdir(parents=True,exist_ok=True)
def put(name,a,bits=32):
    (p/name).write_text(''.join(f'{int(x)&((1<<bits)-1):08x}\n' for x in np.asarray(a).reshape(-1)))
q=np.zeros((8,864),np.int64);q[:,0]=[1,-1,-1,1,0,0,0,0]
v=np.zeros((12,8,8),np.int64)
for g in range(12):
    if g%4==0:v[g,:,1]=-32768 # leader0 removed, negative singleton
    if g%4==1:
        v[g,:,1]=-32768;v[g,:,2]=-32768 # both remaining members negative
    if g%4==2:
        v[g,:,1]=np.arange(8)-4;v[g,:,3]=np.arange(8)+7 # mixed negative/positive
    # g%4==3: entire effective group empty
s=np.full(1536,1023,np.int64);z=np.tile(q[:,0],(40,1));out=np.einsum('pr,gnr->gpn',z,v)
put('q1.hex',q.T,3);put('q2.hex',v.transpose(0,2,1),16);put('k_live.hex',np.any(q!=0,axis=0),1)
put('source.hex',s,10);put('origin.hex',[80,120],16);put('gold.hex',out,32)
meta=json.loads((H/'definition.json').read_text());meta['fixtures']=[c for c in meta['fixtures'] if c['name']!='q2_zero_leader']
meta['fixtures'].append({'name':'q2_zero_leader','origin':[80,120],'spikes':15360,'nonzero_z':160,
 'role':'directed original-leader removed: negative singleton, all-negative multi, mixed-sign multi, empty effective group; repeats force cache hits'})
(H/'definition.json').write_text(json.dumps(meta,indent=2)+'\n')
print('target q2_zero_leader prepared')
