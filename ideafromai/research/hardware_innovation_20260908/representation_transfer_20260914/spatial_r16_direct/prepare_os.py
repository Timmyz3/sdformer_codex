from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent
original=json.loads((H/'profiles.json').read_text());a=json.loads((H/'admission.json').read_text())
assert a['nonzero_weight_vectors']==a['weight_vectors']
profiles={}
for name,p in original.items():
 d=Path(name);source=np.array([int(v,16)&1023 for v in (d/'source.hex').read_text().split()],np.uint16).reshape(96,4,4)
 origin=np.array([int(v,16) for v in (d/'origin.hex').read_text().split()],np.uint32).view(np.int32).astype(np.int64)
 for y in range(4):
  for x in range(4):
   if not(0<=origin[0]+y<240 and 0<=origin[1]+x<320):source[:,y,x]=0
 ev=((source[None]>>np.arange(10)[:,None,None,None])&1)
 bits=np.stack([ev[:,:,y:y+3,x:x+3].reshape(10,864) for y in range(2) for x in range(2)])
 updates=int(bits.sum())*12;reads=int(np.any(bits.reshape(4,10,27,32),axis=3).sum())*12
 assert updates==p['add_issues']
 state=[0]*64
 for k,v in {1:384,2:6144,3:3456,4:540,5:480,6:reads,7:updates,8:480,9:4,10:480,11:480,12:1}.items():state[k]=v
 profiles[name]=dict(base_cycles=sum(state),source_words=p['source_words']*4,weight_words=updates,local_gathers=3456,add_issues=updates,
  psum_reads=480,psum_writes=480,psum_clears=0,bitmap_reads=reads,bitmap_writes=540,base_states=state)
(H/'os_profiles.json').write_text(json.dumps(profiles,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,profiles=len(profiles),bitmap_bytes=1080,bitmap_nonempty_word_bits=270)))
