from pathlib import Path
import sys,struct,json,numpy as np
H=Path(__file__).resolve().parent;B=H.parents[1];OLD=B/'psn/rtl/gp_slice'
sys.path.insert(0,str(OLD))
from prepare_cases import make
from prepare_intersection_cases import encode
D=np.asarray([[(c>>r)&1 for c in range(8)] for r in range(3)],np.int64)
A=np.asarray([[1,-2,3],[-2,1,1],[0,0,0],[1,0,-1],[-3,2,1],[1,1,1],[-1,0,2],[0,1,0],[0,-1,0],[2,1,-2]],np.int64)
# 333/334 checks the implemented branch; exact image byte crossing is336.
# The original334 derivation was an arithmetic error; keep measured data unchanged.
extra=[]
for reverse in [False,True]:
 W=np.zeros((2,384),np.int8)
 for m,n in enumerate(([334,333] if reverse else [333,334])):
  cols=list(range(n-1))+[383]
  W[m,cols]=np.resize(np.array([-1,2,-3,4],np.int8),n)
 codes=np.full((16,384),7,np.uint8);codes[:,::5]=0;codes[:,383]=7
 tau=np.resize(np.array([-1,0,1,5,-5],np.int64),(10,2))
 for route,dec,co in [('packed_class',np.eye(8,dtype=np.int64)[1:],(A@D)[:,1:]),('packed_time',D,A)]:
  extra.append(make(f'boundary_{334 if reverse else 333}_{333 if reverse else 334}_{route}',codes,W,dec,co,tau,[1.25,.75],dict(real=False,route=route,directed='image_size_boundary',identity='333/334 mixed tile density; source holes and C383 final NR4 tail')))
old=(OLD/'intersection_cases.bin').read_bytes();count=struct.unpack('<I',old[4:8])[0]
(H/'cases.bin').write_bytes(b'GPS1'+struct.pack('<I',count+len(extra))+old[8:]+b''.join(map(encode,extra)))
meta=json.loads((OLD/'intersection_cases.json').read_text());meta['cases'] += [dict(name=c['name'],program_length=len(c['program']),**c['metadata']) for c in extra];meta['total_cases']+=len(extra)
(H/'cases.json').write_text(json.dumps(meta,separators=(',',':'))+'\n')
print(f'{count} original full cases + {len(extra)} boundary cases')
