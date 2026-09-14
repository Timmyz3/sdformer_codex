from pathlib import Path
import sys,numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent))
from prepare import readhex,writehex,native,compile_table
p=H.parent/'fixtures/real_0';q=readhex(p/'q1.hex').reshape(864,8);v=readhex(p/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
# Each T has a different channel/spatial signature; gold stays in original T order.
words=np.array([1<<((c+3*y+7*x)%10) for c in range(96) for y in range(4) for x in range(4)],np.int64).reshape(96,4,4)
origin=np.array([17,19]);e,z=native(words,q,origin);gold=np.concatenate([z@v[og*8:og*8+8].T for og in range(12)])
assert all(len(set(tuple(map(int,row)) for row in z[pp*10:pp*10+10]))==10 for pp in range(4))
cl,rep,sizes,_,_=compile_table(q);d=H/'fixtures/t_unique';d.mkdir(parents=True,exist_ok=True)
for name,a in [('source',words),('origin',origin),('q1',q),('q2',v.reshape(12,8,8).transpose(0,2,1)),('k_live',np.any(q!=0,axis=1)),('gold',gold),('class',np.sum(cl<<(6*np.arange(4)),axis=1)),('representative',rep),('ngroups',[max(sizes)])]:writehex(d/(name+'.hex'),a)
print('unique original-T signatures fixture ready')
