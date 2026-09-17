import struct
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parent
d=np.load(p/'source_cases.npz')
with (p/'source.bin').open('wb') as f:
    f.write(d['A_q12'].astype('<i2').tobytes())
    f.write(d['threshold_q28'].astype('<i8').tobytes())
    f.write(d['D'].astype('u1').tobytes())
    for name in ['prefix_tables.npz','prefix_tables_entropy.npz']:
        q=np.load(p/name)
        for key in ['code_nodes64','class_nodes64']:
            v=q[key];f.write(struct.pack('<I',len(v)));f.write(v.astype('<u8').tobytes())
        f.write(q['roots'].astype('<u2').tobytes())
        f.write(q['response_canonical_code'].astype('u1').tobytes())
        rank=np.full((6,16),15,dtype='u1')
        for g in range(6):
            rank[g,q[f'variables_g{g}']]=np.arange(len(q[f'variables_g{g}']),dtype='u1')
        f.write(rank.tobytes())
    cases=[]
    for frame in range(2):
        for pos in range(32):cases.append((f'train{frame}_p{pos}',1,d['X_q16'][frame,:,pos,:].T))
    cases+=[('diagnostic_zero',0,np.zeros((96,10),dtype=np.int32)),
            ('diagnostic_positive_extreme',0,np.full((96,10),(1<<23)-1,dtype=np.int32)),
            ('diagnostic_signed_extreme',0,np.tile(np.array([-1<<23,(1<<23)-1]*5,dtype=np.int32),(96,1)))]
    f.write(struct.pack('<I',len(cases)))
    for name,real,x in cases:
        b=name.encode();f.write(struct.pack('<II',real,len(b)));f.write(b);f.write(x.astype('<i4').tobytes())
print('exported',len(cases),'cases, 2 dictionary-only order policies')
