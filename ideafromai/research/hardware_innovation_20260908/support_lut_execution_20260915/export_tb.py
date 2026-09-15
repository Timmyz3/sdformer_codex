from pathlib import Path
import struct
import numpy as np

p = Path(__file__).resolve().parent
d = np.load(p / 'cases.npz')
fields = [('S','u1'),('W','i1'),('tau','<i8'),('positive_gain','u1'),
          ('constant_channels','u1'),('constant_gate','u1'),('Y','<i4'),('U','<i8'),('gold','u1')]
cases = [{key:d[key][i] for key in [x[0] for x in fields]+['is_real','hblock','case_name']}
         for i in range(len(d['S']))]
# Two fixed algorithm probes and a protocol-only pattern; never pooled with
# unchanged real cases or labelled network accuracy. All answers recomputed.
from probe_response_projection import bounded_zero_sum
W0=cases[0]['W'].astype(np.int64)
projection=W0.copy();pruned=W0.copy();holes=W0.copy()
for g in range(6):
    cols=16*g+np.flatnonzero(d['D'][g,1])
    for h in range(96): projection[cols,h]=bounded_zero_sum(W0[cols,h])[1]
    candidates=[k for k in range(1,16) if np.all(d['D'][g,1:]<=d['D'][g,k],axis=1).sum()==1]
    for h0 in range(0,96,12):
        k=min(candidates,key=lambda k:float(np.square(W0[16*g+np.flatnonzero(d['D'][g,k]),h0:h0+12]).sum()))
        pruned[16*g+np.flatnonzero(d['D'][g,k]),h0:h0+12]=0
        if (h0//12+g)%3==0:holes[g*16:g*16+16,h0:h0+12]=0
holes[80:96]=0
variants=[('variant_integer_projection',projection),('variant_matched_W_prune',pruned),('diagnostic_directory_holes',holes)]
variants.append(('variant_response_classes',np.load(p/'response_class_W.npy')[:96].T.astype(np.int64)))
for name,w in variants:
    c=dict(cases[0]);c['case_name']=name;c['is_real']=0;c['W']=w.astype(np.int8)
    c['Y']=np.einsum('ptc,ch->pth',c['S'].astype(np.int64),w)
    c['U']=np.einsum('ts,psh->pth',d['A'].astype(np.int64),c['Y'])
    c['gold']=np.where(c['constant_channels'][None,None,:],c['constant_gate'][None,:,:],
        np.where(c['positive_gain'][None,None,:],c['U']>=c['tau'][None,:,:],c['U']<=c['tau'][None,:,:]))
    cases.append(c)
with (p / 'cases.bin').open('wb') as f:
    f.write(struct.pack('<I', len(cases)))
    f.write(d['D'].astype('u1').tobytes())
    f.write(d['A'].astype('<i2').tobytes())
    for c in cases:
        name = str(c['case_name']).encode()
        f.write(struct.pack('<III', int(c['is_real']), int(c['hblock']), len(name)))
        f.write(name)
        for key, dtype in fields:
            f.write(c[key].astype(dtype).tobytes())
print('exported', len(cases), 'cases; 32 unchanged real, remaining labelled variants/diagnostics')
