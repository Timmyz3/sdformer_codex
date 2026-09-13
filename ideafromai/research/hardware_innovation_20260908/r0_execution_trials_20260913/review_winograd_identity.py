"""Check the exact bilinear identity and phase expansion from 16 spatial basis inputs.

This is an independent numeric review of the fixed integer function; not RTL cycles.
Run after the compliant native capture is available.
"""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
d=np.load(HERE/'data_and_quality/r0_contiguous_t10.npz')
s=d['source_bits'].astype(np.int64)
w=d['weight_q16'].astype(np.int64)
bt=np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]],np.int64)
g2=np.array([[2,0,0],[1,1,1],[1,-1,1],[0,0,2]],np.int64)
at=np.array([[1,1,1,0],[0,1,-1,-1]],np.int64)
u=np.einsum('ia,ncab,jb->ncij',g2,w,g2)
v=np.einsum('ia,ptcab,jb->ptcij',bt,s,bt)
m=np.einsum('ncij,ptcij->ptnij',u,v)
y4=np.einsum('ai,ptnij,bj->ptnab',at,m,at)
gold=d['golden_accum'].astype(np.int64)
# Evaluate all sixteen spatial unit vectors. This derives the four-phase
# expanded operator independently of the hardware coefficient address layout.
basis=np.eye(16,dtype=np.int64).reshape(16,4,4)
vb=np.einsum('ia,sab,jb->sij',bt,basis,bt)
e4=np.einsum('ai,ncij,sij,bj->ncabs',at,u,vb,at)
direct=np.zeros_like(e4)
for py in range(2):
    for px in range(2):
        for ky in range(3):
            for kx in range(3):
                direct[:,:,py,px,(py+ky)*4+px+kx]=4*w[:,:,ky,kx]
result=dict(scope='Full native captured C96/N96/T10, fixed G2 integer identity; not RTL performance.',
            basis_coefficients=int(e4.size),basis_expansion_mismatches=int(np.count_nonzero(e4!=direct)),
            real_outputs=int(gold.size),native_identity_mismatches=int(np.count_nonzero(y4!=4*gold)),
            residual_mod4_nonzero=int(np.count_nonzero(y4%4)),
            transformed_weight_range=[int(u.min()),int(u.max())],
            transformed_source_range=[int(v.min()),int(v.max())],
            transformed_source_nonzero=int(np.count_nonzero(v)),
            transformed_source_elements=int(v.size))
(HERE/'winograd_identity_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if result['basis_expansion_mismatches'] or result['native_identity_mismatches']:
    raise SystemExit(1)
