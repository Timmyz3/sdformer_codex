"""One fixed train-only integer Kronecker fit; validation is never an input."""
from pathlib import Path
import numpy as np
import json
H=Path(__file__).resolve().parent
OLD=H.parents[1]/'fusion_ten_trials_20260914'/'algorithm_sparse'
f=np.load(OLD/'factors.npz');co=np.load(OLD/'consumer_coefficients.npz')
z=np.load(OLD/'calibration_latent.npz')['z'].reshape(-1,8).astype(float)
W=f['q2'].astype(float).reshape(12,8,4,2)
cov=z.T@z/len(z)
C=cov.reshape(4,2,4,2)
rw=(co['a_q40'].astype(float)/np.mean(co['a_q40']))**2
rw=rw.reshape(12,8)
mat=W.transpose(0,2,1,3).reshape(48,16)
u,s,v=np.linalg.svd(mat,full_matrices=False)
b0=v[0].reshape(8,2);scale=31/np.max(np.abs(b0))
B=np.rint(b0*scale).astype(np.int64)
A=np.clip(np.rint((u[:,0]*s[0]/scale).reshape(12,4)),-4096,4095).astype(np.int64)
history=[]
for step in range(8):
 for o in range(12):
  lhs=np.einsum('l,lj,lp,kjqp->kq',rw[o],B,B,C)
  rhs=np.einsum('l,lj,kjqp,lqp->k',rw[o],B,C,W[o])
  A[o]=np.clip(np.rint(np.linalg.lstsq(lhs+1e-7*np.eye(4),rhs,rcond=None)[0]),-4096,4095)
 for l in range(8):
  lhs=np.einsum('o,ok,oq,kjqp->jp',rw[:,l],A,A,C)
  rhs=np.einsum('o,ok,kjqp,oqp->j',rw[:,l],A,C,W[:,l])
  B[l]=np.clip(np.rint(np.linalg.lstsq(lhs+1e-7*np.eye(2),rhs,rcond=None)[0]),-32,31)
 E=np.einsum('ok,lj->olkj',A,B)-W
 ec=E.reshape(96,8)
 error=np.einsum('ni,ij,nj->n',ec,cov,ec)*rw.reshape(-1)
 history.append(float(error.sum()))
ranked=np.argsort(-error.reshape(12,8).sum(1),kind='stable')
escape=np.zeros(12,dtype=bool);escape[ranked[:2]]=True
Wk=np.einsum('ok,lj->olkj',A,B).reshape(96,8)
Wh=Wk.copy();Wh.reshape(12,8,8)[escape]=f['q2'].reshape(12,8,8)[escape]
q1=f['q1'].astype(np.int64);lo=np.minimum(q1,0).sum(1);hi=np.maximum(q1,0).sum(1)
Sc=[]
for k in range(4):
 Sc.append([sum(max(abs(int(b)*int(lo[k*2+j])),abs(int(b)*int(hi[k*2+j]))) for j,b in enumerate(B[l])) for l in range(8)])
Sbound=int(np.max(Sc));pbound=int(np.max(np.abs(Wh)@np.maximum(-lo,hi)))
assert Sbound<2**18 and pbound<2**31 and np.abs(Wh).max()<2**18
data={k:f[k] for k in f.files};data.update(A=A.astype(np.int16),B=B.astype(np.int8),escape=escape,W_kron=Wk.astype(np.int32),W_hybrid=Wh.astype(np.int32))
np.savez_compressed(H/'fitted.npz',**data)
report=dict(training_frame='thun_00_a_0002.npy',training_windows=336,ALS_steps=8,objective_history=history,A=A.tolist(),B=B.tolist(),escape_groups=np.where(escape)[0].tolist(),weighted_group_errors=error.reshape(12,8).sum(1).tolist(),S_abs_bound=Sbound,p_abs_bound=pbound,expanded_weight_abs_max=int(np.abs(Wh).max()),AEE='not yet measured',RTL_cycles='not yet measured')
(H/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
