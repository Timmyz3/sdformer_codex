"""Train-only frozen K4/threshold calibration; no validation metrics are inputs."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import nnls
from reference import latent,consumer,transform,rank_threshold,MODES
H=Path(__file__).resolve().parent

def main():
 c=np.load(H/'calibration.npz');f=np.load(H/'factors.npz');co=np.load(H/'consumer_coefficients.npz')
 z=latent(c['source_bits'],f['q1']);J=c['identity_q20'].reshape(336,10,96,4).transpose(0,3,1,2)
 p=z@f['q2'].astype(np.int64).T;I=consumer(p,J,co['a_q40'],co['b_q20']);I0=consumer(np.zeros_like(p),J,co['a_q40'],co['b_q20'])
 target=np.sqrt(np.mean((I.astype(float)-I0)**2,axis=-1)).reshape(-1)
 coeff,_=nnls(np.abs(z).reshape(-1,8).astype(float),target)
 positive=coeff[coeff>0];base=float(positive.min()) if len(positive) else 1.
 shifts=np.clip(np.rint(np.log2(np.maximum(coeff,base)/base)),0,7).astype(int)
 score=(np.abs(z)<<shifts).sum(-1);nz=np.count_nonzero(z,axis=-1);taus=np.unique(((score[nz>0]+nz[nz>0]-1)//nz[nz>0]).astype(int))
 target_drop=.25*int(nz.sum());tau=min([0,*taus.tolist()],key=lambda t:(abs(int(nz[score<=t*nz].sum())-target_drop),t))
 values=z.reshape(-1,8);cb=[np.zeros(8,np.int64)]
 for _ in range(3):
  d=np.stack([(np.abs(values-v)<<shifts).sum(-1) for v in cb]).min(0);cb.append(values[int(d.argmax())].copy())
 cb=np.stack(cb)
 for _ in range(8):
  ix=(np.abs(values[:,None,:]-cb[None])<<shifts).sum(-1).argmin(-1)
  for k in range(1,4):
   if np.any(ix==k):cb[k]=np.rint(np.median(values[ix==k],axis=0)).astype(np.int64)
 diff=z[:,:,1:]-z[:,:,:-1];tscore=(np.abs(diff)<<shifts).sum(-1);nonzero=tscore[tscore>0]
 tt=int(np.quantile(nonzero,.25,method='lower')) if len(nonzero) else 0
 params=dict(frozen=True,training_frame=str(c['frame']),training_windows=len(z),budget_target=.25,shifts=shifts.tolist(),nnls_real_coefficients=coeff.tolist(),group_tau=int(tau),rank_tau=rank_threshold(z),codebook=cb.tolist(),codebook_iterations=8,codebook_size=4,residual_ranks=1,temporal_tau=tt,temporal_rank_tau=rank_threshold(diff),quantization='integers; weighted scores use left shifts; lower-index tie resolution')
 (H/'parameters.json').write_text(json.dumps(params,indent=2)+'\n')
 rows=[]
 for mode,name in MODES.items():
  zz,meta=transform(z,mode,params);pp=zz@f['q2'].astype(np.int64).T;II=consumer(pp,J,co['a_q40'],co['b_q20'])
  rows.append(dict(mode=mode,name=name,nonzero_latents=int(np.count_nonzero(zz)),Q2_MAC_vectors=int((np.count_nonzero(zz,axis=-1)*meta['refresh']).sum())*12 if mode not in (3,4) else int(np.count_nonzero(meta['residual']))*12,
   I24_MSE=float(np.mean((II.astype(float)-I)**2)),I24_max_abs_error=int(np.abs(II.astype(np.int64)-I).max()),hold_positions=int(np.count_nonzero(~meta['refresh'])),source_kind='train-only calibration; no AEE'))
 (H/'calibration_scores.json').write_text(json.dumps(dict(complete=True,original_nonzero_latents=int(np.count_nonzero(z)),arms=rows),indent=2)+'\n')
 np.savez_compressed(H/'calibration_latent.npz',z=z.astype(np.int16),identity_q20=J,p=p.astype(np.int32),I24=I)
 print(json.dumps(params),flush=True);print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
