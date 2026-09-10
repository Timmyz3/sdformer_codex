"""Stronger old-encoding control: only the invalid side needs recovery.

Same B32K4 float64 numerical proxy. The repair mask uses stored predicted bits,
kept positions and side extrema, never the oracle locations of actual bit flips.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='4'
import argparse
import json
from pathlib import Path
import sys
import numpy as np

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'handoff_reconciliation_20260906/scripts'))
from screen_repair_sectors import load_sources,read_checkpoint,CKPT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True)
    out=ap.parse_args().output_dir;out.mkdir(parents=True,exist_ok=False)
    source,_=load_sources();state=read_checkpoint(CKPT)['model_state_dict']
    old=json.loads((BASE.parent/'mechanism_rebuild_gh_20260906/records/threshold_packet_sample0_screen.json').read_text())
    layers=[]
    for stage in (0,3):
        pre=f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.'
        spec,S=source[pre+'fc1'];N,C=S.shape;H=spec['output_channels'];T=10;P=N//T;B=32;K=4
        W=state[pre+'fc1.weight'].astype(np.float64)
        theta_s=float(state[pre+'sn1.spiking_neuron.thresh'])
        gamma=state[pre+'bn1.norm_layer.weight'].astype(np.float64)
        beta=state[pre+'bn1.norm_layer.bias'].astype(np.float64)
        A=state[pre+'sn2.spiking_neuron.weight'].astype(np.float64)
        bias=state[pre+'sn2.spiking_neuron.bias'].astype(np.float64).reshape(T,1)
        center=state[pre+'sn2.spiking_neuron.center'].astype(np.float64).reshape(T,1)
        theta=float(state[pre+'sn2.spiking_neuron.thresh']);R=A.sum(1).reshape(T,1)
        nb=(P+B-1)//B;sizes=np.minimum(B,P-np.arange(nb)*B)
        valid=np.arange(B)[None,None,:,None]<sizes[None,:,None,None]
        unknown=np.zeros((T,P,H),dtype=bool);fails=known_errors=0
        off_failed=on_failed=0
        for lo in range(0,H,32):
            hi=min(H,lo+32);hc=hi-lo
            Y=(S.astype(np.float64)@W[lo:hi].T*theta_s).reshape(T,P,hc)
            mu=Y.mean((0,1));std=np.sqrt(((Y-mu)**2).mean((0,1))+1e-5)
            direction=np.sign(gamma[lo:hi])
            V=np.einsum('ts,sph->tph',A,Y,optimize=True)*direction
            tau=(mu*R+std/gamma[lo:hi]*(theta+center-bias-beta[lo:hi]*R))*direction
            ends=np.minimum((np.arange(nb)+1)*B,P)-1
            cs=np.cumsum(Y.sum(0),0);cq=np.cumsum((Y*Y).sum(0),0)
            pmu=cs[ends]/((ends+1)*T)[:,None]
            pvar=np.maximum(cq[ends]/((ends+1)*T)[:,None]-pmu*pmu,0)
            ptau=(pmu[None,:,:]*R[:,None,:]+np.sqrt(pvar+1e-5)[None,:,:]/gamma[lo:hi]
                  *(theta+center[:,None,:]-bias[:,None,:]-beta[lo:hi]*R[:,None,:]))*direction
            padded=np.full((T,nb*B,hc),np.nan);padded[:,:P]=V
            packed=padded.reshape(T,nb,B,hc)
            pred=packed>=ptau[:,:,None,:]
            truth=packed>=tau[:,None,None,:]
            order=np.argsort(np.where(valid,np.abs(packed-ptau[:,:,None,:]),np.inf),axis=2,kind='stable')
            keep=np.zeros_like(pred);np.put_along_axis(keep,order[:,:,:K,:],True,axis=2)
            dropped=valid&~keep
            a=np.where(dropped&~pred,packed,-np.inf).max(2)
            b=np.where(dropped&pred,packed,np.inf).min(2)
            bad_off=tau[:,None,:]<=a;bad_on=tau[:,None,:]>b
            needs=dropped&((~pred&bad_off[:,:,None,:])|(pred&bad_on[:,:,None,:]))
            unknown[:,:,lo:hi]=needs.reshape(T,nb*B,hc)[:,:P]
            fails+=int((bad_off|bad_on).sum())
            off_failed+=int(bad_off.sum());on_failed+=int(bad_on.sum())
            known_errors+=int(((pred!=truth)&dropped&~needs).sum())
        reference=next(l for l in old['layers'] if f'layers.{stage}.' in l['module'])
        ref=next(r for r in reference['rows'] if r['B']==B and r['K']==K)
        assert fails==ref['failed_packets'] and known_errors==0
        touched=unknown.any(0)
        spikes=S.reshape(T,P,C).sum((0,2),dtype=np.int64)
        adds=int(np.dot(touched.sum(1,dtype=np.int64),spikes))
        psn=int(sum(int(unknown[t].sum())*int((A[t]!=0).sum()) for t in range(T)))
        # The two measured A matrices are dense. A zero-aware variant would only
        # need the union of input time rows used by unresolved output consumers.
        assert (A!=0).all()
        np.packbits(unknown,axis=2,bitorder='little').tofile(out/f'stage{stage}_unknown.le.bitpack')
        consumer=(unknown.astype(np.uint16)*(1<<np.arange(T,dtype=np.uint16))[:,None,None]).sum(0,dtype=np.uint16)
        np.save(out/f'stage{stage}_consumer_mask.npy',consumer,allow_pickle=False)
        row=dict(stage=stage,shape_T_P_H=[T,P,H],C=C,failed_packets=fails,
                 off_side_failed=off_failed,on_side_failed=on_failed,
                 known_bit_mismatches=known_errors,unknown_outputs=int(unknown.sum()),
                 unknown_fraction=float(unknown.mean()),unresolved_ph=int(touched.sum()),
                 shared_Y_repair_binary_ADD_terms=adds,shared_Y_repair_PSN_MAC_terms=psn,
                 full_FC1_binary_ADD_terms=int(S.sum(dtype=np.int64))*H,
                 full_PSN_MAC_terms=P*H*int((A!=0).sum()),
                 point64_packet_bytes_aligned=T*nb*H*64,
                 point32_packet_bytes_nominal_not_evaluated=T*nb*H*32,
                 legacy_q4_repair_FC1_fraction=[.08668437157959301,.29438274682621374][stage==3])
        layers.append(row);print(json.dumps(row),flush=True)
    result=dict(scope='same two-layer sample0, B32K4; old point-value encoding with stronger causal directional recovery',
                mechanism='stored predicted bits and keep positions select only dropped values on the invalid side',
                limitations=['Float64 proxy only; no interval-input enclosure producer or original FP32 closure.',
                             'Operation counts omit sparse index/queue/port/patch and threshold predictor/topK costs.',
                             '32bit payload figure is a nominal format comparison, not tested numerical storage.'],layers=layers)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
