"""Independent CPU audit only: no network, GPU, EDA or original-file writes."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE/'papers_lineage_20260916/copy_remaining'))
import remaining_ops

def exact_shipped_logic(y, a, tau):
    y, a, tau = (torch.from_numpy(x) for x in (y,a,tau))
    T, N, H = y.shape
    used = torch.empty((T,N,H), dtype=torch.int64)
    gate = torch.empty((T,N,H), dtype=torch.bool)
    ym = y.abs().amax(dim=0)
    for t in range(T):
        order = torch.argsort(a[t].abs(), descending=True)
        aa = a[t].index_select(0,order)
        av = aa.abs()
        r0 = av.sum()*ym
        c0, m0 = r0 < tau[t], -r0 >= tau[t]
        sums = (y.index_select(0,order)*aa[:,None,None]).cumsum(0)
        suf = av.flip(0).cumsum(0).flip(0)-av
        r = suf[:,None,None]*ym
        cannot, must = sums+r < tau[t], sums-r >= tau[t]
        dec = cannot|must
        first = dec.long().argmax(dim=0)
        has = dec.any(dim=0)
        used[t] = torch.where(c0|m0,0,torch.where(has,first+1,T))
        pick = must.gather(0,first[None])[0]
        gate[t] = torch.where(c0|m0,m0,torch.where(has,pick,sums[-1]>=tau[t]))
    return used.numpy(), gate.numpy()

def main():
    eps = np.float32(2**-25)
    y = np.array([0,1],np.float32).reshape(2,1,1)
    a = np.array([[1,eps],[1,eps]],np.float32)
    tau = np.full((2,1),eps,np.float32)
    used, gate = exact_shipped_logic(y,a,tau)
    true = (a@y.reshape(2,1)).reshape(2,1,1)>=tau[:,None,:]
    wrong = int(np.count_nonzero(gate!=true))
    assert wrong == 2 and np.all(used==1)
    original_extra = remaining_ops.psn_scrooge_extra(torch.from_numpy(y),torch.from_numpy(a),torch.from_numpy(tau))
    result = {'scope':'Independent CPU audit, two prior archived sample traces, not the 10-frame run and not RTL cycles.',
      'fp32_counterexample': {'A':a.tolist(),'Y':y.flatten().tolist(),'tau':tau.tolist(),
         'rest_after_first':float(np.float32(np.float32(1)+eps)-np.float32(1)),
         'true_remaining':float(eps),'used_terms':used.flatten().tolist(),
         'wrong_gates':wrong,'shipped_extra':original_extra},'traces':[]}
    for sid in [0,10]:
        z=np.load(BASE/f'bn_state/trace_s{sid}_stage0.npz')
        s=np.unpackbits(z['source_packed'],axis=1,bitorder='little')[:,:96]
        W=z['W'].astype(np.float64); A=z['A'].astype(np.float32)
        T=A.shape[0]; H=W.shape[0]; P=s.shape[0]//T
        k=z['k'].astype(np.float64); G=z['G'].astype(np.float64)
        mu=W@k/len(s)
        var=np.maximum(np.sum((W@G)*W,axis=1)/len(s)-mu**2,0)
        idx=np.arange(0,P,max(1,P//256))[:256]
        raw=s.reshape(T,P,96)[:,idx].astype(np.float64)@W.T
        y=(z['gamma']*(raw-mu)/np.sqrt(var+1e-5)+z['beta']).astype(np.float32)
        # Existing bn_state numeric receipt locks theta=1 and zero center here.
        assert np.all(z['center']==0)
        tau=np.broadcast_to(1-z['bias'].astype(np.float32).reshape(T,1),(T,H)).copy()
        used, pred=exact_shipped_logic(y,A,tau)
        original=remaining_ops.psn_scrooge_extra(torch.from_numpy(y),torch.from_numpy(A),torch.from_numpy(tau))
        independent=1-used.mean()/T
        assert abs(original-independent)<1e-10
        full=(A@y.reshape(T,-1)).reshape(y.shape)>=tau[:,None,:]
        grp_h=1-used.reshape(T,len(idx),H//8,8).max(axis=3).mean()/T
        grp_t=1-used.max(axis=0).mean()/T
        grp_th=1-used.reshape(T,len(idx),H//8,8).max(axis=(0,3)).mean()/T
        groups=s.reshape(len(s),24,4).any(axis=2).sum()
        nnz=s.sum()
        result['traces'].append({'sample':sid,'source_shape':[T,P,96],'stored_source_dtype':str(z['source_packed'].dtype),
          'stored_source_shape':list(z['source_packed'].shape),'sampled_P':len(idx),
          'lut_count_extra_full_source':1-float(groups)/int(nnz),
          'psn_independent_mac_skip':independent,'shipped_function_skip':original,
          'H8_same_t_lockstep_term_skip':grp_h,'T10_same_h_lockstep_term_skip':grp_t,
          'T10_H8_lockstep_term_skip':grp_th,'fp32_gate_mismatch_in_this_cpu_probe':int(np.count_nonzero(pred!=full)),
          'num_decisions':int(used.size),'used_hist':np.bincount(used.ravel(),minlength=T+1).tolist(),
          'limitations':'Reconstructed post-liveBN using archived all-domain FP64 moments then FP32; not cuDNN golden. Grouped results are term opportunities, not a scheduler or cycle model.'})
    (HERE/'leftover_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
