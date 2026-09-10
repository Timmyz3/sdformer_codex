"""Paid single-U fallback: retain Y, recompute prefix for failed predictions.

Only re-evaluates the saved local FP32 predictor on existing validation Y.
No training/network forward. Parameter-space fixed-point admission is separate.
"""
from pathlib import Path
import json
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import numpy as np
import torch
from train_local import load_inputs


def main():
    torch.set_num_threads(4)
    _,splits,_,_,_,_=load_inputs()
    data=splits['valid']
    result=dict(
        scope='existing local train16/valid4 FP32-norm1 students, including train-selected ordinary prefixes; four frames/64 P4/allH96/T10. Not integer gate replay or whole layer service.',
        formula='prefix_all + sum_(not_accepted gate) nnz(A[row,:]); prefix-only rows are already exact and may bypass needless fallback',
        ideal_retained_C_formula='prefix_all + sum_(not_accepted gate) nnz(A[row,tail])',
        state='retain10Y plus one working U plus40B accept and40B output bitmaps; no hidden C bank. FP32 Y payload1280B, integer Y24 payload960B; reserved workingU48B. Other operand registers/source/parameters remain paid separately.',
        limits=['scalar nonzero-coefficient terms, not MAC cycles','ordinary zero-raw-Y/BN compensation and constant-matrix compilation are not measured on this FP32 capture','no trainedFP C32 integer bound is assumed'],axes={})
    for folder in ('local_train16','word_train16','optimized_prefix_train16'):
        base=HERE.parent/folder
        src=json.loads((base/'result.json').read_text())
        for axis,record in src['axes'].items():
            z=np.load(base/(axis+'.npz'))
            a=torch.from_numpy(z['temporal_q14'].astype(np.float32)/16384)
            b=torch.from_numpy(z['temporal_bias'])
            mean=torch.from_numpy(z['mean']);cov=torch.from_numpy(z['covariance'])
            gamma=torch.from_numpy(z['gamma'])
            theta=float(z['theta_output'])
            prefix=z['prefix'].tolist()
            known=torch.zeros(10);known[prefix]=1
            remaining=a*(1-known)
            offset=mean@remaining.T+b-theta
            sigma=torch.einsum('ts,csu,tu->ct',remaining,cov,remaining).clamp_min(1e-6).sqrt()
            mask=a!=0
            prefix_n=mask[:,prefix].sum(-1)
            full_n=mask.sum(-1);tail_n=full_n-prefix_n
            saved=np.load(base/(axis+'_valid_decisions.npz'))
            counts=dict(gates=0,accepted=0,need_column_differences=0,gate_differences=0,
                        prefix_all=0,ideal_retained_C_terms=0,
                        retained_Y_recompute_terms=0,retained_Y_skip_prefix_only_fallback_terms=0,
                        full_terms=0,unaccepted_gates=0,unaccepted_with_tail=0)
            for start in range(0,len(data['y']),128):
                sl=slice(start,start+128)
                y=data['y'][sl];ch=data['channels'][sl]
                full=y@a.T+b-theta
                pred=(y*known)@a.T+offset[ch][:,:,None,:]
                radius=sigma[ch][:,:,None,:]*gamma
                accepted=pred.abs()>=radius
                failed=~accepted
                gate=torch.where(accepted,pred>=0,full>=0)
                by_lane=(failed[...,None]*mask).amax(-2)
                by_lane=by_lane*(1-known)+known
                need=by_lane.amax((1,2)).bool()
                counts['gate_differences']+=int(np.count_nonzero(gate.numpy()!=saved['gate'][sl]))
                counts['need_column_differences']+=int(np.count_nonzero(need.numpy()!=saved['need_column'][sl]))
                counts['gates']+=gate.numel();counts['accepted']+=int(accepted.sum())
                counts['unaccepted_gates']+=int(failed.sum())
                counts['unaccepted_with_tail']+=int((failed&(tail_n>0)).sum())
                pre=gate.numel()//10*int(prefix_n.sum())
                counts['prefix_all']+=pre
                counts['full_terms']+=gate.numel()//10*int(full_n.sum())
                counts['ideal_retained_C_terms']+=pre+int((failed*tail_n).sum())
                counts['retained_Y_recompute_terms']+=pre+int((failed*full_n).sum())
                counts['retained_Y_skip_prefix_only_fallback_terms']+=pre+int((failed*(tail_n>0)*full_n).sum())
            counts['ideal_term_difference_from_training_report']=counts['ideal_retained_C_terms']-record['valid']['early_psn_terms']
            counts['prefix']=prefix
            counts['extra_recompute_terms']=counts['retained_Y_recompute_terms']-counts['ideal_retained_C_terms']
            counts['recompute_over_ideal']=counts['retained_Y_recompute_terms']/counts['ideal_retained_C_terms']
            counts['recompute_over_full']=counts['retained_Y_recompute_terms']/counts['full_terms']
            result['axes'][folder+'/'+axis]=counts
    (HERE/'recompute_psn_costs.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result['axes'],ensure_ascii=False))


if __name__=='__main__':main()
