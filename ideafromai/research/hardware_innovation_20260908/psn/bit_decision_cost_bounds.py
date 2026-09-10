"""Small parameter-derived bounds; no tensor forward or service simulation."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bn_state'))
from support_service_model import read_torch


def signed_width(low,high):
    width=1
    while low < -(1 << (width-1)) or high >= (1 << (width-1)):
        width+=1
    return width


def main():
    probe=json.loads((ROOT/'psn/bit_decision_probe/result.json').read_text())
    params=read_torch(ROOT/'algorithm/stage2_temporal_codes/integer_parameters.pt')
    basis=np.load(ROOT/'algorithm/stage2_temporal_codes/signed_basis.npz')
    definitions={}
    for tag in ('s2b0','s2b3'):
        q=params[f'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.{tag[-1]}.mlp.']
        B=basis[tag+'_B_int32'].astype(np.int64)
        neg=np.minimum(B,0).sum(1);pos=np.maximum(B,0).sum(1)
        table=((np.arange(64)[:,None]>>np.arange(6))&1)@B.T
        wbound=int(np.abs(q['weight_int8'].astype(np.int64)).sum(1).max())
        ubound=wbound*np.abs(B).sum(1)
        tau=q['threshold_int64'].astype(np.int64)
        Q=-((-tau)//(1 << 14))
        pending_width=signed_width(int(neg.min()+1),int(pos.max()))
        update_low=int((2*neg+1-pos).min());update_high=int((2*pos-neg).max())
        U_width=signed_width(-int(ubound.max()),int(ubound.max()))
        definitions[tag]=dict(
            source_legal_signed_bits=signed_width(-wbound,wbound),
            W_L1_bound=wbound,subset_table_entries_per_output_t=64,
            subset_table_range=[int(table.min()),int(table.max())],
            subset_table_signed_bits=signed_width(int(table.min()),int(table.max())),
            pending_static_range=[int(neg.min()+1),int(pos.max())],
            pending_static_signed_bits=pending_width,
            update_static_range=[update_low,update_high],
            update_signed_bits=signed_width(update_low,update_high),
            sign_initialization_range=[int((Q+neg[:,None]).min()),int((Q+pos[:,None]).max())],
            legal_U_abs_bound=int(ubound.max()),legal_U_signed_bits=U_width,
            tau_range=[int(tau.min()),int(tau.max())],
            resident_96_U48_bytes=576,resident_96_U_minimum_bytes=96*U_width//8,
            resident_96_pending_bytes=96*pending_width//8,
            state_saving_vs_width_matched_U_bytes=96*(U_width-pending_width)//8,
            source_six_vectors_INT15_bytes=96*6*15//8,
            source_six_vectors_INT24_bytes=96*6*24//8,
            single_copy_all_t_LUT_raw_bytes=10*64*pending_width//8,
            ninety_six_independent_ROM_copies_raw_bytes=96*10*64*pending_width//8,
            source_theta=float(q['theta_source']),output_theta=float(q['theta_output']))
    cases=[]
    for row in probe['cases']:
        tag=Path(row['capture']).stem.rsplit('_',1)[1]
        T,P,H=row['T_P_H'];vectors=T*P*(H//96)
        live_sum=6*P
        capture=ROOT/'algorithm/stage2_deployment_diverse_capture/capture'/row['capture']
        if capture.exists():
            with np.load(capture) as f:codes=f['codes']
            coordinates=basis[tag+'_coordinates_int8'][codes]
            live_sum=int(np.any(coordinates!=0,axis=1).sum())
        mac_issues=T*(H//96)*live_sum
        scalar_steps=sum((i+1)*plane['newly_decided'] for i,plane in enumerate(row['per_plane']))
        mean=scalar_steps/(T*P*H)
        compact=row['ideal_compaction_by_spatial_tile']['32']['ideal_vector_bit_steps']
        cases.append(dict(capture=row['capture'],vectors=vectors,
            support_live_word_MAC_issue_lower_bound=mac_issues,
            word_MAC_issues_per_output_vector=mac_issues/vectors,
            independent_DPU_work_lower_bound_per_output_vector=mean,
            fixed_SIMD_steps_per_output_vector=row['fixed_SIMD96_vector_bit_steps']/vectors,
            staged_tile32_steps_per_output_vector=compact/vectors,
            fixed_SIMD_vs_word_MAC_issue_ratio=row['fixed_SIMD96_vector_bit_steps']/mac_issues,
            async_work_vs_word_MAC_issue_ratio=(scalar_steps/96)/mac_issues,
            staged_tile32_vs_word_MAC_issue_ratio=compact/mac_issues,
            max_unhidden_overhead_for_staged_tile32_to_win=mac_issues/vectors-compact/vectors,
            max_unhidden_overhead_for_async_work_bound_to_win=mac_issues/vectors-mean))
    result=dict(kind='parameter/static bounds and optimistic issue counts only',
                definitions=definitions,cases=cases,
                pending_proof='With N=sum min(B,0),P=sum max(B,0), undecided integer R lies in[N+1,P]. Next R=2R-LUT+delta, delta in{-1,0}, fits[2N+1-P,2P-N]. Missing live coordinates only tighten these bounds.',
                limits='A bit step is not a word-MAC beat. Bounds exclude source gather, LUT read realization, comparator/shift feedback, threshold access, dispatch and output. Tile32 layer-wise compaction is not an absolute lower bound for an asynchronous DPU schedule. No area equality or timing claim.')
    (ROOT/'psn/bit_decision_cost_bounds.json').write_text(json.dumps(result,indent=2)+'\n')
    for tag,d in definitions.items():print(tag,d)
    for c in cases:print(c)


if __name__=='__main__':main()
