"""Bounded real PED progressive-rank opportunity, not deployed early exit.

No training or simulator cycles. Executes true signed48 dot/RNE/sat24 for
U96->32 and V32->96 on the original preselected two4x4 anchor windows.
Oracle ranks use full outputs; they are NOT free executable predicates.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
FULL=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
sys.path.insert(0,str(FULL.parent))
from consumer_service import rne24

RANKS=(8,16,24,32)
TOLERANCES=(0.,.01,.05,.10)


def dot(w,x,shift):
    n=w.astype(np.int64)@x.astype(np.int64)
    assert np.max(np.abs(n)) < 2**47
    return rne24(n,shift)


def fee(rank,progressive):
    """Necessary issue/packed-port work; no latency/scheduling fiction.

    Static U keeps all rank groups in RF and reads the source once.
    Progressive U8 requires another source traversal per continuation.
    V expansion is deferred until rank chosen, so NO free240-register Y.
    Final V tiles H32 (80 accumulators), reads selected latent three times.
    These are ideal packed-byte counts; no address/decode or dependencies.
    """
    reads=5760*(rank//8 if progressive else 1)+20*rank*3*3
    writes=20*rank*3+20*96*3
    mac=20*(96*rank+rank*96)//8
    return dict(MAC8_issues=mac,state_read_bytes=reads,state_write_bytes=writes,
        coefficient_read_bytes=2*(96*rank+rank*96),
        extra_source_read_bytes_vs_static=5760*(rank//8-1) if progressive else 0,
        arithmetic_or_port_lower_bound=max(mac,(reads+7)//8,(writes+7)//8,(384*rank+31)//32),
        no_overlap_reservation_proxy=mac+(reads+7)//8+(writes+7)//8+12*rank,
        max_vector_accumulators=80,retained_input_bytes=5760,latent_bytes=20*rank*3)


def error(a,b):
    d=a.astype(np.float64)-b.astype(np.float64)
    denom=float(np.sum(b.astype(np.float64)**2))
    return dict(nrmse=float(np.sqrt(np.sum(d*d)/max(denom,1))),
        rms_q24=float(np.sqrt(np.mean(d*d))),max_abs_q24=int(np.max(np.abs(d))),
        bit_differences=int(np.count_nonzero(a!=b)))


def main():
    result=dict(scope=__doc__,ranks=list(RANKS),relative_L2_tolerances=list(TOLERANCES),
        numerical_contract='Signed48 dot -> original RNE+sat24 at U, at V, then original bias+sat24. No reassociation across RNE.',
        ranking='Stable descending ||U_row||2*||V_col||2 from fixed coefficients only; paired permutation is exact, no refactor or activation oracle.',
        fee_contract='8lane issue and ideal packed SR64/SW64/CR256 lower bounds, plus serial reservation proxy. Not cycles/RTL/PPA. Predicate, gather/decode, refill, gates/BN/projection excluded.',
        chosen_rank_contract='Oracle minimum rank meeting group true-output L2 tolerance across BOTH P2 positions and ALL T10/H96. Only opportunity, never an implemented acceptance rule.',
        input_read_scope='Input I24 is already in same on-chip state. Reads here are SRAM services, not repeated external input.',
        axes={})
    for axis in ('ordinary','lifting_raw'):
        with np.load(FULL/'capture'/axis/'parameters.npz') as qz:q={k:qz[k] for k in qz.files}
        with np.load(FULL/'capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz') as z:
            source=z['full_updated_I24']
        with np.load(FULL/'capture'/axis/'000_zurich_city_09_a_0001.npz') as z:
            expected={label:z[label+'_continuous_q24'] for label in ('corner','interior')}
        U,V=q['U_ped_q16'],q['V_ped_q16']
        score=np.linalg.norm(U.astype(float),axis=1)*np.linalg.norm(V.astype(float),axis=0)
        order=np.argsort(-score,kind='stable')
        record=dict(order=order.tolist(),static_fee={str(r):fee(r,False) for r in RANKS},
            progressive_fee={str(r):fee(r,True) for r in RANKS},windows={})
        for label,(oy,ox) in [('corner',(0,0)),('interior',(120,160))]:
            vals=source[:,:,oy:oy+8:2,ox:ox+8:2]
            x=vals.transpose(1,0,2,3).reshape(96,-1)
            z=dot(U,x,int(q['U_ped_exponent']))
            base=rne24(dot(V,z,int(q['V_ped_exponent']))+q['PED_bias_q24'][:,None],0)
            expected_flat=expected[label].transpose(1,0,2,3).reshape(96,-1)
            assert np.array_equal(base,expected_flat),(axis,label,'full-path capture mismatch')
            out={}
            for r in RANKS:
                selected=order[:r]
                zsel=dot(U[selected],x,int(q['U_ped_exponent']))
                y=rne24(dot(V[:,selected],zsel,int(q['V_ped_exponent']))+q['PED_bias_q24'][:,None],0)
                out[r]=y.reshape(96,10,4,4)
            assert np.array_equal(out[32].reshape(96,-1),base),'permutation altered full output'
            groups=[]
            for dy in range(4):
                for dx in (0,2):
                    ref=out[32][:,:,dy,dx:dx+2]
                    es={r:error(out[r][:,:,dy,dx:dx+2],ref) for r in RANKS}
                    selected={str(tol):next(r for r in RANKS if es[r]['nrmse']<=tol) for tol in TOLERANCES}
                    groups.append(dict(y=oy+2*dy,x=ox+2*dx,errors=es,oracle_rank=selected))
            oracle={}
            for tol in TOLERANCES:
                rr=[g['oracle_rank'][str(tol)] for g in groups]
                basefee=fee(32,False)
                costs={k:sum(fee(r,True)[k] for r in rr) for k in ['MAC8_issues','state_read_bytes','state_write_bytes','coefficient_read_bytes','no_overlap_reservation_proxy']}
                oracle[str(tol)]=dict(rank_histogram={str(r):rr.count(r) for r in RANKS},mean_rank=float(np.mean(rr)),
                    totals=costs,arithmetic_work_reduction=1-costs['MAC8_issues']/(8*basefee['MAC8_issues']),
                    reservation_proxy_reduction=1-costs['no_overlap_reservation_proxy']/(8*basefee['no_overlap_reservation_proxy']))
            record['windows'][label]=dict(output_values=base.size,full32_capture_differences=0,
                static_output_error={str(r):error(out[r].reshape(96,-1),base) for r in RANKS},
                all_group_error_ranges={str(r):[min(g['errors'][r]['nrmse'] for g in groups),max(g['errors'][r]['nrmse'] for g in groups)] for r in RANKS},
                oracle=oracle,groups=groups)
        result['axes'][axis]=record
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({a:{w:dict(static=r['static_output_error'],oracle=r['oracle']) for w,r in x['windows'].items()} for a,x in result['axes'].items()},indent=2))


if __name__=='__main__':main()
