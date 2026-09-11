"""Pro's fixed-reference spatial P2 certificate on real captured lifting inputs.

Opportunity only. No scheduling/energy result. Both point computations and all
RNE boundaries are evaluated to check the certificate; that oracle is not a
cheap implementation. Last five RNEs already folded into gates are excluded.
"""
from pathlib import Path
from collections import Counter
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from consumer_service import rne24,read_npz,LIMIT
sys.path.insert(0,str(HERE.parent/'full_chain'))
from numerical_reference import compare


def pairs(a):
    w=a.shape[-1]//2*2
    return a[:,:,:w:2],a[:,:,1:w:2]


def all8(mask):
    return mask.transpose(1,2,0).reshape(-1,12,8).all(-1)


def measure(identity,q,expected):
    value=identity.astype(np.int64).copy()
    rows=[]
    totals=Counter()
    halfstage=[]
    for layer in range(4):
        for half in range(2):
            accepted_pairs=[]
            equal_pairs=[]
            for pair,(i,j) in enumerate(q['lifting_matchings'][layer]):
                dst,src=(i,j) if half==0 else (j,i)
                a,b=value[dst].copy(),value[src].copy()
                coeff=int(q['lifting_q12'][layer,pair,half])
                numerator=(a<<12)+coeff*b
                y=rne24(numerator,12)
                # Final five nodes are only computed for the independent gate
                # oracle, not included in the proposed certificate savings.
                if layer*2+half < 7:
                    a0,a1=pairs(a); b0,b1=pairs(b)
                    n0,n1=pairs(numerator); y0,y1=pairs(y)
                    r0=n0-(y0<<12)
                    safe=(y0>-LIMIT)&(y0<LIMIT-1)
                    delta=n1-n0
                    distance=np.abs(r0+delta)
                    exact=safe&((distance<2048)|((distance==2048)&((y0&1)==0)))
                    assert np.all(y0[exact]==y1[exact])
                    radius=np.zeros(a0.shape,np.int64)
                    for d in (1,2,4,8):
                        radius=np.where(safe&(abs(coeff)*d < 2048-np.abs(r0)),d,radius)
                    equal=(a0==a1)&(b0==b1)
                    cheap=(radius>0)&(a0==a1)&(np.abs(b1-b0)<=radius)
                    assert np.all(y0[cheap]==y1[cheap])
                    nontrivial=cheap&~equal
                    accepted=equal|cheap
                    accepted_pairs.append(accepted)
                    equal_pairs.append(equal)
                    counts=dict(scalar_tests=int(equal.size),exact_inputs=int(equal.sum()),
                        cheap_certificate_total=int(cheap.sum()),cheap_nontrivial=int(nontrivial.sum()),
                        full_delta_oracle_hits=int(exact.sum()),equal_outputs=int((y0==y1).sum()),
                        eligible_reference_radius=int((radius>0).sum()),
                        SIMD8_tests=int(all8(equal).size),SIMD8_exact_inputs=int(all8(equal).sum()),
                        SIMD8_equal_or_cheap=int(all8(accepted).sum()),
                        SIMD8_increment_over_equal=int((all8(accepted)&~all8(equal)).sum()))
                    rows.append(dict(layer=layer,half=half,pair=pair,q12=coeff,
                        admits_nonzero_radius_for_some_reference=abs(coeff)<2048,**counts))
                    totals.update(counts)
                value[dst]=y
            if accepted_pairs:
                both=all8(np.logical_and.reduce(accepted_pairs))
                exact=all8(np.logical_and.reduce(equal_pairs))
                halfstage.append(dict(halfstage=layer*2+half,groups=int(both.size),
                    all5pairs_all8_equal=int(exact.sum()),all5pairs_all8_equal_or_cheap=int(both.sum()),
                    increment=int((both&~exact).sum())))
    gates=compare(value[q['source_permutation'].astype(int)],q,'source')
    assert np.array_equal(gates,expected)
    return dict(shape=list(identity.shape),adjacent_P2_groups=int(identity.shape[2]*(identity.shape[3]//2)),
        scalar_counts=dict(totals),per_node=rows,whole_halfstage_groups=halfstage,
        source_gate_mismatches=0)


def main():
    capture=HERE.parent/'full_chain/capture/lifting_raw'
    data=read_npz(capture/'000_zurich_city_09_a_0001.npz')
    q=read_npz(capture/'parameters.npz')
    kept=q['lifting_q12'].transpose(0,2,1).reshape(8,5)[:7].reshape(-1)
    result=dict(scope='One real frame, two preselected captured windows; adjacent nonoverlapping spatial P2, channels grouped contiguously H8. No frame-wide or 825-frame distribution.',
        coefficients=dict(total=40,needed_RNE=35,last5_already_folded=5,
            needed_abs_q_lt_half=int((np.abs(kept)<2048).sum()),
            all40_abs_q_lt_half=int((np.abs(q['lifting_q12'])<2048).sum())),
        baseline='Exact equal-input memo receives the same reference storage. Full-delta result is an expensive oracle, not a free certificate.',
        condition='Fixed reference y0=RNE(N0/4096), r0=N0-4096*y0; require interior y0. radius=max d in{1,2,4,8} with |q|*d<2048-|r0|; a==a0 and |b-b0|<=d. Saturation edge or failed class falls back.',
        costs_not_measured=['reference state and its reads','certificate generation/comparison','conditional CSE graph scheduling','SIMD divergence','same-privilege ordinary program'],windows={})
    for label in ('corner','interior'):
        result['windows'][label]=measure(data[label+'_I24'],q,data[label+'_sn1_gate'])
    (HERE/'rounding_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(coefficients=result['coefficients'],windows={k:v['scalar_counts'] for k,v in result['windows'].items()}),indent=2))


if __name__=='__main__':
    main()
