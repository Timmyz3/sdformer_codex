"""Real lifting half-step RNE-margin reuse with fixed spatial P2 references."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[1]
Q=4096
LIMIT=1<<23


def rne(a):
    q,r=np.divmod(a,Q)
    return q+((2*r>Q)|((2*r==Q)&((q&1)!=0)))


def halfstep(a,b,coef,where):
    exact=Q*a+coef*b
    rounded=rne(exact)
    out=np.clip(rounded,-LIMIT,LIMIT-1)
    # Split fixed adjacent x pairs, retain physical groups of8 channels.
    width=a.shape[2]//2*2
    reshape=lambda x:x[:,:,:width].reshape(12,8,x.shape[1],width//2,2).transpose(0,2,3,1,4)
    aa,bb,oo,rr,nn=map(reshape,(a,b,out,rounded,exact))
    a0,a1=aa[...,0],aa[...,1];b0,b1=bb[...,0],bb[...,1]
    y0,y1=oo[...,0],oo[...,1]
    residue=nn[...,0]-Q*rr[...,0]
    margin=Q//2-np.abs(residue)
    legal=(rr[...,0]==y0)&(np.abs(residue)!=Q//2)
    level=np.zeros_like(a0)
    for d in (1,2,4,8):
        level=np.where(legal&(abs(coef)*d<margin),d,level)
    input_equal=(a1==a0)&(b1==b0)
    zero=b1==0
    certificate=legal&(level>0)&(a1==a0)&(np.abs(b1-b0)<=level)
    vector=certificate.all(-1)
    # All lanes of a vector can already use a simple equality/zero bypass
    # even when they take different trivial paths. Do not count those as X.
    trivial=(input_equal|zero)
    novel=certificate&(~trivial)
    novel_vector=vector&(~trivial.all(-1))
    actual=np.empty_like(y1)
    actual[vector]=y0[vector]
    observed=Q*a1[~vector]+coef*b1[~vector]
    actual[~vector]=np.clip(rne(observed),-LIMIT,LIMIT-1)
    assert np.array_equal(actual,y1)
    assert np.array_equal(y0[certificate],y1[certificate])
    return out,dict(**where,coefficient=int(coef),static_nontrivial_possible=bool(0<abs(coef)<Q//2),
        lanes=int(certificate.size),vectors=int(vector.size),
        input_equal_lanes=int(input_equal.sum()),zero_lanes=int(zero.sum()),
        a_equal_lanes=int((a1==a0).sum()),small_b_change_lanes=int((np.abs(b1-b0)<=8).sum()),
        certificate_lanes=int(certificate.sum()),new_lanes=int(novel.sum()),
        certificate_vectors=int(vector.sum()),new_vectors=int(novel_vector.sum()),
        exact_input_vectors=int(input_equal.all(-1).sum()),zero_vectors=int(zero.all(-1).sum()),
        reference_saturated_lanes=int((rr[...,0]!=y0).sum()),
        reference_tie_lanes=int((np.abs(residue)==Q//2).sum()),
        max_reusable_level_counts={str(d):int(np.count_nonzero(level==d)) for d in (0,1,2,4,8)},
        replay_values=int(actual.size),replay_differences=0)


def main():
    capture=OPEN/'stage_20260912/algorithm/hardware_exports/lifting_raw'
    with np.load(capture/'deployed_constants.npz') as z:q={k:z[k] for k in z.files}
    with np.load(capture/'000_zurich_city_09_a_0001.npz') as z:data={k:z[k] for k in z.files}
    matchings=q['lifting_matchings'];coeff=q['lifting_q12']
    report=dict(scope=__doc__,same_function=True,AEE_inherited=False,RTL_cycles=False,
        coverage='Independent reproduction of 20260911 rounding_opportunity.py on the same strict equal-a/horizontal-P2 interface; not a new interface.',
        input='Current lifting R24+onepass actual I24 source halo, fixed neighboring horizontal pixels; not successive video frames.',
        coefficient_values=coeff.tolist(),static_nonzero_eligible=int(np.count_nonzero((np.abs(coeff)>0)&(np.abs(coeff)<Q//2))),
        total_coefficients=int(coeff.size),levels=[1,2,4,8],windows={})
    for label in ('corner','interior'):
        value=data[label+'_I24'].astype(np.int64).copy();rows=[]
        for layer in range(4):
            pairs=matchings[layer]
            for half in (0,1):
                source=1-half
                for pair,(first,second) in enumerate(pairs):
                    target=(first,second)[half];other=(first,second)[source]
                    out,row=halfstep(value[target],value[other],int(coeff[layer,pair,half]),
                        dict(layer=layer,half=half,pair=pair,materialized_norm24=(layer,half)!=(3,1)))
                    value[target]=out;rows.append(row)
        src=value[q['source_permutation']]
        shape=(10,1,1,1)
        gate=np.where(q['source_direction'].reshape(shape)>0,
            src>=q['source_threshold'].reshape(shape),src<=q['source_threshold'].reshape(shape))
        constants=q['source_constant'].reshape(shape)
        gate=np.where(constants>=0,constants.astype(bool),gate)
        gold=data[label+'_sn1_gate'].astype(bool)
        assert np.array_equal(gate,gold),label
        fields=('lanes','vectors','certificate_lanes','certificate_vectors','new_lanes','new_vectors','replay_values','replay_differences')
        totals={name:{f:sum(r[f] for r in rows if (r['materialized_norm24'] or name=='all40')) for f in fields}
            for name in ('all40','materialized35')}
        for v in totals.values():
            v['new_lane_fraction']=v['new_lanes']/v['lanes']
            v['new_vector_fraction']=v['new_vectors']/v['vectors']
        report['windows'][label]=dict(source_gate_values=int(gold.size),source_gate_differences=0,
            rows=rows,totals=totals)
    report['state_requirement']=dict(per_halfstep_vector_reference_bytes=3*8*3,
        per_halfstep_vector_grade_bits=8*3,materialized_halfsteps=35,
        all35_reference_and_grade_bytes=35*(3*8*3+3),
        meaning='Packed lower bound: a0,b0,y0 each8 signed24, plus3-bit grade per lane. No actual allocation in the48-bit RF is implemented. Extra comparisons, packing, writes and reads are not free.')
    report['decision']='Stop only this strict equal-a/horizontal-P2 margin certificate if novel full-SIMD matches are absent. No family-wide rejection; no performance inferred from lane hits.'
    (HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(static_eligible=report['static_nonzero_eligible'],windows={k:v['totals'] for k,v in report['windows'].items()})))


if __name__=='__main__':main()
