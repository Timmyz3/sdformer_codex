"""Bounded actual-input motion-reference residual probe, not a cycle model."""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
FULL=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
sys.path.insert(0,str(FULL))
from numerical_reference import tf32_round

OFFSETS=sorted([(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)],key=lambda d:(abs(d[0])+abs(d[1]),d))
GUIDE=np.arange(0,96,6)


def bits(key):
    while key:
        bit=key&-key
        yield bit.bit_length()-1
        key-=bit


def main():
    result=dict(evidence='Actual source, signed exact dyadic U32 payload replay and operation/byte opportunities; not port cycles, full layer, AEE or PPA.',
        identity='Existing coarse-head ordinary/lifting students, one frozen captured frame; T10 neuron slices after noncausal source completion, not ten consecutive video frames.',
        full_K=864,guide_channels=GUIDE.tolist(),offsets=OFFSETS,
        selector='Minimum centre-guide Hamming distance, whole-tile common displacement; zero offset wins ties. No network output flow or gold Z used.',
        fallback='After current/reference support comparison choose fresh if residual_count+one heuristic parent-vector slot penalty is not lower than current_count. Actual parent is a COPY/LOAD, not an extra ADD. Both difference arms get the same privilege.',
        arms=['fresh_GP','same_position_delta','guide_displacement_delta','full_support_displacement_oracle'],axes={})
    for axis in ['ordinary','lifting_raw']:
        folder=FULL/'capture_full_producers'/axis
        with np.load(folder/'live_parameters.npz') as p:
            u=p['preview_u'][:,:32]
            assert float(p['preview_theta_source'])==1.0
        exponent=np.ceil(np.log2(np.max(np.abs(u),axis=0)/127)).astype(int)
        code=np.rint(u/np.exp2(exponent)).astype(np.int64)
        assert np.array_equal(u,code*np.exp2(exponent))
        # A partial signed correction can contain <=2 complete supports;
        # final sums and all prefixes remain exact at each output's dyadic scale.
        bound=int((2*np.abs(code).sum(0)).max());assert bound<2**24
        with np.load(folder/'000_zurich_city_09_a_0001.npz') as z:
            words=z['full_sn1_words']
            geometry=json.loads(str(z['window_geometry_json']))
            captured={name:z[name+'_preview_Z_shared'].copy() for name in ['corner','interior']}
        ar={}
        for label in ['corner','interior']:
            geo=geometry[label];h,w=geo['gate_shape'];oy,ox=geo['gate_origin']
            positions=[(y,x) for y in range(h) for x in range(w)]
            keys={};guides={}
            for t in range(10):
                for y,x in positions:
                    yy,xx=oy+y,ox+x
                    support=np.zeros((96,3,3),dtype=np.uint8)
                    for ky in range(3):
                        for kx in range(3):
                            sy,sx=yy+ky-1,xx+kx-1
                            if 0<=sy<240 and 0<=sx<320:support[:,ky,kx]=(words[:,sy,sx]>>t)&1
                    keys[t,y,x]=int.from_bytes(np.packbits(support.reshape(-1),bitorder='little').tobytes(),'little')
                    guide=(words[GUIDE,yy,xx]>>t)&1
                    guides[t,y,x]=sum(int(v)<<i for i,v in enumerate(guide))
            def parent(t,y,x,d):
                dy,dx=d;py,px=y+dy,x+dx
                return (t-1,py,px) if t>0 and 0<=py<h and 0<=px<w else None
            raw={k:sum((code[i] for i in bits(v)),start=np.zeros(32,np.int64)) for k,v in keys.items()}
            direct=np.stack([raw[t,y,x] for t in range(10) for y,x in positions]).reshape(10,h,w,32).transpose(0,3,1,2)
            actual=(direct*np.exp2(exponent)[None,:,None,None]).astype(np.float32)
            assert np.array_equal(actual.view(np.uint32),captured[label].view(np.uint32))
            assert np.array_equal(tf32_round(actual).view(np.uint32),tf32_round(captured[label]).view(np.uint32))
            arms={}
            # More optimistic than any implemented guide: every row may choose
            # a different neighbour, with zero detection/cache/parent cost.
            ideal_terms=0
            for t in range(10):
                for y,x in positions:
                    key=keys[t,y,x];cost=key.bit_count()
                    for d in OFFSETS:
                        ref=parent(t,y,x,d)
                        if ref:cost=min(cost,(key^keys[ref]).bit_count())
                    ideal_terms+=cost
            for mode in result['arms']:
                cache={};source_terms=0;reference_vectors=0;zero_fresh=0;repaired=0;reset=0
                selected=[];guide_comparisons=0;support_compares=0;cache_writes=0;numeric_checks=0
                for t in range(10):
                    d=(0,0)
                    if t and mode=='guide_displacement_delta':
                        costs=[]
                        for candidate in OFFSETS:
                            cost=0
                            for y,x in positions:
                                ref=parent(t,y,x,candidate)
                                cost+=(guides[t,y,x]^(guides[ref] if ref else 0)).bit_count()
                                guide_comparisons+=1
                            costs.append(cost)
                        d=OFFSETS[int(np.argmin(costs))]
                    elif t and mode=='full_support_displacement_oracle':
                        costs=[]
                        for candidate in OFFSETS:
                            cost=0
                            for y,x in positions:
                                ref=parent(t,y,x,candidate);key=keys[t,y,x]
                                cost+=min(key.bit_count(),(key^keys[ref]).bit_count()+1) if ref else key.bit_count()
                            costs.append(cost)
                        d=OFFSETS[int(np.argmin(costs))]
                    selected.append(list(d));next_cache={}
                    for y,x in positions:
                        key=keys[t,y,x];ref=parent(t,y,x,d) if mode!='fresh_GP' else None
                        use=False
                        if ref:
                            support_compares+=14 # padded864-bit key, real physical64-bit words
                            delta=(key^keys[ref]).bit_count()
                            use=delta+1<key.bit_count()
                        if use:
                            positive=key&~keys[ref];negative=keys[ref]&~key
                            value=cache[ref[1],ref[2]].copy()
                            for i in bits(positive):value+=code[i]
                            for i in bits(negative):value-=code[i]
                            source_terms+=positive.bit_count()+negative.bit_count();reference_vectors+=1;repaired+=1
                        else:
                            value=sum((code[i] for i in bits(key)),start=np.zeros(32,np.int64))
                            source_terms+=key.bit_count();reset+=1;zero_fresh+=key==0
                        assert np.array_equal(value,raw[t,y,x]);numeric_checks+=32
                        next_cache[y,x]=value
                        if mode!='fresh_GP' and t<9:cache_writes+=1
                    cache=next_cache
                total_ops=4*(source_terms+reference_vectors)
                arms[mode]=dict(source_terms=source_terms,source_ADD8_issues=4*source_terms,
                    heuristic_parent_vector_slot_penalty=4*reference_vectors,
                    total_ADD8_equivalents=total_ops,guide_hamming16_comparisons=guide_comparisons,
                    support_pair64_comparisons=support_compares,unshared_current_plus_reference_pair_key_scan_bytes=support_compares*16,
                    raw_U32_reference_read_bytes=reference_vectors*128,raw_U32_reference_write_bytes=cache_writes*128,
                    reference_cache_two_planes_bytes=2*h*w*(112+128) if mode!='fresh_GP' else 0,
                    repaired_rows=repaired,fresh_rows=reset,fresh_zero_rows=zero_fresh,
                    output_values_checked=numeric_checks,integer_mismatches=0,selected_displacements=selected,
                    displaced_steps=sum(q!=[0,0] for q in selected),
                    oracle=mode.endswith('oracle'),
                    note='Total equivalents includes one heuristic parent-vector slot penalty for route selection. Actual repair needs parent LOAD, not an extra parent ADD; actual source ADD8 and bytes are separate. Fresh zero skipping is available to all. Coefficient traffic/metadata/pipeline/control still unmodeled; no service speedup.')
            base_ops=arms['fresh_GP']['total_ADD8_equivalents']
            for r in arms.values():r['arithmetic_reduction_vs_fresh']=1-r['total_ADD8_equivalents']/base_ops
            ar[label]=dict(shape=[10,h,w,864,32],original_capture_FP32_bit_differences=0,downstream_TF32_bit_differences=0,
                maximum_signed_prefix_bound=bound,
                zero_overhead_per_row_any_neighbor_oracle=dict(source_terms=ideal_terms,
                    source_reduction=1-ideal_terms/arms['fresh_GP']['source_terms'],
                    caveat='Different displacement allowed for every row; free detector/cache/parent load. Upper bound only on source additions at this capture/tile/T reference set, not total service.'),arms=arms)
            print(axis,label,{k:round(v['arithmetic_reduction_vs_fresh'],5) for k,v in arms.items()},flush=True)
        result['axes'][axis]=ar
        (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
