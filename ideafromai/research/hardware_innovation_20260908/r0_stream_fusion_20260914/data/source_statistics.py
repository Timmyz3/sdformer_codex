"""Full-frame descriptive fixed-group support statistics; no layout selection."""
import json
import numpy as np
from model_access import HERE

def hist(a,size):return np.bincount(a.reshape(-1).astype(np.int64),minlength=size).tolist()
def summary(a):return dict(min=int(a.min()),max=int(a.max()),mean=float(a.mean()),quantiles={str(q):float(np.quantile(a,q)) for q in (0,.25,.5,.75,.9,.99,1)})

def main():
    bits=np.load(HERE/'source_bits.npy',mmap_mode='r');b=np.asarray(bits,dtype=np.bool_)
    transitions=b[1:]!=b[:-1];onsets=(~b[:-1])&b[1:];offsets=b[:-1]&(~b[1:])
    spike_word=b.sum(0);d_internal=transitions.sum(0)
    positive=b[0].astype(np.int16)+onsets.sum(0)
    endpoint_prefix=b[0].astype(np.int16)+d_internal
    endpoint_closed=endpoint_prefix+b[-1]
    report=dict(complete=True,source='source_bits.npy',frame='zurich_city_09_a_0001.npy',shape=list(b.shape),
        elements=int(b.size),spikes=int(b.sum()),density=float(b.mean()),temporal_words=int(spike_word.size),
        nonzero_words=int((spike_word!=0).sum()),spikes_per_t=b.sum((1,2,3)).tolist(),spikes_per_c=b.sum((0,2,3)).tolist(),
        word_popcount_histogram=hist(spike_word,11),
        endpoints=dict(internal_T1_to_T9=int(d_internal.sum()),initial_D0=int(b[0].sum()),terminal_return_to_zero=int(b[-1].sum()),
            execution_D0_to_D9=int(endpoint_prefix.sum()),closed_D0_to_D10=int(endpoint_closed.sum()),
            positive_interval_starts=int(positive.sum()),internal_positive=int(onsets.sum()),internal_negative=int(offsets.sum()),
            internal_transition_histogram=hist(d_internal,10),prefix_endpoint_histogram=hist(endpoint_prefix,11),closed_endpoint_histogram=hist(endpoint_closed,12),
            convention='D0=g0; Dt=gt-g(t-1) for t1..9; terminalD10=-g9 only in closedinterval count. No cross-frame continuity assumed.'),
        groups={},scope='actual unpruned full native source; fixed contiguous original-channel groups; no mask/rank/order selection')
    pc=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)
    for width in (4,8):
        grouped=b.reshape(10,96//width,width,240,320)
        code=np.zeros((10,96//width,240,320),dtype=np.uint8)
        for i in range(width):code|=grouped[:,:,i].astype(np.uint8)<<i
        unique=np.zeros(code.shape[1:],dtype=np.uint8)
        freq=hist(code,2**width)
        for k in range(1,2**width):unique+=np.any(code==k,axis=0)
        first=grouped[:,:,:width//2].any(2);second=grouped[:,:,width//2:].any(2)
        H=(first&second).sum(0);K=first.any(0).astype(np.uint8)+second.any(0).astype(np.uint8)
        cpop=pc[code]
        pergroup=[]
        for g in range(96//width):
            pergroup.append(dict(channels=list(range(g*width,(g+1)*width)),spikes=int(cpop[:,g].sum()),
                active_times=int((code[:,g]!=0).sum()),multi_source_times=int((cpop[:,g]>=2).sum()),
                half_overlap_times=int(H[g].sum()),support_histogram=hist(code[:,g],2**width)))
        report['groups'][str(width)]=dict(group_definition='contiguous C in original order; bitj=channel(group*width+j)',
            group_spatial_words=int(unique.size),group_time_positions=int(code.size),support_frequency=freq,
            support_popcount_histogram=hist(cpop,width+1),unique_nonzero_supports_per_T10_word_histogram=hist(unique,11),
            half_union_overlap_H_histogram=hist(H,11),active_half_K_histogram=hist(K,3),
            half_union_overlap_total=int(H.sum()),multi_source_time_positions=int((cpop>=2).sum()),per_group=pergroup)
        assert sum(freq)==code.size and sum(k*n for k,n in enumerate(hist(cpop,width+1)))==b.sum()
    # Disjoint spatial output tiles: no halo duplication in this descriptive map.
    spatial=b.sum((0,1));tile=spatial.reshape(120,2,160,2).sum((1,3))
    np.save(HERE/'spikes_per_output_tile.npy',tile.astype(np.uint32))
    report['spatial']=dict(disjoint2x2_output_tile_spikes=summary(tile),zero_tiles=int((tile==0).sum()),
        occupied_y_rows=int((spatial.sum(1)>0).sum()),occupied_x_columns=int((spatial.sum(0)>0).sum()),
        quadrants=[[int(spatial[y:y+120,x:x+160].sum()) for x in (0,160)] for y in (0,120)],
        note='Tile map sums native pixels in each disjoint2x2 region, not4x4halo source fetch or execution cycles.')
    (HERE/'source_statistics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','spikes_per_c')},indent=2))
if __name__=='__main__':main()
