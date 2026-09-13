"""Read existing real T,C,y,x captures. No inference, training or production writes."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = ROOT / 'ideafromai/research/hardware_innovation_20260908/open_fusion_execution'
SOURCES = {
    'current_ordinary_R24_onepass': BASE/'stage_20260912/algorithm/hardware_exports/ordinary',
    'matched_dense_stage320_snapshot': BASE/'breadth_20260912/algorithm/hardware_exports/dense',
}
# Fixed from the existing interior anchor traversal, before observing statistics.
POSITIONS = [(120,160),(120,162)]
P,T,K,N = 2,10,864,16

def extract(a):
    assert str(a['tensor_order']) == 'T,C,y,x'
    geo = json.loads(str(a['window_geometry_json']))['interior']
    g = a['interior_sn2_gate']
    assert g.shape == (10,96,9,9) and g.dtype == np.bool_
    oy,ox = geo['gate_origin']
    s=np.stack([g[:,:,y-1-oy:y+2-oy,x-1-ox:x+2-ox].reshape(T,K) for y,x in POSITIONS])
    assert s.shape == (P,T,K)
    return s,geo

def stats(s):
    d=np.diff(s.astype(np.int8),axis=1,prepend=np.zeros((s.shape[0],1,s.shape[2]),np.int8))
    a=int(s.sum()); e=int(np.count_nonzero(d)); runs=int((d>0).sum())
    lengths=[]
    for row in s.transpose(0,2,1).reshape(-1,T):
        starts=np.where(np.diff(np.r_[0,row,0].astype(int))==1)[0]
        ends=np.where(np.diff(np.r_[0,row,0].astype(int))==-1)[0]
        lengths += (ends-starts).tolist()
    # A group is the P2 x T10 consumer set sharing one K weight vector.
    union=s.any(axis=(0,1)); du=(d!=0).any(axis=(0,1))
    per_time=[int(s[:,t].sum()) for t in range(T)]
    group_union_by_t=[int(s[:,t].any(axis=0).sum()) for t in range(T)]
    return dict(shape=list(s.shape),spikes=a,endpoint_events=e,rising=runs,
                falling_in_window=int((d<0).sum()),terminal_open_intervals=int(s[:,-1].sum()),
                closed_interval_endpoint_count=2*runs,interval_count=runs,
                interval_length_histogram={str(i):lengths.count(i) for i in range(1,T+1)},
                mean_interval_length=a/runs if runs else 0,
                spike_density=a/s.size,endpoint_density=e/s.size,
                live_K_groups=int(union.sum()),endpoint_live_K_groups=int(du.sum()),
                same_union_K=bool(np.array_equal(union,du)),
                spike_count_by_time=per_time,spatial_union_K_by_time=group_union_by_t,
                direct_vector_add_cycles=2*a,endpoint_vector_add_cycles=2*e,
                prefix_vector_add_cycles=s.shape[0]*(T-1)*2,
                auto_select_endpoint=(2*e+s.shape[0]*(T-1)*2 < 2*a))

def word_pack(values,bits):
    out=0
    for lane,v in enumerate(values): out |= (int(v)&((1<<bits)-1)) << (lane*bits)
    return out

def write_hex(name,values,width):
    (HERE/name).write_text(''.join(f'{int(v):0{width}x}\n' for v in values))

def rounded24(raw):
    q=raw//8; r=raw-q*8
    return np.clip(q+((r>4)|((r==4)&((q&1)!=0))),-(1<<23),(1<<23)-1)

def main():
    report={}
    current=None
    for name,path in SOURCES.items():
        with np.load(path/'000_zurich_city_09_a_0001.npz') as a, np.load(path/'deployed_constants.npz') as c, np.load(path/'live_parameters.npz') as l:
            s,geo=extract(a); w=c['U_conv2_theta_q16'].astype(np.int64)
            assert w.shape==(N,K)
            assert int(c['U_conv2_theta_exponent'])==17
            assert np.ndim(c['sn2_theta'])==0 and float(c['sn2_theta'])==1
            assert np.ndim(l['preview_theta_output'])==0 and float(l['preview_theta_output'])==1
            report[name]=dict(capture=str(path/'000_zurich_city_09_a_0001.npz'),constants=str(path/'deployed_constants.npz'),
                              layout='P,T,K; K=c*9+ky*3+kx, actual original T retained',positions=POSITIONS,
                              gate_geometry=geo,theta=1,weight_exp=17,output_fraction_bits=14,
                              weight_range=[int(w.min()),int(w.max())],tile=stats(s),
                              full_captured_interior_gate=stats(a['interior_sn2_gate'].reshape(1,T,-1)))
            if current is None: current=(s,w)
    s,w=current
    # Directed full-size functional controls, not additional data-performance samples.
    zeros=np.zeros_like(s); ones=np.ones_like(s)
    alternating=np.zeros_like(s); alternating[:,::2,:]=1
    boundary=np.zeros_like(s)
    boundary[:,0,:216]=1; boundary[:,-1,216:432]=1
    boundary[:,2:8,432:648]=1; boundary[:,3:,648:]=1
    cases=[('real_current',s),('all_zero_control',zeros),('all_one_control',ones),
           ('alternating_control',alternating),('interval_boundary_control',boundary)]
    report['functional_cases']={name:stats(x) for name,x in cases}
    report['state_bound']={'universal_abs_K_times_int16_max':K*32768,
        'signed32_accumulator_proved_safe':K*32768 < 2**31,
        'real_max_column_abs_weight_sum':int(np.abs(w).sum(axis=1).max()),
        'note':'Prefix is formed after full K sums, before RNE/saturation; telescoping prefix remains a direct binary dot product.'}
    report['scope']='One fixed P2 tile, complete K864,T10,N16; no full layer, BN, PSN, V96, or full network timing.'
    masks=[]; gold=[]
    for name,x in cases:
        for k in range(K): masks.append(word_pack(x[:,:,k].reshape(-1),1))
        raw=np.einsum('ptk,nk->ptn',x.astype(np.int64),w)
        delta=np.diff(x.astype(np.int64),axis=1,prepend=np.zeros((P,1,K),np.int64))
        rec=np.cumsum(np.einsum('ptk,nk->ptn',delta,w),axis=1)
        assert np.array_equal(raw,rec)
        y=rounded24(raw)
        for p in range(P):
            for t in range(T):
                for hg in range(2): gold.append(word_pack(y[p,t,hg*8:hg*8+8],24))
        report['functional_cases'][name]['integer_output_range']=[int(y.min()),int(y.max())]
    write_hex('spikes.mem',masks,5)
    write_hex('weights.mem',[word_pack(w[:,k],16) for k in range(K)],64)
    write_hex('golden.mem',gold,48)
    (HERE/'statistics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['tile'] for k,v in report.items() if isinstance(v,dict) and 'tile' in v},indent=2))

if __name__=='__main__': main()
