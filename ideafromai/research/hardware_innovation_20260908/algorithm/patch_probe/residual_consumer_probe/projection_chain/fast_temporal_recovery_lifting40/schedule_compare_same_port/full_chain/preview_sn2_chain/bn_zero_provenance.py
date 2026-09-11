"""S1: distinguish producer-known zero vectors from deletable BN/PED work.

No timing extrapolation and no active-only normalization. Read full gate
producer maps of the unchanged students, complete K864 and real image edges.
"""
import json
import numpy as np
from run_chain import HERE,read_npz,difference


def inspect(data,p):
    words=data['full_proj_words']
    padded=np.pad(words,((0,0),(1,1),(1,1)))
    support=np.zeros((120,160),np.uint16)
    for ky in range(3):
        for kx in range(3):
            support|=np.bitwise_or.reduce(padded[:,ky:ky+240:2,kx:kx+320:2],axis=0)
    zero=np.stack([((support>>t)&1)==0 for t in range(10)])
    raw=data['proj_bn_full_input_fp32'];observed=np.all(raw==0,axis=1)
    assert not bool(p['proj_has_bias'])
    violations=int(np.count_nonzero(zero&~observed));assert violations==0
    cont=data['full_continuous_q24'];cont_live=np.any(cont!=0,axis=1)
    normalized=data['proj_bn_full_output_fp32'];normalized_live=np.any(normalized!=0,axis=1)
    raw_dense=raw.nbytes;meta=(zero.size+255)//256*32
    active=int(np.count_nonzero(~zero));n=int(zero.size)
    # Mathematical control only. Omitting zeros from the denominator is a
    # different network function, although sparse libraries often use active
    # coordinate domains. Centered variance also keeps their nonzero mass.
    x=raw.transpose(0,2,3,1).reshape(n,96).astype(np.float64)
    mu=x.mean(0);var=((x-mu)**2).mean(0)
    nz=x[~zero.reshape(-1)]
    omit_mu=nz.mean(0);omit_var=((nz-omit_mu)**2).mean(0)
    count0=int(zero.sum())
    moment_mass=count0*mu**2/n
    wrong_var=((nz-mu)**2).sum(0)/n
    return dict(domain_T_HW=n,K=864,H=96,producer_known_zero_positions=count0,
        producer_known_zero_fraction=count0/n,observed_zero_positions=int(observed.sum()),
        zero_tag_violations=violations,observed_zero_without_producer_certificate=int(np.count_nonzero(observed&~zero)),
        certified_zero_but_continuous_PED_live=int(np.count_nonzero(zero&cont_live)),
        certified_zero_but_normalized_live=int(np.count_nonzero(zero&normalized_live)),
        raw_dense_bytes=raw_dense,raw_tagged_payload_bytes=active*384,bitmap_bytes_per_pass=meta,
        potential_raw_transfer_fraction=(active*384+meta)/raw_dense,
        full_domain_denominator=n,variance_zero_mass_fraction_max=float(np.max(moment_mass/var)),
        wrong_active_only_mean=difference(omit_mu,mu),wrong_active_only_variance=difference(omit_var,var),
        wrong_drop_zero_centered_contribution=difference(wrong_var,var),
        exact_FP32_reduction_preserved=False,net_service=None,
        disposition='Preserve as producer/default-value interface probe; generic sparse/default representation is a baseline, novelty unresolved.')


def main():
    result=dict(scope=__doc__,axes={},RTL=False,new_training=False,new_quantization=False)
    for axis in ('ordinary','lifting_raw'):
        path=HERE.parent/'capture_full_producers'/axis
        data=read_npz(path/'000_zurich_city_09_a_0001.npz');p=read_npz(path/'live_parameters.npz')
        result['axes'][axis]=inspect(data,p)
        previous=read_npz(HERE.parent/'capture'/axis/'000_zurich_city_09_a_0001.npz')
        old_checks={k:difference(data[k],v) for k,v in previous.items() if v.dtype.kind in 'fibu'}
        assert all(v['differences']==0 for v in old_checks.values()),old_checks
        result['axes'][axis]['recapture_numeric_arrays']=len(old_checks)
        result['axes'][axis]['recapture_total_differences']=sum(v['differences'] for v in old_checks.values())
        print(axis,result['axes'][axis],flush=True)
    (HERE/'bn_zero_provenance.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
