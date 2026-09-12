"""GPU-ready ordinary W8/W4 U-only check using the unchanged q16 PED helper.

Encoded low-bit code * row-scale expands exactly to q16 coefficients offline.
This adapter preserves original U/V/bias boundaries and is NOT a compressed
CUDA kernel or an implementation of the sign+residual candidate.
"""
import numpy as np

MODES=('original32','W8','W4')


def select(package,mode,original):
    if mode=='original32':return original['U_ped_q16'],original['V_ped_q16']
    assert mode in ('W8','W4')
    return package[mode+'_U'],package[mode+'_V']


def install(helper,package,mode,original):
    import torch
    assert int(package['U_ped_exponent'])==int(original['U_ped_exponent'])==16
    assert int(package['V_ped_exponent'])==int(original['V_ped_exponent'])==15
    assert np.array_equal(helper.projection_bias.detach().cpu().numpy(),package['PED_bias_q24'])
    assert np.array_equal(original['PED_bias_q24'],package['PED_bias_q24'])
    u,v=select(package,mode,original)
    assert np.array_equal(v,original['V_ped_q16'])
    for name,values,exponent in [('U_ped',u,16),('V_ped',v,15)]:
        values=np.asarray(values,np.int64)
        pos=np.maximum(values,0).sum(1);neg=np.minimum(values,0).sum(1)
        lower=pos*(-2**23)+neg*(2**23-1)
        upper=pos*(2**23-1)+neg*(-2**23)
        bound=int(max(np.max(np.abs(lower)),np.max(np.abs(upper))))
        assert bound<2**47
        helper.matrices[name]=dict(q=torch.as_tensor(values,device=helper.device,dtype=torch.float64),q_numpy=values,exponent=exponent,lower=lower,upper=upper)
        helper.matrix_metadata[name]=dict(shape=list(values.shape),exponent=exponent,input_domain='signed24',dot_abs_bound=bound,exact_saved_coefficients=True,mode=mode,coefficient_origin='U-only symmetric low-bit code times q16 row scale; offline exact expansion')
        helper.constants[name+'_q16']=values.astype(np.int16)
        helper.constants[name+'_exponent']=np.asarray(exponent)
    helper.ready=False


def run_continuous(helper,x):
    helper.frame=dict(clip_counts={},state_ranges={},accumulator_ranges={})
    z=helper.channel_dot('U_ped',x,'PED_U24')
    y=helper.channel_dot('V_ped',z,'PED_V24')
    return helper.write24('PED_bias_output24',y+helper.projection_bias[None,:,None,None])


def check_fixture(helper,package,original,fixture,axis):
    import torch
    checks=[]
    with torch.no_grad():
        for mode in MODES:
            install(helper,package,mode,original)
            for window in ['corner','interior']:
                prefix=axis+'_'+window
                # Stored input/expected matrices are [C,T*H*W]; helper wants T,C,H,W.
                x=fixture[prefix+'_x'].reshape(96,10,4,4).transpose(1,0,2,3)
                actual=run_continuous(helper,torch.as_tensor(x,device=helper.device,dtype=torch.float64)).cpu().numpy().transpose(1,0,2,3).reshape(96,-1)
                expected=fixture[prefix+'_'+mode]
                n=int(np.count_nonzero(actual!=expected));assert n==0,(axis,window,mode,n)
                checks.append(dict(axis=axis,window=window,mode=mode,values=int(expected.size),differences=n))
        install(helper,package,'original32',original)
    return checks
