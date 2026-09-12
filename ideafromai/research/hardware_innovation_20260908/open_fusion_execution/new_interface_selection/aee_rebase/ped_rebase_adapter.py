"""Exact saved PED coefficient replacement, using the existing fixed helper.

Only U_ped/V_ped change. No recompilation/rescaling, gates/BN changes or bias
substitution. Both matrices keep the original signed16 exponent16/15 contract.
"""
import numpy as np
import torch

MODES=('original32','original_ordered24','weight_svd24','activation_whitened24')


def select(q,mode,original):
    if mode=='original32':
        return original['U_ped_q16'],original['V_ped_q16']
    key={'original_ordered24':'original_ordered','weight_svd24':'weight_svd',
         'activation_whitened24':'activation_whitened'}[mode]
    return q[key+'_U'][:24],q[key+'_V'][:,:24]


def install(helper,q,mode,original):
    assert int(original['U_ped_exponent'])==int(q['U_ped_exponent'])==16
    assert int(original['V_ped_exponent'])==int(q['V_ped_exponent'])==15
    assert np.array_equal(helper.projection_bias.detach().cpu().numpy(),q['PED_bias_q24'])
    assert np.array_equal(original['PED_bias_q24'],q['PED_bias_q24'])
    u,v=select(q,mode,original)
    for name,values,exponent in [('U_ped',u,16),('V_ped',v,15)]:
        values=np.asarray(values,np.int64)
        pos=np.maximum(values,0).sum(1);neg=np.minimum(values,0).sum(1)
        lower=pos*(-2**23)+neg*(2**23-1)
        upper=pos*(2**23-1)+neg*(-2**23)
        bound=int(max(np.max(np.abs(lower)),np.max(np.abs(upper))))
        assert bound<2**47
        helper.matrices[name]=dict(q=torch.as_tensor(values,device=helper.device,dtype=torch.float64),
            q_numpy=values,exponent=exponent,lower=lower,upper=upper)
        helper.matrix_metadata[name]=dict(shape=list(values.shape),exponent=exponent,
            input_domain='signed24',dot_abs_bound=bound,exact_saved_coefficients=True,mode=mode)
        helper.constants[name+'_q16']=values.astype(np.int16)
        helper.constants[name+'_exponent']=np.asarray(exponent)
    helper.ready=False


@torch.no_grad()
def run_continuous(helper,x):
    helper.frame=dict(clip_counts={},state_ranges={},accumulator_ranges={})
    latent=helper.channel_dot('U_ped',x,'PED_U24')
    value=helper.channel_dot('V_ped',latent,'PED_V24')
    return helper.write24('PED_bias_output24',value+helper.projection_bias[None,:,None,None])


@torch.no_grad()
def check_fixture(helper,q,original,fixture,axis):
    rows=[]
    for mode in MODES:
        install(helper,q,mode,original)
        for window in ('corner','interior'):
            key=axis+'_'+window
            x=torch.as_tensor(fixture[key+'_x'],device=helper.device,dtype=torch.float64)
            actual=run_continuous(helper,x).to(torch.int64).cpu().numpy()
            expected=fixture[key+'_'+mode]
            differences=int(np.count_nonzero(actual!=expected))
            assert differences==0,(axis,window,mode,differences)
            rows.append(dict(axis=axis,window=window,mode=mode,values=actual.size,differences=differences))
    install(helper,q,'original32',original)
    return rows
