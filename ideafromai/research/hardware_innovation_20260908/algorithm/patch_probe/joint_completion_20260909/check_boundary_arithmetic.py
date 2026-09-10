"""Compare the Q16 numerical hook to integer shifts/RNE on real windows."""
from pathlib import Path
import json
import numpy as np
import torch
from evaluate_boundary_control import Boundary

ROOT=Path(__file__).resolve().parent


def signed_rne(value, shift):
    magnitude=np.abs(value)
    quotient=magnitude>>shift
    remainder=magnitude&((np.int64(1)<<shift)-1)
    half=np.int64(1)<<(shift-1)
    increment=(remainder>half)|((remainder==half)&((quotient&1)!=0))
    return (quotient+increment)*np.sign(value)


def main():
    torch.set_num_threads(4)
    with np.load(ROOT/'boundary_q16_calibration/parameters.npz') as z:
        params={k:z[k].copy() for k in z.files}
    hook=Boundary.__new__(Boundary)
    hook.params={k:torch.from_numpy(params[k]).double() for k in
                 ('identity_scale','output_scale','internal_scale','weight_scale','offset_int')}
    ei,eo,eq=(params[k].astype(np.int64) for k in ('identity_exponent','output_exponent','internal_exponent'))
    ew=np.log2(params['weight_scale']).astype(np.int64)
    view=lambda a:a[None,None,:,None,None]
    differences=values=0
    peak=0
    for file in sorted((ROOT/'integer_w1w2_capture4/capture').glob('*/*/consumer_windows.npz')):
        with np.load(file) as z:
            identity=z['identity'].copy()
            integer_z=z['conv2_Z_int32'].astype(np.int64)
        qi=np.clip(np.rint(identity.astype(np.float64)/view(params['identity_scale'])),-32767,32767).astype(np.int64)
        numerator=(qi<<view(ei-eq))+(integer_z<<view(ew-eq))+view(params['offset_int'])
        expected=np.clip(signed_rne(numerator,view(eo-eq)),-32767,32767)
        hook.identity=torch.from_numpy(identity)
        hook.z2=torch.from_numpy(integer_z).float()
        actual=hook.after(None,None,None).numpy()
        actual=np.rint(actual.astype(np.float64)/view(params['output_scale'])).astype(np.int64)
        differences+=np.count_nonzero(actual!=expected)
        values+=expected.size
        peak=max(peak,int(np.abs(numerator).max()))
    edge_count=edge_difference=0
    for shift in sorted(set((eo-eq).tolist())):
        units=np.array([-32768,-32767,-2,-1,0,1,2,32766,32767],dtype=np.int64)
        edges=(units[:,None]*(1<<shift)+np.array([-(1<<(shift-1)),-1,0,1,1<<(shift-1)])[None]).reshape(-1)
        reference=signed_rne(edges,shift)
        actual=torch.round(torch.from_numpy(edges).double()/(1<<shift)).long().numpy()
        edge_difference+=np.count_nonzero(reference!=actual);edge_count+=len(edges)
    result=dict(real_window_values=values,integer_reference_differences=int(differences),
                observed_internal_abs_max=peak,legal_internal_abs_max=int(params['internal_legal_abs_bound'].max()),
                signed_rounding_edge_cases=edge_count,edge_differences=int(edge_difference),
                scope='Q16 hook versus integer shifts and sign-aware round-to-even on actual W1/W2 captures; no service or PPA claim')
    (ROOT/'boundary_q16_calibration/arithmetic_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    assert values and differences==0 and edge_difference==0


if __name__=='__main__':main()
