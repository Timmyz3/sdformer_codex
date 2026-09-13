"""A800 hook: replace only U16 before constructing existing LiteralForward.
The reconstructed matrix evaluates candidate quality; it is NOT a GPU sparse
kernel timing. Packed values/indices and response_execution.py own execution.
"""
from pathlib import Path
import argparse
import numpy as np


def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def replace_u(parent_constants,package):
    p=arrays(package) if isinstance(package,(str,Path)) else package
    out={k:np.asarray(v).copy() for k,v in parent_constants.items()}
    assert out['U_conv2_theta_q16'].shape==p['U_conv2_theta_q16'].shape==(16,864)
    assert np.array_equal(out['U_conv2_theta_exponent'],p['U_conv2_theta_exponent'])
    assert p['U_conv2_theta_q16'].dtype==np.int16
    out['U_conv2_theta_q16']=p['U_conv2_theta_q16'].copy()
    assert all(np.array_equal(v,parent_constants[k]) for k,v in out.items() if k!='U_conv2_theta_q16')
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--package',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    np.savez_compressed(a.output,**replace_u(arrays(a.parent),a.package))
