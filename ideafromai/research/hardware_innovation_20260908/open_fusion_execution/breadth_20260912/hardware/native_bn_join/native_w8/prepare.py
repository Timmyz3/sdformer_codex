from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[3]
EXPORT=OPEN/'stage_20260912/algorithm/hardware_exports/ordinary'


def main():
    p=dict(np.load(EXPORT/'live_parameters.npz'))
    w=p['proj_weight_fp32'].astype(np.float32)
    rows=w.reshape(96,864)
    peak=np.max(np.abs(rows),axis=1)
    exponent=np.ceil(np.log2(np.where(peak>0,peak/127.0,1.0))).astype(np.int32)
    scale=np.exp2(exponent.astype(np.float64)).astype(np.float32)
    q=np.clip(np.rint(rows/scale[:,None]),-128,127).astype(np.int8)
    expanded=np.float32(q.astype(np.float32)*scale[:,None]).reshape(w.shape)
    assert float(p['proj_theta_output'])==1.0 and not bool(p['proj_has_bias'])
    assert np.all(np.sum(np.abs(q.astype(np.int32)),axis=1)<2**24)
    np.savez_compressed(HERE/'native_w8.npz',q=q,scale=scale,exponent=exponent,
        expanded_weight_fp32=expanded,original_weight_fp32=w)
    info=dict(parent='Previous stage ordinary original_ordered24 + onepass, unchanged actual parent capture.',
        parent_export=str(EXPORT),target_module=str(p['proj_module'])+'.conv',
        replacement='Copy expanded_weight_fp32 into this conv.weight; preserve bias/stride/padding/BN/source/PED and all other modules.',
        tensor_shape=list(w.shape),q_shape=list(q.shape),q_dtype=str(q.dtype),
        rule='Per output row: exponent=ceil(log2(maxabs(original_FP32_W)/127)); scale=2**exponent; q=clip(RNE(W/scale),-128,127).',
        code_bytes=int(q.nbytes),scale_bytes=int(scale.nbytes),coefficient_bytes=int(q.nbytes+scale.nbytes),
        exponent_values=np.unique(exponent).tolist(),q_min=int(q.min()),q_max=int(q.max()),
        max_row_abs_integer_sum=int(np.max(np.sum(np.abs(q.astype(np.int32)),axis=1))),
        max_weight_abs_error=float(np.max(np.abs(expanded-w))),
        original_native_live_unchanged=True,new_function_requires_fresh_quality=True,
        same_upstream_projection_g_and_PED=True)
    (HERE/'deployment.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
