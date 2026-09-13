"""Actual trained FP32 coefficients as exact dyadic integers; full N768."""
import sys
sys.dont_write_bytecode=True
import ctypes,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from execute import HERE,HW,MODULE,capture

subprocess.run(['g++','-O3','-march=native','-std=c++17','-shared','-fPIC',str(HERE/'numeric.cpp'),'-o','/tmp/prosperity_owned_numeric.so'],check=True)
lib=ctypes.CDLL('/tmp/prosperity_owned_numeric.so')
ptr=lambda a:ctypes.c_void_p(a.ctypes.data)
state=torch.load(HW/'system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth',map_location='cpu',weights_only=False)['model_state_dict']
w=state[MODULE+'.weight'].numpy().reshape(768,6912).T.copy()
_,exp=np.frexp(w);scales=np.where(w!=0,exp-24,0).min(0)
wi=np.ldexp(w.astype(np.float64),-scales)
assert np.all(wi==np.rint(wi))
wi=np.ascontiguousarray(wi,dtype=np.int64)
assert np.array_equal(np.ldexp(wi.astype(np.float64),scales),w)
bound=np.abs(wi).sum(0,dtype=np.int64)
assert bound.max()<2**62
_,matrix=capture(1)
a=np.ascontiguousarray(matrix.reshape(10,300,6912).transpose(1,0,2).reshape(3000,6912),dtype=np.uint8)
gold=np.empty((3000,768),np.int64);start=time.monotonic()
lib.direct(ptr(a),ptr(wi),ptr(gold),3000,6912,768)
report=dict(sample=1,geometry=[3000,6912,768],trained_weight_key=MODULE+'.weight',
  weight_min=float(w.min()),weight_max=float(w.max()),weight_zeros=int(np.count_nonzero(w==0)),
  exact_dyadic_scale_min=int(scales.min()),exact_dyadic_scale_max=int(scales.max()),
  exact_integer_absolute_sum_bound=int(bound.max()),exact_weight_reconstruction_mismatches=0,
  actual_weight_bytes=w.nbytes,diagnostic_integer_weight_bytes=wi.nbytes,
  shared_theta_float32=0.9999106526374817,shared_theta_bits='3f7ffa25',
  direct_host_seconds=time.monotonic()-start,layouts={},
  limits=['Exact sums of actual FP32 weight values before the common exact theta factor.',
    'The int64 dyadic reference is an algebraic oracle, not an INT8 deployment or area claim.',
    'No GPU FP32 accumulation-order equivalence, following PSN/PED, or new AEE evaluation is asserted.'])
for name in ('channel_major','tap_major','calibrated_tap_major','spatial_major'):
    if not (HERE/(name+'_numeric_plan.npz')).exists():continue
    data=np.load(HERE/(name+'_numeric_plan.npz'))
    ps=np.concatenate([data[f'p{i}'].reshape(-1) for i in range(12)]).astype(np.int16)
    rs=np.concatenate([data[f'r{i}'].reshape(-1) for i in range(12)]).astype(np.uint16)
    os=np.concatenate([data[f'o{i}'].reshape(-1) for i in range(12)]).astype(np.uint16)
    perm=data['perm'].astype(np.int32);out=np.empty_like(gold);start=time.monotonic()
    lib.forest(ptr(ps),ptr(rs),ptr(os),ptr(perm),ptr(wi),ptr(out),3000,6912,768)
    expected=gold[data['row_index']] if 'row_index' in data else gold
    wrong=int(np.count_nonzero(expected!=out));assert wrong==0
    report['layouts'][name]=dict(exact_output_values=int(out.size),mismatches=wrong,
        host_seconds=time.monotonic()-start,output_numerator_absmax=int(np.abs(out).max()))
    print(name,report['layouts'][name],flush=True)
    (HERE/'numeric_results.json').write_text(json.dumps(report,indent=2)+'\n')
