#!/opt/anaconda3/bin/python3.12
"""Freeze existing parameters only; no new quantization, calibration or training."""
from pathlib import Path
import json,sys
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent;BASE=SOURCE.parent
sys.path.insert(0,str(BASE/'bn_state'))
from support_service_model import read_torch
PREFIX='sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.'
old=read_torch(BASE/'algorithm/integer_s0_valid825/integer_parameters.pt')
student=read_torch(BASE/'algorithm/support_training/forced_code_parameters.pt')
source=np.load(SOURCE/'source_cases.npz',allow_pickle=False)
back=np.load(BASE/'support_lut_execution_20260915/cases.npz',allow_pickle=False)
weights={
 'original':np.asarray(read_torch(BASE/'algorithm/support_training/forced_code_weight_int8.pt')),
 'adapt':np.load(SOURCE/'source_class_adapt/adapt_W.npy',allow_pickle=False),
 'zero':np.load(SOURCE/'zero_response_reopen/Wz.npy',allow_pickle=False)}
assert all(w.shape==(384,96) and w.dtype==np.int8 for w in weights.values())
assert np.array_equal(weights['original'],np.rint(np.asarray(student['W'],dtype=np.float32)*np.asarray(student['theta'],dtype=np.float32)/np.asarray(student['scale'],dtype=np.float32)[:,None]).clip(-127,127).astype(np.int8))
for key in ['A','bias','center','theta']:
 assert np.array_equal(np.asarray(student[key]),source[key+'_fp32'])
assert np.array_equal(np.asarray(student['dictionary'],dtype=np.uint8),source['D'])
# Verify, but do not choose/requantize, already-frozen source formats.
assert np.array_equal(np.rint(source['A_fp32'].astype(np.float64)*2**12).astype(np.int16),source['A_q12'])
tr=(source['theta_fp32'].astype(np.float64)+source['center_fp32'].astype(np.float64)-source['bias_fp32'].astype(np.float64)).reshape(10)
assert np.array_equal(np.rint(tr*2**28).astype(np.int64),source['threshold_q28'])
bundle={'schema_version':np.array(1,dtype=np.int32),'source_A_fp32':source['A_fp32'],'source_bias_fp32':source['bias_fp32'],'source_center_fp32':source['center_fp32'],'source_theta_fp32':source['theta_fp32'],
 'source_A_q12':source['A_q12'],'source_tau_q28':source['threshold_q28'],'dictionary':source['D']}
for name,w in weights.items():bundle['fc1_'+name+'_W_int8']=w
for i in range(2):
 record=old[PREFIX+str(i)+'.mlp.']
 for key,value in record.items():bundle['consumer'+str(i)+'_'+key]=np.asarray(value)
assert np.array_equal(bundle['consumer0_temporal_int16'],back['A'])
ids=[int(np.flatnonzero((back['hblock']==h)&back['is_real'])[0]) for h in range(4)]
assert np.array_equal(np.concatenate([back['W'][i] for i in ids],axis=1).T,weights['original'])
for key,field,axis in [('threshold_int64','tau',1),('positive_gain','positive_gain',0),('constant_channels','constant_channels',0),('constant_gate','constant_gate',1)]:
 assert np.array_equal(bundle['consumer0_'+key],np.concatenate([back[field][i] for i in ids],axis=axis)),key
assert np.array_equal(bundle['consumer0_theta_source'].astype(np.float32),bundle['source_theta_fp32'])
# Preserve provenance and shape/type; bundle arrays are copied, not fitted or clipped.
np.savez_compressed(HERE/'frozen_bundle.npz',**bundle)
def bounds(w,a,signed_input):
 w=w.astype(np.int64);a=a.astype(np.int64)
 yabs=np.abs(w).sum(1)
 lo=-yabs if signed_input else np.minimum(w,0).sum(1)
 hi=yabs if signed_input else np.maximum(w,0).sum(1)
 ulo=np.minimum(a[:,:,None]*lo[None,None,:],a[:,:,None]*hi[None,None,:]).sum(1)
 uhi=np.maximum(a[:,:,None]*lo[None,None,:],a[:,:,None]*hi[None,None,:]).sum(1)
 partial_bound=int(np.max(np.abs(a).sum(1)[:,None]*yabs[None,:]))
 assert int(yabs.max())<2**23 and partial_bound<2**47 and partial_bound<2**53
 return {'Y_signed24_prefix_min':int(lo.min()),'Y_signed24_prefix_max':int(hi.max()),'U_final_min':int(ulo.min()),'U_final_max':int(uhi.max()),'U_any_partial_abs_bound':partial_bound,'Y_abs_sum_bound':int(yabs.max()),'FP64_exact_integer_dot':True}
A0=bundle['source_A_q12'].astype(np.int64)
source_bound=int(np.abs(A0).sum(1).max())*(1<<23)
assert source_bound<2**47 and source_bound<2**53
report={'status':'PASS','training':False,'requantized_weights':False,'reestimated_thresholds':False,
 'sources':{'source_quantized':'support_source_execution_20260915/source_cases.npz','source_student':'algorithm/support_training/forced_code_parameters.pt','original_W':'algorithm/support_training/forced_code_weight_int8.pt','W_double_prime':'support_source_execution_20260915/source_class_adapt/adapt_W.npy','Wz':'support_source_execution_20260915/zero_response_reopen/Wz.npy','frozen_integer_consumers':'algorithm/integer_s0_valid825/integer_parameters.pt','backend_RTL_gold_contract':'support_lut_execution_20260915/cases.npz'},
 'source_widths':{'X':'signed24 Q16, runtime RNE with admission check and no saturation','A':'frozen signed16 Q12','tau':'frozen signed48 Q28','tau_min':int(bundle['source_tau_q28'].min()),'tau_max':int(bundle['source_tau_q28'].max()),'source_any_X24_partial_abs_bound':source_bound,'FP64_exact_integer_dot':True},
 'fc1_bounds':{name:bounds(w,bundle['consumer0_temporal_int16'],False) for name,w in weights.items()},
 'teacher_consumer_bounds':{str(i):bounds(bundle[f'consumer{i}_weight_int8'],bundle[f'consumer{i}_temporal_int16'],True) for i in range(2)},
 'bundle_arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in bundle.items()},'bundle_bytes':(HERE/'frozen_bundle.npz').stat().st_size,
 'frozen_back_contract_equal':True,'retained_other_consumer':'layers.0.swin_blocks.1.mlp. unchanged frozen W/A14/tau'}
(HERE/'BUNDLE.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print(json.dumps({k:report[k] for k in ['status','source_widths','fc1_bounds','teacher_consumer_bounds','bundle_bytes']},separators=(',',':')))
