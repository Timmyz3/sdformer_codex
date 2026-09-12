"""Package only ordinary W8/W4 controls; sign candidates remain probe-only."""
import hashlib
import json
import numpy as np
from probe import HERE,execute

checks=[];files=[]
with np.load(HERE/'replay_fixture_probe_only.npz',allow_pickle=False) as f:
    fixture={k:f[k] for k in f.files}
for axis in ['ordinary','lifting_raw']:
    with np.load(HERE/(axis+'_parameters_probe_only.npz'),allow_pickle=False) as f:
        q={k:f[k] for k in f.files}
    package=dict(PED_bias_q24=q['bias_q24'],U_ped_exponent=np.array(16),V_ped_exponent=np.array(15),original_U=q['original_U'],original_V=q['original_V'])
    for mode in ['W8','W4']:
        expanded=q[mode+'_code'].astype(np.int64)*q[mode+'_scale'][:,None].astype(np.int64)
        assert expanded.min()>=-32768 and expanded.max()<=32767
        package[mode+'_U']=expanded.astype(np.int16);package[mode+'_V']=q['original_V']
        package[mode+'_code']=q[mode+'_code'];package[mode+'_scale_q16']=q[mode+'_scale']
        for window in ['corner','interior']:
            prefix=axis+'_'+window
            actual,_=execute(dict(kind='dense',U=expanded,V=q['original_V']),fixture[prefix+'_x'],q['bias_q24'])
            expected=fixture[prefix+'_'+mode]
            n=int(np.count_nonzero(actual!=expected));assert n==0
            checks.append(dict(axis=axis,window=window,mode=mode,values=int(actual.size),different_values=n))
    out=HERE/(axis+'_lowbit_gpu_parameters.npz');np.savez_compressed(out,**package)
    files.append(dict(path=out.name,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
out=dict(recommendation='W8 first as ordinary strong control, only after root/phase_aee existing queue; no GPU job launched.',not_recommended_without_training=['sign_residual_r0','sign_residual_r4','sign_residual_r8','sign_residual_r16'],precision='Expanded q16 U is exactly equivalent to low-bit-code integer dot then q16 row-scale before original U RNE. Original V q16/e15 and bias untouched; no new RNE.',fixture='replay_fixture_probe_only.npz',adapter='lowbit_ped_adapter.py',cpu_expansion_checks=checks,parameter_files=files,AEE_status='not evaluated; local error cannot predict DSEC AEE',hardware_status='compressed storage and per-row scale must still be implemented/priced; adapter is a functional check only')
(HERE/'gpu_package_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
