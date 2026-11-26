#!/usr/bin/env python3
"""Read actual original W plus selected K own-BN values; no inference/quality."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;DATE=HERE.parent.parent;K=DATE/'later_checkpoint_deployment'
GFILE=DATE/'coverage_recovery_round13/independent_execution_audit/audit_execution.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
 return h.hexdigest()
assert sha(GFILE)=='dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8'
s=importlib.util.spec_from_file_location('_independent_geometry',GFILE);G=importlib.util.module_from_spec(s);s.loader.exec_module(G)
torch.set_num_threads(2)
pt=DATE/'baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt'
assert sha(pt)=='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
p=torch.load(pt,map_location='cpu',weights_only=True);state=p['model_state_dict']
assert p['training']['completed_optimizer_steps']==271482 and len(state)==711
assert all(v.device.type=='cpu' and bool(torch.isfinite(v).all()) for v in state.values())
assert G.tensor_digest(state)=='ad14a7d45091aa75d64df83e1a71bd89b35dfbbf49c9ab84861d5ae0aa270560'
suffix=('running_mean','running_var','num_batches_tracked')
allnames={n.rsplit('.',1)[0] for n in state if n.endswith(tuple('.'+x for x in suffix))}
fixed={n+'.bn'+str(i)+'.norm_layer' for n in G.SELECTED for i in (1,2)};keys={n+'.'+x for n in fixed for x in suffix}
assert len(allnames)==78 and len(fixed)==16 and fixed<=allnames
selected={n:t for n,t in state.items() if not n.endswith(tuple('.'+x for x in suffix))}
assert len(selected)==477 and G.tensor_digest(selected)=='b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
bn=K/'run_v1/bn_buffers.npz';receipt=K/'run_v1/bn_receipt.json';audit=K/'independent_review/ACTUAL_EXECUTION_AUDIT_V1.json'
assert sha(bn)=='4d887a8fce408e3a33cb95bd120d2ed3d865541a39c034205f4172ab7bdd30dc'
assert sha(receipt)=='24d33ba100150775a733fe6ecfabd4e88a1c152ca080adbf1f387a30ce574b9c'
assert sha(audit)=='7455e073fcfb1ff981e228807cf917a4e4657a7ef988d072d7a16cdcbb77e9bd'
arrays={}
with np.load(bn,allow_pickle=False) as z:
 assert len(z.files)==72 and keys<=set(z.files)
 for n in sorted(keys):
  a=z[n].copy();o=state[n].numpy();assert a.shape==o.shape and a.dtype==o.dtype and np.isfinite(a).all()
  if n.endswith('num_batches_tracked'):assert int(a)==176
  if n.endswith('running_var'):assert (a>=0).all()
  selected[n]=torch.from_numpy(a);arrays[n]={'shape':list(a.shape),'dtype':str(a.dtype),'payload_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
assert len(selected)==525 and sum(v.numel() for v in selected.values())==54944214
assert not torch.cuda.is_initialized()
out={'schema':'flow_selected16_independent_source_state_v1','status':'PASS','script_sha256':sha(__file__),'checkpoint_sha256':sha(pt),
 'source_state_sha256':G.tensor_digest(state),'allcurrent_state_sha256':'b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798',
 'selected16_deployment_state_sha256':G.tensor_digest(selected),'state_tensors':525,'state_elements':54944214,'BN_source_npz_sha256':sha(bn),'BN_source_receipt_sha256':sha(receipt),
 'BN_source_execution_audit_sha256':sha(audit),'frozen_BN_names':sorted(fixed),'current_stat_BN_names':sorted(allnames-fixed),'selected_buffer_count':48,'selected_buffer_arrays':arrays,
 'training_or_calibration_performed':False,'GPU_initialized':False,'quality_metrics_read':False,
 'scope':'Actual original source W + actual same-W K own48 BN subset, no new calibration or inference; no quality/benefit claim.'}
path=HERE/'SOURCE_STATE_V1.json';assert not path.exists();path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:out[k] for k in ('status','selected16_deployment_state_sha256','state_tensors','state_elements')},indent=2));print('receipt_sha256',sha(path))
