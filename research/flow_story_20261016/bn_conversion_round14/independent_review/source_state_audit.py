#!/usr/bin/env python3
"""Independent CPU source state digests for the fixed two-checkpoint all78-current test."""
import hashlib,json
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;DATE=HERE.parent.parent
SOURCES={135930:(DATE/'coverage_recovery_round13/source_snapshot/checkpoint_step_000135930.pt','d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'),271482:(DATE/'baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt','1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977')}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def digest(state):
 h=hashlib.sha256()
 for n in sorted(state):
  t=state[n].detach().cpu().contiguous();h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
 return h.hexdigest()
records=[]
for step,(path,expected) in SOURCES.items():
 assert sha(path)==expected
 payload=torch.load(path,map_location='cpu',weights_only=True);state=payload['model_state_dict']
 assert payload['training']['completed_optimizer_steps']==step and payload['contract_sha256']=='961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934'
 assert len(state)==711 and sum(t.numel() for t in state.values())==55001588
 assert all(t.device.type=='cpu' and bool(torch.isfinite(t).all()) for t in state.values())
 buffers={n for n in state if n.endswith(('.running_mean','.running_var','.num_batches_tracked'))};names={n.rsplit('.',1)[0] for n in buffers}
 assert len(buffers)==234 and len(names)==78
 current={n:t for n,t in state.items() if n not in buffers};assert len(current)==477
 records.append({'checkpoint_step':step,'checkpoint_path':str(path),'checkpoint_sha256':expected,'source_native_state_sha256':digest(state),
  'all78_current_state_sha256':digest(current),'all78_current_state_tensors':len(current),'all78_current_state_elements':sum(t.numel() for t in current.values()),
  'removed_BN_buffer_count':len(buffers),'BN_names':sorted(names),'all_state_tensors_CPU_finite':True})
 del payload,state,current
assert not torch.cuda.is_initialized()
out={'schema':'flow_bn_conversion_round14_independent_source_states_v1','status':'PASS','script_sha256':sha(Path(__file__)),
 'source_states':records,'inference_executed':False,'GPU_used':False,'quality_read':False,
 'scope':'Actual original checkpoint bytes and all78-current state structure/digest only. No BN-statistic calculation, inference or quality evidence.'}
p=HERE/'SOURCE_STATES_V1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'states':[{k:r[k] for k in ('checkpoint_step','source_native_state_sha256','all78_current_state_sha256','all78_current_state_tensors','all78_current_state_elements')} for r in records]},indent=2))
