#!/usr/bin/env python3
"""Independent CPU deserialization of both exact source states, no inference."""
import hashlib,json
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;DATE=HERE.parent.parent
EARLY=DATE/'coverage_recovery_round13/source_snapshot/checkpoint_step_000135930.pt'
LATER=DATE/'baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def require(v,msg):
 if not v:raise ValueError(msg)
def digest(state,exclude=()):
 h=hashlib.sha256();exclude=set(exclude)
 for n in sorted(state):
  if n in exclude:continue
  t=state[n].detach().cpu().contiguous();h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
 return h.hexdigest()
require(sha(EARLY)=='d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30','early bytes')
require(sha(LATER)=='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977','later bytes')
early=torch.load(EARLY,map_location='cpu',weights_only=True);later=torch.load(LATER,map_location='cpu',weights_only=True)
require(not torch.cuda.is_initialized(),'No CUDA')
require(early['contract']==later['contract'],'actual full embedded contracts')
require(later['contract_sha256']==hashlib.sha256(json.dumps(later['contract'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'actual canonical contract digest')
require(later['contract_sha256']=='961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934','pinned contract')
require(early['training']['completed_optimizer_steps']==135930 and later['training']['completed_optimizer_steps']==271482,'source progress')
a=early['model_state_dict'];b=later['model_state_dict']
require(set(a)==set(b),'complete tensor name equality')
require(len(b)==711 and sum(t.numel() for t in b.values())==55001588,'native state counts')
for n,t in b.items():
 require(isinstance(t,torch.Tensor) and t.device.type=='cpu' and bool(torch.isfinite(t).all()),'finite CPU state '+n)
 require(t.shape==a[n].shape and t.dtype==a[n].dtype,'same state shape/dtype '+n)
suffixes=('.running_mean','.running_var','.num_batches_tracked');bn={n for n in b if n.endswith(suffixes)}
require(len(bn)==234 and len({n.rsplit('.',1)[0] for n in bn})==78,'all78 BN state names')
private={n for n in bn if '.mlp.bn' in n};require(len(private)==72,'24 MLP BN buffers')
state_without_external={n:t for n,t in b.items() if n not in bn-private}
require(len(state_without_external)==549 and sum(t.numel() for t in state_without_external.values())==54963422,'native deployment key geometry')
require(later['provenance']['source_sha256']==early['provenance']['source_sha256']==later['contract']['source_sha256'],'actual source encoding/version equality')
record={'schema':'flow_later_checkpoint_independent_source_tensor_audit_v1','status':'PASS','script_sha256':sha(Path(__file__)),
 'source_checkpoint':str(LATER),'checkpoint_sha256':sha(LATER),'checkpoint_bytes':LATER.stat().st_size,
 'earlier_checkpoint_sha256':sha(EARLY),'embedded_complete_contract_equal':True,'shared_contract_sha256':later['contract_sha256'],
 'same_all_state_names_shapes_dtypes':True,'all_711_tensors_finite_CPU':True,'native_state_tensors':711,'native_state_elements':55001588,
 'BN_buffer_count':234,'BN_module_count':78,'private_MLP_buffer_count':72,
 'source_native_state_sha256':digest(b),'source_non_BN_state_sha256':digest(b,bn),
 'deployment_expected_state_tensors':549,'deployment_expected_state_elements':54963422,
 'native_BN_state_shapes':{n:{'shape':list(b[n].shape),'dtype':str(b[n].dtype)} for n in sorted(bn)},
 'later_training':{k:later['training'][k] for k in ('completed_optimizer_steps','epoch','next_index','zero_valid_crops')},
 'later_stop_reason':later['stop_reason'],'optimizer_RNG_trajectory_replayed':False,
 'new_BN_generated':False,'inference_executed':False,'GPU_initialized':False,'quality_read':False,
 'scope':'Actual source PT bytes/embedded contract/tensor structures and non-BN values independently checked. No inference, BN calibration, training-progress causal or quality claim.'}
out=HERE/'LATER_SOURCE_TENSOR_AUDIT_V1.json';require(not out.exists(),'No overwrite');out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('status','checkpoint_sha256','source_native_state_sha256','source_non_BN_state_sha256','same_all_state_names_shapes_dtypes')},indent=2))
