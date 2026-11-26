#!/usr/bin/env python3
"""Independent CPU contract comparison. No inference or quality metrics."""
import hashlib,json
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
DATE=HERE.parent.parent
ROOT=DATE.parent
EARLY=ROOT/'flow_story_20261013/breadth/remote_snapshot'
LATE=DATE/'baseline_checkpoint_verification'
PT=DATE/'coverage_recovery_round13/source_snapshot/checkpoint_step_000135930.pt'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def require(v,msg):
 if not v:raise ValueError(msg)
early=read(EARLY/'run_contract.json');late=read(LATE/'run_contract.json')
checkpoint=torch.load(PT,map_location='cpu',weights_only=True)
require(sha(PT)=='d5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30','actual original checkpoint SHA')
contract=late['contract'];digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
require(digest==early['contract_sha256']==late['contract_sha256']==checkpoint['contract_sha256'],'same contract digest')
require(contract==early['contract']==checkpoint['contract'],'complete embedded original contract equality')
meta=read(LATE/'checkpoint_step_000271482.json');cpu=read(LATE/'cpu_checkpoint_audit.json');cpucontract=read(LATE/'cpu_contract_audit.json')
require(meta['contract_sha256']==cpu['contract_sha256']==cpucontract['canonical_contract_sha256']==digest,'later recorded embedded contract closure')
require(cpucontract['embedded_matches_run_contract'] is True and cpucontract['canonical_contract_digest_matches'] is True,'later primary CPU receipt')
require(meta['checkpoint_sha256']=='1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977','later specified checkpoint')
require(meta['completed_optimizer_steps']==cpu['training']['completed_optimizer_steps']==271482,'later fixed step')
require(checkpoint['training']['completed_optimizer_steps']==135930,'earlier fixed step')
require(contract['source_sha256']==checkpoint['provenance']['source_sha256']==cpucontract['provenance']['source_sha256'],'all source pins same')
config=contract['effective_config'];require(config['data_contract']['encoding_id']=='sdformerflow_upstream_voxel_10bins_cpu_v1','native encoding identity')
require(config['model']['num_bins']==10 and config['swin_transformer']['window_size']==[2,9,9],'native shape contract')
require(contract['scope']=='canonical6000' and contract['epochs']==80,'original bounded training protocol')
require(cpu['training']['epoch']*6000+cpu['training']['next_index']==271482,'later cursor arithmetic')
require(checkpoint['training']['epoch']*6000+checkpoint['training']['next_index']==135930,'original cursor arithmetic')
require(read(LATE/'segment_result.json')['stop_reason']=='wall_budget','later incomplete original target')
paths=[EARLY/'run_contract.json',LATE/'run_contract.json',LATE/'checkpoint_step_000271482.json',LATE/'cpu_checkpoint_audit.json',LATE/'cpu_contract_audit.json',LATE/'segment_result.json',PT]
r={'schema':'flow_later_checkpoint_independent_contract_comparison_v1','status':'PASS_COMPARABLE_DEPLOYMENT_PACKAGES',
 'script_sha256':sha(Path(__file__)),'source_sha256':{str(p):sha(p) for p in paths},'shared_complete_contract_sha256':digest,
 'complete_contract_equal_to_actual_135930_embedded':True,'source_dictionary_equal':True,
 'source_files':len(contract['source_sha256']),'encoding_id':config['data_contract']['encoding_id'],
 'encoding_contract_sha256':config['data_contract']['encoding_contract_sha256'],'canonical_manifest_sha256':contract['canonical_manifest_sha256'],
 'training_target_ids_sha256':contract['training_target_ids_sha256'],'earlier_completed_steps':135930,'later_completed_steps':271482,
 'later_epoch':cpu['training']['epoch'],'later_next_index':cpu['training']['next_index'],'training_target_epochs':80,
 'later_state_tensor_count_primary_receipt':cpu['model_state_dict_audit']['tensor_count'],
 'later_state_elements_primary_receipt':cpu['model_state_dict_audit']['elements'],
 'later_checkpoint_independently_deserialized_by_this_review':False,'earlier_checkpoint_independently_deserialized':True,
 'new_inference_or_calibration':False,'GPU_used':False,'quality_read':False,
 'limitations':['Later CPU deserialization/finite facts are prior primary receipts; later PT is not available locally in this review.',
 'No optimizer/RNG/order resume arithmetic replay or exact continuous-trajectory proof.',
 'Only deployment state-package comparison is supported: checkpoint and its own recalibrated BN change together.',
 'Neither original training target completion, paper reproduction gate nor official DSEC testing is established.',
 'New 176-event calibration/24private+54current-stat graph and same check8 source/runtime must be audited separately.']}
out=HERE/'CONTRACT_COMPARISON_V1.json';require(not out.exists(),'No overwrite');out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ('status','shared_complete_contract_sha256','source_files','later_checkpoint_independently_deserialized_by_this_review')},indent=2))
