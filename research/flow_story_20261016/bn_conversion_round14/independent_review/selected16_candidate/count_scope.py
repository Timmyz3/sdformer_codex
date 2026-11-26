#!/usr/bin/env python3
"""Geometry-only BN scope cost comparison; no inference, cycles or quality."""
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
REPO=ROOT.parents[1]
GFILE=ROOT/'coverage_recovery_round13/independent_execution_audit/audit_execution.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(GFILE)=='dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8'
spec=importlib.util.spec_from_file_location('_independent_true_geometry',GFILE);G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
rows=[]
for name,(t,b,h,w,c) in G.GEOMETRY.items():
 selected=name in G.SELECTED
 for ordinal,width in ((1,4*c),(2,c)):
  n=t*b*h*w;rows.append({'name':name+'.bn'+str(ordinal)+'.norm_layer','selected':selected,'shape_TBCHW':[t,b,width,h,w],
   'channels':width,'elements':n*width,'single_complete_FP32_result_bytes':4*n*width})
selected=[x for x in rows if x['selected']];rest=[x for x in rows if not x['selected']]
assert len(selected)==16 and len(rest)==8
assert sum(x['elements'] for x in selected)==322560000 and sum(x['elements'] for x in rest)==115200000
channels=sum(x['channels'] for x in selected);assert channels==12480
fixed={'running_FP32_values':channels*2,'int64_counters':16,'buffer_tensors':48,'buffer_elements_including_counters':channels*2+16,
 'buffers_bytes':channels*2*4+16*8,'full_deployed_state_tensors':477+48,'full_deployed_state_elements':54919238+channels*2+16}
assert fixed['full_deployed_state_tensors']==525 and fixed['full_deployed_state_elements']==54944214
representative={}
for arm in ('native','CC','fine_s2b2'):
 r=[]
 for name,shape in G.GEOMETRY.items():
  if name not in G.SELECTED:continue
  executed=G.expected_layer(name,arm)[0];c=shape[-1]
  r.append({'name':name,'FC_rows':executed,'BN1_output_elements':executed*4*c,'BN2_output_elements':executed*c,
   'BN1_single_result_bytes':executed*4*c*4,'BN2_single_result_bytes':executed*c*4})
 representative[arm]={'selected_private_BN_elements':sum(v['BN1_output_elements']+v['BN2_output_elements'] for v in r),'layers':r}
assert representative['native']['selected_private_BN_elements']==322560000
assert representative['CC']['selected_private_BN_elements']==36864000
assert representative['fine_s2b2']['selected_private_BN_elements']==57216000
sources=[GFILE,REPO/'research/flow_story_20261014/structured_merge/structured_merge_mlp.py',REPO/'research/flow_story_20261016/ordinary_recovery_round12/trainable_representative_mlp.py',
 REPO/'SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py',REPO/'SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_submodules.py',REPO/'SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py']
out={'schema':'flow_selected16_bn_geometry_screen_v1','status':'CANDIDATE_ONLY_NO_QUALITY_OR_CYCLE_CLAIM','script_sha256':sha(__file__),
 'source_sha256':{str(p):sha(p) for p in sources},'all12_private_BN':rows,'selected16_fixed_buffers':fixed,
 'new_current8_vs_fixed24':{'statistic_barriers':8,'BN_input_elements':sum(v['elements'] for v in rest),'single_input_representation_bytes':sum(v['single_complete_FP32_result_bytes'] for v in rest),
  'largest_single_result_bytes':max(v['single_complete_FP32_result_bytes'] for v in rest),'layers':rest},
 'policy_counts':{'current78':{'current':78,'fixed':0},'private24_fixed':{'current':54,'fixed':24},'selected16_fixed':{'current':62,'fixed':16},
  'weighted_global_candidate':{'ordinary_current':62,'selected_weighted_current':16,'fixed':0}},
 'selected_representative_geometry':representative,
 'limitations':['Element counts and single-complete-result capacities, not measured traffic, SRAM requirement, cycles or energy.',
 'Weighted-global requires staged FC1/BN1/PSN2/FC2/BN2; selected16 fixed can keep existing selected producer tile ordering.',
 'The existing54-BN cycle model misses eight current-stat private BNs for this candidate; no full-network denominator is closed.',
 'Candidate not executed; current result quality not read or used to generate these counts.']}
p=Path(__file__).parent/'GEOMETRY_COST_V1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({k:out[k] for k in ('status','selected16_fixed_buffers')},indent=2));print('json_sha256',sha(p))
