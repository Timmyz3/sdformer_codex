#!/usr/bin/env python3
from pathlib import Path
import json
P=Path(__file__).resolve().parent;R=P.parents[2]
load=lambda p:json.loads(p.read_text())
d=load(P/'results/SUPPLY_REPLAY_V1.json');rs=d['results'];old=load(P.parent/'rne4_supply_v1/results/SUPPLY_REPLAY_V1.json')['results']
bg=rs[0];prior=next(x for x in old if x['name']=='patch.res0.conv0' and x['arm']=='Q4_complete_Q8_tail_dual')
for k in prior:
 if k in ('trace_prefix','accepted_beat_trace','address_fnv64'):continue
 assert bg[k]==prior[k],('background changed',k)
strong=min(x['affine_plus_BN_completion_ticks'] for x in old if x['name']=='patch.res0.conv0' and x['arm'].startswith('Q8_'))
source=load(R/'flow_story_20261021/e3_rne_probe/PREPARED_CONTRACT_V1.json');key=source['family_targets'][0]
checks=[];summary=[]
for m in range(6):
 selected=set(tuple(t) for t in source['groups'][m]['tiles'][key]);assert len(selected)==151
 expected_control=0;desc=0
 for first in range(0,768000,128):
  previous_bit=-1;previous_id=None
  for r in range(first,min(first+128,768000)):
   y=r//320%240;x=r%320;full=y<234 and x<315
   bits=8 if not full or (y//9,x//9) in selected else 4
   if bits!=previous_bit:expected_control+=6;previous_bit=bits
   if full:
    idx=y//9*35+x//9
    if idx!=previous_id:desc+=6;previous_id=idx
   else:previous_id=None
 for dyn in [False,True]:
  a=next(x for x in rs if x['selected_macro']==m and x['dynamic_selection']==dyn)
  assert(a['q8_rows'],a['q4_rows'])==(153210,614790)
  assert a['arithmetic_accepted_beats']-bg['arithmetic_accepted_beats']==20548080
  assert a['arithmetic_accepted_beats']==(153210*8+614790*4)*7*6
  assert a['variant_control_ticks']==expected_control
  assert a['descriptor_RF_read_bytes']==desc*4
  assert a['descriptor_service_ticks']==desc+15
  assert a['group_decode_ticks']==(desc if dyn else 0)
  assert a['selection_latch_ticks']==int(dyn)
  assert a['DMA_bytes']-bg['DMA_bytes']==960+(64 if dyn else 0)
  assert a['weight_SRAM_read_bytes']==bg['weight_SRAM_read_bytes']
  assert a['scale_SRAM_read_bytes']==bg['scale_SRAM_read_bytes']
  assert a['peak_weight_RF_bytes']==22016
  assert a['Q8_weight_scale_storage_bytes']+a['RNE4_weight_scale_storage_bytes']==129792
  assert a['scale_RF_read_bytes']==768000*6*64
  assert a['row_index_read_bytes']==a['row_index_write_bytes']==0
  assert a['BN_dependencies']['normalized_output_done']==a['affine_plus_BN_completion_ticks']
  checks.append({'macro':m,'dynamic':dyn,'status':'PASS','independent_full_rows_controls_descriptors_beats':True})
 static,dynamic=rs[1+m*2:3+m*2]
 assert dynamic['affine_plus_BN_completion_ticks']-static['affine_plus_BN_completion_ticks']==desc+34+(dynamic['input_DMA_blocks_PE_ticks']-static['input_DMA_blocks_PE_ticks'])+(dynamic['output_DMA_blocks_PE_ticks']-static['output_DMA_blocks_PE_ticks'])
 summary.append({'macro':m,'selected_complete_tiles':151,'selected_rows':122310,'added_core_beats_per_operator':20548080,'four_operator_added_core_beats_not_network_ticks':82192320,'static_affine_BN_ticks':static['affine_plus_BN_completion_ticks'],'dynamic_affine_BN_ticks':dynamic['affine_plus_BN_completion_ticks'],'dynamic_over_same_static_ticks':dynamic['affine_plus_BN_completion_ticks']-static['affine_plus_BN_completion_ticks'],'dynamic_over_Q4_background_ticks':dynamic['affine_plus_BN_completion_ticks']-bg['affine_plus_BN_completion_ticks'],'remaining_ticks_to_two_rule_Q8_reference':strong-dynamic['affine_plus_BN_completion_ticks'],'predictor_cycles_included':False})
(P/'results/ACCOUNTING_CHECKS_V1.json').write_text(json.dumps({'status':'PASS','Q4_background_equal_frozen_supply':True,'checks':checks},indent=2)+'\n')
report={'schema':'rne4_macro6_cost_summary_v1','Q4_background_affine_BN_ticks':bg['affine_plus_BN_completion_ticks'],'two_rule_Q8_reference_affine_BN_ticks':strong,'reference_scope':'faster of parallel16 and bitplane128; not lower bound on all sparse accelerators','candidates':summary,'scope':'same complete operator; four isolated domains sum is not network latency','predictor_implemented':False,'algorithm_quality_measured_here':False,'endpoint_bound':False,'cost_interpretation':'Q8 upgrades buy accuracy at positive cost versus Q4; dynamic control has no free margin versus same static map'}
(P/'results/SUMMARY_V1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
