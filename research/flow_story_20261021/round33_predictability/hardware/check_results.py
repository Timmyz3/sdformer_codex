#!/usr/bin/env python3
"""Independent arithmetic geometry/bitmap/port-work golden; not RTL reimplementation."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[2]
load=lambda p:json.loads(p.read_text())
words=load(P/'BITMAP_CONTENT_V1.json')['words'];rs=load(P/'results/SUPPLY_REPLAY_V1.json')['results']
assert len(rs)==14
for a in range(6):
 for tile in range(1024):
  assert bool((words[a*32+tile//32]>>(tile%32))&1)==(a*151<=tile<(a+1)*151)
rtl=[load(P/'results'/('rtl_'+k)/'RESULT.json') for k in ['dynamic','static','none']]
assert all(x['status']=='PASS' and x['no_stall_bypass_span_ticks']==1 for x in rtl)
assert rtl[0]['no_stall_active_span_ticks']==rtl[1]['no_stall_active_span_ticks']==3
assert rtl[2]['no_stall_active_span_ticks'] is None
old=load(R/'flow_story_20261021/e4_hardware/rne4_supply_v1/results/SUPPLY_REPLAY_V1.json')['results']
oldbg=next(x for x in old if x['name']=='patch.res0.conv0' and x['arm']=='Q4_complete_Q8_tail_dual')
summary=[];checks=[]
for a in range(7):
 expected8=30900+(122310 if a<6 else 0);expected4=768000-expected8
 full=bypass=0;histbase=[0]*32
 for first in range(0,768000,128):
  previous=None
  for r in range(first,first+128):
   y=r//320%240;x=r%320;complete=y<234 and x<315
   tile=(y//9)*35+x//9 if complete else 910
   key=911 if complete and a==6 else tile
   if key==previous:continue
   if complete and a<6:full+=6;histbase[tile//32]+=6
   else:bypass+=6
   previous=key
 pair=[]
 for dyn in [False,True]:
  z=next(x for x in rs if x['selected_action']==a and x['dynamic_selection']==dyn)
  assert(z['q8_rows'],z['q4_rows'])==(expected8,expected4)
  beats=(expected8*8+expected4*4)*7*6
  assert z['arithmetic_accepted_beats']==beats
  assert z['weight_RF_read_bytes']==beats*256 and z['activation_RF_read_bytes']==beats*16
  assert z['full_bitmap_queries']==full and z['registered_bypass_queries']==bypass
  assert z['query_service_ticks']==3*full+bypass
  assert z['descriptor_RF_read_bytes']==4*full
  storage=(768 if dyn else 128) if a<6 else 0
  assert z['descriptor_storage_bytes']==storage
  assert z['descriptor_init32bit_writes']==storage//4
  assert z['descriptor_service_ticks']==3*full+bypass+storage//4
  assert z['variant_control_ticks']==0 # replaced by actual response latch protocol
  assert z['boundary_handshake_ticks']==2
  assert z['DMA_bytes']==oldbg['DMA_bytes']+storage
  assert z['DMA_transactions']==oldbg['DMA_transactions']+int(a<6)
  assert z['DMA_busy_ticks']==oldbg['DMA_busy_ticks']+(32+storage//64 if a<6 else 0)
  assert z['weight_SRAM_read_bytes']==oldbg['weight_SRAM_read_bytes']
  assert z['scale_SRAM_read_bytes']==oldbg['scale_SRAM_read_bytes']
  assert z['scale_RF_read_bytes']==oldbg['scale_RF_read_bytes']
  assert z['weight_RF_prefill_busy_ticks']==oldbg['weight_RF_prefill_busy_ticks']
  assert z['Q8_weight_scale_storage_bytes']+z['RNE4_weight_scale_storage_bytes']==129792
  assert z['peak_weight_RF_bytes']==22016
  assert z['row_index_read_bytes']==z['row_index_write_bytes']==0
  assert z['peak_input_slot_bytes']<=524288 and z['peak_output_slot_bytes']<=524288
  hist=[0]*192
  if a<6:hist[(a*32 if dyn else 0):(a*32 if dyn else 0)+32]=histbase
  assert z['bitmap_word_read_histogram']==hist
  assert sum(hist)*4==z['descriptor_RF_read_bytes']
  deps=z['BN_dependencies'];assert deps['all_affine_done']<deps['mean_finalized']<deps['variance_finalized']<deps['normalized_output_done']
  assert z['BN_full_domain_passes']==3
  if a==6:assert z['affine_plus_BN_completion_ticks']==oldbg['affine_plus_BN_completion_ticks']+2
  checks.append({'action':a,'dynamic':dyn,'status':'PASS','actual_bit_contents_and_all_word_addresses':True,'independent_row_core_control_init_DMA_counts':True,'limits_and_BN_order':True})
  pair.append(z)
 st,dy=pair;delta=dy['affine_plus_BN_completion_ticks']-st['affine_plus_BN_completion_ticks']
 assert delta==(170 if a<6 else 0)
 for k in ['arithmetic_accepted_beats','query_service_ticks','input_DMA_blocks_PE_ticks','output_DMA_blocks_PE_ticks','full_bitmap_queries','registered_bypass_queries']:
  assert st[k]==dy[k]
 summary.append({'action':a,'static_affine_BN_ticks':st['affine_plus_BN_completion_ticks'],'dynamic_affine_BN_ticks':dy['affine_plus_BN_completion_ticks'],'dynamic_over_same_static_ticks':delta,'full_bitmap_queries':full,'registered_bypass_queries':bypass,'dynamic_minus_static_is_only_serialized_bitmap_init':True})
checks_out={'status':'PASS','checks':checks,'RTL_requests_from_actual_bitmaps':True,'word_histogram_independent_golden':True,'whole_network':False,'scoring_head_included':False}
(P/'results/ACCOUNTING_CHECKS_V1.json').write_text(json.dumps(checks_out,indent=2)+'\n')
dc={k:load(P/'results'/('dc_'+k)/'summary.json') for k in ['dynamic','static','none']}
assert all(v['clock_period_ns']==4 and v['worst_setup_slack_ns']>0 for v in dc.values())
report={'schema':'round33_bitmap_control_result_v1','status':'EXECUTED_CPU_RTL_DC_ISOLATED_CONTROL_AND_OPERATOR_MODEL','candidates':summary,'RTL':rtl,'DC':dc,'predictor_execution_included':False,'feature_reads_included':False,'head_weights_storage_included':False,'whole_network':False,'old513674_result_replaced':False,'ordinary_comparison':'same bitpack, read32, extract, address, response and ports; dynamic conservatively initializes six masks, static one; none initializes zero'}
(P/'results/SUMMARY_V1.json').write_text(json.dumps(report,indent=2)+'\n')
for x in summary:print(x)
print('all14 independent checks PASS; isolated controller RTL and DC PASS')
