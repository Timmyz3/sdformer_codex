#!/usr/bin/env python3
"""No quality reading: independent actual-mask/count checks and budget summary."""
from pathlib import Path
import hashlib,json,itertools,math
H=Path(__file__).resolve().parent
C=H.parent.parent/'e2_group_damage/PATCH6_CONTRACT_V1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def req(ok,msg):
 if not ok:raise ValueError(msg)
req(sha(C)=='a6926959cd24167e73fff38c1f7bb0024955cf9c035a50fd10ed081703bb862a','Frozen actual E2 contract')
c=json.loads(C.read_text());d=json.loads((H/'results/SUPPLY_V1.json').read_text());old=json.loads((H.parent/'results/SUPPLY_REPLAY_V1.json').read_text())
bg=d['results'][0];oldbg=next(r for r in old['results'] if r['name']=='patch.res0.conv0' and r['engine']=='bitplane128' and r['precision_complete']==6)
req(all(bg[k]==v for k,v in oldbg.items()),'Unchanged E4 background exactly replays every old field')
req(len(d['results'])==46,'One background plus15pairs x3 control arms')
target=c['family_targets'][0];groups={g['id']:set(map(tuple,g['tiles'][target])) for g in c['groups']}
req(all(len(x)==151 for x in groups.values()) and len(set.union(*groups.values()))==906,'6 disjoint151-tile groups')
complete=737100;tail=30900;low=302*81*10;mid=complete-low
expected=(4*low+6*mid+8*tail)*7*6
req(bg['arithmetic_accepted_beats']-expected==c['pair_cancelled_kernel_ticks']//4,'Exact actual contract cancellation')
lookups=0
for r in range(768000):
 y=(r//320)%240;x=r%320
 if y<234 and x<315 and (x%9==0 or r%128==0):lookups+=6
req(lookups==519480,'Independent complete tile-span lookup count incl batch breaks')
rows=[]
for a,b in itertools.combinations(range(6),2):
 arms={r['control']:r for r in d['results'] if r['pair']==[a,b]};req(set(arms)=={0,1,2},'All control baselines')
 selected=groups[a]|groups[b];req(len(selected)==302,'No overlap')
 for ctl,r in arms.items():
  req(r['arithmetic_accepted_beats']==expected,'All pair arithmetic equal')
  req(r['descriptor_word_reads']==(lookups if ctl else 0),'Exact descriptor word reads')
  req(r['descriptor_control_ticks']==lookups*ctl,'Paid read/decode cycles')
  req(r['descriptor_read_bytes']==(lookups*4 if ctl else 0),'32bit stats-RF word granularity')
  req(r['DMA_bytes']==bg['DMA_bytes']+(960 if ctl else 0),'Only table DMA added; all8 weights retained')
  req(r['pair_mask_setup_ticks']==(3 if ctl==2 else 0),'Dynamic pair setup separate from unknown predictor')
  req(r['weight_SRAM_read_bytes']==bg['weight_SRAM_read_bytes'],'No claimed weight refill cancellation')
  req(r['BN_dependencies']['normalized_output_done']==r['affine_plus_BN_completion_ticks'],'Same full BN endpoint')
  for beat in r['accepted_beat_trace']:
   y=(beat['row']//320)%240;x=beat['row']%320
   bits=8 if (y>=234 or x>=315) else 4 if (y//9,x//9) in selected else 6
   req(beat['bits']==bits and beat['plane']>=8-bits,'Saved actual mask row precision and no low-plane request')
  dma=sorted((e for e in r['trace_prefix'] if e['resource']=='DMA'),key=lambda e:e['start']);req(all(x['end']<=y['start'] for x,y in zip(dma,dma[1:])),'Sample DMA exclusive')
  desc=[e for e in r['trace_prefix'] if e['kind']=='tile_descriptor'];req(all(e['address']%4==0 and e['address']+4<=960 and e['end']-e['start']==ctl for e in desc),'Saved descriptor addresses and paid cycles')
 dyn=arms[2]['affine_plus_BN_completion_ticks'];fixed=arms[1]['affine_plus_BN_completion_ticks'];lower=arms[0]['affine_plus_BN_completion_ticks'];base=bg['affine_plus_BN_completion_ticks']
 rows.append({'pair':[a,b],'single_operator':{'Q6_background_ticks':base,'static_compiled_lower_bound_ticks':lower,'static_precision_table_ticks':fixed,'dynamic_group_table_ticks':dyn,'additional_serial_cost_must_be_less_than_for_any_gain_vs_Q6':base-dyn,'dynamic_overhead_vs_same_pair_static_table':dyn-fixed,'dynamic_overhead_vs_static_compiled_lower_bound':dyn-lower},'four_isolated_operator_service_sums_not_network_latency':{'Q6_background':4*base,'dynamic_pair':4*dyn,'maximum_additional_serial_budget_for_any_gain_vs_Q6_exclusive':4*(base-dyn),'dynamic_overhead_vs_same_pair_static_table':4*(dyn-fixed),'dynamic_overhead_vs_static_compiled_lower_bound':4*(dyn-lower)}})
summary={'schema':'flow_e2_patch6_supply_budget_v1','status':'PASS_COUNTS_AND_BOUNDED_SERVICE_MODEL','contract_sha256':sha(C),'new_GPU_forwards':0,'GT_read':False,'pair_core_cancelled_ticks_all_four':c['pair_cancelled_kernel_ticks'],'single_conv_Q4_rows':low,'single_conv_Q6_rows':mid,'single_conv_Q8_tail_rows':tail,'single_conv_descriptor_word_reads':lookups,'single_conv_dynamic_descriptor_control_ticks':2*lookups,'single_conv_descriptor_DMA_bytes':960,'pairs':rows,'limits':['Additional budget excludes equality: a scorer at the stated maximum leaves zero gain vsQ6. It is not implemented or measured.','Four isolated operator service sums omit intervening PSN/residual dependencies and are not whole-network latency.','Same pair static controller is always cheaper. Runtime selection needs actual accuracy/cost advantage across examples, not merely positive budget vsQ6.','All values use nested truncation, never independent-scale RNE4 costs.']}
(H/'results/SUMMARY_V1.json').write_text(json.dumps(summary,indent=2)+'\n')
checks={'schema':'flow_e2_patch6_supply_checks_v1','status':'PASS','complete_operator_arms':46,'exact_background_replay':True,'all15_pair_mask_and_work_counts':True,'descriptor_span_count':lookups,'source_binding':{str(p):sha(p) for p in [C,H/'patch6_supply.cpp',H/'tile_groups.h',H/'results/SUPPLY_V1.json',H/'results/SUMMARY_V1.json']},'independent_agent_review':False,'full_RTL_added':False}
(H/'results/CHECKS_V1.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'status':'PASS','pair_service_range':[min(r['single_operator']['dynamic_group_table_ticks'] for r in rows),max(r['single_operator']['dynamic_group_table_ticks'] for r in rows)],'four_op_max_extra_serial_budget_range':[min(r['four_isolated_operator_service_sums_not_network_latency']['maximum_additional_serial_budget_for_any_gain_vs_Q6_exclusive'] for r in rows),max(r['four_isolated_operator_service_sums_not_network_latency']['maximum_additional_serial_budget_for_any_gain_vs_Q6_exclusive'] for r in rows)]}))
