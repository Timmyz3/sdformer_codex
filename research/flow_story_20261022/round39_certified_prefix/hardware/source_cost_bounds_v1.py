#!/usr/bin/env python3
"""R39 pre-result source decomposition. No raw arrays, selectors, GT or GPU.

This is a lower-cost relaxation of one prescribed frozen-resource rowband
schedule, not a lower bound over all possible hardware or summation schemes.
"""
from pathlib import Path
import hashlib,json,math
P=Path(__file__).resolve().parent
R=P.parents[2]
ROOT=R.parent
OLD=ROOT/'research/flow_story_20261021/round34_deployability/hardware'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ticks(d):return sum(v*(3 if k in ('add','sub','mul') else 16 if k in ('div','sqrt') else 1) for k,v in d.items())
N=19200;T=10;C=16
cheap={'add':C*(2*(T*N-1)+T*(N-1)+3*(T-1)),'abs':C*T*N,'compare':C*(2*T*N+1),'sign':C*T*N,'integer_count_update':C*2*T*N,'convert':C*2,'mul':C*T*3,'div':C*6}
late={'add':C*((T-1)*N+((T-1)*N-1)+(N-1)),'sub':C*(T-1)*N,'abs':C*T*N,'div':C*3}
base=json.loads((OLD/'actual_v2/rowband128.json').read_text())
combined={k:cheap.get(k,0)+late.get(k,0) for k in set(cheap)|set(late)}
assert combined==base['ops'],(combined,base['ops'])
# Vector IOS->RES copies: 16 words/tick, each port serialized in old source.
dma=C*T*N*4//64+40*32
copy=C*T*N//16*2
gather=dma+copy
cheap_mem={'raw_RES_read':C*T*N,'raw_IOS_write':C*T*N,'raw_IOS_read':C*T*N,'feature_IOS_write':C*6}
late_mem={'raw_RES_read':C*T*N,'raw_IOS_write':C*T*N,'raw_IOS_read':C*T*N,'prev_RES_read':C*(T-1)*N,'prev_IOS_write':C*(T-1)*N,'prev_IOS_read':C*(T-1)*N,'sum_RES_read':C*(T-1)*N,'sum_IOS_write':C*(T-1)*N,'sum_IOS_read':C*(T-1)*N,'sum_next_IOS_write':C*T*N,'sum_next_IOS_read':C*T*N,'sum_next_RES_write':C*T*N,'prev_next_RES_write':C*T*N,'temporal_RES_read':C*N,'feature_IOS_write':C*2}
# Original fused source consumes raw once; split late path duplicates this.
duplicated_raw=3*C*T*N
original_memory=sum(cheap_mem.values())+sum(late_mem.values())-duplicated_raw+copy
assert original_memory==sum(m['serial_service_ticks'] for m in base['memory'])
assert ticks(combined)+original_memory+dma==base['feature_ready_tick']
first=ticks(cheap)+sum(cheap_mem.values())+gather
per=(ticks(late)+sum(late_mem.values()))//C
assert (ticks(late)+sum(late_mem.values()))%C==0
head=json.loads((OLD/'actual_v2/HEAD_PREFIX_RESULTS_V1.json').read_text())['models']
heads={'folded128':head['X128_folded_constant_v']['action_ready_tick'],'ordinary128':head['X128_calibrated_residual_g1']['action_ready_tick']}
assert heads=={'folded128':8599,'ordinary128':26646}
rows=[]
for k in range(17):
 bands=(k+3)//4
 floor=first+k*per+bands*(gather//4)
 row={'completed_late_cells':k,'second_pass_rowbands':bands,'second_pass_DMA_bytes':bands*3072000,'second_pass_DMA_transactions':bands*10,'feature_source_floor_ticks_no_zero_cells':floor,'feature_source_floor_ticks_all_zero_cells':floor-512,'delta_vs_original_features_ticks':floor-base['feature_ready_tick'],'free_certificate_prefix_saving_by_head':{}}
 for name,h in heads.items():
  row['free_certificate_prefix_saving_by_head'][name]={}
  for z in (0,16):
   original=base['feature_ready_tick']-32*z+h
   new=floor-32*z
   row['free_certificate_prefix_saving_by_head'][name][f'zero_mass_cells_{z}']={'old_full_prefix_ticks':original,'optimistic_new_ticks':new,'saved_ticks':original-new,'saved_fraction':(original-new)/original,'passes_10pct_necessary_gate':10*(original-new)>=original}
 rows.append(row)
assert rows[-1]['feature_source_floor_ticks_no_zero_cells']-base['feature_ready_tick']==duplicated_raw+gather
out={
 'schema':'r39_source_cost_relaxation_v1','status':'pre-real-result source decomposition only',
 'scope':'one R34 frozen-resource/source rowband schedule with all interval, metadata, head and control services optimistically zero; not a universal hardware lower bound',
 'raw_inputs_read':0,'actual_selector_executions':0,'GT_read':False,'new_RTL':False,
 'sources':{str(p.relative_to(ROOT)):sha(p) for p in [OLD/'stream_features_v2.cpp',OLD/'head_prefix_v2.py',OLD/'actual_v2/rowband128.json',OLD/'actual_v2/HEAD_PREFIX_RESULTS_V1.json',P/'source_cost_bounds_v1.py']},
 'original_feature_ticks':base['feature_ready_tick'],'original_head_ticks':heads,
 'geometry':{'cells':C,'T':T,'spatial_per_cell':N,'all_input_words':C*T*N,'raw_input_bytes':C*T*N*4,'rowband_bytes':3072000,'RES_capacity_bytes':4194304,'IOS_capacity_bytes':524288,'scalar_resource_bytes_assumed_not_instrumented':2048},
 'original_exact_reconciliation':{'ops_match':True,'memory_service_match':True,'total_ticks_match':True,'arithmetic_ticks':ticks(combined),'memory_excluding_DMA_ticks':original_memory,'DMA_ticks':dma},
 'cheap_pass':{'operations':cheap,'arithmetic_ticks':ticks(cheap),'scalar_memory_service_ticks_by_role':cheap_mem,'all_frame_gather':{'DMA_ticks':dma,'IOS_to_RES_two_port_serial_copy_ticks':copy,'combined_ticks':gather},'total_ticks_nonzero_mass':first,'zero_mass_cell_discount':32},
 'late_full_pass':{'operations':late,'arithmetic_ticks':ticks(late),'scalar_memory_service_ticks_by_role':late_mem,'per_cell_body_ticks':per,'per_new_rowband_DMA_and_copy_ticks':gather//4,'all_16_cells_with_reread_ticks':per*C+gather},
 'schedule_rows':rows,
 'mandatory_positive_costs_omitted_for_negative_screen':['192 predivision positive/negative sums and mass values saved if bounds not immediately consumed: 768B plus read/write service','128-feature interval endpoints, interval scalar scratch and bound constants','bound arithmetic including explicit outward nextafter and numerical guards','each sequential FP32 interval head and coefficient/endpoint port accesses','certification comparisons and lowest-ID tie handling','completion-to-check queue transitions and any fallback duplicated reference operations'],
 'capacity_countercondition':{'pin_last_rowband_plus_one_cell_plus_prev_sum_bytes':3072000+768000+153600,'fits_4MiB':3072000+768000+153600<4194304,'native_T_major_cell_gather_transactions':1200,'cell_gather_DMA_ticks':1200*32+768000//64,'cell_gather_two_port_copy_ticks':768000//64*2,'first_12_cell_gather_ticks':12*(1200*32+3*768000//64),'rowband_all_16_gather_ticks':gather,'explanation':'retaining last band while cell-gathering first12 uses less external bytes but more startup service; this is only a geometry comparison, not an implemented second scheduler'},
 'numerical_limit':'Old C++ features differ slightly from primary NumPy FP32 reference. This partition may reject the prescribed source-like configuration but cannot establish a positive exact-policy deployment. Such a claim requires a new exact chain and actual memory replay.',
}
(P/'SOURCE_COST_BOUNDS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'cheap_pass':first,'late_body_per_cell':per,'rowband_refetch':gather//4,'full_miss':rows[-1]['feature_source_floor_ticks_no_zero_cells'],'source_reconciles':True,'json_sha256':sha(P/'SOURCE_COST_BOUNDS_V1.json')},indent=2))
