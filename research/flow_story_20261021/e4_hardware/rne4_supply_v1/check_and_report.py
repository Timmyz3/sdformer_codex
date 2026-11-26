#!/usr/bin/env python3
"""Independent geometry/count/capacity checks, logical storage, and concise table."""
from pathlib import Path
from collections import defaultdict
import json, hashlib, math
P=Path(__file__).resolve().parent; R=P.parents[2]
load=lambda p:json.loads(p.read_text())
mask=load(R/'flow_story_20261020/round30/hardware/MASK_CONTRACT_V1.json')
inv=load(R/'flow_story_20261019/round27/hardware/UNIFIED_SERVICE_V1.json')['inventory']['affines']
results=load(P/'results/SUPPLY_REPLAY_V1.json')['results']
ceil=lambda n,d:(n+d-1)//d
geos=[(240,320,96,96,864),(30,40,384,1536,384),(30,40,1536,384,1536),(120,160,96,384,96)]
checks=[]; rowsummary=[]
for i,(h,w,cin,cout,k) in enumerate(geos):
 armset=results[i*5:(i+1)*5]; key=next(a for a,g in mask['geometries']['full9'].items() if g['H']==h and g['W']==w)
 sel=set(tuple(x) for x in mask['evaluation_tiles'][key]); total=10*h*w; complete=10*(h//9*9)*(w//9*9); tail=total-complete
 expectedrows=[(total,0),(total,0),(0,total),(tail,complete),(tail+len(sel)*810,complete-len(sel)*810)]
 for j,a in enumerate(armset):
  hi,lo=expectedrows[j]; lanes=16 if j==0 else 128; chunks=ceil(k,lanes); pad=chunks*lanes;obs=cout//16
  beats=total*obs*chunks if j==0 else (hi*8+lo*4)*obs*chunks
  assert(a['q8_rows'],a['q4_rows'])==(hi,lo)
  assert a['arithmetic_accepted_beats']==beats
  assert a['scale_RF_read_bytes']==total*obs*64
  assert a['output_store_ticks']==total*obs
  assert a['scale_vector_ticks']==total*obs*2
  assert a['weight_RF_read_bytes']==beats*256
  assert a['activation_RF_read_bytes']==beats*(4 if j==0 else 16)
  assert a['Q8_weight_scale_storage_bytes']==(0 if j==2 else pad*cout+4*cout)
  assert a['RNE4_weight_scale_storage_bytes']==(0 if j<2 else pad*cout//2+4*cout)
  assert a['Q8_weight_scale_storage_bytes']+a['RNE4_weight_scale_storage_bytes']<=1048576
  assert a['peak_input_slot_bytes']<=524288 and a['peak_output_slot_bytes']<=524288 and a['peak_weight_RF_bytes']<=32768
  assert a['tail_precision']==(4 if j==2 else 8)
  assert a['BN_full_domain_passes']==3
  b=a['BN_dependencies'];assert b['all_affine_done']<b['mean_finalized']<b['variance_finalized']<b['normalized_output_done']
  # Reconstruct refill/domain scheduling from exact geometry, independent of C++.
  fills8=fills4=descreads=controls=indexwrite=indexread=0
  batch=a['batch_rows'];grouped=a['row_schedule']=='batch_local_two_variant_partition'
  def bits(r):
   if j<2:return 8
   if j==2:return 4
   y=(r//w)%h;x=r%w
   if y>=h//9*9 or x>=w//9*9:return 8
   return 8 if j==4 and (y//9,x//9) in sel else 4
  for first in range(0,total,batch):
   rr=list(range(first,min(total,first+batch)));bb=[bits(r) for r in rr]
   if grouped:
    fills8+=obs*int(8 in bb);fills4+=obs*int(4 in bb)
    controls+=len(rr)+obs*(int(8 in bb)+int(4 in bb));indexwrite+=4*len(rr);indexread+=4*len(rr)*obs
   else:
    fills8+=obs*int(j!=2);fills4+=obs*int(j>=2)
    if j>=3:controls+=obs*sum(x!=y for x,y in zip(bb,[-1]+bb[:-1]))
   if j==4:
    previous=None;d=0
    for r in rr:
     y=(r//w)%h;x=r%w
     if y>=h//9*9 or x>=w//9*9:previous=None;continue
     idx=y//9*(w//9)+x//9
     if idx!=previous:d+=1;previous=idx
    descreads+=d*(1 if grouped else obs)
  w8=chunks*(1 if j==0 else 8)*256;w4=chunks*4*256
  assert a['weight_SRAM_read_bytes']==fills8*w8+fills4*w4
  assert a['weight_RF_prefill_busy_ticks']==fills8*w8//256+fills4*w4//256
  assert a['scale_SRAM_read_bytes']==(fills8+fills4)*64
  assert a['scale_RF_refill_ticks']==fills8+fills4
  assert a['variant_control_ticks']==controls
  assert a['descriptor_RF_read_bytes']==descreads*4
  assert a['descriptor_service_ticks']==descreads+(ceil(h//9*(w//9),64) if j==4 else 0)
  assert a['row_index_write_bytes']==indexwrite and a['row_index_read_bytes']==indexread
  assert a['row_index_service_ticks']==(indexwrite+indexread)//4
  checks.append({'operator':a['name'],'arm':a['arm'],'independent_row_and_beat_counts':'PASS','independent_refill_control_index_counts':'PASS','bounds_and_BN_order':'PASS'})
 strong=min(armset[:2],key=lambda x:x['affine_plus_BN_completion_ticks'])
 for a in armset:
  rowsummary.append(dict({k:a[k] for k in ['name','arm','row_schedule','q8_rows','q4_rows','arithmetic_accepted_beats','affine_plus_BN_completion_ticks','affine_completion_ticks','Q8_weight_scale_storage_bytes','RNE4_weight_scale_storage_bytes','peak_weight_RF_bytes','DMA_bytes','variant_control_ticks','descriptor_service_ticks','row_index_service_ticks']},strong_Q8_arm=strong['arm'],speedup_vs_strong_Q8=strong['affine_plus_BN_completion_ticks']/a['affine_plus_BN_completion_ticks'],reduction_vs_strong_Q8=1-a['affine_plus_BN_completion_ticks']/strong['affine_plus_BN_completion_ticks']))
ledger=[]
for a in inv:
 if a['scope'] not in ('patch4','MLP12'):continue
 k,co,h,w=a['K'],a['Cout'],a['H'],a['W']; key=a['weight'][:-7]
 if a['scope']=='MLP12':key=key.rsplit('.',1)[0]
 g=mask['geometries']['full9'][key]; assert(h,w)==(g['H'],g['W'])
 q8=k*co;s=4*co;q4=k*co//2;pad=ceil(k,128)*128
 physical8=pad*co+s;physical4=pad*co//2+s
 complete=10*(h//9*9)*(w//9*9);tail=a['rows']-complete;mh=g['Q8_complete_tiles']*810
 rfhi=ceil(ceil(k,128)*8*256+64,256)*256;rflo=ceil(ceil(k,128)*4*256+64,256)*256
 ledger.append({'name':a['name'],'K':k,'Cout':co,'rows':a['rows'],'logical_Q8_weight_bytes':q8,'logical_Q4_packed_weight_bytes':q4,'each_scale_bytes':s,'logical_dual_weight_scale_bytes':q8+q4+2*s,'padded_Q8_weight_scale_bytes':physical8,'padded_Q4_weight_scale_bytes':physical4,'dual_single_SRAM_slot_fits':physical8+physical4<=1048576,'Q8_single_SRAM_slot_fits':physical8<=1048576,'Q4_single_SRAM_slot_fits':physical4<=1048576,'Q8_output16_RF_fits':rfhi<=32768,'Q4_output16_RF_fits':rflo<=32768,'dual_output16_RF_fits':rfhi+rflo<=32768,'Q4_actual_Q8_tail_rows':tail,'M_Q8_rows':tail+mh,'M_Q4_rows':complete-mh,'cycle_coverage':'only representative geometries in four explicit domains; no whole-network sum'})
assert len(ledger)==28
storage={'status':'LOGICAL_STORAGE_NOT_WHOLE_NETWORK_CYCLES','operators':ledger,'totals':{k:sum(a[k] for a in ledger) for k in ['logical_Q8_weight_bytes','logical_Q4_packed_weight_bytes','each_scale_bytes','logical_dual_weight_scale_bytes']},'needs_weight_DMA_striping':[a['name'] for a in ledger if not a['dual_single_SRAM_slot_fits']],'needs_Q8_RF_reduction_chunking':[a['name'] for a in ledger if not a['Q8_output16_RF_fits']]}
(P/'results/ACCOUNTING_CHECKS_V1.json').write_text(json.dumps({'status':'PASS','arms':checks,'scope':'integer counts/address model; no numeric output or RTL validation'},indent=2)+'\n')
(P/'results/STORAGE_LEDGER_28_V1.json').write_text(json.dumps(storage,indent=2)+'\n')
(P/'results/SUMMARY_V1.json').write_text(json.dumps({'status':'FINITE_OPERATOR_MODEL_NOT_NETWORK_OR_RTL','results':rowsummary},indent=2)+'\n')
for i in range(4):
 print(rowsummary[i*5]['name'])
 for a in rowsummary[i*5:i*5+5]:print(a['arm'],a['affine_plus_BN_completion_ticks'],'speedup',round(a['speedup_vs_strong_Q8'],6),'RF',a['peak_weight_RF_bytes'],'DMA',a['DMA_bytes'])
print('storage totals',storage['totals']);print('slot overflow',len(storage['needs_weight_DMA_striping']),'RF overflow',len(storage['needs_Q8_RF_reduction_chunking']))
