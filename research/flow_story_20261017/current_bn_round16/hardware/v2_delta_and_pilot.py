#!/usr/bin/env python3
"""Independent geometry-only V2 inventory. No V2 schedule or forward executed."""
from collections import Counter
from pathlib import Path
import json
import service_model as m

HERE=Path(__file__).resolve().parent

def main():
 report_path=HERE/'actual_v1/SERVICE_WORK_REPORT.json'
 report=json.loads(report_path.read_text());g=m.geometry();arms={}
 for arm in ('FF','CC','P','checker'):
  rows=[];compact_work=Counter();dense_work=Counter();dense_elements=compact_elements=coarse_elements=bn1_compact_elements=0
  for name,(t,b,h,w,c),selected in g:
   if not selected:continue
   kind='FF' if arm=='FF' or (arm=='P' and '.layers.2.swin_blocks.2.' in name) else ('checker' if arm=='checker' else 'CC')
   points=m.points(h,w,m.masks(h,w,kind));r=len(points);coarse=sum(not p['fine'] for p in points)
   for bn,width in (('bn1',4*c),('bn2',c)):
    n=t*b*r;nd=t*b*h*w;cr=t*b*coarse;ec=n*width;ed=nd*width
    compact_elements+=ec;dense_elements+=ed;coarse_elements+=cr*width
    if bn=='bn1':bn1_compact_elements+=ec
    for target,nn,cc in ((compact_work,n,cr),(dense_work,nd,0)):
     target.update(m.mean_work(nn,width,cc));target.update(m.variance_work(nn,width,cc));target.update(m.norm_work(nn,width))
     target.update(BN_mean_scale=width,BN_variance_scale=width,BN_epsilon_add=width,BN_rsqrt=width)
    rows.append({'name':name+'.'+bn,'dense_elements':ed,'compact_elements':ec,'dense_tensor_bytes':ed*4,'compact_tensor_bytes':ec*4})
  require=lambda ok,msg:m.require(ok,msg)
  require(dense_elements==322560000,'Selected16 full domain is fixed')
  arms[arm]={'records':rows,'selected16_dense_BN_elements':dense_elements,'V1_compact_weighted_BN_elements':compact_elements,
   'additional_full_domain_BN_elements':dense_elements-compact_elements,'V1_weighted_BN_issue_slots':m.issue_slots(compact_work),
   'V2_dense_BN_issue_slots':m.issue_slots(dense_work),'additional_BN_scalar_issue_slots':m.issue_slots(dense_work)-m.issue_slots(compact_work),
   'same_materialized_policy_V1_bytes':20*compact_elements,'same_materialized_policy_V2_bytes':20*dense_elements,
   'materialized_BN_path_additional_bytes':20*(dense_elements-compact_elements),
   'V2_BN1_gather_payload_bytes':4*bn1_compact_elements,
   'V2_separately_stored_BN1_gather_read_write_producer_read_bytes':12*bn1_compact_elements,
   'FC_PSN_repetition_factor':1,'normalization_boundaries_selected8':16}
 # Single-layer screening cannot supply the multi-layer whole-net gate.
 layer_fc=[(name,t*b*h*w*8*c*c) for name,(t,b,h,w,c),_ in g]
 m.require(len({n for _,n in layer_fc})==1,'All 12 layers have equal nominal FC MAC')
 target=next((name,s) for name,s,_ in g if '.layers.2.swin_blocks.2.' in name)
 name,(t,b,h,w,c)=target;r=len(m.points(h,w,m.masks(h,w,'CC')))
 native=t*b*h*w;compact=t*b*r;cancel=(native-compact)*8*c*c;total=sum(n for _,n in layer_fc)
 # Complete T PSN1/PSN2: T MAC plus bias and threshold for every scalar.
 psn_saved=(native-compact)*5*c*(t+2)
 pool_added=t*b*c*(h*w-r+r)
 dense_known=report['comparisons']['native/spill']['known_dense_producer_plus_external54_issue_slots']
 pilot={'target':name,'native_target_FC_rows_each':native,'compact_target_FC_rows_each':compact,
  'whole12_FC_rows_each_after_single_layer_CC':558000-native+compact,
  'per_MLP_dense_FC_MAC':layer_fc[0][1],'all12_dense_FC_MAC':total,'cancelled_FC_MAC':cancel,
  'maximum_cancelled_fraction_of_twelve_MLP_FC_MAC':cancel/total,
  'conditional_single_pass_PSN_scalar_issues_cancelled':psn_saved,'pool_scalar_issues_added':pool_added,
  'V2_conditional_total_known_producer_scalar_issues_cancelled':cancel+psn_saved-pool_added,
  'optimistic_fraction_of_known_native_work_under_shared_PE_assumption':(cancel+psn_saved-pool_added)/dense_known,
  'scope':'No FC/PSN recomputation; all current BN remains dense. MAC/issue necessary condition only. It is not a speedup or full-net cycle bound.'}
 m.require((native,compact,pilot['whole12_FC_rows_each_after_single_layer_CC'])==(12000,1400,547400),'Single target actual geometry')
 out={'schema':'flow_current_bn_round16_V2_geometry_delta_v1','status':'CONDITIONAL_INVENTORY_ONLY_NO_SCHEDULE_OR_NUMERICAL_ADMISSION',
  'source_sha256':{str(Path(__file__)):m.sha(__file__),str(Path(m.__file__)):m.sha(m.__file__),str(report_path):m.sha(report_path)},
  'arms':arms,'single_layer_pilot':pilot,
  'materialized_policy':'For BOTH hypothetical BN paths: save Z once (4E bytes), then three unfused Z reads and one normalized output write (16E). V2 expansion is the dense Z store, not a second store. This DIFFERENT common inventory must not be substituted into the V1 producer-fused schedule.',
  'gather_policy':'BN1 sample one normalized value per actual representative. The read payload is stated separately; a spill implementation additionally writes and rereads this compact payload. BN2 remains dense and needs no final gather/repeat. Final residual/output and ordinary common62 are not included again in this delta.',
  'important_limits':['No V2 SRAM/port schedule is supplied. Full BN values cannot be assumed to fit in the declared 2MiB activation SRAM.',
   'Native PyTorch BN numerical admission does not validate an RTL BN reduction order. Physical arithmetic/frequency/area remain unknown.',
   'Expand/gather address generation, mask/scorer, full-grid row-order conversion, residual scheduling and common attention/conv must still be charged.',
   'Producer-fused mean or fused final residual can reduce traffic, but only with a newly specified dependency/capacity schedule.',
   'Ordinary and flow-information methods must share this executor and mask budget. Normalization restoration is infrastructure, not the flow-specific contribution.']}
 with (HERE/'V2_DELTA_AND_SINGLE_LAYER_PILOT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'CC_BN_delta_elements':arms['CC']['additional_full_domain_BN_elements'],'CC_BN_delta_issue_slots':arms['CC']['additional_BN_scalar_issue_slots'],'CC_materialized_bytes_delta':arms['CC']['materialized_BN_path_additional_bytes'],'pilot':pilot},indent=2))

if __name__=='__main__':main()
