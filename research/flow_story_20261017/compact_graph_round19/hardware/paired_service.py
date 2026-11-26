#!/usr/bin/env python3
"""Same-resource dense/compact services for R19's frozen V2 graph.

Finite blocked operation schedule, not numerical FP RTL or whole-net cycles.
Both BN domains remain full, unfused, and materialized. No weighted reductions.
"""
from collections import Counter
from dataclasses import asdict
import argparse,hashlib,importlib.util,json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;DATE=HERE.parents[1]
HELPER=DATE/'current_bn_round16/hardware/service_model.py'
SCORE=DATE/'ordinary_controls_round18/hardware/results/SERVICE_ACCOUNT_V2.json'
ACTUAL=HERE.parent/'independent_execution_audit/ACTUAL_EXECUTION_AUDIT_V1.json'
PRIMITIVE=DATE/'current_bn_round16/implementation/native_dense_bn_v2.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def validate_schedule(sch):
 sch.validate()
 for event in sch.events:
  if event['kind']!='task':continue
  traffic=event.get('local_activation_read_bytes',0)+event.get('local_activation_write_bytes',0)
  require(event['end']-event['start']>=M.ceildiv(traffic,sch.cfg.activation_banks*sch.cfg.bytes_per_word),'Local 1RW activation bank service lower bound')
require(sha(HELPER)=='0860a1408c522994dc5ca8dc72993ab9903f76cdb2c3feb0e55b6e313fe7627f','Frozen finite resources/math helper')
require(sha(SCORE)=='f1e6a8ec2670f0bccb5daea8d9ff518888bb7b4fd1158e5834e64ccbc6aaddce','Frozen R18 same-score cost assumptions')
sp=importlib.util.spec_from_file_location('_r19_full_bn_service',HELPER);M=importlib.util.module_from_spec(sp);sys.modules[sp.name]=M;sp.loader.exec_module(M)
T,C,H,W=10,384,30,40
def cover(arm):
 require(arm in ('FF','CC','single'),'Known resource mask')
 out=[]
 for sy in range(0,H,9):
  for sx in range(0,W,9):
   ey,ex=min(H,sy+9),min(W,sx+9);fine=arm=='FF' or (arm=='single' and sy==sx==0);step=1 if fine else 3
   for y in range(sy,ey,step):
    for x in range(sx,ex,step):out.append({'y':y,'x':x,'area':(min(y+step,ey)-y)*(min(x+step,ex)-x),'fine':fine})
 require(sum(p['area'] for p in out)==H*W,'Complete true-tail cover')
 return out

def layer(graph,arm,cfg=None,events=False):
 require(graph in ('native','dense_expanded','compact'),'Explicit graph')
 if graph=='native':require(arm=='FF','Native unmodified full input')
 cfg=cfg or M.Config();cfg.validate();sch=M.Schedule(cfg)
 require((cfg.pe_rows,cfg.pe_columns,cfg.activation_banks,cfg.weight_banks,cfg.statistic_vector)==(8,32,16,32,16),'This blocking schedule has fixed 8x32 PE, 16 activation banks, 32 weight banks and SIMD16')
 points=cover(arm);r=len(points) if graph=='compact' else H*W
 # A fixed complete-T batch of8 spatial points; three widest FP32 tensors.
 block=8;working=block*T*4*C*4*3
 require(working+cfg.reserved_statistics_and_metadata_bytes<=cfg.activation_sram_bytes,'Complete T input/intermediate/output live storage')
 require(6*4*C*4+len(points)*20<=cfg.reserved_statistics_and_metadata_bytes,'Statistics plus true-cell descriptors fit reserved metadata')
 def dma(name,n,direction,requires=(),weight=False):sch.dma(name,n,direction,working,requires,weight=weight)
 def task(name,ticks,work=None,requires=(),local_r=0,local_w=0):
  w=dict(work or {});w.update(local_activation_read_bytes=local_r,local_activation_write_bytes=local_w)
  require(ticks>=M.ceildiv(local_r+local_w,cfg.activation_banks*4),'Local 1RW activation bank service lower bound')
  sch.task(name,ticks,requires=requires,scratch=working,work=w)
  sch.events[-1].update(local_activation_read_bytes=local_r,local_activation_write_bytes=local_w)
 def pool_mean(n,adds,scales):
  # Conservative disjoint arithmetic and local SRAM slots. Channel-spaced
  # partial sums use the declared register pipeline; two FP drains cover
  # reduction and final scale. No read/write overlap is credited here.
  arithmetic=M.ceildiv(adds,16)+M.ceildiv(scales,16)+2*cfg.fp_latency
  local=M.ceildiv((n+T*C)*4,cfg.activation_banks*4)
  task('pool/mean',arithmetic+local,{'pool_add':adds,'pool_scale':scales,
   'pool_arithmetic_ticks':arithmetic,'pool_local_service_ticks':local},local_r=n*4,local_w=T*C*4)
 def chunks(n):return [(s,min(block,n-s)) for s in range(0,n,block)]
 full=T*H*W*C
 if graph=='dense_expanded':
  dma('dense/clone_read',full*4,'pool_source_read');dma('dense/clone_write',full*4,'dense_input_write')
  coarse=[p for p in points if not p['fine']]
  for p in coarse:
   n=T*C*p['area'];dma('pool/coarse_input',n*4,'pool_source_read')
   adds=T*C*(p['area']-1);scales=T*C
   pool_mean(n,adds,scales)
   task('pool/repeat',2*M.ceildiv(n,16),{'pool_repeat_elements':n},local_r=n*4,local_w=n*4)
   dma('pool/dense_repeat_store',n*4,'dense_input_write')
 elif graph=='compact':
  for p in points:
   n=T*C*p['area'];dma('pool/original_cell',n*4,'pool_source_read')
   adds=T*C*(p['area']-1);scales=T*C # literal pack divide, including fine/area1
   pool_mean(n,adds,scales)
   dma('pool/representative_store',T*C*4,'compact_input_write')
  dma('pool/cell_descriptors',len(points)*20,'descriptor_write')
 sch.barrier('input_ready')
 def producer(which,cin,cout,input_source,dependency):
  for start,n in chunks(r):
   dma(which+'/input',n*T*cin*4,input_source,(dependency,))
   dma(which+'/temporal_parameters',440,'weight_read',weight=True)
   pt=M.matmul_ticks(n*cin,T,T,cfg)
   task(which+'/PSN_matmul',pt,{'PSN_MAC':n*cin*T*T,'PE_reserved_ticks':pt,'PE_FMA_issue_ticks':M.ceildiv(n*cin,8)*T,
      'PSN_PE_activation_operand_bytes_consumed':n*cin*T*T*4},(dependency,),local_r=n*cin*T*4,local_w=n*cin*T*4)
   task(which+'/PSN_bias_threshold',4*M.ceildiv(n*T*cin,16)+4,{'PSN_bias_add':n*T*cin,'PSN_threshold':n*T*cin},local_r=n*T*cin*4,local_w=n*T*cin*4)
   for col in range(0,cout,32):
    width=min(32,cout-col);dma(which+'/weight_stripe',cin*width*4,'weight_read',weight=True)
    tt=M.matmul_ticks(n*T,cin,width,cfg)
    task(which+'/FC',tt,{which+'_rows':n*T if col==0 else 0,which+'_MAC':n*T*cin*width,
       'PE_reserved_ticks':tt,'PE_FMA_issue_ticks':M.ceildiv(n*T,8)*cin},(dependency,),
       local_r=M.ceildiv(n*T,8)*cin*min(8,n*T)*4,local_w=n*T*width*4)
   dma(which+'/produced_store',n*T*cout*4,which+'_representative_store' if graph=='compact' else which+'_full_Z_store')
  sch.barrier(which+'_produced',(dependency,))
 def expand_z(which,width):
  if graph=='compact':
   dma(which+'/descriptors',len(points)*20,'descriptor_read',(which+'_produced',))
   for p in points:
    dma(which+'/representative_read',T*width*4,which+'_representative_read',(which+'_produced',))
    n=T*width*p['area']
    task(which+'/expand',2*M.ceildiv(n,16),{'FC_expand_elements':n},(which+'_produced',),local_r=n*4,local_w=n*4)
    dma(which+'/full_domain_Z',n*4,which+'_full_Z_store')
  sch.barrier(which+'_full_Z_ready',(which+'_produced',))
 def norm(which,width):
  fc='FC1' if which=='BN1' else 'FC2';expand_z(fc,width)
  dma(which+'/affine',width*8,'parameter_read',(fc+'_full_Z_ready',))
  for start,n in chunks(H*W):
   elements=n*T*width;dma(which+'/mean_full_Z',elements*4,which+'_Z_read',(fc+'_full_Z_ready',))
   work=M.mean_work(n*T,width,0)
   if start:work['BN_mean_add']+=width
   task(which+'/mean',M.stats_ticks(n*T,width,'mean',cfg,False),work,(fc+'_full_Z_ready',),local_r=elements*4)
  task(which+'/mean_scale',M.ceildiv(width,16)+4,{'BN_mean_scale':width});sch.barrier(which+'_mean',(fc+'_full_Z_ready',))
  for start,n in chunks(H*W):
   elements=n*T*width;dma(which+'/variance_full_Z',elements*4,which+'_Z_read',(which+'_mean',))
   work=M.variance_work(n*T,width,0)
   if start:work['BN_variance_add']+=width
   task(which+'/centered_variance',M.stats_ticks(n*T,width,'variance',cfg,False),work,(which+'_mean',),local_r=elements*4)
  task(which+'/var_scale_eps_rsqrt',2*M.ceildiv(width,16)+8+width+15,{'BN_variance_scale':width,'BN_epsilon_add':width,'BN_rsqrt':width},(which+'_mean',))
  sch.barrier(which+'_statistics_ready',(which+'_mean',))
  for start,n in chunks(H*W):
   elements=n*T*width;dma(which+'/normalize_full_Z',elements*4,which+'_Z_read',(which+'_statistics_ready',))
   task(which+'/normalize',M.stats_ticks(n*T,width,'norm',cfg),M.norm_work(n*T,width),(which+'_statistics_ready',),local_r=elements*4,local_w=elements*4)
   dma(which+'/full_normalized_store',elements*4,which+'_full_output_store')
  sch.barrier(which+'_full_output_ready',(which+'_statistics_ready',))
 producer('FC1',C,4*C,'compact_input_read' if graph=='compact' else 'dense_input_read','input_ready');norm('BN1',4*C)
 if graph=='compact':
  dma('BN1/gather_descriptors',len(points)*20,'descriptor_read',('BN1_full_output_ready',))
  for start,n in chunks(r):
   e=n*T*4*C;dma('BN1/representative_gather',e*4,'BN1_gather_read',('BN1_full_output_ready',))
   task('BN1/gather_copy',2*M.ceildiv(e,16),{'BN1_gather_elements':e},('BN1_full_output_ready',),local_r=e*4,local_w=e*4)
   dma('BN1/compact_gather_store',e*4,'BN1_gather_store')
 sch.barrier('BN1_next_producer_ready',('BN1_full_output_ready',))
 producer('FC2',4*C,C,'BN1_compact_gather_read' if graph=='compact' else 'BN1_dense_producer_read','BN1_next_producer_ready');norm('BN2',C)
 for _,n in chunks(H*W):
  e=n*T*C;dma('output/BN2_branch',e*4,'BN2_branch_read',('BN2_full_output_ready',));dma('output/residual',e*4,'residual_read')
  task('output/residual_add',3*M.ceildiv(e,16)+4,{'residual_add':e},('BN2_full_output_ready',),local_r=e*8,local_w=e*4)
  dma('output/full_grid',e*4,'output_write')
 sch.barrier('output_ready',('BN2_full_output_ready',));validate_schedule(sch)
 require(sch.work['FC1_rows']==sch.work['FC2_rows']==r*T,'True producer rows')
 require(sch.traffic['FC1_full_Z_store']==T*H*W*4*C*4 and sch.traffic['FC2_full_Z_store']==full*4,'Full-domain FC expansion stores')
 require(sch.traffic['BN1_Z_read']==T*H*W*4*C*4*3 and sch.traffic['BN2_Z_read']==full*4*3,'Three actual full-domain BN reads')
 require(sch.work.get('BN_area_multiply',0)==0,'Never weighted compact BN')
 out={'graph':graph,'arm':arm,'shape_TBHWC':[T,1,H,W,C],'producer_representatives':r,'FC_rows_each':r*T,
   'full_BN_domain_elements':23040000,'resource_config':asdict(cfg),'complete_T_point_batch':block,
   'activation_peak_bytes':sch.peak,'modeled_PE_reserved_ticks':sch.work['PE_reserved_ticks'],
   'modeled_PE_FMA_issue_ticks':sch.work['PE_FMA_issue_ticks'],'modeled_blocking_total_service_ticks':sch.tick,
   'work':dict(sch.work),'external_transactions_bytes':dict(sch.traffic),'external_total_bytes':sum(sch.traffic.values()),
   'local_activation_read_bytes':sch.work['local_activation_read_bytes'],'local_activation_write_bytes':sch.work['local_activation_write_bytes'],
   'barriers':[x['name'] for x in sch.events if x['kind']=='barrier'],
   'cost_scope':'Assumed finite microarchitecture; PE reserved ticks include recurrence/drain. No FP numerical RTL or full-network speedup.'}
 if events:out['events']=sch.events
 return out,sch

def report():
 require(sha(ACTUAL)=='9c5baf417bbab272689ae6ef996725b0aece6818dd4cec8440dd58f1091ade55','Actual new graph execution evidence')
 require(sha(PRIMITIVE)=='e874e976ce4833a0638092ca78612e8224139eb1cc3b49d9ce0098488a3bb731','Unmodified full-domain native BN primitive')
 rows={};scores=read(SCORE)['per_deployed_score']
 for graph,arms in [('native',('FF',)),('dense_expanded',('FF','CC','single')),('compact',('FF','CC','single'))]:
  for arm in arms:rows[graph+'/'+arm]=layer(graph,arm)[0]
 pairs={};deployments={}
 for arm in ('FF','CC','single'):
  d,c=rows['dense_expanded/'+arm],rows['compact/'+arm]
  pairs[arm]={'dense_PE_reserved_ticks':d['modeled_PE_reserved_ticks'],'compact_PE_reserved_ticks':c['modeled_PE_reserved_ticks'],
   'PE_reserved_tick_difference_dense_minus_compact':d['modeled_PE_reserved_ticks']-c['modeled_PE_reserved_ticks'],
   'service_tick_difference_dense_minus_compact':d['modeled_blocking_total_service_ticks']-c['modeled_blocking_total_service_ticks'],
   'external_byte_difference_dense_minus_compact':d['external_total_bytes']-c['external_total_bytes'],
   'local_activation_read_byte_difference_dense_minus_compact':d['local_activation_read_bytes']-c['local_activation_read_bytes'],
   'local_activation_write_byte_difference_dense_minus_compact':d['local_activation_write_bytes']-c['local_activation_write_bytes'],
   'same_full_BN_domain_elements':d['full_BN_domain_elements'],'same_score_cases':{}}
  for name,score in scores.items():
   # The inherited scorer writes twelve binary64 results externally. Here we
   # explicitly reload those 96 bytes into the selector's finite input queue.
   extra=score['score']['hypothetical_score_schedule_ticks']+14+M.ceildiv(96,64)
   common_bytes=score['score']['original_source_read_bytes']+96+96
   if score['extra_native_score_PSN1']:
    psn=score['extra_native_score_PSN1'];extra+=psn['hypothetical_ticks']
    common_bytes+=psn['input_read_bytes']+psn['native_spike_write_bytes']+psn['temporal_parameter_bytes']
   pairs[arm]['same_score_cases'][name]={'common_score_plus_selector_ticks_assumed':extra,
    'common_score_plus_selector_external_bytes':common_bytes,'selector_score_reload_bytes':96,
    'dense_target_plus_same_score_ticks_assumed':d['modeled_blocking_total_service_ticks']+extra,
    'compact_target_plus_same_score_ticks_assumed':c['modeled_blocking_total_service_ticks']+extra,
    'dense_target_plus_same_score_external_bytes':d['external_total_bytes']+common_bytes,
    'compact_target_plus_same_score_external_bytes':c['external_total_bytes']+common_bytes,
    'quality_transfer_between_graphs':False,'whole_network_speedup':None}
   if arm=='single':
    native=rows['native/FF'];desc=score['score'];conversion_elements=desc['conversion_ticks']//3
    split={'descriptor_auxiliary_ALU_ticks':desc['serialized_ALU_ticks']+conversion_elements,
     'descriptor_local_SRAM_ticks':desc['local_SRAM_access_ticks']+desc['pool_repeat_read_write_ticks']+2*conversion_elements,
     'descriptor_external_DMA_ticks':desc['external_DMA_ticks']+M.ceildiv(96,64),
     'selector_transaction_ticks':14,'extra_native_PSN_PE_reserved_ticks':0,
     'extra_native_PSN_PE_FMA_issue_ticks':0,'extra_native_PSN_auxiliary_bias_threshold_ticks':0,'extra_native_PSN_external_DMA_ticks':0}
    if score['extra_native_score_PSN1']:
     split['extra_native_PSN_PE_reserved_ticks']=M.matmul_ticks(H*W*C,T,T,M.Config())
     split['extra_native_PSN_PE_FMA_issue_ticks']=M.ceildiv(H*W*C,8)*T
     split['extra_native_PSN_auxiliary_bias_threshold_ticks']=4*M.ceildiv(full:=T*H*W*C,16)+4
     split['extra_native_PSN_external_DMA_ticks']=M.ceildiv(full*8+440,64)
    # FMA issue ticks are a subset of PE reserved ticks, not another duration.
    require(sum(v for k,v in split.items() if k!='extra_native_PSN_PE_FMA_issue_ticks')==extra,'Complete score/selector resource duration decomposition')
    candidate=c['modeled_blocking_total_service_ticks']+extra;baseline=native['modeled_blocking_total_service_ticks']
    deployments[name]={'comparison':'compact k1 + its score + selector versus native FF without any score',
     'native_no_score_target_service_ticks':baseline,'compact_target_without_score_ticks':c['modeled_blocking_total_service_ticks'],
     'score_plus_selector_ticks':extra,'score_resource_split_assumed':split,
     'candidate_target_with_score_service_ticks':candidate,
     'net_local_service_ticks_saved_vs_native':baseline-candidate,'net_local_service_fraction_saved_vs_native':(baseline-candidate)/baseline,
     'native_PE_reserved_ticks':native['modeled_PE_reserved_ticks'],
     'candidate_PE_reserved_ticks_including_score_PSN':c['modeled_PE_reserved_ticks']+split['extra_native_PSN_PE_reserved_ticks'],
     'native_external_bytes':native['external_total_bytes'],'candidate_external_bytes_including_score':c['external_total_bytes']+common_bytes,
     'net_external_bytes_saved_vs_native':native['external_total_bytes']-c['external_total_bytes']-common_bytes,
     'auxiliary_score_lane_area_timing_and_numerics':'NOT_IMPLEMENTED_OR_SYNTHESIZED',
     'whole_network_speedup':None,'precision_or_quality_admission':None}
 return {'schema':'flow_r19_full_domain_bn_paired_service_v1','status':'CONDITIONAL_RESOURCE_MODEL_ONLY','source_sha256':{str(p):sha(p) for p in (HELPER,SCORE,ACTUAL,PRIMITIVE,Path(__file__))},
  'graphs':rows,'paired_same_mask':pairs,'deployment_vs_native_no_score':deployments,
  'assumptions':['Same finite2MiB activation/512KiB weight resources; blocks8 completeT points, one shared serialized64B/tick external bus.',
   'FP32PE8x32 recurrence latency4 and output draining; no FP MAC=one-cycle shortcut. Statistics SIMD16 separate pipeline with full channel vectors.',
   'Both graphs pay full Z store, three full-domain BN reads, normalized output store, and final branch/residual reads. No producer-fused mean.',
   'Compact additionally stores/reads representative FC Z then expands, and BN1 gathers to compact storage before second producer.',
   'Pool mean arithmetic and local read/write slots are serialized, including two FP pipeline drains. Every compute task checks aggregate 16x1RW local-byte capacity; detailed physical bank addresses remain unimplemented.',
   'Repeat and FC expansion conservatively reread a local representative for every destination scalar; no local broadcast saving is credited.',
   'External byte inventory and local activation operation accesses are distinct. Weight-bank reads/RF accesses and physical addresses need separate RTL.',
   'Detailed FP rounding/reduction tree not proven equivalent to GPU; this is a service feasibility model, not a validated numeric accelerator.',
   'Scores are same R18 finite-lane assumption; not synthesized. Native PSN source service is extra for PSN-based scores in both coarse graphs.',
   'Same-score comparator explicitly reloads twelve external binary64 scores (96 bytes) before the selector; the common selector then consumes its measured unstalled 14-tick transaction.',
   'The fixed eight-point producer order reloads weight stripes per batch; this is one shared conservative blocking choice, not a claim that the dense baseline or compact schedule is optimal.',
   'Deployment comparison uses native with no predictor as baseline and compact k1 plus its complete ordinary scorer and selector as candidate. Same-score pairs only isolate execution graph costs.',
   'FP64/integer descriptor arithmetic is on an additional hypothetical auxiliary lane, explicitly timed, not zero-cost PE work; native score PSN uses the shared FP32 PE and is included in candidate PE reservations. Lane area/timing/precision remain unimplemented.',
   'Diagnostic reconstructed moments/cell exact scans, captures/serialization excluded from production but must remain separately documented.',
   'Dense-expanded and compact have different numerical outputs; no R17/R18 quality transferred. R19 quality only applies after own actual execution audit.',
   'No remaining11MLP/attention/conv/decoder whole-network schedule; no whole-network cycle/energy claim.'],
  'diagnostic_cost_per_compact_call':{'moment_reductions':4,'moment_read_elements':46080000,'cell_exact_comparison_read_elements':46080000,'BN2_diagnostic_gather_elements':'10*representatives*384','capture_hash_serialization_bytes':'NOT_MEASURED'}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New exclusive result')
 out=report();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'sha256':sha(a.output),'pairs':{k:{x:y for x,y in v.items() if x!='same_score_cases'} for k,v in out['paired_same_mask'].items()}}))
