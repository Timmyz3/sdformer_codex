#!/usr/bin/env python3
"""R18 exact logical work + explicitly assumed finite-resource schedules.

Only k1 RTL ticks are implementation-tested. FP32/64 schedules are service
assumptions, not numerical datapaths, CPU instruction counts or whole-net cycles.
No quality files, model forward, GPU or source mutation.
"""
from collections import Counter
from dataclasses import asdict
import hashlib,importlib.util,json,math
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'current_bn_round16/hardware/service_model.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,m):
 if not x:raise ValueError(m)
def load(p,n):
 import sys
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m

NAMES=('amplitude_abs_mean','feature_rms','pooling_residual_mse','spatial_neighbor_abs_difference',
 'temporal_neighbor_abs_difference','psn1_activity','psn1_spatial_neighbor_hamming')

def independent_tile_work(t=10,c=384):
 n=t*81*c;q=t*9*c;v=t*72*c;e=(t-1)*81*c
 return dict(zip(NAMES,[
 {'u_reads':n,'sn1_reads':0,'absolute':n,'add':n-1,'divide':1},
 {'u_reads':n,'sn1_reads':0,'multiply':n,'add':n-1,'divide':1,'sqrt':1},
 {'u_reads':2*n,'sn1_reads':0,'representative_reads':n,'add':8*q+n-1,'divide':q+1,'subtract':n,'multiply':n,'representative_elements':q,'expanded_representative_write_elements':n},
 {'u_reads':4*v,'sn1_reads':0,'subtract':2*v,'absolute':2*v,'add':2*v-1,'divide':3},
 {'u_reads':2*e,'sn1_reads':0,'subtract':e,'absolute':e,'add':e-1,'divide':1},
 {'u_reads':0,'sn1_reads':n,'integer_add':n-1,'divide':1},
 {'u_reads':0,'sn1_reads':4*v,'not_equal':2*v,'boolean_count_elements':2*v,'integer_add':2*(v-1),'add':1,'divide':3}]))

def score_schedule(name,ops,cfg):
 # Six C64 stripes per tile. Full T10 and all9x9 positions of each stripe fit.
 # Four FP64 arrays and one original FP32 stripe are a conservative local
 # lifetime reservation; processing instructions below execute serially.
 stripe_n=10*81*64;live=stripe_n*(4*8+4)+cfg.reserved_statistics_and_metadata_bytes
 require(live<=cfg.activation_sram_bytes,'Finite score stripe storage')
 scalar_latency={'absolute':1,'add':4,'subtract':4,'multiply':4,'divide':16,'sqrt':32,'integer_add':1,'not_equal':1}
 # Literal scalar load/compute/store implementation: no FP operation overlap;
 # every operand and result gets its own local 64-bit SRAM access tick.
 # A 64-bit word consumes two banks. Serial service <=16 one-RW32-bit banks.
 local_access={k:2 if k in ('absolute','sqrt','multiply') else 3 for k in scalar_latency}
 frame={k:12*v for k,v in ops.items()}
 instructions=sum(frame.get(k,0) for k in scalar_latency)
 alu_ticks=sum(frame.get(k,0)*v for k,v in scalar_latency.items())
 sram_ticks=sum(frame.get(k,0)*local_access[k] for k in scalar_latency)
 materialization_ticks=2*frame.get('expanded_representative_write_elements',0)
 converted=12*10*81*384
 conversion_ticks=3*converted # original read32, convert, write64; serialized.
 external_source_bytes=4*converted
 source_dma_ticks=math.ceil(external_source_bytes/cfg.dram_bytes_per_tick)
 score_write_bytes=12*8
 # Reduction accumulators and stripe merging are covered by exact n-1 adds;
 # any chosen tree must separately reproduce the frozen FP64 score semantics.
 return {'logical_operations_per_frame':frame,'original_source_read_bytes':external_source_bytes,
  'source_kind':'native_full_grid_PSN1' if name.startswith('psn1') else 'pre_MLP_u',
  'finite_stripes':72,'score_local_peak_bytes':live,'FP64_or_integer_instructions':instructions,
  'local_SRAM_access_ticks':sram_ticks,'pool_repeat_read_write_ticks':materialization_ticks,'serialized_ALU_ticks':alu_ticks,
  'conversion_ticks':conversion_ticks,'external_DMA_ticks':source_dma_ticks+math.ceil(score_write_bytes/64),
  'hypothetical_score_schedule_ticks':sram_ticks+materialization_ticks+alu_ticks+conversion_ticks+source_dma_ticks+math.ceil(score_write_bytes/64),
  'latency_assumptions':scalar_latency,'two_bank_64bit_local_access':True,
  'numeric_reduction_order_implemented_or_validated':False,
  'scope':'Single declared serial FP64/integer lane, no measured score datapath or speed claim.'}

def cover(mask):
 out=[]
 for ty in range(4):
  for tx in range(5):
   h=min(9,30-ty*9);w=min(9,40-tx*9);fine=(ty,tx)==mask
   require(h>0 and w>0,'Real tail')
   for y in range(0,h,1 if fine else 3):
    for x in range(0,w,1 if fine else 3):
     area=min(1 if fine else 3,h-y)*min(1 if fine else 3,w-x)
     out.append((area,fine))
 require(sum(a for a,_ in out)==1200,'Full real spatial coverage')
 return out

def report():
 require(sha(OLD)=='0860a1408c522994dc5ca8dc72993ab9903f76cdb2c3feb0e55b6e313fe7627f','Frozen finite-resource geometry helper')
 m=load(OLD,'r18_dense_service_helper');cfg=m.Config()
 dense,sch=m.layer('s2b2_actual_dense_target',(10,1,30,40,384),m.masks(30,40,'FF'),'spill',cfg)
 require(dense['work']['FC1_rows']==dense['work']['FC2_rows']==12000,'R17 dense target rows')
 # This ONLY calls the old all-fine/full-domain branch; no weighted compact
 # reductions, old V1 compact ticks or old whole-net ratio are imported.
 require(dense['work'].get('BN_area_multiply',0)==0,'No weighted BN reused')
 pc=cover((0,0));coarse_area=sum(a for a,f in pc if not f);coarse_reps=sum(not f for _,f in pc)
 require(len(pc)==212 and coarse_area==1119 and coarse_reps==131,'Independent singleton tail geometry')
 for y in range(3):
  for x in range(4):require(cover((y,x))==pc or (len(cover((y,x)))==212 and sum(a for a,f in cover((y,x)) if not f)==1119),'All complete singleton same service geometry')
 full=10*1200*384
 pool={'clone_read_bytes':4*full,'clone_write_bytes':4*full,'coarse_source_read_bytes':4*10*coarse_area*384,
       'coarse_expansion_write_bytes':4*10*coarse_area*384,'pool_add':10*384*(coarse_area-coarse_reps),
       'pool_scale':10*384*coarse_reps,'peak_one_cell_bytes':(9+1)*10*384*4}
 pool['hypothetical_ticks']=math.ceil(sum(pool[k] for k in ('clone_read_bytes','clone_write_bytes','coarse_source_read_bytes','coarse_expansion_write_bytes'))/64)+math.ceil((pool['pool_add']+pool['pool_scale'])/16)+4
 require(pool['peak_one_cell_bytes']+cfg.reserved_statistics_and_metadata_bytes<=cfg.activation_sram_bytes,'Cell fits')
 # Native full-grid PSN1 is an actual required predictor producer for two
 # spike scores. R17 archives save it but a new coarse action cannot assume
 # those native spikes also equal PSN1(pool(u)). No free cached native pass.
 psn={'MAC':1200*384*100,'bias_add':full,'threshold':full,'input_read_bytes':full*4,
      'native_spike_write_bytes':full*4,'temporal_parameter_bytes':440,
      'native_PSN_output_reused_as_coarse_PSN_output':False}
 psn['hypothetical_ticks']=m.matmul_ticks(1200*384,10,10,cfg)+4*m.ceildiv(full,16)+4+math.ceil((8*full+440)/64)
 models={}
 for name,ops in independent_tile_work().items():
  score=score_schedule(name,ops,cfg);extra=psn['hypothetical_ticks'] if name.startswith('psn1') else 0
  models[name]={'score':score,'k1_RTL_unstalled_transaction_ticks':14,'extra_native_score_PSN1':psn if extra else None,
   'target_dense_MLP_plus_pool_score_selection_ticks_assumed':dense['modeled_blocking_schedule_ticks']+pool['hypothetical_ticks']+score['hypothetical_score_schedule_ticks']+14+extra,
   'target_dense_FC_rows_each':12000,'whole12_dense_FC_rows_each':558000,
   'actual_dense_cancelled_FC_rows':0,'actual_dense_speedup_claim':None}
 # V2 is a different unadmitted-as-expanded-mapping graph. Its producer work
 # can be inventoried, but none of R17/R18 quality transfers to this graph.
 compact={'status':'SEPARATE_CONDITIONAL_GRAPH_INVENTORY_ONLY','representatives':212,'target_FC_rows_each':2120,
  'whole12_FC_rows_each':548120,'target_FC_MAC':2*2120*384*1536,'target_PSN_MAC':212*384*5*100,
  'native_dense_BN_elements':10*1200*(1536+384),'expand_FC_z_written_elements':10*1200*(1536+384),
  'BN1_gather_read_elements':2120*1536,'BN1_gather_write_elements':2120*1536,
  'score_PSN1_cost_remains_additional_when_using_native_spike_scores':True,
  'measured_cycles':None,'R17_R18_quality_applicable':False,
  'note':'R16 equivalence failure remains. Independent approximate algorithm may be tested against GT under a new contract, never as recovered equivalence. Dense BN and expand/gather/retention remain; no V1 weighted savings.'}
 bn=m.external_bn_inventory()
 other_mlp_BN_elements=sum(t*b*h*w*5*c for _,(t,b,h,w,c),_ in m.geometry())-23040000
 require(other_mlp_BN_elements+23040000==437760000,'All24 MLP BN elements')
 return {'schema':'flow_r18_hardware_work_and_assumed_schedule_v1','status':'CONDITIONAL_SCHEDULE_NOT_NETWORK_ACCELERATION',
  'source_sha256':{str(OLD):sha(OLD),str(Path(__file__)):sha(__file__)},'resource_config':asdict(cfg),
  'scope':'Actual-dense operator graph with assumed finite blocking services; neither NumPy execution timing nor native-BN bitwise RTL; only selector has tested RTL ticks.',
  'dense_target':dense,'dense_k1_pool_clone_and_expand':pool,'per_deployed_score':models,'hypothetical_compact_separate':compact,
  'unmodeled_common_work':{'other11_MLP_BN_elements':other_mlp_BN_elements,'external54_BN_elements':sum(x['elements'] for x in bn),
    'remaining_MLP_FC_rows_each':546000,'attention_conv_decoder_full_schedule':'UNKNOWN','common_producer_cost_cancelled':False},
  'limitations':['No compatible complete-network cycle denominator; no latency/energy/speedup claim.',
   'FP64 score arithmetic and FP32 PE/BN are timing assumptions, not synthesized/bitwise-matched datapaths.',
   'Per-policy cost includes one descriptor, not all seven simultaneously; archive/hash/decompression are offline audit work.',
   'Local 64-bit SRAM accesses serialize two banks; full bank-address placement for FP/MLP remains unimplemented.',
   'Source-dependent floating reduction trees can change rankings; selector RTL guarantees only exact supplied score-word ordering.',
   'u original full18.432MB remains in external storage; no full-frame residency in 2MiB SRAM is assumed.',
   'One score native PSN and coarse executed PSN are distinct services; computing all seven offline with cached sn1 proves no free production pass.']}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=report();dest=a.output;require(not dest.exists(),'Never overwrite report');dest.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'status':out['status'],'output_sha256':sha(dest),'dense_target_ticks':out['dense_target']['modeled_blocking_schedule_ticks']}))
