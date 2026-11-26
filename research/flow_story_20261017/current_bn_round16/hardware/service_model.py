#!/usr/bin/env python3
"""Finite-activation-storage current-BN service model, not RTL or whole-net latency.

Operation/FC rows are exact for the stated graph. Ticks are a declared blocked
producer + SIMD16/statistics + serialized DMA machine, not GPU timing or DC.
"""
from collections import Counter
from dataclasses import dataclass,asdict
import hashlib,json,math
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
GFILE=REPO/'research/flow_story_20261016/coverage_recovery_round13/independent_execution_audit/audit_execution.py'
BNFILE=REPO/'research/flow_story_20261014/damage_allocation_probe/hardware_contract_complete_bn_audit.json'
def require(x,message):
 if not x:raise ValueError(message)
def ceildiv(a,b):return (a+b-1)//b
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@dataclass(frozen=True)
class Config:
 activation_sram_bytes:int=2*1024*1024
 reserved_statistics_and_metadata_bytes:int=128*1024
 weight_sram_bytes:int=512*1024
 activation_banks:int=16
 weight_banks:int=32
 bytes_per_word:int=4
 dram_bytes_per_tick:int=64
 pe_rows:int=8
 pe_columns:int=32
 fma_latency:int=4
 fp_latency:int=4
 statistic_vector:int=16
 rsqrt_latency:int=16
 accumulator_register_bytes:int=4096
 def validate(self):
  require(self.activation_sram_bytes>self.reserved_statistics_and_metadata_bytes>0,'Finite usable activation SRAM')
  require(self.weight_sram_bytes>0 and self.activation_banks>=self.pe_rows and self.weight_banks>=self.pe_columns,'Declared operand bank service')
  require(self.bytes_per_word==4 and 0<self.dram_bytes_per_tick<=4*self.activation_banks,'FP32/DMA port')
  require(self.statistic_vector<=self.activation_banks and self.statistic_vector<=self.pe_rows*self.pe_columns,'Statistics bank/lane service')
  require(self.accumulator_register_bytes>=4*self.pe_rows*self.pe_columns,'Dot-product register capacity')


def geometry():
 require(sha(GFILE)=='dd57f9f315e1c5f38961208496a37f46db577c47b8507029a55b5bc216d8c8e8','Pinned independently audited real geometry')
 # Reconstruct explicitly, not from the implementation runner ledger.
 return [(f'sttmultires_unet.encoders.swin3d.layers.{s}.swin_blocks.{i}.mlp',(10,1,h,w,c),s in (0,2))
         for s,depth,h,w,c in ((0,2,120,160,96),(1,2,60,80,192),(2,6,30,40,384),(3,2,15,20,768)) for i in range(depth)]


def masks(h,w,kind):
 require(kind in ('FF','CC','checker'),'Known nonlearned mask')
 return [[kind=='FF' or (kind=='checker' and (gy+gx)%2==0) for gx in range(ceildiv(w,9))] for gy in range(ceildiv(h,9))]


def points(h,w,mask):
 require(len(mask)==ceildiv(h,9) and all(len(r)==ceildiv(w,9) and all(type(v) is bool for v in r) for r in mask),'True tile Boolean mask')
 out=[]
 for gy,row in enumerate(mask):
  for gx,fine in enumerate(row):
   y0=gy*9;x0=gx*9;y1=min(h,y0+9);x1=min(w,x0+9);stride=1 if fine else 3
   for y in range(y0,y1,stride):
    for x in range(x0,x1,stride):
     yy=min(y1,y+stride);xx=min(x1,x+stride)
     out.append({'y':y,'x':x,'h':yy-y,'w':xx-x,'area':(yy-y)*(xx-x),'fine':fine})
 require(sum(p['area'] for p in out)==h*w,'No missing/extra represented pixels')
 return out


def matmul_ticks(m,k,n,cfg):
 # Row tile8/output tile32, scalar K recurrence at FP32-FMA latency4.
 # A Cin x min(32,N) weight stripe stays in finite weight SRAM across row tiles.
 require(k*min(n,cfg.pe_columns)*4<=cfg.weight_sram_bytes,'Full output stripe must fit finite weight bank')
 cycles=0
 for row in range(0,m,cfg.pe_rows):
  rr=min(cfg.pe_rows,m-row)
  for col in range(0,n,cfg.pe_columns):
   nn=min(cfg.pe_columns,n-col)
   interval=max(cfg.fma_latency,ceildiv(rr,cfg.activation_banks),ceildiv(nn,cfg.weight_banks))
   cycles+=k*interval+ceildiv(rr*nn,cfg.activation_banks)
 return cycles


def stats_ticks(rows,width,kind,cfg,weighted=False):
 # SIMD16 streaming channel pipeline. Accumulators are channel-banked in
 # reserved statistics RF (16 read + 16 write), not the activation 1RW banks.
 # Channel recurrence cannot issue faster than the final add's latency.
 vv=ceildiv(width,cfg.statistic_vector)
 if kind in ('mean','variance'):
  stages=(2 if weighted else 1) if kind=='mean' else (4 if weighted else 3)
  return rows*max(vv,cfg.fp_latency)+(stages*cfg.fp_latency-1)
 # Activation banks are 1RW: every normalized scalar consumes one read and
 # one write. They cannot both be credited to the same bank in one tick.
 if kind=='norm':return 2*rows*vv+4*cfg.fp_latency-1
 raise ValueError('Unknown stats service')


class Schedule:
 def __init__(self,cfg):
  cfg.validate();self.cfg=cfg;self.tick=0;self.peak=cfg.reserved_statistics_and_metadata_bytes;self.events=[];self.ready=set();self.work=Counter();self.traffic=Counter()
 def barrier(self,name,requires=()):
  require(set(requires)<=self.ready,'Unsatisfied statistics/producer dependency')
  self.ready.add(name);self.events.append({'kind':'barrier','name':name,'tick':self.tick,'requires':list(requires)})
 def task(self,name,duration,*,requires=(),scratch=0,resource='compute',bytes_=0,direction=None,work=None):
  require(set(requires)<=self.ready,'Read before required complete statistics')
  require(duration>=0 and scratch+self.cfg.reserved_statistics_and_metadata_bytes<=self.cfg.activation_sram_bytes,'Activation capacity')
  self.peak=max(self.peak,scratch+self.cfg.reserved_statistics_and_metadata_bytes)
  if bytes_:
   require(resource in ('activation_dma','weight_dma') and bytes_%4==0,'Real FP32 transaction bytes')
   require(duration==ceildiv(bytes_,self.cfg.dram_bytes_per_tick),'Serialized physical DMA-port service')
   if resource=='weight_dma':require(bytes_<=self.cfg.weight_sram_bytes,'A single live weight stripe must fit')
   self.traffic[direction]+=bytes_
  self.events.append({'kind':'task','name':name,'start':self.tick,'end':self.tick+duration,'resource':resource,'scratch_bytes':scratch,'requires':list(requires),'bytes':bytes_,'direction':direction})
  self.tick+=duration
  if work:self.work.update(work)
 def dma(self,name,bytes_,direction,scratch,requires=(),weight=False):
  self.task(name,ceildiv(bytes_,self.cfg.dram_bytes_per_tick),requires=requires,scratch=scratch,resource='weight_dma' if weight else 'activation_dma',bytes_=bytes_,direction=direction)
 def validate(self,expand_bus=False):
  ready=set();previous=0;beats=[]
  for e in self.events:
   if e['kind']=='barrier':
    require(set(e['requires'])<=ready,'Barrier dependence');ready.add(e['name']);continue
   require(e['start']>=previous and e['end']>=e['start'],'Serialized resource overlap')
   require(set(e['requires'])<=ready,'Dependent producer issued early')
   require(e['scratch_bytes']+self.cfg.reserved_statistics_and_metadata_bytes<=self.cfg.activation_sram_bytes,'Live SRAM capacity')
   previous=e['end']
   if e['bytes']:
    require(e['end']-e['start']==ceildiv(e['bytes'],self.cfg.dram_bytes_per_tick),'DMA beat conservation')
    if e['resource']=='weight_dma':require(e['bytes']<=self.cfg.weight_sram_bytes,'Weight stripe capacity')
    if expand_bus:
     left=e['bytes']
     for tick in range(e['start'],e['end']):
      size=min(left,self.cfg.dram_bytes_per_tick);require(0<size<=self.cfg.activation_banks*4,'Activation bank port excess');left-=size
      beats.append({'tick':tick,'bytes':size,'resource':e['resource'],'name':e['name']})
     require(left==0,'Transaction conservation')
  if expand_bus:require(len({b['tick'] for b in beats})==len(beats),'Shared DRAM port conflict')
  require(previous==self.tick,'Complete schedule terminal time')
  return beats


def mean_work(rows,width,coarse_rows):
 return {'BN_mean_add':max(0,rows-1)*width,'BN_area_multiply':coarse_rows*width}
def variance_work(rows,width,coarse_rows):
 return {'BN_variance_sub':rows*width,'BN_variance_square':rows*width,'BN_variance_add':max(0,rows-1)*width,'BN_area_multiply':coarse_rows*width}
def norm_work(rows,width):return {'BN_normalize_sub':rows*width,'BN_normalize_multiply':2*rows*width,'BN_normalize_add':rows*width}


def layer(name,shape,mask,policy,cfg,keep_events=False):
 require(policy in ('spill','recompute'),'Explicit lifetime policy')
 t,b,h,w,c=shape;require(t==10 and b==1,'Complete native T10/B1')
 cfg.validate();p=points(h,w,mask);r=len(p);coarse=sum(not x['fine'] for x in p);coarse_area=sum(x['area'] for x in p if not x['fine']);fine=coarse==0
 # 4C normalized input + 4C PSN2 output + C FC2 output, all complete T;
 # metadata12 bytes per representative. Producer uses no temporal truncation.
 per_point=t*b*9*c*4+12
 block=(cfg.activation_sram_bytes-cfg.reserved_statistics_and_metadata_bytes)//per_point
 require(block>=1,'No complete-T producer point fits finite SRAM')
 chunks=[(start,min(block,r-start)) for start in range(0,r,block)]
 require(6*(4*c+c)*4<=cfg.reserved_statistics_and_metadata_bytes,'Both BN moment/affine/coefficient vectors fit reserved storage')
 sch=Schedule(cfg);full=t*b*h*w*c;u=t*b*r*c
 def scratch(n):return n*per_point
 sch.dma('BN/affine_parameters',(4*c+c)*2*4,'parameter_read',0)
 if not fine:
  for start,n in chunks:
   q=p[start:start+n];covered=sum(x['area'] for x in q);source=t*b*covered*c;reps=t*b*n*c
   sch.dma('pool/full_source',source*4,'input_read',scratch(n))
   operations=t*b*c*sum(x['area']-1 for x in q if not x['fine'])
   scales=t*b*c*sum(not x['fine'] for x in q)
   sch.task('pool/real_tail_area',ceildiv(operations+scales,cfg.statistic_vector)+cfg.fp_latency,scratch=scratch(n),work={'pool_add':operations,'pool_scale':scales})
   sch.dma('pool/compact_U',reps*4,'compact_U_write',scratch(n));sch.dma('pool/area_position',n*12,'descriptor_write',scratch(n))
 sch.barrier('U_ready')
 # These reads include whole-T point groups; both policies retain compact U in
 # external memory. The all-fine path aliases original U, with no pool/copy.
 def read_u(start,n):
  sch.dma('producer/U',t*b*n*c*4,'input_read' if fine else 'compact_U_read',scratch(n),('U_ready',))
  if not fine:sch.dma('producer/descriptor',n*12,'descriptor_read',scratch(n),('U_ready',))
 def psn(n,width,which):
  mm=n*b*width
  require((t*t+t)*4<=cfg.weight_sram_bytes,'PSN complete temporal coefficients fit weight SRAM')
  sch.dma(which+'/temporal_weight_bias',(t*t+t)*4,'weight_read',scratch(n),weight=True)
  sch.task(which,matmul_ticks(mm,t,t,cfg)+4*ceildiv(mm*t,cfg.statistic_vector)+cfg.fp_latency,scratch=scratch(n),work={which+'_MAC':mm*t*t,which+'_bias_add':mm*t,which+'_threshold':mm*t})
 def fc(n,cin,cout,which):
  rows=t*b*n
  # Each complete K x output-stripe is loaded, then consumed before the next;
  # never pretend that the full layer's weight matrix fits in finite SRAM.
  for column in range(0,cout,cfg.pe_columns):
   width=min(cfg.pe_columns,cout-column);weight_bytes=cin*width*4
   require(weight_bytes<=cfg.weight_sram_bytes,'Live weight stripe capacity')
   sch.dma(which+'/weight_stripe',weight_bytes,'weight_read',scratch(n),weight=True)
   sch.task(which+'/stripe',matmul_ticks(rows,cin,width,cfg),scratch=scratch(n),work={which+'_rows':rows if column==0 else 0,which+'_MAC':rows*cin*width})
 def bn_input(start,n,width,stage):
  sch.dma(stage+'/stored_input',t*b*n*width*4,stage+'_read',scratch(n),(stage+'_mean',))
  if not fine:sch.dma(stage+'/descriptor',n*12,'descriptor_read',scratch(n),('U_ready',))
 def moment(start,n,width,kind,tag):
  weighted=any(not x['fine'] for x in p[start:start+n]);cr=t*b*sum(not x['fine'] for x in p[start:start+n])
  work=mean_work(t*b*n,width,cr) if kind=='mean' else variance_work(t*b*n,width,cr)
  # Cross-block sums need one addition/channel per block after the first.
  if start:work[('BN_mean_add' if kind=='mean' else 'BN_variance_add')]+=width
  sch.task(tag+'/'+kind,stats_ticks(t*b*n,width,kind,cfg,weighted),requires=(tag+'_mean',) if kind=='variance' else (),scratch=scratch(n),work=work)
 def normalize(n,width,tag,dependency):
  sch.task(tag+'/normalize',stats_ticks(t*b*n,width,'norm',cfg),requires=(dependency,),scratch=scratch(n),work=norm_work(t*b*n,width))
 def finish_mean(width,tag):
  sch.task(tag+'/mean_scale',ceildiv(width,cfg.statistic_vector)+cfg.fp_latency,work={'BN_mean_scale':width});sch.barrier(tag+'_mean')
 def finish_var(width,tag):
  sch.task(tag+'/var_eps_rsqrt',2*ceildiv(width,cfg.statistic_vector)+2*cfg.fp_latency+width+cfg.rsqrt_latency-1,requires=(tag+'_mean',),work={'BN_variance_scale':width,'BN_epsilon_add':width,'BN_rsqrt':width})
  sch.barrier(tag+'_ready',(tag+'_mean',))
 def first(start,n):read_u(start,n);psn(n,c,'PSN1');fc(n,c,4*c,'FC1')
 def second(n):normalize(n,4*c,'BN1','BN1_ready');psn(n,4*c,'PSN2');fc(n,4*c,c,'FC2')
 def emit(start,n):
  normalize(n,c,'BN2','BN2_ready');covered=sum(x['area'] for x in p[start:start+n]);out=t*b*covered*c
  sch.dma('output/original_residual',out*4,'residual_read',scratch(n),('BN2_ready',))
  sch.task('output/repeat_plus_residual',3*ceildiv(out,cfg.statistic_vector)+cfg.fp_latency,requires=('BN2_ready',),scratch=scratch(n),work={'residual_add':out,'repeat_fanout_elements':t*b*c*sum(x['area'] for x in p[start:start+n] if not x['fine'])})
  sch.dma('output/full_grid',out*4,'output_write',scratch(n),('BN2_ready',))
 if policy=='spill':
  for start,n in chunks:
   first(start,n);moment(start,n,4*c,'mean','BN1');sch.dma('BN1/Z1_store',t*b*n*4*c*4,'BN1_write',scratch(n))
  finish_mean(4*c,'BN1')
  for start,n in chunks:
   bn_input(start,n,4*c,'BN1');moment(start,n,4*c,'variance','BN1')
  finish_var(4*c,'BN1')
  for start,n in chunks:
   bn_input(start,n,4*c,'BN1');second(n);moment(start,n,c,'mean','BN2');sch.dma('BN2/Z2_store',t*b*n*c*4,'BN2_write',scratch(n))
  finish_mean(c,'BN2')
  for start,n in chunks:bn_input(start,n,c,'BN2');moment(start,n,c,'variance','BN2')
  finish_var(c,'BN2')
  for start,n in chunks:bn_input(start,n,c,'BN2');emit(start,n)
 else:
  for start,n in chunks:first(start,n);moment(start,n,4*c,'mean','BN1')
  finish_mean(4*c,'BN1')
  for start,n in chunks:first(start,n);moment(start,n,4*c,'variance','BN1')
  finish_var(4*c,'BN1')
  for start,n in chunks:first(start,n);second(n);moment(start,n,c,'mean','BN2')
  finish_mean(c,'BN2')
  for start,n in chunks:first(start,n);second(n);moment(start,n,c,'variance','BN2')
  finish_var(c,'BN2')
  for start,n in chunks:first(start,n);second(n);emit(start,n)
 sch.barrier('output_ready',('BN2_ready',));sch.validate()
 expected1=t*b*r*(1 if policy=='spill' else 5);expected2=t*b*r*(1 if policy=='spill' else 3)
 require(sch.work['FC1_rows']==expected1 and sch.work['FC2_rows']==expected2,'Actual scheduled producer rows including recomputation')
 require(sch.traffic['output_write']==full*4 and sch.traffic['residual_read']==full*4,'Full grid output/residual conservation')
 if policy=='spill':require(sch.traffic['BN1_write']==u*16 and sch.traffic['BN1_read']==u*32 and sch.traffic['BN2_write']==u*4 and sch.traffic['BN2_read']==u*8,'Stored Z write/two read conservation')
 else:require(not any(sch.traffic[x] for x in ('BN1_write','BN1_read','BN2_write','BN2_read')),'Recompute has no hidden Z retention')
 output={'name':name,'shape_TBHWC':list(shape),'policy':policy,'representatives':r,'original_spatial_points':h*w,'coarse_representatives':coarse,'coarse_covered_pixels':coarse_area,
  'block_representatives':block,'blocks':len(chunks),'nominal_FC_rows_each':t*b*r,'work':dict(sch.work),'traffic_bytes':dict(sch.traffic),
  'activation_peak_bytes':sch.peak,'global_BN_boundaries':2,'moment_finalization_boundaries':4,'modeled_blocking_schedule_ticks':sch.tick,
  'time_scope':'Declared uncalibrated abstract microarchitecture; no RTL/DC/cycle-accurate datapath or whole-net claim'}
 if keep_events:output['events']=sch.events
 return output,sch


def external_bn_records():
 require(sha(BNFILE)=='9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567','Pinned real external54 shape/source evidence')
 value=json.loads(BNFILE.read_text());rows=value['external_BN']['records']
 for name,digest in value['native_source_sha256'].items():require(sha(REPO/'SDformer/third_party/SDformerFlow'/name)==digest,'Actual native shape source unchanged')
 require(len(rows)==54 and sum(x['elements'] for x in rows)==976584960,'Real external54 inventory')
 return rows


def external_bn_inventory():
 out=[]
 for row in external_bn_records():
  width=row['channels'];n=row['reduction_elements_per_channel'];work=Counter(mean_work(n,width,0));work.update(variance_work(n,width,0));work.update(norm_work(n,width));work.update(BN_mean_scale=width,BN_variance_scale=width,BN_epsilon_add=width,BN_rsqrt=width)
  out.append({'name':row['name'],'shape':row['native_input_shape'],'elements':row['elements'],'work':dict(work),
   'unfused_external_input_reads':3*row['elements']*4,'external_output_write':row['elements']*4,'global_BN_boundaries':1,'moment_finalization_boundaries':2,
   'scope':'Complete common54 arithmetic/byte inventory only; no serial54 cycle estimate reused. Producer-fused mean may reduce common traffic; surrounding producers unmodeled.'})
 return out


def issue_slots(work):
 # Explicit optimistic shared-PE assumption: one scalar FMA/ADD/MUL/threshold
 # issue each. Rsqrt is separate and intentionally not converted into FMA slots.
 return sum(v for k,v in work.items() if k not in ('FC1_rows','FC2_rows','repeat_fanout_elements','BN_rsqrt'))


def suite(cfg):
 g=geometry();all_summaries={};external=external_bn_inventory();common_external=Counter()
 for row in external:common_external.update(row['work'])
 for arm in ('native','FF','CC','P','checker'):
  for policy in ('spill','recompute'):
   rows=[];work=Counter();traffic=Counter()
   for name,shape,selected in g:
    h,w=shape[2:4];kind='FF'
    if selected and arm in ('CC','P'):kind='CC'
    if arm=='P' and '.layers.2.swin_blocks.2.' in name:kind='FF'
    if selected and arm=='checker':kind='checker'
    # The four unselected full-grid MLPs share the same spill service in every
    # comparison; only selected producer lifetime choice is being screened.
    record,_=layer(name,shape,masks(h,w,kind),policy if selected else 'spill',cfg)
    rows.append(record);work.update(record['work']);traffic.update(record['traffic_bytes'])
   nominal=sum(x['nominal_FC_rows_each'] for x in rows)
   if arm in ('native','FF','CC','P'):require(nominal=={'native':558000,'FF':558000,'CC':153600,'P':164200}[arm],'Independent whole12 nominal row total')
   all_summaries[arm+'/'+policy]={'layers':rows,'producer_work':dict(work),'producer_traffic_bytes':dict(traffic),'nominal_FC_rows_each':nominal,
    'executed_FC1_rows':work['FC1_rows'],'executed_FC2_rows':work['FC2_rows'],'producer_issue_slots_hypothetical':issue_slots(work),
    'producer_blocking_schedule_ticks':sum(r['modeled_blocking_schedule_ticks'] for r in rows),'activation_peak_bytes':max(r['activation_peak_bytes'] for r in rows)}
 require(all_summaries['native/spill']['producer_work']==all_summaries['FF/spill']['producer_work'],'Same dense/current and allfine shared compact graph')
 base=all_summaries['native/spill'];comparisons={};known_denominator=base['producer_issue_slots_hypothetical']+issue_slots(common_external)
 for key,row in all_summaries.items():
  saved=base['producer_issue_slots_hypothetical']-row['producer_issue_slots_hypothetical']
  comparisons[key]={'cancelled_minus_added_producer_issue_slots':saved,'producer_work_exhausted':saved<=0,'known_dense_producer_plus_external54_issue_slots':known_denominator,
   'optimistic_savings_fraction_of_known_work':saved/known_denominator,
   'ten_percent_net_PE_gate':'UNRESOLVED: attention/conv/other operators and actual PE implementation/scheduling/predictor still missing',
   'same_mask_ordinary_compact_difference':0,'scope':'Operation-issue feasibility screen with explicit common shared PE assumption, not FLOPs-to-cycle conversion or measured speedup'}
 ordinary8=[]
 for name,(t,b,h,w,c),selected in g:
  if selected:continue
  for bn,width in (('bn1',4*c),('bn2',c)):
   ordinary8.append({'name':name+'.'+bn,'shape_TBCHW':[t,b,width,h,w],'elements':t*b*h*w*width})
 require(len(ordinary8)==8 and sum(x['elements'] for x in ordinary8)==115200000,'Unselected ordinary8 current-BN inventory')
 return {'schema':'flow_current_bn_round16_finite_service_v1','status':'CPU_SCHEDULE_AND_WORK_SCREEN_ONLY','config':asdict(cfg),
  'source_sha256':{str(GFILE):sha(GFILE),str(BNFILE):sha(BNFILE),str(Path(__file__)):sha(__file__)},
  'contracts':{'T':10,'B':1,'tile':9,'coarse_cell_stride':3,'normalization':'population current statistics weighted by actual spatial coverage; complete T and all spatial/channel domains',
   'BN_boundaries_selected8':16,'ordinary_current_BN_unselected4':8,'ordinary_current_BN_external':54,'total_current_BN':78,
   'two_pass_moment_finalization_boundaries_selected8':32,'ordinary_score_and_flow_score_share_executor':True,
   'SRAM_ports':'activation16 banks each1RW word/tick; weight32 banks each1R word/tick; shared DRAM64bytes/tick, serialized jobs; statistics channel RF16R+16W within reserved128KiB',
   'FC_mapping':'row8/output32 blocked dot, FP32FMA latency4 serial K recurrence; output stripe <=512KiB; weights reread from external once per spatial producer block; accumulator registers4KiB',
   'statistics_mapping':'SIMD16 channel pipeline, FP32 latency4, channel recurrence wait>=4; separate sequential mean and centered variance; coefficients/partial vectors fit reserved SRAM',
   'PSN_mapping':'completeT10 matrix per independent spatial/channel column; FP32spike storage; no bitpacking or temporal pruning'},
  'strategies':all_summaries,'common_unselected8_BN':{'records':ordinary8,'elements':115200000,'scope':'Already charged inside the four unselected MLPs, not added a second time.'},'common_external54':external,'common_external54_work':dict(common_external),'comparisons':comparisons,
  'unknown_or_excluded':['No validated FP32 RTL/DC latency, area, achievable frequency, energy or IEEE-equivalent arithmetic datapath.',
   'Attention, convolution, patch/down/up sampling and surrounding producer schedules absent; no compatible whole-net denominator.',
   'Score predictor, mask computation, index/control/address generation and physical layout conversion costs not closed; assumed zero only in optimistic work screen.',
   'Detailed SRAM address-bank allocation and PE routing/weight broadcasting not RTL implemented. Capacity/bus/dependency schedule tests do not establish physical timing.',
   'Inter-layer overlap and optimal cache reuse not solved. Blocking activation/weight transfer schedule is one explicit design candidate, not an optimal engine.',
   'External54 BN byte inventory assumes unfused existing inputs; it is not reused as a serial baseline cycle count.',
   'No quality evidence. Area-weighted compact represents the expanded approximate graph, not original uncompressed flow.'],
  'next_gate':'Only if independent CPU mathematics holds and producer work remains positive should a same-W quality probe and full hardware denominator be considered; no automatic GPU/RTL launch.'}
