#!/usr/bin/env python3
"""Independent closed-form accounting and saved transaction checks. No model/GPU."""
import hashlib,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
def require(x,m):
 if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
supply=read(HERE/'results/SUPPLY_REPLAY_V1.json')
inv=read(REPO/'research/flow_story_20261019/round27/hardware/UNIFIED_SERVICE_V1.json')['inventory']['affines']
prefix='sttmultires_unet.encoders.swin3d.layers.'
names={'patch.res0.conv0':('patch.res0.conv0',240,320,True),'stage2.fc1':(prefix+'2.swin_blocks.2.mlp.fc1',30,40,False),'stage2.fc2':(prefix+'2.swin_blocks.2.mlp.fc2',30,40,False),'stage0.fc1_K96':(prefix+'0.swin_blocks.0.mlp.fc1',120,160,False)}
checks=[]
for r in supply['results']:
 source,h,w,conv=names[r['name']];op=next(x for x in inv if x['name']==source)
 n,k,rows=op['Cout'],op['K'],op['rows'];ci=op['Cin'];par=r['engine']=='parallel16';precision=r['precision_complete'];width=16 if par else 128
 require((r['rows'],r['K'],r['Cout'])==(rows,k,n),'Frozen actual geometry')
 complete=10*(h//9*9)*(w//9*9);tail=rows-complete
 beats=math.ceil(k/width)*math.ceil(n/16)*(rows if par else complete*precision+tail*8)
 require(beats==r['arithmetic_accepted_beats'],'Independent full-domain accepted beat count')
 require(r['weight_RF_read_bytes']==beats*256,'All requested weight beat bytes')
 require(r['activation_RF_read_bytes']==beats*(4 if par else 16),'All requested activation word bytes')
 require(r['scale_vector_ticks']==2*rows*(n//16),'Exactly one conversion/scale per16 outputs')
 source_reads=10*(3*h-2)*(3*w-2)*ci//8 if conv else rows*ci//8
 require(r['source_input_word_read_bytes']==source_reads,'Independent halo-clipped source-word reads')
 padk=math.ceil(k/width)*width
 require(r['packed_input_write_bytes']==rows*math.ceil(padk/32)*4,'Packed input writes including hardware K zeros')
 jobs=math.ceil(rows/r['batch_rows']);blockbytes=math.ceil(k/width)*(256 if par else 8*256)
 require(r['weight_SRAM_read_bytes']==jobs*(n//16)*blockbytes,'RF all8 prefill, no false weight-SRAM savings')
 require(r['weight_RF_prefill_busy_ticks']==jobs*(n//16)*math.ceil(blockbytes/256),'Fixed256B1RW refill service')
 require(r['peak_input_slot_bytes']<=524288 and r['peak_output_slot_bytes']<=524288 and r['peak_weight_RF_bytes']<=32768,'Finite allocated capacities')
 d=r['BN_dependencies'];require(d['all_affine_done']<d['mean_finalized']<d['variance_finalized']<d['normalized_output_done'],'Full current-BN barriers')
 require(r['BN_full_domain_passes']==3,'Conservative separate mean/variance/normalize passes')
 tr=r['trace_prefix'];dma=sorted((x for x in tr if x['resource']=='DMA'),key=lambda x:x['start'])
 require(all(a['end']<=b['start'] for a,b in zip(dma,dma[1:])),'Sample DMA exclusivity')
 loads=[x for x in tr if x['kind']=='halo_input'];packs=[x for x in tr if x['kind']=='gather_and_pack'];fills=[x for x in tr if x['kind']=='prefill_RF']
 for pk in packs:
  require(all(pk['end']<=x['start']+32 or pk['start']>=x['end'] for x in loads),'Sample1RW pack/DMA input disjoint')
 for b in r['accepted_beat_trace']:
  require(b['weight_RF_address']%256==0,'64 distinct weight banks on every sampled beat')
  require(all(not(x['start']+32<=b['tick']<x['end']) for x in loads),'Sample accepted beats respect input DMA bank ownership')
  require(all(not(x['start']<=b['tick']<x['end']) for x in fills),'Sample accepted beats respect1RW weight RF refill')
  if not par:require(b['plane']>=8-b['bits'] and b['plane']<8,'Q6 bit2 and Q4 bit4 entry')
 checks.append({'name':r['name'],'engine':r['engine'],'bits':precision,'source_inventory_name':source,'status':'PASS','rows':rows,'complete_rows':complete,'true_tail_rows':tail})
rtl=read(HERE/'results/RTL_INTEGER_V1.json')
raw=list(map(int,(REPO/'research/flow_story_20261020/round30/actual_v1/final_vector_export_v1/final_vectors_v1.txt').read_text().split()));number=raw[0];pos=1;ks=[]
for i in range(number):
 k,oldmode=raw[pos:pos+2];pos+=2;values=raw[pos:pos+17*k];pos+=17*k
 require(all(x in (0,1) for x in values[:k]),'Actual old binary operands')
 require(all(-128<=x<=127 for x in values[k:]),'Actual old q8 operands')
 ks.append(k)
require(pos==len(raw) and number==32,'Entire R30 operation payload')
allks=[k for k in (1,16,96,128,129,256,384,768,864,1536,3072) for _ in range(5)]+ks
require(rtl['case_count']==3*len(allks)==261,'165 synthetic +96 reused operand/precision transactions')
for run in rtl['runs']:
 expected=sum(math.ceil(k/128)*18 for k in allks) if run['engine']=='binary_popcount128' else sum(math.ceil(k/16)*3 for k in allks)
 require(run['accepted_beats']==expected,'Independent RTL descriptor beat totals')
 require(run['checked_int32_values']==261*16,'All integer outputs compared in simulation')
 if not run['stalls']:require(run['cycles']==expected+1,'Actual no-bubble steady-state service')
require(rtl['malformed_protocol_rejections']==11,'Protocol6 +invalid precision5')
dc=read(HERE/'results/dc_q6/summary.json');require(dc['clock_period_ns']==4 and dc['worst_setup_slack_ns']>=0,'New Q6 protocol DC meets same4ns')
area_text=(HERE/'results/dc_q6/area.rpt').read_text();require(str(dc['area_library_units']) in area_text,'Area bound to new raw report')
report={'schema':'flow_e4_independent_accounting_checks_v1','status':'PASS','supply_checks':checks,'rtl_cases':261,'old_actual_operands':32,'new_Q6_full_model_forward':False,'new_Q6_protocol_DC':dc,'checked_input_files':{str(p.relative_to(REPO)):sha(p) for p in [HERE/'supply_replay.cpp',HERE/'results/SUPPLY_REPLAY_V1.json',HERE/'rtl/plane_pe.sv',HERE/'rtl/tb_plane_pe.cpp',HERE/'results/RTL_INTEGER_V1.json',HERE/'results/dc_q6/summary.json']},'GPU_used':False,'GT_read':False,'limits':['Closed-form full-domain accounting plus saved prefix address/port checks. Full per-beat addresses were checked at generation time but only bounded samples are saved.','Author checks, not independent-agent review. No numerical BN, power, SRAM synthesis or full-network verification.']}
out=HERE/'results/ACCOUNTING_CHECKS_V1.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'supply_arms':len(checks),'RTL_cases':261,'DC_area':dc['area_library_units']}))
