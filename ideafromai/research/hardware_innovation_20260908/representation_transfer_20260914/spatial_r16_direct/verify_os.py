from pathlib import Path
import json
H=Path(__file__).resolve().parent;A=H.parent/'spatial_r16_rtl'
profiles=json.loads((H/'os_profiles.json').read_text());rows=[];summaries=[]
keys=['source_words','weight_words','local_gathers','add_issues','psum_reads','psum_writes','psum_clears','bitmap_reads','bitmap_writes']
for name in ['small','held','disjoint']:
 manifest=(H/f'{name}.txt').read_text().splitlines();n=len(manifest)
 for stall in [0,1]:
  records=[json.loads(v) for v in (H/f'os_results_{name}_{stall}.jsonl').read_text().splitlines()]
  assert len(records)==2*n
  for index,r in enumerate(records):
   assert r['command']==index and r['stall']==stall and r['fixture']==manifest[index%n]
   p=profiles[r['fixture']]
   for key in keys:assert r[key]==p[key],(name,index,key,r[key],p[key])
   expected=p['base_states'].copy();expected[2]+=r['source_stalls'];expected[7]+=r['weight_stalls'];expected[11]+=r['output_stalls']
   assert r['state_cycles']==expected,(name,index,'states')
   assert r['cycles']==sum(expected) and r['outputs']==3840
   assert r['configuration_cycles']==1537+(10368 if index==0 else 0)
   assert r['start_cycles']==1 and r['service_cycles']==r['cycles']+r['configuration_cycles']+1
   if not stall:assert r['source_stalls']==r['weight_stalls']==r['output_stalls']==0
   if index>=n:
    for key in keys+['cycles','state_cycles','source_stalls','weight_stalls','output_stalls']:
     assert r[key]==records[index-n][key],(name,index,'repeat',key)
  rows+=records;first=records[:n]
  s=dict(arm='output_stationary_bitmap',set=name,stall=stall,tiles=n,core_cycles=sum(r['cycles'] for r in first),cold_stream_service=sum(r['service_cycles'] for r in first),
   warm_stream_service=sum(r['service_cycles'] for r in records[n:]),static_configuration_cycles=10368,
   source_configuration_cycles=n*1536,origin_configuration_cycles=n,start_cycles=n,
   counters={key:sum(r[key] for r in first) for key in keys},
   source_payload_read_bytes=sum(r['source_words'] for r in first)*10//8,weight_read_bytes=sum(r['weight_words'] for r in first)*32,
   psum_read_bytes=sum(r['psum_reads'] for r in first)*32,psum_write_bytes=sum(r['psum_writes'] for r in first)*32,
   bitmap_read_bytes=sum(r['bitmap_reads'] for r in first)*4,bitmap_write_bytes=sum(r['bitmap_writes'] for r in first)*8,
   configuration_bus_bytes=sum(r['configuration_cycles'] for r in first)*32)
  for label,path in [('factor',A/f'results_{name}_{stall}.jsonl'),('gustav_event',H/f'results_{name}_{stall}.jsonl')]:
   other=[json.loads(v) for v in path.read_text().splitlines()][:n]
   assert len(other)==n and [r['fixture'] for r in other]==manifest
   core=sum(r['cycles'] for r in other);service=sum(r['cycles']+r['configuration_cycles']+1 for r in other)
   s[label]=dict(core_cycles=core,cold_stream_service=service,other_core_reduction_vs_OS=1-core/s['core_cycles'],other_cold_reduction_vs_OS=1-service/s['cold_stream_service'])
  summaries.append(s)
report=dict(passed=True,arm='output_stationary_bitmap',commands=len(rows),outputs=sum(r['outputs'] for r in rows),
 checked=['all raw outputs in RTL TB','actual source-built bitmap activity and nonempty word counts','every source/W/bitmap/psum counter','every FSM state','paid backpressure cycles','configuration/start accounting','no-reset different-tile and repeated pass'],summaries=summaries,
 scope='raw final output; ordinary OS strong control with same function and execution ports; coefficient capacity still differs',
 resource=dict(ALU32=8,multipliers_used=0,factor_multipliers_available_but_idle=8,weight_read_port_bits=256,source_payload_bits=10,psum_row_bits=256,
  source_bytes=1920,native_window_bytes=20,psum_bytes=15360,weight_bytes=331776,weight_support_bytes=1296,
  bitmap_physical_bank_count=8,bitmap_physical_rows_per_bank=40,bitmap_physical_word_bits=32,bitmap_physical_bytes=1280,bitmap_logical_bytes=1080,
  bitmap_write_max_bits=64,bitmap_read_max_bits=32,bitmap_common_row_address=True,bitmap_added_pair_bank_enable_decoder=True,
  bitmap_nonempty_word_support_bits=270,phase_register_bytes=40,phase_register_use='10x32 bitmap assembly OR first8x32 output-stationary acc',result_hold_bytes=32,
  extra_40_by_8_accumulator=False,weight_vectors_per_configuration=10368,physical_configuration_bits_per_beat=256),
 overflow_assertions='active during every granted live MAC, no failure',build='Verilator 4.028; no Warning/Error in final build_os.log')
(H/'OS_SUMMARY.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,commands=report['commands'],outputs=report['outputs'],summaries=[{k:v for k,v in s.items() if k in ['set','stall','core_cycles','cold_stream_service','factor','gustav_event']} for s in summaries]),separators=(',',':')))
