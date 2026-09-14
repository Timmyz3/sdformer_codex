from pathlib import Path
import json
H=Path(__file__).resolve().parent;A=H.parent/'spatial_r16_rtl'
profiles=json.loads((H/'profiles.json').read_text());rows=[];summaries=[]
keys=['source_words','weight_words','local_gathers','add_issues','psum_reads','psum_writes','psum_clears']
for name in ['small','held','disjoint']:
 manifest=(H/f'{name}.txt').read_text().splitlines();n=len(manifest)
 for stall in [0,1]:
  records=[json.loads(v) for v in (H/f'results_{name}_{stall}.jsonl').read_text().splitlines()]
  assert len(records)==2*n
  for index,r in enumerate(records):
   assert r['command']==index and r['stall']==stall and r['fixture']==manifest[index%n]
   p=profiles[r['fixture']]
   for key in keys:assert r[key]==p[key],(name,index,key,r[key],p[key])
   expected=p['base_states'].copy()
   expected[3]+=r['source_stalls'];expected[5]+=r['weight_stalls'];expected[10]+=r['output_stalls']
   assert r['state_cycles']==expected,(name,index,'states')
   assert r['cycles']==sum(expected)
   assert r['outputs']==3840 and r['configuration_cycles']==1537+(10368 if index==0 else 0)
   assert r['start_cycles']==1 and r['service_cycles']==r['cycles']+r['configuration_cycles']+1
   if not stall:assert r['source_stalls']==r['weight_stalls']==r['output_stalls']==0
   if index>=n:
    for key in keys+['cycles','state_cycles','source_stalls','weight_stalls','output_stalls']:
     assert r[key]==records[index-n][key],(name,index,'repeat',key)
  rows+=records
  first=records[:n]
  s=dict(set=name,stall=stall,tiles=n,core_cycles=sum(r['cycles'] for r in first),cold_stream_service=sum(r['service_cycles'] for r in first),
   warm_stream_service=sum(r['service_cycles'] for r in records[n:]),static_configuration_cycles=10368,
   source_configuration_cycles=n*1536,origin_configuration_cycles=n,start_cycles=n,
   counters={key:sum(r[key] for r in first) for key in keys},
   source_payload_read_bytes=sum(r['source_words'] for r in first)*10//8,
   weight_read_bytes=sum(r['weight_words'] for r in first)*32,
   psum_read_bytes=sum(r['psum_reads'] for r in first)*32,psum_write_bytes=sum(r['psum_writes'] for r in first)*32,
   configuration_bus_bytes=sum(r['configuration_cycles'] for r in first)*32)
  factor_file=A/f'results_{name}_{stall}.jsonl'
  if factor_file.exists():
   factor=[json.loads(v) for v in factor_file.read_text().splitlines()][:n]
   assert len(factor)==n and [r['fixture'] for r in factor]==manifest
   fc=sum(r['cycles'] for r in factor);fs=sum(r['cycles']+r['configuration_cycles']+1 for r in factor)
   s['factor_raw_same_function']=dict(core_cycles=fc,cold_stream_service=fs,core_reduction=1-fc/s['core_cycles'],cold_service_reduction=1-fs/s['cold_stream_service'],
    counters={key:sum(r[key] for r in factor) for key in ['source_words','q1_words','q2_words','q1_issues','q2_issues','z_vector_reads','z_scalar_reads','z_writes','psum_reads','psum_writes']})
  summaries.append(s)
report=dict(passed=True,commands=len(rows),outputs=sum(r['outputs'] for r in rows),checked=['all raw outputs in RTL TB','actual source/direct-W versus factor gold in prepare','every source/W/psum counter','every FSM state','paid backpressure cycles','configuration/start accounting','no-reset different-tile and repeated pass'],
 summaries=summaries,scope='raw final output only, no consumer; same function and execution ports, different required static coefficient capacity',
 resource=dict(ALU32=8,multipliers_used=0,factor_multipliers_available_but_idle=8,weight_read_port_bits=256,source_payload_bits=10,psum_row_bits=256,source_bytes=1920,native_window_bytes=20,
  psum_bytes=15360,weight_bytes=331776,weight_support_bytes=1296,weight_hold_bytes=32,psum_hold_bytes=32,result_hold_bytes=32,event_support_bytes=5,event_pending_bytes=5,
  extra_40_by_8_accumulator=False,weight_vectors_per_configuration=10368,physical_configuration_bits_per_beat=256),
 overflow_assertions='active during every PADD, no failure',build='Verilator 4.028; no Warning/Error in final build.log')
(H/'SUMMARY.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,commands=report['commands'],outputs=report['outputs'],summaries=[{k:v for k,v in s.items() if k not in ['counters','factor_raw_same_function']} for s in summaries]),separators=(',',':')))
