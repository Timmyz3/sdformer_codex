#!/usr/bin/env python3
"""Full recorded-address bounds/ports plus real persisted handoff bytes."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;O=P/'actual_v2'
dtype=np.dtype([('start','<u8'),('end','<u8'),('address','<u4'),('words','<u4'),('mem','u1'),('write','u1'),('vector','u1'),('reserved','u1')]);assert dtype.itemsize==28
def read(p):return json.loads(p.read_text())
reports=[]
for name,servicefile in [('rowband160','zurich_city_07_a_0010.json'),('rowband128','rowband128.json'),('cell160','cell160.json')]:
 d=read(O/servicefile);a=np.memmap(O/(name+'.trace'),mode='r',dtype=dtype);assert len(a)==d['trace_records']
 assert np.all(a['end']>=a['start']) and np.all(a['start'][1:]>=a['end'][:-1]);assert np.all(a['mem']<3) and np.all(a['address']%4==0)
 assert np.all(a['end']-a['start']==np.where(a['vector']!=0,(a['words'].astype(np.uint64)+15)//16,a['words']))
 with (O/(name+'.trace.dma.csv')).open() as f:dm=[{k:int(v) for k,v in r.items()} for r in csv.DictReader(f)]
 ordered=sorted(dm,key=lambda r:r['external_byte_address']);position=0
 for r in ordered:assert r['external_byte_address']==position;position+=r['bytes']
 assert position==12288000 and len(dm)==d['DMA_transactions']
 ingress=a[(a['mem']==0)&(a['write']==1)&(a['address']==0)&(a['vector']==1)&(a['words']>8)]
 assert len(ingress)==len(dm)
 for event,r in zip(ingress,dm):
  assert event['start']==r['start']+32 and event['end']==r['end'] and event['words']==r['bytes']//4
  assert r['end']-r['start']==32+r['bytes']//64
 banks=[]
 for mid,capacity in [(0,524288),(1,4194304),(2,2048)]:
  mask=a['mem']==mid;sub=a[mask];assert np.all(sub['address'].astype(np.uint64)+4*sub['words']<=capacity)
  actual=d['memory'][mid];assert int(np.max(sub['address'].astype(np.uint64)+4*sub['words']))==actual['high_water_address'] if len(sub) else actual['high_water_address']==0
  for write,key in [(0,'bank_reads'),(1,'bank_writes')]:
   use=sub[sub['write']==write];q=use['words']//16;rem=use['words']%16;starts=(use['address']//4)%16
   hist=[int(np.sum(q,dtype=np.uint64)+np.sum((b-starts.astype(np.int64))%16<rem)) for b in range(16)];assert hist==actual[key]
   assert int(np.sum(use['words'],dtype=np.uint64))*4==actual['write_bytes' if write else 'read_bytes']
  ticks=int(np.sum(sub['end']-sub['start'],dtype=np.uint64))
  if mid==0:ticks-=sum(r['bytes']//64 for r in dm)
  assert ticks==actual['serial_service_ticks'];banks.append({'mem':mid,'records':len(sub),'recorded_memory_bytes_and_banks_PASS':True})
 math_ticks=sum(v*(3 if k in ('add','sub','mul') else 16 if k in ('div','sqrt') else 1) for k,v in d['ops'].items())
 assert math_ticks+sum(m['serial_service_ticks'] for m in d['memory'])+d['DMA_ticks']==d['feature_ready_tick']
 # Persistent X region must never overlap DMA, staging or gradient writes.
 protected=(a['mem']==0)&(a['write']==1)&(a['address']<400640)&(a['address'].astype(np.uint64)+a['words']*4>400000)
 featurewrites=a[protected];assert len(featurewrites)==(32 if name!='rowband128' else 16)
 assert np.all(featurewrites['address']>=400000) and np.all(featurewrites['address']+featurewrites['words']*4<=400640)
 assert sum(featurewrites['words'])==(160 if name!='rowband128' else 128)
 reports.append(dict(layout=name,trace_records=len(a),full_DMA_exactly_once_input_coverage=True,all_recorded_memory_bounds_and_ports_PASS=True,protected_feature_write_count=len(featurewrites),ticks_recomputed=d['feature_ready_tick'],scalar_RF_addresses_instrumented=False,bank_checks=banks))
execution=read(O/'STREAM_EXECUTION_V1.json');payloads=[]
for row in execution['rows']:
 x=np.fromfile(O/row['feature_file'],dtype=np.float32);memory=np.fromfile(O/(row['feature_file']+'.ios'),dtype=np.float32);assert len(memory)==524288//4;assert np.array_equal(x,memory[100000:100160]);payloads.append(hashlib.sha256(memory[100000:100160].tobytes()).hexdigest())
heads=read(O/'HEAD_PREFIX_RESULTS_V1.json')
for name,h in heads['models'].items():
 events=read(O/(name+'_HEAD_EVENTS_V1.json'));n=128 if name.startswith('X128') else 160
 feature=[e for e in events if e['space']=='IOS' and not e['write'] and e.get('address',0)>=400000]
 assert sum(e['words'] for e in feature)>=n
 assert all(400000<=e['address'] and e['address']+e['words']*4<=400000+n*4 for e in feature)
 assert h.get('feature_stream_action_flips_fixed',0)==0 and h.get('feature_stream_action_flips_allow',0)==0
out=dict(schema='r34_hardware_v2_full_trace_handoff_audit',status='PASS_RECORDED_MEMORIES_AND_ACTUAL_IOS_HANDOFF',reports=reports,persisted_120_feature_payload_SHA256=payloads,head_same_address_reads_verified=True,scalar_register_scope='not address instrumented; bounded allocation and 2R1W/result-routing assumptions in RESOURCE_SCOPE_V2.md',full_system_latency_proven=False)
(O/'TRACE_HANDOFF_AUDIT_V2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='persisted_120_feature_payload_SHA256'},indent=2))
