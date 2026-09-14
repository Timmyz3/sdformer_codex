from pathlib import Path
import argparse,json
import numpy as np
H=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--function',choices=['q11','q13'],default='q11');ap.add_argument('--sets',nargs='+',default=['small','held','disjoint']);o=ap.parse_args()
D=H/'q11' if o.function=='q11' else H
profiles=json.loads((D/('profiles.json' if o.function=='q11' else 'os_profiles.json')).read_text())
def readhex(path):return np.array([int(v,16) for v in path.read_text().split()],np.uint32)
sets={};total=0;allrows=[];goldcheck={}
for name in o.sets:
 names=(D/f'{name}.txt').read_text().splitlines();n=len(names);summaries=[]
 for stall in [0,1]:
  rows=[json.loads(v) for v in (D/f'consumer_{name}_{stall}.jsonl').read_text().splitlines()]
  assert len(rows)==2*n
  for i,r in enumerate(rows):
   assert r['fixture']==names[i%n] and r['command']==i and r['stall']==stall
   p=profiles[r['fixture']]
   for key,value in p.items():
    if key not in ['base_cycles','base_states']:assert r[key]==value,(name,i,key,r[key],value)
   for key in ['raw_values','j_values','wide_values','i24_values']:assert r[key]==3840
   expected=p['base_states'].copy();expected[2]+=r['source_stalls'];expected[7]+=r['weight_stalls'];expected[11]+=r['output_stalls'];expected[0]=r['state_cycles'][0]
   assert expected[0]>=6 and r['state_cycles']==expected,(name,i,'states')
   assert r['cycles']==sum(expected)-expected[0] and r['c_cycles']==sum(expected)
   assert r['configuration_cycles']==1537+(10392 if i==0 else 0)
   fixed=dict(c_raw_words=480,c_identity_words=480,c_coefficient_words=24,c_mul_issues=480,c_add_issues=960,c_round_issues=480,c_output_words=480,c_conversion_issues=480,c_wide_waits=0)
   for key,value in fixed.items():assert r[key]==value,(name,i,key,r[key],value)
   if r['fixture'] not in goldcheck:
    path=Path(r['fixture']);fp=readhex(path/'identity.hex').view(np.float32).astype(np.float64)
    jr=np.rint(fp*(1<<20));j=np.clip(jr,-2**31,2**31-1).astype(np.int64)
    assert np.array_equal(j.astype(np.int32).view(np.uint32),readhex(path/'j.hex'))
    wide=readhex(path/'wide.hex').view(np.int64);q=wide//(1<<26);rem=wide-q*(1<<26)
    rounded=q+((rem>(1<<25))|((rem==(1<<25))&((q&1)!=0)))
    assert np.array_equal(np.clip(rounded,-2**23,2**23-1).astype(np.int32).view(np.uint32),readhex(path/'i24.hex'))
    goldcheck[r['fixture']]=(int(np.count_nonzero(jr!=j)),int(np.count_nonzero((rounded<-2**23)|(rounded>2**23-1))))
   conversion_sat,final_sat=goldcheck[r['fixture']]
   assert r['c_conversion_saturations']==conversion_sat and r['c_saturations']==final_sat
   assert r['c_cycles']==3385+r['c_join_wait_cycles']+r['c_output_stalls']+r['c_wide_waits']
   if not stall:
    assert r['source_stalls']==r['weight_stalls']==r['c_identity_stalls']==r['c_output_stalls']==0
    assert r['c_cycles']==p['base_cycles']+2425
   if i>=n:
    for key,value in r.items():
     if key not in ['command','configuration_cycles']:assert value==rows[i-n][key],(name,stall,i,key)
  first=rows[:n];numeric=[key for key,value in first[0].items() if isinstance(value,int) and key not in ['stall','command','configuration_cycles']]
  s={key:sum(r[key] for r in first) for key in numeric}
  s.update(stall=stall,tiles=n,cold_stream_service=sum(r['c_cycles']+r['configuration_cycles']+1 for r in first),warm_stream_service=sum(r['c_cycles']+1538 for r in first),static_configuration_cycles=10392,
   source_configuration_cycles=n*1536,origin_configuration_cycles=n,start_cycles=n,
   identity_bytes=n*15360,raw_join_bytes=n*15360,consumer_coefficient_read_bytes=n*768,
   external_coefficient_configuration_bytes=10392*32,source_origin_configuration_bytes=n*1537*32)
  summaries.append(s);allrows+=rows;total+=len(rows)
 sets[name]=dict(fixtures=n,commands=4*n,summaries=summaries)
report=dict(passed=True,function=o.function,arm='output_stationary_expanded_W32_to_native_FP32_I24',commands=total,raw_values=total*3840,j_values=total*3840,wide_values=total*3840,i24_values=total*3840,exact_native_source_to_i24=True,
 checked=['all raw/J/wide/I24 and ordered last output in actual RTL TB','each producer source/W/bitmap/psum service and every FSM state','finite-state consumer cycle identity','actual finite FP32 conversion and saturation','ready/BP and no-reset repeat equality','static configuration once plus each native input load/origin/start'],sets=sets,
 resource=dict(producer='unchanged os_core.sv, see OS_SUMMARY.json',consumer='read-only original i24_consumer.sv and wide_phase_alu.sv',consumer_multipliers='8 signed32x32',consumer_adders='8x64 shared phase chain',consumer_contexts=1,identity_port_bits=256,raw_join_port_bits=256,consumer_coefficient_port_bits=256,output_port_bits=256),
 scope='same-function full native-source and FP32-identity through final I24; no training, EDA, or new mechanism')
(D/'CONSUMER_SUMMARY.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,function=o.function,commands=total,i24_values=total*3840,sets={name:[{k:r[k] for k in ['stall','cycles','c_cycles','cold_stream_service']} for r in value['summaries']] for name,value in sets.items()}),separators=(',',':')))
