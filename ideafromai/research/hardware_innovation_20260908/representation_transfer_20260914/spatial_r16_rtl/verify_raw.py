from pathlib import Path
import json
H=Path(__file__).resolve().parent
p=json.loads((H/'profiles.json').read_text());out={};total=0
for name in ['small','held','disjoint']:
 rs=[]
 for stall in [0,1]:
  rows=[json.loads(x) for x in (H/f'results_{name}_{stall}.jsonl').read_text().splitlines()]
  names=(H/f'{name}.txt').read_text().splitlines();assert len(rows)==2*len(names)
  for i,r in enumerate(rows):
   assert r['fixture']==names[i%len(names)] and r['command']==i and r['stall']==stall
   e=p[r['fixture']]
   for k,v in e.items():
    if k not in ['base_cycles','base_states']:assert r[k]==v,(name,i,k,r[k],v)
   assert r['outputs']==3840 and r['z_values']==1280
   assert r['cycles']==e['base_cycles']+sum(r[k] for k in ['source_stalls','weight_stalls','output_stalls'])
   states=r['state_cycles'];base=e['base_states']
   for j,(a,b) in enumerate(zip(states,base)):
    if j==3:assert a==b+r['source_stalls']
    elif j in [6,12]:assert a>=b
    elif j==17:assert a==b+r['output_stalls']
    else:assert a==b,(name,i,j,a,b)
   assert states[6]+states[12]==base[6]+base[12]+r['weight_stalls']
   assert r['configuration_cycles']==1537+(1152 if i==0 else 0)
   if not stall:assert sum(r[k] for k in ['source_stalls','weight_stalls','output_stalls'])==0
   if i>=len(names):
    for k,v in r.items():
     if k not in ['command','configuration_cycles']:assert v==rows[i-len(names)][k],(name,stall,i,k)
  total+=len(rows);rs+=rows
 ready=[r for r in rs if r['stall']==0 and r['command']<len(names)]
 keys=['cycles','source_words','q1_words','q2_words','local_gathers','q1_issues','q2_issues','z_vector_reads','z_scalar_reads','z_writes','psum_reads','psum_writes','cache_writes']
 out[name]={'commands':len(rs),'fixtures':len(names),'ready_mean':{k:sum(r[k] for r in ready)/len(ready) for k in keys},'ready_state_mean':[sum(r['state_cycles'][j] for r in ready)/len(ready) for j in range(19)]}
 # Cold service includes native tile config, source coordinate, all weight configuration, start command.
 out[name]['ready_mean']['per_tile_reset_cold_service_cycles']=out[name]['ready_mean']['cycles']+1537+1152+1
 out[name]['ready_mean']['resident_service_cycles']=out[name]['ready_mean']['cycles']+1537+1
 out[name]['ready_cold_stream_service']=sum(r['cycles']+r['configuration_cycles']+1 for r in ready)
(H/'raw_verification.json').write_text(json.dumps({'passed':True,'commands':total,'raw_values':total*3840,'z_values':total*1280,'independent_schedule_and_counters':True,'sets':out},indent=2)+'\n')
print(json.dumps({'passed':True,'commands':total,'sets':{k:v['ready_mean'] for k,v in out.items()}}))
