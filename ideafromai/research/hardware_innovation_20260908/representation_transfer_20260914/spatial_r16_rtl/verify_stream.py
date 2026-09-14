from pathlib import Path
import json
H=Path(__file__).resolve().parent;p=json.loads((H/'profiles.json').read_text());out={};total=0
for name in ['small','held','disjoint']:
 rs=[];names=(H/f'{name}.txt').read_text().splitlines()
 for stall in [0,1]:
  rows=[json.loads(x) for x in (H/f'stream_{name}_{stall}.jsonl').read_text().splitlines()]
  assert len(rows)==2*len(names),(name,stall,len(rows))
  for i,r in enumerate(rows):
   assert r['fixture']==names[i%len(names)] and r['command']==i and r['stall']==stall
   e=p[r['fixture']]
   for k,v in e.items():
    if k not in ['base_cycles','base_states']:assert r[k]==v,(name,i,k,r[k],v)
   for k in ['raw_values','j_values','wide_values','i24_values']:assert r[k]==3840
   assert r['z_values']==1280
   assert r['cycles']==e['base_cycles']+sum(r[k] for k in ['source_stalls','weight_stalls','output_stalls'])
   st=r['state_cycles'];base=e['base_states'];assert sum(st)==r['c_cycles'] and sum(st)-st[0]==r['cycles']
   for j,(a,b) in enumerate(zip(st,base)):
    if j==0:assert a>=6
    elif j==3:assert a==b+r['source_stalls']
    elif j in [6,12]:assert a>=b
    elif j==17:assert a==b+r['output_stalls']
    else:assert a==b,(name,i,j,a,b)
   assert st[6]+st[12]==base[6]+base[12]+r['weight_stalls']
   assert r['configuration_cycles']==1537+(1176 if i==0 else 0)
   fixed={'c_raw_words':480,'c_identity_words':480,'c_coefficient_words':24,'c_mul_issues':480,'c_add_issues':960,'c_round_issues':480,'c_output_words':480,'c_conversion_issues':480,'c_conversion_saturations':0,'c_wide_waits':0}
   for k,v in fixed.items():assert r[k]==v,(name,i,k,r[k],v)
   assert r['c_cycles']==3385+r['c_join_wait_cycles']+r['c_output_stalls']+r['c_wide_waits']
   if not stall:
    assert r['source_stalls']==r['weight_stalls']==r['c_identity_stalls']==r['c_output_stalls']==0
    assert r['c_cycles']==e['base_cycles']+2425
   # Every resident second pass reproduces every numerical event and paid cycle.
   if i>=len(names):
    for k,v in r.items():
     if k not in ['command','configuration_cycles']:assert v==rows[i-len(names)][k],(name,stall,i,k)
  rs+=rows;total+=len(rows)
 summaries=[]
 for stall in [0,1]:
  rr=[r for r in rs if r['stall']==stall and r['command']<len(names)]
  keys=[k for k,v in rr[0].items() if isinstance(v,int) and k not in ['stall','command','configuration_cycles']]
  sums={k:sum(r[k] for r in rr) for k in keys}
  sums.update(stall=stall,tiles=len(rr),cold_stream_service=sum(r['c_cycles']+r['configuration_cycles']+1 for r in rr),resident_stream_service=sum(r['c_cycles']+1538 for r in rr),static_configuration_cycles=1176)
  summaries.append(sums)
 out[name]={'commands':len(rs),'fixtures':len(names),'summaries':summaries}
report={'passed':True,'commands':total,'raw_values':total*3840,'z_values':total*1280,'j_values':total*3840,'wide_values':total*3840,'i24_values':total*3840,'exact_native_source_to_i24':True,'checked':['all raw/Z/J/wide/I24 in RTL testbench','every producer state and resource counter against independent source/factor schedule','consumer finite-state cost identity','ready/BP and no-reset 2 passes identical','configuration+start actual paid cycles'],'sets':out,'scope':'factor complete I24; direct sibling currently raw only; no same-endpoint full-consumer speed ratio inferred'}
(H/'stream_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'commands':total,'sets':{k:[{f:r[f] for f in ['stall','cycles','c_cycles','cold_stream_service']} for r in v['summaries']] for k,v in out.items()}}))
