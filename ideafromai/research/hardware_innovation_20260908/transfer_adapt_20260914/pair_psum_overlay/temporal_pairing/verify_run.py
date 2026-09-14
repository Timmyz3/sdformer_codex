from pathlib import Path
import sys,json,argparse
import numpy as np
H=Path(__file__).resolve().parent;O=H.parent
sys.path.insert(0,str(O))
from verify import profile
from prepare import readhex,native
from probe import model,PERM
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['small','short','held','disjoint'],default='small');stage=p.parse_args().stage
rows=json.loads((H/f'results_{stage}.json').read_text());cache={};checks=0
def eq(a,b):
 global checks
 checks+=1;assert a==b,(a,b)
nt=1 if stage=='small' else len(set(r['fixture'] for r in rows))
for path in dict.fromkeys(r['fixture'] for r in rows):
 d=Path(path);pa=d if stage=='small' else O/'fixtures/real_0'
 ref=profile(d,pa)
 q=readhex(pa/'q1.hex').reshape(864,8);v=readhex(pa/'q2.hex').reshape(12,8,8).transpose(0,2,1).reshape(96,8)
 cl=(readhex(pa/'class.hex')[:,None]>>(6*np.arange(4)))&63;ng=int(readhex(pa/'ngroups.hex')[0])
 ev,z=native(readhex(d/'source.hex').reshape(96,4,4),q,readhex(d/'origin.hex'))
 a=model(ev.reshape(864,4,10),list(range(10)),q,v,cl,ng);b=model(ev.reshape(864,4,10),PERM,q,v,cl,ng)
 eq(a['core_cycles'],ref[20]['base_cycles']);eq(a['native_core_cycles'],ref[14]['base_cycles'])
 c=dict(ref[20]);c.update(base_cycles=b['core_cycles'],count_checks=b['source_blocks'],count_bank_reads=b['count_bank_reads'],count_bank_writes=b['count_bank_writes'],aux_reads=b['first_touch_read_vectors']+b['retire_blocks'],aux_writes=b['source_blocks'],aux_issues=b['source_blocks'])
 ref[21]=c;ref['pairing_original']=a;ref['pairing_permuted']=b;cache[path]=ref
for r in rows:
 pred=cache[r['fixture']][r['mode']]
 for k,v in pred.items():
  if k!='base_cycles':eq(r[k],v)
 eq(r['cycles'],pred['base_cycles']+sum(r[k] for k in ['source_stalls','weight_stalls','output_stalls']))
 eq(sum(r['state_cycles']),r['cycles'])
 static=1824 if r['mode']==14 else 2721+(r['mode']==21)
 cfg=(1537+static if r['command']==0 else 0) if stage=='small' else 1537+(static if r['command']==0 else 0)
 eq(r['configuration_cycles'],cfg)
 if r['mode'] in [20,21]:
  x=cache[r['fixture']]['pairing_original' if r['mode']==20 else 'pairing_permuted']
  eq(r['state_cycles'][21],x['source_blocks']);eq(r['state_cycles'][23],x['source_blocks']);eq(r['state_cycles'][26],x['retire_blocks']);eq(r['aux_events'],x['retire_issues'])
matched=0
old_file={'small':O/'results.json','short':O/'results_stream_small.json','held':O/'results_stream_64.json'}.get(stage)
if old_file:
 old=json.loads(old_file.read_text());ix={(Path(r['fixture']).name,r['mode'],r['stall'],r['command']):r for r in old}
 for r in rows:
  key=(Path(r['fixture']).name,r['mode'],r['stall'],r['command'])
  if r['mode'] in [14,20] and key in ix:
   for k,v in ix[key].items():
    if k not in ['fixture','reference']:eq(r[k],v)
   matched+=1
summary={}
if stage!='small':
 for m in [14,20,21]:
  for s in [0,1]:
   for repeat in [0,1]:
    rr=[r for r in rows if r['mode']==m and r['stall']==s and r['command']//nt==repeat]
    v={k:sum(r[k] for r in rr) for k in rr[0] if k not in ['mode','stall','command','state_cycles','fixture']}
    v['start_beats']=len(rr);v['service_cycles']=v['cycles']+v['configuration_cycles']+len(rr);v['retire_blocks']=sum(r['state_cycles'][26] for r in rr)
    summary[f'm{m}_s{s}_repeat{repeat}']=v
checksout=dict(passed=True,checks=checks,commands=len(rows),raw_values=sum(r['outputs'] for r in rows),original14_20_all_fields_reproduced=matched,independent_gold_values=len(cache)*3840,permutation=PERM,summary=summary)
(H/f'checks_{stage}.json').write_text(json.dumps(checksout,indent=2)+'\n');(H/f'profiles_{stage}.json').write_text(json.dumps(cache,indent=2)+'\n')
print(json.dumps({k:v for k,v in checksout.items() if k!='summary'},indent=2))
if summary:print(json.dumps({k:{'core':v['cycles'],'service':v['service_cycles'],'updates':v['count_checks'],'retire_blocks':v['retire_blocks']} for k,v in summary.items()},indent=2))
