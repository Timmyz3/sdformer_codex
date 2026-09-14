from pathlib import Path
import json
import numpy as np
H=Path(__file__).resolve().parent;N=H.parent;M=N/'spatial_moment3';R=N/'spatial_winograd/pruned_replay/moment';rows=[];same=set()
for key in ['q1','consumer']:assert (H/f'parameters/{key}.hex').read_bytes()==(M/f'parameters/{key}.hex').read_bytes()==(R/f'parameters/{key}.hex').read_bytes(),key
read=lambda p:np.array([int(v,16) for v in p.read_text().split()],np.uint32)
parentq=read(M/'parameters/q2.hex').reshape(2,12,8,3,8)
assert np.array_equal(read(H/'parameters/q2.hex').reshape(2,12,8,2,8),parentq[:,:,:,[0,2],:])
profile=json.loads((H/'profiles.json').read_text());mp=json.loads((M/'profiles.json').read_text());exact=0
for stage in ['small','held','disjoint','sequences']:
 names=(H/f'{stage}.txt').read_text().splitlines();other=(M/f'{stage}.txt').read_text().splitlines();assert len(names)==len(other)
 for a,b in zip(names,other):
  for field in ['source','origin','gold','identity','j','wide','i24']:assert (Path(a)/f'{field}.hex').read_bytes()==(Path(b)/f'{field}.hex').read_bytes(),(stage,a,field)
  same.add(a)
  source=read(Path(a)/'source.hex').reshape(96,4,4);origin=read(Path(a)/'origin.hex').view(np.int32)
  for y in range(4):
   for x in range(4):
    if not(0<=origin[0]+y<240 and 0<=origin[1]+x<320):source[:,y,x]=0
  ev=((source[None]>>np.arange(10)[:,None,None,None])&1).astype(bool)
  q1=read(H/'parameters/q1.hex').reshape(2,288,8)
  increment=0
  for ss in range(2):
   for k in range(288):
    if q1[ss,k].any():
     g=ev[:,k//3,k%3:k%3+2,:]
     increment+=int((~g[:,:,0]&~g[:,:,1]&g[:,:,2]).sum())
  assert profile[a]['q1_issues']-mp[b]['q1_issues']==increment
 for consumer in [False,True]:
  prefix='consumer' if consumer else 'raw';cycle='c_cycles' if consumer else 'cycles'
  for stall in [0,1]:
   current=[json.loads(x) for x in (H/f'{prefix}_{stage}_s{stall}.jsonl').read_text().splitlines()]
   moment=[json.loads(x) for x in (M/f'{prefix}_{stage}_s{stall}.jsonl').read_text().splitlines()]
   ordinary=[json.loads(x) for x in (R/f'{prefix}_{stage}_m0_s{stall}.jsonl').read_text().splitlines()]
   general=[json.loads(x) for x in (R/f'{prefix}_{stage}_m1_s{stall}.jsonl').read_text().splitlines()]
   assert len(current)==len(moment)==len(ordinary)==len(general)==2*len(names)
   for i,(b,m,o,g) in enumerate(zip(current,moment,ordinary,general)):
    for k in ['source_words','q1_words','local_gathers','psum_reads','psum_writes']:assert b[k]==m[k]==o[k]==g[k],(stage,i,k)
    assert b['q1_issues']>=m['q1_issues']==o['q1_issues']==g['q1_issues']
    if not stall:
     assert b[cycle]-m[cycle]==b['q2_issues']-m['q2_issues']+3*(b['q1_issues']-m['q1_issues'])-1832
     exact+=1
   n=len(names)
   service=lambda a:sum(x[cycle]+x['configuration_cycles']+1 for x in a[:n])
   sums=lambda a:{k:sum(x[k] for x in a[:n]) for k in ['q1_issues','q2_issues','q1_words','q2_words','source_words','cache_writes','z_vector_reads','z_scalar_reads','z_writes','psum_reads','psum_writes']}
   bc,mc=sums(current),sums(moment)
   row=dict(stage=stage,consumer=consumer,stall=stall,tiles=n,box2=service(current),moment3=service(moment),ordinary3tap=service(ordinary),general4M=service(general),
    box2_change_vs_moment3=service(current)/service(moment)-1,box2_improvement_vs_ordinary=1-service(current)/service(ordinary),
    core=sum(x[cycle] for x in current[:n]),configuration=sum(x['configuration_cycles'] for x in current[:n]),box2_counters=bc,moment3_counters=mc,
    ready_difference_breakdown=dict(Q1_extra_cycles=3*(bc['q1_issues']-mc['q1_issues']),Q2_MAC_difference=bc['q2_issues']-mc['q2_issues'],fixed_core_saving=1832*n,static_configuration_saving=192),
    counts={k:sum(x[k] for x in current[:n]) for k in ['count_constructs','count2_fields','count_nonzero_fields']})
   rows.append(row)
report=dict(passed=True,same_function_fixtures=len(same),source_proven_Q1_increment='per row/t: s0=s1=0 and s2=1, actual Q1-live masked; every increment 3 paid states',exact_ready_per_command_checks=exact,
 formula='box2_core - moment3_core = delta_Q2_MAC + 3*delta_Q1_issue - 1832; cold config additionally -192 once',comparisons=rows)
(H/'comparison.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,same_function_fixtures=len(same),fullconsumer=[r for r in rows if r['consumer'] and r['stage']!='small']),separators=(',',':')))
