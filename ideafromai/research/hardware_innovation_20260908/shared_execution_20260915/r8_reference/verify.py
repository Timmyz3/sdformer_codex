"""Independent native arithmetic and resource obligations for the common RR input protocol."""
from pathlib import Path
import ast,json
import numpy as np
H=Path(__file__).resolve().parent;B=H.parents[1]
def extract(path,name,env):
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 node.args.defaults=[]
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
 return env[name]
checks=0
def eq(a,b,label=''):
 global checks
 checks+=1
 assert a==b,(label,a,b)
base={'np':np}
native=extract(B/'transfer_adapt_20260914/rr_borrow/verify.py','predict',base)
countmodel=extract(B/'transfer_adapt_20260914/pair_psum_overlay/temporal_pairing/probe.py','model',{'np':np})
env={'np':np,'eq':eq,'native':native,'countmodel':countmodel,'PERM':[8,2,6,3,9,4,1,5,7,0],'raw_values':0,'maxcount':0}
profile=extract(B/'representation_transfer_20260914/bitmap_rr/verify.py','profile',env)
cases=json.loads((H/'fixtures.json').read_text());cache={};stage_names={}
for stage in ['small','held','disjoint','sequences','swap']:
 stage_names[stage]=[Path(s).name for s in (H/f'{stage}.txt').read_text().splitlines()]
rows=[]
for stage in stage_names:
 p=H/f'results_{stage}.jsonl'
 if p.exists():rows += [json.loads(l) for l in p.read_text().splitlines()]
for name in sorted(set(name for r in rows for name in stage_names[r['stage']])):
 c=cases[name];p=Path(c['path']);word=np.fromfile(p/'source.bin','<u2').reshape(96,4,4)
 vals=[np.fromfile(p/f,d).reshape(480,8) for f,d in [('raw.bin','<i4'),('identity_fp32.bin','<f4'),('identity.bin','<i4'),('gold.bin','<i4')]]
 cache[name]=profile(word,c['input_origin_yx'],H/'parameters',*vals)
 raw,j=vals[0].astype(np.int64),vals[2].astype(np.int64)
 ab=np.fromfile(H/'parameters/param7.bin','<i4').reshape(12,2,8).astype(np.int64)
 wide=raw.reshape(12,40,8)*ab[:,0,None,:]+((j.reshape(12,40,8)+ab[:,1,None,:])<<20)
 eq(bool(np.array_equal(wide.reshape(480,8),np.fromfile(p/'wide.bin','<i8').reshape(480,8))),True,'wide independent')
for r in rows:
 mode,n=r['mode'],r['tiles'];pp=[cache[name][mode] for name in stage_names[r['stage']]]
 pred={k:sum(p[k] for p in pp) for k in pp[0]}
 for key,value in pred.items():
  if key!='base_cycles':eq(r['core_'+key],value,(r['stage'],mode,key))
 eq(r['core_cycles'],pred['base_cycles']+sum(r[k] for k in ['core_source_stalls','core_weight_stalls','core_output_stalls','core_arbitration_stalls']),('core cycles',r['stage'],mode))
 eq(r['consumer_cycles'],3385*n+r['consumer_join_wait_cycles']+r['consumer_output_stalls']+r['consumer_wide_waits'],'consumer cycles')
 eq(r['window_cycles'],r['consumer_cycles']+n+n//2,'window')
 eq(r['total_cycles'],r['window_cycles']+r['launch_cycles']+sum(r[k] for k in ['static_words','parameter_stalls','source_load_words','origin_words','source_load_stalls','origin_stalls'])+1,'total')
 eq(r['shared_alu_grants'],r['proof_issues']+(0 if mode==4 else pred['first_issues'])+pred['mac_issues']+pred['repair_issues']+pred['normalization_issues']+(pred['aux_issues'] if mode==21 else 0),'ALU grants')
 eq(r['shared_weight_grants'],pred['weight_words']+pred['metadata_reads'],'weight grants')
 eq(r['shared_psum_grants'],960*n+(pred['aux_reads']+pred['aux_writes'] if mode==21 else 0),'psum grants')
 eq(r['shared_z_grants'],pred['z_vector_reads']+pred['z_scalar_reads']+pred['z_writes'],'z grants')
 eq(r['core_arbitration_stalls'],r['conflict_cycles']+r['borrow_consumer_stalls'],'RR conflicts')
 eq(r['shared_wide_grants'],r['consumer_add_issues']+r['borrow_grants'],'wide grants')
 eq(r['borrow_grants'],pred['first_issues'] if mode==4 else 0,'borrow grants')
 for key in ['outputs','raw_outputs','J_outputs','wide_outputs']:eq(r[key],3840*n,key)
 for key in ['external_source_words','source_load_words']:eq(r[key],1536*n,key)
 eq(r['origin_words'],n);eq(r['padding_words'],0)
 eq(r['first_tile'],0);eq(len(pp),n)
# Ready retains every existing compute/arb count; new BP intentionally loads padding too.
matched=0
for stage in ['held','disjoint']:
 p=B/'representation_transfer_20260914/bitmap_rr'/f'results_{stage}.jsonl'
 old=[json.loads(l) for l in p.read_text().splitlines()]
 for r in rows:
  if r['stage']!=stage or r['stall']!=0:continue
  ref=next(x for x in old if x['mode']==r['mode'] and x['stall']==0 and x['command']==r['command'])
  for k,v in ref.items():
   if k.startswith(('core_','consumer_','shared_','borrow_')) or k in ['total_cycles','window_cycles','conflict_cycles','wide_conflict_cycles','both_compute_cycles','proof_issues']:
    eq(r[k],v,('old ready unchanged',stage,r['mode'],k))
  matched+=1
res=dict(passed=True,checks=checks,commands=len(rows),raw_J_wide_I24_each=sum(r['outputs'] for r in rows),independent_fixtures=len(cache),independent_raw_J_wide_I24_each=env['raw_values'],maximum_count=env['maxcount'],old_ready_records_equal=matched,stages=sorted({r['stage'] for r in rows}),per_context_Z_bytes=2080,total_Z_bytes=4160,Z_service_bits=416,physical_cache_bytes_per_context=384,source_words_per_tile=1536,origin_words_per_tile=1,go_cycles=1)
(H/'verification.json').write_text(json.dumps(res,indent=2)+'\n')
(H/'profiles.jsonl').write_text(''.join(json.dumps(dict(fixture=k,modes=v),separators=(',',':'))+'\n' for k,v in cache.items()))
print(json.dumps(res),flush=True)
