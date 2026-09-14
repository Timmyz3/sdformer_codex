"""Independent native arithmetic, class legality, service obligations and old controls."""
from pathlib import Path
import ast,json
import numpy as np
H=Path(__file__).resolve().parent;B=H.parents[1];D=B/'r8_consumer_fusion_20260914/data'
def extract(path,name):
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 node.args.defaults=[]
 env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
 return env[name]
native=extract(B/'transfer_adapt_20260914/rr_borrow/verify.py','predict')
countmodel=extract(B/'transfer_adapt_20260914/pair_psum_overlay/temporal_pairing/probe.py','model')
PERM=[8,2,6,3,9,4,1,5,7,0]
cases={r['name']:r for r in json.loads((H/'fixtures.json').read_text())}
rows=[]
for stage in ['small','short','held','disjoint']:
 p=H/f'results_{stage}.jsonl'
 if p.exists():rows.extend(dict(json.loads(s),stage=stage) for s in p.read_text().splitlines())
checks=0;cache={};raw_values=0;maxcount=0
def eq(a,b,label=''):
 global checks
 checks+=1;assert a==b,(label,a,b)
def profile(word,origin,param,rawgold,jfp,jgold,igold):
 global raw_values,maxcount
 q=np.fromfile(param/'param4.bin','<i4').reshape(864,8).astype(np.int64)
 v=np.fromfile(param/'param5.bin','<i4').reshape(12,8,8).transpose(0,2,1).reshape(96,8).astype(np.int64)
 ab=np.fromfile(param/'param7.bin','<i4').reshape(12,2,8).astype(np.int64)
 cls=(np.fromfile(param/'param8.bin','<u4').reshape(864,8)[:,0,None]>>(6*np.arange(4)))&63
 rep=np.fromfile(param/'param9.bin','<i4').reshape(32,8).astype(np.int64);ng=int(np.fromfile(param/'param10.bin','<u4')[0])
 valid=np.array([[0<=origin[0]+y<240 and 0<=origin[1]+x<320 for x in range(4)] for y in range(4)])
 word=word*valid
 n,raw=native(word,q.T,v,int(valid.sum())*96)
 raw=raw.reshape(4,10,12,8).transpose(2,0,1,3).reshape(480,8)
 eq(bool(np.array_equal(raw,rawgold.reshape(480,8))),True,'native raw')
 j=np.clip(np.rint(jfp.astype(np.float64)*(1<<20)),-2**31,2**31-1).astype(np.int64).reshape(480,8)
 eq(bool(np.array_equal(j,jgold.reshape(480,8))),True,'FP identity to J20')
 wide=raw.reshape(12,40,8)*ab[:,0,None,:]+((j.reshape(12,40,8)+ab[:,1,None,:])<<20)
 quo=wide>>26;rem=wide&((1<<26)-1)
 rounded=quo+((rem>(1<<25))|((rem==(1<<25))&((quo&1)!=0)))
 eq(bool(np.array_equal(np.clip(rounded,-2**23,2**23-1).reshape(480,8),igold.reshape(480,8))),True,'I24 RNE')
 raw_values+=3840
 e=np.zeros((864,4,10),np.int64)
 for k in range(864):
  ch,tap=divmod(k,9)
  for pp in range(4):e[k,pp]=(int(word[ch,pp//2+tap//3,pp%2+tap%3])>>np.arange(10))&1
 eq(bool(np.array_equal(np.any(q!=0,axis=1),np.fromfile(param/'param6.bin','<u4').reshape(864,8)[:,0])),True,'k_live')
 direct=q.copy();counts=np.zeros((4,32,4,10),np.int64)
 for g in range(4):
  for k in range(864):
   code=int(cls[k,g]);pair=q[k,2*g:2*g+2]
   eq(code<=32 or code==63,True,'class range')
   if code==0:eq(bool(np.all(pair==0)),True,'zero class')
   elif code!=63:
    eq(code<=ng,True);eq(bool(np.array_equal(pair,rep[code-1,2*g:2*g+2])),True,'representative')
    direct[k,2*g:2*g+2]=0;counts[g,code-1]+=e[k]
  for code in range(1,33):
   size=int(np.count_nonzero(cls[:,g]==code));eq(size==0 or 2<=size<=255,True,'count static bound')
 recovered=e.reshape(864,40).T@direct
 for g in range(4):recovered[:,2*g:2*g+2]+=counts[g].reshape(32,40).T@rep[:,2*g:2*g+2]
 eq(bool(np.array_equal(recovered,e.reshape(864,40).T@q)),True,'count representation')
 maxcount=max(maxcount,int(counts.max()));eq(int(counts.max())<=255,True)
 u0=int(((e[:,0]|e[:,1])+(e[:,2]|e[:,3]))[np.any(q!=0,axis=1)].sum())
 common=dict(source_words=n['source'],local_source_reads=n['K'],second_weight_words=n['V'],z_scalar_reads=n['M'],mac_issues=n['M'],psum_reads=480,psum_writes=480)
 modes={}
 for mode in [0,1,2,3,4]:
  u=u0 if mode==0 else n['U1'] if mode==1 else n['U2'];repair=n['repairs'] if mode==2 else 0;norm=10 if mode==2 else 0
  pred=dict(common,first_issues=u,merged_updates=n['A']-u,weight_words=n['Q']+n['V'],repair_issues=repair,repair_fields=n['fields'] if mode==2 else 0,normalization_issues=norm,z_vector_reads=u+40+repair+norm,z_writes=u+10+repair+norm,base_cycles=5427+n['K']+2*n['Q']+3*u+n['M']+2*(repair+norm))
  pred.update({x:0 for x in ['aux_reads','aux_writes','aux_issues','aux_weight_words','aux_events','metadata_reads','count_checks','count_bank_reads','count_bank_writes']})
  modes[mode]=pred
 for mode,order in [(20,list(range(10))),(21,PERM)]:
  x=countmodel(e,order,q,v,cls,ng);ev=e[:,:,order]
  need=np.any(cls==63,axis=1)&np.any(q!=0,axis=1)&np.any(ev,axis=(1,2))
  dual=int(((ev[:,0]&ev[:,1])+(ev[:,2]&ev[:,3]))[need].sum())
  u=x['direct_U']+x['retire_issues']
  modes[mode]=dict(common,first_issues=u,merged_updates=dual,weight_words=x['direct_A']+x['slots']+n['V'],repair_issues=0,repair_fields=0,normalization_issues=0,z_vector_reads=u+40,z_writes=u+10,base_cycles=x['core_cycles']-10,metadata_reads=x['A'],count_checks=x['source_blocks'],count_bank_reads=x['count_bank_reads'],count_bank_writes=x['count_bank_writes'],aux_reads=x['first_touch_read_vectors']+x['retire_blocks'],aux_writes=x['source_blocks'],aux_issues=x['source_blocks'],aux_weight_words=x['slots'],aux_events=x['retire_issues'])
 return modes
def ordered(x):return x.reshape(10,12,8,4).transpose(1,3,0,2).reshape(480,8)
maps={}
def get(name,tile=None):
 key=name if tile is None else f'native_{tile}'
 if key in cache:return cache[key]
 if tile is None:
  c=cases[name];p=Path(c['path']);origin=c['input_origin'];word=np.fromfile(p/'source.bin','<u2').reshape(96,4,4)
  vals=[np.fromfile(p/f,dtype).reshape(480,8) for f,dtype in [('raw.bin','<i4'),('identity_fp32.bin','<f4'),('identity.bin','<i4'),('gold.bin','<i4')]]
 else:
  for k,file in [('source','first_source_words.npy'),('raw','raw_p_full.npy'),('fp','identity_fp32_full.npy'),('j','identity_q20_full.npy'),('i','i24_new_full.npy')]:
   if k not in maps:maps[k]=np.load(D/file,mmap_mode='r')
  origin=[2*(tile//160)-1,2*(tile%160)-1];p=H/'fixtures/real_0';word=np.zeros((96,4,4),np.uint16)
  for y in range(4):
   for x in range(4):
    yy,xx=origin[0]+y,origin[1]+x
    if 0<=yy<240 and 0<=xx<320:word[:,y,x]=maps['source'][:,yy,xx]
  vals=[ordered(maps[k][tile]) for k in ['raw','fp','j','i']]
 cache[key]=profile(word,origin,p,*vals);return cache[key]
for r in rows:
 mode,n=r['mode'],r['tiles']
 pp=[get(r['fixture'],tile)[mode] for tile in range(r['first_tile'],r['first_tile']+n)] if r['stage']!='small' else [get(r['fixture'])[mode]]*n
 pred={k:sum(p[k] for p in pp) for k in pp[0]}
 for key,value in pred.items():
  if key!='base_cycles':eq(r['core_'+key],value,(r['fixture'],mode,key))
 eq(r['core_cycles'],pred['base_cycles']+sum(r[k] for k in ['core_source_stalls','core_weight_stalls','core_output_stalls','core_arbitration_stalls']),('core cycles',r['fixture'],mode))
 eq(r['consumer_cycles'],3385*n+r['consumer_join_wait_cycles']+r['consumer_output_stalls']+r['consumer_wide_waits'],'consumer cycles')
 eq(r['window_cycles'],r['consumer_cycles']+n+n//2,'window')
 eq(r['total_cycles'],r['window_cycles']+r['launch_cycles']+sum(r[k] for k in ['static_words','parameter_stalls','source_load_words','origin_words','source_load_stalls'])+1,'total')
 eq(r['shared_alu_grants'],r['proof_issues']+(0 if mode in [3,4] else pred['first_issues'])+pred['mac_issues']+pred['repair_issues']+pred['normalization_issues']+pred['aux_issues'],'ALU grants')
 eq(r['shared_weight_grants'],pred['weight_words']+pred['metadata_reads'],'weight grants')
 eq(r['shared_psum_grants'],960*n+pred['aux_reads']+pred['aux_writes'],'psum grants')
 eq(r['shared_z_grants'],pred['z_vector_reads']+pred['z_scalar_reads']+pred['z_writes'],'z grants')
 eq(r['core_arbitration_stalls'],r['conflict_cycles']+r['borrow_consumer_stalls'],'RR conflicts')
 eq(r['shared_wide_grants'],r['consumer_add_issues']+r['borrow_grants'],'wide grants')
 eq(r['borrow_grants'],pred['first_issues'] if mode in [3,4] else 0,'borrow grants')
 eq(r['outputs'],3840*n);eq(r['raw_outputs'],3840*n);eq(r['J_outputs'],3840*n)
# Read old receipts only: unchanged baseline modes retain all old diagnostics.
matched=0;index={(r['fixture'] if r['stage']=='small' else None,r['first_tile'],r['tiles'],r['mode'],r['stall'],r['command']):r for r in rows if not r['cross_mode']}
for folder,files in [(B/'transfer_adapt_20260914/rr_borrow',['results.json','results_small.json','results_64.json']),(B/'transfer_adapt_20260914/rr_borrow/adapt_rr',['results.json','results_small.json','results_64.json'])]:
 for file in files:
  path=folder/file
  if not path.exists():continue
  for r in json.loads(path.read_text()):
   key=(r.get('fixture'),r.get('first_tile'),r['tiles'],r['mode'],r['stall'],r['command'])
   if key not in index:continue
   for k,v in r.items():
    if k not in ['wall_seconds_so_far','fixture']:eq(index[key][k],v,('old control',key,k))
   matched+=1
result=dict(passed=True,checks=checks,commands=len(rows),rtl_values_each_stage=sum(r['outputs'] for r in rows),independent_tiles=len(cache),independent_raw_J_I24_each=raw_values,max_observed_count=maxcount,old_control_records_equal=matched,stages=sorted(set(r['stage'] for r in rows)),class_table_copies=1,representative_table_copies=1,producer_alu=8,producer_multipliers=8,contexts=2,z_port_bits=416,per_context_z_bytes=520,per_context_psum_bytes=15360,class_bytes=2592,representative_bytes=96,permutation_bits=40)
(H/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
(H/'profiles.jsonl').write_text(''.join(json.dumps(dict(fixture=k,modes=v),separators=(',',':'))+'\n' for k,v in cache.items()))
print(json.dumps(result),flush=True)
