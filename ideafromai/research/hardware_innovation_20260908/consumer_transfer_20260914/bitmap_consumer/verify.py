from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;B=H.parents[1];D=B/'r8_consumer_fusion_20260914/data'
rows=[]
for stage in ['small','short','held','disjoint']:
 p=H/f'results_{stage}.jsonl'
 if p.exists():rows += [dict(json.loads(s),stage=stage) for s in p.read_text().splitlines()]
cases={r['name']:r for r in json.loads((H.parent/'count_rr/fixtures.json').read_text())}
cache={};maps={};checks=0
def eq(a,b,label=''):
 global checks
 checks+=1;assert a==b,(label,a,b)
def ordered(x):return x.reshape(10,12,8,4).transpose(1,3,0,2).reshape(480,8)
def profile(name,tile=None):
 key=name if tile is None else f'native_{tile}'
 if key in cache:return cache[key]
 if tile is None:
  c=cases[name];p=Path(c['path']);oy,ox=c['input_origin'];words=np.fromfile(p/'source.bin','<u2').reshape(96,4,4)
  rawgold,jfp,jgold,igold=[np.fromfile(p/f,dtype).reshape(480,8) for f,dtype in [('raw.bin','<i4'),('identity_fp32.bin','<f4'),('identity.bin','<i4'),('gold.bin','<i4')]]
 else:
  for k,file in [('source','first_source_words.npy'),('raw','raw_p_full.npy'),('fp','identity_fp32_full.npy'),('j','identity_q20_full.npy'),('i','i24_new_full.npy')]:
   if k not in maps:maps[k]=np.load(D/file,mmap_mode='r')
  oy,ox=2*(tile//160)-1,2*(tile%160)-1;p=H.parent/'count_rr/fixtures/real_0';words=np.zeros((96,4,4),np.uint16)
  for y in range(4):
   for x in range(4):
    if 0<=oy+y<240 and 0<=ox+x<320:words[:,y,x]=maps['source'][:,oy+y,ox+x]
  rawgold,jfp,jgold,igold=[ordered(maps[k][tile]) for k in ['raw','fp','j','i']]
 q=np.fromfile(p/'param4.bin','<i4').reshape(864,8).astype(np.int64);v=np.fromfile(p/'param5.bin','<i4').reshape(12,8,8).transpose(0,2,1).reshape(96,8).astype(np.int64)
 ab=np.fromfile(p/'param7.bin','<i4').reshape(12,2,8).astype(np.int64)
 valid=np.array([[0<=oy+y<240 and 0<=ox+x<320 for x in range(4)] for y in range(4)])
 words=words*valid;bits=((words[None]>>np.arange(10)[:,None,None,None])&1).astype(np.int64)
 e=np.stack([bits[:,:,pos//2:pos//2+3,pos%2:pos%2+3].reshape(10,864) for pos in range(4)]).reshape(40,864)
 kl=np.any(q!=0,axis=1);e=e*kl;z=e@q
 raw=np.concatenate([z@v[og*8:og*8+8].T for og in range(12)])
 eq(bool(np.array_equal(raw,rawgold)),True,(key,'raw'))
 j=np.clip(np.rint(jfp.astype(np.float64)*(1<<20)),-2**31,2**31-1).astype(np.int64)
 eq(bool(np.array_equal(j,jgold)),True,(key,'J20'))
 wide=raw.reshape(12,40,8)*ab[:,0,None,:]+((j.reshape(12,40,8)+ab[:,1,None,:])<<20)
 quotient=wide>>26;rem=wide&((1<<26)-1)
 i=np.clip(quotient+((rem>(1<<25))|((rem==(1<<25))&((quotient&1)!=0))),-2**23,2**23-1).reshape(480,8)
 eq(bool(np.array_equal(i,igold)),True,(key,'I24'))
 bitmap=e.reshape(40,54,16);n=bitmap.sum(2);live=n>0;single=n==1
 qb=((q[None]>>np.arange(3)[:,None,None])&1).reshape(3,54,16,8);ql=qb.any((2,3))
 counts=np.einsum('pbk,lbkr->plbr',bitmap,qb)
 eq(bool(np.array_equal(z,counts[:,0].sum(1)+2*counts[:,1].sum(1)-4*counts[:,2].sum(1))),True,'signed planes')
 act=e.reshape(4,10,864).astype(bool);vl=v.reshape(12,8,8).any(1);zl=z!=0
 info=dict(K=int(kl.sum()),Q=int(e.any(0).sum()),U=int((act[0]|act[1]).sum()+(act[2]|act[3]).sum()),DU=int((act[0]&act[1]).sum()+(act[2]&act[3]).sum()),V=int((vl&zl.any(0)).sum()),M=int((zl.sum(0)*vl.sum(0)).sum()),source=int(valid.sum())*96,N=int(live.sum()),single=int(single.sum()),planes=int((live[:,None,:]&ql).sum()),hybrid_planes=int(((live&~single)[:,None,:]&ql).sum()))
 for m,mask in [('full',live),('hybrid',live&~single)]:info[m+'_overlap']=int((mask[:,None,:]&(ql[1:]&ql[:-1])).sum())
 cache[key]=info;return info
for r in rows:
 pp=[profile(r['fixture'],tile) for tile in range(r['first_tile'],r['first_tile']+r['tiles'])] if r['stage']!='small' else [profile(r['fixture'])]
 p={k:sum(a[k] for a in pp) for k in pp[0]};n=r['tiles'];m=r['mode'];bp=sum(r[k] for k in ['core_source_stalls','core_weight_stalls','core_output_stalls'])
 if m==14:
  ex=dict(cycles=5437*n+p['K']+2*p['Q']+3*p['U']+p['M']+bp,first_issues=p['U'],weight_words=p['Q']+p['V'],z_vector_reads=p['U']+40*n,z_writes=p['U']+20*n,aux_reads=0,aux_writes=0,aux_issues=0,aux_weight_words=0,aux_events=0,bitmap_native_reads=0,bitmap_native_issues=0,dual_updates=p['DU'])
 else:
  plane=p['hybrid_planes'] if m==10 else p['planes'];one=p['single'] if m==10 else 0
  scans=p['N'] if m in [13,10] else 2160*n;work=3*p['N']+plane if m==15 else 4*(p['N']-one)+2*one
  ex=dict(cycles=5517*n+(864*n if m==15 else 0)+scans+work+p['M']+bp,first_issues=plane+one,weight_words=plane+one+p['V'],z_vector_reads=40*n,z_writes=60*n,aux_reads=scans,aux_writes=864*n,aux_issues=plane,aux_weight_words=plane,bitmap_native_reads=one,bitmap_native_issues=one,dual_updates=0)
  if not r['stall']:ex['aux_events']=0 if m==15 else p['hybrid_overlap' if m==10 else 'full_overlap']
 ex.update(second_weight_words=p['V'],mac_issues=p['M'],z_scalar_reads=p['M'],source_words=p['source'],local_source_reads=p['K'],psum_reads=480*n,psum_writes=480*n)
 for k,v in ex.items():eq(r['core_'+k],v,(r['fixture'],m,k))
 eq(r['consumer_cycles'],3385*n+r['consumer_join_wait_cycles']+r['consumer_output_stalls'],'consumer obligations')
 eq(r['total_cycles'],r['consumer_cycles']+r['static_words']+r['parameter_stalls']+r['source_load_words']+r['origin_words']+r['source_load_stalls']+2*n+1,'wrapper obligations')
 for k in ['outputs','raw_outputs','J_outputs']:eq(r[k],3840*n)
 eq(r['consumer_add_issues'],960*n);eq(r['consumer_mul_issues'],480*n);eq(r['consumer_conversion_issues'],480*n)
 eq(r['static_words'],1848 if r['command']==0 else 0)
# The component leaf is reused unchanged, with every native/bitmap obligation still checked above.
eq((H/'decomp_core.sv').read_text(),(H.parent/'bitmap_pipeline/decomp_core.sv').read_text(),'leaf unchanged')
out=dict(passed=True,commands=len(rows),rtl_values_each_stage=sum(r['outputs'] for r in rows),independent_tiles=len(cache),independent_raw_J_I24_each=3840*len(cache),counter_checks=checks,leaf_reused_unchanged=True,contexts=1,z_port_bits=208,bitmap_bytes=4320,extra_coefficient_copy_bytes=2592,pop16_trees=8,stages=sorted(set(r['stage'] for r in rows)))
(H/'verification.json').write_text(json.dumps(out,indent=2)+'\n');(H/'profiles.jsonl').write_text(''.join(json.dumps(dict(fixture=k,**v))+'\n' for k,v in cache.items()));print(json.dumps(out),flush=True)
