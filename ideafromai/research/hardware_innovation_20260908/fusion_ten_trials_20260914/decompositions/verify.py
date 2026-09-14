"""Read-only independent algebra and exact architecture-level transaction audit."""
from pathlib import Path
import json,re,numpy as np
H=Path(__file__).resolve().parent
OLD=H.parents[1]/'r8_consumer_fusion_20260914/packed_rtl'
old=json.loads((OLD/'results.json').read_text())
def words(p):return np.array([int(x,16) for x in p.read_text().split()],dtype=np.int64)
def signed(a,b):
 a=a&((1<<b)-1);return np.where(a&(1<<(b-1)),a-(1<<b),a)
all_info={}
for name in ['q1_bitplanes','q2_da','q1_dictionary']:
 h=H/name;rows=json.loads((h/'results.json').read_text());states=re.search(r'\{IDLE,(.*?)\} state_t', (h/'decomp_core.sv').read_text(),re.S).group(1)
 names=['IDLE']+[x.strip() for x in states.split(',')];idx={s:i for i,s in enumerate(names)};details=[];old_checks=0;table_values=0;da_width_hist={}
 for fixture in sorted({x['fixture'] for x in rows}):
  p=h/'fixtures'/fixture;q=signed(words(p/'q1.hex'),3).reshape(864,8).T;v=signed(words(p/'q2.hex'),16).reshape(12,8,8).transpose(0,2,1).reshape(96,8)
  sw=words(p/'source.hex');src=((sw[None,:]>>np.arange(10)[:,None])&1).reshape(10,96,4,4)
  oy,ox=signed(words(p/'origin.hex'),16);valid=(np.arange(4)[:,None]+oy>=0)&(np.arange(4)[:,None]+oy<240)&(np.arange(4)[None,:]+ox>=0)&(np.arange(4)[None,:]+ox<320)
  src*=valid[None,None];x=np.stack([src[:,:,a//2:a//2+3,a%2:a%2+3].reshape(10,864) for a in range(4)]);xf=x.reshape(40,864)
  z=xf@q.T;gold=z@v.T;gw=gold.reshape(4,10,12,8).transpose(2,0,1,3).reshape(-1)
  assert np.array_equal(gw,signed(words(p/'gold.hex'),32))
  live=np.any(q!=0,axis=0);assert np.array_equal(live,words(p/'k_live.hex').astype(bool))
  xx=x*live;pair=(xx[0]|xx[1]).sum()+(xx[2]|xx[3]).sum();qreads=int(np.count_nonzero(np.any(x,axis=(0,1))&live))
  vv=np.any(v.reshape(12,8,8)!=0,axis=1);rl=np.any(z!=0,axis=0);second=int(np.count_nonzero(vv&rl));mac=sum(np.count_nonzero(z[:,vv[g]]) for g in range(12))
  extra={}
  if name=='q1_bitplanes':
   xb=(xf*live).reshape(40,54,16);bp=np.stack([((q&7)>>b)&1 for b in range(3)]).reshape(3,8,54,16)
   bp_live=np.any(bp,axis=(1,3));nonnull=np.any(xb,axis=2);nplanes=int(np.sum(nonnull[:,None,:]*bp_live[None,:,:]));nsource=int(nonnull.sum())
   zp=np.zeros((40,8),np.int64)
   for b in range(3):zp+=(1<<b)*(1 if b<2 else -1)*np.einsum('fkv,rkv->fr',xb,bp[b])
   assert np.array_equal(zp,z)
   extra={'active_bitmap16_words':nsource,'coefficient_plane_reads':nplanes,'bitmap_bits_overwritten':34560}
  elif name=='q2_da':
   updates=0;positions=0;lut_nonzero=0;zout=np.zeros_like(gold)
   for og in range(12):
    vs=v[og*8:og*8+8];lut=np.zeros((2,16,8),np.int64)
    for g in range(2):
     for m in range(16):lut[g,m]=vs[:,g*4:g*4+4]@np.array([(m>>r)&1 for r in range(4)])
    assert lut.min()>=-(1<<17) and lut.max()<(1<<17);table_values+=256;lut_nonzero+=int(np.any(lut,axis=2).sum())
    for pos,zrow in enumerate(z):
     zz=zrow*vv[og]
     if not np.any(zz):continue
     positions+=1;w=next(w for w in range(1,14) if np.all(zz>=-(1<<(w-1))) and np.all(zz<(1<<(w-1))))
     da_width_hist[str(w)]=da_width_hist.get(str(w),0)+1
     for b in range(w):
      for g in range(2):
       m=sum(int((zz[g*4+r]>>b)&1)<<r for r in range(4))
       if np.any(lut[g,m]):updates+=1;zout[pos,og*8:og*8+8]+=(1<<b)*(1 if b<w-1 else -1)*lut[g,m]
   assert np.array_equal(zout,gold)
   extra={'DA_LUT_updates':updates,'DA_positions':positions,'built_lut_nonzero_vectors':lut_nonzero}
  else:
   cls=words(p/'class.hex');rep=words(p/'representative.hex');ng=int(words(p/'ngroups.hex')[0]);assert 0<=ng<=32
   counts=np.zeros((40,ng),np.int64)
   for g in range(ng):
    ids=cls==g+1;assert ids.sum()>=2 and np.all(q[:,ids]==q[:,rep[g]][:,None]);counts[:,g]=xf[:,ids].sum(axis=1)
   assert np.all(cls<=ng) and np.all(counts<=864)
   zd=xf[:,cls==0]@q[:,cls==0].T
   if ng:zd+=counts@q[:,rep[:ng]].T
   assert np.array_equal(zd,z)
   direct=xx[:,:,cls==0];directpair=int((direct[0]|direct[1]).sum()+(direct[2]|direct[3]).sum())
   activegroup=int(np.any(counts,axis=0).sum()) if ng else 0;retire=int(np.count_nonzero(counts))
   direct_q=int(np.count_nonzero(np.any(x,axis=(0,1))&live&(cls==0)))
   cblocks=0;class_active_k=0
   for k in np.flatnonzero(cls):
    pr=np.concatenate([xx[0,:,k]|xx[1,:,k],xx[2,:,k]|xx[3,:,k]])
    if np.any(pr):class_active_k+=1;cblocks+=sum(np.any(pr[a:a+8]) for a in [0,8,16])
   extra={'dictionary_groups':ng,'dictionary_active_groups':activegroup,'retired_nonzero_counts':retire,'count_update_blocks':int(cblocks),'direct_pair_updates':directpair,'direct_Q1_reads':direct_q,'classified_active_k':class_active_k}
  for r in [x for x in rows if x['fixture']==fixture]:
   st={n:r['state_cycles'][i] for n,i in idx.items()};assert sum(r['state_cycles'])==r['cycles'] and r['outputs']==3840
   assert r['source_words']==int(valid.sum())*96 and r['second_weight_words']==second
   assert st['L_LOAD']==1536+r['source_stalls'] and st['L_START']==96 and st['L_GATHER']==864 and st['KNEXT']==864
   assert st['ZCLEAR']==20 and st['ZSCAN']==40 and st['STORE']==st['DRAIN_READ']==480 and st['DRAIN_SEND']==480+r['output_stalls'] and st['FINISH']==1
   assert r['psum_reads']==r['psum_writes']==480 and r['local_source_reads']==int(live.sum())
   assert r['configuration_cycles']==((4258 if name=='q1_dictionary' else 3361) if r['command']==0 else 0)
   if r['mode']==14:
    a=next(o for o in old if o['fixture']==fixture and o['mode']==15 and o['stall']==r['stall'] and o['command']==r['command'])
    for key in ['cycles','source_words','weight_words','second_weight_words','local_source_reads','z_vector_reads','z_scalar_reads','z_writes','first_issues','dual_updates','psum_reads','psum_writes','mac_issues','source_stalls','weight_stalls','output_stalls']:
     assert r[key]==a[key],(name,fixture,key,r[key],a[key]);old_checks+=1
    assert r['state_cycles'][:19]==a['state_cycles'] and sum(r['state_cycles'][19:])==0
   elif name=='q1_bitplanes':
    assert r['first_issues']==r['aux_issues']==r['aux_weight_words']==nplanes
    assert r['aux_reads']==st['BM_SCAN']==2160 and r['aux_writes']==st['BM_PACK']==864
    assert st['BM_QREAD']==nsource*3+r['weight_stalls']-(st['VLOAD']-96)
    assert st['BM_POP']==nplanes and st['BM_POS']==st['BM_STORE']==40
    assert r['weight_words']==nplanes+second and r['mac_issues']==mac
    assert r['z_writes']==60 and r['z_vector_reads']==40 and r['z_scalar_reads']==mac
   elif name=='q2_da':
    assert st['DA_BUILD']==r['aux_writes']==384 and st['DA_POS']==st['DA_ENCODE']==positions
    assert st['DA_MAC']==r['aux_events']==updates and r['aux_reads']==r['aux_issues']==360+updates
    assert r['weight_words']==qreads+second and r['mac_issues']==0 and r['z_scalar_reads']==0
    assert r['first_issues']==pair and r['z_writes']==pair+20 and r['z_vector_reads']==pair+40+positions
   else:
    assert st['DCLEAR']==3*ng and st['C_CHECK']==3*class_active_k and st['C_READ']==st['C_ADD']==cblocks
    assert st['G_START']==(ng+1 if ng else 0) and st['G_READ']==3*activegroup and st['G_SELECT']==3*activegroup+retire
    assert st['G_ZREAD']==st['G_MAC']==retire and r['aux_events']==retire
    assert r['aux_reads']==cblocks+3*activegroup and r['aux_writes']==cblocks+3*ng and r['aux_issues']==cblocks
    assert r['weight_words']==direct_q+activegroup+second and r['aux_weight_words']==activegroup and r['mac_issues']==mac
    assert r['first_issues']==directpair+retire and r['z_vector_reads']==directpair+retire+40 and r['z_writes']==directpair+retire+20
  for m in [14,15]:
   for sf in [0,1]:
    rr=[r for r in rows if r['fixture']==fixture and r['mode']==m and r['stall']==sf]
    assert len(rr)==2 and all(rr[0][k]==rr[1][k] for k in rr[0] if k not in ['command','configuration_cycles'])
  details.append({'fixture':fixture,**extra})
 out={'runs':len(rows),'gold_values_recomputed':14*3840,'old_baseline_scalar_metric_matches':old_checks,'old_baseline_state_array_matches':56,'state_names':names,'lut_values_checked':table_values,'da_width_histogram':da_width_hist,'fixtures':details,
 'scope':'Read-only full integer algebra and transaction derivation; measured cycles remain RTL results. No new simulations or performance predictions used as evidence.'}
 (h/'checks.json').write_text(json.dumps(out,indent=2)+'\n');all_info[name]={'runs':len(rows),'gold':14*3840,'old_metrics':old_checks};print(name,'verified',len(rows),'runs')
(H/'checks.json').write_text(json.dumps(all_info,indent=2)+'\n')
