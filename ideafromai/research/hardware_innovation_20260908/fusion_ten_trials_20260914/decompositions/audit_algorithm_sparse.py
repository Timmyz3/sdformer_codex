"""Independent integer semantics/encoding audit only; no source writes or RTL/GPU execution."""
from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;P=H.parent/'algorithm_sparse';params=json.loads((P/'parameters.json').read_text());rows=json.loads((P/'pre_zero_bypass/results.json').read_text())
def word(p):return np.array([int(v,16) for v in p.read_text().split()],dtype=np.int64)
def signed(x,b=32):
 x=x&((1<<b)-1);return np.where(x&(1<<(b-1)),x-(1<<b),x)
def unpack(a):return a.reshape(12,4,10,8).transpose(1,2,0,3).reshape(4,10,96)
q1=signed(word(P/'constants/q1.hex')).reshape(864,8).T;q2=signed(word(P/'constants/q2.hex')).reshape(12,8,8).transpose(0,2,1).reshape(96,8)
cb=np.array(params['codebook'],np.int64);shift=np.array(params['shifts']);assert np.all(cb[0]==0)
assert np.all(abs(q1)<=3) and q2.min()>=-32768 and q2.max()<=32767
pt=signed(word(P/'constants/prototypes.hex')).reshape(12,4,8).transpose(1,0,2).reshape(4,96);assert np.array_equal(pt,cb@q2.T)
c=signed(word(P/'constants/consumer.hex')).reshape(24,8);a=c[::2].reshape(96);b=c[1::2].reshape(96)
all_values=0;jvalues=0;alias_checks=0;delta_selected={7:0,8:0};delta_rejected={7:0,8:0};maxdiff=0
for case in sorted({r['fixture'] for r in rows}):
 f=P/'fixtures'/case;src=((word(f/'source.hex')[None,:]>>np.arange(10)[:,None])&1).reshape(10,96,4,4)
 oy,ox=signed(word(f/'origin.hex'),16);valid=(np.arange(4)[:,None]+oy>=0)&(np.arange(4)[:,None]+oy<240)&(np.arange(4)[None,:]+ox>=0)&(np.arange(4)[None,:]+ox<320);src*=valid
 patch=np.stack([src[:,:,p//2:p//2+3,p%2:p%2+3].reshape(10,864) for p in range(4)]);z=patch@q1.T
 identity=unpack(word(f/'identity.hex').astype('<u4').view('<f4').astype(np.float64));assert np.isfinite(identity).all()
 J=np.clip(np.rint(identity*2**20),-2**31,2**31-1).astype(np.int64);assert np.array_equal(J,unpack(signed(word(f/'j.hex'))));jvalues+=J.size
 modes={}
 for mode in range(9):
  fun={7:6,8:5}.get(mode,mode);zz=z.copy();proto_code=np.zeros((4,10),int);resrank=np.zeros((4,10),int);resval=np.zeros((4,10),np.int64)
  if fun==1:zz[((abs(z)<<shift).sum(-1))<=params['group_tau']*np.count_nonzero(z,axis=-1)]=0
  elif fun==2:zz[abs(z)<=np.array(params['rank_tau'])]=0
  elif fun in [3,4]:
   for p in range(4):
    for t in range(10):
     code=0 if fun==4 else int(np.argmin(((abs(cb-z[p,t]))<<shift).sum(-1)))
     dif=z[p,t]-cb[code];r=int(np.argmax(abs(dif)<<shift));zz[p,t]=cb[code];zz[p,t,r]=z[p,t,r];proto_code[p,t]=code;resrank[p,t]=r;resval[p,t]=dif[r]
  elif fun in [5,6]:
   for p in range(4):
    for t in range(1,10):
     diff=z[p,t]-zz[p,t-1]
     hold=((abs(diff)<<shift).sum()<=params['temporal_tau']) if fun==5 else abs(diff)<=np.array(params['temporal_rank_tau'])
     zz[p,t]=np.where(hold,zz[p,t-1],z[p,t])
  raw=zz@q2.T
  if fun in [3,4]:
   rr=pt[proto_code]+resval[...,None]*q2.T[resrank];assert np.array_equal(rr,raw)
  if mode in [7,8]:
   encoded=zz.copy();replay=np.zeros_like(raw)
   for p in range(4):
    acc=np.zeros(96,np.int64)
    for t in range(10):
     delta=False
     if t:
      diff=zz[p,t]-zz[p,t-1];maxdiff=max(maxdiff,int(abs(diff).max()));fits=np.all((diff>=-4096)&(diff<=4095))
      beneficial=np.count_nonzero(diff)<np.count_nonzero(zz[p,t]);delta_rejected[mode]+=int(beneficial and not fits)
      delta=fits and beneficial
      if delta:encoded[p,t]=diff;delta_selected[mode]+=1
     if not delta:acc[:]=0
     acc+=encoded[p,t]@q2.T;replay[p,t]=acc
   assert np.array_equal(replay,raw)
  assert np.array_equal(raw,unpack(signed(word(f/f'gold_{mode}.hex'))))
  wide=raw*a+((J+b)<<20);quo,rem=np.divmod(wide,2**26);I=np.clip(quo+((rem>2**25)|((rem==2**25)&((quo&1)!=0))),-2**23,2**23-1)
  assert np.array_equal(I,unpack(signed(word(f/f'i24_{mode}.hex'))));modes[mode]=(raw,I);all_values+=raw.size
 for lo,hi in [(6,7),(5,8)]:
  assert np.array_equal(modes[lo][0],modes[hi][0]) and np.array_equal(modes[lo][1],modes[hi][1]);alias_checks+=2
# Generic guard arithmetic only; overflow pairs below are outside the fixed-Q1 binary-source temporal contract.
# Each retained rank is an earlier real z_r, so |delta_r| <= sum_k |q1[r,k]| <= 2592 (or 3456 for full signed3).
guard=[]
for prev,new in [(-2592,2592),(2592,-2592),(-1503,2592),(-1504,2592),(1504,-2592),(1505,-2592)]:
 dif=new-prev;fits=-4096<=dif<=4095
 enc=dif if fits else new;decoded=prev+enc if fits else enc;assert decoded==new
 guard.append({'difference':dif,'select_delta':fits})
for r in rows:
 assert r['outputs']==3840 and r['configuration_cycles']==(3450 if r['command']==0 else 0)
 assert r['consumer_raw_words']==r['consumer_identity_words']==r['consumer_mul_issues']==r['consumer_round_issues']==r['consumer_output_words']==r['consumer_conversion_issues']==480
 assert r['consumer_add_issues']==960 and r['consumer_coefficient_words']==24
 assert r['total_cycles']==r['consumer_cycles']+1 and r['cycles']==sum(r['state_cycles'])
 assert r['encoder_cycles']=={0:0,1:320,2:160,3:1160,4:200,5:296,6:152,7:152,8:296}[r['mode']]
 assert r['prototype_reads']==0 or r['mode']==3
out={'scope':'Read-only independent integer transforms, prototype identity, exact J/I24 and delta replay; pre_zero_bypass/results.json counter snapshot; final zero bypass reviewed statically without rerun',
 'run_records_checked':len(rows),'raw_values_recomputed':all_values,'I24_values_recomputed':all_values,'J_values_recomputed':jvalues,
 'prototype_values_verified':pt.size,'same_function_gold_alias_checks':alias_checks,'delta_selected_fixture_vectors':delta_selected,'signed13_guard_rejections_in_fixtures':delta_rejected,'max_changed_delta_abs':maxdiff,
 'mathematical_guard_cases_not_RTL':guard,'common_Q2_mult_bits':[16,14],
 'static_temporal_delta_domain':{'retained_rank_is_prior_real_z':True,'bound_fixed_q1_minus3_to3':2592,'bound_full_signed3':3456,'signed13_overflow_reachable':False,'guard_boundary_samples_scope':'Generic arithmetic outside this fixed-Q1 binary-source temporal domain; not a missing legal RTL fixture'},
 'quality_scope':'No825 quality pass inferred. Read current quality receipts separately; 7 shares6 function,8 shares5.'}
(H/'ALGORITHM_SPARSE_AUDIT_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
