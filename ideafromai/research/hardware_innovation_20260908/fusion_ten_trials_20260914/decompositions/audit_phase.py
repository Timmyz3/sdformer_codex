"""Independent read-only audit; does not invoke RTL, training, or author's verify script."""
from pathlib import Path
import json,itertools,numpy as np,re
H=Path(__file__).resolve().parent;P=H.parent/'phase_borrow';OLD=H.parents[1]/'r8_consumer_fusion_20260914/consumer_packed'
# Direct signed13 field carry isolation, independent of neural workload.
carry_checks=0
for vals in itertools.product([-2592,-1,0,1,2592],repeat=4):
 for q in [-3,-1,0,1,3]:
  for active in range(16):
   lhs=sum((a&8191)<<(13*i) for i,a in enumerate(vals));rhs=sum(((q if active>>i&1 else 0)&8191)<<(13*i) for i in range(4))
   out=0;cin=0
   for bit in range(64):
    if bit in [0,13,26,39]:cin=0
    a=lhs>>bit&1;b=rhs>>bit&1;out|=(a^b^cin)<<bit;cin=(a&b)|((a^b)&cin)
   for i,a in enumerate(vals):
    z=out>>(13*i)&8191;z=z-8192 if z&4096 else z;assert z==a+(q if active>>i&1 else 0)
   carry_checks+=1
# FP conversion and exact64 sum/RNE/sat against independently supplied fixture gold.
rows=json.loads((P/'results.json').read_text());fixture_checks=0;finite_words=0
for n in sorted({r['fixture'] for r in rows}):
 f=P/'fixtures'/n
 raw=np.fromfile(f/'raw.bin',dtype='<i4').astype(np.int64).reshape(12,40,8)
 identity=np.fromfile(f/'identity_fp32.bin',dtype='<f4').astype(np.float64).reshape(12,40,8)
 assert np.all(np.isfinite(identity));finite_words+=identity.size
 j=np.clip(np.rint(identity*2**20),-2**31,2**31-1).astype(np.int64)
 jgold=np.fromfile(f/'identity.bin',dtype='<i4').astype(np.int64).reshape(12,40,8);assert np.array_equal(j,jgold)
 c=np.fromfile(f/'param7.bin',dtype='<i4').astype(np.int64).reshape(24,8)
 wide=raw*c[::2,None,:]+((c[1::2,None,:]+j)<<20)
 quot,rem=np.divmod(wide,2**26);val=np.clip(quot+((rem>2**25)|((rem==2**25)&((quot&1)!=0))),-2**23,2**23-1)
 gold=np.fromfile(f/'gold.bin',dtype='<i4').astype(np.int64).reshape(12,40,8);assert np.array_equal(val,gold);fixture_checks+=raw.size
# Shared explicit main adder replaces former expression; conversion/RNE functions unchanged.
new=(P/'i24_consumer.sv').read_text();old=(OLD/'i24_consumer.sv').read_text()
assert 'add_result[l]=wide_hold[l]+add_rhs[l]' not in new
for name in ['fp32_q20','rne26']:
 pat=r'function automatic logic signed \[.*?\] '+name+r'\(.*?endfunction'
 assert re.search(pat,new,re.S).group(0)==re.search(pat,old,re.S).group(0)
assert (P/'consumer_stream.sv').read_text().count('wide_phase_alu wide_alu(')==1
# No-stall same-function mode2 vs prior packed15: only10 fewer clear cycles/tile.
matched=0
for fn in ['results.json','results_64.json','results_full.json']:
 a=json.loads((P/fn).read_text());b=json.loads((OLD/fn).read_text())
 for r in a:
  if r['mode']!=2 or r['stall']:continue
  keys=['command']+(['fixture'] if 'fixture' in r else ['first_tile','tiles'])
  rr=next(x for x in b if x['mode']==15 and not x['stall'] and all(x[k]==r[k] for k in keys))
  t=r['retired_tiles']
  for key in ['total_cycles','core_cycles','consumer_cycles','core_z_writes']:
   assert r[key]==rr[key]-10*t,(fn,key,r[key],rr[key]);matched+=1
full={r['mode']:r for r in json.loads((P/'results_full.json').read_text())}
assert full[2]['total_cycles']-full[3]['total_cycles']==3*(full[2]['core_first_issues']-full[3]['core_first_issues'])
for mode,r in full.items():
 assert r['outputs']==r['raw_outputs']==r['J_outputs']==73728000
 assert r['core_borrow_waits']==r['consumer_wide_waits']==0
out={'scope':'Independent finite math and read-only source/results audit; no RTL reexecution','carry_boundary_vectors':carry_checks,'fp32_J_values_checked':finite_words,'I24_values_checked':fixture_checks,
 'old_mode15_to_new_mode2_clear_adjusted_checks':matched,'FP_conversion_and_RNE_functions_byte_identical':True,
 'shared_wide_main_adder_instances':1,'full_cycles':{m:r['total_cycles'] for m,r in full.items()},
 'stalled_small_core_borrow_wait_sum':sum(r['core_borrow_waits'] for r in rows if r['stall']),
 'stalled_small_consumer_wide_wait_sum':sum(r['consumer_wide_waits'] for r in rows if r['stall']),
 'full_percent_savings':100*(full[2]['total_cycles']-full[3]['total_cycles'])/full[2]['total_cycles']}
(H/'PHASE_AUDIT_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
