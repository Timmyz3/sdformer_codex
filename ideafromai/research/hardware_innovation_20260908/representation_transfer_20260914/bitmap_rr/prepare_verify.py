from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parents[1]
s=(B/'consumer_transfer_20260914/count_rr/verify.py').read_text()
s=s.replace('import ast,json','import ast,json,sys').replace('checks=0;cache={};raw_values=0;maxcount=0','if len(sys.argv)>1:rows=[r for r in rows if r["stage"] in sys.argv[1:]]\nchecks=0;cache={};raw_values=0;maxcount=0')
s=s.replace(" return modes\ndef ordered",''' # Rebuild native signed3 bitplanes independently of the RTL representation.
 kl=np.any(q!=0,axis=1);ev=e*kl[:,None,None]
 bm=ev.reshape(54,16,40).transpose(2,0,1)
 live=bm.sum(2)>0;single=bm.sum(2)==1
 qb=((q[None]>>np.arange(3)[:,None,None])&1).reshape(3,54,16,8)
 plive=qb.any((2,3));pc=np.einsum('pbk,lbkr->plbr',bm,qb)
 eq(bool(np.array_equal(pc[:,0].sum(1)+2*pc[:,1].sum(1)-4*pc[:,2].sum(1),ev.reshape(864,40).T@q)),True,'bitmap signed3 arithmetic')
 stripes=int(live.sum());one=int(single.sum());groups=int(live.any(0).sum())
 planes=int(((live&~single)[:,None,:]&plive).sum())
 prefetch=int((live.any(0)[None,:]&plive).sum())
 for mm in modes.values():
  for cc in ['bitmap_native_reads','bitmap_native_issues','cache_reads','cache_writes','bitmap_hold_writes','bitmap_hold_reads']:mm[cc]=0
 modes[8]=dict(common,base_cycles=5481+3*stripes+3*(stripes-one)+2*one+3*groups+n['M'],
  first_issues=planes+one,merged_updates=0,weight_words=prefetch+one+n['V'],repair_issues=0,repair_fields=0,normalization_issues=0,
  z_vector_reads=stripes+40,z_writes=stripes+10,aux_reads=stripes,aux_writes=864,aux_issues=planes,aux_weight_words=prefetch,aux_events=0,
  metadata_reads=0,count_checks=0,count_bank_reads=0,count_bank_writes=0,
  bitmap_native_reads=one,bitmap_native_issues=one,cache_reads=planes,cache_writes=3*groups,bitmap_hold_writes=0,bitmap_hold_reads=0)
 row_count=int(live.reshape(4,10,54).any(0).sum())
 modes[7]=dict(modes[8],base_cycles=modes[8]['base_cycles']-(stripes-row_count),z_vector_reads=row_count+40,z_writes=row_count+10,bitmap_hold_writes=stripes-row_count,bitmap_hold_reads=stripes)
 return modes
def ordered''')
s=s.replace("+pred['normalization_issues']+pred['aux_issues'],'ALU grants')","+pred['normalization_issues']+(pred['aux_issues'] if mode in [20,21] else 0),'ALU grants')")
s=s.replace("960*n+pred['aux_reads']+pred['aux_writes'],'psum grants')","960*n+(pred['aux_reads']+pred['aux_writes'] if mode in [20,21] else 0),'psum grants')")
s=s.replace("for stage in ['small','short','held','disjoint']:","for stage in ['small','short','held','disjoint','swap']:",1)
start=s.index('# Read old receipts only:')
s=s[:start]+'''# Reproduce all existing same-structure strong controls without rewriting old receipts.
matched=0;index={(r['fixture'],r['first_tile'],r['tiles'],r['mode'],r['stall'],r['command'],r['stage']):r for r in rows if not r['cross_mode']}
for stage in ['small','short','held','disjoint']:
 for r in [json.loads(l) for l in (B/'consumer_transfer_20260914/count_rr'/f'results_{stage}.jsonl').read_text().splitlines()]:
  key=(r['fixture'],r['first_tile'],r['tiles'],r['mode'],r['stall'],r['command'],stage)
  if r['cross_mode'] or key not in index:continue
  for k,v in r.items():
   if k!='wall_seconds_so_far':eq(index[key][k],v,('old control',key,k))
  matched+=1
initial_reproduced=0
for before in json.loads((H/'initial_mode8_summary.json').read_text()):
 matches=[r for r in rows if r['mode']==8 and all(r[k]==before[k] for k in ['stage','first_tile','tiles','stall','command'])]
 if not matches:continue
 after=matches[0]
 for k,v in before.items():eq(after[k],v,('initial mode8',k))
 initial_reproduced+=1
result=dict(passed=True,checks=checks,commands=len(rows),rtl_values_each_stage=sum(r['outputs'] for r in rows),independent_tiles=len(cache),independent_raw_J_I24_each=raw_values,max_observed_count=maxcount,old_control_records_equal=matched,stages=sorted(set(r['stage'] for r in rows)),class_table_copies=1,representative_table_copies=1,producer_alu=8,producer_multipliers=8,contexts=2,z_port_bits=416,per_context_z_bytes=520,per_context_psum_bytes=15360,class_bytes=2592,representative_bytes=96,permutation_bits=40,plane_coefficient_copies=1,plane_coefficient_bytes=2592,plane_live_bits=162,pop16_trees=8,per_context_bitmap_bytes=80,per_context_bitmap_live_bits=40,per_context_qblock_overlay_bytes=48)
result['initial_mode8_records_reproduced']=initial_reproduced
(H/'verification.json').write_text(json.dumps(result,indent=2)+'\\n')
(H/'profiles.jsonl').write_text(''.join(json.dumps(dict(fixture=k,modes=v),separators=(',',':'))+'\\n' for k,v in cache.items()))
print(json.dumps(result),flush=True)
'''
(H/'verify.py').write_text(s)
