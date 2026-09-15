from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;R=H.parent/'r8_reference';N=H.parents[1]/'representation_transfer_20260914'
refs=[json.loads(x) for x in (R/'comparison.jsonl').read_text().splitlines()];ours=json.loads((H/'SUMMARY.json').read_text())['records']
readhex=lambda p:np.array([int(x,16) for x in p.read_text().split()],np.uint32)
source_values=identity_values=origin_values=0
for stage in ['held','disjoint','sequences']:
 a=(H/f'{stage}.txt').read_text().splitlines();b=(R/f'{stage}.txt').read_text().splitlines();assert len(a)==len(b)
 for x,y in zip(a,b):
  x,y=Path(x),Path(y)
  for left,right,dtype in [('source.hex','source.bin','<u2'),('origin.hex','origin.bin','<u4'),('identity.hex','identity_fp32.bin','<u4')]:
   av=readhex(x/left);bv=np.fromfile(y/right,dtype=dtype);assert np.array_equal(av,bv),(stage,x,y,left)
  source_values+=1536;origin_values+=2;identity_values+=3840
comparisons=[]
for s in ours:
 if s['stage']=='small':continue
 stage=s['stage'];stall=s['stall'];repeat=s['repeat']
 rrows=[x for x in refs if x['stage']==stage and x['stall']==stall and x['command']==repeat and x['initial_mode']==x['mode'] and x['next_mode'] is None]
 assert len(rrows)==4,(stage,stall,repeat,len(rrows))
 c=json.loads((H/f'results_{stage}_b{s["borrow"]}_s{stall}.jsonl').read_text().splitlines()[repeat]);r7=next(x for x in rrows if x['mode']==7)
 row=dict(stage=stage,borrow=s['borrow'],stall=stall,repeat=repeat,tiles=s['tiles'],spatial_service=s['service_cycles'],R8_services={str(x['mode']):x['service_cycles'] for x in rrows},
  slowdown_vs_R8_bitmap7=s['service_cycles']/r7['service_cycles']-1,
  spatial=dict(window=c['window_cycles'],static_words=c['static_words'],source_compute_words=c['core_source_words'],Q1_issues=c['core_q1_issues'],Q2_MAC=c['core_q2_issues'],W_words=c['shared_weight_grants'],Z_grants=c['shared_z_grants'],psum_grants=c['shared_psum_grants'],ALU_grants=c['shared_alu_grants'],borrow_grants=c['borrow_grants'],arbitration=c['core_arbitration_stalls'],raw_holding_wait=c['core_output_stalls'],consumer_wide_wait=c['consumer_wide_waits']),
  R8_bitmap7={k:r7[k] for k in ['window_cycles','static_words','core_first_issues','core_mac_issues','shared_weight_grants','shared_z_grants','shared_psum_grants','shared_alu_grants','core_arbitration_stalls','core_output_stalls']})
 comparisons.append(row)
quality=json.loads((N/'quality/unconstrained/deployed_valid/spatial_integer_summary.json').read_text())
report=dict(passed=True,same_inputs=dict(source_words=source_values,origin_scalar_words=origin_values,FP32_identity_words=identity_values,sets=['held64','disjoint64','18sequences36tiles']),function_comparison='different frozen functions on identical upstream native gate/FP32 identity input; own gold, quality reported independently',
 quality_source=str(N/'quality/unconstrained/deployed_valid/spatial_integer_summary.json'),R8_quality_source=str(R/'admission.json'),comparisons=comparisons)
(H/'comparison.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,same_inputs=report['same_inputs'],borrow_cold=[r for r in comparisons if r['borrow'] and r['repeat']==0 and not r['stall']]),separators=(',',':')))
