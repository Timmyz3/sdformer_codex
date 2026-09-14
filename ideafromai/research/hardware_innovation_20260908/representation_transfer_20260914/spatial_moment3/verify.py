from pathlib import Path
import argparse,json
H=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--stages',nargs='+',default=['small','held','disjoint','sequences']);ap.add_argument('--input',type=Path);o=ap.parse_args();H=o.input.resolve() if o.input else H
p=json.loads((H/'profiles.json').read_text());ad=json.loads((H/'admission.json').read_text());ids=ad['state_ids'];out={};checks=0
def eq(a,b,label=''):
 global checks
 checks+=1;assert a==b,(label,a,b)
total=0
for consumer in [False,True]:
 prefix='consumer' if consumer else 'raw';sets={};commands=0
 for stage in o.stages:
  names=(H/f'{stage}.txt').read_text().splitlines();summaries=[]
  for stall in [0,1]:
   rows=[json.loads(v) for v in (H/f'{prefix}_{stage}_s{stall}.jsonl').read_text().splitlines()];eq(len(rows),2*len(names));commands+=len(rows)
   for i,r in enumerate(rows):
    eq(r['fixture'],names[i%len(names)]);eq(r['mode'],1);eq(r['stall'],stall);eq(r['command'],i)
    e=p[r['fixture']]
    for key,value in e.items():
     if key not in ['base_cycles','base_states']:eq(r[key],value,(stage,consumer,i,key))
    eq(r['configuration_cycles'],1537+((1176 if consumer else 1152) if i==0 else 0))
    eq(r['z_values'],1280);eq(r['d_values'],1280) # 960 real D fields +320 explicitly zero padding fields
    if consumer:
     for key in ['raw_values','j_values','wide_values','i24_values']:eq(r[key],3840)
    else:eq(r['outputs'],3840)
    eq(r['cycles'],e['base_cycles']+r['source_stalls']+r['weight_stalls']+r['output_stalls'])
    state=e['base_states'].copy();state[ids['L_LOAD']]+=r['source_stalls'];state[ids['DRAIN_SEND']]+=r['output_stalls']
    for j,value in enumerate(state):
     if j==0 and consumer:eq(r['state_cycles'][j]>=6,True)
     elif j not in [ids['QREAD'],ids['VLOAD']]:eq(r['state_cycles'][j],value,(stage,consumer,i,'state',j))
    eq(r['state_cycles'][ids['QREAD']]+r['state_cycles'][ids['VLOAD']],state[ids['QREAD']]+state[ids['VLOAD']]+r['weight_stalls'])
    if consumer:
     eq(sum(r['state_cycles']),r['c_cycles']);eq(sum(r['state_cycles'])-r['state_cycles'][0],r['cycles'])
     fixed=dict(c_raw_words=480,c_identity_words=480,c_coefficient_words=24,c_mul_issues=480,c_add_issues=960,c_round_issues=480,c_output_words=480,c_conversion_issues=480,c_conversion_saturations=0,c_wide_waits=0)
     for key,value in fixed.items():eq(r[key],value,(stage,i,key))
     eq(r['c_cycles'],3385+r['c_join_wait_cycles']+r['c_output_stalls'])
    if not stall:
     eq(r['source_stalls']+r['weight_stalls'],0)
     if consumer:eq(r['c_cycles'],e['base_cycles']+2425);eq(r['c_identity_stalls']+r['c_output_stalls'],0)
     else:eq(r['output_stalls'],0)
    if i>=len(names):
     for key,value in r.items():
      if key not in ['command','configuration_cycles']:eq(value,rows[i-len(names)][key],('repeat',key))
   for repeat in [0,1]:
    group=rows[repeat*len(names):(repeat+1)*len(names)]
    summary=dict(stall=stall,repeat=repeat,tiles=len(group),core=sum(r['cycles'] for r in group),service=sum(r['c_cycles' if consumer else 'cycles']+r['configuration_cycles']+1 for r in group),configuration=sum(r['configuration_cycles'] for r in group),start_cycles=len(group),
      counters={key:sum(r[key] for r in group) for key in e if key not in ['base_cycles','base_states']})
    if consumer:summary['consumer_cycles']=sum(r['c_cycles'] for r in group)
    summaries.append(summary)
   sets[stage]=dict(commands=len(names)*4,summaries=summaries)
 out[prefix]=dict(passed=True,commands=commands,raw_values=commands*3840,real_D_values=commands*960,D_padded_fields=commands*1280,sets=sets)
 if consumer:out[prefix].update(J_values=commands*3840,wide_values=commands*3840,I24_values=commands*3840)
 total+=commands
report=dict(passed=True,function=ad['function'],commands=total,checks=checks,stages=o.stages,results=out,
 resource=dict(ALU32=8,producer_mult19x13=8,q1_bytes=4608,q2_bytes=7488,q2_cache_bytes=312,Z_bytes=1280,psum_bytes=15360,source_bytes=1920,native_window_bytes=20,M_total_bytes=96,M_extra_vs_native_acc_bytes=64,transform_tail_bytes=32,z_hold_bytes=32,q1_hold_bytes=8,position_support_bytes=80,q1_live_bytes=72,q2_live_bytes=72,block_and_remaining_bits=48,
  source_port_bits=10,weight_port_bits=256,Z_vector_port_bits=256,Z_scalar_port_bits=32,psum_port_bits=256,consumer_contexts=1,consumer_mult32x32=8,consumer_alu64=8,original_g_and_transformed_coeff_copies=False),
 numerical_scope='raw/Z/D/J/wide/I24 actual RTL; M prefix assertions and independent CPU arithmetic, no per-M RTL monitor',
 sequence_scope=ad['sequence_scope'],novelty='ordinary constrained FIR factorization/fast convolution; no new formula claim')
(H/'SUMMARY.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,commands=total,checks=checks,summary={prefix:{stage:[{k:s[k] for k in ['stall','repeat','service']} for s in row['summaries']] for stage,row in value['sets'].items()} for prefix,value in out.items()}),separators=(',',':')))
