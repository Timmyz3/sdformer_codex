from pathlib import Path
import json,argparse
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stages',nargs='+',default=['small','held','disjoint','sequences']);a=p.parse_args()
profiles=json.loads((H/'profiles.json').read_text());ad=json.loads((H/'admission.json').read_text());ids=ad['state_ids'];schema=json.loads((H/'counter_schema.json').read_text());checks=0;records=[]
def eq(a,b,label=''):
 global checks
 checks+=1;assert a==b,(label,a,b)
for stage in a.stages:
 for borrow in [0,1]:
  for stall in [0,1]:
   path=H/f'results_{stage}_b{borrow}_s{stall}.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
   eq(len(rows),8 if stage=='swap' else 2)
   for ri,r in enumerate(rows):
    names=Path(r['manifest']).read_text().splitlines();n=len(names);eq(n,r['tiles']);eq(r['borrow'],borrow);eq(r['stall'],stall)
    exp={k:sum(profiles[x][k] for x in names) for k in profiles[names[0]] if k not in ['base_cycles','base_states']}
    for k,v in exp.items():eq(r['core_'+k],v,(stage,borrow,stall,ri,k))
    for k in ['raw_values','J_values','wide_values','I24_values']:eq(r[k],3840*n,k)
    eq(r['Z_fields'],1280*n);eq(r['D_fields'],1280*n)
    eq(r['static_words'],1176 if ri==0 else 0);eq(r['source_load_words'],1536*n);eq(r['origin_words'],n);eq(r['output_beats'],480*n)
    batches=(n+1)//2;eq(r['batches'],batches);eq(r['launch_cycles'],batches)
    state=[sum(profiles[x]['base_states'][i] for x in names) for i in range(64)]
    denied=[sum(c[i] for c in r['context_denied']) for i in range(64)];observed=[sum(c[i] for c in r['context_states']) for i in range(64)]
    for i in range(64):eq(observed[i],state[i]+denied[i]+(r['core_output_stalls'] if i==ids['DRAIN_SEND'] else 0),(stage,borrow,stall,ri,'state',i))
    eq(sum(denied),r['core_source_stalls']+r['core_weight_stalls']+r['core_arbitration_stalls'])
    eq(sum(observed),r['core_cycles']);eq(r['core_cycles'],sum(state)+sum(denied)+r['core_output_stalls'])
    grant=[exp['source_words'],exp['q1_words']+exp['q2_words'],exp['z_vector_reads']+exp['z_scalar_reads']+exp['z_writes'],exp['psum_reads']+exp['psum_writes'],exp['q2_issues']+exp['transform_issues']+exp['reconstruction_issues']+exp['stripe_add_issues']+(0 if borrow else exp['q1_issues']),exp['q1_issues'] if borrow else 0]
    eq(r['observed_grants'],grant)
    for k,v in zip(['shared_source_grants','shared_weight_grants','shared_z_grants','shared_psum_grants','shared_alu_grants','borrow_grants'],grant):eq(r[k],v,k)
    eq(r['shared_wide_grants'],grant[5]+960*n)
    fixed=dict(raw_words=480,identity_words=480,coefficient_words=24,mul_issues=480,add_issues=960,round_issues=480,output_words=480,conversion_issues=480,conversion_saturations=0)
    for k,v in fixed.items():eq(r['consumer_'+k],v*n,k)
    eq(r['consumer_cycles'],3385*n+r['consumer_join_wait_cycles']+r['consumer_output_stalls']+r['consumer_wide_waits'])
    eq(r['window_cycles'],r['consumer_cycles']+n+n//2)
    ts=[0]*16;ts[1]=r['static_words']+r['parameter_stalls'];ts[2]=r['source_load_words']+r['source_load_stalls'];ts[3]=r['origin_words']+r['origin_stalls'];ts[4]=batches;ts[5]=r['window_cycles']-n//2;ts[6]=n//2;ts[7]=1
    eq(r['top_states'],ts);eq(r['total_cycles'],sum(ts));eq(r['go_cycles'],1)
    if 'request_hold_checks' in r:eq(r['request_hold_checks'],r['parameter_stalls']+r['source_load_stalls']+r['origin_stalls']+r['consumer_identity_stalls'])
    if not stall:
     for k in ['parameter_stalls','source_load_stalls','origin_stalls','core_source_stalls','core_weight_stalls','consumer_identity_stalls','consumer_output_stalls']:eq(r[k],0,k)
    if not borrow:eq(r['consumer_wide_waits'],0)
    row=dict(stage=Path(r['manifest']).stem,job=ri,borrow=borrow,stall=stall,repeat=r['repeat'],tiles=n,service_cycles=r['total_cycles']+1,total_cycles=r['total_cycles'],window_cycles=r['window_cycles'],configuration=r['static_words'],source_load=r['source_load_words']+r['source_load_stalls']+r['origin_words']+r['origin_stalls'],
      core_work_cycles=sum(state),core_cycles=r['core_cycles'],arbitration_stalls=r['core_arbitration_stalls'],output_stalls=r['core_output_stalls'],grants=dict(zip(['source','weight','z','psum','alu','borrow'],grant)),consumer_cycles=r['consumer_cycles'],consumer_wide_waits=r['consumer_wide_waits'])
    records.append(row)
report=dict(passed=True,jobs=len(records),tile_executions=sum(r['tiles'] for r in records),raw_J_wide_I24_each=sum(r['tiles']*3840 for r in records),Z_D_physical_fields_each=sum(r['tiles']*1280 for r in records),checks=checks,records=records,sequence_scope=ad['sequence_scope'],function=ad['function'])
(H/('verification_swap.json' if a.stages==['swap'] else 'SUMMARY.json')).write_text(json.dumps(report,separators=(',',':'))+'\n')
print(json.dumps(dict(passed=True,jobs=len(records),tile_executions=report['tile_executions'],checks=checks,service=[{k:r[k] for k in ['stage','borrow','stall','repeat','service_cycles']} for r in records]),separators=(',',':')))
