#!/opt/anaconda3/bin/python3.12
"""Independently derive group retirement from actual Y/A/tau and reconcile RTL."""
from pathlib import Path
import csv,json
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'support_lut_execution_20260915'
STATES=['IDLE','PREQ','PRSP','TABLE','PN_INIT','PN_POS','PN_NEG','YLOAD','PINIT','YREAD','GROUP','SIGN_SUM','SIGN_NEG','PLANE_SUM','PREFIX','BOUND_POS','BOUND_NEG','LOWER','UPPER','OUTPUT','DONE']
d=np.load(OLD/'cases.npz',allow_pickle=False)
records=[json.loads(x) for x in (HERE/'results_all.jsonl').read_text().splitlines()]
assert len(records)==296
by_name={str(n):i for i,n in enumerate(d['case_name'])}
expected={}
a=d['A'].astype('int64');pos=np.maximum(a,0).sum(axis=1)[None,:,None];neg=np.minimum(a,0).sum(axis=1)[None,:,None]
for name in dict.fromkeys(x['case'] for x in records):
 i=by_name[name];y=d['S'][i].astype('int64')@d['W'][i].astype('int64');u=np.einsum('ts,psh->pth',a,y)
 groupmax=np.abs(y).reshape(32,10,12,8).max(axis=(1,3));e=np.zeros((32,12),dtype='int64')
 for b in range(24):e=np.where(groupmax>=1<<b,b+1,e)
 e=np.maximum(e,1);eh=np.repeat(e,8,axis=1)
 v=-np.einsum('ts,psh->pth',a,(y<0).astype('int64'));locked=np.broadcast_to(d['constant_channels'][i][None,None],u.shape).copy();gate=np.broadcast_to(d['constant_gate'][i][None],u.shape).copy().astype(bool)
 depth=np.zeros((32,12),dtype='int64');active_groups=np.ones((32,12),dtype=bool)
 for step in range(1,int(e.max())+1):
  m=np.maximum(eh-step,0);active_h=step<=eh
  dot=np.einsum('ts,psh->pth',a,(y>>m[:,None])&1)
  v=np.where(active_h[:,None],2*v+dot,v)
  lo=(v<<m[:,None])+neg*((1<<m[:,None])-1);hi=(v<<m[:,None])+pos*((1<<m[:,None])-1)
  low_hit=np.where(d['positive_gain'][i][None,None],lo>=d['tau'][i][None],lo>d['tau'][i][None])
  high_hit=np.where(d['positive_gain'][i][None,None],hi<d['tau'][i][None],hi<=d['tau'][i][None])
  now=(low_hit|high_hit)&(~locked)&active_h[:,None]
  gate=np.where(now,np.where(low_hit,d['positive_gain'][i][None,None],~d['positive_gain'][i][None,None]),gate)
  locked|=now
  all_locked=locked.reshape(32,10,12,8).all(axis=(1,3))
  retire=active_groups&all_locked
  depth[retire]=step;active_groups&=~retire
 assert np.array_equal(v,u) and np.array_equal(gate,d['gold'][i]) and not active_groups.any()
 expected[name]={'full_planes':int(e.sum()),'cert_planes':int(depth.sum()),'early_groups':int((depth<e).sum()),'full_plane_histogram':dict(zip(*[x.tolist() for x in np.unique(e,return_counts=True)])),'cert_depth_histogram':dict(zip(*[x.tolist() for x in np.unique(depth,return_counts=True)]))}
 for r in [x for x in records if x['case']==name]:
  ex=expected[name]; assert r['planes']==ex['cert_planes' if r['mode'] else 'full_planes']
  assert r['early_groups']==(ex['early_groups'] if r['mode'] else 0)
  assert sum(r['state_cycles'])==r['cycles']
  assert r['state_cycles'][13]==r['state_cycles'][14]==r['planes']
  for k in [8,9,10,11,12]:assert r['state_cycles'][k]=={8:32,9:320,10:384,11:384,12:384}[k]
  for k in [15,16,17,18]:assert r['state_cycles'][k]==(r['planes']-(384-r['early_groups']) if r['mode'] else 0)
  assert r['bound_checks']==80*(r['state_cycles'][17]+r['state_cycles'][18])
  assert r['y_rows_read']==320 and r['y_rows_written']==320
  assert r['gate_checks']==r['y_checks']==30720
  assert r['exponent_checks']==384 and r['u_checks']==(0 if r['mode'] else 30720)
agg=[]
for mode in [0,1]:
 for bp in [0,1]:
  for warm in [0,1]:
   rr=[x for x in records if x['real'] and x['mode']==mode and x['bp']==bp and x['warm']==warm]
   assert len(rr)==32
   sums={k:sum(x[k] for x in rr) for k in ['cycles','psn_service','param_words','y_rows_written','y_rows_read','table_rows_written','planes','early_groups']}
   st=np.array([x['state_cycles'] for x in rr]).sum(axis=0)
   assert sums['cycles']==int(st.sum())
   agg.append({'mode':mode,'bp':bp,'warm':warm,'commands':32,**sums,'cycles_plus_go':sums['cycles']+32,'state_cycles':dict(zip(STATES,map(int,st)))})
rtl=list(csv.DictReader((OLD/'rtl_cycles.csv').open()))
native=[]
for bp in [0,1]:
 rr=[x for x in rtl if x['case'].startswith('real_') and int(x['mode'])==2 and int(x['bp'])==bp]
 assert len(rr)==32
 native.append({'mode':'original_support_mode2_native96MAC','bp':bp,'commands':32,'complete_cycles':sum(int(x['cycles']) for x in rr),'psn_cycles':sum(int(x['psn_cycles']) for x in rr),'mac96_issues':sum(int(x['mac']) for x in rr),'fc_cycles':sum(int(x['fc_cycles']) for x in rr),'scope':'original complete FC1->PSN; candidate is post-Y only; native differs in arithmetic/table/holding resources'})
summary={'status':'PASS','scope':'Post-Y T10 full/cert subset RTL, same full/cert resources. Native is a different-resource reference; no FC1 production integrated here.',
 'commands':len(records),'real_commands':sum(x['real'] for x in records),'real_cases':32,'independent_sources':8,'diagnostic_cases':5,
 'coverage':{k:sum(x[k] for x in records) for k in ['gate_checks','u_checks','y_checks','bound_checks','exponent_checks']},
 'subset_table_values_checked':sum(x['table_rows_written'] for x in records)*10,
 'aggregate_real':agg,'native_reference':native,'expected_by_case':expected,
 'counter_semantics':{'cycles':'sum of states PREQ..DONE after accepted start, excludes go; cycles_plus_go adds one per command','psn_service':'PINIT..OUTPUT, including Y row reads and output ready stalls; excludes external Y transfer, parameter service and table construction','warm':'A and LUT/P/N retained from immediately preceding cold command; per-command thresholds/flags and complete Y still reloaded'},
 'resources':{'alu48':80,'multiplier':0,'table_payload_B':1280,'table_16bit_banks':20,'read_muxes_per_table_bank':8,'table_physical_data_copies':1,'Y24_persistent_B':92160,'Y24_T10_holding_B':2880,'Y24_old_single_row_holding_B':288,'A16_B':200,'tau48_B':5760,'flags_B':144,'prefix48_B':480,'dot48_B':480,'pos_neg_tail48_B':240,'gate_locked_lowerhit_lowergate_bits':320,'exponent_bits':60,'parameter_service_bits':128,'Y_row_service_bits':2304,'gate_group_bits':80},
 'limitations':['No synthesis/timing/power/area claim; register LUT has real eight-address mux cost per bank.','Y is CPU-recomputed input at an explicit post-Y boundary, held/read in RTL; FC1 last-writer handoff is not integrated.','No complete BitL import, no learned constant-MVM/CSE layout, no PSN producer fusion.','Fixed original32 real cases share8 projected-source tiles; signed/gain/constant/tie diagnostics are synthetic.','Only this H8/T10 layout is stopped if slower; no family-level impossibility inferred.']}
(HERE/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,separators=(',',':'))+'\n')
print(json.dumps({'status':'PASS','commands':len(records),'coverage':summary['coverage'],'real':agg,'native':native},ensure_ascii=False))
