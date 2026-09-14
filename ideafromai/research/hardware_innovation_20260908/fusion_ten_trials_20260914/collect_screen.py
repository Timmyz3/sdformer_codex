"""Read completed measurements into one scope-labelled table; no new estimates."""
from pathlib import Path
import csv,json
H=Path(__file__).resolve().parent
rows=[]
def read(p):return json.loads((H/p).read_text())
def pair(path,mode,scope):
 a=[r for r in read(path) if r['mode']==mode and r['stall']==0 and r['command']==0]
 if scope=='real8':a=[r for r in a if r['fixture'].startswith('real_')]
 if not a:raise ValueError((path,mode,scope))
 keys=['cycles','total_cycles','configuration_cycles','outputs']
 return {k:sum(r.get(k,0) for r in a) for k in keys}
def quality(mode,split='valid'):
 if mode is None:return None
 candidates=[H/'algorithm_sparse'/f'quality_{split}_{mode}.json']
 if mode==6:candidates.insert(0,candidates[0].with_name(f'quality_{split}_{mode}_parallel.json'))
 available=[(p,json.loads(p.read_text())) for p in candidates if p.exists()]
 if not available:return {'status':'pending'}
 p,a=next(((p,a) for p,a in available if a.get('complete')),available[0])
 return {'status':'complete' if a.get('complete') else 'pending','AEE':a.get('result',{}).get('AEE_frame_mean'),'NB0':a.get('NB0_same_set'),'frames':len(a.get('frames',[])),'source':str(p.relative_to(H))}
def add(i,name,path,ctrl,cand,scope,label,key='total_cycles',q=None,qc=None):
 a,b=pair(path,ctrl,scope),pair(path,cand,scope)
 r=dict(id=i,mechanism=name,scope=label,control_mode=ctrl,candidate_mode=cand,
  control_cycles=a[key],candidate_cycles=b[key],cycle_change_pct=100*(b[key]/a[key]-1),
  control_cold_with_configuration=a[key]+a['configuration_cycles'],candidate_cold_with_configuration=b[key]+b['configuration_cycles'],
  source=path,quality=quality(q),control_quality=quality(qc))
 rows.append(r)
for i,n,f in [(1,'Q1位平面/bitmap','q1_bitplanes'),(2,'Q2分组DA','q2_da'),(3,'Q1重复完整列字典','q1_dictionary')]:
 add(i,n,f'decompositions/{f}/results.json',14,15,'real8','八个封存真实tile，线性核内，不含配置/consumer','cycles')
add(4,'完成latent整组剪枝','algorithm_sparse/results.json',2,1,'real8','八个当前真实tile，完整consumer，go后',q=1,qc=2)
add(5,'K4原型+单rank残差','algorithm_sparse/results.json',0,3,'real8','八个当前真实tile，完整consumer，go后')
rows[-1]['diverse10_quality']=quality(3,'diverse')
add(6,'整latent时间保持','algorithm_sparse/results.json',7,8,'real8','八个当前真实tile，完整consumer，同Δ权限',q=5,qc=6)
add(7,'取消p物化写','dataflow/d1_forward/results_64.json',1,2,'stream','64连续tile，冷配置到最终I24')
add(8,'横邻halo轮转','dataflow/d2_halo/results_full.json',0,1,'stream','19200tile单帧组件，冷配置到最终I24')
add(9,'双context阶段错位','dataflow/d3_interleave/results_full.json',2,1,'stream','19200tile单帧组件，对普通同时ready/RR')
add(10,'借用consumer宽链','phase_borrow/results_full.json',2,3,'stream','19200tile单帧组件，同416bit z口')
if (H/'temporal_direction/results.json').exists():
 add(11,'T10共同方向有限缩放','temporal_direction/results.json',1,2,'real8','八个当前真实tile，冷配置到I24，对full/prev/anchor Δ')
extra={}
for m in [0,1,2]:extra[f'dual_context_mode{m}']=pair('dataflow/d3_interleave/results_full.json',m,'stream')['total_cycles']
for m in [0,1,2,3,4,5,6,7,8]:extra[f'lossy_compute_mode{m}']=pair('algorithm_sparse/results.json',m,'real8')['total_cycles']
if (H/'joint_selected/results_full.json').exists():
 for m in [2,3]:extra[f'joint_same_halo_forward_mode{m}']=pair('joint_selected/results_full.json',m,'stream')['total_cycles']
records=0
for d in ['decompositions/q1_bitplanes','decompositions/q2_da','decompositions/q1_dictionary','algorithm_sparse','dataflow/d1_forward','dataflow/d2_halo','dataflow/d3_interleave','phase_borrow','temporal_direction']:
 for f in ['results.json','results_3.json','results_64.json','results_full.json']:
  p=H/d/f
  if p.exists():records+=len(json.loads(p.read_text()))
out=dict(initial_planned_mechanisms=10,distinct_mechanisms=len(rows),rtl_command_records=records,scope='Verilator screening only; scopes differ by row, do not pool absolute cycles or multiply ratios',quality_complete=all(r[k] is None or r[k]['status']=='complete' for r in rows for k in ['quality','control_quality']),rows=rows,additional_controls=extra)
(H/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
keys=['id','mechanism','scope','control_mode','candidate_mode','control_cycles','candidate_cycles','cycle_change_pct','control_cold_with_configuration','candidate_cold_with_configuration','source']
with (H/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
s=[f'# {len(rows)}项实际RTL筛选表','', '最初计划十项，追加精确T10方向一项；组合实验不重复计idea。正的周期变化表示变慢；每行使用自己的同工作负载强对照，绝对周期不得跨行拼接。','', '| # | 接口 | 强控周期 | 候选周期 | 周期变化 | 范围 |','|---|---|---:|---:|---:|---|']
for r in rows:s.append(f"| {r['id']} | {r['mechanism']} | {r['control_cycles']} | {r['candidate_cycles']} | {r['cycle_change_pct']:+.3f}% | {r['scope']} |")
s+=['','剪枝/时间两项表中分母是普通逐rank强控；它们相对完整exact的结果另在comparison.json与各自报告列出。所有比值只描述所测RTL时间线，没有同频同面积的ASIC结论。','']
(H/'PERFORMANCE_TABLE.md').write_text('\n'.join(s))
print(json.dumps({'items':len(rows),'commands':records,'quality_complete':out['quality_complete'],'extras':extra},ensure_ascii=False))
