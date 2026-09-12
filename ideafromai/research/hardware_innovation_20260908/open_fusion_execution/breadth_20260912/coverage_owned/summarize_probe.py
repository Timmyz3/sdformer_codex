from pathlib import Path
from collections import defaultdict
import csv
import json
import statistics

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[1]
ROOT=OPEN.parent
r=json.loads((HERE/'shared_index_results.json').read_text())['rows']
groups=defaultdict(list)
for x in r:
    if not x['real'] or x['temporal'] or x['stress']:continue
    group='trained_2of4' if 'row_2of4' in x['name'] else 'trained_C16' if 'C16' in x['name'] else 'original_integer'
    groups[group,x['mode']].append(x)
rows=[]
for (g,m),v in groups.items():
    rows.append(dict(group=g,mode=m,actual_capture_count=len(v),
        source_plus_W_frontend_slots_mean=statistics.mean(x['source_plus_W_frontend_slots'] for x in v),
        W8_reads_mean=statistics.mean(x['counts'].get('physical_W8_reads',0) for x in v),
        NRV32_reads_mean=statistics.mean(x['counts'].get('physical_NRV32_reads',0) for x in v),
        required_S_additions_mean=statistics.mean(x['counts'].get('required_S_additions_unpriced',0) for x in v)))
checks=0
old=list(csv.DictReader((ROOT/'psn/rtl/gp_slice/intersection_results.tsv').open(),delimiter='\t'))
for x in r:
    if not x['real'] or x['mode'] not in ('dense_private','index_private'):continue
    y=next(y for y in old if y['case']==x['name'] and int(y['intersection'])==int(x['mode'].startswith('index')) and int(y['reduce'])==0 and int(y['stress'])==int(x['stress']))
    assert x['counts']['physical_W8_reads']==int(y['w_reads']) and x['source_SR64_reads_equivalent']==int(y['source_reads'])
    checks+=1
ratios=[]
for g in sorted({x['group'] for x in rows}):
    a={x['mode']:x for x in rows if x['group']==g}
    base=min((a[m] for m in a if m.startswith('dense')),key=lambda x:x['source_plus_W_frontend_slots_mean'])
    best=min((a[m] for m in a if m.startswith('index')),key=lambda x:x['source_plus_W_frontend_slots_mean'])
    ratios.append(dict(group=g,strong_dense=base['mode'],strong_index=best['mode'],
        best_index_over_best_dense=best['source_plus_W_frontend_slots_mean']/base['source_plus_W_frontend_slots_mean'],
        dense_merge_service_reduction=1-a['dense_merge']['source_plus_W_frontend_slots_mean']/a['dense_private']['source_plus_W_frontend_slots_mean'],
        index_merge_service_reduction=1-a['index_merge']['source_plus_W_frontend_slots_mean']/a['index_private']['source_plus_W_frontend_slots_mean']))
out=dict(rows=rows,comparisons=ratios,total_cases=len(r),actual_student_cases=sum(x['real'] for x in r),
    directed_cases=sum(not x['real'] for x in r),gate_bits_checked=sum(x['gate_bits'] for x in r),
    S_values_checked=sum(x['S_values'] for x in r),differences=sum(x['differences'] for x in r),
    old_scalar_RTL_private_W8_and_source64_equivalent_counts_matched=checks,
    conclusion='With actual NRV response service, merging reduces W traffic but does not improve dense frontend completion. Index merging helps private index, yet the best indexed variant still loses to the best dense variant for all3 actual-W groups. No whole-chain/RTL/PPA speedup or new AEE.')
(HERE/'shared_index_summary.json').write_text(json.dumps(out,indent=2)+'\n')
with (HERE/'shared_index_summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))

# Supplement source candidates that are outside the305 view schema.
supp=json.loads((OPEN/'stage_20260912/literature/literature_supplement.json').read_text())
families=['F23','F23','F23','F08','F11','F01','F17','F08','F23','F20','F12','F08','F13','F09']
extra=[]
for item,family in zip(supp,families):
    extra.append(dict(ref='SUPP-'+item['id'],name=item['name'],family=family,
        provenance='stage_20260912/literature/literature_supplement.json',
        status='缺指定原作正文/工件；机制未完整迁入' if item['id'] in ('L07','L08','L09') else '见家族已试接口；未完整迁原作',
        boundary=item['complete_original_implementation_missing']))
for ref,name,family,boundary in [
    ('GROK-G1','typed last-use','F11','真实最后读取对照已做；类型名称未带来净服务'),
    ('GROK-G2','causal spatial wake','F13','不能因不是G1或已有空间先验就判整个接口不适配'),
    ('GROK-G3','global dynamic BN','F10','完整外部gate/PED起点native→实算BN→join及两块目录复用已测；真实I24前段整层/新参数整链未闭'),
    ('GROK-G4','train binary support density','F04','旧停标题不是禁止同预算训练；使用最新NB0门'),
    ('GROK-H2','8088 wait class','F11','已定位原source压力波形，不是PED最后读取证据'),
    ('QUEUE-FLIVE','F_live>1','F02','固定同56字S/双NR4的F2已实跑352次，当前布局未胜最强F1；其余推进布局/完整阵列与16+80交织未试'),
    ('QUEUE-PED','LoD/whitening/PED representation','F07','两学生×五表示十帧含affine/diag已测；CR/RF两种5+5解码已收费；完整消费者内压缩残差/step/门就绪仍开放')]:
    extra.append(dict(ref=ref,name=name,family=family,provenance='EXECUTION_QUEUE.md / Grok原报告',status='见家族状态',boundary=boundary))
with (HERE/'supplemental_interfaces.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(extra[0]));w.writeheader();w.writerows(extra)
