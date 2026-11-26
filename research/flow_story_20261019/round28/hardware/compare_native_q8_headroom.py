#!/usr/bin/env python3
"""Derive same-machine headroom against both ordinary controls; no model rerun."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'round27/hardware'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    original=OLD/'UNIFIED_SERVICE_V1.json';q4file=OLD/'Q4_COST_ADDENDUM_V1.json'
    assert sha(original)=='f1aca17e9a6f7e3afc6f57260fce313ae8197d61eedb5c5c9c0d65603ac6ad14'
    assert sha(q4file)=='42e15c7ab3327a147b3522fd1d16b80da62989c7aa0481553814113cd2e4f21f'
    d=read(original);v=next(x for x in read(q4file)['variants'] if x['variant']=='primary');assert d['machine']==v['machine']
    selected={x['implementation']:x['summary'] for x in d['results'] if x['scope']=='MLP12_patch4'}
    q4=v['Q4_total'];metrics=['PE_arithmetic_ticks','PE_reserved_ticks','ticks'];comparisons={}
    for candidate,cost in [('Q4',q4),('Q6',selected['Q6_serial'])]:
        comparisons[candidate]={}
        for metric in metrics:
            rows={}
            for name in ('native','Q8_strong'):
                baseline=selected[name][metric];value=cost[metric]
                rows[name]=dict(baseline=baseline,candidate=value,reduction_fraction=1-value/baseline,
                               remaining_after_10_percent_gate=(9*baseline-10*value)/10)
            tight=min(rows,key=lambda k:rows[k]['remaining_after_10_percent_gate'])
            comparisons[candidate][metric]=dict(controls=rows,tighter_control=tight,
                joint_remaining_after_10_percent_gate=rows[tight]['remaining_after_10_percent_gate'])
    out=dict(schema='flow_r28_same_machine_native_and_Q8_headroom_v1',status='DERIVED_CONDITIONAL_COST_ONLY',
        sources={str(original):sha(original),str(q4file):sha(q4file),str(Path(__file__)):sha(Path(__file__))},
        scope='all12 MLP plus fixed4 patch residual convolutions; complete9 low precision, true tails Q8',
        machine=d['machine'],comparisons=comparisons,
        future_budget_rule='additional_restore_to_Q8_and_score_and_select_and_switch_PE <= min(0.9*C_native-C_Q4, 0.9*C_Q8strong-C_Q4)',
        limitations=['Same fixed uncalibrated resource model; no parameter selection, RTL/DC, or new simulation.',
          'PE arithmetic, PE reserved, and summed operator service are distinct metrics; service sum is not an executed graph-wide causal trace.',
          'Positive unused budget is a necessary cost allowance, not measured net dynamic-policy savings or quality qualification.',
          'No native sensitivity reruns were available in the frozen source; this derivation reports the primary fixed machine only.',
          'Q4 encoding is the fixed nested signed drop4 action, not a claim that generic low-bit quantization alternatives are fully optimized.'],
        GPU_used=False,model_forwards=0,quality_values_read=False,R27_frozen_files_modified=False,R28_launch_conditions_changed=False)
    output=HERE/'NATIVE_Q8_HEADROOM_V1.json';assert not output.exists();output.write_text(json.dumps(out,indent=2)+'\n')
    lines=['# Q4 同机普通对照与剩余成本空间','',
        '只从已冻结 R27 模型与 Q4 成本附表计算，未改变机器参数、执行图、恢复合同或 R28 启动条件。主比较同时面对 native FP32 spike gate-add 与较强 Q8；不能仅选择较慢对照。这里的 tick 是明确假设下的成本模型量，不是已综合芯片周期。','',
        '| 指标 | Q4 候选 | native 对照 | 相对 native 降幅 | 对 native 过 10% 后余量 | 较强 Q8 对照 | 相对 Q8 降幅 | 对 Q8 过 10% 后余量 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    labels={'PE_arithmetic_ticks':'PE 算术 tick','PE_reserved_ticks':'PE 占用 tick','ticks':'算子服务相加 tick'}
    for metric in metrics:
        c=comparisons['Q4'][metric]['controls'];n,q=c['native'],c['Q8_strong']
        lines.append(f"| {labels[metric]} | {n['candidate']:,} | {n['baseline']:,} | {100*n['reduction_fraction']:.6f}% | {n['remaining_after_10_percent_gate']:,.1f} | {q['baseline']:,} | {100*q['reduction_fraction']:.6f}% | {q['remaining_after_10_percent_gate']:,.1f} |")
    b=comparisons['Q4']['PE_arithmetic_ticks']['joint_remaining_after_10_percent_gate']
    lines += ['',f'较紧的普通对照是 native。以 PE 算术账为例，未来恢复部分窗口到 Q8 的增量、预测器、选择、模式切换与额外搬运占用相关 PE 的费用合计，必须不超过 **{b:,.1f} tick**，才能在这个固定模型中同时满足两个对照的净 10% 条件。这仍不构成已经通过：真实动态算法尚未计入、精度必须另测、其他存储/服务瓶颈也要检查。', '',
       'Q6 对两个对照的条件余量均为负；不能通过选择 Q8 分母或加入非负成本的窗口恢复和评分来补救。Q4 仍有正的必要空间，但相对仅 Q8 比较的空间更小。本表只使用现有主机器的 native 数据，没有补做或挑选敏感性参数；不改变本轮已经确定的实验。','']
    md=HERE/'NATIVE_Q8_HEADROOM_V1.md';assert not md.exists();md.write_text('\n'.join(lines))
    freeze=dict(schema='flow_r28_native_Q8_headroom_freeze_v1',files={p.name:sha(p) for p in (Path(__file__),output,md)},quality_read=False)
    f=HERE/'NATIVE_Q8_HEADROOM_FREEZE_V1.json';assert not f.exists();f.write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(dict(json_sha256=sha(output),freeze_sha256=sha(f),Q4_primary=comparisons['Q4']['PE_arithmetic_ticks']),indent=2))
if __name__=='__main__':main()
