"""Summarize actual fixed full-window executions; no experiments here."""
from pathlib import Path
import csv,json
HERE=Path(__file__).resolve().parent
AXES=['dense_twopot','lifting_twopot','contiguous34']
LABELS=['dense 两项','lifting40 两项','普通分组34']
AEE=[1.209053834287375,1.235243344790693,1.4214610958585636]
def read(axis,mode,stress=False):
    return json.loads((HERE/(axis+'_'+mode+('_stress' if stress else '')+'.json')).read_text())
def bytes_used(r):
    c=r['counts']
    return dict(SR=8*c['SR64_reads'],SW=8*c['SW64_writes'],CR=32*c['CR256_reads'],CW=32*c['CW256_writes'],
        external_input=c['DMA_input_slots']//5*32,
        external_PED=c['PED_DMA_output_slots']//5*32,
        external_projection=c['A_projection_egress_DMA_slots']//5*32)
rows=[];cases=[]
for axis,label,aee in zip(AXES,LABELS,AEE):
    for stress in (False,True):
        baseline=read(axis,'direct_CSE_P1_Z',stress);candidate=read(axis,'batch_joint_CSE',stress)
        row=dict(axis=axis,function=label,stress=stress,valid825_AEE=aee,
            baseline_slots=baseline['service_slots'],candidate_slots=candidate['service_slots'],
            saved_slots=baseline['service_slots']-candidate['service_slots'],
            net_reduction=1-candidate['service_slots']/baseline['service_slots'],
            baseline_bytes=bytes_used(baseline),candidate_bytes=bytes_used(candidate))
        if not stress:
            serial=read(axis,'batch_serial_CSE')
            row.update(batch_serial_slots=serial['service_slots'],
                same_layout_overlap_saved=serial['service_slots']-candidate['service_slots'],
                batching_tax_vs_best_direct=serial['service_slots']-baseline['service_slots'])
        rows.append(row)
        modes=['direct_CSE_P1_Z','batch_joint_CSE'] if stress else ['direct_CSE_P2','direct_CSE_P1_Z','batch_serial_CSE','batch_joint_CSE']
        for mode in modes:
            r=read(axis,mode,stress)
            assert all(v==0 for k,v in r['checks'].items() if k.endswith('_differences'))
            cases.append(dict(axis=axis,mode=mode,stress=stress,service_slots=r['service_slots'],checks=r['checks'],
                bytes=bytes_used(r),source_work_RF=r['source_work_RF'],PED_hblock=r['PED_hblock'],
                ROM_words=r['combined_source_ROM_words'],state_high_water=r['state_high_water']))
result=dict(scope='Complete A local4x4 PED/8x8 gates and B complete11x11 source; not two-window inference/full layer/network.',
    evidence='Actual-payload CPU shared-ISA service slots, not RTL cycles or PPA.',
    cases=cases,comparisons=rows,
    fixed_resources=dict(RF=[96,8,48],state_bytes=131072,coefficient_bytes=131072,source_ROM_words=512,
        SR_bits=64,SW_bits=64,CR_bits=256,issue=1,DMA='32 B / 5 slots, one shared staging buffer'),
    inference_identity='AT-LIF {0,theta}; static theta absorbable into W; continuous I24/PSN/PED separately executed.',
    conclusion='Positive limited service, but ordinary34 also benefits and is faster. This fixed batch/round-robin layout does not establish a novel title mechanism. Other producer-consumer interfaces remain untested.')
(HERE/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (HERE/'comparison.csv').open('w') as f:
    fields=['axis','stress','valid825_AEE','baseline_slots','candidate_slots','saved_slots','net_reduction']
    writer=csv.DictWriter(f,fields);writer.writeheader()
    writer.writerows({k:r[k] for k in fields} for r in rows)
lines=['# 完整 A 消费者 × B 源：真实同资源交错结果','',
'**固定布局有正收益，但还不能作为新标题。** 12 个 ready、6 个固定背压执行均完成，完整边界检查零差。普通源获得完整 CSE、立即 PED、P1 留 Z 的权限；候选的 sn2 迁移与 46,080 B anchor 保存真实计费。','',
'| 函数 | fresh825 AEE | ready 最强普通 | ready 交错 | ready 减少 | 固定背压减少 |',
'|---|---:|---:|---:|---:|---:|']
for i,(axis,label,aee) in enumerate(zip(AXES,LABELS,AEE)):
    a,b=rows[2*i:2*i+2]
    lines.append(f"| {label} | {aee:.9f} | {a['baseline_slots']:,} | {a['candidate_slots']:,} | {100*a['net_reduction']:.3f}% | {100*b['net_reduction']:.3f}% |")
lines += ['', 'AEE来自每个函数自己的完整825；没有新训练。质量见[两项函数](../../../algorithm/source_constant_valid825/README.md)与[普通34](../../../algorithm/README.md)。这里的服务是CPU载荷模型槽数，不能与源RTL周期、native后段或整网指标相乘。', '',
'## 工作边界与真实终点','',
'A 使用 corner 的全部 9×9 源输入、K864 preview/sn2、8×8 updated/投影门和4×4 PED；B 使用同帧 interior 的全部11×11源。每项实测检查61,440个updated值、61,440个投影门、15,360个PED值、116,160个B源门；实际外送12,288 B投影门和46,080 B连续PED逐字节核对。B后续preview、native投影、全域BN/join及上游I24生产不在范围内。', '',
'完整源与连续消费者共享96×8×48RF、SR64/SW64、CR256、128KiB状态/128KiB系数、512字源ROM、一个issue和一套ready/pending。DMA输入与输出共用32B暂存；互斥锁不保证逐事务公平，因此不声称最优调度。', '',
'## 把保存税与重叠收益拆开','',
'| 函数 | 同保存布局串行 | 重叠省槽 | 保存/分块相对强普通多槽 | 最终净省槽 |',
'|---|---:|---:|---:|---:|']
for r in rows[::2]:
    lines.append(f"| {r['function']} | {r['batch_serial_slots']:,} | {r['same_layout_overlap_saved']:,} | {r['batching_tax_vs_best_direct']:,} | {r['saved_slots']:,} |")
lines += ['', '“同布局串行→交错”不是最终分母。最强普通立即消费updated，无需迁移sn2、保存全部anchors，且P1留Z更少重读。候选每项另付1,536次sn2 SW64、5,760次anchor SW64及相应SR；片上字节并未全部下降。', '',
'| 函数 / ready布局 | SR B | SW B | CR B | 外部输入 B |',
'|---|---:|---:|---:|---:|']
for r in rows[::2]:
    for key,label in [('baseline_bytes','立即消费P1'),('candidate_bytes','保存后交错')]:
        b=r[key];lines.append(f"| {r['function']} / {label} | {b['SR']:,} | {b['SW']:,} | {b['CR']:,} | {b['external_input']:,} |")
lines += ['', '容量内真实地址复用见[预定计划](PLAN.md)。sn2最后消费和A投影门实际外送结束后，才启用会覆盖这些区域的B源与PED暂存；没有把金值填入执行SRAM。各函数程序/分块沿用上一级实验中固定的CSE布局，没有扫描队列、分块或背压。', '',
'## 创新性裁决与尚未尝试','',
'相比首个P2实验，完整消费者让净收益扩大，说明不能凭过小工作组合杀整个家族。但普通34获得更大净收益，普通dense也可交错；小RF、PoT/CSE、分块和两流轮转仍属公共实现手段。当前结果没有建立lifting独占的X，也没有新的RTL/PPA证据。', '',
'此布局留作公共对照，不继续靠扩大窗口或扫队列争取标题。仍开放的执行接口是：将源输出的实际生产顺序直接接完整preview/后继，减少物化或重复读取；以及受限公共子图的跨消费者复用。必须与立即消费的普通实现比较，并保留原RNE/阈值边界；未实现之前不宣称这些接口会提速。', '',
'[结果JSON](summary.json) · [简表CSV](comparison.csv) · [执行代码](run.py) · [独立审阅](REVIEW.md)。重跑固定压力可用 `run.py AXIS --stress --mode direct_CSE_P1_Z,batch_joint_CSE`，解释器为 `/opt/anaconda3/bin/python3.12`。']
(HERE/'README.md').write_text('\n'.join(lines)+'\n')
print('summarized',len(cases),'cases')
