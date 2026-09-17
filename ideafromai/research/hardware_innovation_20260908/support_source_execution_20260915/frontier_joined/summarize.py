"""Summarize actual joined runs; never add independently timed leaves."""
from pathlib import Path
from collections import defaultdict
import csv,json

HERE=Path(__file__).resolve().parent
METRICS=['cycles_to_last_gate','cycles_to_done','source_phase_cycles','backend_phase_cycles',
         'backend_boot_cycles','backend_fc_cycles','backend_psn_cycles','words','bytes',
         'dictionary_words','source_config_words','source_X_words','source_graph_words',
         'backend_config_words','backend_coeff_words','source_produced_pairs','source_channel_refs',
         'source_scalar_mac','backend_vector_mac','backend_vector_updates','bridge_writes','bridge_reads',
         'req_stall','out_stall']
ARMS={(1,0):'static64 + next-X PF',(2,0):'one code',(2,1):'resident frontier code',
      (3,0):'one class',(3,1):'resident frontier class'}
def load(stem):
    rows=list(csv.DictReader((HERE/(stem+'.csv')).open()))
    for r in rows:
        for k in r:
            if k not in ('case','function'):r[k]=int(r[k])
        assert r['hblocks']==4 and r['backend_mode']==4
        assert r['source_scalar_mac']==10*r['source_produced_pairs']
        assert r['bytes']==16*r['words']
    return rows
def group(rows,omit_selection=False):
    out=defaultdict(lambda:{k:0 for k in METRICS})
    for r in rows:
        if not r['real'] or r['pass'] or (omit_selection and r['case']=='train0'):continue
        k=f"{r['mode']}_{r['frontier']}_{r['bp']}"
        for m in METRICS:out[k][m]+=r[m]
    return dict(out)
def compare(rows,a,b):
    pairs=defaultdict(dict)
    for r in rows:
        if r['real'] and not r['pass'] and r['case']!='train0':pairs[(r['case'],r['bp'])][(r['mode'],r['frontier'])]=r
    res={}
    for bp in (0,1):
        ps=[v for (_,pressure),v in pairs.items() if pressure==bp]
        base=sum(v[a]['cycles_to_last_gate'] for v in ps)
        cand=sum(v[b]['cycles_to_last_gate'] for v in ps)
        res[str(bp)]={'baseline':base,'candidate':cand,'reduction_pct':100*(base-cand)/base,
                     'negative_frames':[case for (case,pressure),v in pairs.items() if pressure==bp and v[b]['cycles_to_last_gate']>v[a]['cycles_to_last_gate']]}
    return res
def main():
    data={};lines=['# 四槽源执行的完整子链结果','',
        '真实 X→源 PSN→最近码/响应类→RTL D 展开→FC1 H384→后级完整 T10 PSN。只报实际 joined RTL 的最后 gate 周期，不相加源叶和后端叶。',
        '', '主分母为 31 个未参与 pair 选择的训练帧，各固定 32 个抽样位置；仍属于学生训练缓存，非 valid825。先前 frame0 保留训练选择角色，另外两例为诊断。后端固定 τ/增益函数，未闭动态 BN、FC2、shortcut、整网 AEE。',
        '', '资源：106 个分别实例化的乘法单元（源10+后端96，阶段串行）、256KiB共同参数池、8×128bit每bank一笔在途、3840B门桥、192B D。各臂共同给128B X holding和128B图cache；相对最初源核增加96B X。根均读物理6174，消除了仅根bank位置不同的对照偏差。', '']
    total=0
    for name in ('old','new'):
        rows=load('expanded_'+name);smoke=load('smoke_'+name)
        assert len(rows)==340 and len(smoke)==80
        aggregate=group(rows,True)
        data[name]={'scope':'31 unselected training frames x fixed P32, H384',
                    'all32':group(rows),'unselected31':aggregate,
                    'commands':len(rows)+len(smoke),
                    'comparisons':{label:compare(rows,a,b) for label,a,b in [
                        ('frontier_code_vs_one_code',(2,0),(2,1)),
                        ('frontier_code_vs_static',(1,0),(2,1)),
                        ('frontier_class_vs_frontier_code',(2,1),(3,1)),
                        ('frontier_class_vs_one_class',(3,0),(3,1))]}}
        total+=len(rows)+len(smoke)
        lines+=['## '+('旧最近响应 W′' if name=='old' else '源费用选择 W″'),'',
                '| 同权重函数执行臂 | ready周期 | BP周期 | 源实际标量MAC | ready/BP总字节 |',
                '|---|---:|---:|---:|---:|']
        for (mode,frontier),label in ARMS.items():
            a,b=(aggregate[f'{mode}_{frontier}_{bp}'] for bp in (0,1))
            lines.append(f"| {label} | {a['cycles_to_last_gate']} | {b['cycles_to_last_gate']} | {a['source_scalar_mac']} | {a['bytes']}/{b['bytes']} |")
        lines+=['','| 比较，正数表示周期减少 | ready | BP |','|---|---:|---:|']
        for label,c in data[name]['comparisons'].items():
            lines.append(f"| {label} | {c['0']['reduction_pct']:.4f}% | {c['1']['reduction_pct']:.4f}% |")
        lines+=['']
    data['verification']={'commands':total,'Y_U_gate_checks_each':total*32*10*384,
                          'scope':'all actual produced (P,c,t), final codes, D writes/reads, full backend Y/U/gate, BP holds, physical banks, repeat without reset'}
    lines+=['## 判断','',
        '普通 code 已获得前沿并行与驻留增量；class 必须只取相对同权限 code 的额外收益。W″与W′是不同近似权重，跨表不能声称无损加速或继承AEE。前沿/决策图/缓存归借入底座；目前测试用于确定按需生产接口是否值得保留，尚不构成新颖性或PPA证明。',
        '',f'全部 {total} 个完整 H384 命令通过，Y/U/gate 各 {total*32*10*384:,} 次核对；完整源输出不喂给DUT，仅逐 active(t,c)检查实际结果。无 VCS/DC/PT/Formality、PPA、动态BN或整网测量。',
        '', '启动时首次脚本在编译期间被修改导致shell偏移执行失败，未产生可用结果；本表仅使用稳定脚本重启后的PASS运行。',
        '', '运行：`bash run.sh`；再导出 `prepare.py --expanded [--adapt]`，按 `run_expanded.sh` 运行并执行本汇总。']
    (HERE/'SUMMARY.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(data['verification'],ensure_ascii=False))
if __name__=='__main__':main()
