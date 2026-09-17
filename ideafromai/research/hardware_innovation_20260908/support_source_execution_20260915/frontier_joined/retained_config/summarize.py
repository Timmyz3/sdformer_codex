from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import summarize as parent
HERE=Path(__file__).resolve().parent
SEMANTIC=['source_produced_pairs','source_channel_refs','source_scalar_mac','backend_vector_mac',
          'backend_vector_updates','backend_coeff_words','backend_config_words','bridge_writes','bridge_reads']
def read(path):
    rows=list(csv.DictReader(path.open()))
    for r in rows:
        for k in r:
            if k not in ('case','function'):r[k]=int(r[k])
    assert len(rows)==340
    return rows
def key(r):return r['case'],r['mode'],r['frontier'],r['bp'],r['pass']
def main():
    data={};lines=['# 最终强控制：P32内静态参数只加载一次','',
        '真实源到完整FC1 H384/T10 PSN链；31个未用于pair选择的训练帧，每帧固定P32。所有模式P0冷加载，P1..31复用既有静态寄存器；每次top命令重装，X holding和图cache仍逐P清空。五臂均活动前沿PF、统一root bank；这项参数驻留是普通基线权限。',
        '', '本表为cycles_to_last_gate，不是DONE、孤立叶或整网周期。每个函数分别比较，同W下code/class整数完全等价；W′/W″改变网络权重，不能跨表借质量。','']
    for name in ('old','new'):
        rows=read(HERE/f'expanded_{name}.csv')
        before=read(HERE.parent/'active_prefetch'/f'expanded_{name}.csv')
        index={key(r):r for r in before}
        for r in rows:
            assert r['source_config_words']==(30 if r['mode']<2 else 21)
            for k in SEMANTIC:assert r[k]==index[key(r)][k],(key(r),k)
        agg=parent.group(rows,True)
        data[name]={'all32':parent.group(rows),'unselected31':agg,
                    'comparisons':{label:parent.compare(rows,a,b) for label,a,b in [
                        ('frontier_code_vs_static',(1,0),(2,1)),('frontier_code_vs_one_code',(2,0),(2,1)),
                        ('frontier_class_vs_code',(2,1),(3,1)),('frontier_class_vs_one_class',(3,0),(3,1))]}}
        lines+=['## W'+('′' if name=='old' else '″'),'',
                '| 同函数执行臂 | ready / BP周期 | 源标量MAC | ready / BP字节 |', '|---|---:|---:|---:|']
        for (mode,frontier),label in parent.ARMS.items():
            a,b=[agg[f'{mode}_{frontier}_{bp}'] for bp in (0,1)]
            lines.append(f"| {label} | {a['cycles_to_last_gate']}/{b['cycles_to_last_gate']} | {a['source_scalar_mac']} | {a['bytes']}/{b['bytes']} |")
        lines+=['','| 比较，减少为正 | ready | BP |','|---|---:|---:|']
        for label,c in data[name]['comparisons'].items():
            lines.append(f"| {label} | {c['0']['reduction_pct']:.4f}% | {c['1']['reduction_pct']:.4f}% |")
        lines+=['']
    data['verification']={'commands':680,'Y_U_gate_checks_each':83558400,
                          'static_words_per_tile':30,'graph_words_per_tile':21,
                          'semantic_work_equal_to_previous_active_PF':True}
    lines+=['## 判读与资源','',
        '普通前沿code共享全部供数与驻留优化。只有class相对这个强code的剩余差，才是消费者等价改变源义务的候选增量；参数驻留和预取本身不计创新。',
        '', '680个完整H384命令通过；Y/U/gate各83558400次核对，partial(t,c)、批内独立channel refs、所有物理请求和反压均由TB检查。所有源实际MAC、后端系数/更新/PSN MAC和桥接语义计数与旧活动PF逐条相同，只有静态配置加载和服务时序改变。',
        '', '维持106个独立乘法单元、256KiB外部参数/X池；另有Y90KiB、routes7.5KiB、U/tau各5.625KiB及其它局部状态、3840B门桥、192B D、128B X和128B图cache。复用已有参数寄存器，不增加容量；不称同面积或同Fmax。仍没有动态BN/FC2/shortcut、有效新AEE、VCS/DC/PT/Formality或PPA。']
    (HERE/'SUMMARY.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(data['verification'],ensure_ascii=False))
if __name__=='__main__':main()
