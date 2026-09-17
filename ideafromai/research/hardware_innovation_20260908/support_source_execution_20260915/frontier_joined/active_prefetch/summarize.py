from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import summarize as parent

HERE=Path(__file__).resolve().parent
def read(path):
    rows=list(csv.DictReader(path.open()))
    for r in rows:
        for k in r:
            if k not in ('case','function'):r[k]=int(r[k])
    assert len(rows)==340
    return rows
def key(r):return r['case'],r['mode'],r['frontier'],r['bp'],r['pass']
def main():
    data={};unchanged=0
    lines=['# 当前活动前沿预取：统一根bank后的完整子链','',
        '同32训练帧固定P32＋2诊断，H384完整FC1/PSN。主表只合计frame1..31，未用于pair选择但仍属于训练缓存。两臂来自实际独立RTL运行；周期列为cycles_to_last_gate，不混done握手。',
        '', '| 函数及强控 | PF旧规则 ready/BP | PF活动规则 ready/BP | 减少 ready/BP |', '|---|---:|---:|---:|']
    for name in ('old','new'):
        old=read(HERE.parent/f'expanded_{name}.csv');new=read(HERE/f'expanded_{name}.csv')
        oldmap={key(r):r for r in old}
        for r in new:
            if not r['frontier']:
                assert r==oldmap[key(r)],key(r)
                unchanged+=1
        a,b=parent.group(old,True),parent.group(new,True)
        data[name]={'old':a,'active':b,'comparisons':{
            'code_vs_static':parent.compare(new,(1,0),(2,1)),
            'class_vs_code':parent.compare(new,(2,1),(3,1))}}
        for mode,label in ((2,'code'),(3,'class')):
            xs=[a[f'{mode}_1_{bp}']['cycles_to_last_gate'] for bp in (0,1)]
            ys=[b[f'{mode}_1_{bp}']['cycles_to_last_gate'] for bp in (0,1)]
            pct=[100*(x-y)/x for x,y in zip(xs,ys)]
            lines.append(f'| W{"′" if name=="old" else "″"} {label} | {xs[0]}/{xs[1]} | {ys[0]}/{ys[1]} | {pct[0]:.4f}%/{pct[1]:.4f}% |')
    data['verification']={'commands':680,'Y_U_gate_checks_each':680*32*10*384,
                          'static_one_channel_unchanged_records':unchanged,
                          'source_reference_counter':'independent unique channel set per actual producer batch asserted in TB'}
    lines+=['', '## 最终同函数比较','', '| W″，活动PF | ready | BP |','|---|---:|---:|']
    for label,comp in data['new']['comparisons'].items():
        lines.append(f"| {label} | {comp['0']['reduction_pct']:.4f}% | {comp['1']['reduction_pct']:.4f}% |")
    lines+=['', f'680个完整命令PASS，Y/U/gate各{680*32*10*384:,}次核对。{unchanged}条static/one-channel记录与父运行逐字段相同；只改变前沿模式预取对象。增加了每次producer批内不同channel的独立计数断言，闭合source_channel_refs，而非直接信核内counter。',
        '', '代价与判断：活动PF多发图请求，不能仅用周期推出能耗下降。106个独立运算单元、所有局部数组、256KiB参数池/8bank/128B X与128B图cache都保持；模式选择、容量和root物理bank未变。它是把已借入的预取适配到真正active请求的供数修复，普通code全部同享，不作为响应类别独有创新。',
        '', '精度范围仍为各自近似权重的同函数整数核验；无新AEE、动态BN/FC2/shortcut、PPA或整网周期。源叶的graphword增量/局部负例另见../../frontier_source/active_prefetch/README.md；本目录只给真实joined差分。']
    (HERE/'SUMMARY.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(data['verification'],ensure_ascii=False))
if __name__=='__main__':main()
