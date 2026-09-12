from pathlib import Path
import json

HERE=Path(__file__).resolve().parent


def main():
    execution=json.loads((HERE/'execution.json').read_text())
    baseline=json.loads((HERE.parent/'integrated_r32.json').read_text())
    pressure=json.loads((HERE.parent/'integrated_r32_ordinary_interior_stress.json').read_text())
    compilation=json.loads((HERE/'compilation.json').read_text())
    params=json.loads((HERE/'parameters.json').read_text())
    aee=json.loads((HERE.parent.parent/'algorithm/source34/aee/run.json').read_text())
    rows=[]
    for row in execution['rows']:
        same=[x for x in (pressure if row['stress'] else baseline)['rows'] if x['window']==row['window']]
        ordinary=next(x for x in same if x['axis']=='ordinary')['source_program']
        lifting=next((x['source_program'] for x in same if x['axis']=='lifting_raw'),None)
        rows.append(dict(window=row['window'],stress=row['stress'],source34_slots=row['service_slots'],
            dense_source_slots=ordinary['service_slots'],dense_reduction_percent=100*(1-row['service_slots']/ordinary['service_slots']),
            lifting_source_slots=lifting['service_slots'] if lifting else None,
            lifting_reduction_percent=100*(1-row['service_slots']/lifting['service_slots']) if lifting else None,
            source34_physical_port_bytes=row['physical_port_bytes'],
            source34_gate_checks=row['checks'],
            input_output_bytes_equal_dense=(row['counts']['SR64_reads']==ordinary['counts']['SR64_reads'] and
                row['counts']['SW64_writes']==ordinary['counts']['SW64_writes'])))
    result=dict(parent=params['parent'],changed_fields=params['changed_fields'],support=params['groups'],
        arithmetic_nodes=compilation['arithmetic_nodes'],program=compilation['program'],rows=rows,
        scope='Source stage only on actual fixed I24. Fresh diverse10 fails NB0; changed downstream service was not replayed.',
        AEE_status='Fresh ordinary diverse10 complete; stop this untrained parameter point, preserve the trained structured-source family.',
        AEE=dict(path='../../algorithm/source34/aee/run.json',frames=aee['summary']['frames'],
            frame_mean=aee['summary']['AEE_frame_mean'],holdout9=aee['holdout9_AEE'],
            NB0_frame_mean=aee['NB0_AEE'],better_than_NB0=aee['better_than_NB0'],
            actual_helper_fixtures=aee['actual_helper_fixtures']),
        ordinary_baseline_link='../final_combo_alignment/alignment.json proves current R24+onepass source inputs/program unchanged.',
        conclusion='Ordinary contiguous34 is faster at this source resource point but the untrained student fails NB0 quality. A matched trained structural control remains open; this result does not establish lifting-specific novelty.')
    rtl_path=HERE.parent/'rtl_source/structured_control_results.json'
    if rtl_path.exists():
        rtl=json.loads(rtl_path.read_text())
        result['common_source_RTL']=dict(path='../rtl_source/structured_control_results.json',
            cases=len(rtl['rows']),hardware_unchanged=rtl['hardware_unchanged'],
            differences=sum(x['differences'] for x in rtl['rows']),
            gate_bits_checked=sum(x['gate_bits_checked'] for x in rtl['rows']),
            source34_ready_cycles_per_H8=301,dense_ready_cycles_per_H8=497,lifting_ready_cycles_per_H8=410,
            scope='Source Verilator only, state-request interface excluding cold DMA/consumers; not integrated CPU slots or PPA.')
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table='\n'.join(f"| {r['window']} {'压力' if r['stress'] else 'ready'} | {r['dense_source_slots']:,} | {r['lifting_source_slots'] if r['lifting_source_slots'] else '未测'} | {r['source34_slots']:,} | {r['dense_reduction_percent']:.2f}% |" for r in rows)
    (HERE/'README.md').write_text(f'''# 普通连续3/3/4源强对照

**源执行及真实十帧已完成：AEE {aee['summary']['AEE_frame_mean']:.6f}，未达到同集NB0的{aee['NB0_AEE']:.6f}。** 停当前免训参数点，保留训练后的普通结构源对照；这不能证明 lifting 独有新颖性，也不是整个残差链结果。

B：现有 dense ordinary 已享完整常量 CSE，但没有本学生的普通结构源。A：迁移旧 dependency 的连续3/3/4支撑，共34个槽，只保留当前 ordinary `original_ordered24 + onepass` 父臂的原 As_q16 系数。旧工作拟合的是 r1.sn2；本次在 r1.sn1 新试，仅借支撑，旧系数、bias、精度均不继承。X：本臂不主张独占创新，作为 lifting 必须面对的普通结构 PSN 对照。

`deployed_constants.npz` 和父组合全部字段逐项比较，**仅 As_q16 改变**，As exponent15、source τ/θ、方向、consumer gain/阈值、Conv2/F、PED R24 和单遍 BN 均不变。`source34_adapter.install(helper, path)` 在当前 ordinary `FixedTemporalForward` 创建之后直接替换定点矩阵，不从 float 重新选择 exponent，不修改原 helper 文件。真实 I24 上独立重建的父门与实际 GPU 捕获两窗0差。

官方 da4ml0.6.0，同普通父臂 `wmc/auto` 整矩阵分解/CSE与固定 `last_use_pressure` 调度：**101个加减节点、124条实际程序、RF峰26个向量**。两级流水/寄存器寿命596个向量0差，1024域角点与4096随机向量的51,200个输出同时对独立整数 dot 和官方 DAIS0差。原 RNE15/sat24 与 inclusive τ 比较的精确整数前像仍可编译为最终门；不是删除必要的数值边界。

| 同范围源阶段 | ordinary dense | lifting40 | ordinary 34项 | 相对 dense 少服务 |
|---|---:|---:|---:|---:|
{table}

以上是 `execute_source.py` 在原完整集成所用 `IntegratedMachine` 上重新执行的真实输入、RF整数载荷与 SRAM 门写出，非节点数替换拍数。两边同96×8×48 RF、128KiB state、128KiB coefficient、8192B固定指令ROM、1R64/1W64 state和32B/5slot DMA。源系数已在双方的同容量静态ROM程序中，没有外部每帧系数加载。所有 I24 冷DMA/重读、指令取用、依赖等待、门收集/写出已收费。corner读233,280B/写248,832B，interior读348,480B/写371,712B，**与 dense 完全相同**；没有借低发放删除输出或消费者。三例310,080个门值均与新独立gold0差。实际源程序最高使用RF26附近但不缩减共同96向量容量，输出collector仍RF95。

源门明显变化：corner553/77,760个1（父4164，差3993位），interior885/116,160个1（父6670，差6703位）。因此不能把未变父臂的sn2、Conv2/PED输出或完整链服务借给本臂。当前仅给源阶段费用；新参数实际十帧未过质量门，未继续回放改变后的完整消费者，不把此36.88%当可用整链加速。

主代理已用**不改模块的同一公共源RTL**完成两窗×ready/stress四例，对真实I24、新CPU门gold与每个RF写回均0差，累计387,840门位。ready每H8：普通source34为301周期，dense497，lifting410；这些程序是不同新函数，仍须各自精度。[RTL结果](../rtl_source/structured_control_results.json)与[入口](../rtl_source/run_structured.py)另列。RTL边界从SR64接口开始，没有冷DMA/下游消费者，不能与上表CPU集成源阶段的服务槽混作一个倍率。`rtl_inputs/`保留真实两窗输入/新门gold及manifest。

本目录CPU执行标签仍为 **CPU payload slot prototype**，公共RTL是 **Verilator功能/周期原型**；均非VCS/DC/PT/Formality闭环、非PPA。ordinary父源费用来自原集成表的源分段；[最终组合对齐](../final_combo_alignment/README.md)已证明 current R24+onepass 没改该源输入/算术。

最终[真实网络评价](../../algorithm/source34/aee/run.json)：diverse10 AEE **{aee['summary']['AEE_frame_mean']:.6f}**，排除首帧后9帧 **{aee['holdout9_AEE']:.6f}**，均未过对应NB0。actual helper的193,920个Q24值及193,920个门与新CPU函数0差。仅停止本次未恢复的3/3/4遮罩，不能据此杀结构稀疏家族，也不能替代lifting与同预算训练普通结构的比较。

主要产物：[参数](parameters.json)、[编译](compilation.json)、[源服务](summary.json)、[实际执行](execution.json)、[GPU adapter](source34_adapter.py)。编译器沿用[da4ml官方实现](https://github.com/calad0i/da4ml)；本次不新增文献/训练/生产RTL/EDA。
''')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
