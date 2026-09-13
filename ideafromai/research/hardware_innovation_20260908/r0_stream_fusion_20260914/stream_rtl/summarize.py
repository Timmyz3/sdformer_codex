from pathlib import Path
import csv,json
H=Path(__file__).resolve().parent
full=json.loads((H/'results_full.json').read_text())
assert len(full)==6 and all(r['tiles']==19200 for r in full)
small=json.loads((H/'results64.json').read_text());assert len(small)==24
checks=json.loads((H/'results_full_checks.json').read_text());assert checks['passed']
index={(r['arm'],r['mode']):r for r in full}
summary={'complete':True,'full_frame_jobs':6,'tiles_per_job':19200,'full_frame_checked_outputs':sum(r['checked_outputs'] for r in full),
 'continuous64_jobs':24,'continuous64_checked_outputs':sum(r['checked_outputs'] for r in small),
 'scope':'One go per complete frame, one persistent RTL instance, one W/mask load, all source/origin/config/stalls/outputs charged; complete r0 Q16 linear layer only.',
 'full_results':full,'benefits':{}}
for arm in ('dense_q16','block_magnitude25','cin_fullcost25'):
 a,b=index[(arm,5)],index[(arm,6)]
 summary['benefits'][arm]={'saved_total_cycles':a['total_cycles']-b['total_cycles'],
 'total_reduction_percent':100*(1-b['total_cycles']/a['total_cycles']),
 'core_reduction_percent':100*(1-b['core_cycles']/a['core_cycles'])}
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
fields=['arm','mode','tiles','total_cycles','core_cycles','static_weight_words','static_mask_words','source_load_words','external_source_words','padding_words','origin_words','source_load_stalls','parameter_stalls','core_source_words','core_weight_words','core_psum_reads','core_psum_writes','core_update_issues','core_sum_issues','core_merge_issues','output_beats','core_source_stalls','core_weight_stalls','core_output_stalls','checked_outputs','wall_seconds']
for filename,rs in [('full_costs.csv',full),('continuous64_costs.csv',small)]:
 with (H/filename).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rs)
old=H.parent.parent/'r0_source_retirement_20260913/pair_parent_merge/resource_contract.json'
contract=json.loads(old.read_text());contract['scope']='stream_wrapper plus unmodified pair_parent_merge leaf, modes5/6'
contract['leaf_previous_stage_additions']=contract.pop('additional_common_registers_vs_previous_module')
contract['loading']={'static_weight_and_mask_cycles_once_per_instance':10656,'source_and_origin_cycles_per_tile':1537,'same_instance_next_job_static_cycles':0,'note':'All source/origin loading repeats for every job; only W/mask remains resident. Input padding still costs source SRAM configuration writes.'}
contract['backpressure']='Parameter response, external source response, core source/W permits and result ready use whole-job cycle schedules; not reset between tiles.'
contract['scope_limits']=['No overlapped tile load/compute or double buffer','No DDR/cache timing model','No BN/PSN/residual downstream RTL','No EDA or energy claim','No full Prosperity/ELSA/Phi reproduction']
contract['wrapper']={'source_buffers':1,'additional_tile_buffers':0,'overlap_load_compute':False,
 'frame_source_port':'At most one requested 10bit source response accepted per cycle; native global address in SV; response may stall; no DDR latency model',
 'static_configuration_port':'At most one128bit configuration beat/cycle; 10368W rows then288 single-mask-bit padded beats; resident until reset',
 'source_loading':'1536 core10bit writes/tile including padding; origin one extra write; halo requests repeated and paid',
 'result':'one256bit beat+15bit tile_id+9bit row+tile_last/job_last; all480 beats accepted before retirement',
 'additional_functional_declared_bits':74,'diagnostic_64bit_counters':21,'retired_tiles_diagnostic_bits':32,
 'job':'first_tile/count/mode accepted only idle; no command queue; legal jobs tested; two64tile jobs reuse static parameters',
 'cycle_identity':'total = sum(core_cycles) + staticW + staticmask + parameter_stall + source_load + origin + source_load_stall + 2*tiles +1',
 'not_implemented':['line buffer or cross-tile halo reuse','overlapped source loading and compute','upstream suppression for retired C4','DDR/cache timing','BN/PSN/residual/fullnetwork bittrue','EDA/Fmax/energy']}
(H/'resource_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
lines=[]
for arm in ('dense_q16','block_magnitude25','cin_fullcost25'):
 a,b=index[(arm,5)],index[(arm,6)];q=summary['benefits'][arm]
 lines.append(f"| {arm} | {a['total_cycles']:,} | {b['total_cycles']:,} | {q['total_reduction_percent']:.2f}% | {a['core_cycles']:,} | {b['core_cycles']:,} |")
table='\n'.join(lines)
dense=index[('dense_q16',6)];cost=index[('cin_fullcost25',6)]
report=f'''# 整帧连续 r0 执行：mode5/6 共用真实加载与结果完成边界

2026-09-14。**三个mask、两模式的六个完整帧作业全部通过：每作业一次go，连续19,200个tile，累计442,368,000个signed32输出逐值匹配。** 首64tile跨行套件另有24作业、5,898,240个输出全绿，覆盖源/参数/内部端口/输出背压及同实例第二次作业。见[完整结果](results_full.json)、[64tile结果](results64.json)、[汇总](SUMMARY.json)、[完整费用CSV](full_costs.csv)、[64tile费用CSV](continuous64_costs.csv)。

## 实际执行边界

`pair_parent_merge.sv`原样复制上一阶段最终版本；新增[stream_wrapper.sv](stream_wrapper.sv)拥有tile循环、源地址、origin、启动、身份和完成状态。一次go首先以共同128bit口付10,368个W行和288个mask配置拍，静态驻留；之后同实例连续加载、执行并完成整帧。没有TB独立leaf实例求和。

SV根据tile行主序和加载索引产生完整源地址`c*76800+y*320+x`，由外部每拍至多一个10bit弹性响应供数。图外不发外部请求，仍向内部tile SRAM写零并付配置拍；origin由SV计算且另付一拍。TB只响应请求地址、施加许可并检查独立整层gold，不预gather源、不发送origin、支持码、消费者集合或中间和。

固定一个源buffer；相邻tile的halo重复请求、全部收费，加载不与当前计算重叠。输出每拍8个signed32加tile_id/row/tile_last/job_last。结果被阻塞时整个包保持；480beat全部握手且leaf.done后，wrapper才退休tile并推进下一次加载。最后一个tile也经同样完成路径，最终done不能提前。W/mask在第二个同实例作业中确实不再装载。

## 三臂完整帧的实际周期

全部为冷作业、无外部阻塞、完整240×320输出，每个tile同时含T10、Cin96、Cout96、3×3全部K864。总拍已经包含静态参数、每tile所有源/origin加载、clear/drain和wrapper控制。

| mask | mode5总拍 | mode6总拍 | 总拍减少 | mode5核心拍 | mode6核心拍 |
|---|---:|---:|---:|---:|---:|
{table}

三个mask的每作业共同装载成本是10,656个静态配置拍、29,491,200个tile源写拍、19,200个origin拍、38,400个launch/完成捕获拍以及最后1个finish拍，共29,559,457拍。外部有效源请求29,276,544次，padding写214,656次；全帧输出9,216,000个256bit beat。所有这些均由SV计数，不把加载或最后输出省略。

dense mode6运行时内部source读29,276,544词、W读153,790,128个128bit行、psum读/写各259,932,480个八lane beat；Cin fullcost分别为21,957,408、115,600,812、197,030,040。完整C4退休只取消内部source/执行，当前wrapper仍加载全部源词，外部29,276,544请求不因Cin mask减少。mode5/6同mask的源和W读完全一致，净差来自付费MERGE后取消第二个psum pass及重复终止扫描。

## 连续、背压、重启与验证

首64tile固定id128..191，经过tile行0→1接缝，三臂两模式各两种许可波形、每个实例两次完整作业。许可按整个作业周期计数，**不在每tile重置**；第二作业从0重启许可波形。每个输出逐tile身份、地址、last和lane值检查；源请求地址、参数请求与结果包在阻塞时保持。首job静态W/mask各10368/288，第二job均0，源/origin则两个job都重新付费。

整帧作业从id0到19199连续执行，覆盖所有行、四条图界及最终tile。整帧为无阻塞测试；带阻塞的实际连续覆盖是64tile套件，未将其写成整帧背压证明。两套运行均核对[输入导出的原生几何/支持费用](predict_work.py)和[完整计数检查](verify.py)，gold来自[数据manifest](../data/manifest.json)与[完整gold校验](../data/gold_progress.json)，计算顺序独立于RTL支持码/父和调度。

每作业总账恒等式为`Σcore + staticW + staticmask + parameter_stalls + source_load + origin + source_load_stalls + 2*tiles +1`。核心计数再与完整源数组的native几何/pair支持预测逐字段比较，阻塞拍单列；不把预测值冒充RTL输出。

## 速度瓶颈与适用范围

64tile预跑每单作业约0.55–0.82秒，结合整帧精确工作量估计单帧数分钟，低于30分钟限额后才执行完整帧。最终六作业各自仿真耗时{min(r['wall_seconds'] for r in full):.1f}–{max(r['wall_seconds'] for r in full):.1f}秒；这是主机Verilator运行耗时，不是硬件帧时延。构建为Verilator4.028 `--cc --exe -CFLAGS -O3`后make，未做EDA。

当前dense mode6中，psum读/写仍合计519,864,960拍；逐时间选择与其后的读改写仍是大项，运行时W读取也有153,790,128拍。输入/配置和wrapper费用约占总拍{100*29559457/dense['total_cycles']:.2f}%，单靠隐藏本次加载不是主要加速来源。顺序tile供数目前只闭合执行边界，未优化halo复用或计算/加载重叠。

双方仍一个10bit内部source读口、8×16bit共同行W口、原8×480×32bit psum且读写分拍、八条32bit数据加法链。wrapper只增74bit按声明计的功能状态，没有额外tilebuffer或psum端口；64bit诊断统计另计，详见[资源合同](resource_contract.json)。外部源使用可阻塞的单词响应接口，没有DDR/cache时延模型，不能据此推断外部存储PPA。

mode6父和转发/归并仍为已有product reuse执行A；完整帧验证扩大了真实范围，没有单凭约5%的周期差升级为标题X。输入、静态mask与精度和上一阶段一致；这里只验证整个r0 Q16整数线性层，未连接后续BN/PSN/残差或给出全网bittrue/AEE结论。旧目录只读，本阶段不另开idea。

复现：`prepare.py`→`build.py`→`predict_work.py`→`run.py`；完整帧参数`--first 0 --count 19200 --stalls 0 --repeats 1 --output results_full.json`，再`verify.py results_full.json`和`summarize.py`。
'''
(H/'REPORT.md').write_text(report)
(H/'README.md').write_text('# Continuous native r0 stream RTL\n\n完整结果与边界见[REPORT.md](REPORT.md)，费用见[full_costs.csv](full_costs.csv)。\n\n- 六个整帧单go连续作业，442,368,000输出全绿。\n- 24个跨行64tile作业，5,898,240输出全绿，含背压/参数驻留/同实例重启。\n- `stream_wrapper.sv`拥有全部tile/源地址/origin/启动/输出身份与完成状态。\n- 原样`pair_parent_merge.sv`保持mode5/6同底座。\n- `resource_contract.json`列出真实接口和未实现的重叠/halo复用/DDR/PPA边界。\n')
print(json.dumps(summary['benefits'],indent=2))
