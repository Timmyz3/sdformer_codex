#!/usr/bin/env python3
from pathlib import Path
import csv,json
H=Path(__file__).resolve().parent
s=json.loads((H/'SUMMARY.json').read_text())
contract={
 'scope':'One common c4_execution module; modes3/4/5 runtime selected. Mode3 cycles exactly match prior consumer_enumeration.',
 'shape':{'T':10,'Cin':96,'Cout':96,'input_hw':[4,4],'output_hw':[2,2],'kernel':[3,3]},
 'source':{'words':1536,'word_bits':10,'read_ports':1,'bytes':1920,'layout':'c*16+y*4+x; origin boundary gating in RTL'},
 'weight':{'banks':8,'rows_per_bank':10368,'bits_per_bank_word':16,'common_row_reads_per_cycle':1,'bytes':165888,'layout':'bank=n%8,row=(n//8)*864+c*9+tap'},
 'psum':{'banks':8,'rows_per_bank':480,'bits_per_bank_word':32,'bytes':15360,'access':'one common row; PS_READ and ADD_WRITE separate cycles; clear/drain paid'},
 'static_mask':{'bits':288,'bytes':36,'layout':'O8 x Cin4 x all9taps','access':'fixed 12bit same-Cg decode and 48dest combinational geometry predicate; not an added SRAM read bank'},
 'data_alu':{'lanes':8,'width_bits':32,'shared_by':['pair_sum','C4_build','psum_update'],'note':'Address/control incrementers and priority logic separate; no synthesis/area/Fmax claim'},
 'additional_common_registers_vs_previous_module':{
   'four_source_holds_bits':40,'four_weight_vectors_bits':512,'one_signed18_scratch_bits':144,
   'demanded_pattern_bitmap_bits':15,'weight_pending_bits':4,'scratch_remaining_bits':4,
   'source_index_and_pair_index_declared_bits':64,'mode_extension_bits':1,
   'total_functional_declared_bits':784,'additional_pattern_copy_counter_bits':32,
   'note':'Old WA/WB/WAB, pair source holds and old mask_a/mask_b remain; union budget physically shared by runtime modes, not duplicated per mode. 784bits=98B functional declared state; diagnostics extra4B. Integer loop bounds can synthesize smaller; no EDA estimate.'},
 'register_read_permissions':{'four_source_holds':'all40bits exposed to10 four-bit support encoders after four serialized source SRAM reads',
  'four_weight_holds':'one vector selectable for copy/build, two vectors selectable for ordinary pair pre-sum; register reads, not extra weight SRAM ports',
  'psum':'no internal arbitrary multi-read; one row in PS_READ, one row in DRAIN_READ'},
 'loading':{'actual_first_command_per_instance_cycles':12193,'actual_second_same_input_command_cycles':0,
   'static_weight_and_mask_cycles':10656,'fresh_source_and_origin_cycles':1537,
   'note':'TB descriptor configuration_cycles=12193 appears in both command rows but was physically loaded once. Core counters exclude cfg. Full host source/W load is retained even for retired groups.'},
 'backpressure':'source_allow/weight_allow/result_ready waveforms restart n=0 each command; output address and data must hold while !ready',
 'scope_limits':['No full-image stream or double buffer','No BN/PSN/residual downstream RTL','No EDA or energy claim','No full Prosperity/ELSA/Phi reproduction']}
(H/'resource_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
with (H/'benefits.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['arm','mode','core_cycles','fresh_source_origin_cycles','source_words','weight_words','psum_updates','sum_issues','pattern_copies','stressed_core_cycles'])
 for arm,ms in s['formal_arms'].items():
  for m in ('3','5','4'):
   r=ms[m]['0'];w.writerow([arm,m,r['cycles'],r['cycles_with_fresh_source_origin'],r['source_words'],r['weight_words'],r['update_issues'],r['sum_issues'],r['pattern_copies'],ms[m]['1']['cycles']])
rows=[]
for arm,ms in s['formal_arms'].items():
 a,b,c=ms['3']['0'],ms['5']['0'],ms['4']['0']
 rows.append(f"| {arm} | {a['cycles']:,} | {b['cycles']:,} | {c['cycles']:,} | {(c['cycles']/b['cycles']-1)*100:.2f}% | {b['source_words']:,} | {b['weight_words']:,} |")
table='\n'.join(rows)
report=f'''# 固定单 scratch 的 C4 支持码归约：功能通过，未超过共同供数强控制

2026-09-13。完整 [结果](results.json)、[汇总](SUMMARY.json)、[费用 CSV](benefits.csv)、[资源合同](resource_contract.json) 已封口；本目录不继续改变构造策略。

本次完成了真实原生 C96/N96/T10、4×4→2×2、3×3 全 K864 线性叶的三模式 RTL。**648 runs、2,488,320 个 signed32 输出全匹配；6,480 项独立事务/完整周期核对全通过。** 固定一个 signed18 scratch 逐实际支持码构造的 mode4，在五个真实分支均慢于共同 C4 供数的 mode5，差 7.53%–7.90%。mode5 比旧 mode3 快 5.54%–6.34%；这是供数、暂存和消费者循环合并的 A 收益。

## 同硬件对照与实际机制

[前一轮 mode3](../../r0_execution_trials_20260913/consumer_enumeration/REPORT.md) 原样保留，两源时间字沿原生源坐标枚举消费者，必要时构造 WA+WB，每对分别更新 psum。新 mode5 一次读同坐标的四个 T10 源字、四个按需 W 向量，共用一次合法消费者枚举，但仍按两对分别 psum 读改写。最后 W 读取直接准备第一个非空 pair，第一对末直接启动第二个非空 pair；不存在人为的 pair 选择拍或空对扫描。两者所有 source/W 请求、pair_sum 次数、psum 更新逐 fixture 相同，旧 mode3 周期也逐项复现。

mode4 继承 mode5 的四源/四 W 供数，RTL 从四个源时间字产生十个 4bit 码，并只枚举出现的非零码。每个目的 `(og,p)` 下，码 k 先复制首 W 到唯一 signed18×8 scratch，支付一拍；其余项经共同八条 32bit 加法链逐项加入，支付 popcount(k)−1 拍。随后只更新属于该码的 t，完成后复用 scratch。没有十五项和表、离线支持码、TB 中间和、预 gather 或额外 psum 端口。原生 geometry/origin、静态 mask、全部循环、背压、clear 与 drain 均在 SV。

## 主实测：八块之和，command0、无外部背压

| 同函数 mask | mode3 两源枚举 | mode5 C4供数两对 | mode4 单scratch码归约 | mode4 比5慢 | source 10bit词 | W 128bit词 |
|---|---:|---:|---:|---:|---:|---:|
{table}

每行三模式共用相同 source、Wq16、live mask 与整数 gold。前三列为包含 clear/drain 的实测核心拍；**每个新 tile 的 source/origin 装入另付 1,537 拍**，八块共 12,296 拍。计入后 dense 的 mode3/5/4 为 427,320 / 403,416 / 434,304，Cin fullcost 为 325,092 / 306,612 / 328,956。静态 W+mask 的 10,656 拍可单次驻留；TB 实际每个新实例完整配置 12,193 拍，因此 cold-instance 八块总拍另加 97,544。第二 command 用同一配置和同一 tile 功能重启，实际配置为零；JSON 中重复打印的 `configuration_cycles=12193` 是一次配置的描述，不可再收费，也不可把第二次重启当新输入免费。

Cin 两臂将六个完整 C4 组退休，source 内部读词由 9,600 降至 7,200。输入配置仍发送全部 1,536 源词，源退休只取消运行时 SRAM 读和对应执行，未声称 host/上游传输已取消。完整 C4 退休和 block mask 都在三模式执行同一 mask。原 physical25 仅作为开发回归，未与新训练校准质量混合统计。

## 费用为什么没有回本

对一个 C4 native 源和目的，设 U 为非零支持的 t 数，H 为两对同 t 都活动的数量，K 为有活动的 pair 数，M 为实际非零码数，J 为所有实际码的 popcount−1 之和，Q 为两 pair 的必要父和构造数，D 为需要读取的 W 行数。两种共同供数模式都付相同四源读取、geometry、mask 与 D 个 W 请求。

mode5 目的服务为 `D + Q + 3(U+H) + K + 2` 拍；mode4 为 `D + J + 3U + 2M + 2` 拍。这里 3 次状态是时间选取、psum 读、psum 写；每码多一复制拍与一次 NEXT_TIME 终止拍。故 mode4−mode5=`(J−Q)+2M−K−3H`，包括全部选择和构造费用。

dense 实测 mode4 少 6,360 次八 lane psum 更新，即各少 6,360 次 256bit 读与写；但多 5,076 次向量加法，并支付 40,596 次码复制。控制剩余少 2,064 拍，最终仍多 30,888 拍。该收益与费用均由 DUT counters 和 [独立闭式检查](verify_ledger.py) 对齐。固定全一功能输入则因同码重复十次，mode4 429,218 拍低于 mode5 729,890 拍；它只证明该构造在高重复支持上可获益，不进入真实 workload 汇总。

## 资源、验证与结论边界

共同预算为 source 1×10bit 读口、W 8×16bit 同行读、psum 8×32bit 原 480 行且读写分拍、八条 32bit 数据加法链。原有所有寄存器保留，另加四源 hold 40bit、四 W 向量 512bit、单 scratch 144bit、support/remaining metadata 23bit、两个按 SV 声明计的 32bit 索引及一位 mode 扩展，共 784bit 功能状态（98B）；新计数器另 32bit。三模式实例化同一份联合资源。四源并行码生成和临时 W 选择是寄存器连线/选择器，不能视作额外 source/W SRAM 多读。组合选择器深度及 Fmax/面积尚未经过 EDA，因此只比较此同模块的周期与事务。

验证含 40 个正式真实 tile、8 个原 physical 开发 tile、6 个固定功能输入，每个模式两种 command-relative 背压波形、每次两条不 reset command。`all_supports_extreme` 在一个固定满形状输入中覆盖 0..15 全码、重复时间码、四项全 −32768 / +32767 与混合符号；另有全零、全一、静态全删、角点和图外毒值。所有 480 个输出 beat 逐 lane 检查，背压期间 data/addr 必须稳定。[ledger_checks.json](ledger_checks.json) 从实际配置的 source/mask/origin 独立计算原生依赖和计数，不向 TB 提供机制。

输入来自独立数据分支 [数据与质量](../data/README.md)：固定 thun 训练帧的 336 栅格 tile 做校准，硬件评价为旧 zurich 的八块；未按 mode4 结果重选 mask。五臂 diverse10 均过原较严 NB0 门，具体精度/质量归该报告；RTL 检查仅覆盖此整数线性叶，未接全图输入流水、BN/PSN/残差或下游量化。

按二值支持复用父和/子集和是已有 product-sparsity 思路。本次提供可执行强 A 与被真实费用否定的一个有界构造点，**不将约束下的负结果扩展成对 Prosperity 或整个 C4/product-reuse 家族的否定，也不将循环合并称作新 X。**

复现：`/opt/anaconda3/bin/python3.12 run.py`（Verilator4.028 `--cc --exe` 后独立 make），再运行 `verify_ledger.py`。前轮目录均只读。
'''
(H/'REPORT.md').write_text(report)
(H/'README.md').write_text('# Native C4 demand sum RTL\n\n固定单 scratch 构造已封口；结论与边界见 [REPORT.md](REPORT.md)，三模式合同见 [resource_contract.json](resource_contract.json)。\n\n- `c4_execution.sv`：mode3 来源追踪、mode5 共同 C4 供数两源强控、mode4 仅实际码归约。\n- `tb.cpp`：原生配置、逐输出检查、背压与不复位重启。\n- `run.py`：完整编译/正式五臂/开发与定向功能回归。\n- `verify_ledger.py`：原生几何与闭式计数核对。\n- `results.json` / `SUMMARY.json` / `benefits.csv`：648 runs、2,488,320 输出全绿。\n')
