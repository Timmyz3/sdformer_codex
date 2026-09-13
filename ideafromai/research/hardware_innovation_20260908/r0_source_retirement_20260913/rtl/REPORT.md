# 固定单 scratch 的 C4 支持码归约：功能通过，未超过共同供数强控制

2026-09-13。完整 [结果](results.json)、[汇总](SUMMARY.json)、[费用 CSV](benefits.csv)、[资源合同](resource_contract.json) 已封口；本目录不继续改变构造策略。

本次完成了真实原生 C96/N96/T10、4×4→2×2、3×3 全 K864 线性叶的三模式 RTL。**648 runs、2,488,320 个 signed32 输出全匹配；6,480 项独立事务/完整周期核对全通过。** 固定一个 signed18 scratch 逐实际支持码构造的 mode4，在五个真实分支均慢于共同 C4 供数的 mode5，差 7.53%–7.90%。mode5 比旧 mode3 快 5.54%–6.34%；这是供数、暂存和消费者循环合并的 A 收益。

## 同硬件对照与实际机制

[前一轮 mode3](../../r0_execution_trials_20260913/consumer_enumeration/REPORT.md) 原样保留，两源时间字沿原生源坐标枚举消费者，必要时构造 WA+WB，每对分别更新 psum。新 mode5 一次读同坐标的四个 T10 源字、四个按需 W 向量，共用一次合法消费者枚举，但仍按两对分别 psum 读改写。最后 W 读取直接准备第一个非空 pair，第一对末直接启动第二个非空 pair；不存在人为的 pair 选择拍或空对扫描。两者所有 source/W 请求、pair_sum 次数、psum 更新逐 fixture 相同，旧 mode3 周期也逐项复现。

mode4 继承 mode5 的四源/四 W 供数，RTL 从四个源时间字产生十个 4bit 码，并只枚举出现的非零码。每个目的 `(og,p)` 下，码 k 先复制首 W 到唯一 signed18×8 scratch，支付一拍；其余项经共同八条 32bit 加法链逐项加入，支付 popcount(k)−1 拍。随后只更新属于该码的 t，完成后复用 scratch。没有十五项和表、离线支持码、TB 中间和、预 gather 或额外 psum 端口。原生 geometry/origin、静态 mask、全部循环、背压、clear 与 drain 均在 SV。

## 主实测：八块之和，command0、无外部背压

| 同函数 mask | mode3 两源枚举 | mode5 C4供数两对 | mode4 单scratch码归约 | mode4 比5慢 | source 10bit词 | W 128bit词 |
|---|---:|---:|---:|---:|---:|---:|
| dense | 415,024 | 391,120 | 422,008 | 7.90% | 9,600 | 41,088 |
| block_magnitude25 | 323,132 | 303,052 | 326,364 | 7.69% | 9,600 | 30,977 |
| cin_magnitude25 | 320,116 | 302,380 | 325,960 | 7.80% | 7,200 | 31,560 |
| cin_fullcost25 | 312,796 | 294,316 | 316,660 | 7.59% | 7,200 | 30,840 |
| mixed_retirement25 | 321,196 | 300,848 | 323,513 | 7.53% | 9,600 | 30,765 |

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
