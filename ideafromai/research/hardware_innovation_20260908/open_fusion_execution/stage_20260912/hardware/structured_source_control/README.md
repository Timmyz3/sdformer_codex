# 普通连续3/3/4源强对照

**源执行及真实十帧已完成：AEE 1.686686，未达到同集NB0的1.454603。** 停当前免训参数点，保留训练后的普通结构源对照；这不能证明 lifting 独有新颖性，也不是整个残差链结果。

B：现有 dense ordinary 已享完整常量 CSE，但没有本学生的普通结构源。A：迁移旧 dependency 的连续3/3/4支撑，共34个槽，只保留当前 ordinary `original_ordered24 + onepass` 父臂的原 As_q16 系数。旧工作拟合的是 r1.sn2；本次在 r1.sn1 新试，仅借支撑，旧系数、bias、精度均不继承。X：本臂不主张独占创新，作为 lifting 必须面对的普通结构 PSN 对照。

`deployed_constants.npz` 和父组合全部字段逐项比较，**仅 As_q16 改变**，As exponent15、source τ/θ、方向、consumer gain/阈值、Conv2/F、PED R24 和单遍 BN 均不变。`source34_adapter.install(helper, path)` 在当前 ordinary `FixedTemporalForward` 创建之后直接替换定点矩阵，不从 float 重新选择 exponent，不修改原 helper 文件。真实 I24 上独立重建的父门与实际 GPU 捕获两窗0差。

官方 da4ml0.6.0，同普通父臂 `wmc/auto` 整矩阵分解/CSE与固定 `last_use_pressure` 调度：**101个加减节点、124条实际程序、RF峰26个向量**。两级流水/寄存器寿命596个向量0差，1024域角点与4096随机向量的51,200个输出同时对独立整数 dot 和官方 DAIS0差。原 RNE15/sat24 与 inclusive τ 比较的精确整数前像仍可编译为最终门；不是删除必要的数值边界。

| 同范围源阶段 | ordinary dense | lifting40 | ordinary 34项 | 相对 dense 少服务 |
|---|---:|---:|---:|---:|
| corner ready | 516,618 | 432054 | 326,106 | 36.88% |
| interior ready | 771,738 | 645414 | 487,146 | 36.88% |
| interior 压力 | 807,312 | 未测 | 526,596 | 34.77% |

以上是 `execute_source.py` 在原完整集成所用 `IntegratedMachine` 上重新执行的真实输入、RF整数载荷与 SRAM 门写出，非节点数替换拍数。两边同96×8×48 RF、128KiB state、128KiB coefficient、8192B固定指令ROM、1R64/1W64 state和32B/5slot DMA。源系数已在双方的同容量静态ROM程序中，没有外部每帧系数加载。所有 I24 冷DMA/重读、指令取用、依赖等待、门收集/写出已收费。corner读233,280B/写248,832B，interior读348,480B/写371,712B，**与 dense 完全相同**；没有借低发放删除输出或消费者。三例310,080个门值均与新独立gold0差。实际源程序最高使用RF26附近但不缩减共同96向量容量，输出collector仍RF95。

源门明显变化：corner553/77,760个1（父4164，差3993位），interior885/116,160个1（父6670，差6703位）。因此不能把未变父臂的sn2、Conv2/PED输出或完整链服务借给本臂。当前仅给源阶段费用；新参数实际十帧未过质量门，未继续回放改变后的完整消费者，不把此36.88%当可用整链加速。

主代理已用**不改模块的同一公共源RTL**完成两窗×ready/stress四例，对真实I24、新CPU门gold与每个RF写回均0差，累计387,840门位。ready每H8：普通source34为301周期，dense497，lifting410；这些程序是不同新函数，仍须各自精度。[RTL结果](../rtl_source/structured_control_results.json)与[入口](../rtl_source/run_structured.py)另列。RTL边界从SR64接口开始，没有冷DMA/下游消费者，不能与上表CPU集成源阶段的服务槽混作一个倍率。`rtl_inputs/`保留真实两窗输入/新门gold及manifest。

本目录CPU执行标签仍为 **CPU payload slot prototype**，公共RTL是 **Verilator功能/周期原型**；均非VCS/DC/PT/Formality闭环、非PPA。ordinary父源费用来自原集成表的源分段；[最终组合对齐](../final_combo_alignment/README.md)已证明 current R24+onepass 没改该源输入/算术。

最终[真实网络评价](../../algorithm/source34/aee/run.json)：diverse10 AEE **1.686686**，排除首帧后9帧 **1.651881**，均未过对应NB0。actual helper的193,920个Q24值及193,920个门与新CPU函数0差。仅停止本次未恢复的3/3/4遮罩，不能据此杀结构稀疏家族，也不能替代lifting与同预算训练普通结构的比较。

主要产物：[参数](parameters.json)、[编译](compilation.json)、[源服务](summary.json)、[实际执行](execution.json)、[GPU adapter](source34_adapter.py)。编译器沿用[da4ml官方实现](https://github.com/calad0i/da4ml)；本次不新增文献/训练/生产RTL/EDA。
