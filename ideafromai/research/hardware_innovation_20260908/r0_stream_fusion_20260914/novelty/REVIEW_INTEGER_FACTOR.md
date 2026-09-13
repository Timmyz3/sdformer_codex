# 窄整数 R8 因子：独立静态审阅

2026-09-14。**最终共同 `k_live` 版本未发现阻断性的算术、地址或共用数据通路问题。mode7 对同函数 expanded21 direct 的真实八块核心周期减少 53.07%，是值得保留的完整强 A；尚无独立 X 证据。** 本核不是 D1 中“按系数码分桶”的实现：`F_ZADD` 将八个 Q1 码分别加到八个 rank 累加器，没有系数直方图、分桶 popcount 或组和表。mode7/8 分别体现常规 V 驻留与 psum 驻留。

阅读范围：完整 [SV](../integer_factor/integer_factor.sv)、[TB](../integer_factor/tb.cpp)、[PLAN](../integer_factor/PLAN.md)、[量化/fixture 生成](../integer_factor/prepare.py)、[运行脚本](../integer_factor/run.py)、[因子定义](../integer_factor/definition.json)、[最终 SUMMARY](../integer_factor/SUMMARY.json)、[完整结果](../integer_factor/results.json)及[新链质量记录](../data/integer_factor_diverse10.json)。另对当前 factors.npz 做只读的零列、零向量、native fanout 与位宽核查。本审阅没有重新运行 RTL/GPU；168 runs 的功能结论来自实现方已执行并保存的结果，独立审阅负责源码与数据之间的对应关系。本文不替代另一代理的完整事务账本。

## 1. 被验证的函数及位宽

`K=864, R=8, N=96, T=10`，入口是原生 C96×4×4 的十时间位源字，叶输出为 N96×2×2×T10。因子模式在 SV 内按卷积 tap 聚集输入，TB 只装入合法源/常量并比较 int64 生成的 gold。

`z[r,p,t] = Σ_k Q1[r,k] g[k,p,t]`

`P[o,p,t] = Σ_r Q2[o,r] z[r,p,t] = Σ_k E[o,k] g[k,p,t]`，其中 `E=Q2·Q1`。

模式 6/7/8 共享这个精确整数函数，没有 latent 中间舍入。网络值还需 `P·output_scale[o]`；该缩放、BN/残差/PSN 后继不在本 RTL 周期内。新 Q1/Q2 相对原 W 或浮点 SVD 有损，不能把“同函数精确”扩大到原网络函数。

|值|SV 表示及必要上界|静态结论|
|---|---|---|
|Q1|signed3，合法码 −3…3|符号扩展至32后加到 z；未使用 −4|
|z|signed13，`|z|≤864×3=2592`|`F_ZREAD` 与 `F_ZADD` 分拍，写回13位不丢有效位|
|Q2 / 乘积|signed16；`16×13→signed29`|两个乘法操作数均显式 `$signed`，乘积再符号扩展至32|
|expanded E|signed21；一般界 `|E|≤8×32768×3=786432`|实际范围 −216576…248830；没有截回旧 W16|
|两父和 / 四源和|signed22 / signed23|各级加法先在32位完成，再取足够宽的结果；负数扩展同步加宽|
|P及部分累加|signed32；`|P|≤8×32768×2592=679477248`|直接与因子次序的部分和也受此绝对和界约束，无隐含饱和/RNE差异|

新增 `max_positive/max_negative` 使用全 ±3、Q2=−32768 和全一源，gold 分别达到 ±679477248；`zero_factor` 覆盖全部 Q1 列无效。最终 14 fixtures×3 modes×2 背压×2 连续命令 = **168 runs，645120 个 raw 输出比较通过**。这既包括真实八块，也包括零/全一/角边界及上述符号上界；合成例仅用于功能边界，不作真实速度收益。

地址核查：Q1 的 `k=c×9+ky×3+kx` 与 Python patch 展平一致；Q2 装入顺序为 `(og,r,lane)`，SV 读取 `v_mem[lane][og×8+r]`；latent 行为 `fp=p×10+t`，最终 psum 行为 `og×40+fp`。直接路径的 `dest_needed` 用正在选择的目的 p 求 k，后续 `c4_effective` 用已锁存 p；非阻塞状态切换之间一致。被 k_live 过滤的 W 保留旧寄存器值不会泄漏，因为 pair 需求、父和构造、逐 t merge 和更新均使用过滤后的源 mask。

## 2. 共用资源与最强零控制

数据通路实际只有八处 `add_result=lhs+rhs`，以及八处 signed16×signed13 乘法。WA+WB、父和 merge、Q1 累加与后因子 MAC 的加法都走这八条32位链。mode6 同样处于这份拥有乘法器和全部状态的 SV 内；这支持当前运行时合同的公平比较，**不证明分别裁剪后的 mode6 与 factor 芯片同面积**。

|资源|本核实际权限；三个模式共同拥有|
|---|---|
|源|1536×10bit，一拍至多一个原生十时间位字；padding 在 SV 判定|
|expanded W|8 个21bit bank，共同行地址，每个读取拍共168bit；配置接口256bit，不是旧128bit W|
|Q1 / Q2|Q1 为8×3bit同址向量；Q2 为8×16bit同址向量；与 expanded W 的发起状态互斥，均受 `weight_allow` 限制|
|latent|8 bank×40 row×13bit = 4160bit；第一阶段同址8读或8写，第二阶段从 `fr` 指定的一个 bank 读13bit标量；**没有同拍“8向量读再加1标量读”**。零判断复用同一标量地址，不构成额外独立读口|
|psum|8 bank×480 row×32bit = 122880bit；mode6、mode7 的读/更新分拍；mode8 在本地八个 psreg 中累加后一次写入；drain 共用256bit输出|
|附加状态|C4源/权重暂存、两个父和、scratch、Q/V/z hold 与标量选择等均在共同 SV；地址计算、优先选择及跨 bank mux 的时序/面积尚未物理测量|

存储声明是可综合数组及相应 mux/使能，并非已绑定某单口 SRAM 宏。`source_allow/weight_allow` 模拟本地读许可等待；获准的该拍读取本地数组，不代表已测外部 CR 请求/响应协议、DDR 或 NoC。当前结论不包含 Fmax、PPA 或完整 r0 block。

共同 `k_live[864]` 随配置实际装入，定义为 `any_r(Q1[r,k] != 0)`。只读复核得：

- 死列仅 k=580、583；expanded 的24个整 N8 行均来自这两个列×12组，**没有额外整行零未给 direct 跳过**；Q2 没有全零 N8 向量。
- mode7/8 在 `F_SOURCE` 前跳死列，mode6 在目的 tap 对应的 `dest_needed/c4_effective` 上取消该列 W 和 psum 义务。真实八块的 source 读取分别为23364与9600：因子路径按 k/p 再聚集源，额外读取已付费，未把 im2col 输入当免费预生成。
- 枚举每个 native `(c,y,x)` 的合法 2×2 目的 fanout 后，真实 Q1 **没有任何整体无消费者的原生源字**。故 direct 仍读取原生源，不构成真实分母的遗漏。全零因子的合成例中 direct 没有进一步提前退休全部源，mode7 也仍会取96个 Q2 向量；两者都不是全零专用最优布局，该例只证明功能。

`k_live`、源和常量必须完整配置且相互一致；RTL 不在线验证用户提供的 E 是否等于 Q2Q1，也不从任意 E 推导额外零行。现有生成器和 fixture 满足这一合同。不能把不一致配置视为受支持函数。

## 3. 实测的含义与边界

下表直接引用最终 SUMMARY 的真实八块、command0 合计，不将零/全一或连续第二命令重复纳入收益。每块完整输出3840个值。

|同函数模式|无背压核心周期|有背压核心周期|无背压 W 向量读|无背压 psum 读 / 写|
|---|---:|---:|---:|---:|
|6 expanded21，native父和direct|370240|388015|40992|70692 / 70692|
|7 窄整数R8，V驻留|173739|178445|2726|21396 / 21396|
|8 窄整数R8，psum驻留|180651|194168|19514|3840 / 7680|

mode7 相对 mode6 核心周期减少 **53.07% / 54.01%**（无/有背压）；各块加实际1536次源装入及1次origin装入后为 **51.37% / 52.35%**。常量稳态驻留是这里的边界。TB 首命令之前真正配置14017拍，包含 source、origin、block mask、expanded W、Q1、Q2 和 k_live；第二命令没有再次配置，JSON仍打印同一 setup 值，不能再次收取。冷启动若按共同数组全部装入，应另加这一实际配置费用，不能只报紧排位数。

mode8 降低 psum 流量，但 Q2 读取从768增加到17556（八块合计），核心反而比 mode7 慢3.98%，有背压慢8.81%。这是有用的调度取舍实测；不能因为 psum 单项更低而称 mode8 更优。Q1/Q2 的权重缩减以及 latent 变成连续整数之后的读取/乘法，均已出现于 SV 和计数。

TB 在每 fixture 内不 reset 运行两次同一模式，逐拍验证输出在背压期间稳定、地址连续且与 gold 相符。它未覆盖不 reset 时切换 mode 或中途改配置；协议限定只在 IDLE 配置，数据缓存的跨模式运行仍属未测覆盖。测试没有覆盖独立 SRAM 宏的 read-during-write 语义，当前状态安排未依赖同拍读写。

[质量记录](../data/integer_factor_diverse10.json)已完成十帧516735个有效像素，AEE frame mean **1.35332576563 < 历史 NB0 1.45460286107**；不训练、不扫格式。整数 z/P 在 FP64 中计算并检查，无中间RNE；随后 FP64 scale 转原浮点消费者。仍是 `fullnet_bittrue=false, valid825=false`。RTL 使用旧3090八块，AEE使用本轮A800链；质量记录明确两份源有19/122880 bit差异，不能写成同一捕获逐位贯通。

## 4. 新颖性差分与结论

[UCNN §III](https://www.kartikhegde.net/media/UCNN_ISCA.pdf)已给重复权重的输入分组与部分和共享；[SmartExchange §III–IV](https://arxiv.org/pdf/2005.03403)已给离散系数/连续基、稀疏索引和基驻留；[StrassenNets §2](https://proceedings.mlr.press/v80/tschannen18a/tschannen18a.pdf)已给离散前后加法与连续乘积项。这些是此前精读的最近邻，阅读范围与未复现模块见本阶段 [来源表](source_table.csv)。本次没有为了维持数量重新声称新读全文，也没有声称复现上述完整加速器。

当前实现的实质是：已过质量门的昂贵 r0 被固定为小整数 R8 因子，在同函数 expanded21 parent direct 与两种完整收缩顺序之间作了真实硬件比较。这个 A 有明确收益且质量未借旧 SVD 冒充，足以继续保留；它比只磨父和的几百分点更值得作为后续接口底座。

**X 仍未成立。** 源按 k 发起、窄 latent、零跳过、V/psum 驻留均可由普通因子执行表达。原 D1 提出的按码分桶、或者进一步限制真实 T10 上整 Q2 字的需求并集，本核未实现，不能借本次53%归给它们。若后续选择这条做一次恢复训练或表示改造，必须继续以当前更快的 mode7、相同零过滤和相同完整输出义务为强 A。当前证据只淘汰较差的 mode8 布局作为首选，不淘汰 psum 驻留家族，也不为论文接收率作保证。
