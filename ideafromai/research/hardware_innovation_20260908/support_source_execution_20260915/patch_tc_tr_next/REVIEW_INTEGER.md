# 固定 TC/TR 整数编译独立审阅

实读 `compile_fixed.py`、PLAN、INTEGER_RESULT，以及实际调用的 `compile_integer_factors.domains`、`factor_reference.spatial_regions/sparse_forward` 和捕获几何。仅用 Python 3.12 做参数/边界小核验，没有重编模型、训练、网络推理或 RTL。**未发现本次固定参数的数值与区域映射错误；它已给出可用于 RTL 的两个新整数函数，不是硬件通过或新 AEE。**

## 映射与交换条件

- 捕获 `source_gate_words[64,864,4]` 的 bit=t；解包 transpose 得 `X[G,T10,P4,K864]`，与当前 U 的 K 轴一致。四份捕获只使用这些词、group_ids 和文件名，没有把旧 Yi/gate 当新函数答案。
- 捕获定义 group_id=y×80+x/4，P4 为同一行的四个水平邻点。区域函数为 floor((group_id//80)/30)，恰好八个 30 行带；最后 group19199 对应 y239/x316..319，不越行。每份固定 64 个 group、每带 8 个，没有按活动率选区域。
- regional masks[8,24] 每 TC 对应连续 J4 个 latent，repeat(4) 正确得到 rank96 的选择；ordinary 重组为八个全开的 J4 TC。decode_tc、紧凑包 tc、恢复 js=4tc+[0..3] 使用同一全局 latent ID，没有把紧凑包序号误当 TR 索引。
- 写成整数式，Z=X·Ui·M(region)，Yi=Z·Vi，Ui_out=Aq·Yi=(Aq·Z)·Vi。M 对同 P 的整个 T10 不变，Vi 的每 h 尺度也不随 t 变，且两条路径没有中间 RNE/截断，所以交换严格成立。这不允许越过后续阈值、time-dependent mask 或中间重新量化；A 在 latent 域先做时须真正存/读更宽 Q。

`sparse_forward` 的布局是 V 按 h 主序，addr=h×R+r，再分 bank=addr%8。**R32/R96 都是 8 倍数，因此同 r 的八个 h 落同一个 bank。** 一个 J4×H8 TR 块有 32 个标量地址，却不能当成一次无冲突八 bank 读取。当前 NumPy 地址参考只证明取值正确；未来 RTL 必须付每 bank 串行服务，或另建并收费的物理重排。stats 中 scalar addresses/products 不是周期。

## 量化、阈值与断开尾

ordinary 原 rank48 文件的 v[32:] 实际全零，live 精确为0..31。独立核对原 U8×2^ue 等于 theta×保存 U、sign×2^shift×nz 等于保存 V；删去16个断开 latent 不改变该函数。旧 masks 全开，重组 J4 也未删有效 mask。普通 R32 的所有执行顺序/硬件分母都应允许这项静态删除，不用空尾充作工作量。

regional 是新量化学生：先 theta 折 U，逐 latent 取 dyadic scale，再 RNE 到 INT8；本次独立核对原始 rounded 值全部在 [−127,127]，clip 实际未触发。V 非零项按固定 log2-RNE 投影到 sign×2^shift、shift∈[−15,0]，不声称这是系数 L2 最优投影。逐 h 选共同指数 eh 后，对齐 Vi=sign×2^(ue+shift−eh)，本次有效 shift 为0..14；整数计算无需中间右移。

BN gain 全正。新整数函数的实数判据精确写作 κ[h]·Ui_out+offset[t,h]>=0，其中 κ=gain×2^(eh−14)，offset=bn_bias×sum(Aq)/2^14+temporal_bias−theta_output。由 Fraction 算 ceil(−offset/κ) 合法。两模型各 960 个阈值独立核验均满足 κ·tau+offset>=0 且 κ·(tau−1)+offset<0，等号和负 offset 无错。该编译不支持未另行处理的负 gain/零 gain，不是动态 BN 统计供数实现。

NPZ 中 int64 `v` 是参考展开系数，不等于硬件 INT64 权重容量。本次可以用 sign+aligned_shift 的 5bit 表示有效系数；若实际存原 `v_shift`，解码还需要 ue/eh 的真实配置。后续资源报告须固定一种编码与加载协议，不能把两种字段混用而免除对齐元数据。

## 任意归约的位宽

独立对八个 mask 重算 z 下/上界与 triangle：
`zabs=max(−sum(min(Ui,0)),sum(max(Ui,0)))`；
`Yabs=zabs·abs(Vi)`；
`Qabs=sum_s abs(Aq)·zabs`；
`Uabs=sum_s abs(Aq)·Yabs`。
结果与 INTEGER_RESULT 全同。任意 source 子集的单 latent 和仍在 z 区间内，后续绝对和界覆盖任意 latent/time 顺序及部分归约；无需假定不同 latent 的源独立。报告较紧的 ordered prefix 则依赖全局递增 latent 顺序，不能用它缩任意调度的 accumulator。

| 固定模型 | Z signed | Q signed | 任意序 Yi signed | 任意序输出 U signed | 输出绝对界 |
|---|---:|---:|---:|---:|---:|
| ordinary R32 | 15 | 30 | 32 | 46 | 34482957466552 |
| regional R96 / active48 | 15 | 30 | 32 | 47 | ≤45763415338464 |

因此公共 48bit 输出算术有余量；ordered Yi30/U44 不能直接替代上表。所有这些整数乘积及归约绝对界小于2^53，FP64 可精确模拟**这份整数函数**；不表示任意保存 FP32 实数权重的 FP64 计算等于原网络 FP32 执行。当前源是任意0/1的保守域，比四份 capture 大，不能把观察到的小值反用于缩位。

## 已完成与未完成

每模型四份 capture 共核对 983040 个 Y 与最终 U/gate；dense masked、实际 TC/TR 地址恢复、A→Q→V 与 V→Y→A 整数输出全等。ordinary 的 Y 对保存 dyadic U/V 的 FP64 参照 RMSE 为0，但 Aq14 改动仍产生2个门差；regional 的 Y 相对 RMSE 约19.72%–19.96%，产生9390个门差。这些是局部量化变化，均不能借旧学生 AEE，也不能由门差比例判整网好坏。

regional 的八个不同 mask、每区48 active 已保留；ordinary R32 是强普通控制，但与 regional 是不同函数，不能仅以 A 域 MAC 次数减半/降到三分之一宣布架构收益。当前计数、mask/payload 字节和完整数组也没有证明有限容量、bank 服务、参数冷装、背压或实际 Y/Q 生命周期。后续应先对每份函数分别比较两种执行顺序，再比较质量已闭合的不同学生。

本步新增的是一份清楚、可复核的整数执行合同；线性算子交换与 TC/TR 都属借入机制，不把这次编译单列整网创新。TR bank 冲突、Q30 驻留和完整终点供数，是实际 RTL 必须收取的费用，当前未被宣称已完成。

