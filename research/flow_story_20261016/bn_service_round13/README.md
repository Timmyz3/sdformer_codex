# R13：当前统计 BN 的完整串行服务合同

**这轮把 54 处当前统计 BN 的输入重读、跨 chunk 统计、地址归约、系数生成和归一化连成了一个可执行、可复算的完整服务合同；没有形成可用于性能比较的高吞吐 BN 引擎，也没有关闭整网周期分母。** 该合同刻意采用已完成 RTL 所支持的“一个未退休归约命令”边界，帮助识别缺项和定位必须改进的执行瓶颈。不能把这个保守日程当作普通基线来放大新方法收益。

来源是冻结的 [54 BN 实际形状账本](../../flow_story_20261014/damage_allocation_probe/hardware_contract_complete_bn_audit.json)，SHA `9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567`，以及 [R12 部署成本边界](../ordinary_recovery_round12/DEPLOYMENT_COST_BOUNDARY.md)。本轮没有执行模型、读取 GT/权重、使用 GPU/SSH，或者修改旧冻结实现及结果。

## 具体执行了什么

[service_model.py](service_model.py) 使用 population variance 的 centered two-pass 顺序：

`sum(x) → μ → sum((x−μ)²) → v → rsqrt(v+eps) → a=γ·invstd, b=β−aμ → y=ax+b`。

每个通道统计域先完成均值，再读取第二遍输入得到方差，生成并保存 a/b 后才允许第三遍输入归一化。第一 chunk 的 partial 直接保存；其余 chunk 显式读取旧 partial、加法并写回。均值、方差、a、b 占用独立的四组有界 state 地址。无 running-stat 更新。

归约工作区是两个不同的 4096-word scratch：每个 pair 真正执行 `src0 read → src1 read → add → dst write`；奇数 carry 真正执行 `src read → copy → dst write`。目标写退休后才接受下一个命令；每一级的所有写退休后才能交换区。输入原值不会留存在 scratch：第一遍归约已经破坏它，所以第二遍和归一化确实从外部重新读取，没有免费原值副本、免费 transpose 或免费 gather。

原生通道连续的 39 处 BN 一次分组全部 C；通道主序的 15 处逐通道处理。本轮每个外部事务仅一个 word，按真实 shape/stride 计算地址，因此一个 chunk 允许越过原生空间行边界。该选择产生 **11 类** full/tail chunk；旧模型按连续段切分得到的 15 类属于另一个调度，不能混称。每个外部地址先由 dense-permutation stride 检查证明其合法映射，再由样本/通道坐标生成，未把 channel-major 数据假设成 channel-contiguous。

输入、γ、β、输出分别占 `[0,E)`、`[E,E+C)`、`[E+C,E+2C)`、`[E+2C,2E+2C)`，输入输出不会互相覆盖。所有输出写入完成才结束 BN。最大一层需要 294,912,096 word 外部空间；假定容量为 300,000,000 word，均按 FP32 的 4 B/word 解释。这里假定该层输入已存在外部存储；上游生产者写入及下游消费者不是本服务合同的一部分。

## 有限资源和时间假定

| 资源 | 明确采用的合同 |
| --- | --- |
| scratch | 两块 ×4096 word，16 bank，bank=`addr mod 16`，每 bank 1R＋1W；读 2 周期，写 1 周期 |
| scalar state | 4096 word，1R＋1W，各 1 周期；真实最大活动 3072 word |
| operand registers | 保守预留 8 word；所有事务/算术严格顺序，pair 需要两个操作数与结果；系数阶段还保留 a，最大逻辑活跃值不超过此容量；未实现 RTL 寄存器分配 |
| 外部存储接口 | 300,000,000 word 有限空间；单 word、单 outstanding、读/写各 33 周期，不重叠；这是接口延迟假定，不是 DRAM 行/突发/刷新模型 |
| FP32 add/sub/mul | 各 1 lane，II=1、latency=4；每次等待退休，实际没有利用流水并发 |
| rsqrt | 1 lane，II=8、latency=16；实际数值 IP/异常处理待核 |
| copy | 1 lane，II=1、latency=1 |
| 控制 | config/finish、命令 accept/retire、两种统计屏障各收费 1 周期 |

全部事务严格顺序，事件的下一次 issue 等于前次 retire。因此任何 bank 不可能超订阅，单 outstanding 不需要无穷队列；也因此该合同极其保守。状态内存端口并没有沿用旧 BN 事件模型里的免费标量广播。完整配置保存在 [实际形状结果](actual54_serial_service.json)。

**所有延迟/II 都是本轮声明的假定，均没有新的 SRAM 宏或 FP32 单元 DC 证据。** 已完成的 [bn_reduce_addr RTL/DC](../bn_address_rtl/README.md)只支持相同地址/退休协议，不包含上述存储、FP 数据通路、跨 chunk 状态或 DMA。本轮也没有对新完整消费者和原 RTL 做逐拍协同仿真；不能声称模型周期经过 RTL 测得或完整模型已经综合。

## 已执行检查与真实形状结果

[六项 CPU 检查](cpu_checks.json)全部通过：

- 两种布局、奇偶长度、单样本、多通道、跨 chunk 的 **48 个完整 BN 小例**，共 65,618 个展开事件。独立事务回放核查时间连续性、地址/容量、存储值版本、命令退休、FP32 标量运算结果以及完整输出覆盖。
- 独立 FP64 数学 BN 参考与这些小例的最大绝对输出差为 **3.411397182517817e−7**。这是有限合成数据上的顺序误差检查，不是 PyTorch 原生 BN FP32 等价证明，也不覆盖真实激活的数值范围。
- 常值输入、故意损坏读值/时间事件的拒绝、容量不足和不支持 stride 的拒绝、参数变化不改变事务计数。
- 全部 54 条实际 BN 共 19,056 个统计通道、976,584,960 个元素；11 类实际 chunk 对应 36,466 个 pair、3,368 个 carry 逐条检查通道覆盖标签与样本集合，证明该类展开没有重复、丢失或跨域相加。
- 真实 54 条记录的完整结果采用确定性重复 motif 的精确计数，**没有对 976,584,960 个实际值逐个执行数值 BN，也没有导出数十亿条事件**。48 个小例检查展开执行与压缩计数/周期完全相等；大形状还利用 stride 置换和 chunk 分区证明地址覆盖，并抽查首/中/末坐标。

[独立阶段分解](actual54_phase_schedule.json)还逐条给出全部 54 BN 的首组 mean-ready、normalize-ready、完整退休时刻和重复组起点公式；`derive_phase_schedule.py`不导入服务模型，按四阶段重新推导，全部操作数与周期合计完全匹配。通道主序按每个通道独立完成三个读取阶段，再复用 state；不要求无关通道全部统计结束才允许已完成通道归一化。

完整小例见 [9×3 全服务 trace](toy9x3_full_service_trace.json)。该例 region 容量改成 12 word，强制 4＋4＋1 样本分块，检查跨块累加和最后单样本；不是 4096 容量下的真实模型 trace。

| 全 54 BN，按指定串行服务执行 | 数量 |
| --- | ---: |
| FP add＋sub | 3,906,339,840 |
| FP multiply | 1,953,246,144 |
| rsqrt | 19,056 |
| 外部输入读取 | 2,929,754,880 word（3E） |
| 外部 γ/β 读取 | 38,112 word（2C） |
| 外部输出写入 | 976,584,960 word |
| scratch 读取；scratch 写入各自 | 3,963,387,072 word |
| state 读取 | 2,975,364,912 word |
| state 写入 | 45,648,144 word |
| 显式 carry copy | 102,638,208 |
| 归约命令 accept；retire 各自 | 2,010,217,152 |
| 54 个独立 BN 严格串行的条件周期之和 | **171,384,347,670** |

三项算术计数与冻结原生几何账本逐项相等。scratch/state/控制是所选具体日程新闭合的费用，不能当作最优 BN 的必需成本。

**171,384,347,670 不是芯片测量、整网周期、最强 dense 下界或论文性能数字。** 该日程中单 word 外部访问无重叠，单命令归约完全放弃并发，因而极慢。不能用其大分母来宣称压缩价值小，也不能拿它与旧 626,142,077 抽象端口周期作“新硬件加速”比较；两者资源和执行合同不同。当前完整整网周期、净 PE 收益、延迟收益仍为 `null`。

[参数敏感性](sensitivity.json)在保持全部事务量不变下检查外部 read 65/1 周期、add 8 周期和 rsqrt 32 周期。它用于检验可复算性以及指出当前串行瓶颈，不是硬件参数测量或选型推荐。

## 对主线和下轮硬件的直接约束

本轮得到一个明确限制：**不能给已综合的单命令地址控制器配上旧多 lane、32-operation 模型的周期，再把二者宣称为同一已实现引擎。** 单命令消费者的依赖必须收费；要利用 PE 并发，必须真正做出多操作队列、operand/return 标识、bank 仲裁、跨 chunk 状态端口和对应退休屏障，或者采用另一个经过验证的并行归约方案。这不是光流专属增量，而是所有普通和光流评分方案共同需要的执行底座。

下一步只有两项值得沿本线继续：

1. 用同一组实际 scratch/state/外部端口和已表征 FP 单元，替换这里的单事务调度，落实普通并行、burst/缓存、输入复用以及 producer/consumer 融合。当前事务 trace 可以作小例 golden，但不能预先保留本轮极慢周期当 baseline。
2. 将 `producer-ready → statistics-done → normalized-output-ready → consumer` 接回完整 attention/MLP/conv 图，并与 PSN/FC/pool/代表恢复使用同一资源实例。比较相同执行底座上的 native/FF、固定 coarse 和未来评分策略，之后才能判断 10% 净 PE 与 8% 延迟门。

尚未关闭：真实 FP32/rsqrt 数据通路和数值等价，实际 SRAM/DRAM 参数与外部突发规则，并发 controller RTL/DC，跨模块缓冲寿命/拥塞，普通融合强基线，完整网络图日程，以及任务预测器自身周期。这里没有额外新增与主线无关的 RTL 组件。

## 复现与初次失败记录

```bash
cd research/flow_story_20261016/bn_service_round13
python3 -B service_model.py --output NEW_ACTUAL54.json
# test 脚本会拒绝覆盖 cpu_checks.json；复制本目录到新实验路径后运行：
python3 -B test_service_model.py
```

初次测试结果保留于 [cpu_checks.initial_failed.json](cpu_checks.initial_failed.json)：一个断言把“从未发生”的 Counter 缺项与显式零项当成不一致，另一个沿用了旧连续段调度的 15 类 chunk 预期。修复是按非零项比较并根据本轮逐 word 合同列出实际 11 类；未改 BN 数学、数据或资源门槛。随后显式分离输入/参数/输出外部地址空间并增加容量检查，最终六项检查重新通过。之前通过的中间检查也保留为 `cpu_checks.before_external_address.json`。
