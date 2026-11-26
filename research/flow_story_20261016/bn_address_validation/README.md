# BN 局部归约的地址、carry 与 bank 回压验证

**本轮关闭了局部归约中“奇数 carry 在哪里、多通道如何寻址、端口冲突如何等待”的实现空白；没有得到完整 BN 或整网周期。** 代码不运行模型、不读权重/GT、不占 GPU。它是 [前轮数学 DAG＋抽象端口模型](../../flow_story_20261015/currentstat_bn_cycle/README.md)的独立下层验证，旧源码和旧敏感性结果保持封存。新模型没有把局部地址日程直接加到旧周期上。

输入形状来自 [54 处外部 current-stat BN 的冻结账本](../../flow_story_20261014/damage_allocation_probe/hardware_contract_complete_bn_audit.json)，SHA `9a92bd764878d36b5ca67af8ef0a39806830e755dd0f9a91217c81e165e27567`。其中 15 处原生通道主序、39 处原生通道连续；共 19,056 个统计域、976,584,960 个输入元素。这是形状来源，不表示本次执行了全部这些元素的 BN。

## 已实现的具体动作

[address_model.py](address_model.py)使用两个互不重叠的 scratch 区。样本优先、通道连续的第 `s,c` 个值在 `base+s*C+c`，bank 为 `address mod banks`，row 为整数除法。每一级对 `(2p,c)` 和 `(2p+1,c)` 实际读 SRAM、等待 add、写另一块的 `(p,c)`；奇数剩余样本经过单独的 **read→copy→write** 放到下一块的连续位置。全部本级写退休后才换区，源区在所有读取完成前不会复用。无隐式 transpose、carry 搬移或免费 gather。

每个事件保留实际地址、bank、row、issue/retire、所属操作、数值和样本覆盖标签。读取延迟、写入延迟、add 的 lane/II/latency、copy lane/latency、每 bank 读写端口、共享读写端口都明确配置。操作队列有限；每个已准入操作保守预留 3 个寄存器 word，直到最后事务退休。被拒绝的 bank/算术请求重试，记录的是**被拒绝请求次数，不是可相加的 stall 周期**。一次 tick 先处理退休，再发射；已就绪写优先，操作顺序固定。此日程可行不等于最优。

默认参数均是人为敏感性条件：4,096-word 区×2，共 8,192 word SRAM；16 bank、每 bank 1R＋1W；read latency 2、write latency 1；8 add lane、II 1、latency 3；8 copy lane、latency 1；最多 32 个操作、96 word 寄存器。没有 SRAM 宏、FP32 单元、频率或 DC 标定。

## 已运行的证据

[8 项 CPU 检查](cpu_checks.json)通过，包括 45 个奇偶长度×多通道组合的精确整数求和 golden、carry 目的地址、有限容量/队列、冲突回压、配置不变和故意破坏事件的拒绝。数据中每通道带独立覆盖标签：禁止跨通道混合、重复或丢失样本。独立事务回放不调用调度器；重新检查 bank/端口、II、每个 src/dst、读值版本、操作依赖、读前覆写、级间屏障及准确的寄存器生命周期。整数 golden 使用独立逐通道求和；**不验证 PyTorch 原生 BN 的 FP32 归约顺序或误差**。

[实际布局结果](actual_layout_address_results.json)覆盖全部 54 条 BN 记录对应的 **15 类真实 full/tail chunk 形状**，逐类运行地址日程和事务回放；另外枚举核验 256 个首/末原生连续段、首/末通道及 full/tail 段的实际 stride 源地址，验证无重复且确为连续区间。没有遍历全部 976,584,960 个地址。每类 trace 的 SHA 和重跑参数均保留；[5×3 的完整 trace](odd_5x3_full_trace.json)可直接检查 `5→3→2→1` 的两轮奇数 carry 压紧。

以实际出现的 `5 samples×768 channels` 局部块为例：3,072 次 add、1,536 次显式 copy、7,680 次归约 SRAM read、8,448 次 SRAM write（含初始装载）、768 次结果读出。默认日程为 1,162 个**合成参数局部周期**。这些事务是该 ping-pong 算法的成本，不能视作所有 BN 实现不可避免的流量。

| [已运行敏感性](synthetic_sensitivity.json) | 5×3 玩具块 | 5×768 实际 chunk 形状 |
| --- | ---: | ---: |
| 默认 16 bank、独立 1R＋1W、32 个操作 | 21 | 1,162 |
| 每 bank 读写共享 1 个端口 | 21 | 1,255 |
| 仅允许 1 个操作、3 word 寄存器 | 117 | 33,024 |
| 仅 4 bank，其余同默认 | 28 | 3,086 |

表格只说明地址/队列/端口会限制同一局部日程，**不是压缩收益、完整 BN 周期、整网加速或 PPA**。初始输入值假定已可提供；输入/结果的逻辑外部 word 数有记录，外部 DMA 和访存延迟未定价。

## 与本线实验的关系和剩余工作

R11 各普通/动态 MLP 策略都保留同一外部 BN 图；本组件必须同等提供给 dense、固定保护、普通评分和光流评分基线。不能把普通 ping-pong 归约优化写成光流专属贡献。更省搬移的原址树、更多独立通道调度、输入复用和 producer/consumer 融合仍是普通基线可用的机会。

下一步才能连接跨 chunk 的统计累加寄存器、mean/variance/rsqrt/normalize 的完整实际数据流，以及 MLP/attention/conv 共享资源和有限输入/输出 buffer。仍需验证实际 FP32 数学顺序或受控数值近似、SRAM 宏/FP32 单元参数、RTL/DC，以及在同一精度图和普通基线下计算净 PE/延迟。当前 `full_BN_cycles`、`full_network_cycles`、硬件收益全部为 `null`；10% 净 PE 门未验证。

```bash
# 重新运行检查（不覆盖冻结 receipt）
cd research/flow_story_20261016/bn_address_validation
python3 -B -m unittest -v test_address_model
# 新文件名，避免覆盖证据
python3 -B address_model.py --config example_base.json --output NEW_ADDRESS_RESULTS.json
```
