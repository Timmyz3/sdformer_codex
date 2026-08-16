# Local5 首遍精确 Relation Vault DSE 与晋级门槛

## 1. 本轮问题

Local5 的完整投影不是现有 `OUT_DIM=32` 单 tile 回放。四个 stage 分别有 `3/6/12/24` 个输入 head，也有 `3/6/12/24` 个输出通道 tile。完整执行必须同时决定 relation、term 和 Acc32 的循环顺序。

两个直接方案都有明显代价：

1. 输出 tile 外层：只保留一个 Acc32 tile，但每个输出 tile 都重算 score、Shiftmax5 和 relation；
2. head 或 head-group 外层：relation 只生成一次，但不同 head-group 之间要保存和恢复所有输出 tile 的 Acc32 部分和。

本轮先用真实 post-G0 profile100 数据建立强基线和容量模型，不先扩 RTL。

## 2. 被否决的简单方案

简单增加 `G` 个 relation 槽、按 head-group 驻留的方案，在串行前端模型下看似可加速；但给输出 tile 重算基线加入双上下文理想重叠后，relation build 可以与 projection 隐藏。此时分组驻留仍要承担跨 group 的 Acc32 spill/read，且每增加一个 relation 槽都增加 SRAM 宏面积。

在 `256/512/1024 bit/cycle` 部分和搬运敏感性中，该方案没有稳定的面积归一吞吐优势，因此结论为：

- `[模型]` 多槽 head-group relation residency：`REJECT_BEFORE_RTL`；
- 不把多 SRAM 槽、ping-pong 或循环重排包装成 DATE 贡献。

## 3. 新候选：暴露感知的精确 Relation Memoization

### 3.1 数据流

```text
第 0 个输出 tile：
Q/K -> score -> Shiftmax5 -> relation transpose
    -> source-major descriptor -> TCFM5/Acc32
    -> 同时统计 exact service、尝试写 packed relation
    -> service < 450 且容量够：commit
    -> 否则：rollback

第 1..O-1 个输出 tile：
packed relation vault -> exact descriptor replay
    -> TCFM5/Acc32

容量 miss 或 projection 已隐藏 relation build：
原 score/Shiftmax5/relation 路径精确重算
```

每个活跃 source 直接占一个 112-bit 物理宏字：

```text
source_id9 + K32 + 5 x gate9 + valid5 = 91 bit payload
+ 21 bit padding = 112 bit physical row
```

source-id 和 valid mask 直接保存，不依赖 replay 时重建。每个 head 只需一个小型目录项 `{resident, base_ptr, length, service}`；最多 24 个目录项由寄存器保存。

单 head 最坏容量为 450 行：

```text
450 x 112 = 50,400 bit < 512 x 112 = 57,344 bit
```

固定 relation 存储为：

```text
450 x (K32 + 5 x gate9 + valid5) = 36,900 bit
```

固定 relation 的逻辑 payload 位数更小，但需要为全部 450 source 保留固定地址；memoization 把非零 product 的 source 压成连续 112-bit 记录，使多个 critical head 可以共享同一个 512 行宏。收益来自记录数稀疏，不来自跨字 bit packing。

### 3.2 暴露感知 admission

强重算基线允许 relation 前端和 projection 后端双上下文重叠，因此每个 head/output-tile 的理想周期是：

```text
max(relation_build=450, projection_service)
```

当 `projection_service >= 450` 时，缓存 relation 不缩短关键路径，只减少工作；有限 SRAM 应优先给 `projection_service < 450` 的 head。首个输出 tile 已经精确产生全部 term，因此 service count 是观测值，不是 sparsity predictor。

旧 FCSR 使用三行 Q/K buffer，不能把整窗 `H x 450 x 64 bit` 当成已存在的空闲 scratch。此前“用整窗 Q/K 生命周期覆盖大 vault”的假设已否决。

修订候选要求首遍由 FCSR 三行 gate ring 和 source frontier 在线生产 descriptor，使 Direct 物理基线中的 7 KiB 完整 relation-plane 宏可以改作 memoization。candidate packet 顺序写入可回滚区；head 结束后满足 critical/fits 才原子提交，否则恢复写指针。当前 FCSR 与 memoization 尚未形成单顶层，所以只能记为 `[待验证]`，不能写成已实现的零额外 SRAM 结构。

在线状态只需要两个边界指针：`committed_ptr` 指向最后一个已提交 packet，`speculative_ptr` 跟随当前 head 的顺序写。head 严格串行，非 critical 或 overflow head 在启动下一 head 前回滚，所以其临时占用不会永久挤掉后续 critical head。容量模型已按这一在线顺序执行，不再预先跳过非 critical head。

首遍和 replay 的单端口访问按原生 112-bit word 计数。对任意活跃 source 数 `A in [0,450]`：

```text
packet_words = A
projection_service >= 15 + A
packet_words + 1 <= projection_service
```

因此在“FCSR ring 与 vault 宏物理分离”的前提下，一拍最多一次的 speculative write 或 replay read 可以被 descriptor/term service 覆盖。该不等式由单元测试穷举全部 451 个 `A`；端口独立性和随机反压仍需 RTL 验证。

## 4. 结果

可复现入口：

```bash
python scripts/model_local5_relation_vault.py
python -m unittest tests/test_model_local5_relation_vault.py
```

产物：

- `results/local5_relation_vault_dse_20260806/report.json`
- `results/local5_relation_vault_dse_20260806/report.md`

输入为 full-resolution、T450、100 sample、4800 个真实 head-window group。联合 head 容量由同 stage 真实 group 独立 bootstrap，故容量拟合率是 `[模型]`，不是同 window 实测值。

| 方案 | 整帧周期代理 | relation build 减少 | 开放宏面积比 | 面积归一吞吐代理 |
|---|---:|---:|---:|---:|
| 双上下文理想重算 | 1.000x | 0 | 1.000x | 1.000x |
| critical-only，4 KiB | 1.325x | 58.92% | 1.000x | 1.325x |
| critical-only，7 KiB | 1.333x | 60.35% | 1.000x | 1.333x |
| critical-only，16 KiB | 1.334x | 60.86% | 1.098x | 1.215x |

这里的面积只是 Nangate45+FakeRAM45 开放代理：7 KiB 以内沿用 Direct 已计入的一组 relation 宏，尚未计入 FCSR ring、packer、replay 和控制标准单元。它不是 DC 面积。

7 KiB 同容量单变量消融为：`first-fit all` 缓存更多总工作，relation build 减少 64.66%，但周期只有 `1.316x`；`critical-only` 的 build 减少较少，为 60.35%，周期反而达到 `1.333x`。原因是前者会让对周期无收益的 dense head 占用容量，挤掉 `service < 450` 的 critical head。该对照是暴露感知 admission 的直接动机，而不是“稀疏度越高越缓存”的事后命名。

7 KiB critical-only 的分 stage 结果：

| Stage | heads/tiles | 驻留 head 比例 | 周期代理 | relation build 减少 |
|---:|---:|---:|---:|---:|
| S0 | 3 | 75.72% | 1.258x | 50.48% |
| S1 | 6 | 94.78% | 2.370x | 78.98% |
| S2 | 12 | 70.71% | 1.375x | 64.81% |
| S3 | 24 | 45.04% | 1.140x | 43.16% |

112-bit 1RW 在线分账显示，各 stage 的 pack/replay 端口暴露周期均为 `0.000`；这来自上述解析不等式，不是忽略写入。每 synthetic window 的 speculative/discarded word 均值为：

| Stage | speculative write words | discarded words | capacity miss/head-window |
|---:|---:|---:|---:|
| S0 | 187.14 | 156.73 | 0.000 |
| S1 | 118.52 | 52.78 | 0.000 |
| S2 | 933.19 | 788.73 | 0.061 |
| S3 | 2715.18 | 2360.67 | 3.043 |

discarded write 很多，尤其 S3；它们虽然在当前相序模型中不增加周期，却会增加动态能量。后续可以在累计 service 达到 450 时提前终止 speculative packing，但在 SAIF 前不能宣称能耗收益。

`1.333x` 来自逐真实 group 计算 `max(relation_build=450, projection_service)`，不是对均值先取 `max`。稀疏 group 中 relation 前端会暴露在关键路径上，因此 vault 即便面对理想双上下文重算基线仍有周期空间。

## 5. 与外部工作的关系

- Bishop 的 TTB 提供 token/time bundle 的表示范式；这里打包的是 Local5 source topology、K payload 和五方向 gate，并保留 exact fallback。
- Bishop 的 density stratifier 启发按工作量分层；这里不分流到异构核，而用首遍精确 service 判断前端是否暴露，并决定 relation 是否值得驻留。
- Phi 的 pattern/residual 思路启发“只保留有效 pattern payload”；这里将活跃 source relation 写成连续原生宏字，并用 exact fallback 处理容量 miss，不使用预测 codebook。
- Prosperity 的 exact reuse 启发跨执行阶段复用；这里复用单元不是 partial product，而是对 output tile 不变的完整 relation descriptor。
- FLAT/Sanger 的 stationary dataflow 启发驻留操作数选择；这里驻留的是五邻域 relation，并与 output-tile 循环和 Acc32 生命周期联合排序。

候选贡献不能写成“借鉴 TTB”“使用压缩”或“复用 SRAM”。可辩护的本土化主张是：

> Local5 的 relation 对输出通道 tile 完全不变。首遍执行把精确 attention 拓扑和服务代价同时转化为可重放操作数，只给前端暴露的 head 分配有限 relation SRAM；后续 tile 在不搬运跨 head Acc32 部分和的前提下消除关键路径重算，并用容量 miss 的 exact fallback 保持数值语义。

## 6. 当前不能宣称的内容

1. `1.333x` 不是 RTL 实测、端到端 FPS、ASIC PPA 或 EDP；
2. 不能把 bootstrap 驻留率写成真实同 window p99；
3. 不能把 7 KiB 以内写成零面积；只能说 SRAM 宏容量代理不增加，且这一前提依赖 FCSR 先替代完整 relation plane；
4. relation build 减少不能直接换算成功耗；
5. 当前只有 112-bit 1RW 相序模型，还没有 pack/replay RTL、随机反压或 Acc32 miter。

## 7. 晋级门槛

按优先级执行：

1. 新 rank-1 profile 导出同一真实 window 的完整 head 集合，报告 packet occupancy、critical-head 命中和 fallback；
2. 先以真实 T450 证明 FCSR 三行 source frontier 可替代完整 relation plane；
3. 冻结 7 KiB relation 宏的候选写入、回滚和 replay 端口相序，完成冲突表；
4. 实现叶级 pack/replay，不先连接正式 scheduler；
5. 以重算路径为金参考，覆盖 450 source、容量边界、exact fallback、随机反压和 Acc32；
6. 再连接完整输出 tile 循环，与双槽理想重算基线同约束比较总周期、SRAM 事务和 SAIF；
7. 只有面积归一吞吐非负且后续 DC/PTPX 的 EDP 有收益时，才晋级为 DATE 主贡献。

## 8. 双线边界

本轮没有停止 Motion。Motion 继续等待真实 full-resolution T450 profile，并维护 SCS/NMF/DCTF/TESC 回归。暴露感知 memoization 的原则可在 Motion 上重新评估，但 Motion 的复用对象应是 NMF/SCS term 或 gated-K 目录，而不是 Local5 五邻域 relation；必须用 T450 的服务时间和存储容量单独判断，不能直接套用 Local5 的 `1.333x`。

## 9. 第一轮 DATE 独立评审与整改

独立审稿人对修正后的 memoization 初稿给出 `2/5 Reject`，认可 relation/output-tile 不变性、critical-only admission 和 exact fallback 的组合有架构潜力，但指出初稿仍把 `service` 和 packet size 当作 head 开始前已知，并没有支付 speculative write、rollback、112-bit 宏对齐和 1RW 端口代价。

本轮据此完成以下整改：

1. admission 改成严格在线、head 串行的 `committed_ptr/speculative_ptr` 模型；
2. 非 critical head 也先发生 speculative write，head 边界才 rollback；
3. packet 最终化简为每个活跃 source 一个原生 112-bit 宏字，无跨字 bit-reservoir；
4. 显式记录 speculative、discarded、capacity miss 和 pack/replay 端口 stall；
5. 穷举证明 `active_sources+1 <= 15+active_sources <= service`，所以独立 vault 宏的一次 1RW 访问可被当前 service 覆盖；
6. 明确 7 KiB 宏只有在 FCSR 替代完整 relation plane 后才能专用于 memoization，FCSR 与 memoization 仍未 RTL 闭环。

整改后 `1.333x` 基本不变，但证据等级仍是 `[prof]+[模型]`。其含义从“忽略 pack 访问的理想缓存”收紧为“在 112-bit 单端口、顺序 head、显式回滚条件下无端口 stall 的周期模型”。
