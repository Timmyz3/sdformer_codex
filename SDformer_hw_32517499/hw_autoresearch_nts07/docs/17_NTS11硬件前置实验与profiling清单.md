# NTS11 硬件前置实验与 profiling 清单

**版本**：2026-06-17  
**对象**：NTS11bd / NTS11bj 统一全 encoder H60 路线  
**目标**：在正式进入 RTL / 架构图定稿前，把硬件主线需要的数据证据补齐。

---

## 1. 当前已完成的统计

已有中文报告：

```text
/root/private_data/work/sdformer_codex/SDformer/
neuron_experiments/H9_bipolar_self_attention/results/
hardware_stats_nts11_mainline/hardware_stats_summary.md
```

已覆盖：

| 类别 | 状态 | 说明 |
|------|------|------|
| valid825 主指标 | 已完成 | NB0 / NTS07b / NTS09e / NTS11bd / NTS11bj |
| AEE / AAE | 已完成 | NTS11bj ep2 当前综合最好 |
| total_spikes | 已完成 | NTS11bj ep2 为 29.0414G |
| energy | 已完成 | NTS11bj ep2 为 23032.66 uJ |
| H60 block 覆盖 | 已完成 | NTS11 为 12/12，NTS07/09 为 6/12 |
| ATLIF 数量 | 已完成 | NTS11 为 105 = 27 ternary + 78 binary |
| Shiftmax 数量 | 已完成 | NTS11 为 12，NTS07/09 为 6 |
| mixed datapath 审计 | 已完成初版 | NTS11 无 native attention block 估计残留 |
| layer category spikes | 已完成 | Q/K、MLP、downsample、decoder、resblock 分类 |
| stage spikes | 已完成 | S0/S1/S2/S3/decoder/resblock/other |
| hot layer | 已完成 | NTS11 的 downsample 是明显热点 |

当前主线对比：

| 方案 | AEE | AAE | total_spikes | energy | H60 blocks | ATLIF |
|------|-----:|-----:|-------------:|-------:|-----------:|------:|
| NB0 ep59 | 1.4872 | 9.9300 | 44.0488G | 37638.01 | 0 | 0 |
| NTS07b ep29 | 1.4855 | 9.7418 | 36.8003G | 31581.35 | 6 | 34 |
| NTS09e ep29 | 1.4891 | 9.7333 | 38.9417G | 31845.17 | 6 | 34 |
| NTS11bd ep19 | 1.5647 | 9.9213 | 29.1676G | 23108.92 | 12 | 105 |
| NTS11bj ep2 | 1.5159 | 9.9611 | 29.0414G | 23032.66 | 12 | 105 |

---

## 2. 当前 profiling 状态

2026-06-17 GPU 已空闲，已补跑 NTS11 P0 profiling。新增中文报告：

```text
/root/private_data/work/SDformer/hw_autoresearch_nts07/docs/
20_NTS11_P0_profiling实测结果.md
```

新增输出目录：

```text
/root/private_data/work/sdformer_codex/SDformer/
neuron_experiments/H9_bipolar_self_attention/results/
nts11_hardware_p0_profiles/
├── nts11bj_ep2_valid40/
└── nts11bd_ep19_valid40/
```

当前状态：

| profiling 类型 | 状态 | 说明 |
|----------------|------|------|
| valid2 smoke hook | 已完成 | H60 记录 24 次，验证 hook 正确 |
| NTS11bj ep2 valid40 + H60 gate hook | 已完成 | H60 记录 480 次 |
| NTS11bd ep19 valid40 + H60 gate hook | 已完成 | H60 记录 480 次 |
| Token-Time Bundle density | 已完成 short | TTB1/TTB2/TTB4 代理统计已输出 |
| ATLIF activity snapshot | 已完成 short | ternary/binary activity、正负事件率已输出 |
| activation/skip storage | 已完成 short | S0/S1/S2 pre-downsample skip 与 S3 final-stage output 已分开统计 |
| full valid825 + forward hook | 未跑 | 建议等最终 checkpoint 选定后作为论文正式统计 |

---

## 3. P0 必补实验

### 3.1 H60 score / gate profiling

目的：证明 NTS11 的 single Shiftmax token gate 有可利用的结构，而不是只报总 spike 数。

需要输出：

| 指标 | 粒度 | 用途 |
------|------|------|
| TX score mean/std/min/max | block/head/window | 判断 TX 分数动态范围 |
| SC score mean/std/min/max | block/head/window | 判断 SC 是否真正贡献 |
| TX 与 SC 比例 | block/head | 支撑 TX/SC consensus engine |
| fused score 分布 | block/head/window | 决定 score 定点位宽 |
| Shiftmax gate entropy | block/head/window | 判断 gate 是否集中 |
| top-1/top-4 gate mass | block/head/window | 判断 token skip 可能性 |
| effective token count | block/head/window | 支撑 token pruning / bundle skip |
| empty 或 near-uniform gate 比例 | block/head | 判断是否存在无效 attention |

推荐脚本：

```text
profile_h60_gate_stats.py
```

建议输出：

```text
results/nts11_gate_profile/
├── h60_gate_stats.json
├── h60_gate_stats.md
└── h60_gate_by_block.csv
```

最小运行口径：

| 级别 | samples | 用途 |
|------|---------|------|
| smoke | 2 | 验证 hook 正确 |
| short | 40 | 写论文趋势图 |
| formal | 825 | 最终表，可选 |

当前状态：已完成 NTS11bj ep2 和 NTS11bd ep19 的 `samples=40`。正式论文表格建议最终再跑一次 `samples=825`。

### 3.2 Token-Time Bundle density profiling

目的：判断 Bishop 式 Token-Time Bundle 是否能迁移到 NTS11。

统计对象：

```text
bundle = window × timestep × token_group
```

需要输出：

| 指标 | 用途 |
------|------|
| empty bundle ratio | 支撑空 bundle skip |
| low-density bundle ratio | 支撑 sparse engine |
| high-density bundle ratio | 支撑 dense engine |
| per-stage bundle density | 决定 S0/S1/S2/S3 调度差异 |
| per-category density | 区分 H60、MLP、downsample |
| TTB-1 / TTB-2 / TTB-4 对比 | 选择 bundle 深度 |

推荐脚本：

```text
profile_bundle_density.py
```

建议先做三个 bundle 方案：

| 方案 | 定义 | 预期 |
|------|------|------|
| TTB-1 | 单 timestep | 最细粒度，控制复杂 |
| TTB-2 | 2 timestep 打包 | 推荐初始点 |
| TTB-4 | 4 timestep 打包 | 控制简单，但容易混入无效 token |

当前状态：已完成 NTS11bj ep2 和 NTS11bd ep19 的 `samples=40`，TTB1/TTB2/TTB4 均已输出。实测 TTB1/TTB2 有空 bundle 跳过价值，TTB4 过粗。

### 3.3 NTS11bj ATLIF 活性快照

当前缺口：NTS11bj 的 valid825 有主指标，但现有落盘日志缺完整 ATLIF 活性 summary。

需要补：

| 指标 | 用途 |
------|------|
| threshold_mean / max | 判断阈值是否异常 |
| ternary_activity_mean | 判断 Q/K 和 downsample 三值活性 |
| binary_activity_mean | 判断 all_non_qk 活性 |
| ternary_pos_neg_ratio | 判断正负三值是否平衡 |
| zero_pos / zero_neg modules | 判断是否有三值模块失活 |

建议方式：

1. 不重训；
2. 只加载 NTS11bj ep2 checkpoint；
3. 跑 `valid2` 或 `valid40`；
4. 在 eval 结束时打印 ATLIF summary。

当前状态：已完成 NTS11bj ep2 和 NTS11bd ep19 的 `valid40` ATLIF summary。NTS11bd ep19 中 ternary ATLIF activity 约 15.14%，binary ATLIF activity 约 5.64%。

### 3.4 downsample hotspot 消融

当前统计显示：

| 方案 | downsample_event |
|------|-----------------:|
| NB0 ep59 | 2.284G |
| NTS07b ep29 | 2.189G |
| NTS09e ep29 | 2.201G |
| NTS11bd ep19 | 4.416G |
| NTS11bj ep2 | 4.102G |

结论：NTS11 的 downsample 三值路径是当前硬件热点。

需要补至少一个消融：

| 消融 | 目的 |
|------|------|
| downsample binary | 看能否保留统一数据流同时降低 downsample traffic |
| downsample ternary but gate/skip | 看能否硬件跳过低密度 downsample bundle |
| downsample 保持三值但单独压 threshold | 看是否能降低 spikes 且不伤 AEE |

这不是马上进 RTL 的阻塞项，但会影响最终架构是否需要给 downsample 单独设计热路径。

---

## 4. P1 建议补充

### 4.1 Shiftmax 定点精度 sweep

目标：决定 Shiftmax gate 的硬件位宽。

候选：

| gate 位宽 | 风险 |
|-----------|------|
| INT8 | 面积小，可能伤 AAE |
| INT10 | 折中 |
| INT12 | 更稳，面积略高 |
| FP16 参考 | 只作 golden |

### 4.2 gated-K accumulator 位宽

需要估算：

```text
acc_width = gate_width + log2(tokens_per_window) + sign_margin
```

建议从软件中统计输出范围，决定 INT16 / INT20 / INT24。

### 4.3 event SRAM 格式

需要明确：

| 数据 | 格式 |
|------|------|
| binary event | 1-bit packed |
| ternary event | 2-bit packed |
| Q/K event tile | head-major 或 token-major |
| gate buffer | per head/window |
| score buffer | 是否保留，还是 streaming |

### 4.4 layer descriptor 表

每个 logical site 不实例化硬件，而写 descriptor：

```text
stage
block
module_type
input_shape
output_shape
event_mode
threshold_addr
center_addr
engine_id
```

---

## 5. 进入硬件实现前的通过条件

建议满足以下条件再开始 RTL：

| 条件 | 状态 |
|------|------|
| NTS11 主 checkpoint 选定 | 基本完成，NTS11bj ep2 暂优 |
| valid825 主指标对齐 | 已完成 |
| mixed datapath 审计 | 已完成初版 |
| H60 gate stats | 已完成 short |
| bundle density stats | 已完成 short |
| NTS11bj ATLIF summary | 已完成 short |
| NTS11 能耗/面积模型更新 | 待补 |
| 小白版数据流文档 | 已完成初版，仍需继续细化接口表 |
| 模块接口表 | 待补 |

当前建议：

1. 先写数据流和接口文档；
2. 用已补出来的数据决定是否采用 TTB-1/TTB-2 和 sparse/dense stratifier；
3. 补 full valid825 profiling 和 downsample hotspot 消融；
4. 再进入 RTL skeleton。
