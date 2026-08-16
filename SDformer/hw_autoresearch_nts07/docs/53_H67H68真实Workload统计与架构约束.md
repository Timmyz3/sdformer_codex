# H67/H68 真实 Workload 统计与架构约束

**日期**：2026-07-13  
**对象**：H67 Motion-XOR TTX、H68 Castling/TTX 部署图  
**原始数据**：H67/H68 各 100 个验证样本、每样本 12 个 attention block 调用  
**机器可读结果**：`results/h67_h68_profile100_arch_features.json`  
**完整统计表**：`results/h67_h68_profile100_arch_features.md`

## 1. 本轮完成了什么

本轮没有占用正在训练 H69 的 GPU，而是重新分析已有的两个 profile100：

- H67：`h67_ep19_true_ttb_profile100_20260712/nts11_hardware_p0_profile.json`；
- H68：`h68_ep19_true_ttb_profile100_20260713/nts11_hardware_p0_profile.json`。

新增脚本：

```text
hw_autoresearch_nts07/scripts/analyze_h67_h68_profile100_arch_features.py
```

脚本完成以下重组：

1. 校验 100 个样本中 12 个 block 的调用顺序和记录完整性；
2. 按样本计算 attention row 周期代理的 p10/p50/p90/p99；
3. 按 stage/block 统计 pair-empty、K-zero、motion-zero、Delta=0；
4. 按 stage/block 统计 active-entry、K-zero fold class 的分位数；
5. 统计 TTB1/2/4/8 的空 bundle 和 active-lane 覆盖；
6. 给出哪些事实足以冻结架构，哪些必须等待新 ordered trace。

## 2. 全网结果

| 模型 | pair 全空 | K-zero | motion-zero | Delta=0 | active 项/行 | fold 类/行 | 周期代理 p99/p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H67 | 73.90% | 83.11% | 83.18% | 74.00% | 18.38 | 2.27 | 1.102 |
| H68 | 74.20% | 83.29% | 83.36% | 74.30% | 18.40 | 2.24 | 1.091 |

这里的 `pair` 是同一空间 token/head 的 `{Q0,Q1,K0,K1}` 时间对。`pair-empty` 才能精确注入两个固定 silent score；单时间片 `Q/K=0` 不能直接删除。

三个结论已经足够稳定：

1. **H67 与 H68 的 workload 几乎相同。** 没有证据支持分别实例化两套物理核。H67 应作为功能超集，H68 编译期关闭 Motion-XOR 和缩小 class 逻辑。
2. **时间对联合处理优先级高。** 当前 H67 接口每个时间片重复携带 64-bit K pair；改为每拍一个 128-bit `{Q0,Q1,K0,K1}` pair，可把每行前端 issue 从 162 次降到 81 次，并消除 K pair 的重复传输。
3. **Delta=0 几乎完全由 pair-empty 构成。** H67 中两者只相差约 0.10 个百分点。因此不能把“非空时间复用”包装成主要收益；更值得利用的是 pair-empty、K-zero class folding 和 block 间不均衡。

### 2.1 81-pair 前端与 context 预筛选

使用每个样本、每个 block 的 active/fold 实测均值，按真实 6720-row 顺序重放 81-cycle pair 前端和共享 SCS 后端：

| 模型 | context | 周期代理/帧 | 相对当前 162-token | 相对 pair 单 context |
|---|---:|---:|---:|---:|
| H67 | 1 | 843238 | -39.23% | 基线 |
| H67 | 2 | 607690 | -56.20% | -27.93% |
| H67 | 4 | 607489 | -56.22% | -27.96% |
| H67 | 8 | 607089 | -56.25% | -28.00% |
| H68 | 1 | 827927 | -39.67% | 基线 |
| H68 | 2 | 612195 | -55.39% | -26.06% |
| H68 | 4 | 612000 | -55.40% | -26.08% |
| H68 | 8 | 611618 | -55.43% | -26.13% |

该两阶段模型尚无独立 commit 阶段、逐 row burst、bank conflict 和 SRAM 延迟。它只能说明 2-context 足以重叠 `pair front -> SCS`，不能单独冻结最终物理 context 数。

### 2.2 供数与双提交端口感知重放

旧 profile 的 `ttb_tok1_kzero` 是两时间片 K 都为零的 pair 数，结合单时间片 `zaf_kzero_token_ratio`，可以精确恢复三种 pair：

| 模型 | 双 K-zero | 单 K-zero | 双 active |
|---|---:|---:|---:|
| H67 | 83.11% | 11.09% | 5.80% |
| H68 | 83.29% | 10.70% | 6.00% |

按行建立 `fetch -> commit -> SCS` 三阶段模型后：

| 模型 | 供数与 commit 结构 | 2-context 周期/帧 | 4-context 周期/帧 | 4 相对 2 |
|---|---|---:|---:|---:|
| H67 | 128-bit，分 bank 单写口，无 PCCC 合并 | 1077711 | 1037358 | -3.74% |
| H67 | 128-bit，分 bank 单写口，PCCC 全合并上界 | 709702 | 613965 | -13.49% |
| H68 | 128-bit，分 bank 单写口，无 PCCC 合并 | 1081856 | 1041900 | -3.69% |
| H68 | 128-bit，分 bank 单写口，PCCC 全合并上界 | 702601 | 618279 | -12.00% |

这里“全合并”是假定所有双 K-zero pair 的两个 score 属于同一 class 的乐观上界，真实结果一定在无合并和全合并之间。新结论是：

1. 128-bit pair 供数并不能单独保证高收益；无 PCCC 时 H67 相对当前 162-token 只下降约 22.33%；
2. 即使 PCCC 全合并，64-bit/拍供数的 H67 也只下降约 18.29%，低于当前 25% 晋级门槛；
3. PCCC 可能是解除 commit 瓶颈的必要机制，而不只是附加微优化；
4. context 必须参数化 `1/2/4`：先实现 2-context，是否物理实例化 4 个等待真实同类率和 ordered FIFO stall；
5. 暂不考虑 8-context。

## 3. 分阶段结果

| 模型 | stage | 行/帧 | pair 全空 | K-zero | active 项/行 | fold 类/行 |
|---|---:|---:|---:|---:|---:|---:|
| H67 | S0 | 2640 | 56.48% | 72.66% | 31.47 | 2.75 |
| H67 | S1 | 1440 | 94.92% | 96.08% | 3.63 | 1.36 |
| H67 | S2 | 2160 | 82.49% | 89.11% | 10.88 | 2.34 |
| H67 | S3 | 480 | 67.94% | 74.61% | 24.43 | 2.13 |
| H68 | S0 | 2640 | 56.23% | 72.32% | 32.35 | 2.71 |
| H68 | S1 | 1440 | 95.68% | 96.74% | 2.97 | 1.33 |
| H68 | S2 | 2160 | 83.03% | 89.44% | 10.56 | 2.29 |
| H68 | S3 | 480 | 68.88% | 75.66% | 23.18 | 2.16 |

S0 占全网 attention row 的 39.3%，S2 占 32.1%。因此：

- 只优化最深层 S3 不能代表整网；
- S0 决定吞吐和前端带宽；
- S1 提供最强的静默门控机会；
- S2 block 数最多，最适合验证多上下文和 block-aware 调度。

## 4. block 级异质性是当前最强架构信号

H67 的典型 block：

| block | 行/帧 | pair 全空 | active 项/行 | fold 类/行 | 架构含义 |
|---|---:|---:|---:|---:|---|
| S0B0 | 1320 | 47.37% | 59.89 | 1.51 | 高吞吐、active replay 压力最大 |
| S0B1 | 1320 | 65.60% | 3.05 | 3.99 | active 很少，但 class commit 较多 |
| S1B0 | 720 | 98.39% | 0.00 | 1.47 | 接近纯 silent/class 注入 |
| S2B3 | 360 | 99.98% | 0.00 | 1.02 | 可用极简路径处理，但仍需精确分母 |
| S2B5 | 360 | 73.69% | 26.97 | 1.47 | active replay 和 gated-K 较重 |
| S3B1 | 240 | 63.18% | 34.53 | 1.37 | 深层 active 输出较重 |

这比“全局 firing rate 低”更重要。S0B0 与 S0B1 行数相同，但 active-entry 相差约 19.6 倍。固定全局阈值或固定稀疏核会在不同 block 上出现完全不同的利用率。

因此当前应冻结：

- 配置粒度至少到 `stage/block`；
- descriptor 中携带表示模式、pair issue 宽度、context 配额和 class commit 策略；
- 是否进一步做到 row-level 动态路由，等待 ordered trace 的服务时间和 burst 统计。

## 5. TTB bundle 结果

| 模型 | bundle | empty | active<=4 | active<=8 | active<=16 | 非空 bundle 平均 active lane |
|---|---:|---:|---:|---:|---:|---:|
| H67 | 1 | 73.90% | 19.04% | 23.38% | 25.77% | 3.68 |
| H67 | 4 | 60.96% | 19.69% | 26.51% | 31.87% | 9.50 |
| H67 | 8 | 55.26% | 18.05% | 25.62% | 32.76% | 15.82 |
| H68 | 1 | 74.20% | 18.40% | 22.79% | 25.41% | 3.84 |
| H68 | 4 | 61.52% | 19.33% | 25.84% | 30.95% | 9.94 |
| H68 | 8 | 55.88% | 17.77% | 25.21% | 32.02% | 16.54 |

这说明 bundle 变大后 empty 比例下降、非空 payload 增长。TTB4/8 可以提高 metadata 摊销，但不能根据平均值直接断言更省能。最终要把：

```text
metadata bit + index packet + 对齐填充 + bank transaction + FIFO 空转
```

与固定 128-bit temporal-pair bitmap 比较。

## 6. 新 profile 已补哪些统计

软件 collector 已扩展为：

### 6.1 时间对充分统计量

- `q_count/k_count/overlap/same_zero/motion/update` 直方图；
- TTX/H68 与 H67 Q7 score 直方图；
- pair-empty、双 K-zero、单 K-zero、双 active；
- 两时间片 score 是否相等；
- 双 K-zero 同 class/双 class，可评估 pair-coalesced class commit；
- row active-entry、全部 score class、真实 K-zero fold class、score span。

### 6.2 表示与存储流量

- 4×32-bit bitmap；
- 四条 count+index packet；
- union-index + 4-bit membership packet；
- 每 pair 自适应 oracle 下界；
- 4/8/16/32 lane 前端工作周期下界。

### 6.3 光流与空间结构

- 输入事件密度、active pixel 比例；
- GT 光流幅值 mean/p50/p90/max、近零比例；
- 光流空间梯度、样本 AEE；
- 9×9 window 的时间持续 token、变化 token；
- 水平、垂直、两种对角 active 邻接；
- 4/8-bank 下 row-major、diagonal、XOR 映射的冲突周期。

空间统计用于检验“运动边缘形成方向性/对角局部性”是否真实存在。只有 diagonal mapping 的平均和 p99 bank cycle 明显优于 row-major，才允许把 ASADI 一类 ANN 对角数据布局迁移到本设计。

## 7. 当前未完成项

GPU 正被 H69 及后续串行软件队列使用。新的 profile100 watcher：

```text
neuron_experiments/H9_bipolar_self_attention/entrypoints/
run_ttb_cycle_profile_v2_after_round3.py
```

会在软件队列结束后依次运行 TTX/H67/H68，并生成：

- ordered pair trace；
- finite-FIFO replay；
- temporal-pair representation DSE；
- sample workload CSV 和相关性；
- 空间局部性与 bank mapping 对照。

仍未得到的结果：

1. pair/row 连续 burst 和最长 run；
2. 1/2/4/8 context 的真实 occupancy/stall；
3. 同拍 class collision 与合并写比例；
4. union packet 相对 bitmap 的真实 SRAM transaction；
5. diagonal/XOR bank mapping 的净冲突收益；
6. 输入事件、光流大小、边界强度与内部稀疏度的相关性。

## 8. 当前可以和不可以做的决定

### 可以冻结

- H67 功能超集、H68 编译期特化；
- 统一同构执行底座；
- temporal-pair 驻留接口；
- class-stationary SCS 后端；
- 参数化 1/2/4 个 row context、首版启用 2-context 的 RTL 骨架；
- block-aware descriptor；
- 三种表示和 bank mapping 必须做同约束 DSE。

### 不能冻结

- 稀疏/稠密异构双核；
- 完整蝶形压紧网络；
- 8 个以上 row context；
- diagonal bank mapping；
- ETCR 非空时间复用作为主贡献；
- 任意近似 pruning 或 weak-token omission。

## 9. 验证状态

```text
python -m py_compile：通过
test_bsa_attention + test_binary_temporal_pair_arch：56 tests，通过
旧 profile100 架构分析器：通过，生成 JSON/中文 MD
```

新统计代码已通过 CPU 单元测试，但完整 100 样本结果仍须等待 GPU 队列。论文中必须区分“统计器已验证”和“真实 workload 数值已获得”。
