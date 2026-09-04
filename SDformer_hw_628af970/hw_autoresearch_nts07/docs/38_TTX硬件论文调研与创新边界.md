# TTX 硬件论文调研与创新边界

**日期**：2026-07-10  
**目的**：从已有工作吸收设计范式，但明确引用、差异和真正需要验证的新增贡献

---

## 1. 原则

不能把已有论文模块换名后包装成新创新。可以做的是：

1. 引用已有范式；
2. 指出它们不适配 TTX 的原因；
3. 从 TTX 特有公式和 no-carrier 数据流推导新结构；
4. 用 RTL、等价性和 PPA 对比证明新增结构不是文字组合。

本文中“候选创新”表示尚需更完整 prior-art 检索和 DC/PPA 证据，不提前声称世界首创。

---

## 2. 核心文献

### 2.1 Bishop：TTB 与 error-constrained pruning

论文：**Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning**，ISCA 2025。  
链接：<https://arxiv.org/abs/2505.12281>

可借鉴：

1. Token-Time Bundle 作为工作单元；
2. active/inactive bundle 标记；
3. structured sparsity 与硬件调度结合；
4. dense/sparse workload 分流。

不能直接照搬：

1. Bishop 的 attention 是 TTB AAC 和 ECP pruning；
2. TTX 是同 token Q/K score，不构建通用 attention map；
3. 本项目不做 error-constrained pruning 主线；
4. 本项目的 ZAF 保留 K-zero token 的归一化贡献，是 exact folding。

我们的差异：

```text
Bishop: prune/skip bundle activity
TTX:    aggregate zero-value token denominator by exact score class
```

### 2.2 FireFly-T：双引擎、SRAM 重排、AND-PopCount

论文：**FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer Acceleration with Dual-Engine Overlay Architecture**。  
链接：<https://arxiv.org/html/2505.12771v1>

可借鉴：

1. sparse engine + binary attention engine；
2. descriptor/orchestrator 复用同一硬件；
3. SRAM write granularity 完成隐式数据重排；
4. 针对 FPGA LUT6 的 AND-PopCount compressor。

不能直接照搬：

1. FireFly-T 面向 QK attention matrix，需要转置/重排；
2. TTX score 是 token-aligned Q/K dyad，不需要 `N×N` matrix，也不需要 V transpose；
3. LUT6 compressor 是 FPGA 专用优化，不能直接当 ASIC 创新；
4. 本项目应使用 ASIC compressor tree 或 library-aware synthesis。

我们的差异：

```text
FireFly-T binary engine: AND-PopCount systolic attention
TTX score engine:       overlap + silence score，单 token dyadic stream
```

### 2.3 Softermax：base-2 与在线归一化

论文：**Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers**，DAC 2021。  
链接：<https://arxiv.org/abs/2103.09301>

可借鉴：

1. base-2 exponent；
2. low-precision softmax；
3. online normalization；
4. 算法与硬件共同修改 normalization。

不能声称：

1. base-2 exp 是我们的创新；
2. power-of-two denominator 是一般意义上的全新 softmax；
3. LUT exp2 本身足以成为论文贡献。

我们的差异：

1. Shiftmax 已经是 TTX 软件算法的一部分；
2. ZAF 聚合的是重复 K-zero score class；
3. 归一化分母被完整保留，不是简单省略 zero-value token；
4. FGK 进一步不物化 zero-value 输出。

### 2.4 Sparse Spike-Driven Transformer accelerator

论文：**An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer**。  
链接：<https://arxiv.org/abs/2501.07825>

可借鉴：

1. 把 spike position 编码成地址；
2. 双 spike 输入用地址比较跳过零；
3. sparse linear 只累加 active weight；
4. attention、linear、maxpool 统一使用 encoded spikes。

不能直接照搬：

1. 它实现 SDSA mask-add，不是 TTX Shiftmax；
2. 它不保留 K-zero token 的 normalization contribution；
3. TTX `HEAD_DIM=32`，bitmap popcount 可能比地址 merge 更合适；
4. FGK 的共享 gate/threshold late scaling 来自本项目 ATLIF + TTX 公式。

### 2.5 Spike-IAND-Former accelerator

论文：**Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing**。  
链接：<https://arxiv.org/abs/2503.19643>

可借鉴：

1. parallel tick batching；
2. reconfigurable neuron；
3. vectorized PE 支持 conv/matrix；
4. ASIC 评估标准：28nm、DC、PT-PX、SRAM、500MHz、FPS。

不能照搬：

1. 它把 residual ADD 改成 IAND；本项目软件仍是两处 ADD residual；
2. 我们不能为了硬件故事私自删除 residual；
3. TTX ATLIFPSN 是 temporal matrix mixer，不等于传统递推 LIF；
4. 当前只实现 TTX attention subsystem，不声称全网 PE array。

### 2.6 Spiking Transformer 3D accelerator

论文：**Spiking Transformer Hardware Accelerators in 3D Integration**，ICCAD 2024。  
链接：<https://arxiv.org/abs/2411.07397>

可借鉴：memory/logic placement、attention/neuron 数据移动分析、2D/3D 物理设计对比方法。

与本项目不同：ZAF/FGK 是逻辑和数据表示创新，不依赖 3D 工艺；若未来做 3D 只能作为物理实现扩展。

### 2.7 SpAtten 与一般 sparse attention

论文：**SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning**。  
链接：<https://arxiv.org/abs/2012.09852>

SpAtten 使用 token/head pruning 和 top-k engine。TTX gate 实测接近均匀，不能照搬 top-k 故事。ZAF 的跳过条件是 `K=0 → output=0`，并保留 denominator，因此不是 attention importance pruning。

### 2.8 Event optical flow 工作

参考：

1. **Neuromorphic Optical Flow and Real-time Implementation with Event Cameras**：<https://arxiv.org/abs/2304.07139>
2. **Spike-FlowNet**：<https://arxiv.org/abs/2003.06696>
3. **Event-Based Optical Flow Estimation with STBP SNN**：<https://doi.org/10.3390/mi14010203>

这些工作支持 event optical flow 的低活动/低延迟动机，但不直接提供 TTX attention hardware。论文相关工作中应把“应用算法”与“spiking transformer accelerator”分开。

---

## 3. 融合后形成的设计

| 来源范式 | 吸收内容 | TTX 修改 | 形成结果 |
|---|---|---|---|
| Bishop TTB | bundle work issue | 不做 ECP，增加 token-class exact folding | TTB + ZAF 两级稀疏 |
| FireFly-T | shared binary engine / orchestrator | 去掉 QK matrix/transpose，改 token dyad | 单 TTX row engine |
| Softermax | base-2 / low precision | 加入 repeated-logit multiplicity | class-folded Shiftmax |
| Sparse SDSA | active spike encoding | 输出保持 K bitmap，不生成 dense activation | FGK stream |
| Spike-IAND HW | tick/dataflow评估 | 保留真实 ADD residual，限制在 attention subsystem | 软件一致的边界 |

这不是“把五篇论文模块拼起来”。真正由 TTX 推导的部分是：

1. `K=0` 时 output 恒零但 denominator 不可删；
2. `K=0` score 只依赖 `q_active`，最多 33 类；
3. ATLIF binary amplitude 是共享 threshold；
4. gate 是 token/head scalar；
5. 因而可以 exact class folding + factorized output + late scaling。

---

## 4. 候选创新点

### 4.1 ZAF-Shiftmax

候选 claim：

> 面向 no-carrier binary selector attention，提出零 K 活动类折叠：不发出必为零的 value token，同时以 multiplicity-weighted class exponent 精确保留其 Shiftmax denominator contribution。

必须证明：

1. dense 与 folded gate exact/bit-accurate；
2. K-zero ratio 和 fold-class 数来自真实 workload；
3. cycle/energy 优于 dense；
4. 与 Bishop ECP、SpAtten pruning 的差异。

### 4.2 FGK stream + gate-late accumulation

候选 claim：

> 利用 binary ATLIF 的共享 threshold 和 TTX token/head scalar gate，以 bitmask + scale 表示 gated-K，并在下游 projection 中先做 spike-selected accumulation、后做共享缩放，避免 dense multi-bit activation materialization。

必须证明：

1. 与 dense `K×gate` 数学/定点等价；
2. SRAM/NoC bit 数降低；
3. multiplier 数或动态功耗降低；
4. 跨 head 累加顺序正确。

### 4.3 TTX dyadic score ISA

候选 claim：

> 将全 encoder attention 固定为 token-aligned binary dyadic score ISA，删除 QK matrix、SC、carrier、K-mag，并通过 descriptor 在 12 blocks 复用单一 row engine。

这更像系统/协同贡献，单独新颖性弱，必须和算法精度、INT8、ZAF/FGK 组合呈现。

---

## 5. 不应作为创新点的内容

1. 单纯的 AND + popcount；
2. 单纯的 exp2 LUT；
3. 单纯的 power-of-two denominator；
4. 单纯的 TTB 命名；
5. 单纯的 descriptor controller；
6. 把 105 个 PyTorch module 时分复用；
7. 1-bit SRAM 本身；
8. 把 H60 的 `mu` tie 0。

这些是实现基础或协同背景，不足以单独形成 DATE 创新。

---

## 6. Novelty 风险

| 风险 | 等级 | 缓解 |
|---|---|---|
| repeated-logit histogram softmax 可能有相邻工作 | 中 | 扩展 IEEE/ACM/patent 检索，限定 no-carrier zero-value exact folding |
| bitmask + scale 类似 block floating/shared scale | 中 | 强调 spike-selected sum 后 late gate，给 projection 等价推导 |
| TTB 被认为照搬 Bishop | 高 | 主图中明确引用 Bishop；TTB 只作前端，核心 claim 放 ZAF |
| Shiftmax 被认为 Softermax 变体 | 中 | 不 claim base-2；claim multiplicity folding 与 zero-output suppression |
| 只有 generic synthesis 无 ASIC PPA | 高 | 补 DC/PT-PX/SRAM 或至少 FPGA implementation |

---

## 7. DATE 贡献建议

论文贡献可写成四点：

1. **算法硬件协同**：TTX 将 all-binary optical-flow encoder attention 固定为 no-carrier TX-only selector，并验证 INT8 score/gate 几乎无损。
2. **ZAF-Shiftmax**：对 K-zero token 做 exact activity-class folding，在保留 normalization 的同时减少 exp 和输出事务。
3. **FGK 数据流**：用 K bitmap + gate + threshold 表示注意力输出，并在 sparse projection 中 late-scale，避免 dense gated-K 物化。
4. **统一 row engine**：以 descriptor 复用 12 blocks，结合 TTB work issue 和 1-bit event storage；真实 TTX profiling 支撑每 stage 的稀疏收益。

文章中必须给 Bishop、FireFly-T、Softermax、Sparse SDSA accelerator 明确引用，不应把来源范式隐藏成原创。

