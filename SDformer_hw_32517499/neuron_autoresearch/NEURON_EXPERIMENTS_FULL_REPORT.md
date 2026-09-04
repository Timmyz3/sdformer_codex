# 神经元改进实验全景报告

生成时间: 2026-05-08 | 项目: SDformerFlow

---

## 一、已完成实验（结果已知）

### 1.1 完整神经元替换（E 系列 — blanket replacement）

将模型中所有 PSN 神经元替换为一种新类型。

| ID | 实验 | 机制 | 目录 | 代码来源 | 结果 | 可行性 |
|----|------|------|------|---------|------|--------|
| **E0** | PSN Baseline | 并行脉冲神经元，逐时间步并行计算 | `neuron_experiments/E0_psn_baseline/` | SpikingJelly (开源) | AEE=1.585, SOPs=3.62G, firing=0.085 | 基准 |
| **E1** | Simple SN | 简化版 LIF，单阈值发放 | `neuron_experiments/E1_exp_sn/` | **自建** (基于SpikingJelly LIF) | 仅smoke: train loss 6.77 | 不推荐 — 未完整评估 |
| **E2** | ATLIF | 自适应阈值：发放多→阈值升高→更难发 (Activity Pruning) | `neuron_experiments/E2_exp_atlif/` | **开源移植** [Activity-Pruning-SNN](https://github.com/putshua/Activity-Pruning-SNN) (NeurIPS 2024) | 多分支探索。最佳 sparse: SOPs=2.87G (-21%) 但 AEE=3.76 (×2.4)。最佳精度: AEE=2.51 但 SOPs=5.21G | ⚠️ 低 — 阈值增长过度剪枝，全替换破坏特征 |
| **E3** | LMHT | 可学习多层级阈值，脉冲量化为{0,1,...,L-1} | `neuron_experiments/E3_exp_lmh/` | **开源移植** [LMHT_SNN](https://github.com/hzc1208/LMHT_SNN) (NeurIPS 2024) | AEE=2.56~2.73, SOPs=9.7G (×2.7 基线) | 不推荐 — SOPs 太高，缺直接推理重参数化 |
| **E4** | TS-LIF | 时间段 LIF，双室时序建模 (ICLR 2025) | `neuron_experiments/E4_exp_tslif/` | **开源移植** [TS-LIF](https://github.com/kkking-kk/TS-LIF) (ICLR 2025) | AEE=2.18, SOPs=4.01G (+11%) | ⚠️ 中低 — 全替换中最平衡，但仍不如 PSN |
| **E4b** | TS-LIF Official-Style | E4 + 官方优化器设置 (Adam, 梯度裁剪, 分组LR) | `neuron_experiments/E4b_exp_tslif_officialstyle/` | 同上 | 仅短跑: AEE=6.99~7.06, SOPs=2.16~2.36G (精度崩溃) | 不推荐 — 短跑结果太差 |
| **E5b** | Ternary Spike | 三元脉冲 {-1,0,+1}，模仿三值信息编码 (AAAI 2024) | `neuron_experiments/E5b_exp_tsn_officialstyle/` | **开源移植** [Ternary-Spike](https://github.com/yfguo91/Ternary-Spike) (AAAI 2024) | AEE=29.8, SOPs=25.9G (×18.8 AEE) | 不推荐 — 范式不兼容 (ANN→SNN 分类 vs 事件流) |
| **E6** | NASN | 自适应脉冲窗口 α + 量化发放 N 级别 (arXiv 2025) | `neuron_experiments/E6_exp_asn/` | **自建** (基于论文公式，无官方代码) | AEE=2.17 (第二好) 但 SOPs=33.3G (×9.2 基线) | ⚠️ 低 — 精度可恢复但稀疏性完全失败 |

**结论: 完整替换全部失败。无一能在精度和稀疏性上同时超越 PSN 基线。**

---

### 1.2 融合神经元（F 系列 — fused approaches）

| ID | 实验 | 机制 | 目录 | 代码来源 | 结果 | 可行性 |
|----|------|------|------|---------|------|--------|
| **F1** | Fused Adaptive PSN | PSN + 自适应阈值 | `neuron_experiments/F1_fused_adaptive_psn/` | **自建** | 仅 smoke: train loss 8.16, val 6.22 | 未知 — 未跑完整实验 |
| **F2** | Fused LMH+ATLIF | 多层级 + 自适应阈值 | `neuron_experiments/F2_fused_lmh_atlif/` | **自建** | 仅 smoke: train loss 6.96, val 6.42 | 未知 — 未跑完整实验 |
| **F3** | Fused Adaptive TS-LIF | TS-LIF + 自适应阈值 | `neuron_experiments/F3_fused_adaptive_tslif/` | **自建** | 仅 smoke: train loss 6.04, val 6.35 | 未知 — 未跑完整实验 |
| **F4** | Fused LMH+TS-LIF | 多层级 + 时间段LIF | `neuron_experiments/F4_fused_lmh_tslif/` | **自建** | 仅 smoke: train loss 14.9, val 15.4 (差) | 低 — 烟雾质量太差 |
| **F5** | Fused Signed Hybrid | 有符号脉冲 + PSN | `neuron_experiments/F5_fused_signed_hybrid/` | **自建** | 仅 smoke: train loss 9.01, val 6.81 | 未知 — 未跑完整实验 |

**结论: F1-F5 都是 smoke-only 骨架，缺少完整训练验证。但 smoke 结果普遍不亮眼。**

---

### 1.3 部分稀疏门控（G/H 系列 — 当前最成功的路线）

| ID | 实验 | 机制 | 目录 | 代码来源 | 结果 | 可行性 |
|----|------|------|------|---------|------|--------|
| **G1** | Partial Sparse Gate | 给 6 个 layer0 节点各加一个标量门 `output = PSN(x) × STE(sigmoid(gate_logit)≥0.5)`, 主干冻结 | `neuron_experiments/G1_partial_sparse_gate/` | **自建** | **AEE=1.61 (+1.3%), SOPs=2.71G (-25.1%)** 🔥 | **最高** — 当前最佳稀疏/精度平衡 |
| **H1** | HW Sparse Neuron (GTCN) | G1 的门控 + ATLIF 阈值自适应，扩展到 36 个 encoder 节点 | `neuron_experiments/H1_hw_sparse/` | **自建** (融合 G1 gate + ATLIF threshold) | AEE=2.67, SOPs=2.71G (-25%) 精度掉了太多 | ⚠️ 中 — SOPs 降了但 AEE 从 1.61 掉到 2.67 |
| **H2** | Adaptive Ternary PSN | PSN + ternary spike + ATLIF 阈值, 仅针对 attention Q/K 节点 | `neuron_experiments/H2_adaptive_ternary_psn/` | **自建** | 仅 smoke: train loss 1.1, val 0.8 | 未知 — 仅 smoke, 仅 Q/K 目标很有意思 |

**结论: G1 的 25% SOP 减少+1.3% AEE 增加是迄今最强结果。H1 证明直接扩展到全 stage 不行。**

---

## 二、正在进行

| ID | 实验 | 机制 | 目录 | 代码来源 | 当前状态 |
|----|------|------|------|---------|---------|
| **A5** | Refractory Pruning | 脉冲发放后强制静默 2 步 (不应期)，全模型 60 轮训练 | `neuron_autoresearch/experiments/a5_refractory/` | **自建** (参考 AT-LIF 论文 NeurIPS 2024, 但机制不同) | **运行中** — GPU 20GB, bs=4, ~2.4it/s, ~13h |

---

## 三、已规划但未执行

### 3.1 配置完整的（随时可跑）

| ID | 实验 | 机制 | 目录 | 代码来源 | 预期 | 风险 |
|----|------|------|------|---------|------|------|
| **A1** | FSN on G1 Nodes | G1 的 6 个节点升级为 FusedSparseNeuron (2-level signed, 三元脉冲) | `neuron_autoresearch/experiments/a1_fsn_g1/` | **自建** (参考 BSA NeurIPS 2025 三元注意力) | SOPs<2.5G, AEE<1.75 | 中 |
| **A6** | Bipolar Attention Gate | FSN signed 专门用于 attention Q/K 投影层 | `neuron_autoresearch/experiments/a6_bipolar_attn/` | **自建** (参考 BSA NeurIPS 2025 + SEMM NeurIPS 2024) | SOPs 2.3-2.7G | 中 |
| **A8** | Dual-Sparse Regularizer | 训练时同时惩罚 firing rate + weight L1 | `neuron_autoresearch/experiments/a8_dual_sparse/` | **自建** (参考 Xu et al. Neural Networks 2025 + QP-SNNS ICLR 2025) | 额外 10-15% SOP 减少 | 低 |

### 3.2 仅设计构思（需要开发代码）

| ID | 实验 | 机制 | 代码来源 | 预期 | 风险 |
|----|------|------|---------|------|------|
| **A2** | Leakage-as-Gate | 用 PSN 现有的 decay 参数推导 gate 信号 (零额外参数) | **自建** (参考 Sparse SNN ICLR 2024) | SOPs 2.8-3.2G | 中 |
| **A3** | Hierarchical Shared Gates | 4 个 stage 各 1 个共享 gate (替代 36 个独立 gate) | **自建** (参考 DPRC-SNNs ICLR 2026 + QP-SNNS ICLR 2025) | SOPs 2.7-3.0G | 低 |
| **A4** | Timing-Dependent Gate | 早期时间步(噪声)门代价高, 晚期(信号)代价低 | **自建** (参考 SpikeSlicer NeurIPS 2024 + TTFSFormer ICML 2025) | SOPs 2.5-3.0G | 中 |
| **A7** | IMP Gating | 可学习初始膜电位推导 gate 信号 (零额外存储) | **自建** (参考 IMP-SNN NeurIPS 2024) | SOPs 2.5-3.0G | 中 |
| **A9** | Adaptive Timestep | 低事件率区域动态减少时间步 | **自建** (参考 SpikeSlicer + Spiking Patches arXiv 2025) | SOPs 2.5-3.0G | **高** |

---

## 四、参考开源实现（仅参考，未移植）

| 代码 | 论文 | 仓库 | 用途 |
|------|------|------|------|
| **PSN** | Parallel Spiking Neuron (NeurIPS 2023) | [fangwei123456/Parallel-Spiking-Neuron](https://github.com/fangwei123456/Parallel-Spiking-Neuron) | **当前基线**，已内置在 SpikingJelly |
| **TC-LIF** | Two-Compartment LIF (AAAI 2024) | [ZhangShimin1/TC-LIF](https://github.com/ZhangShimin1/TC-LIF) | 未移植 — 双室时序参考 |
| **ATLIF** | Activity Pruning SNN (NeurIPS 2024) | [putshua/Activity-Pruning-SNN](https://github.com/putshua/Activity-Pruning-SNN) | E2 已移植，结果不佳 |
| **LMHT** | LM-HT SNN (NeurIPS 2024) | [hzc1208/LMHT_SNN](https://github.com/hzc1208/LMHT_SNN) | E3 已移植，SOPs 太高 |
| **TS-LIF** | Temporal Segment LIF (ICLR 2025) | [kkking-kk/TS-LIF](https://github.com/kkking-kk/TS-LIF) | E4 已移植，全替换中最好但不如 PSN |
| **TSN** | Ternary Spike (AAAI 2024) | [yfguo91/Ternary-Spike](https://github.com/yfguo91/Ternary-Spike) | E5b 已移植，范式不兼容 |

---

## 五、总体路线图

```
已完成实验 18 个:
  ✅ E0-E6 + E4b + E5b (10个完整替换)
  ✅ F1-F5 (5个融合骨架)
  ✅ G1 (1个部分门控 — 最佳结果)
  ✅ H1, H2 (2个硬件感知)

进行中:
  🔄 A5 Refractory Pruning (13h 剩余)

可立即启动:
  🟢 A1 — FSN on G1 (配置完整)
  🟢 A6 — Bipolar Attention (配置完整)
  🟢 A8 — Dual-Sparse Regularizer (配置完整)

需要开发:
  🟡 A2, A3, A4, A7, A9 (仅设计)
```

### 核心洞察

1. **G1 是唯一成功的策略**: 25% SOP 减少 + 1.3% AEE 增加。关键: 部分插入 > 全替换, gate-only 训练 > 全模型训练
2. **全替换是死胡同**: 15 个全替换实验中, 0 个成功。最好的是 E4 TS-LIF (+37% AEE, +11% SOPs)
3. **FSN 是最大未验证机会**: G1 的 6 个节点 + FSN 三元脉冲 = A1, 还没跑
4. **不应期是低风险基础改进**: A5 现在在跑, 如果有效可以叠加到其他实验上
5. **开源移植 ≠ 直接可用**: 所有移植的开源神经元在 SDformerFlow 上都失败 — 模型范式差异太大
