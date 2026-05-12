# H1 实验设计记录

## 核心创新: Gate-Threshold Co-Optimized Neuron (GTCN)

### 两级控制机制

| Level | 机制 | 硬件映射 | 来源 |
|-------|------|---------|------|
| **Coarse Gate** | 标量 learnable logit → sigmoid → STE 二值门 | clock-gating, gate=0 时跳过整个 spike_unit | G1 已验证 |
| **Fine Threshold** | 发放率反馈 → threshold_bias 累积 → 调制输入 | activity monitor + 慢速偏置更新 | ATLIF 论文 |

### Forward 公式
```
effective_x = x - threshold_bias          ← ATLIF 阈值偏置
out = base_neuron(effective_x)            ← PSN 脉冲神经元
gated = out * STE(sigmoid(gate_logit))    ← 粗粒度门控

if training:
    running_rate = EMA(spike_density)
    threshold_bias += activity_eta × (running_rate - target_rate)
```

## 扩展: FusedSparseNeuron (FSN) — 融合顶刊工作

在 GTCN 基础上加入:

| 顶刊工作 | 机制 | 融入 FSN 方式 | 配置参数 |
|---------|------|-------------|---------|
| **LMHT** (ICLR'24) | 多层级脉冲 {0,1,...,L-1}×th | `num_levels=3` → LMH 风格三阶量化 | `fsn_num_levels` |
| **Ternary Spike** | 三值脉冲 {-1,0,+1} | `signed=True` → 三值输出, 适合光流极性 | `fsn_signed` |
| **GTCN** (our) | gate+ATLIF | coarse gate + adaptive threshold (基础) | `init_logit, activity_eta` |

### 硬件优势
- 多层级 spike 用 2-3 个比较器替代单个比较器 (flash-ADC 风格)
- 下游仍用 AND-popcount (无乘法器)
- 每 spike 承载更多信息 → 可支撑更高稀疏率

## 训练配置演进

### Smoke 扫参结果
| init_logit | reg_lambda | valid_loss | gate 状态 |
|------------|-----------|-----------|----------|
| -2.0 (12%开) | 0.02 | 5.61 | 0/36 open |
| 0.0 (50%开) | 0.02 | 2.49 | 15/36 open |
| **2.0 (88%开)** | **0.005** | **0.78** | 36/36 open |
| 2.0 | 0.02 | 0.78 | 36/36 open |

### 全量训练参数 (H1 full)
- `stage_selection: all_stages_proj` → 36 gates (4 stages × depths × 3 nodes)
- `init_logit: 2.0` → gates 初始开放 (88%), 保证精度起点
- `reg_lambda: 0.02` → L1 正则逐步推动 gates 关闭
- `activity_eta: 0.001` → ATLIF 阈值自适应 (启用)
- `target_rate: 0.05` → 目标发放率 5%
- `freeze_backbone: true` → 只训练 36 个 gate_logit 参数
- `batch_size: 16`, `n_epochs: 20`
- ~9 min/epoch, 总约 3 小时

### 速度对比
- 之前 (G1 full): batch_size=1, ~20 min/epoch
- 现在 (H1 full): batch_size=16 + freeze_backbone, ~9 min/epoch
- 原因: freeze_backbone → 55M 参数不需要梯度, 反向传播极快

## 文件结构

```
src/models/modules/spiking_neurons/
  hw_sparse_neuron.py       # GTCN (gate + ATLIF threshold)
  fused_sparse_neuron.py    # FSN (GTCN + LMH multi-level + ternary)

neuron_experiments/H1_hw_sparse/
  overlay/models/STSwinNet_SNN/
    sparse_gate.py           # install_sparse_gates / install_hw_sparse_gates / install_fsn_gates
  configs/
    smoke.yml                # 1 epoch, 1 sample, 1 batch
    short_gate_only.yml      # 5 epoch, 8 batch, 16 samples
    full.yml                 # 20 epoch, 16 batch, 40 samples
  entrypoints/
    train.py                 # Source-patching 训练入口
    profile_sops.py          # SOPs profiling 入口
  results/
    h1_full_20260507.log     # 全量训练日志
```

## 论文故事线 (3 contributions)

1. **GTCN**: Gate-Threshold Co-Optimized Neuron — 粗粒度硬件时钟门控 + 细粒度 ATLIF 自适应阈值
2. **FSN**: Fused Sparse Neuron — 融合 LMH 多层级脉冲和三元脉冲, 更高信息密度
3. **Multi-Level Structured Scheduler** (Phase 4): Timestep→Window→Token→Head 联合调度, mask 直接导出给硬件控制器
