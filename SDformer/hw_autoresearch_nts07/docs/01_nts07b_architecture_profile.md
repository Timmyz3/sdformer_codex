# NTS-11bc 网络架构与张量剖面（统一 H60 主线）

> **硬件主线**：`docs/16_统一H60注意力硬件方案.md`  
> 11aa / 07b 混用注意力方案已废弃（`docs/15`）。

## 1. 顶层结构

```
MS_SpikingformerFlowNet_en4
└── MS_Spikingformer_MultiResUNet
    ├── encoders: MS_Spiking_SwinTransformer3D_v2  (4 stages)
    ├── resblocks: 2 × MS_ResBlock (bottleneck)
    ├── decoders: 4 × upsample conv + skip
    └── preds: 4 × flow head → temporal sum → interpolate
```

**配置锚点（短测）**：`configs/generated/nts11bc_hw_h60_all12_ds_w720_fastlr_s1224.yml`  
**Sweep**：`configs/generated/nts11bd_u12_*_s1224.yml`

## 2. 编码器四阶段几何

| Stage | depth | dim (C) | heads | window (T×H×W) | patch merge | 分辨率 (H×W) |
|-------|-------|---------|-------|----------------|-------------|--------------|
| S0 | 2 | 96 | 3 | 2×9×9 | 1,1,2,2 | 240×320 |
| S1 | 2 | 192 | 6 | 2×9×9 | 1,1,2,2 | 120×160 |
| **S2** | **6** | **384** | **12** | **2×9×9** | 1,1,2,2 | **60×80** |
| S3 | 2 | 768 | 24 | 2×9×9 | — | 30×40 |

**统一 H60 注意力**：`target_blocks` = **全部 12** encoder blocks（`0:0`…`3:1`）

## 3. 时间维与输入表示

| 参数 | 值 | 硬件含义 |
|------|-----|----------|
| num_bins | 10 | 事件体素时间 bin 数 |
| num_steps (PSN) | 10 | 膜电位并行时间步 |
| polarity split | pos/neg | 输入 2×10 → fold 为 T=10, C=2（再经 head conv 扩到 48） |
| crop | 288×384 | 训练/验证；推理可扩至 480×640 |
| norm_input | minmax | 仅对非零元素归一化 |

## 4. NTS-07b 注意力算子图（H60，推理冻结）

对每个 S2 block 的每个 head、每个 window、每个 query token：

```
1. Q_orig, K_orig ← Linear+BN+SN (PSN/ATLIF 前向)
2. Q_event = ternary_sign(Q_orig)     # qk: symmetric_bsa_tsn
3. K_event = ternary_sign(K_orig)
4. TX_scores = Σ_d [ same_nonzero + α0·same_zero − β·opposite − γ·single_active ]
5. SC_scores = Σ_d (Q_event * K_event) / head_dim    # NTS-07b: μ_schedule→0.05
6. scores = TX + μ·SC;  optional center_scores (减行均值)
7. gate = Shiftmax(scores)              # 2^x / 2^ceil(log2 sum)
8. attn = K_orig ⊙ gate                 # 无 carrier
9. out = Linear_proj(attn) + residual
```

**推理关闭项**（相对早期 NTX 线）：

- `k_magnitude_alpha = 0`
- `mismatch_penalty = 0`（β=0 时 opposite 项可硬件省略或保留常数 0）
- `single_active_penalty = 0`
- 无 `carrier_q = sn2(sum(Q))`
- 无 `target_rate` 反馈路径

## 5. 脉冲神经元路径（11aa 定稿：双模 ATLIF-PSN）

| 位置 | 模式 | 硬件 `ternary_en` | 注意力引擎 |
|------|------|-------------------|------------|
| **全线 Q/K** | 三值 ATLIF | **1** | **全线→H60** |
| **downsample.sn ×3**（layers 0/1/2） | 三值 ATLIF | **1** | —（后接 Sparse MAC） |
| **sn2_q**（配置可有） | 二值 ATLIF | **0** | **H60 推理不参与** |
| **all_non_qk**（Patch/MLP/decoder…） | 二值 ATLIF | **0** | Sparse MAC |

不再保留独立 PSN 第三条推理路径。

硬件在 **推理 checkpoint** 导出：

- `neuron_mode` per SN 层（ternary / binary）
- `pos_thresh` / `neg_thresh`（二值层 `neg_thresh` 不参与比较）

## 6. 计算量分布（NB0 剖面，NTS-07b 同拓扑）

| 子系统 | 理论 GFLOPs | 有效占比 | NTS-07b 变化 |
|--------|-------------|----------|--------------|
| Patch Embed + head | ~307 | ~31% | 相同引擎 |
| S0/S1 MLP+attn | ~330 | ~34% | attn 仍为原生 QKFormer |
| **S2 MLP+attn** | **~224** | **~23%** | **attn→H60 Binary** |
| S3 + bottleneck + decoder | ~184 | ~19% | 相同引擎 |

NTS-07b 的 SOPs 下降主要来自 **S2 attention + 全局 firing 降低**，不是拓扑变化。

## 7. 稀疏度实测（valid825 ep29）

| 指标 | NB0 | NTS-07b ep29 | Δ |
|------|-----|--------------|---|
| AEE | 1.585 | 1.485 | −6.3% |
| SOPs | 3.622G | 3.358G | −7.3% |
| global firing | 8.50% | 7.94% | −0.56pp |
| effective_flops | ~117G | ~91G | ~−22%（profile 口径） |

**硬件映射**：effective_flops 下降 ≈ Sparse MAC 的有效 MAC 数；S2 H60 进一步将 attention 从 O(N·D) MAC 降为 O(N·D) **popcount + shift**。

## 8. 片上存储预算（单帧推理）

| 缓冲 | 尺寸估算 | 策略 |
|------|----------|------|
| 当前 window Q/K | 2×98 tokens × 32 ch × 2b | Window SRAM 512KB |
| Stage feature tile | max 60×80×384 × 1b | DRAM tile + 64KB cache |
| 权重 tile | 256KB burst | Weight buffer |
| Metadata (masks) | < 4KB | 专用 narrow SRAM |

总计目标 **< 2MB on-chip**（不含完整 feature map，采用流式 window 计算）。