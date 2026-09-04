# 可优化硬件策略（Autoresearch 搜索空间）

本文档定义 autoresearch 循环可探索的 **硬件策略维度**，每项含：动机、参数、预期影响、风险。

---

## 优化目标

| 指标 | 符号 | 方向 | 权重 |
|------|------|------|------|
| 有效能耗 | `effective_energy_mj` | ↓ | 主目标 |
| 有效周期 | `effective_cycles` | ↓ | 次目标 |
| 片上 SRAM | `sram_kb` | ↓ | 约束 < 2048 |
| 精度漂移 | `epe_drift` | ↓ | 约束 < 0.02 |

---

## 维度 A：Window 调度（TTB）

**借鉴**：Bishop ISCA'25 Token-Time Bundle

| 参数 | 范围 | 默认 |
|------|------|------|
| `bundle_tokens` | 32, 64, 98 | 98 |
| `prefetch_windows` | 1, 2 | 2 |
| `skip_empty_windows` | on/off | on |

**策略**：当 window 内 `token_mask` 全零时跳过 H60 引擎，直接旁路 residual。

**预期**：周期 −5~15%（依赖 firing map）  
**风险**：控制逻辑面积 +8%

---

## 维度 B：H60 流水线深度

| 参数 | 选项 | 权衡 |
|------|------|------|
| `tx_sc_parallel` | 0=串行, 1=并行 | 并行：周期减半，面积 +30% |
| `shiftmax_pipeline` | 2/3/4 级 | 更深：频率↑，延迟↑ |
| `gate_k_fused` | 0/1 | 融合：省 1 次 SRAM 读写 |

---

## 维度 C：Sparse MAC 配置

| 参数 | 范围 | 默认 |
|------|------|------|
| `pe_count` | 64, 128, 256 | 128 |
| `bit_serial_width` | 1, 2, 4 | 1 |
| `zero_skip` | on/off | on |
| `weight_width` | 8, 4 | 8 |

**NTS-07b 特化**：firing 7.9% → `zero_skip=on` 近乎必选。

---

## 维度 D：存储层次

| 参数 | 选项 | 影响 |
|------|------|------|
| `window_sram_kb` | 256, 512, 1024 | 512 够 98×32×2b QK |
| `weight_buffer_kb` | 128, 256 | 256 减少 DRAM 冲突 |
| `metadata_on_chip` | yes/no | mask 常驻片上 |

---

## 维度 E：Stage 引擎绑定（软件已固定）

NTS-07b 已冻结 S2→H60；autoresearch **不改** stage 绑定，但可探索：

| 参数 | 说明 |
|------|------|
| `s2_only_h60` | 必须为 1（论文故事） |
| `legacy_qkformer_opt` | S0/S1/S3 是否也用简化 popcount |

---

## 维度 F：数值近似

| 参数 | 选项 | 精度影响 |
|------|------|----------|
| `shiftmax_lut_bits` | 6, 8, 10 | 8 足够 |
| `tx_alpha0_fixed` | 0, 1/64, 1/32 | NTS-07b 用 0.02≈1/64 |
| `sc_divider` | shift / div | 应用 shift（÷32） |
| `mu_fixed_q8` | 13/256≈0.05 | 推理冻结 |

---

## 已排除策略（NTS-07b 推理语义不允许）

| 策略 | 原因 |
|------|------|
| 引入 carrier 路径 | 改变 H60 算子图 |
| K_mag 通道 | α=0，硬件应裁掉 |
| Softmax 替代 Shiftmax | 破坏 dyadic 性质 |
| 全 stage H60 | 与 NTS-07b checkpoint 不匹配 |
| FFN ternary 推理 | 训练已固定 binary |

---

## Autoresearch 优先级队列

1. **P0**：`skip_empty_windows` + `zero_skip` + `tx_sc_parallel=1`
2. **P1**：`gate_k_fused=1` + `weight_width=8`
3. **P2**：`pe_count` 扫 64/128/256
4. **P3**：`legacy_qkformer_opt` 对非 S2 stage

脚本入口：`scripts/nts07_perf_model.py --grid docs/04_grid_default.json`