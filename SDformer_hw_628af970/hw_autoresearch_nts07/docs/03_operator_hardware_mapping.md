# 算子清单与硬件引擎映射

## 1. NTS-07b 推理算子全集

| ID | 算子 | 出现位置 | 复杂度 | 引擎 |
|----|------|----------|--------|------|
| O1 | VoxelGrid scatter-add | 输入 | O(events) | Event Scatter |
| O2 | Conv2d 3×3 | Patch/Res/Decoder | O(HWCk²) | Sparse/Dense MAC |
| O3 | BatchNorm | 全线 | O(HWC) | Fused in MAC pipe |
| O4 | ATLIF-PSN temporal mix | Patch+全线 SN | O(T²) per pixel | Dense MAC mini / 融合进 SN |
| O5 | **统一 ATLIF encode** (`ternary_en=1`) | **全线 Q/K + downsample.sn×3** | O(HWC) compare | `atlif_unified_encode_unit` |
| O6 | **统一 ATLIF encode** (`ternary_en=0`) | Patch/FFN/其余 SN | O(HWC) compare | 同一算子，1-bit 出口 |
| O7 | QKFormer attn | S0/S1/S3 | O(N·D) popcount | Legacy Binary |
| **O8** | **TX α-XNOR score** | **S0–S3 全线** | **O(N·D) popcount** | **H60 TX unit** |
| **O9** | **SC signed consensus** | **S0–S3 全线** | **O(N·D) popcount** | **H60 SC unit** |
| **O10** | **Score fuse TX+μSC** | **S2** | **O(N) add** | **ALU** |
| **O11** | **Shiftmax** | **S2** | **O(N) 2^x+div** | **Shiftmax LUT** |
| **O12** | **K × gate** | **S2** | **O(N·D) mul** | **Gating MUX** |
| O13 | Linear proj | Attn out | O(N·D²) | Sparse MAC |
| O14 | MLP 4× | 全线 | O(N·D²) | Sparse MAC |
| O15 | Bilinear interp | 输出 | O(HW) | Dense MAC |

---

## 2. H60 核心算子微架构

### O8: TX α-XNOR（ternary 扩展）

软件（`bsa_attention.py`）：

```python
score = (+1)*same_nonzero + alpha0*same_zero - beta*opposite - gamma*single_active
```

NTS-07b：`alpha0=0.02, beta=0, gamma=0` → 硬件仅需三类 popcount：

| 类 | 条件 | 权重 |
|----|------|------|
| strong match | q==k, both active | +1 |
| silence reward | both silent | +0.02 → **LUT 近似为 +0（或 1/50 定点）** |
| opposite | q==-k, both active | 0（β=0 跳过） |

**RTL**：`tx_sc_score_unit.v` 模式 `TX_ONLY`

### O9: SC signed consensus

```python
score = sum(q_event * k_event) / head_dim   # q,k ∈ {-1,0,+1}
```

**RTL**：逐通道 XNOR-sign 乘法表 + signed adder tree → 6-bit score

| q | k | product |
|---|---|---------|
| +1 | +1 | +1 |
| -1 | -1 | +1 |
| +1 | -1 | -1 |
| 0 | * | 0 |

### O11: Shiftmax

```python
shifted = scores - max(scores)
num = 2^shifted
den = 2^ceil(log2(sum(num)))
gate = num / den
```

**硬件**：

1. Row max（6-bit comparator tree）
2. Per-token `2^(score-max)` → **7-entry LUT**（score 范围 −32..+32 → shifted −64..0，截断到 −16..0）
3. Sum + ceil_log2 → **桶形移位除法**

**RTL**：`shiftmax_unit.v`

---

## 3. 精度映射表

| 软件 | 训练 | 推理硬件 | 量化策略 |
|------|------|----------|----------|
| Q/K float pre-threshold | FP32 | FP16 scratch | 仅 SN 前暂存 |
| Q/K ternary | {-th,0,+th} | 2-bit {00,01,10} | 阈值来自 checkpoint |
| TX/SC score | float | 6-bit signed | 饱和到 ±32 |
| Shiftmax gate | float | 8-bit unsigned | [0,1] Q0.8 |
| MLP weight | FP32 | INT8 | per-channel scale |
| PSN weight | FP32 | FP16 | 10×10 全精度 |

---

## 4. Stage → Engine 绑定表（NTS-11bc 统一 H60）

| stage_id | block_count | attn_engine | mlp_engine | neuron_qk | neuron_ffn / 其余 |
|----------|-------------|-------------|------------|-----------|-------------------|
| 0 | 2 | **H60** | SPARSE_MAC | ATLIF-tern | ATLIF-bin |
| 1 | 2 | **H60** | SPARSE_MAC | ATLIF-tern | ATLIF-bin |
| 2 | 6 | **H60** | SPARSE_MAC | ATLIF-tern | ATLIF-bin |
| 3 | 2 | **H60** | SPARSE_MAC | ATLIF-tern | ATLIF-bin |

**无 Legacy QKFormer 行。** downsample.sn（S0–S2）三值编码后走 SPARSE_MAC。  
~~混用绑定表（11aa）~~ 见 `docs/15`。

---

## 5. 算子融合机会

| 融合 | 收益 | 风险 |
|------|------|------|
| BN + SN compare | 省 1 次 DRAM 写回 | 需校准阈值 |
| TX + SC 并行 popcount | 2× 吞吐 | 面积 +30% |
| Shiftmax + K-gate | 省 gate SRAM | 关键路径变长 |
| MLP SN1+Linear | bit-serial 直连 | 量化敏感 |

Autoresearch 搜索空间见 `docs/04`。