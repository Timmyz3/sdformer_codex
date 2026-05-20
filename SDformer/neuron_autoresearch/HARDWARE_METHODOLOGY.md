# 硬件量化和论文方法学框架

---

## 一、注意力归一化的硬件成本对比

### 三种归一化方案的 arithmetic

| 方案 | 核心运算 | 28nm 估算 | 位置 |
|------|---------|:---:|------|
| **softmax** | Σ exp(x_i) / Σ exp, 每行 n 次 e^x | ~0.5 mm²/head | 标准 Transformer |
| **Shiftmax** (BSA) | Σ 2^x_i / 2^ceil(log2(Σ)), LUT-based | ~0.08 mm²/head | bsa_attention.py:60-70 |
| **ShiftNorm** | Σ x_i / 2^ceil(log2(Σ)), 无指数 | ~0.04 mm²/head | bsa_attention.py:73-89 |
| **L1Norm** | Σ x_i / Σ (exact), 除法器 | ~0.06 mm²/head | bsa_attention.py:92-101 |

### 面积分解 (28nm, per-attention-head, n=162 tokens)

| 组件 | softmax | Shiftmax | ShiftNorm | L1Norm |
|------|:---:|:---:|:---:|:---:|
| exp(·) | 162× FPU | — | — | — |
| 2^x LUT (4K×8bit) | — | 1× 16Kbit | — | — |
| log2(·) | — | 1× LUT | 1× LUT | — |
| ceil(·) | — | 1× 组合 | 1× 组合 | — |
| 除法器 | 1× 24bit | 1× 24bit | — | 1× 24bit |
| 加法树 | Σ exp | Σ 2^x | Σ x_i | Σ x_i |
| 总计 | ~0.5 mm² | **~0.08 mm²** | **~0.04 mm²** | **~0.06 mm²** |

### signed_consensus 前端的硬件

| 组件 | 28nm area | 说明 |
|------|:---:|------|
| sign(·) | 几乎零 | 三元脉冲已经是 ±thre |
| Q_sign × K_sign | d × XNOR | d=32, ~0.001 mm² |
| Σ agreement | popcount (adder tree) | ~0.003 mm² |
| normalize (/head_dim) | 右移 (head_dim=32) | 零面积 |

**合计 signed_consensus 前端 + ShiftNorm: ~0.047 mm²/head, 12 heads ~0.57 mm²。** 对比 softmax 12 heads ~6 mm²: 10× 面积缩减。

---

## 二、论文方法学框架

### 核心贡献三元组

```
贡献 1: Symmetric Ternary ATLIF (对称三元阈值)
  — 证明 neg_thre = thre 恢复正负平衡发放
  — target_rate 机制提供可解释的双向阈值调节
  — 输出严格 {+θ, 0, -θ}，完备的三值范式

贡献 2: Signed Consensus Attention (符号共识注意力)
  — Q_sign × K_sign popcount 替代 Q·K^T 点积
  — 硬件: XNOR + popcount + head_dim 右移
  — head_dim normalization 保留方向信息

贡献 3: Hardware-Clean Normalization (硬件清洁归一化)  
  — ShiftNorm/L1Norm 替代 softmax/Shiftmax
  — 零指数运算，零 LUT (L1Norm)
  — 10× 面积缩减 vs softmax
```

### 实验逻辑链

```
1. 负发放死亡 → S1 对称阈值 → 1:1 pos/neg 平衡
2. AAE 爆炸 → signed_consensus + head_dim norm → AAE 改善
3. SOPs 过高 → target_rate + low LR → SOPs 可控
4. 硬件论证 → ShiftNorm/L1Norm 面积量化 → 10× 缩减
```

### 关键对照实验表 (论文 Figure)

| 实验 | 三元 | 注意力 | 归一化 | AAE | SOPs |
|------|:---:|------|:---:|:---:|:---:|
| PSN baseline | — | QK gating | 无 | 7.50 | 3.62G |
| H9a | asym (30×) | compat | Shiftmax | 7.64 | **3.08G** |
| H23e | sym | signed_consensus | Shiftmax | **7.37** | 3.59G |
| (target) H13t | sym | signed_consensus | L1Norm | TBD | TBD |
| (target) | sym | signed_consensus | ShiftNorm | TBD | TBD |

### 论文写作骨架

```
1. Introduction
   - Event cameras + SNNs: latency, sparsity, energy
   - SDformerFlow: state-of-art but attention not designed for ternary
   - Our contribution: symmetric ternary + signed consensus + HW norm

2. Related Work  
   - Ternary SNN: TSN (AAAI 2024), BSA (NeurIPS 2025), QP-SNN (ICLR 2025)
   - SNN Attention: Spike-driven Transformer (ICLR 2024), QSD-Transformer (ICLR 2025)
   - Hardware SNN: LoAS (MICRO 2024), MINT (ASP-DAC 2024)

3. Method
   3.1 Symmetric Ternary ATLIF: S1 neg=thre fix
   3.2 Signed Consensus Attention: XNOR popcount + head_dim norm
   3.3 Hardware-Clean Normalization: ShiftNorm/L1Norm analysis

4. Experiments
   4.1 Negative firing recovery (H9a→H13n)
   4.2 Attention mode ablation (H18-H27)
   4.3 SOPs-sparsity tradeoff  
   4.4 Hardware area quantification

5. Conclusion
```

### 当前实验中最适合做 Figure 的结果

**Figure 1**: 负发放恢复 — H9a neg_mean=0.0004 vs H13n neg_mean=0.023, 57× improvement

**Figure 2**: AAE vs attention mode — signed_consensus (7.37) vs alpha_xnor (7.63) vs strict BSA (8.08) vs Hamming (8.15)

**Figure 3**: 硬件面积 — softmax 6mm² → Shiftmax 0.96mm² → ShiftNorm 0.48mm² → L1Norm 0.72mm² for 12 heads

**Figure 4**: SOPs vs AEE tradeoff — 所有 42 个 valid40 的散点图，H23e 和 H9a 高亮
