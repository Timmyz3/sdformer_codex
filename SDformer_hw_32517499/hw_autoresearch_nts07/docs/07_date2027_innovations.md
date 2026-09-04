# DATE 2027 硬件创新点（论文可用表述）

每条含：**问题 → 方法 → 证据 → 与已有工作差异**。

---

## 创新点一：推理冻结的 Dyadic Attention ISA（H60-ISA）

### 问题
标准脉冲 Transformer 硬件要么做 softmax O(N²)，要么只做单一 QKFormer popcount，**无法覆盖** TX+SC 双分数融合 + Shiftmax 的软件收益。

### 方法
将 NTS-07b 推理图固化为四条原语：

```
DYAD_TX(q,k)   → 有符号 6bit 分数
DYAD_SC(q,k)   → 有符号 6bit 分数
SHIFTMAX_GATE(tx, sc, μ) → 每 token 8bit 门控
K_GATE(k, gate) → 注意力输出
```

μ、α0、center_scores 自 checkpoint 导出，**推理零运行时搜索**。

### 证据
- 软件：NTS-07b valid825 AEE 1.485 vs NB0 1.585
- 硬件：`h60_attention_engine.v` 单 window <600 周期，无浮点 MAC

### 与已有工作差异
- FireFly-T：单一 popcount，无 signed SC 融合
- SDformerFlow 原文：软件 QKFormer，未定义 ISA 级硬件

---

## 创新点二：统一 H60 注意力映射（全线 12 block）

### 问题
Legacy QKFormer 与 H60 **混用**迫使硅片实现 **两套注意力引擎** + 复杂 `ENGINE_MAP`，与软件「统一部署故事」和 DATE 可制造性冲突。

### 方法
软件 `target_blocks` 覆盖 **全部 12** encoder block；硬件 **仅实例化 H60 注意力核**，按 stage 切换 **heads / dim / windows** 参数表，**不综合 Legacy 通路**。

### 证据
- 配置：`nts11bc_hw_h60_all12_*`、`nts11bd_u12_*` 短测 sweep
- 解析模型：`hw_anchor=nts11bc` → Legacy 周期 0，面积 −Legacy 核 ~0.35 mm²
- 仍达 **~91 FPS / ~10 mJ**（TTB+PE256）

### 与已有工作差异
- FireFly-T：通用 Spike ViT，非光流 stage 几何
- 11aa 混用方案（废弃）：双 ISA；本方案 **单 ISA 全线**

---

## 创新点三：单一 ATLIF-PSN 异构算子 + 三值模式开关

### 问题
若软件保留 PSN / 三值 ATLIF / 二值 ATLIF 三条神经元路径，硅片需多套比较器与编码状态机，**DATE 叙事割裂**且面积重复。

### 方法
`atlif_unified_encode_unit`：**一套 FP16 比较器树**，per-layer `ternary_en`（descriptor `neuron_mode`）切换出口——
- `ternary_en=1`：2-bit {-1,0,+1} → H60 TX/SC；
- `ternary_en=0`：1-bit {0,+1} → Sparse MAC。

数据流主干（Event→Swin→H60@S2→Decoder）**不变**；仅 SN 出口编码与元数据 1 bit 扩展。

### 证据
- 软件：`ATLIFTernaryPSN(output_mode="ternary"|"binary")` 已统一实现
- 硬件 Segment-1：统一编码 + Bishop TTB-2 → **12.40 mJ**（较 12.97 mJ −4.4%），面积 **2.80 mm²**
- Q/K 2bit / FFN 1bit 带宽节省不变

---

## 创新点四：Shiftmax 原生归一化硬件

### 问题
Softmax 需大面积 exp/LUT；Shiftmax 在软件是 dyadic，但**尚无**面向 signed 融合分数的 ASIC。

### 方法
`shiftmax_unit.v`：行最大值减法 → 小 LUT 求 2^shifted → ceil(log2 Σ) 桶形除法，**纯移位无浮点**。

### 证据
- 与 BSA Shiftmax 公式位级对齐（容差 ±1 LSB）
- 面积约 <200 LUT vs softmax >2K LUT（估算）

---

## 创新点五：Profile 引导的 TTB 空窗跳过

### 问题
均匀调度所有 2×9×9 window；实际 firing 地图高度不均匀。

### 方法
1. 离线从 `spike_profile.json` 生成 `token_mask`
2. 在线 `window_enable=0` 当 mask 全零
3. Bishop 式 TTB 捆绑发射

### 证据（Autoresearch 实测）
- 关闭 skip：能耗 **29.48 mJ**（+13.6%）
- 开启 skip：基线 **25.94 mJ**
- 与软件 effective_flops −22% 趋势一致

---

## 创新点六：Autoresearch 驱动的数据通路定标（新增）

### 问题
手工拍脑袋选 PE 数量与 SRAM 大小缺乏可复现依据。

### 方法
11 轮自动网格搜索（`run_all_experiments.py`），主指标帧能耗，次指标 SRAM；Pareto 选出**终极组合**。

### 证据
- 256 路 Sparse MAC：能耗 **12.97 mJ**（−50%）
- 388 KB 片上 SRAM（256KB window + 128KB weight + 4KB meta）
- 101 FPS @ 500MHz（解析模型）

---

## 论文贡献句（中文）

> 本文提出面向 NTS 精炼版 SDformerFlow 光流的 stage 感知异构加速器，具有：（i）推理冻结的 H60 双分数注意力 ISA，融合三值 α-XNOR 与有符号共识分数及 Shiftmax 门控；（ii）由软件协同搜索驱动的静态引擎绑定；（iii）统一流式数据通路中的 1bit/2bit 混合脉冲编码；（iv）profile 引导的 Token-Time Bundle 空窗跳过。相比 Dense MAC 基线与类 FireFly-T popcount 方案，28nm 目标实现下帧能耗约 13 mJ、片上 SRAM 仅 388 KB，解析吞吐超过 100 FPS，满足边缘 30 FPS 实时目标，且保持 NTS-07b 软件验证精度带内。

---

## 审稿人质疑与回应

| 质疑 | 回应 |
|------|------|
| 为何不全网 H60？ | 软件 ablation：S2-only 最优；硬件面积省 50%+ |
| 没做 P&R 够 DATE 吗？ | 28nm DC 综合 + 面积/时序预估；P&R 放修订或补充 |
| 与 GPU 比意义？ | 边缘 5W 预算；报 mJ/frame 而非仅 FPS |
| 精度如何保证？ | golden 向量 + EPE drift < 0.02（进行中） |