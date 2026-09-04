# 执行摘要：NTS-11bc 统一 H60 硬件加速器

**版本**：2026-06-13  
**软件锚点（待短测定型）**：`nts11bc` / `nts11bd_u12_*` — **全线 12 block H60 注意力**  
**废弃硬件线**：11aa / 07b 的 Legacy+H60 **混用** → `docs/15`  
**目标会议**：DATE 2027

---

## 1. 我们要做什么

为 `MS_SpikingformerFlowNet_en4` 设计 **单一 H60 注意力 ISA** 的异构 SNN 加速器：

1. **不再混用** Legacy QKFormer 与 H60；encoder **12 block 全部 H60**
2. 神经元仍为 **双模 ATLIF**（三值 Q/K + downsample 等；二值 all_non_qk）
3. 硅片 **三引擎**：Scatter + Sparse MAC + **H60** + Dense MAC（无 Legacy 核）

---

## 2. 相对混用方案（11aa）的核心变化

| 维度 | 11aa（废弃） | **11bc+（主线）** |
|------|--------------|-------------------|
| 注意力 | S0/S1 Legacy + S2/S3 H60 | **S0–S3 全部 H60** |
| 硬件引擎数 | 4（含 Legacy） | **3**（去掉 Legacy） |
| 控制器 | ENGINE_MAP 分两路 | **注意力恒 H60** |
| carrier / sn2_q | S0/S1 需要 | **H60 推理不用** |
| DATE 叙事 | 「stage 混绑」难讲 | **「统一注意力映射」** |

---

## 3. 三引擎架构

```
Event → [Scatter] → [Sparse MAC] → [H60 ×12 blocks] → [Sparse MAC MLP] → [Dense MAC] → Flow
```

| 引擎 | 负责算子 |
|------|----------|
| **Event Scatter** | VoxelGrid |
| **Sparse MAC** | Conv / MLP / downsample 卷积 |
| **H60 Binary** | **全部 encoder 注意力**（TX+SC+Shiftmax） |
| **Dense MAC** | Decoder |

---

## 4. DATE 创新点（v2）

1. **Unified H60 Attention ISA** — 12 block 同一套原语，stage 仅改 heads/windows  
2. **Unified ATLIF-PSN + ternary_en** — 双模神经元，单编码算子  
3. **Shiftmax-Native Unit** — 无 softmax MAC  
4. **TTB-Aware Scheduler** — S0 高 window 数更依赖空窗跳过  

（原「Stage 混绑 Legacy/H60」创新点 **已废弃**。）

---

## 5. 解析模型（provisional @ `nts11bc_anchor.json`）

| 指标 | 结果 | 目标 |
|------|------|------|
| 能耗 | **~10.4 mJ/帧** | ≤ 22 mJ |
| FPS | **~91** @ 500MHz | ≥ 30 |
| SRAM | 388 KB | ≤ 2 MB |
| 面积 | **~2.45 mm²**（无 Legacy 核） | — |

短测完成后用赢家 `spike_profile` 刷新 firing / AEE。

---

## 6. 交付物与阅读顺序

| 顺序 | 文档 |
|------|------|
| 1 | **`docs/16_统一H60注意力硬件方案.md`** — 主线 |
| 2 | `docs/02_end_to_end_dataflow.md` — 周期/DMA 补充 |
| 3 | `docs/05_module_interface_spec.md` — Step3 接口 |

---

## 7. 下一步

1. 等 **11bd 短测** valid10 → 升全量 → valid825  
2. 赢家 config 写入 `hw_anchor` + checkpoint 路径  
3. Yosys 综合 **仅 H60 路径**（Legacy 不综合）