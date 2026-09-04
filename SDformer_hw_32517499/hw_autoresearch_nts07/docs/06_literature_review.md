# 文献对标与可借鉴报告（精简版）

面向 DATE 2027 硬件章节，按与 NTS-07b 相关性排序。

> **扩展版（20+ 篇、四条检索线、Segment-2 idea 池）**：见 `docs/13_扩展文献库与可借鉴清单.md`  
> **硬件主线（统一 H60）**：`docs/16_统一H60注意力硬件方案.md`  
> **硬件入门顺序**：`docs/14_硬件小白入门路线图.md`

---

## 第一梯队：必须精读并引用

### 1. SDformerFlow（Tian & Andrade-Cetto，ICPR 2024 / arXiv 2409.04082）

| 维度 | 内容 | 我们借鉴 |
|------|------|----------|
| 模型 | PSN + 3D Swin + O(N) QKFormer 注意力 | 基线拓扑 |
| DSEC | 主训练/验证/官方 benchmark | 软件评测协议 |
| MVSEC | MDR 训练 → MVSEC 测试 | 泛化对照 |
| 能效 | FLOPS×firing×T×E_AC | 能耗模型框架 |

**差异（创新空隙）：** 原文未做 stage 局部替换、未做 H60 双分数融合、未给出 ASIC RTL。

---

### 2. FireFly-T（IEEE TC 2026，arXiv ~2505.12771）

| 维度 | 内容 | 我们借鉴 |
|------|------|----------|
| 引擎 | 双稀疏 + **二进制 AND-PopCount** | H60 TX/SC 单元 |
| 注意力 | 脉冲注意力无矩阵乘 | 面积/功耗对标基线 |
| 指标 | FPS/W @ 边缘 | DATE 能效表格式 |

**论文画法：** 微观结构图 — FireFly-T popcount vs 本工作 Shiftmax 融合 H60（多一级 score 融合，仍无 MAC）。

---

### 3. Bishop ISCA'25（arXiv 2505.12281）

| 维度 | 内容 | 我们借鉴 |
|------|------|----------|
| 调度 | **Token-Time Bundle (TTB)** | window×timestep 打包 |
| 架构 | 异构 core + ECP | 四引擎异构 |
| 稀疏 | 结构化 token 跳过 | `skip_empty_windows`（实验证实 +13.6% 能耗若关闭） |

---

### 4. ASNA-Flow（IEEE TVLSI 2025）

| 维度 | 内容 | 我们借鉴 |
|------|------|----------|
| 前端 | 异步事件光流 | Event Scatter 单元 |
| 能效 | **7.9 mW，104 FPS** | 边缘功耗时钟 |
| 数据流 | 体素增量更新 | scatter-add 微架构 |

---

### 5. ADMFlow / MDR（Luo et al.，arXiv 2303.11011）

| 维度 | 内容 | 我们借鉴 |
|------|------|----------|
| 数据 | 多密度渲染训练集 | MVSEC 泛化协议 |
| 指标 | MVSEC dt1 AEE | Table II 对标 |

---

## 第二梯队：补充引用

| 论文 | 会议/年份 | 借鉴点 |
|------|-----------|--------|
| Spiking Transformer 3D Accelerator | ICCAD'24 | 3D 激活片上缓冲 |
| Reconfigurable Spiking Transformer Accel. | arXiv 2503.19643 | 并行 timestep |
| SENECA | 2023 | 三级存储、事件驱动 |
| SpinalFlow | ISCA'20 | SNN 数据流经典 |
| QKFormer | 2024 | 线性复杂度注意力原理 |
| BSA | 2025 | Shiftmax 归一化 |
| OF_EV_SNN | Front. Neurosci. 2023 | DSEC SNN 基线 |
| Prosperity | HPCA'25 | 乘积稀疏 |
| Phi | ISCA'25 | 模式化层次稀疏 |

---

## 论文对标表（可直接进正文）

| 加速器 | 年份 | 领域 | 注意力 | 稀疏 | 工艺 | 功耗 | 吞吐 | 片上存储 |
|--------|------|------|--------|------|------|------|------|----------|
| SpinalFlow | ISCA'20 | 通用 SNN | — | spike | 28nm | — | — | — |
| ASNA-Flow | TVLSI'25 | 事件光流 | 模型法 | event | — | 7.9mW | 104FPS | — |
| FireFly-T | TC'26 | 脉冲 ViT | PopCount | 双稀疏 | FPGA/ASIC | — | — | — |
| Bishop | ISCA'25 | Transformer | 异构核 | TTB | — | — | — | — |
| **本工作（NTS-07b）** | DATE'27 | **事件光流 SNN** | **H60 双分数** | **异构脉冲** | **28nm** | **~13mJ/帧** | **30+FPS** | **388KB** |

---

## 引用策略（DATE 审稿口味）

1. **引言**：SDformerFlow（任务）+ FireFly-T（脉冲 attention 硬件）+ Bishop（调度）
2. **相关工作**：分三条线 — 事件光流 ASIC / 脉冲 Transformer 加速器 / 软硬件协同
3. **架构**：与 FireFly-T 并排微观结构图；强调 **stage 绑定 H60** 差异
4. **评测**：ASNA-Flow 能效数量级 + SDformerFlow 精度协议 + 综合面积（待 Yosys）