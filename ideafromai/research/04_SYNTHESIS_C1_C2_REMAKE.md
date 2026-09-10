# C1 / C2 重做 — 创新机制总排序（Grok Bot / iscas_ssh）

**日期:** 2026-09-05  
**来源:** 合并 `01_ann_sparsity` + `02_snn_atlif` + `03_opticalflow`  
**原则:** 新机制 > 换乘法顺序；光流语义 / ATLIF 实值 / 稀疏门控 三线交叉；不碰已有文件只新增

---

## 一句话诊断

现在的 C1（精确积捕获）/ C2+TSBG（换序/分发）在审稿人眼里像 **通用 SNN MAC 岛**。要翻盘，必须让调度/累加 **吃光流数据性质**（方向可预测、遮挡、孔径、时间相干）和 **你们 ATLIF 实值载荷**，而不是再讲 zero-skip。

---

## 推荐打包（直接可当 ISCAS 贡献故事）

### C1* 重做包（优先做）— 新颖度 ~9

| 代号 | 机制一句话 | 挂论文 | 替代掉的旧叙事 |
|---|---|---|---|
| **OP-STW / DPP-Skip** | 用方向/残差预测唤醒 spike-tile，不是零激活跳过 | ERAFT ISCAS'25; Liu TCAS-I'25 | “有限容量精确捕获” |
| **PRRC** | 金字塔残差预算：粗层写 residual，细层只捕残差窗 | FlowAcc DATE'22; Ultra-Flow | 平坦 tile 容量 |
| **OGEC** | 遮挡/未匹配 mask：匹配走精确路径，未匹配走传播填补 | GMFlow CVPR'22 | 一律 exact |

**Claim 句式（安全）:**  
“首个由 **光流预测残差** 驱动 PE wake / exact-product 预算的 spikeformer OF 前端（相对 always-on / zero-skip）。”

### C2* 重做包（替换“换乘法顺序”）— 新颖度 ~9

| 代号 | 机制一句话 | 挂论文 | 替代掉的旧叙事 |
|---|---|---|---|
| **HBG-RP** | `{gate, payload}`：gate 控时钟/访存，payload 才乘 | FireFly-v2 TCAD'24; AT-LIF NeurIPS'25 对照 | 纯二值 AAC |
| **ADP-MAC** | 双边 bit 稀疏 + 负载均衡，保住 ATLIF 幅度 | BBS MICRO'24; BitParticle | “A×B 换序” |
| **ARM-Acc / MFBD** | Acc 上下文 = 运动假设 / motion-bundle，不是符号槽 | hARMS'22; VideoFlow/TMA ICCV'23 | 普通 multi-context |
| **SP-Gate / SS-FSA** | 用 attention-mass / score-stationary 调度，不是换序 | SpAtten HPCA'21; Bishop ISCA'25 | TSBG 换序叙事 |

**Claim 句式:**  
“C2 不是乘法重排：是 **实值 ATLIF 载荷 + 运动假设/注意力质量门控** 的双轨 datapath。”

### 可选杀手锏（前级或联合）

| 代号 | 说明 | 新颖度 |
|---|---|---|
| **TDE-Prior → Residual Transformer** | 便宜 TDE-3 给粗速度，transformer 只算残差流 | ~9 |
| **VGTS + ReMem-Tok** | 幅度感知时间跳步 + 窗口重叠膜电位复用 | ~8.5–9 |
| **EESUC** | 残差早停（ERAFT/TMA 思路落到 spike T） | ~8 |

---

## Top-10 单点机制（总榜）

1. **HBG-RP** — 二值门控 + 实值载荷（你们算法差异点）  
2. **OP-STW / DPP-Skip** — 光流方向预测 → tile wake  
3. **ARM-Acc** — 孔径多假设 Acc（最 OF）  
4. **MFBD** — Motion-bundle 投递（VideoFlow/TMA → C2）  
5. **ADP-MAC** — 双边 bit + 保幅度（钉死 C2 换皮）  
6. **OGEC** — 遮挡门控精确路径  
7. **TDE-Prior residual** — 生物先验 + 残差 spikeformer  
8. **VGTS** — 幅度+漏电预测的时间跳步  
9. **ReMem-Tok** — shifted-window 膜复用  
10. **MSBC / SP-Gate** — 光流显著度 × attention cascade  

---

## 算法纸（尚无匹配硬件）— 可挂 “first HW for X”

- SDformerFlow (ICPR'24 / arXiv:2409.04082) — spikeformer 稠密事件光流  
- TMA (ICCV'23) — 事件时序运动聚合  
- VideoFlow (ICCV'23) — 多帧运动传播  
- GMFlow (CVPR'22) — 全局匹配 + 遮挡传播  
- Spike-driven Transformer / V2 / QSD / SFA — 算法侧多比特/近似发放  
- AT-LIF (NeurIPS'25) — 注意：官方是 `{0,θ}`；你们若 **不可吸收进 W 的实值** 要写清差异  
- PSN (NeurIPS'23) — 并行时空，适合光流短 T  
- TDE-3 — 时差编码先验  

---

## 明确不要当主贡献（DO NOT CLAIM）

- 普通 zero-skip / 零激活跳过  
- 裸 N:M、静态混精、WS vs AS 分类学  
- 单独 bit-serial  
- “首个 spiking Transformer 硬件”（Bishop/FireFly-T/SpikeTA 已有）  
- Motion-XOR/TTX 当 AEE 算法创新（只可当地址/复用键）  
- “换个乘法顺序”任何变体  

---

## 建议落地顺序（工程）

1. **纸面定 C1\*/C2\* 贡献句**（本文件打包）  
2. ep34 上量：方向可预测率、遮挡占比、attention-mass 直方图、ATLIF `|amp|>ε` 稀疏度  
3. 先 RTL 原型 **HBG-RP + OP-STW**（机制清晰、和现有 TB 兼容）  
4. 再 **ARM-Acc 或 MFBD** 换掉 C2 叙事  
5. 对比梯子：always-on / zero-skip / 旧 C1·C2 / 新机制；指标 AEE + SOP/J + PE wake  

---

## 详细底稿

- `01_ann_sparsity_mechanisms.md`  
- `02_snn_atlif_realvalued_mechanisms.md`  
- `03_opticalflow_data_hw_algo.md`  

---

## Round-2 升级指针（2026-09-05）

包名与 claim 以 **`09_ROUND2_SYNTHESIS_C1_C2_UPGRADE.md`** 为准（合并 07/08；本文件保留为 R1 基线）。
