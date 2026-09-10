# ROUND 2 合成 — C1* / C2* 包名升级（Grok Bot / iscas_ssh）

**日期:** 2026-09-05（Asia/Shanghai）  
**输入:** Round-1 `04_SYNTHESIS` + Round-2 `07_transformer_attention` + `08_video_temporal_sparsity`  
**规则:** 只新增；不覆盖 `04`；不碰 `hw_autoresearch_nts07`；R2 **增强** R1 包，**不替换** HBG-RP / ARM-Acc。

---

## 一句话（相对 Round-1）

R1 定了「光流语义 + ATLIF 实值 + 稀疏门控」三线；R2 补了 **eager 相关预测 / 运动轨迹打包 / 时空分型稀疏 / 运动翘曲残差缓冲**——把 C1* 从「方向 wake」抬到「预测+翘曲+事件触发」，把 C2* 从「注意力质量门」抬到「分型稀疏织物 + Mask-Add 双轨」。

---

## 升级后的推荐打包（ISCAS 故事用这版）

### C1* — Temporal Residual + Eager Front-End（新颖度目标 ~9）

| 层级 | 代号 | 一句话 | 来源轮次 | 挂论文（代表） |
|---|---|---|---|---|
| 核心（保留） | **OP-STW / DPP-Skip** | 方向/残差 → spike-tile wake | R1 | ERAFT, FlowAcc |
| 核心（保留） | **PRRC** | 金字塔残差预算 | R1 | FlowAcc / Ultra-Flow |
| 核心（保留） | **OGEC** | 遮挡：精确 vs 传播 | R1 | GMFlow |
| **R2 主升** | **ECP-QKV** | QKV **投影前** 用便宜相关/粗流预测跳过投影 MAC | R2-07 | FACT ISCA’23 |
| **R2 主升** | **MW-ΔBuf** | 运动翘曲 Ref/膜，只算残差 spike | R2-08 | MotionDeltaCNN ICCV’23 |
| R2 辅 | **Δ-MaskPipe / DF-OS** | DiffFrame / delta mask 稀疏更新 + OS 捕获 | R2-08 | VideoTime3 / DeltaCNN |
| R2 辅 | **EV-Wake** | DVS/体素能量门控 exact 路径 | R2-08 | DVS–CIS ISCAS’25 等 |
| R2 辅 | **HeatFlow-Tok** | 金字塔级渐进 token 选择 | R2-07 | HeatViT HPCA’23 |

**C1* 安全 claim（升级句）:**  
“Exact-product / PE wake 由 **eager motion-correlation（ECP-QKV）+ motion-warped temporal residual（MW-ΔBuf）+ event/occlusion gate** 调度，而非 zero-skip 或平坦 tile 配额。”

**工程优先序（C1*）:** OP-STW → **ECP-QKV predictor** → MW-ΔBuf → OGEC/EV-Wake → PRRC/HeatFlow。

---

### C2* — Dual-Rail + SpatioTemporal Sparse Fabric（新颖度目标 ~9）

| 层级 | 代号 | 一句话 | 来源轮次 | 挂论文（代表） |
|---|---|---|---|---|
| 核心（**保留不换**） | **HBG-RP** | `{gate, payload}`：门控时钟/访存，载荷才乘 | R1 | FireFly-v2; ATLIF 差异 |
| 核心（保留） | **ADP-MAC** | 双边 bit 稀疏 + 保幅度 | R1 | BBS / BitParticle |
| 核心（**保留不换**） | **ARM-Acc** | Acc = 孔径多假设 | R1 | hARMS |
| 核心（保留） | **MFBD** | Motion-bundle 投递 | R1 | VideoFlow / TMA |
| 核心（保留→特化） | **SP-Gate** | attention-mass 调度 | R1 | SpAtten |
| **R2 主升** | **SMAM-RP** | Mask-Add 在 gate；实值 ATLIF MAC 仅 mask=1 | R2-07 | SMAM arXiv:2501.07825 |
| **R2 主升** | **Motion-TTB + OF-ECP** | `(tile,Δt,hyp)` 打包 + OF 误差界门控 | R2-07 | Bishop ISCA’25 |
| **R2 主升** | **STH-Gate** | Spatial vs Temporal head/token 分型稀疏 SDSA | R2-08 | Sparse VideoGen ICML’25 |
| R2 辅 | **Trajectory-Reorder Attn** | 按轨迹/对极线重排再 SDSA | R2-07 | PARO |
| R2 辅 | **TokMerge-OF / TS-Shift / SinkSparse / CW-Reuse** | 合并 token / 零 MAC 时移 / sink 块稀疏 / 通道前缀复用 | R2-08 | ToMe/TSM/RainFusion/Kaleido |

**C2* 安全 claim（升级句）:**  
“C2* **不是**乘法重排：是 **双轨 spike-gate + 不可吸收 ATLIF 载荷（HBG-RP/SMAM-RP）**，外加 **分型时空稀疏（STH-Gate）与运动轨迹包（Motion-TTB/MFBD）**。”

**工程优先序（C2*）:** HBG-RP → **SMAM-RP dual rail** → Motion-TTB packer → STH-Gate → Trajectory-Reorder / TokMerge。

---

## 联合杀手故事（ISCAS 篇幅）

1. **前端** ECP-QKV + OP-STW + MW-ΔBuf + OGEC/EV-Wake → 决定 *什么* 进 SDSA  
2. **中段** Motion-TTB/MFBD + STH-Gate/SP-Gate/OF-ECP → 决定 *哪些* Q/K bundle 跑  
3. **后端** SMAM-RP + ADP-MAC → gate 上 Mask-Add，payload 上实值 MAC  
4. **消融梯子:** always-on / zero-skip / 旧 C1·C2 / Bishop 纯二值 AAC / 逐块关掉 ECP·MW·STH·SMAM

---

## 更新后 Top-12（跨 R1+R2，写贡献句用）

1. **ECP-QKV** — 投影前相关预测（R2，~9）  
2. **MW-ΔBuf** — 运动翘曲残差缓冲（R2，~9）  
3. **HBG-RP** — 门控+实值载荷（R1，钉死差异点）  
4. **SMAM-RP** — Mask-Add × ATLIF payload（R2）  
5. **Motion-TTB + OF-ECP** — 运动时间包 + 误差界（R2）  
6. **STH-Gate** — 时空头分型稀疏（R2）  
7. **OP-STW** — 方向/残差 wake（R1，仍核心）  
8. **ARM-Acc / Trajectory-Reorder** — 孔径/轨迹 Acc（R1+R2）  
9. **MFBD** — motion-bundle 投递（R1）  
10. **ADP-MAC** — 双边 bit + 保幅度（R1）  
11. **OGEC / EV-Wake** — 遮挡+事件门（R1+R2）  
12. **TokMerge-OF / HeatFlow-Tok** — token 减负 / 金字塔选择（R2）

---

## 明确不要当主贡献（R1+R2 合并）

- zero-skip / 裸 N:M / 静态混精 / WS vs AS 分类学  
- “换乘法顺序”任何变体  
- “首个 spiking Transformer 硬件”（Bishop / SMAM / FireFly-T 等已有）  
- 纯 cascade prune / 弱 omit / 纯 score-stationary 当唯一卖点  
- FlashAttention ASIC 口号、PagedAttention 边缘 OF 叙事  
- “首个 video sparse attention / DiffFrame / TSM-FPGA”（SVG / VideoTime3 / Han Lab 已有）  
- 未持有具体引用的 EVA 挂名  

可谨慎用的 **first-HW for X（OF-spikeformer 限定）:** SVG 式 dual-head SDSA；Motion-warped Δ for event OF-T；token-merge for spikeformer OF；block-sparse SDSA + OF sink。

---

## 与现有 Codex 卡的关系

| Card / 块 | R2 后建议 |
|---|---|
| **Card A OP-STW** | 仍先做；规格里预留 **ECP-QKV** 接口（wake bitmap → 可选跳过 projection）与 **MW-ΔBuf** 输入 |
| **Card B HBG-RP** | 仍先做；规格对齐 **SMAM-RP**（gate Mask-Add / payload MAC） |
| 下一张建议 | **Card C:** ECP-QKV predictor；**Card D:** Motion-TTB packer；**Card E:** STH-Gate（可后置） |

R1 最小路径 OP-STW+HBG-RP（~4–6 pw）**不变**；R2 名称为 **第二层故事与后续卡**，不阻塞 A/B。

---

## 建议下一步（工程，不自动开干）

1. ep34 统计：方向可预测率、翘曲残差能量、spatial vs temporal attention mass、`|amp|>ε`、可合并 token %  
2. RTL：独立树只做 Card A/B；ECP/Motion-TTB/SMAM-RP 写进 microarch 草图再开卡  
3. 仍开放调研缺口（未做 Round-3）：CIM for spike/OF；event-camera 全栈；ISSCC/VLSI/HotChips edge NPU  

---

## 底稿索引

- R1: `01`–`04`；microarch `05`；plan `06`  
- R2: `07_transformer_attention_accelerators.md`；`08_video_temporal_sparsity_accelerators.md`  
- 本文件: `09_ROUND2_SYNTHESIS_C1_C2_UPGRADE.md`（**覆盖叙事优先级，不删改 `04`**）

*End of ROUND 2 synthesis.*
