# POST TDE3+TMA Hard Knives — Fork A enhancers (Grok Bot / iscas_ssh)

**日期:** 2026-09-06（Asia/Shanghai）  
**Scope:** TCAS-II letter · SDformer C1*/C2* **Fork A only**  
**Hard filter:** ONLY mechanisms that **ENHANCE** OP-STW / HBG-RP / OGEC×PRRC→exact_capture.  
**Do NOT push:** Grok46 pure-binary ATLIF / MX3P island swap / vague CIM·analog silicon.  
**Prefer:** event priors · temporal aggregation · control-loop / early-exit · sparse attention fabrics · **first-HW / synthesizable digital RTL**.  
**Committed / lower priority (do not rehash as top):** TDE3-Prior · TMA-Agg (Card G) · EvQ-Win · PredExit-OF · BitHyp-Prior.  
**This file:** what is **STILL MISSING** after those five + stack; flag anything **sharper** than EvQ/PredExit/BitHyp as **RED**.

**Existing stack (incremental delta only):**  
OP-STW, TDE3-Prior, wake_merge, HBG-RP, OGEC, PRRC, exact_capture, TMA-Agg, ECP-QKV, MW-ΔBuf, SMAM-RP, Motion-TTB, STH-Gate, ADP-MAC, ARM-Acc, MFBD, SP-Gate, EvQ-Win/PredExit/BitHyp (idea stage).

---

## 一句话（给 teammate）

TDE3+TMA 把「生物粗速先验 + 时序运动聚合」钉死了；**仍缺**的是：(i) **双向/非线性**时序（比线性 TMA lookup 更硬），(ii) **置信度驱动的残差 scrub / exact 预算**（比 PredExit 二元早停更锐），(iii) **无预测器 / bit-guard 稀疏注意力**（比 EvQ 邻居窗更贴 SDSA MAC），(iv) **抑制型 Barlow–Levick / 匹配反馈控制环**（与 TDE3/BitHyp 正交的数字先验）。本轮 8 把刀全部是 Fork A 增强，可合成数字 RTL。

---

## Gap analysis — after TDE3 + TMA, still missing …

| Gap | Why TDE3/TMA/EvQ/PredExit/BitHyp 不够 | Knife(s) below |
|---|---|---|
| Uni-directional temporal only | TMA = forward split + **linear** lookup；无 backward / future-flow / occlusion-from-exit | **BiSAT-Agg** |
| Linear motion assumption | TMA/MW-ΔBuf 假设匀速；快抖动 / 非线性长序列崩 | **NL-STMFA** |
| Binary early-exit | PredExit = residual&lt;ε halt；不 scrub 已污染的迭代态 | **SCI-CleanExit** |
| Occlusion ≠ confidence budget | OGEC 是 exact vs propagate；缺 **pixel confidence → PRRC/exact_capture 配额** | **CFP-ConfGate** |
| Wake 仍靠方向字段 | OP-STW 有方向/残差；缺 **动态方向预测直接砍 PE**（frame OF FPGA 已证） | **DynDir-Wake** |
| Tw / slice 开环 | AdaptSlice 有；缺 **匹配距离闭环** 调 event 曝光窗 | **AEC-MatchFB** |
| TDE-3 = Reichardt 促进型 | 无 **Barlow–Levick 抑制 veto** 数字银行 | **BL-VetoPrior** |
| EvQ = 邻居 gather | 管窗口几何，不管 **attention score 逐 bit 剪枝** | **BUI-GuardSDSA** |
| BitHyp = occupancy hyp | 不管 **log 前导零** 的廉价 score 稀疏预测 | **LZ-SparsePred**（辅） |

---

## NEW remake candidates（≥6；不含 TDE3/TMA/EvQ/PredExit/BitHyp 原名）

### BiSAT-Agg — bidirectional temporal corr + SATMA adaptive fuse → MFBD/Motion-TTB
- **Near papers:** Xu et al., *BAT: Learning Event-based Optical Flow with Bidirectional Adaptive Temporal Correlation*, **arXiv:2503.03256** (2025; DSEC-Flow 1st vs TMA/E-RAFT); Liu et al., *TMA*, **ICCV 2023** (baseline to beat); VideoFlow, **ICCV 2023** (multi-frame bidirectional cousin, frames).
- **Delta vs our stack:** TMA-Agg = forward split + linear lookup；**仍缺** backward corr（future-flow / exit-occlusion）与 **SATMA**（deformable sparse attn 抑不一致运动特征）。OP-STW/HBG/exact_capture 无双向时间织物；MFBD/Motion-TTB 可吃 fused fwd+bwd motion bundles。
- **RTL sketch interface:**
```systemverilog
// bisat_agg: N temporal groups, fwd+bwd corr tiles → fused motion pack
input  logic        corr_valid_i, bwd_en_i;
input  logic [7:0]  group_id_i;          // 0..2N-1
input  flow_t       df_tile_i;           // f/N linear seed (then SATMA corrects)
output logic        fuse_valid_o;
output motion_pack_t m_fuse_o;           // → MFBD / Motion-TTB
output logic        occ_exit_hint_o;     // → OGEC (object leaving FOV)
```
- **Novelty self-score:** **9.0 / 10** · first-HW posture: **algo→first digital HW for BAT-style bi-temporal agg under spikeformer OF**（谨慎：BAT 本身是 GPU algo；勿称 first event OF HW）。
- **Overclaim traps:** 勿称 “first temporal OF HW”（EDFLOW/TMA GPU/ASNA）；勿把 deformable attn 说成已有 ASIC；抖动剧烈时 BAT 自承 backward 增益有限。
- **RED?** **yes** — 比 EvQ-Win（几何窗）更贴 letter 时序刀；比 BitHyp/PredExit 更直接升级 Card G TMA-Agg。

---

### NL-STMFA — nonlinear motion-guided spatio-temporal feature aware residual warp
- **Near papers:** *E-NMSTFlow: Nonlinear Motion-Guided and Spatio-Temporal Aware Network for Unsupervised Event-Based Optical Flow*, **ICRA 2025** / arXiv:2505.05089 (STMFA + AMFE；vs TamingCM +19% MVSEC / +13% DSEC unsupervised).
- **Delta vs our stack:** MW-ΔBuf / TMA-Agg 偏线性对齐；**仍缺** 非线性运动补偿 + 跨尺度 previous-hidden aggregation（STMFA）与 self-attn 自适应增强（AMFE）。可喂 OP-STW residual 与 PRRC 细层窗。
- **RTL sketch interface:**
```systemverilog
// nl_stmfa: prev multi-scale F_{t-1} → nonlinear warp residual mask
input  logic        feat_valid_i;
input  feat_vec_t   f_prev_l_i [0:3];    // pyramid prev hidden
input  flow_t       f_lin_i;             // TMA/BitHyp linear seed
output logic        nl_mask_valid_o;
output residual_t   r_nl_o;              // → MW-ΔBuf / OP-STW wake
output logic [3:0]  amfe_w_o;            // attn weights → STH-Gate
```
- **Novelty self-score:** **8.5 / 10** · first-HW: **algo-only → candidate first RTL for NL motion residual under event OF-T**.
- **Overclaim traps:** 无现成 FPGA；unsupervised loss 不要写进 TCAS 主贡献；勿与 CIM 混谈。
- **RED?** **soft-yes** — 比 TMA 线性刀更锐；对 EvQ/BitHyp 是正交增强而非直接替代。

---

### SCI-CleanExit — warp-consistency self-cleaning iteration channel + residual scrub gate
- **Near papers:** Lin et al., *SciFlow: Empowering Lightweight Optical Flow Models with Self-Cleaning Iterations*, **CVPRW EVW 2024** / arXiv:2404.08135 (SCI + RFL；轻量 RAFT；近零推理开销)；ERAFT FPGA early-exit, **ISCAS 2025**（PredExit 灵感源，frame）。
- **Delta vs our stack:** PredExit-OF = **halt** when residual&lt;ε；**仍缺** 每迭代用 **warp consistency SCI map** 作额外通道，主动 scrub 已污染估计（error propagation）。对 exact_capture：SCI 低置信区 → 继续 refine / 高置信 → 释放 PRRC 预算。增强 HBG-RP：gate 可跟 SCI 质量时钟门控。
- **RTL sketch interface:**
```systemverilog
// sci_clean: F1 vs warp(F2,f) → G_sci ∈[0,1] → scrub / exit
input  logic        it_valid_i;
input  feat_t       f1_i, f2_i;
input  flow_t       f_hat_i;
output logic [7:0]  g_sci_o;             // Q8.0 quality map (tile)
output logic        scrub_en_o;          // force another update
output logic        exit_ok_o;           // sharpens PredExit
output logic        exact_hold_o;        // → exact_capture budget
```
- **Novelty self-score:** **8.5 / 10** · first-HW: **first event-spikeformer OF with SCI scrub channel**（SciFlow 是 frame/lite algo+on-device；勿称 first OF FPGA）。
- **Overclaim traps:** RFL 仅训练；letter 主卖 SCI 推理通道；勿把 Snapdragon 数字当你们 ASIC 结果。
- **RED?** **yes** — 明确 **sharper than PredExit-OF**（scrub ≠ binary exit）。

---

### CFP-ConfGate — confidence-induced flow propagation → OGEC×PRRC exact budget
- **Near papers:** Deng et al., *EMD-Flow / Explicit Motion Disentangling…*, **ICCV 2023**, DOI:10.1109/ICCV51070.2023.00873 (MMA + **CFP** confidence maps → dense init，轻量 refine)；GMFlow **CVPR 2022**（OGEC 遮挡传播近亲）。
- **Delta vs our stack:** OGEC = match→exact / unmatch→propagate；PRRC = 金字塔残差窗；**仍缺** **自带置信度图** 驱动：高 conf 区 propagate+早释 exact 配额，低 conf 区 exact_capture 加码。比 PredExit 更细（像素/ tile 级预算，而非整帧退出）。
- **RTL sketch interface:**
```systemverilog
// cfp_confgate: corr peak → conf → exact vs propagate + PRRC quota
input  logic        corr_valid_i;
input  logic [15:0] corr_peak_i;
input  logic [7:0]  prrc_budget_i;
output logic [7:0]  conf_o;
output logic        exact_req_o;         // → exact_capture
output logic        prop_en_o;           // → OGEC fill
output logic [7:0]  prrc_grant_o;        // residual window grant
```
- **Novelty self-score:** **8.5 / 10** · first-HW: **first HW wiring CFP-style conf → spikeformer exact_capture/PRRC**（EMD-Flow 为 frame algo）。
- **Overclaim traps:** 勿称 first occlusion OF；与 OGEC 写清 **enhance not replace**；conf 阈值需 ep34 校准。
- **RED?** **yes** — 比 PredExit 更贴 OGEC×PRRC→exact_capture 主叙事。

---

### DynDir-Wake — dynamic direction prediction PE skip for OP-STW
- **Near papers:** *An Ultra-High Performance and Scalable Optical Flow Hardware Accelerator…*, **IEEE TCAS-I 2025**, DOI:10.1109/TCSI.2025.3572918（FPGA；**adaptive OF based on dynamic direction prediction**；405 FPS ZCU104）；FlowAcc **DATE 2022**；Gong et al. OF tracking accel **TCAS-I 2023**.
- **Delta vs our stack:** OP-STW 已有方向 wake；**仍缺** 硬件已验证的 **动态方向预测 → 跳过无关搜索/PE** 控制环（frame OF FPGA 成熟，事件/spike 侧未钉）。可与 BitHyp/TDE3 方向字段融合，但不重复 BitHyp 假设打分器。
- **RTL sketch interface:**
```systemverilog
// dyndir_wake: pred_dir histogram → tile PE enable mask
input  logic        evt_valid_i;
input  logic [2:0]  dir_bin_i;           // from TDE3/BitHyp/plane
input  logic [7:0]  dir_conf_i;
output logic [N-1:0] pe_wake_o;         // → OP-STW
output logic        search_skip_o;       // skip non-pred dirs
```
- **Novelty self-score:** **7.5 / 10** · first-HW: **careful** — frame OF FPGA 已有；claim 限定 **event-spikeformer tile wake with dyn-dir FSM**。
- **Overclaim traps:** 勿把 TCAS-I’25 数字搬到事件芯片；勿称 first OF FPGA。
- **RED?** **no** — 增强 OP-STW 很实，但对 EvQ/PredExit/BitHyp 不是更锐的 letter 刀。

---

### AEC-MatchFB — area-event-count slice exposure closed-loop from match distance
- **Near papers:** Liu & Delbruck, *EDFLOW*, **IEEE TCSVT 2022**, DOI:10.1109/TCSVT.2022.3156653（ABMOF + SFAST；**area event count** 曝光由平均匹配距离反馈）；hARMS **IEEE Access 2022**（RFB 小历史）。
- **Delta vs our stack:** AdaptSlice-Tw / EvQ 开环或几何窗；**仍缺** **匹配质量 → Tw/event-count 闭环**（控制论味道，贴 TCAS）。直接调 OP-STW 输入密度与 PRRC 时间窗。
- **RTL sketch interface:**
```systemverilog
// aec_matchfb: mean match dist → area_event_count setpoint
input  logic        match_valid_i;
input  logic [7:0]  match_dist_i;        // ABMOF-like or hyp score
output logic [15:0] aec_target_o;        // events per slice
output logic        slice_fire_o;        // → SPE / TMA-Agg split
output logic        wake_boost_o;        // → OP-STW when match poor
```
- **Novelty self-score:** **8.0 / 10** · first-HW: EDFLOW **已有** ABMOF 闭环；remake = **闭环接到 SDformer SPE/PRRC/OP-STW**（系统级新）。
- **Overclaim traps:** 勿称 first event OF FPGA（EDFLOW 在先）；ABMOF 大 BRAM 勿照搬进 spikeformer。
- **RED?** **no** — 强工程刀， sharpness 低于 BiSAT/SCI/CFP/BUI。

---

### BL-VetoPrior — Barlow–Levick inhibition-veto digital direction bank (beyond TDE-3)
- **Near papers:** Haessig et al., *Spiking Optical Flow… TrueNorth*, arXiv:1710.09820 / **IEEE TNNLS**-era TrueNorth OF（**Barlow–Levick** inhibition DS units）；Gutierrez-Galan et al., digital TDE, **IEEE TNNLS 2021**；TDE-3 **Frontiers Neurosci. 2025** / arXiv:2402.11662（Reichardt 促进+第三抑制，**不同**电路原语）。
- **Delta vs our stack:** TDE3-Prior = facilitator–trigger (+inhibit reset)；**仍缺** **纯抑制 veto / null-direction 封禁** 的 BL 银行，与 TDE 并联作方向先验，喂 OP-STW / ARM-Acc。数字 RTL（计数器+时间窗 AND-NOT）可综合，避模拟。
- **RTL sketch interface:**
```systemverilog
// bl_veto: preferred enhance + null veto in Δt window
input  logic        fac_spike_i, trg_spike_i, null_spike_i;
input  logic [7:0]  dt_win_i;
output logic        pd_spike_o;          // preferred direction
output logic        veto_o;              // null-dir suppress
output logic [2:0]  dir_code_o;          // → OP-STW / ARM-Acc
```
- **Novelty self-score:** **8.0 / 10** · first-HW: TrueNorth 已有 BL-OF；claim = **first synthesizable digital BL-veto bank gating spikeformer residual OF**（与 TDE3 对照消融）。
- **Overclaim traps:** 勿称 first bio OF HW；勿混 TDE-3 第三突触与 BL veto；TrueNorth ≠ 你们 RTL。
- **RED?** **soft-yes** — 比 BitHyp 更「生物+数字」正交；对 EvQ/PredExit 非直接替代。

---

### BUI-GuardSDSA — bit-wise uncertainty-interval guard filter for predictor-free SDSA sparsity
- **Near papers:** Wang et al., *PADE: A Predictor-Free Sparse Attention Accelerator…*, **HPCA 2026** / arXiv:2512.14322（**BUI-GF** bit-serial guard；无独立 sparsity predictor）；Wang et al., *SOFA*, **MICRO 2024**（leading-zero log sparsity pred；对照）；SpAtten **HPCA 2021**；Sanger / DOTA（PADE 对比基）。
- **Delta vs our stack:** EvQ-Win = 事件邻居队列（几何）；SP-Gate/STH-Gate = attention-mass / 头分型；**仍缺** **逐 bit 不确定区间护栏剪枝**——无额外预测器开销，直接砍 QK trivial tokens。与 HBG-RP 双轨：guard 剪 gate 轨，payload ATLIF 仅在幸存 token 上乘。比 EvQ 更贴 SDSA 能耗刀。
- **RTL sketch interface:**
```systemverilog
// bui_guard: bit-round score interval → keep/drop before payload MAC
input  logic        bit_round_valid_i;
input  logic [7:0]  score_msb_i;         // bit-serial partial score
input  logic [7:0]  ubl_i, lb_i;         // uncertainty bounds
output logic        token_keep_o;        // → HBG gate / SMAM mask
output logic        payload_en_o;        // ATLIF MAC enable
output logic        ooo_ready_o;         // BS-OOE style issue
```
- **Novelty self-score:** **9.0 / 10** · first-HW: PADE 是 **LLM attn ASIC 论文**；claim 限定 **first BUI-guard SDSA for dense event OF spikeformer**（勿称 first sparse attn HW）。
- **Overclaim traps:** PADE/SOFA 非事件 OF；bit-serial 与 ATLIF int8 payload 位宽对齐要写清；勿吸收 payload 进纯二值。
- **RED?** **yes** — **sharper than EvQ-Win** for attention fabric knife。

---

### LZ-SparsePred — leading-zero / log-add-only attention sparsity predictor (SOFA-style)
- **Near papers:** Wang et al., *SOFA*, **MICRO 2024**, DOI via MICRO’24 proceedings / arXiv:2407.10416（**leading-zero computing** log-based add-only sparsity pred + cross-stage tiling）。
- **Delta vs our stack:** BitHyp = occupancy hyp 打分；EvQ = 邻居；**仍缺** **极便宜 log/LZ 分数稀疏预测** 在 SDSA 前级。可作 BUI-Guard 的轻前端或 SP-Gate 特征。
- **RTL sketch interface:**
```systemverilog
// lz_sparsepred: approx log|q·k| via LZ + add → top-k mask
input  logic [15:0] q_i, k_i;
output logic [4:0]  lz_score_o;
output logic        pred_keep_o;         // cheap mask → BUI / EvQ
```
- **Novelty self-score:** **7.5 / 10** · first-HW: SOFA 已有 LLM；OF-spike remake 中等新颖。
- **Overclaim traps:** 勿称 first sparsity predictor HW；精度损失要在 DSEC 上量。
- **RED?** **no** — 便宜辅刀；锐度低于 BUI-GuardSDSA。

---

## Ranked Top list（letter knife order）

| Rank | Name | Why (Fork A) | RED? |
|---|---|---|---|
| 1 | **BiSAT-Agg** | 直接升级 Card G：TMA→双向+SATMA；喂 MFBD/OGEC | **yes** |
| 2 | **BUI-GuardSDSA** | 比 EvQ 更贴 SDSA MAC；无预测器；贴 HBG-RP 双轨 | **yes** |
| 3 | **SCI-CleanExit** | 比 PredExit 锐：scrub + exit；贴 exact_capture 预算 | **yes** |
| 4 | **CFP-ConfGate** | 置信度→OGEC×PRRC×exact；像素级控制环 | **yes** |
| 5 | **NL-STMFA** | 破线性 TMA/MW 假设；长序列残差 | soft-yes |
| 6 | **BL-VetoPrior** | 与 TDE3 正交的抑制先验；数字可综合 | soft-yes |
| 7 | **AEC-MatchFB** | EDFLOW 闭环接到 SPE/OP-STW；工程硬 | no |
| 8 | **DynDir-Wake** | 强化 OP-STW；frame FPGA 可迁 | no |
| — | LZ-SparsePred | BUI 轻前端 | no |

**建议 letter 主刀组合（不换 HBG/OP-STW）：**  
`OP-STW + HBG-RP + OGEC×PRRC→exact` ← **BiSAT-Agg + CFP-ConfGate + SCI-CleanExit + BUI-GuardSDSA**  
（TDE3/TMA/EvQ/PredExit/BitHyp 降为消融基线或辅先验。）

---

## DO NOT CLAIM

- **First event optical-flow hardware** — EDFLOW TCSVT’22, ASNA-Flow TVLSI’25, hARMS Access’22, Aung ISCAS’18, TrueNorth OF, EventShiftFlow’26 已在。  
- **First temporal aggregation for event OF** — TMA ICCV’23（algo）；你们最多 claim **first HW for BAT/TMA-style under spikeformer**。  
- **First sparse attention accelerator** — SpAtten/Sanger/DOTA/SOFA/PADE 已在（LLM/ViT）。  
- **First spiking Transformer HW** — Bishop/FireFly-T/Xpikeformer/SMAM/SPARTA。  
- **Grok46 pure-binary ATLIF / MX3P island swap** — Fork A 禁推。  
- **Vague CIM / analog silicon** — 本文件零 CIM 主贡献。  
- 勿把 **E-RAFT（algo·events）** 与 **ERAFT（FPGA·frames）** 混为一谈。  
- 勿把 BAT/SciFlow/EMD-Flow/E-NMSTFlow 的 **GPU/frame 精度数字** 写成你们芯片结果。  
- 勿 rehash TDE3 / TMA-Agg / EvQ-Win / PredExit / BitHyp 为「新 top」。

---

## Sources（verified; no fabricated venues）

### Temporal aggregation beyond TMA
1. Liu et al., *TMA*, **ICCV 2023**, DOI:10.1109/ICCV51070.2023.00888 / arXiv:2303.11629.  
2. Xu et al., *BAT*, **arXiv:2503.03256** (2025).  
3. *E-NMSTFlow*, **ICRA 2025** / arXiv:2505.05089.  
4. Shi et al., *VideoFlow*, **ICCV 2023**.

### Early-exit / confidence / residual control
5. Lin et al., *SciFlow*, **CVPRW EVW 2024** / arXiv:2404.08135.  
6. Deng et al., *EMD-Flow*, **ICCV 2023**, DOI:10.1109/ICCV51070.2023.00873.  
7. ERAFT FPGA, **ISCAS 2025**, DOI:10.1109/ISCAS56072.2025.11043529（frame；PredExit 近亲）.  
8. Xu et al., *GMFlow*, **CVPR 2022**.

### Event OF HW / FPGA / ASIC（digital）
9. Liu & Delbruck, *EDFLOW*, **IEEE TCSVT 2022**, DOI:10.1109/TCSVT.2022.3156653.  
10. Wang et al., *ASNA-Flow*, **IEEE TVLSI 2025**, DOI:10.1109/TVLSI.2025.3600953.  
11. Stumpp et al., *hARMS*, **IEEE Access 2022**, DOI:10.1109/ACCESS.2022.3172396.  
12. *Ultra-high performance OF FPGA*, **IEEE TCAS-I 2025**, DOI:10.1109/TCSI.2025.3572918.  
13. FlowAcc, **DATE 2022**, DOI:10.23919/DATE54114.2022.9774506.  
14. Alonso Bizzi et al., *EventShiftFlow*, **arXiv:2605.28312** (2026) — BitHyp 源；本轮不重推。  
15. Xu et al., event OF on SENECA, **arXiv:2407.20421**.

### Sparse attention fabrics（digital RTL-friendly）
16. Wang et al., *PADE*, **HPCA 2026** / arXiv:2512.14322.  
17. Wang et al., *SOFA*, **MICRO 2024** / arXiv:2407.10416.  
18. Gao et al., *ESDA*, **FPGA 2024**, DOI:10.1145/3626202.3637558（SubMan-Pipe 已命名；本轮不重推）.  
19. Yang et al., *EvGNN*, **IEEE TCAS-AI 2024**（EvQ-Win 源；不重推）.

### Bio / digital timing & coincidence beyond TDE-3
20. Gutierrez-Galan et al., digital TDE, **IEEE TNNLS 2021**, DOI:10.1109/TNNLS.2021.3108047.  
21. TDE-3, **Frontiers Neurosci. 2025** / arXiv:2402.11662.  
22. Haessig et al., TrueNorth BL-style spiking OF, arXiv:1710.09820.  
23. Greatorex et al., timing OF / synaptic gating, **CVPRF 2026** / egomotion arXiv:2501.11554；TDE circuit arXiv:2501.10155（混合信号旁支；主推 BL-Veto 数字银行时仅作对照）.

### Stack / prior ideafromai
24. `11_event_camera_stack_accelerators.md`, `09_ROUND2_SYNTHESIS_C1_C2_UPGRADE.md`, `04_SYNTHESIS_C1_C2_REMAKE.md`, `12_ROUND3_SYNTHESIS_CIM_EVENT.md`.

---

## 工程下一步（不自动开干）

1. ep34：双向 corr 一致性、SCI map vs AEE、conf→exact 配额、BUI keep-rate。  
2. RTL 优先序：**BiSAT-Agg microcode**（扩 Card G）→ **CFP-ConfGate**（接 OGEC/PRRC）→ **SCI-CleanExit**（接 PredExit）→ **BUI-Guard**（接 HBG/SMAM）。  
3. 消融梯子：TMA-only / +BiSAT / +SCI / +CFP / +BUI；指标 AEE + PE wake + SOP/J。  
4. 基线引用：EDFLOW、ASNA-Flow、E-RAFT(GPU)、TMA(GPU)、BAT(GPU)、ERAFT(frame)、FireFly-T-style attn energy。

---

*End of 18_POST_TDE3_TMA_HARD_KNIVES.md — Grok Bot executor · 2026-09-06 CST*
