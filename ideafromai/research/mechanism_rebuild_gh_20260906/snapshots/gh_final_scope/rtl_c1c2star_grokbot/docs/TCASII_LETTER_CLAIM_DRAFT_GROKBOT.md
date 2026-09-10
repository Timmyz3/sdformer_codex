# TCAS-II Letter Claim Draft (Grok Bot) — 3-Knife Spine

**Status:** DRAFT — 3-knife letter spine (proposal vs capture discipline)  
**Date:** 2026-09-06 (Asia/Shanghai)  
**Tree:** `sdformer_c1c2star_grokbot` / `tcasii/c1c2star-oss`  
**Hygiene:** No silicon µ²/mW; no 18-module laundry list; OpenROAD = implementation completeness only.

---

## English (≤1 page)

**Title (working):** Motion-Scheduled Exact Capture and Hybrid Binary-Gate Real Payload for Spiking Dense Optical Flow

**Three knives only.**

| Knife | Name | Claim edge |
|---|---|---|
| **1** | **OP-STW** (+ optional **TDE3-Prior** lane) | Optical-flow Δ + event density schedules *which tiles wake* for exact work. **TDE3-Prior** adds a digital time-difference prior that seeds OF wake (bio TDE-3 texture guard) — enhancer of OP-STW, **not** zero-skip. **Novelty enhancers under knife-1:** **BL-VetoPrior** (inhibition veto bank) + **NL-STMFA** (nonlinear residual wake) — orthogonal to TDE3 facilitation / linear MW·TMA. |
| **2** | **HBG-RP** | Binary *gate* clocks PEs while carrying a **non-absorbable** ATLIF **int8 proposal** payload. **Explicit:** frozen ep35 capture is **binary** today; int8 is a co-design *proposal*, not a deployed identity. |
| **3** | **OGEC×PRRC→exact_capture** | Exact products enqueue only under match-gate ∧ residual budget ∧ finite capacity — not a flat quota. (**CFP-ConfGate** + **SCI-CleanExit** sharpen this exact path via conf-gated exact/prop + scrub/exit hold; **TMA-Agg** = supporting temporal fabric; **BiSAT-Agg** upgrades TMA temporal knife; **BUI-GuardSDSA** guards HBG MAC — still 3-knife letter, these are enhancers.) |

**Supporting (not letter knives):** ECP-QKV, MW-ΔBuf, SMAM-RP, ADP-MAC, STH/SP/ARM/MFBD, TDE3/TMA/BiSAT, CFP/SCI (knife-3 enhancers), TID/EDC (Card I exact/wake side), BUI (HBG guard), stats — ablation / roadmap layers only.

**Evidence posture.** Directed RTL sims + Yosys/abc cell counts; C1*/C2* ablation ladders (wake_pop / proj_skip / exact_hit / mac_en). OpenROAD sky130hd best-effort = *completeness* (DRC=0 routes), **not** silicon PPA. **No** µ²/mW from routed area; **no** AEE without algo co-sim.

**Hard negatives.** Not Bishop/AAC binary-only; not FACT/FlightVGM/ERAFT end-to-end; not multiply-reorder; not DualRail-CIM physical arrays; not MX3P/Grok46 binary fork; not laundry-list of ~18 modules as 18 innovations.

---

## 中文（一页内）

**题目（工作稿）：** 面向脉冲稠密光流的运动调度精确捕获与混合二元门控实幅载荷加速

**三刀脊柱（仅此三刀）。**

| 刀 | 名称 | 刀刃 |
|---|---|---|
| **1** | **OP-STW**（+可选 **TDE3-Prior**） | 光流差分 + 事件密度调度“哪些 tile 唤醒做精确计算”。TDE3-Prior 以数字时差先验增强 OF wake（生物 TDE-3 纹理抑制），**不是**跳零。**刀1 新颖增强：BL 抑制否决库 + NL-STMFA 非线性残差唤醒。** |
| **2** | **HBG-RP** | 二元门控驱动 PE，同时保留 **不可吸收** 的 ATLIF **int8 提案**载荷。**明示：** ep35 冻结捕获今日为**二值**；int8 是协同设计提案，不是已部署身份。 |
| **3** | **OGEC×PRRC→exact_capture** | 精确乘积仅在匹配门控 ∩ 残差预算 ∩ 有限容量下入队。**CFP-ConfGate** + **SCI-CleanExit** 以置信门控与 scrub/exit hold 锐化该 exact 路径；**TMA-Agg** 为支撑性时间织物；**BiSAT-Agg** 升级 TMA 时序刀，**BUI-GuardSDSA** 护栏 HBG MAC——仍为三刀信件，二者为增强器。 |

**支撑层（不进信件主刀）：** ECP / MW / SMAM / ADP / STH·SP·ARM·MFBD / TDE3·TMA·BiSAT / CFP·SCI（锐化刀3）/ BUI（HBG 护栏）/ stats —— 仅消融或路线图。

**证据边界。** RTL 定向仿真 + 消融计数；OpenROAD 仅作实现完整度。无硅基 µ²/mW；无联合仿真不报 AEE。

**硬性否定。** 非纯二元 AAC；非端到端精度吞并；非乘法重排；非 DualRail-CIM 物理阵列；非模块动物园当贡献列表。

---

## Suggested Table I (ablation)

| Rung | C1* | Metric hooks |
|---|---|---|
| always-ish | all-tile wake | wake_pop↑ exact_hit↑ |
| OP-STW | flow/event wake | wake_pop↓ |
| +ECP | +corr skip | proj_skip↑ |
| +MW | +Δ residual OR | delta_nz, wake_pop mid |
| +OGEC/PRRC (`EXACT`) | gated exact | exact_hit **capped** vs MW |

See `out/ABLATION_C1_LADDER.md` for measured RTL numbers. TDE3-Prior / TMA-Agg prove-by counters in Card G TB logs.
