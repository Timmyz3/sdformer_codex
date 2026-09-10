# ROUND 3 合成 — CIM + Event-Camera → C1*/C2*（Grok Bot / iscas_ssh）

**日期:** 2026-09-05  
**输入:** `10_cim_spike_opticalflow_accelerators.md` + `11_event_camera_stack_accelerators.md`  
**相对 R2 `09`:** 本轮补 **存算一体织物** 与 **事件相机全栈**；不替换 HBG-RP / ARM-Acc / OP-STW。

---

## C1* 再升级（叠在 R2：ECP-QKV / MW-ΔBuf / EV-Wake 上）

| 代号 | 作用 | 来源 |
|---|---|---|
| **MW-CIM-TileGate** | 翘曲残差能量同时门控 PE wake **与** CIM tile | R3-10 |
| **Het-OF-CIM** | frame/ref 走 nvCIM，event/spike 走 SRAM-CIM | R3-10 |
| **PRRC-CIM-Modes** | 金字塔层切换 CIM 精度/模式 | R3-10 |
| **TDE3-Prior / BitHyp-Prior** | 超便宜生物/比特假设先验 → OP-STW/ECP | R3-11 |
| **AdaptSlice-Tw / SkipEdge-ROI / EV-Wake+** | 自适应时间片 + 传感器 ROI + 加深事件唤醒 | R3-11 |
| **ShiftTS-SPE** | 时间面/移位编码进 SPE | R3-11 |

**Claim 增量:** 前端不仅预测相关，还把 **事件先验与 CIM tile 活性** 绑在同一 wake bitmap 上。

---

## C2* 再升级（叠在 R2：SMAM-RP / Motion-TTB / STH-Gate 上）

| 代号 | 作用 | 来源 |
|---|---|---|
| **DualRail-CIM / Fused-WVMEM-CIM** | gate=spike-CIM，payload=DigiCIM；W∥Vmem 融合 | R3-10 |
| **XForm-Split-SDSA / NeuroGate-CIM** | 静态 NVM vs 动态 SDSA；神经元早停门 | R3-10 |
| **TMA-Agg** | 时序运动聚合（强 first-HW 候选） | R3-11 |
| **EvQ-Win / DualEng-SDSA / SubMan-Pipe** | 事件邻域窗 / 双引擎 SDSA / 子流形管线 | R3-11 |

**Claim 增量:** 双轨不仅是数字 MAC 分轨，还可落到 **DualRail-CIM 物理织物**；时间侧用 **TMA-Agg / EvQ-Win** 喂 MFBD。

---

## 联合故事（R1–R3）

1. **传感/先验** TDE3/BitHyp/AdaptSlice/EV-Wake+  
2. **前端** ECP-QKV + MW-ΔBuf + MW-CIM-TileGate / Het-OF  
3. **中段** Motion-TTB + STH-Gate + TMA-Agg / EvQ-Win  
4. **后端** SMAM-RP + DualRail-CIM / ADP-MAC  

**RTL 仍先 Card A/B**；建议后续 **Card F = CIM fabric（DualRail/Het/Fused）**，event 侧 **Card G = TMA-Agg / EvQ-Win**。

---

## DO NOT CLAIM（R3）

- 首个 CIM / SNN / Transformer / event-OF / DVS 硅  
- 泛泛 TOPS/W SRAM-CIM、单靠 ADC-less  
- 把 E-RAFT（算法）当成 ERAFT（帧 FPGA）  
- Softmax-in-RRAM 当 spikeformer 主卖点  

---

## 仍开放

ISSCC/VLSI/HotChips **edge NPU** 横向对标（Round-4 候选）。

## 底稿

- `10_cim_…` · `11_event_…` · 本文件 `12_ROUND3_SYNTHESIS_CIM_EVENT.md`  
- 包名叙事以 `09`（R2）+ 本文件（R3）叠加为准；`04` 仍为 R1 基线。
