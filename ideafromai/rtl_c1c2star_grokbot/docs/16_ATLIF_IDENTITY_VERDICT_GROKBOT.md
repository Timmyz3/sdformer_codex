# ATLIF identity verdict — who is right? (Grok Bot after file dig)

**Date:** 2026-09-05 · **Method:** read ismd `ideafromai` Grok46 + Codex packs + original `nts11_hardware_p0_profile.md/json` under `synopsys_date_dual/...ep35...`

---

## Short verdict

| Claim | Verdict | Scope |
|---|---|---|
| **Grok46:** captured ATLIF outputs are **binary** (93 modules, ternary=0) | **Supported by profile file** | That **ep35 / one-sample** P0 profile |
| **Grok46:** those numbers are sealed **ep34** paper identity | **Overstated** | Profile header is **checkpoint_epoch35.pth**, samples=1; Grok46 `05` itself says “not sealed ep34” |
| **Grok Bot:** HBG **int8 non-absorbable payload** | **Correct as NEW co-design draft** | Our `ATLIF_contract_r1` explicitly says *new co-design only*, not “ep34 already int8” |
| **Grok Bot:** int8 is already true of frozen deploy capture | **Not supported** | No evidence in the profile that fires carry int8 amplitudes |
| **Codex independent:** C1/C2 novelty weak | **Agree** (orthogonal to ATLIF bitwidth) | Keep C1/C2 as baseline/context unless remade |

**Neither pack is “all wrong.”** They answer different questions.

---

## Evidence dug (primary)

**File:**  
`/home/zhumd/work/synopsys_date_dual/h67_ep35_real_tile_trace_s1_p64_2d_r2_20260822/.../nts11_hardware_p0_profile.md`

| Field in profile | Value |
|---|---|
| Checkpoint | `.../checkpoint_epoch35.pth` (H67 ep40 run folder, **epoch35**) |
| Samples | **1** |
| ATLIF modules installed | 105 `ATLIFTernaryPSN` |
| ATLIF recorded | **93** |
| Activity snapshot | ternary **0** modules · binary **93** · pos_rate=activity · **neg_rate=0** |
| Threshold mode | `official_atlif` · inference threshold = checkpoint static param |

JSON `atlif_summary` matches: `binary_activity_mean≈0.056`, ternary means 0, `official_atlif_modules=105`.

Grok46 `05_profile_evidence.md` correctly labels this as **directional / ep35 path / one sample**, and tells Codex **not** to paste into sealed ep34 tables — but `00_READ_THIS_FIRST` still freezes “all 93 captured outputs binary” as ep34 identity. **That freeze step is the soft spot.**

---

## What our contract already admitted

`sdformer_c1c2star_grokbot/docs/ATLIF_contract_r1_grokbot.md`:

- Status: **DEFAULT DRAFT**
- Scope: **NEW co-design only**
- NeurIPS’25 AT-LIF `{0,θ}` **≠** this contract; document difference
- If amps absorbable into W → downgrade HBG-RP

So HBG-RP RTL is **not falsified** by the binary profile; it is a **different product assumption** that needs AEE/Pareto vs binary deploy.

---

## How to speak in TCAS-II without lying

**Safe (Fork A — continue existing RTL):**  
“We co-design a **non-absorbable int8 ATLIF payload** datapath (HBG-RP / SMAM-RP / ADP-MAC) and OF-scheduled wake (OP-STW / ECP / MW). Deployed H67 profiles on ep35 show **binary fire** today; int8 is the **proposed** contract, not the sealed capture. Binary MX3P (Grok46) remains a **leaf** candidate under frozen binary identity.”

**Unsafe:**  
“ep34 already has int8 ATLIF amplitudes” — **no file supports this**.  
“System 1.5× from attention MX3P alone” — Grok46 envelope caveat; needs T0 on ep34.

---

## Action while “先做现有的”

1. Keep building / verifying **C1\*/C2\* OSS + OpenROAD** stack (already: regress, C1/C2 ablations, HBG floorplan ~121 u² floorplan-only).  
2. Label every paper sentence: **co-design proposal** vs **frozen capture fact**.  
3. On sd5 A800 later: train/eval int8-payload Pareto vs binary baseline (the missing truth for Fork A).  
4. Optionally still run Grok46 T0–T5 on **ep34** to decide if MX3P is leaf-only — without stopping current HW.

---

## One-line for the user

**今天部署捕获更像二值（有 ep35 profile 为证）；我们的实值载荷是另开的新合同（合同文件自己这么写的）。继续做现有 RTL 可以，但论文里必须写成 co-design，不能写成“冻结 ep34 已经是 int8”。**
