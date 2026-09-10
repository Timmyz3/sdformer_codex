# 21 — TCAS-II Novelty Boost Top5 (Fork A)

**Date:** 2026-09-06 (Asia/Shanghai)  
**Fork:** **A only** — three-knife spine unchanged  
**Purpose:** novelty boost after independent rescore ≈ **3.43** (Borderline→target lift via sharper first-HW remakes)  
**Author:** Grok Bot (ideafromai research)  
**Hard filter:** enhance **ONLY** OP-STW(+TDE3) / HBG-RP int8 *proposal* / OGEC×PRRC→exact(+CFP/SCI). Supporting TMA / BiSAT / BUI already landed (Card G/H).  
**Ban:** MX3P / pure-binary ATLIF island swap; vague CIM / analog / PIM (do **not** remake ASTER); restating Card G/H (TDE3/TMA/CFP/SCI/BiSAT/BUI) as “new top”.

---

## Gap vs Card G/H + soft knives

| Already landed / soft | Why not enough for ≈3.43→letter lift | What Top5 adds |
|---|---|---|
| **Card G:** TDE3-Prior, TMA-Agg | Bio prior + linear Tw split/lookup; no HTR residual fabric, no corr-free deblur loop | ResHTR residual refiner; TID deblur loop |
| **Card H:** BiSAT, CFP, SCI, BUI | Bidirectional fuse / conf budget / scrub / bit-guard — **done**; do not restate | Orthogonal remakes only |
| **NL-STMFA** (soft) | Nonlinear warp residual idea; heavy pyramid / no regional-noise HTR training posture | **ResHTR-Refiner** = linear seed + HF residual + regional-noise style tile pattern (sharper, HW-shaped) |
| **BL-VetoPrior** (soft) | Inhibition bank orthogonal to TDE3; not residual / sparsity fabric | Out of Top5; keep as ablation prior |
| **AEC-MatchFB** (soft) | Exposure closed-loop; EDFLOW already has ABMOF loop | Out of Top5 |
| **DynDir-Wake** (soft) | Frame-OF FPGA dyn-dir skip; weaker letter novelty | Out of Top5 |
| **EvQ / PredExit / BitHyp** | Geometry wake / binary halt / occupancy hyp — idea-stage, lower sharpness | WinTok co-sparsity ≫ EvQ; TID/EDC ≫ PredExit binary; AdjEvt ≫ BitHyp for MAC fabric |

**One-line gap:** after G/H the letter still lacks (i) **HTR residual refine** on TMA/BiSAT seeds, (ii) **Δ-feature fuse** before exact, (iii) **window×token co-sparsity** for wake/BUI, (iv) **adjacent-event digital compress** into HBG/SDSA, (v) **corr-volume-free iterative deblur** control loop on exact/PRRC path.

---

## Top5 remakes

### 1. ResHTR-Refiner — residual HTR refine on linear TMA/BiSAT seed

**One-line mechanism:** Decompose tile flow into **global linear seed** (from `c2s_tma_agg` / `c2s_bisat_agg`) + **high-frequency residual refiner**; inject **regional-noise–style** residual patterns on HTR tiles → residual wake / PRRC fine levels.

**Near papers:** Zhou et al., *ResFlow: Fine-tuning Residual Optical Flow for Event-based High Temporal Resolution Motion Estimation*, **arXiv:2412.09105** (2024); TMA ICCV 2023 (linear temporal agg baseline); BAT arXiv:2503.03256 (bi-temporal cousin already wired as BiSAT).

**Delta vs stack:** NL-STMFA = nonlinear warp residual sketch; **ResHTR** is sharper: explicit **linear seed + shared residual refiner** + regional-noise training *posture* mapped to digital tile noise banks — wires to MW-ΔBuf / OP-STW residual wake / PRRC fine levels. **Not** Card G TMA restatement (TMA supplies seed only).

**Module ports (SV-ish):**
```systemverilog
module c1s_reshtr_refiner #(
  parameter int N_TILE   = 8,
  parameter int FLOW_W   = 8,
  parameter int FEAT_W   = 8,
  parameter int RES_W    = 8,
  parameter int NOISE_S  = 4,          // regional upsample scale (ResFlow S≈6 → HW 4)
  parameter int N_ITER   = 2,          // residual refine iters (keep small for RTL)
  parameter logic [RES_W-1:0] TH_WAKE = 8'd24,
  parameter bit ABLATE_SEED_ONLY = 1'b0  // 1 = pass linear seed, zero residual
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [FLOW_W-1:0]            seed_flow   [N_TILE], // ← TMA/BiSAT linear
  input  logic [FEAT_W-1:0]            feat_htr    [N_TILE], // local HTR / slice feat
  input  logic [N_TILE-1:0]            tile_en,
  input  logic                         noise_en_i,           // TB: regional noise on
  input  logic [RES_W-1:0]             noise_tile  [N_TILE], // precomputed Up(G,S) proxy
  output logic [FLOW_W-1:0]            flow_htr    [N_TILE], // seed + residual
  output logic [RES_W-1:0]             residual    [N_TILE],
  output logic [N_TILE-1:0]            res_wake,             // → wake_merge / OP-STW
  output logic [3:0]                   prrc_fine_lvl,        // → PRRC fine window
  output logic                         valid_o
);
```

**Wiring to existing modules:**
| From / To | Signal |
|---|---|
| ← `c2s_tma_agg.agg_flow` / `c2s_bisat_agg.fuse_flow` | `seed_flow` |
| ← slice / MW feat | `feat_htr` |
| → `c1s_wake_merge` (OR with op_stw\|tde\|delta) | `res_wake` |
| → `c1s_op_stw_predictor` residual lane | `residual` / `res_wake` |
| → `c1s_mw_delta_buf` | residual delta |
| → `c1s_prrc_ledger` fine grant policy | `prrc_fine_lvl` |

**Novelty /10 + first-HW posture:** **~9.0** · careful first-HW: **first synthesizable digital ResFlow-class linear-seed + HTR residual refiner under spikeformer OF schedule** (algo is GPU; claim HW fabric only).

**Overclaim traps:** Do not claim inventing ResFlow / HTR GT; do not claim 150 Hz ASIC AEE; regional noise is **training-inspired HW stimulus**, not “learned noise ASIC”; freeze global stage ≡ use TMA/BiSAT as frozen seed.

**Prove-by counters:**
| Counter | Meaning |
|---|---|
| `cnt_res_nz` | tiles with \|residual\| > 0 |
| `sum_abs_res` | residual energy |
| `cnt_res_wake` | popcount(res_wake) sum |
| `cnt_fine_bump` | prrc_fine_lvl increments |
| Ablation | SEED_ONLY vs FULL: wake_pop / capture under same MW |

---

### 2. EDC-ΔFuse — multi-scale temporal Δ-feat + adaptive fuse before exact

**One-line mechanism:** Build **multi-scale temporal feature-difference maps**, adaptively fuse with **low-res correlation** motion features; plug-in refine **before** `exact_capture` to sharpen ECP/OGEC match quality.

**Near papers:** Liu et al., *EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation*, **arXiv:2506.03512** (2025); TMA ICCV 2023 / E-RAFT 3DV 2021 (corr baselines EDC improves); IDNet ICRA 2024 (corr-free cousin — see #5).

**Delta vs stack:** ECP-QKV / OGEC use match/corr peaks; CFP gates conf — **still missing** explicit **Δ-feat multi-scale** path fused with low-res corr as a **refine stage** into exact. Sharper than NL-STMFA (warp residual) and PredExit (binary halt): continuous Δ-motion features, not just exit.

**Module ports (SV-ish):**
```systemverilog
module c1s_edc_delta_fuse #(
  parameter int N_TILE   = 8,
  parameter int FEAT_W   = 8,
  parameter int CORR_W   = 16,
  parameter int MOT_W    = 8,
  parameter int N_SCALE  = 3,            // s = {1,2,5} style
  parameter logic [MOT_W-1:0] TH_DETAIL = 8'd32,
  parameter bit ABLATE_CORR_ONLY = 1'b0,
  parameter bit ABLATE_DIFF_ONLY = 1'b0
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [FEAT_W-1:0]            feat_t0   [N_TILE],
  input  logic [FEAT_W-1:0]            feat_tn   [N_TILE], // warped / adjacent
  input  logic [CORR_W-1:0]            corr_lo   [N_TILE], // low-res corr / ECP peak
  input  logic [N_TILE-1:0]            match_ok,           // ← OGEC pre
  output logic [MOT_W-1:0]             mot_diff  [N_TILE],
  output logic [MOT_W-1:0]             mot_fuse  [N_TILE],
  output logic [N_TILE-1:0]            detail_hit,         // boundary / texture
  output logic [N_TILE-1:0]            exact_boost,        // → AND/OR with CFP exact_req
  output logic                         fuse_valid
);
```

**Wiring to existing modules:**
| From / To | Signal |
|---|---|
| ← `c1s_ecp_qkv_predictor` / corr bank | `corr_lo` |
| ← feat / MW warped neighbors | `feat_t0`, `feat_tn` |
| ← `c1s_ogec_gate` match pre | `match_ok` |
| → `c1s_cfp_confgate.corr_peak` (optional enrich) | fused peak proxy |
| → combine with `c1s_cfp_confgate.exact_req` | `exact_boost` |
| → `c1s_exact_capture_wrap` via OGEC×PRRC | indirect refine-before-exact |
| → `c1s_ogec_gate` | detail-aware match assist |

**Novelty /10 + first-HW posture:** **~8.7** · first-HW: **first digital EDC-style Δ-feat×low-res-corr fuse as plug-in refine before spikeformer exact_capture** (EDCFlow = GPU algo; claim RTL fuse fabric).

**Overclaim traps:** Do not claim beating TMA/EDCFlow AEE on ASIC; do not replace OGEC — **enhance** match/exact path; high-res cost volume banned (EDC’s point is *avoiding* it).

**Prove-by counters:**
| Counter | Meaning |
|---|---|
| `sum_mot_diff` / `sum_mot_fuse` | Δ vs fused energy |
| `cnt_detail_hit` | boundary tiles |
| `cnt_exact_boost` | tiles boosting exact |
| Ablation | CORR_ONLY / DIFF_ONLY / FULL: `capture_cnt`, `detail_hit` |

---

### 3. WinTok-CoSparse — window×token co-sparsification → wake / BUI keep

**One-line mechanism:** Score **window × token** jointly (scene-adaptive co-sparsity); emit `wake_bitmap` + BUI-compatible **keep mask** — sharper than EvQ geometry-only neighbor gather.

**Near papers:** *SAST: Scene Adaptive Sparse Transformer* for event detection, **CVPR 2024**; EvGNN IEEE TCAS-AI 2024 (EvQ-Win geometry prior — contrast); PADE HPCA 2026 / arXiv:2512.14322 (BUI bit-guard already landed — co-sparse is *front* mask, not bit-interval).

**Delta vs stack:** EvQ = window geometry; BUI = bit-uncertainty keep on scores; SP/STH = mass/head gates. **WinTok** adds **joint window×token sparsification score** → wake_bitmap **and** keep mask into BUI/HBG — letter-sharp vs EvQ.

**Module ports (SV-ish):**
```systemverilog
module c1s_wintok_cosparse #(
  parameter int N_TILE   = 8,
  parameter int N_TOK    = 8,            // tokens per tile window
  parameter int SCORE_W  = 8,
  parameter int WIN_W    = 4,
  parameter logic [SCORE_W-1:0] TH_WIN = 8'd40,
  parameter logic [SCORE_W-1:0] TH_TOK = 8'd48,
  parameter bit ABLATE_WIN_ONLY = 1'b0,  // EvQ-like geometry
  parameter bit ABLATE_TOK_ONLY = 1'b0
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [SCORE_W-1:0]           win_score [N_TILE],
  input  logic [SCORE_W-1:0]           tok_score [N_TILE][N_TOK],
  input  logic [WIN_W-1:0]             win_id    [N_TILE],
  output logic [N_TILE-1:0]            wake_bitmap,        // → wake_merge / OP-STW
  output logic [N_TILE-1:0]            win_keep,
  output logic [N_TOK-1:0]             tok_keep  [N_TILE], // → BUI / SMAM
  output logic                         sparse_valid
);
```

**Wiring to existing modules:**
| From / To | Signal |
|---|---|
| ← event / SPE window stats | `win_score`, `win_id` |
| ← SDSA / token pre-scores | `tok_score` |
| → `c1s_wake_merge` | `wake_bitmap` |
| → `c1s_op_stw_predictor` | wake side-channel |
| → `c2s_bui_guard_sdsa` (AND with token_keep path) | `tok_keep` |
| → `c2s_smam_rp` / `c2s_sp_gate` | keep masks |

**Novelty /10 + first-HW posture:** **~8.5** · first-HW: **first window×token co-sparse scorer driving OF wake_bitmap + BUI keep under SDformer stack** (SAST = detection algo; do not claim first sparse transformer HW).

**Overclaim traps:** SAST is event *detection*, not OF — claim **mechanism remake** only; do not subsume BUI (BUI remains bit-guard; WinTok is co-sparse *proposal*); not EvQ rename.

**Prove-by counters:**
| Counter | Meaning |
|---|---|
| `cnt_wake` | popcount(wake_bitmap) sum |
| `cnt_tok_keep` | kept tokens |
| `cnt_win_drop` | windows killed |
| Ablation | WIN_ONLY (EvQ-like) vs TOK_ONLY vs FULL: payload_fire / wake_pop |

---

### 4. AdjEvt-Compress — adjacent-position event compression before SDSA/HBG MAC

**One-line mechanism:** **Adjacent-position event compression** (digital sparsity fabric) before SDSA / HBG MAC — cut redundant spatial event tokens; **not** CIM.

**Near papers:** *ExSpike*, **arXiv:2606.20414** (FPGA full-event; adjacent-position event compression + attention core); contrast ASNA-Flow spatial locality (already claimed elsewhere — do not rehash).

**Delta vs stack:** BitHyp = occupancy hyp; EvQ = neighbor queue; BUI = score bit-guard. **AdjEvt** compresses **event positions** themselves into denser tokens before MAC — orthogonal sparsity *input* fabric for knife-2 HBG-RP.

**Module ports (SV-ish):**
```systemverilog
module c2s_adjevt_compress #(
  parameter int N_TILE   = 8,
  parameter int N_EVT    = 16,           // raw events / tile window
  parameter int N_OUT    = 8,            // compressed slots
  parameter int COORD_W  = 8,
  parameter int POL_W    = 1,
  parameter int TS_W     = 8,
  parameter logic [COORD_W-1:0] TH_ADJ = 8'd1, // adjacent chebyshev ≤1
  parameter bit ABLATE_BYPASS = 1'b0     // 1 = pass-through first N_OUT
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [COORD_W-1:0]           ev_x [N_EVT],
  input  logic [COORD_W-1:0]           ev_y [N_EVT],
  input  logic [POL_W-1:0]             ev_p [N_EVT],
  input  logic [TS_W-1:0]              ev_t [N_EVT],
  input  logic [N_EVT-1:0]             ev_valid,
  output logic [COORD_W-1:0]           cmp_x [N_OUT],
  output logic [COORD_W-1:0]           cmp_y [N_OUT],
  output logic [POL_W-1:0]             cmp_p [N_OUT],
  output logic [TS_W-1:0]              cmp_t [N_OUT],
  output logic [N_OUT-1:0]             cmp_valid,
  output logic [7:0]                   compress_ratio_q8,  // N_kept/N_in proxy
  output logic                         cmp_fire            // → HBG / SDSA issue
);
```

**Wiring to existing modules:**
| From / To | Signal |
|---|---|
| ← event front / SPE | raw `ev_*` |
| → `c2s_hbg_rp_packetizer` token/event in | compressed stream |
| → `c2s_bui_guard_sdsa` (fewer tokens to score) | `cmp_valid` population |
| → `c2s_smam_rp` / SDSA schedule | `cmp_fire` |
| **Not** | any CIM / crossbar analog macro |

**Novelty /10 + first-HW posture:** **~8.5** · first-HW: **first adjacent-position event compress fabric in front of HBG-RP int8 proposal MAC** under this stack (ExSpike = FPGA full-event paper — cite as near; claim *integration posture* + digital-only, not “first event FPGA”).

**Overclaim traps:** ExSpike already FPGA — do **not** claim first event-compression FPGA; ban CIM/PIM wording; do not absorb HBG payload into pure binary.

**Prove-by counters:**
| Counter | Meaning |
|---|---|
| `cnt_in_evt` / `cnt_out_slot` | compression |
| `avg_ratio_q8` | compress_ratio |
| `cnt_cmp_fire` | issue pulses |
| Ablation | BYPASS vs COMPRESS: `payload_fire`, MAC cycles proxy |

---

### 5. TID-DeblurLoop — iterative event deblur without correlation volume

**One-line mechanism:** Hardware **temporal iterative deblurring (TID)** control loop: motion-compensate / deblur event bins with prior flow seed → residual Δflow → warm-start next step — **no 4D correlation volume**; feeds OGEC×PRRC→exact and OP-STW residual path.

**Near papers:** Wu, Paredes-Vallés, de Croon, *Lightweight Event-based Optical Flow Estimation via Iterative Deblurring* (**IDNet**), **ICRA 2024** / **arXiv:2211.13726** (ID + TID schemes; corr-volume-free); contrast E-RAFT / TMA (corr-heavy); EVA-Flow arXiv:2307.05033 (anytime / on-the-fly cousin — runner only).

**Delta vs stack:** PredExit = residual&lt;ε halt; SCI = scrub/exit on warp quality; TMA/BiSAT = temporal agg **with** lookup seeds. **TID-DeblurLoop** is sharper: **explicit deblur→residual→warm-start** FSM without building corr volumes — directly attacks exact-path compute and residual wake. **Replaces** prior SwinShift-SDSA candidate (would have restated SDformerFlow shifted-window algo; weaker vs G/H SDSA already present).

**Module ports (SV-ish):**
```systemverilog
module c1s_tid_deblur_loop #(
  parameter int N_TILE   = 8,
  parameter int FLOW_W   = 8,
  parameter int FEAT_W   = 8,
  parameter int BIN_W    = 8,
  parameter int N_BIN    = 4,            // event bins per Tw
  parameter int N_ITER   = 2,            // ID-style iters; TID uses 1 + time
  parameter logic [FLOW_W-1:0] TH_DRES = 8'd6,
  parameter bit MODE_TID = 1'b1,         // 1=TID online; 0=ID multi-iter same batch
  parameter bit ABLATE_NO_DEBLUR = 1'b0  // pass raw bins (IDNet ablative cousin)
) (
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         valid_i,
  input  logic [BIN_W-1:0]             evt_bin   [N_TILE][N_BIN],
  input  logic [FLOW_W-1:0]            flow_seed [N_TILE], // ← TMA/BiSAT/OP-STW
  input  logic [FLOW_W-1:0]            flow_hat_prev [N_TILE], // TID prior
  input  logic                         iter_fire_i,
  output logic [BIN_W-1:0]             evt_deblur[N_TILE][N_BIN],
  output logic [FLOW_W-1:0]            dflow     [N_TILE], // residual ΔF
  output logic [FLOW_W-1:0]            flow_out  [N_TILE], // seed + ΔF
  output logic [FLOW_W-1:0]            flow_hat_next [N_TILE],
  output logic [N_TILE-1:0]            deblur_wake,        // → OP-STW / wake_merge
  output logic                         exact_pref,         // prefer exact when Δ large
  output logic                         loop_valid
);
```

**Wiring to existing modules:**
| From / To | Signal |
|---|---|
| ← `c2s_tma_agg` / `c2s_bisat_agg` | `flow_seed` |
| ← prior cycle `flow_hat_next` | `flow_hat_prev` |
| → `c1s_op_stw_predictor` / `c1s_wake_merge` | `deblur_wake` |
| → `c1s_ogec_gate` / `c1s_cfp_confgate` | `exact_pref` side-channel |
| → `c1s_prrc_ledger` / `c1s_exact_capture_wrap` | large Δ → hold/boost exact |
| → `c1s_sci_cleanexit` (optional) | deblur quality vs SCI scrub |
| **Not** | 4D corr SRAM macro |

**Novelty /10 + first-HW posture:** **~8.8** · first-HW: **first synthesizable TID/IDNet-class iterative deblur loop (corr-free) under SDformer C1\* exact/wake schedule** (IDNet = Jetson/GPU algo; claim digital FSM + deblur datapath, not “first event OF”).

**Overclaim traps:** Do not claim inventing IDNet/TID; do not report Jetson 8 ms latency as ASIC; ID vs TID modes must be ablated; MVSEC weakness of IDNet is algo-data — do not hide; **no** corr-volume “also supported” weasel that reintroduces TMA-scale SRAM.

**Prove-by counters:**
| Counter | Meaning |
|---|---|
| `cnt_deblur` | deblur pulses |
| `sum_abs_dflow` | residual magnitude |
| `cnt_exact_pref` | exact_pref assertions |
| `cnt_wake` | deblur_wake pop |
| Ablation | NO_DEBLUR vs ID vs TID: exact_hit / wake_pop under fixed PRRC budget |

---

## Ranked letter impact (which knife each boosts)

| Rank | Remake | Primary knife | Secondary | Why letter-impact |
|---|---|---|---|---|
| 1 | **ResHTR-Refiner** | **K1 OP-STW(+TDE3)** residual wake | K3 PRRC fine | HTR residual on TMA/BiSAT seed — sharpest new narrative vs soft NL-STMFA |
| 2 | **TID-DeblurLoop** | **K3 OGEC×PRRC→exact** | K1 wake | Corr-free iterative refine — orthogonal to CFP/SCI, sharpens exact budget story |
| 3 | **EDC-ΔFuse** | **K3 exact path** (pre-exact refine) | K1 detail wake | Δ-feat×corr fuse → exact_boost / OGEC; enhances ECP |
| 4 | **AdjEvt-Compress** | **K2 HBG-RP** MAC sparsity | BUI/SMAM | Digital event compress into HBG — knife-2 payload path |
| 5 | **WinTok-CoSparse** | **K1 wake** + K2 keep | BUI front | Co-sparse ≫ EvQ; feeds wake_bitmap + tok_keep |

---

## Suggested Card I order (pick any 2 for RTL after CFP/SCI pipe integration)

CFP/SCI (+ BiSAT/BUI) already in pipe. **Card I** = two leaf modules max this wave:

| Priority | Pair option | Rationale |
|---|---|---|
| **I-A (recommended)** | **`c1s_tid_deblur_loop` + `c1s_edc_delta_fuse`** | Both land on **knife-3 exact path** next to CFP/SCI/OGEC/PRRC; shared TB harness (`exact_hit`, grant starve); clearest rescore lever after ≈3.43 |
| **I-B** | **`c1s_reshtr_refiner` + `c1s_wintok_cosparse`** | Knife-1 wake depth (residual + co-sparse); good if letter text leads with OP-STW |
| **I-C** | **`c2s_adjevt_compress` + `c1s_wintok_cosparse`** | Knife-2 fabric + wake; defer if HBG packetizer I/O freeze is risky |

**Default recommendation:** **I-A** (TID + EDC) immediately after CFP/SCI integration; park ResHTR as I+1 when TMA/BiSAT seeds are stable.

---

## Optional runners (not Top5)

| Runner | Cite carefully | Why not Top5 |
|---|---|---|
| **Greatorex timing OF / ego-motion** | Greatorex et al., event timing OF, **CVPRF 2026** / related timing arXiv:2501.11554 (and TDE-circuit companion in stack notes) | Strong bio-timing prior; overlaps TDE3 lane — keep as **ablation/contrast**, not new top knife |
| **EVA-Flow** | Ye et al., *Towards anytime optical flow estimation with event cameras*, **arXiv:2307.05033** (anytime / on-the-fly; cited vs IDNet TID latency) | Real paper; anytime narrative overlaps PredExit/TID — cite as related, do not remake as 6th knife |
| **IDNet full ID multi-iter** | Same as #5 paper | TID loop is the HW-friendly slice; full ID×N is heavier RTL |

---

## DO NOT CLAIM

1. **Still exactly 3 letter knives** — Top5 are **enhancers**, not knife-4…N laundry.  
2. **Do not restate** Card G/H modules (TDE3, TMA, CFP, SCI, BiSAT, BUI) as novelty tops.  
3. **No MX3P / pure-binary ATLIF island swap**; HBG remains **int8 proposal** vs ep35 binary capture.  
4. **No CIM / analog / PIM / ASTER remake**; AdjEvt is **digital** sparsity only.  
5. **No inventing** ResFlow / EDCFlow / SAST / ExSpike / IDNet algorithms — **first-HW / stack-wiring** posture only.  
6. **No µ²/mW / AEE / DSEC tables** until joint sim; prove-by = RTL counters + ablations.  
7. **No “first event OF HW / first sparse-attn HW / first temporal OF HW”** global claims.  
8. **No SwinShift-SDSA** in this card (dropped: would skim SDformerFlow shifted-window + existing SDSA/HBG/BUI).

---

## Sources

1. Zhou et al., *ResFlow*, **arXiv:2412.09105**, 2024.  
2. Liu et al., *EDCFlow*, **arXiv:2506.03512**, 2025.  
3. *SAST: Scene Adaptive Sparse Transformer* (event detection), **CVPR 2024**.  
4. *ExSpike* (FPGA full-event; adjacent-position compression), **arXiv:2606.20414**.  
5. Wu, Paredes-Vallés, de Croon, *IDNet* (iterative deblurring; ID/TID), **ICRA 2024** / **arXiv:2211.13726**.  
6. Liu et al., *TMA*, **ICCV 2023**.  
7. Gehrig et al., *E-RAFT*, **3DV 2021**.  
8. Xu et al., *BAT*, **arXiv:2503.03256**, 2025 (BiSAT near).  
9. Ye et al., *EVA-Flow / anytime event OF*, **arXiv:2307.05033**.  
10. Greatorex et al., timing OF / ego-motion, **CVPRF 2026** (timing arXiv:2501.11554 family — cite carefully).  
11. Wang et al., *PADE* BUI-GF, **HPCA 2026** / arXiv:2512.14322 (BUI landed — contrast).  
12. Tian & Andrade-Cetto, *SDformerFlow*, **arXiv:2409.04082** / ICPR lineage (stack algo; **not** remade as Top5 here).  
13. Internal stack docs: `18_CARD_G_*`, `18_POST_TDE3_TMA_HARD_KNIVES.md`, `19_CARD_H_*`, `20_ISMD_INDEX_CARD_GH_*.md`.

---

**Teammate one-liner:** Fork A Top5 = **ResHTR / EDC-ΔFuse / WinTok-CoSparse / AdjEvt-Compress / TID-DeblurLoop** — all synthesizable digital enhancers of the three knives; Card I default **TID + EDC** after CFP/SCI.
