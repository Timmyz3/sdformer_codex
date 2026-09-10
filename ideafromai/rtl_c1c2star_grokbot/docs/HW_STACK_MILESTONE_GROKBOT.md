# HW Stack Milestone (Grok Bot) — TCAS-II full skeleton close

**Tree:** `/workspace/sdformer_c1c2star_grokbot/`  
**Branch:** `tcasii/c1c2star-oss`  
**Date:** 2026-09-05 (Asia/Shanghai)  
**Constraint:** HW ONLY under this tree; never `nts07`; no push from this milestone.

**Regression:** **PASS** (18/18 `sim_*.sh`, see `out/REGRESSION_REPORT.md`).  
Skeletons **closed**: exact-capture, C1* stats, C2* stats. Side-wired on tops. Full `sim_*.sh` regression harness live.

## Module inventory (PASS / cells)

Yosys generic cells (NOT µm² / NOT DC). Unpacked-array designs use flat wrappers under `flows/oss/wrappers/`.  
*Cell counts refreshed by latest `flows/oss/run_all.sh` / `out/SUMMARY_TCASII_OSS.md` — placeholders below updated after regression.*

| Module | Path | Sim | Cells | Wire bits | Role |
|---|---|---|---|---|---|
| OP-STW | `rtl_c1star/c1s_op_stw_predictor.sv` | **PASS** | **4870** | 9097 | Card A — flow/event tile wake |
| HBG-RP | `rtl_c2star/c2s_hbg_rp_packetizer.sv` | **PASS** | **24** | 55 | Card B — binary gate + ATLIF payload |
| ECP-QKV | `rtl_c1star/c1s_ecp_qkv_predictor.sv` | **PASS** | **769** | 1861 | Eager corr → proj enable |
| MW-ΔBuf | `rtl_c1star/c1s_mw_delta_buf.sv` | **PASS** | **3521** | 7300 | Motion-warped residual Δ |
| SMAM-RP | `rtl_c2star/c2s_smam_rp.sv` | **PASS** | **56** | 86 | Dual-rail Mask-Add × payload |
| Motion-TTB | `rtl_c2star/c2s_motion_ttb_packer.sv` | **PASS** | **2065** | 2228 | Wake → motion time-bundles |
| STH-Gate | `rtl_c2star/c2s_sth_gate.sv` | **PASS** | **161** | 324 | Spatial/temporal head enables |
| OGEC | `rtl_c1star/c1s_ogec_gate.sv` | **PASS** | **25** | 80 | Exact vs propagate (+ABLATE_FORCE_EXACT) |
| Front-pipe | `rtl_c1star/c1s_front_pipe.sv` | **PASS** | **1238** | 2879 | Thin C1*: OP-STW→ECP←MW @ N_TILE=8 |
| Back-pipe | `rtl_c2star/c2s_back_pipe.sv` | **PASS** | **250** | 685 | Thin C2*: HBG→SMAM + STH @ N_HEAD=8 |
| PRRC | `rtl_c1star/c1s_prrc_ledger.sv` | **PASS** | **116** | 186 | Pyramid residual budget ledger |
| ADP-MAC | `rtl_c2star/c2s_adp_mac.sv` | **PASS** | **565** | 588 | Bilateral bit-sparse MAC |
| ARM-Acc | `rtl_c2star/c2s_arm_acc.sv` | **PASS** | **529** | 656 | Multi-hypothesis aperture acc |
| MFBD | `rtl_c2star/c2s_mfbd.sv` | **PASS** | **23** | 75 | Motion-bundle delivery |
| SP-Gate | `rtl_c2star/c2s_sp_gate.sv` | **PASS** | **89** | 196 | Attention-mass schedule gate |
| Exact-capture | `rtl_c1star/c1s_exact_capture_wrap.sv` | **PASS** | **114** | 197 | OGEC×PRRC gated exact latch/count |
| C1* stats | `rtl_c1star/c1s_stats.sv` | **PASS** | **376** | 485 | wake / proj-skip / delta_nz window |
| C2* stats | `rtl_c2star/c2s_stats.sv` | **PASS** | **234** | 242 | mac_en / skip / gate_fire window |

**Tops (thin integration):**
- `rtl_c1star/c1s_top.sv` — OP-STW + ECP + MW + OGEC + PRRC + **exact_capture** + **c1s_stats**
- `rtl_c2star/c2s_top.sv` — back_pipe + Motion-TTB + ADP/ARM/MFBD/SP + **c2s_stats**

## In / out of TCAS-II first cut

| In first cut | Status |
|---|---|
| OP-STW, HBG-RP spine | **in** |
| ECP, MW, OGEC, PRRC, exact_capture, C1* stats | **in** |
| SMAM, STH, Motion-TTB, ADP, ARM, MFBD, SP, C2* stats | **in** |
| front_pipe / back_pipe | **in** |
| Full `sim_*.sh` regression + `out/REGRESSION_REPORT.md` | **in** |
| Relative Yosys cells + power proxy | **in** (NOT silicon) |

| Out / later | Reason |
|---|---|
| DualRail-CIM / MW-CIM-TileGate / EV-Wake | roadmap packs |
| Silicon µm², STA, PrimePower | need liberty + OpenROAD |
| AEE / PE wake % from algo co-sim | needs ep34 harness |
| Full SDSA PE array | beyond letter thin cut |

## Flows

| Item | Path |
|---|---|
| Sim scripts | `flows/oss/sim_*.sh` |
| Full regression | `flows/oss/run_all.sh` → `out/REGRESSION_REPORT.md` + `out/SUMMARY_TCASII_OSS.md` |
| Exit-nonzero wrapper | `flows/oss/regress.sh` |
| Synth | `flows/oss/run_synth.sh` |
| Innovation review | `docs/TCASII_INNOVATION_REVIEW_GROKBOT.md` |
| OpenROAD status | `docs/OPENROAD_STATUS_GROKBOT.md` |
| This milestone | `docs/HW_STACK_MILESTONE_GROKBOT.md` |

## Stop rule

After Phase D (OpenROAD best-effort doc / Yosys-only): STOP for parent iteration.


## 2026-09-05 follow-up (Grok Bot)

- Liberty-mapped Yosys synth for HBG/OGEC (`out/synth/*_mapped.stat.txt`); OpenROAD binary still PENDING.
- C1* ablation ladder: `flows/oss/ablation_c1_ladder.sh` → `out/ABLATION_C1_LADDER.md`.
- Claim draft: `docs/TCASII_LETTER_CLAIM_DRAFT_GROKBOT.md`.
- RTL sharpen: ADP `FORBID_REORDER=1` + HBG EPS contract asserts in TB.

## 2026-09-05 OpenROAD continue (Grok Bot)

- OpenROAD wrapper OK: `/workspace/tools/openroad` (`v2.0-17198-g8396d0866`).
- OGEC **floorplan only**: **586 u² @ 8%** → `out/openroad/ogec_floorplan.*`.
- HBG **placed** (global+detailed), **not routed**: **121 u² @ 8%** → `out/openroad/hbg_placed.*`.
- Extra mapped synth: SMAM **730.700800**, OP-STW@N_TILE=8 **3071.696000**.
- Status: `docs/OPENROAD_STATUS_GROKBOT.md`. Regress still **PASS**.


## 2026-09-06 Card G + 3-knife hygiene (Grok Bot)

- **Claim draft** compressed to 3 knives (OP-STW(+TDE3) / HBG-RP proposal / OGEC×PRRC→exact); TMA-Agg = support.
- **New RTL:** `c1s_tde3_prior`, `c1s_wake_merge`, `c2s_tma_agg` (+ TB + `sim_tde3_prior` / `sim_tma_agg`).
- **C1 ablation** rung **EXACT**: PRRC budget caps `exact_hit` 49→19 vs MW (same wake).
- OpenROAD still = completeness only; **no** µ²/mW / PDN / SPEF / signoff claims this turn.
- Docs: `docs/CARD_G_TDE3_TMA_GROKBOT.md`; independent-review remediation § appended (re-score pending).
