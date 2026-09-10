# Card I — TID-DeblurLoop + EDC-ΔFuse (Grok Bot)

**Date:** 2026-09-06 (Asia/Shanghai)  
**Tree:** `sdformer_c1c2star_grokbot` only · `// GROKBOT NEW FILE -- iscas_ssh`  
**Sketch:** `ideafromai/research/22_CARD_I_IF_TID_EDC.md`  
**Order:** I1 TID → I2 EDC (ResHTR stub not this wave)

---

## 中文摘要

1. **TID-DeblurLoop（`c1s_tid_deblur_loop`）**：对 event bins 做 **corr-free** 运动补偿/去模糊 → `dflow` / `flow_hat_next` / `deblur_wake` / `exact_pref`。无 4D corr SRAM。
2. **EDC-ΔFuse（`c1s_edc_delta_fuse`）**：多尺度时序 Δ-feat × 低分辨 corr 融合 → `detail_hit` / `exact_boost`（与 CFP.exact_req 组合）。与 TID 正交。

推荐组合：`exact_en |= edc.exact_boost | {N{tid.exact_pref}}`（CFP 主门）；`wake |= tid.deblur_wake`；`capture_en &= ~sci.exact_hold`。

---

## EN — Prove-by / Do-not-claim

| | TID-DeblurLoop (I1) | EDC-ΔFuse (I2) |
|---|---|---|
| **What** | Corr-free MC/deblur loop on evt bins | Δ-feat × low-res corr fuse |
| **Outputs** | deblur_wake, exact_pref, flow_hat_next | detail_hit, exact_boost, mot_fuse |
| **Ablation** | NO_DEBLUR: dflow/wake → 0 | CORR_ONLY / DIFF_ONLY vs FULL |
| **Do-not-claim** | invent IDNet; Jetson≠ASIC; AEE | beat EDCFlow AEE; full-pair corr |

### Files
| Path | Role |
|---|---|
| `rtl_c1star/c1s_tid_deblur_loop.sv` | I1 RTL |
| `tb_c1star/tb_c1s_tid_deblur_loop.sv` | I1 TB |
| `flows/oss/sim_tid_deblur_loop.sh` | I1 sim |
| `rtl_c1star/c1s_edc_delta_fuse.sv` | I2 RTL |
| `tb_c1star/tb_c1s_edc_delta_fuse.sv` | I2 TB |
| `flows/oss/sim_edc_delta_fuse.sh` | I2 sim |

### CFP/SCI glue (Priority A, same wave)
| Path | Role |
|---|---|
| `rtl_c1star/c1s_cfp_sci_exact_glue.sv` | AND exact_req; grant AND; hold→capture_en; scrub→pe_clk_en |
| `tb_c1star/tb_c1s_cfp_sci_exact_glue.sv` | exact_hit CFP&lt;ALWAYS; hold freezes vs SCI-bypass |
| `flows/oss/sim_cfp_sci_exact_glue.sh` | glue sim |
