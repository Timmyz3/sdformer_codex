# REGRESSION REPORT (Grok Bot)

**Generated (UTC):** 2026-09-06T05:07:49Z  
**Local (Asia/Shanghai):** 2026-09-06T13:07:49+0800  
**Tree:** `sdformer_c1c2star_grokbot`  
**Branch:** `tcasii/c1c2star-oss`  
**Overall:** **PASS**  
**Sim scripts:** 30

## Pass/fail table

| Script | Result | Detail |
|---|---|---|
| `sim_ablation_c2_ladder` | **PASS** | rc=0 |
| `sim_adp_mac` | **PASS** | rc=0 |
| `sim_arm_acc` | **PASS** | rc=0 |
| `sim_back_pipe` | **PASS** | rc=0 |
| `sim_bisat_agg` | **PASS** | rc=0 |
| `sim_bl_veto_prior` | **PASS** | rc=0 |
| `sim_bui_guard_sdsa` | **PASS** | rc=0 |
| `sim_c1s_stats` | **PASS** | rc=0 |
| `sim_c2s_stats` | **PASS** | rc=0 |
| `sim_cfp_confgate` | **PASS** | rc=0 |
| `sim_cfp_sci_exact_glue` | **PASS** | rc=0 |
| `sim_ecp_qkv` | **PASS** | rc=0 |
| `sim_edc_delta_fuse` | **PASS** | rc=0 |
| `sim_exact_capture` | **PASS** | rc=0 |
| `sim_front_pipe` | **PASS** | rc=0 |
| `sim_hbg_rp` | **PASS** | rc=0 |
| `sim_mfbd` | **PASS** | rc=0 |
| `sim_motion_ttb` | **PASS** | rc=0 |
| `sim_mw_delta` | **PASS** | rc=0 |
| `sim_nl_stmfa` | **PASS** | rc=0 |
| `sim_ogec` | **PASS** | rc=0 |
| `sim_op_stw` | **PASS** | rc=0 |
| `sim_prrc` | **PASS** | rc=0 |
| `sim_sci_cleanexit` | **PASS** | rc=0 |
| `sim_smam_rp` | **PASS** | rc=0 |
| `sim_sp_gate` | **PASS** | rc=0 |
| `sim_sth_gate` | **PASS** | rc=0 |
| `sim_tde3_prior` | **PASS** | rc=0 |
| `sim_tid_deblur_loop` | **PASS** | rc=0 |
| `sim_tma_agg` | **PASS** | rc=0 |

## Notes

- Exit non-zero if any sim fails (see `flows/oss/regress.sh`).
- Yosys cell counts are generic — **NOT** µm² / NOT DC.
- No liberty → no silicon power / STA claims.
