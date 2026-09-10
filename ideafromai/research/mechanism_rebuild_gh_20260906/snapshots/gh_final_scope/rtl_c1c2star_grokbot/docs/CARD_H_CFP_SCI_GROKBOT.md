# Card H — CFP-ConfGate + SCI-CleanExit (Grok Bot)

**Date:** 2026-09-06 (Asia/Shanghai)  
**Tree:** `sdformer_c1c2star_grokbot` only  
**Tag:** `// GROKBOT NEW FILE -- iscas_ssh`  
**Scope this turn:** H2 + H3 only (**not** BiSAT / BUI)

---

## 中文摘要

Card H 前两把 RED 刀锐化 **刀3（exact 路径）**，不替换 OGEC / PRRC / exact_capture：

1. **CFP-ConfGate（`c1s_cfp_confgate`）**：`corr_peak`→`conf`；高 conf → `prop_pref`（释 PRRC 软配额）；低/中 conf → `exact_req`。`ABLATE_ALWAYS_EXACT` 消融对照。
2. **SCI-CleanExit（`c1s_sci_cleanexit`）**：`|f1−f2_warp|` 反相得 `g_sci`；低质 → `scrub_en`；均值高 → `exit_ok`；脏迭代 `exact_hold` 冻结 exact 入队。`ABLATE_EXIT_ONLY` 关 scrub。

**勿称：** first occlusion OF；无联合仿真 AEE；SciFlow 帧端精度；CIM。

---

## EN — What / Wiring / Prove-by / Do-not-claim

| | CFP-ConfGate (H2) | SCI-CleanExit (H3) |
|---|---|---|
| **What** | Peak→conf gate: exact_req vs prop_pref + soft `prrc_grant` | SCI quality → scrub / exit_ok / exact_hold |
| **Wired** | Between OGEC match & PRRC; `exact_en &= exact_req`; grant soft-scales budget | `capture_en &= ~exact_hold`; scrub → iter FSM; optional `bisat_occ_exit_i` |
| **Policy** | conf≥TH_HI → prop; else exact (if match_ok) | g=255-sat(|delta|); scrub if g<TH_SCRUB |
| **Ablation** | ALWAYS_EXACT vs CFP (`capture_cnt` ↓ under CFP) | EXIT_ONLY vs FULL (`cnt_scrub` ↑ under FULL) |

### Prove-by (example TB)

**CFP** (`sim_cfp_confgate.sh`):
- Case1 split: exact=`00001111` prop=`11110000` grant=8
- Ablation: `capture_cfp=12` vs `capture_ALWAYS_EXACT=28`
- Counters: `sum_conf=5520` `cnt_exact_req=20` `cnt_prop_pref=16` `cnt_grant_starve=1`

**SCI** (`sim_sci_cleanexit.sh`):
- Clean: g=255 exit=1 hold=0; dirty: g=55 scrub=all hold=1
- Ablation: FULL `cnt_scrub=12` vs EXIT_ONLY `0`; holds both=3
- Counters: `cnt_iter=5` `cnt_exit=3` `cnt_exact_hold=3`

### Do-not-claim
- Enhance knife-3 exact path — **not** a 4th letter knife laundry item.
- Not first event-OF HW / occlusion OF / SciFlow ASIC AEE.
- No µ²/mW; no CIM; Fork A int8 proposal untouched.

## Files

| Path | Role |
|---|---|
| `rtl_c1star/c1s_cfp_confgate.sv` | H2 RTL |
| `tb_c1star/tb_c1s_cfp_confgate.sv` | H2 TB |
| `flows/oss/sim_cfp_confgate.sh` | H2 sim |
| `rtl_c1star/c1s_sci_cleanexit.sv` | H3 RTL |
| `tb_c1star/tb_c1s_sci_cleanexit.sv` | H3 TB |
| `flows/oss/sim_sci_cleanexit.sh` | H3 sim |
