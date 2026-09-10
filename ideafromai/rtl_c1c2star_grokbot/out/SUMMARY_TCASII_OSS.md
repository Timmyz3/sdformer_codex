# TCAS-II OSS flow SUMMARY

**Generated (UTC):** 2026-09-05T18:32:19Z  
**Local (Asia/Shanghai):** 2026-09-06T02:32:19+0800  
**Tree:** `sdformer_c1c2star_grokbot` (Grok Bot)  
**Branch intent:** `tcasii/c1c2star-oss`  
**Tools:** iverilog/vvp + yosys(+abc) + gtkwave-ready VCD + power_proxy.py  
**Regression overall:** **PASS** (see `out/REGRESSION_REPORT.md`)

## Simulation (all `sim_*.sh`)

| Script | Result | Log |
|---|---|---|
| `sim_ablation_c2_ladder` | **PASS** | `out/sim/ablation_c2_FULL.log` |
| `sim_adp_mac` | **PASS** | `out/sim/c2s_adp_mac_sim.log` |
| `sim_arm_acc` | **PASS** | `out/sim/c2s_arm_acc_sim.log` |
| `sim_back_pipe` | **PASS** | `out/sim/c2s_back_pipe_sim.log` |
| `sim_bisat_agg` | **PASS** | `out/sim/c2s_bisat_agg_sim.log` |
| `sim_bui_guard_sdsa` | **PASS** | `out/sim/c2s_bui_guard_sdsa_sim.log` |
| `sim_c1s_stats` | **PASS** | `out/sim/c1s_stats_sim.log` |
| `sim_c2s_stats` | **PASS** | `out/sim/c2s_stats_sim.log` |
| `sim_cfp_confgate` | **PASS** | `out/sim/c1s_cfp_confgate_sim.log` |
| `sim_ecp_qkv` | **PASS** | `out/sim/c1s_ecp_qkv_sim.log` |
| `sim_exact_capture` | **PASS** | `out/sim/c1s_exact_capture_sim.log` |
| `sim_front_pipe` | **PASS** | `out/sim/c1s_front_pipe_sim.log` |
| `sim_hbg_rp` | **PASS** | `out/sim/c2s_hbg_rp_sim.log` |
| `sim_mfbd` | **PASS** | `out/sim/c2s_mfbd_sim.log` |
| `sim_motion_ttb` | **PASS** | `out/sim/c2s_motion_ttb_sim.log` |
| `sim_mw_delta` | **PASS** | `out/sim/c1s_mw_delta_sim.log` |
| `sim_ogec` | **PASS** | `out/sim/c1s_ogec_sim.log` |
| `sim_op_stw` | **PASS** | `out/sim/c1s_op_stw_sim.log` |
| `sim_prrc` | **PASS** | `out/sim/c1s_prrc_sim.log` |
| `sim_sci_cleanexit` | **PASS** | `out/sim/c1s_sci_cleanexit_sim.log` |
| `sim_smam_rp` | **PASS** | `out/sim/c2s_smam_rp_sim.log` |
| `sim_sp_gate` | **PASS** | `out/sim/c2s_sp_gate_sim.log` |
| `sim_sth_gate` | **PASS** | `out/sim/c2s_sth_gate_sim.log` |
| `sim_tde3_prior` | **PASS** | `out/sim/c1s_tde3_prior_sim.log` |
| `sim_tma_agg` | **PASS** | `out/sim/c2s_tma_agg_sim.log` |

## Yosys area (generic cells — NOT µm² / NOT DC)

| Module | Cells | Wire bits | Stat file |
|---|---|---|---|
| `c1s_op_stw` | **4870** | 9097 | `out/synth/c1s_op_stw.stat.txt` |
| `c2s_hbg_rp` | **24** | 55 | `out/synth/c2s_hbg_rp.stat.txt` |
| `c1s_ecp_qkv` | **769** | 1861 | `out/synth/c1s_ecp_qkv.stat.txt` |
| `c1s_mw_delta` | **3521** | 7300 | `out/synth/c1s_mw_delta.stat.txt` |
| `c2s_smam_rp` | **56** | 86 | `out/synth/c2s_smam_rp.stat.txt` |
| `c2s_motion_ttb` | **2065** | 2228 | `out/synth/c2s_motion_ttb.stat.txt` |
| `c2s_sth_gate` | **161** | 324 | `out/synth/c2s_sth_gate.stat.txt` |
| `c1s_ogec` | **25** | 80 | `out/synth/c1s_ogec.stat.txt` |
| `c1s_front_pipe` | **1238** | 2879 | `out/synth/c1s_front_pipe.stat.txt` |
| `c2s_back_pipe` | **250** | 685 | `out/synth/c2s_back_pipe.stat.txt` |
| `c1s_prrc` | **116** | 186 | `out/synth/c1s_prrc.stat.txt` |
| `c2s_adp_mac` | **565** | 589 | `out/synth/c2s_adp_mac.stat.txt` |
| `c2s_arm_acc` | **529** | 656 | `out/synth/c2s_arm_acc.stat.txt` |
| `c2s_mfbd` | **23** | 75 | `out/synth/c2s_mfbd.stat.txt` |
| `c2s_sp_gate` | **89** | 196 | `out/synth/c2s_sp_gate.stat.txt` |
| `c1s_exact_capture` | **114** | 197 | `out/synth/c1s_exact_capture.stat.txt` |
| `c1s_stats` | **376** | 485 | `out/synth/c1s_stats.stat.txt` |
| `c2s_stats` | **234** | 242 | `out/synth/c2s_stats.stat.txt` |

## Power proxy (NOT silicon)

### Card A
```
============================================================
POWER PROXY — NOT SILICON POWER / NOT PRIMETIME
No liberty (.lib), no C_load, no Vdd — heuristic only.
============================================================
label:          c1s_op_stw
vcd:            out/waves/c1s_op_stw.vcd
stat:           out/synth/c1s_op_stw.stat.txt
yosys cells:    4870
vcd changes:    107
toggle events:  107
alpha:          1.0
PROXY score:    5.21e+05  (= alpha * toggles * cells)
Interpret as relative activity×area score across ablations only.
```

### Card B
```
============================================================
POWER PROXY — NOT SILICON POWER / NOT PRIMETIME
No liberty (.lib), no C_load, no Vdd — heuristic only.
============================================================
label:          c2s_hbg_rp
vcd:            out/waves/c2s_hbg_rp.vcd
stat:           out/synth/c2s_hbg_rp.stat.txt
yosys cells:    24
vcd changes:    109
toggle events:  109
alpha:          1.0
PROXY score:    2.62e+03  (= alpha * toggles * cells)
Interpret as relative activity×area score across ablations only.
```

## Tool gaps

- **No liberty (.lib)** → no PrimeTime-class STA, no silicon power, no µm² area.
- **OpenROAD** → see `docs/OPENROAD_STATUS_GROKBOT.md`.
- Yosys **abc** used as internal pass when available.
- Verilator present but primary path is iverilog.
