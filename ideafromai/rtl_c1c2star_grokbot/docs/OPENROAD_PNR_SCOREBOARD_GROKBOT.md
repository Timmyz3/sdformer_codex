# OpenROAD P&R Scoreboard / OpenROAD 布局布线记分板 (Grok Bot)

**中文：** sky130hd 上 C1*/C2* 模块的 best-effort 综合→布局→CTS→布线结果汇总；**非签核**（无 PDN / 无 OpenRCX SPEF / 无多角 STA）。面积为 `report_design_area` 单元面积利用率，不可当作硅片 µm² 声称。DEF 见 `out/openroad/*_routed.def`。

**English:** Best-effort synth→floorplan→place→CTS→GR→DR scoreboard on sky130hd for C1*/C2* blocks. **NOT signoff** (no PDN, no OpenRCX SPEF, no multi-corner STA). Area = cell-area util from `report_design_area` — do **not** quote as silicon µm². Routed DEFs under `out/openroad/`.

| Field | Meaning |
|---|---|
| Mapped u² | Yosys `stat -liberty` chip area after liberty-map |
| Routed u² / util | Post-CTS+DR `report_design_area` |
| DRC | Detailed-route violations (empty `*_route_drc.rpt` ⇒ 0) |
| STA setup/hold | OpenROAD `report_checks` @ 10 ns, IO max=1.0 / min=0.5 ns, placement parasitics — **NOT SIGNOFF** |

**Tree:** `sdformer_c1c2star_grokbot` · **Branch:** `tcasii/c1c2star-oss`  
**Date (Asia/Shanghai):** 2026-09-06 ~12:40+08 · **OpenROAD:** v2.0-17198-g8396d0866  
**Caveats (all rows):** no PDN (`snets=0`); no SPEF/RCX; single `tt_025C_1v80`; ideal/propagated CTS clock as tool allows; local commit only; **no push**; never `nts07`.

## Full table — 16 leaf + 2 pipes + 7 Card G/H

| # | Module | Params | Cells (map) | Mapped u² | Die/core µm | Routed u² / util | DRC | STA setup / hold @10ns | Routed DEF |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HBG-RP | — | 15 | 121.4 | (fp 8%) | **121 / 8%** | **0** | +6.465 / +0.204 | `hbg_routed.def` |
| 2 | OGEC | N_TILE=8 | 41 | 585.6 | 50×50 / 40×40 | **613 / 41%** | **0** | (not run) | `ogec_routed.def` |
| 3 | SMAM-RP | — | 66 | 730.7 | — | **757 / 39%** | **0** | (not run) | `smam_routed.def` |
| 4 | OP-STW | N_TILE=8 | 423 | 3071.7 | 100×100 / 90×90 | **3098 / 40%** | **0** | +5.945 / +0.205 | `op_stw_routed.def` |
| 5 | ADP-MAC | — | 404 | 3399.5 | 110×110 / 100×100 | **3426 / 35%** | **0** | +4.049 / +0.205 | `adp_mac_routed.def` |
| 6 | ECP-QKV | N_TILE=8 | 74 | 768.2 | 60×60 / 50×50 | **796 / 33%** | **0** | +8.248 / +0.205 | `ecp_qkv_routed.def` |
| 7 | STH-Gate | N_HEAD=8 | 147 | 1525.2 | 80×80 / 70×70 | **1614 / 34%** | **0** | +7.657 / +0.205 | `sth_gate_routed.def` |
| 8 | PRRC ledger | N_LEVEL=3 | 133 | 1248.7 | 70×70 / 60×60 | **1275 / 37%** | **0** | +6.677 / +0.205 | `prrc_routed.def` |
| 9 | MW-ΔBuf | N_TILE=8 | 520 | 4719.5 | 130×130 / 120×120 | **4842 / 35%** | **0** | +6.452 / +0.206 | `mw_delta_routed.def` |
| 10 | Motion-TTB | N_TILE=8 | 591 | 4653.2 | 130×130 / 120×120 | **4826 / 34%** | **0** | +4.178 / +0.205 | `motion_ttb_routed.def` |
| 11 | ARM-Acc | N_HYP=4 | 454 | 4033.9 | 120×120 / 110×110 | **4207 / 35%** | **0** | +4.759 / +0.206 | `arm_acc_routed.def` |
| 12 | SP-Gate | N=8 | 73 | 788.3 | 60×60 / 50×50 | **815 / 33%** | **0** | +8.266 / +0.205 | `sp_gate_routed.def` |
| 13 | MFBD | MAX_B=4 | 27 | 454.2 | 50×50 / 40×40 | **480 / 32%** | **0** | +8.331 / +0.205 | `mfbd_routed.def` |
| 14 | ExactCapt | N_TILE=8 | 126 | 1306.3 | 70×70 / 60×60 | **1329 / 39%** | **0** | +5.276 / +0.205 | `exact_capture_routed.def` |
| 15 | C1*-stats | N_TILE=8 | 304 | 2874.0 | 100×100 / 90×90 | **2944 / 38%** | **0** | +4.450 / +0.206 | `c1s_stats_routed.def` |
| 16 | C2*-stats | — | 240 | 2258.4 | 90×90 / 80×80 | **2307 / 37%** | **0** | +7.423 / +0.206 | `c2s_stats_routed.def` |
| 17 | **front_pipe** | **N_TILE=8 flat** | **1179** | **11347.1** | **200×200 / 190×190** | **11791 / 33%** | **0** | **+5.764 / +0.221** | `front_pipe_routed.def` |
| 18 | **back_pipe** | **N_HEAD=8 flat** | **246** | **2703.8** | **100×100 / 90×90** | **2825 / 36%** | **0** | **+6.343 / +0.206** | `back_pipe_routed.def` |

**Totals:** **25/25 DRC=0** (16 leaf + 2 pipes + **7 Card G/H**). Largest routed cell-area: front_pipe **11791 u²**. Card G/H rows 19–25 below.

## Pipe notes / 管道说明

| Pipe | Flatten path | Mapped synth | Wire / vias | Status |
|---|---|---|---|---|
| `c1s_front_pipe_synth` | OP-STW+ECP+MW wrappers @ N_TILE=8 + glue; Yosys `flatten` after abc | `out/synth/c1s_front_pipe_mapped.{v,stat.txt}` | ~42168 µm / 8995 | **ROUTED DRC=0** |
| `c2s_back_pipe_synth` | HBG+SMAM RTL + STH wrapper @ N_HEAD=8; Yosys `flatten` | `out/synth/c2s_back_pipe_mapped.{v,stat.txt}` | ~9289 µm / 2136 | **ROUTED DRC=0** |

Prior status had skipped pipes as “hierarchical/heavy”; this turn flattened cleanly and completed full P&R.

## Artifact index

All under `out/openroad/`: `*_placed.def`, `*_cts.def`, `*_routed.def`, `*_route.guide`, `*_route_drc.rpt`, `*_sta_iodelay_report.txt`, `*_routed_area.txt`.  
Mapped netlists: `out/synth/*_mapped.v`.  
Detailed attempt log: `docs/OPENROAD_STATUS_GROKBOT.md`.  
Summary: `out/SUMMARY_TCASII_OSS.md`.


## Card G/H addendum (2026-09-06 ~12:40+08 Asia/Shanghai)

Liberty-map + floorplan→place→CTS→GR→DR + STA IO delays for Card G (TDE3 / wake_merge / TMA) and Card H (CFP / SCI / BiSAT / BUI). Default **N_TILE=8** (BUI: N_TOKEN=8). Driver: `flows/oss/map_and_pnr_card_gh.sh`. **NOT signoff** (no PDN / no SPEF / single tt corner).

| # | Module | Params | Cells (map) | Mapped u² | Die/core µm | Routed u² / util | DRC | STA setup / hold @10ns | Routed DEF |
|---|---|---|---|---|---|---|---|---|---|
| 19 | **TDE3-Prior** | **N_TILE=8** | **1062** | **8683.3** | **165×165 / 155×155** | **9149 / 39%** | **0** | **+4.854 / +0.206** | `tde3_prior_routed.def` |
| 20 | **wake_merge** | **N_TILE=8** | **33** | **355.3** | **40×40 / 30×30** | **382 / 47%** | **0** | **+8.357 / +0.205** | `wake_merge_routed.def` |
| 21 | **TMA-Agg** | **N_TILE=8** | **785** | **6817.8** | **150×150 / 140×140** | **6960 / 36%** | **0** | **+3.417 / +0.206** | `tma_agg_routed.def` |
| 22 | **CFP-ConfGate** | **N_TILE=8** | **471** | **4296.6** | **120×120 / 110×110** | **4459 / 37%** | **0** | **+2.290 / +0.206** | `cfp_confgate_routed.def` |
| 23 | **SCI-CleanExit** | **N_TILE=8** | **802** | **7250.7** | **155×155 / 145×145** | **7388 / 35%** | **0** | **-0.019 / +0.206** | `sci_cleanexit_routed.def` |
| 24 | **BiSAT-Agg** | **N_TILE=8** | **867** | **6705.2** | **145×145 / 135×135** | **6830 / 38%** | **0** | **+3.557 / +0.206** | `bisat_agg_routed.def` |
| 25 | **BUI-GuardSDSA** | **N_TOKEN=8** | **380** | **2791.4** | **95×95 / 85×85** | **2828 / 40%** | **0** | **+6.878 / +0.205** | `bui_guard_sdsa_routed.def` |

**Card G/H:** **7/7 DRC=0**. SCI-CleanExit setup WNS **−0.019 ns** @10 ns (placement parasitics only — **NOT SIGNOFF**; do not claim timing closed). CSV: `out/openroad/card_gh_pnr_results.csv`.

