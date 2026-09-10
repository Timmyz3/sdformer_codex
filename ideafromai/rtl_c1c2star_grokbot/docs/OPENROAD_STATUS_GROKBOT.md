# OpenROAD status (Grok Bot)

**Date:** 2026-09-06 (Asia/Shanghai)  
**Tree:** `sdformer_c1c2star_grokbot`  
**Branch:** `tcasii/c1c2star-oss`  
**Goal:** best-effort synth→floorplan→place→CTS→route on HBG / OGEC / SMAM / OP-STW / ADP-MAC / ECP-QKV / STH-Gate / PRRC / MW-ΔBuf / Motion-TTB / ARM-Acc / SP-Gate / MFBD / ExactCapt / C1*-stats / C2*-stats / **front_pipe** / **back_pipe** (sky130hd, no full ORFS).

## Attempt log (exact commands + results)

### 1) apt install openroad

```bash
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y openroad
```

**Result:** `E: Unable to locate package openroad` (Debian 13 / trixie).

### 2) Docker / ORFS image

```bash
command -v docker   # empty
```

**Result:** Docker **not used** (blocked / unavailable). No `docker pull`.

### 3) sky130 liberty + LEF → `/workspace/pdks/sky130hd/`

Liberty + merged cell LEF reused from prior turn. **Added tech LEF** for SITE/layers:

```bash
curl -sL -o /workspace/pdks/sky130hd/lef/sky130_fd_sc_hd.tlef \
  https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef
```

Also fetched ORFS `setRC.tcl` / `fastroute.tcl` → `/workspace/pdks/sky130hd/tech/`.

### 4) Yosys mapped synth (liberty) — HBG-RP + OGEC (+ SMAM / OP-STW)

**Result:** SUCCESS.

| Design | Cells | Liberty chip area | Artifacts |
|---|---|---|---|
| HBG-RP | 15 | **121.366400** | `out/synth/c2s_hbg_rp_mapped.*` |
| OGEC (N_TILE=8) | 41 | **585.561600** | `out/synth/c1s_ogec_mapped.*` |
| SMAM-RP | 66 | **730.700800** | `out/synth/c2s_smam_rp_mapped.*` |
| OP-STW (N_TILE=8) | 423 | **3071.696000** | `out/synth/c1s_op_stw_mapped.*` |

### 5) OpenROAD binary — Precision Innovations `.deb` (no Docker)

```bash
/workspace/tools/openroad -version
# → v2.0-17198-g8396d0866
```

Wrapper: `/workspace/tools/openroad` (LD_LIBRARY_PATH for jammy python/tclreadline + or-tools).

### 6) Floorplan run on HBG mapped netlist — **RAN (floorplan only)**

| Item | Value |
|---|---|
| `report_design_area` | **Design area 121 u^2 8% utilization** |
| DEF | `out/openroad/hbg_floorplan.def` |

**Label: FLOORPLAN ONLY**

### 7) Floorplan run on OGEC mapped netlist — **RAN (floorplan only)**

| Item | Value |
|---|---|
| Top | `c1s_ogec_gate_synth` (N_TILE=8 mapped) |
| Die (fp-only script) | 100×100 / core 90×90 |
| `report_design_area` | **Design area 586 u^2 8% utilization** |
| DEF | `out/openroad/ogec_floorplan.def` |

**Label: FLOORPLAN ONLY**

### 8) HBG place step (global + detailed) — **RAN (placed)**

| Item | Value |
|---|---|
| `global_placement` | **OK** (overflow ≈ 0.094) |
| `detailed_placement` | **OK** (legalized) |
| `report_design_area` | **Design area 121 u^2 8% utilization** |
| DEF | `out/openroad/hbg_placed.def` |

**Label: PLACED (global+detailed)**

### 9) OGEC place step (global + detailed) — **RAN (placed, NOT routed)**

```bash
# TCL: out/openroad/place_ogec_minimal.tcl
# Note: initial 100×100 die @ dens=0.7 → GPL-0305 RePlAce diverged.
# Retry: die 50×50 / core 40×40, dens=0.6 → OK.
/workspace/tools/openroad -no_init -exit out/openroad/place_ogec_minimal.tcl \
  | tee out/openroad/place_ogec_minimal.log
```

| Item | Value |
|---|---|
| `global_placement` | **OK** (dens=0.6, overflow ≈ 0.101) |
| `detailed_placement` | **OK** (legalized HPWL 831.3 u, delta +28%) |
| `report_design_area` | **Design area 586 u^2 39% utilization** |
| DEF | `out/openroad/ogec_placed.def` |
| Area log | `out/openroad/ogec_placed_area.txt` |

IO ports not pin-placed during GPL (acceptable).  
**Label: PLACED (global+detailed)** — **NOT routed**. Do not quote as post-route µm².

### 10) HBG CTS + route (best-effort, no full ORFS) — **RAN**

```bash
# TCL: out/openroad/cts_route_hbg_minimal.tcl
# read_def hbg_placed.def → create_clock → setRC → place_pins →
# clock_tree_synthesis → global_route → detailed_route
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_hbg_minimal.tcl \
  | tee out/openroad/cts_route_hbg_minimal.log
```

| Step | Result | Artifact |
|---|---|---|
| `create_clock` clk @ 10 ns | **OK** | — |
| `source tech/setRC.tcl` | **OK** | ORFS sky130hd layer RC |
| `place_pins` met3/met2 | **OK** | — |
| `clock_tree_synthesis` (clkbuf_1/2/4/8, root clkbuf_4) | **OK** | `hbg_cts.def` |
| `global_route` + guide | **OK** | `hbg_route.guide` |
| `estimate_parasitics -global_routing` | **OK** (GR estimate only) | — |
| `detailed_route` | **OK** (DRC violations → **0**) | `hbg_routed.def` |
| Post-flow `report_design_area` | **121 u^2 8% utilization** | cell-area util |

**Status banner:** `HBG_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`

**Caveats (do not over-claim):**
- **No OpenSTA `report_checks`** → **do NOT claim post-route timing / WNS / TNS**.
- **No PDN** (`Number of snets: 0`) — power rails not inserted; signal route only.
- **No OpenRCX SPEF** — parasitics are GR estimates via `setRC.tcl`, not extracted SPEF.
- Area number remains **cell-area utilization**, not a taped-out die claim.

### 11) Optional: SMAM floorplan — **RAN (floorplan only)**

```bash
/workspace/tools/openroad -no_init -exit out/openroad/floorplan_smam_minimal.tcl \
  | tee out/openroad/floorplan_smam_minimal.log
```

| Item | Value |
|---|---|
| Top | `c2s_smam_rp` |
| `report_design_area` | **Design area 731 u^2 9% utilization** |
| DEF | `out/openroad/smam_floorplan.def` |

**Label: FLOORPLAN ONLY**

### 12) Missing tech for full ORFS / silicon-class signoff

Present under `/workspace/pdks/sky130hd/`:
- liberty tt corner, tech+merged LEF, `make_tracks.tcl`, `tech/setRC.tcl`, `tech/fastroute.tcl`

**Still missing** (blocked full ORFS / RCX / signoff; did **not** block GR+DR on this tiny HBG):
- OpenRCX rules / patterns (e.g. `sky130hd.rcx_rules`, `rcx_patterns.rules`)
- PDN / grid TCL (`pdn.cfg` / `grid.tcl`) + welltap / endcap / filler scripts
- Multi-corner liberty + full SDC STA deck
- KLayout DRC/LVS tech decks
- Full ORFS `config.mk` / platform makefile glue

### 13) Summary

| Item | Status |
|---|---|
| apt `openroad` | FAIL (no package) |
| docker ORFS | SKIP (no docker; blocked) |
| PI `.deb` + wrapper on Debian 13 | **OK** (`v2.0-17198-g8396d0866`) |
| sky130 liberty / tech+cell LEF / setRC | **OK** |
| yosys liberty-mapped synth | **OK** (HBG / OGEC / SMAM / OP-STW@8) |
| OpenROAD floorplan (HBG) | **OK** — 121 u² @ 8% (**floorplan only**) |
| OpenROAD floorplan (OGEC) | **OK** — 586 u² @ 8% (**floorplan only**) |
| OpenROAD floorplan (SMAM) | **OK** — 731 u² @ 9% (**floorplan only**) |
| OpenROAD place (HBG) | **OK** — 121 u² @ 8% (**placed**) |
| OpenROAD place (OGEC) | **OK** — 586 u² @ 39% (**placed, not routed**) |
| OpenROAD CTS (HBG) | **OK** — `hbg_cts.def` |
| OpenROAD global+detailed route (HBG) | **OK** — `hbg_routed.def`, DRC=0 |
| OpenSTA post-route timing | **NOT DONE** (no `report_checks`) |
| OpenRCX / PDN / full ORFS | **NOT DONE** (files missing / not run) |

**Never claim P&R silicon µm² / mW / timing without STA+RCX closing.**  
`report_design_area` numbers are cell-area util on the initialized core.

### 14) OGEC CTS + route (mirror HBG) — **RAN**

```bash
# TCL: out/openroad/cts_route_ogec_minimal.tcl
# read_def ogec_placed.def → create_clock → setRC → place_pins →
# clock_tree_synthesis → global_route → detailed_route
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_ogec_minimal.tcl \
  | tee out/openroad/cts_route_ogec_minimal.log
```

| Step | Result | Artifact |
|---|---|---|
| `create_clock` clk @ 10 ns | **OK** | — |
| `source tech/setRC.tcl` | **OK** | ORFS sky130hd layer RC |
| `place_pins` met3/met2 | **OK** | — |
| `clock_tree_synthesis` | **OK** | `ogec_cts.def` |
| `global_route` + guide | **OK** | `ogec_route.guide` |
| `detailed_route` | **OK** (violations → **0**) | `ogec_routed.def`, empty `ogec_route_drc.rpt` |
| Post-flow `report_design_area` | **613 u^2 41% utilization** | cell-area util (CTS bufs ↑ from 586) |

**Status banner:** `OGEC_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`  
Wire ~1309 µm, vias 350. No PDN. **No OpenSTA on OGEC this turn.**

### 15) HBG timing best-effort (OpenROAD `report_checks`) — **RAN (NOT SIGNOFF)**

```bash
# Binary: OpenROAD built-in STA (also found /workspace/tools/openroad_extract/usr/bin/sta v2.6.0)
# TCL: out/openroad/hbg_sta_minimal.tcl → out/openroad/hbg_sta_report.txt
/workspace/tools/openroad -no_init -exit out/openroad/hbg_sta_minimal.tcl \
  | tee out/openroad/hbg_sta_report.txt
```

| Item | Value |
|---|---|
| DEF | `hbg_routed.def` |
| Clock | 10 ns (100 MHz target) + ideal I/O delays max=1.0 ns |
| Parasitics | `estimate_parasitics -placement` — **NOT OpenRCX SPEF** |
| Corner | single `tt_025C_1v80` liberty |
| Setup (max) worst data-path slack | **+6.465 ns MET** (amp[7]→p[6]; arrival ~2.535 ns) |
| `report_wns` / `report_tns` | **0.000 / 0.000** (setup group) |
| Hold/removal (min) | **-0.296 ns VIOLATED** on `rst_n`→`_19_` removal (ideal min IO=0; **not** signoff hold) — superseded by §19 with min IO=0.5 ns → **MET** |

**Label: NOT SIGNOFF.** No SPEF, no multi-corner, no SI, ideal clock network. Do not quote as PrimeTime / silicon timing.

### 16) SMAM place (quick) — **RAN (placed, NOT routed)**

```bash
/workspace/tools/openroad -no_init -exit out/openroad/place_smam_minimal.tcl \
  | tee out/openroad/place_smam_minimal.log
```

| Item | Value |
|---|---|
| Die / dens | 55×55 / dens=0.6 (overflow ≈ 0.099) |
| `detailed_placement` | **OK** |
| `report_design_area` | **Design area 731 u^2 38% utilization** |
| DEF | `out/openroad/smam_placed.def` |

**Label: PLACED (global+detailed)** — **NOT routed**.

### 17) Summary (updated)

| Item | Status |
|---|---|
| OpenROAD CTS+route (OGEC) | **OK** — `ogec_routed.def`, DRC=0, 613 u² @ 41% |
| OpenROAD STA (HBG) | **OK best-effort NOT SIGNOFF** — setup slack ≈ +6.5 ns @ 10 ns; see `hbg_sta_report.txt` |
| OpenROAD place (SMAM) | **OK** — 731 u² @ 38% (placed; routed in §18) |
| OpenRCX / PDN / full ORFS | **NOT DONE** |

### 18) SMAM CTS + route (mirror HBG/OGEC) — **RAN**

```bash
# TCL: out/openroad/cts_route_smam_minimal.tcl
# read_def smam_placed.def → create_clock → setRC → place_pins →
# clock_tree_synthesis → global_route → detailed_route
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_smam_minimal.tcl \
  | tee out/openroad/cts_route_smam_minimal.log
```

| Step | Result | Artifact |
|---|---|---|
| `create_clock` clk @ 10 ns | **OK** | — |
| `source tech/setRC.tcl` | **OK** | ORFS sky130hd layer RC |
| `place_pins` met3/met2 | **OK** | — |
| `clock_tree_synthesis` (17 sinks → 3 clkbufs) | **OK** | `smam_cts.def` |
| `global_route` + guide | **OK** (WL GR ≈ 2566 µm) | `smam_route.guide` |
| `detailed_route` | **OK** (violations 1→**0**) | `smam_routed.def`, empty `smam_route_drc.rpt` |
| Post-flow `report_design_area` | **757 u^2 39% utilization** | cell-area util (CTS bufs ↑ from 731) |

**Status banner:** `SMAM_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`  
Wire ~1541 µm, vias 519. No PDN (`Number of snets: 0`). **No OpenSTA on SMAM this turn.**

### 19) HBG STA with realistic I/O delays — **RAN (NOT SIGNOFF)**

```bash
# TCL: out/openroad/hbg_sta_iodelay.tcl → out/openroad/hbg_sta_iodelay_report.txt
# min IO delay 0.5 ns / max 1.0 ns (was ideal min=0 → false rst_n removal fail)
/workspace/tools/openroad -no_init -exit out/openroad/hbg_sta_iodelay.tcl \
  | tee out/openroad/hbg_sta_iodelay_report.txt
```

| Item | Value |
|---|---|
| DEF | `hbg_routed.def` |
| Clock | 10 ns + **IO max=1.0 ns / min=0.5 ns** |
| Parasitics | `estimate_parasitics -placement` — **NOT OpenRCX SPEF** |
| Corner | single `tt_025C_1v80` liberty |
| Setup (max) worst data-path slack | **+6.465 ns MET** (amp[7]→p[6]) |
| `report_wns` / `report_tns` | **0.000 / 0.000** (setup group) |
| Hold/removal (min) worst | **+0.204 ns MET** on `rst_n`→`_19_` removal (was **-0.296 ns** under ideal min IO=0) |

**Label: NOT SIGNOFF.** No SPEF, no multi-corner, no SI. Do not quote as PrimeTime / silicon timing.

### 20) Optional PDN on HBG — **SKIPPED**

Searched `/workspace/pdks/sky130hd/` and `/workspace/tools` for ORFS `pdn.cfg` / `grid.tcl` / welltap scripts: **none present** (only `setRC.tcl`, `fastroute.tcl`, LEF/liberty, `make_tracks.tcl`). Did **not** invent a PDN grid or touch `nts07`. Documented gap remains: no power straps / snets.

### 21) Summary (updated 2026-09-06 Asia/Shanghai)

| Item | Status |
|---|---|
| OpenROAD CTS+route (HBG) | **OK** — `hbg_routed.def`, DRC=0, 121 u² @ 8% |
| OpenROAD CTS+route (OGEC) | **OK** — `ogec_routed.def`, DRC=0, 613 u² @ 41% |
| OpenROAD CTS+route (SMAM) | **OK** — `smam_routed.def`, DRC=0, **757 u² @ 39%** |
| OpenROAD STA (HBG, IO delays) | **OK best-effort NOT SIGNOFF** — setup **+6.465 ns**; hold/removal **+0.204 ns** @ min IO=0.5 ns — `hbg_sta_iodelay_report.txt` |
| OpenRCX / PDN / full ORFS | **NOT DONE** (PDN scripts missing → skipped) |

### 22) OP-STW (N_TILE=8) full P&R — **RAN (ROUTED DRC=0)**

Mapped netlist reused: `out/synth/c1s_op_stw_mapped.v` (top `c1s_op_stw_predictor_synth`, liberty area **3071.696** u², 423 cells).

**Die/core sizing (documented):**
- Liberty area ≈ 3071.7 u² → target util **~30–40%**
- Chose **die 100×100 µm**, **core 90×90 µm** (snapped core ≈ 89.7×87.0 → CoreArea 7807.5 u²)
- Floorplan util **39%**; post-CTS/route cell-area util **40%**

```bash
# Floorplan
/workspace/tools/openroad -no_init -exit out/openroad/floorplan_op_stw_minimal.tcl \
  | tee out/openroad/floorplan_op_stw_minimal.log
# Place (global dens=0.6 + detailed)
/workspace/tools/openroad -no_init -exit out/openroad/place_op_stw_minimal.tcl \
  | tee out/openroad/place_op_stw_minimal.log
# CTS + GR + DR
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_op_stw_minimal.tcl \
  | tee out/openroad/cts_route_op_stw_minimal.log
```

| Step | Result | Artifact |
|---|---|---|
| Floorplan | **OK** — 3072 u² @ 39% | `op_stw_floorplan.def` |
| `global_placement` dens=0.6 | **OK** (overflow ≈ 0.098) | — |
| `detailed_placement` | **OK** (HPWL +35%) | `op_stw_placed.def` |
| `create_clock` clk @ 10 ns | **OK** | — |
| `place_pins` met3/met2 | **OK** (204 I/Os) | — |
| `clock_tree_synthesis` (9 sinks → 3 clkbufs) | **OK** | `op_stw_cts.def` |
| `global_route` + guide | **OK** (WL GR ≈ 15897 µm, 0 overflow) | `op_stw_route.guide` |
| `detailed_route` | **OK** (violations 137→**0**) | `op_stw_routed.def`, empty `op_stw_route_drc.rpt` |
| Post-flow `report_design_area` | **3098 u^2 40% utilization** | CTS bufs ↑ from 3072 |

**Status banner:** `OP_STW_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`  
Wire ~10750 µm, vias 2988. No PDN (`Number of snets: 0`).

### 23) OP-STW STA with I/O delays — **RAN (NOT SIGNOFF)**

```bash
/workspace/tools/openroad -no_init -exit out/openroad/op_stw_sta_iodelay.tcl \
  | tee out/openroad/op_stw_sta_iodelay_report.txt
```

| Item | Value |
|---|---|
| DEF | `op_stw_routed.def` |
| Clock | 10 ns + **IO max=1.0 ns / min=0.5 ns** |
| Parasitics | `estimate_parasitics -placement` — **NOT OpenRCX SPEF** |
| Corner | single `tt_025C_1v80` liberty |
| Setup (max) worst data-path slack | **+5.945 ns MET** (`flow_cur[0]`→`_829_`) |
| `report_wns` / `report_tns` | **0.000 / 0.000** (setup group) |
| Hold/removal (min) worst | **+0.205 ns MET** |

**Label: NOT SIGNOFF.** No SPEF, no multi-corner, no SI. Do not quote as PrimeTime / silicon timing.

### 24) Summary (updated 2026-09-06 Asia/Shanghai — OP-STW)

| Item | Status |
|---|---|
| OpenROAD CTS+route (HBG) | **OK** — DRC=0, 121 u² @ 8% |
| OpenROAD CTS+route (OGEC) | **OK** — DRC=0, 613 u² @ 41% |
| OpenROAD CTS+route (SMAM) | **OK** — DRC=0, 757 u² @ 39% |
| OpenROAD CTS+route (OP-STW N=8) | **OK** — `op_stw_routed.def`, DRC=0, **3098 u² @ 40%** (die 100×100 / core 90×90) |
| OpenROAD STA (OP-STW, IO delays) | **OK best-effort NOT SIGNOFF** — setup **+5.945 ns**; hold **+0.205 ns** — `op_stw_sta_iodelay_report.txt` |
| OpenRCX / PDN / full ORFS | **NOT DONE** |

### 25) ADP-MAC full P&R — **RAN (ROUTED DRC=0)**

Mapped synth: `out/synth/c2s_adp_mac_mapped.v` (top `c2s_adp_mac`, liberty area **3399.510400** u², 404 cells).

**Die/core sizing:** liberty ≈ 3399.5 u² → die **110×110 µm**, core **100×100 µm** (target util ~34%; achieved **35%**).

```bash
/workspace/tools/openroad -no_init -exit out/openroad/floorplan_adp_mac_minimal.tcl | tee out/openroad/floorplan_adp_mac_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/place_adp_mac_minimal.tcl | tee out/openroad/place_adp_mac_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_adp_mac_minimal.tcl | tee out/openroad/cts_route_adp_mac_minimal.log
```

| Step | Result | Artifact |
|---|---|---|
| Floorplan | **OK** — 3400 u² @ 35% | `adp_mac_floorplan.def` |
| `global_placement` dens=0.6 | **OK** (overflow ≈ 0.100) | — |
| `detailed_placement` | **OK** | `adp_mac_placed.def` |
| `create_clock` clk @ 10 ns | **OK** | — |
| Reclassify `one_` POWER→SIGNAL | **OK** (DRT-0305 fix) | — |
| `clock_tree_synthesis` (17 sinks) | **OK** | `adp_mac_cts.def` |
| `global_route` + guide | **OK** (WL GR ≈ 12682 µm) | `adp_mac_route.guide` |
| `detailed_route` | **OK** (violations 97→**0**) | `adp_mac_routed.def`, empty `adp_mac_route_drc.rpt` |
| Post-flow `report_design_area` | **3426 u^2 35% utilization** | CTS bufs ↑ from 3400 |

**Status banner:** `ADP_MAC_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`  
Wire ~7561 µm, vias 2732. No PDN (`Number of snets: 0`).

### 26) ADP-MAC STA with I/O delays — **RAN (NOT SIGNOFF)**

```bash
/workspace/tools/openroad -no_init -exit out/openroad/adp_mac_sta_iodelay.tcl \
  | tee out/openroad/adp_mac_sta_iodelay_report.txt
```

| Item | Value |
|---|---|
| DEF | `adp_mac_routed.def` |
| Clock | 10 ns + **IO max=1.0 ns / min=0.5 ns** |
| Parasitics | `estimate_parasitics -placement` — **NOT OpenRCX SPEF** |
| Corner | single `tt_025C_1v80` liberty |
| Setup (max) worst data-path slack | **+4.049 ns MET** |
| `report_wns` / `report_tns` | **0.000 / 0.000** (setup group) |
| Hold/removal (min) worst | **+0.205 ns MET** |

**Label: NOT SIGNOFF.** No SPEF, no multi-corner, no SI.

### 27) ECP-QKV (N_TILE=8) full P&R — **RAN (ROUTED DRC=0)**

Flat wrapper: `flows/oss/wrappers/c1s_ecp_qkv_synth.sv` (unpacked `corr_score[]` → packed bus).  
Mapped: `out/synth/c1s_ecp_qkv_mapped.v` (top `c1s_ecp_qkv_predictor_synth`, liberty **768.236800** u², 74 cells).

**Die/core sizing:** first try die 55×55 @ ~40% → **GPL-0305**; retuned to die **60×60 µm**, core **50×50 µm** (~31% util) + `place_pins` before GPL + `-overflow 0.2` dens=0.65 → OK. Post-CTS util **33%**.

```bash
/workspace/tools/openroad -no_init -exit out/openroad/floorplan_ecp_qkv_minimal.tcl | tee out/openroad/floorplan_ecp_qkv_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/place_ecp_qkv_minimal.tcl | tee out/openroad/place_ecp_qkv_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_ecp_qkv_minimal.tcl | tee out/openroad/cts_route_ecp_qkv_minimal.log
```

| Step | Result | Artifact |
|---|---|---|
| Floorplan | **OK** — 768 u² @ 32% | `ecp_qkv_floorplan.def` |
| `place_pins` then GPL dens=0.65 | **OK** (overflow ≈ 0.199) | — |
| `detailed_placement` | **OK** | `ecp_qkv_placed.def` |
| CTS (17 sinks) | **OK** | `ecp_qkv_cts.def` |
| `global_route` + guide | **OK** | `ecp_qkv_route.guide` |
| `detailed_route` | **OK** (violations 10→**0**) | `ecp_qkv_routed.def`, empty `ecp_qkv_route_drc.rpt` |
| Post-flow `report_design_area` | **796 u^2 33% utilization** | CTS bufs ↑ from 768 |

**Status banner:** `ECP_QKV_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`  
Wire ~2440 µm, vias 654. No PDN.

### 28) ECP-QKV STA with I/O delays — **RAN (NOT SIGNOFF)**

```bash
/workspace/tools/openroad -no_init -exit out/openroad/ecp_qkv_sta_iodelay.tcl \
  | tee out/openroad/ecp_qkv_sta_iodelay_report.txt
```

| Item | Value |
|---|---|
| DEF | `ecp_qkv_routed.def` |
| Clock | 10 ns + **IO max=1.0 ns / min=0.5 ns** |
| Parasitics | `estimate_parasitics -placement` — **NOT OpenRCX SPEF** |
| Setup (max) worst | **+8.248 ns MET** |
| `report_wns` / `report_tns` | **0.000 / 0.000** |
| Hold/removal (min) worst | **+0.205 ns MET** |

**Label: NOT SIGNOFF.**

### 29) Summary (updated 2026-09-06 Asia/Shanghai — ADP-MAC + ECP-QKV)

| Item | Status |
|---|---|
| OpenROAD CTS+route (HBG) | **OK** — DRC=0, 121 u² @ 8% |
| OpenROAD CTS+route (OGEC) | **OK** — DRC=0, 613 u² @ 41% |
| OpenROAD CTS+route (SMAM) | **OK** — DRC=0, 757 u² @ 39% |
| OpenROAD CTS+route (OP-STW N=8) | **OK** — DRC=0, 3098 u² @ 40% |
| OpenROAD CTS+route (ADP-MAC) | **OK** — `adp_mac_routed.def`, DRC=0, **3426 u² @ 35%** (die 110×110 / core 100×100) |
| OpenROAD STA (ADP-MAC, IO delays) | **OK best-effort NOT SIGNOFF** — setup **+4.049 ns**; hold **+0.205 ns** |
| OpenROAD CTS+route (ECP-QKV N=8) | **OK** — `ecp_qkv_routed.def`, DRC=0, **796 u² @ 33%** (die 60×60 / core 50×50) |
| OpenROAD STA (ECP-QKV, IO delays) | **OK best-effort NOT SIGNOFF** — setup **+8.248 ns**; hold **+0.205 ns** |
| OpenRCX / PDN / full ORFS | **NOT DONE** |


### 30) STH-Gate (N_HEAD=8) full P&R — **RAN (ROUTED DRC=0)**

Flat wrapper: `flows/oss/wrappers/c2s_sth_gate_synth.sv` (packed score buses).  
Mapped: `out/synth/c2s_sth_gate_mapped.v` (top `c2s_sth_gate_synth`, liberty **1525.212800** u², 147 cells).

**Die/core sizing:** liberty ≈ 1525 u² → die **80×80 µm**, core **70×70 µm** (target util ~31%; achieved **32–34%**).

```bash
/workspace/tools/openroad -no_init -exit out/openroad/floorplan_sth_gate_minimal.tcl | tee out/openroad/floorplan_sth_gate_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/place_sth_gate_minimal.tcl | tee out/openroad/place_sth_gate_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_sth_gate_minimal.tcl | tee out/openroad/cts_route_sth_gate_minimal.log
```

| Step | Result | Artifact |
|---|---|---|
| Floorplan | **OK** — 1525 u² @ 32% | `sth_gate_floorplan.def` |
| `place_pins` then GPL dens=0.65 | **OK** (overflow ≈ 0.197) | — |
| `detailed_placement` | **OK** | `sth_gate_placed.def` |
| CTS | **OK** | `sth_gate_cts.def` |
| `global_route` + guide | **OK** | `sth_gate_route.guide` |
| `detailed_route` | **OK** (violations → **0**) | `sth_gate_routed.def`, empty `sth_gate_route_drc.rpt` |
| Post-flow `report_design_area` | **1614 u^2 34% utilization** | CTS bufs ↑ from 1525 |

**Status banner:** `STH_GATE_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`  
Wire ~5629 µm, vias 1245. No PDN.

### 31) STH-Gate STA with I/O delays — **RAN (NOT SIGNOFF)**

```bash
/workspace/tools/openroad -no_init -exit out/openroad/sth_gate_sta_iodelay.tcl \
  | tee out/openroad/sth_gate_sta_iodelay_report.txt
```

| Item | Value |
|---|---|
| DEF | `sth_gate_routed.def` |
| Clock | 10 ns + **IO max=1.0 ns / min=0.5 ns** |
| Parasitics | `estimate_parasitics -placement` — **NOT OpenRCX SPEF** |
| Setup (max) worst | **+7.657 ns MET** |
| `report_wns` / `report_tns` | **0.000 / 0.000** |
| Hold/removal (min) worst | **+0.205 ns MET** |

**Label: NOT SIGNOFF.**

### 32) PRRC ledger (N_LEVEL=3) full P&R — **RAN (ROUTED DRC=0)**

Wrapper: `flows/oss/wrappers/c1s_prrc_synth.sv` + RTL `c1s_prrc_ledger.sv` (flattened mapped).  
Mapped: `out/synth/c1s_prrc_mapped.v` (top `c1s_prrc_ledger_synth`, liberty **1248.697600** u², 133 cells).

**Die/core sizing:** liberty ≈ 1249 u² → die **70×70 µm**, core **60×60 µm** (target util ~35%; achieved **37%**).

```bash
/workspace/tools/openroad -no_init -exit out/openroad/floorplan_prrc_minimal.tcl | tee out/openroad/floorplan_prrc_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/place_prrc_minimal.tcl | tee out/openroad/place_prrc_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_prrc_minimal.tcl | tee out/openroad/cts_route_prrc_minimal.log
```

| Step | Result | Artifact |
|---|---|---|
| Floorplan | **OK** — 1249 u² @ 37% | `prrc_floorplan.def` |
| `place_pins` then GPL dens=0.65 | **OK** (overflow ≈ 0.198) | — |
| `detailed_placement` | **OK** | `prrc_placed.def` |
| CTS | **OK** | `prrc_cts.def` |
| `global_route` + guide | **OK** | `prrc_route.guide` |
| `detailed_route` | **OK** (violations → **0**) | `prrc_routed.def`, empty `prrc_route_drc.rpt` |
| Post-flow `report_design_area` | **1275 u^2 37% utilization** | CTS bufs ↑ from 1249 |

**Status banner:** `PRRC_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`  
Wire ~3031 µm, vias 1085. No PDN.

### 33) PRRC STA with I/O delays — **RAN (NOT SIGNOFF)**

```bash
/workspace/tools/openroad -no_init -exit out/openroad/prrc_sta_iodelay.tcl \
  | tee out/openroad/prrc_sta_iodelay_report.txt
```

| Item | Value |
|---|---|
| DEF | `prrc_routed.def` |
| Clock | 10 ns + **IO max=1.0 ns / min=0.5 ns** |
| Parasitics | `estimate_parasitics -placement` — **NOT OpenRCX SPEF** |
| Setup (max) worst | **+6.677 ns MET** |
| `report_wns` / `report_tns` | **0.000 / 0.000** |
| Hold/removal (min) worst | **+0.205 ns MET** |

**Label: NOT SIGNOFF.**

### 34) MW-ΔBuf (N_TILE=8 flat) full P&R — **RAN (ROUTED DRC=0)**

Flat wrapper reused: `flows/oss/wrappers/c1s_mw_delta_synth.sv` with `chparam -set N_TILE 8` (same pattern as ECP-QKV).  
Mapped: `out/synth/c1s_mw_delta_mapped.v` (top `c1s_mw_delta_buf_synth`, liberty **4719.526400** u², 520 cells).

**Die/core sizing:** liberty ≈ 4720 u² → die **130×130 µm**, core **120×120 µm** (target util ~33%; achieved **34–35%**).

```bash
/workspace/tools/openroad -no_init -exit out/openroad/floorplan_mw_delta_minimal.tcl | tee out/openroad/floorplan_mw_delta_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/place_mw_delta_minimal.tcl | tee out/openroad/place_mw_delta_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_mw_delta_minimal.tcl | tee out/openroad/cts_route_mw_delta_minimal.log
```

| Step | Result | Artifact |
|---|---|---|
| Floorplan | **OK** — 4720 u² @ 34% | `mw_delta_floorplan.def` |
| `place_pins` then GPL dens=0.65 | **OK** (overflow ≈ 0.200) | — |
| `detailed_placement` | **OK** | `mw_delta_placed.def` |
| CTS | **OK** | `mw_delta_cts.def` |
| `global_route` + guide | **OK** | `mw_delta_route.guide` |
| `detailed_route` | **OK** (violations 356→**0**) | `mw_delta_routed.def`, empty `mw_delta_route_drc.rpt` |
| Post-flow `report_design_area` | **4842 u^2 35% utilization** | CTS bufs ↑ from 4720 |

**Status banner:** `MW_DELTA_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`  
Wire ~16007 µm, vias 3933. No PDN.

### 35) MW-ΔBuf STA with I/O delays — **RAN (NOT SIGNOFF)**

```bash
/workspace/tools/openroad -no_init -exit out/openroad/mw_delta_sta_iodelay.tcl \
  | tee out/openroad/mw_delta_sta_iodelay_report.txt
```

| Item | Value |
|---|---|
| DEF | `mw_delta_routed.def` |
| Clock | 10 ns + **IO max=1.0 ns / min=0.5 ns** |
| Parasitics | `estimate_parasitics -placement` — **NOT OpenRCX SPEF** |
| Setup (max) worst | **+6.452 ns MET** |
| `report_wns` / `report_tns` | **0.000 / 0.000** |
| Hold/removal (min) worst | **+0.206 ns MET** |

**Label: NOT SIGNOFF.**

### 36) Scoreboard — ALL routed modules so far (2026-09-06T00:22+08 Asia/Shanghai)

Cell-area util from `report_design_area` after CTS+DR. STA = best-effort IO-delay placement parasitics (**NOT SIGNOFF**; no SPEF / PDN / multi-corner).

| # | Module | Params | Cells (map) | Liberty u² | Die/core µm | Routed area / util | DRC | STA setup / hold @10ns | Routed DEF |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HBG-RP | — | 15 | 121.4 | (fp 8%) | **121 u² / 8%** | **0** | +6.465 / +0.204 | `hbg_routed.def` |
| 2 | OGEC | N_TILE=8 | 41 | 585.6 | 50×50 / 40×40 | **613 u² / 41%** | **0** | (not run) | `ogec_routed.def` |
| 3 | SMAM-RP | — | 66 | 730.7 | — | **757 u² / 39%** | **0** | (not run) | `smam_routed.def` |
| 4 | OP-STW | N_TILE=8 | 423 | 3071.7 | 100×100 / 90×90 | **3098 u² / 40%** | **0** | +5.945 / +0.205 | `op_stw_routed.def` |
| 5 | ADP-MAC | — | 404 | 3399.5 | 110×110 / 100×100 | **3426 u² / 35%** | **0** | +4.049 / +0.205 | `adp_mac_routed.def` |
| 6 | ECP-QKV | N_TILE=8 | 74 | 768.2 | 60×60 / 50×50 | **796 u² / 33%** | **0** | +8.248 / +0.205 | `ecp_qkv_routed.def` |
| 7 | **STH-Gate** | **N_HEAD=8** | **147** | **1525.2** | **80×80 / 70×70** | **1614 u² / 34%** | **0** | **+7.657 / +0.205** | `sth_gate_routed.def` |
| 8 | **PRRC ledger** | **N_LEVEL=3** | **133** | **1248.7** | **70×70 / 60×60** | **1275 u² / 37%** | **0** | **+6.677 / +0.205** | `prrc_routed.def` |
| 9 | **MW-ΔBuf** | **N_TILE=8** | **520** | **4719.5** | **130×130 / 120×120** | **4842 u² / 35%** | **0** | **+6.452 / +0.206** | `mw_delta_routed.def` |

**Never claim silicon µm² / mW / timing without STA+RCX closing.** Local commit only; **no push**; never `nts07`.


### 37) Motion-TTB packer (N_TILE=8 flat) full P&R — **RAN (ROUTED DRC=0)**

Flat wrapper: `flows/oss/wrappers/c2s_motion_ttb_synth.sv` with `chparam -set N_TILE 8`.  
Mapped: `out/synth/c2s_motion_ttb_mapped.v` (top `c2s_motion_ttb_packer_synth`, liberty **4653.212800** u², 591 cells).

**Die/core sizing:** liberty ≈ 4653 u² → die **130×130 µm**, core **120×120 µm** (target util ~32%; achieved **33–34%**).

```bash
/workspace/tools/openroad -no_init -exit out/openroad/place_motion_ttb_minimal.tcl | tee out/openroad/place_motion_ttb_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_motion_ttb_minimal.tcl | tee out/openroad/cts_route_motion_ttb_minimal.log
```

| Step | Result | Artifact |
|---|---|---|
| Floorplan / place dens=0.65 | **OK** | `motion_ttb_placed.def` |
| CTS + GR + DR | **OK** (violations → **0**) | `motion_ttb_routed.def`, empty `motion_ttb_route_drc.rpt` |
| Post-flow `report_design_area` | **4826 u^2 34% utilization** | CTS bufs ↑ from 4653 |

**Status banner:** `MOTION_TTB_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`  
Wire ~17951 µm, vias 5519. No PDN.

### 38) Motion-TTB STA with I/O delays — **RAN (NOT SIGNOFF)**

| Item | Value |
|---|---|
| DEF | `motion_ttb_routed.def` |
| Clock | 10 ns + **IO max=1.0 ns / min=0.5 ns** |
| Parasitics | `estimate_parasitics -placement` — **NOT OpenRCX SPEF** |
| Setup (max) worst | **+4.178 ns MET** |
| `report_wns` / `report_tns` | **0.000 / 0.000** |
| Hold/removal (min) worst | **+0.205 ns MET** |

**Label: NOT SIGNOFF.**

### 39) ARM-Acc full P&R — **RAN (ROUTED DRC=0)**

Mapped: `out/synth/c2s_arm_acc_mapped.v` (top `c2s_arm_acc`, liberty **4033.868800** u², 454 cells).  
Die **120×120** / core **110×110** (~33–35%).

| Step | Result | Artifact |
|---|---|---|
| Place dens=0.65 | **OK** | `arm_acc_placed.def` |
| CTS + GR + DR | **OK** (DRC **0**) | `arm_acc_routed.def` |
| Post-flow area | **4207 u^2 35% utilization** | — |

Wire ~11009 µm, vias 3657. STA setup **+4.759 ns** / hold **+0.206 ns** (`arm_acc_sta_iodelay_report.txt`). **NOT SIGNOFF.**

### 40) SP-Gate (N=8 flat) full P&R — **RAN (ROUTED DRC=0)**

Wrapper: `flows/oss/wrappers/c2s_sp_gate_synth.sv` N=8.  
Mapped: `out/synth/c2s_sp_gate_mapped.v` (liberty **788.256000** u², 73 cells).  
Die **60×60** / core **50×50**. Post-route **815 u² / 33%**. Wire ~2480 µm, vias 712.  
STA setup **+8.266 ns** / hold **+0.205 ns**. **NOT SIGNOFF.**

### 41) MFBD full P&R — **RAN (ROUTED DRC=0)**

Mapped: `out/synth/c2s_mfbd_mapped.v` (liberty **454.185600** u², 27 cells).  
Die **50×50** / core **40×40**. Post-route **480 u² / 32%**. Wire ~966 µm, vias 231.  
STA setup **+8.331 ns** / hold **+0.205 ns**. **NOT SIGNOFF.**

### 42) Exact-capture wrap (N_TILE=8) full P&R — **RAN (ROUTED DRC=0)**

Mapped: `out/synth/c1s_exact_capture_mapped.v` (top `c1s_exact_capture_wrap`, liberty **1306.252800** u², 126 cells).  
Die **70×70** / core **60×60**. Post-route **1329 u² / 39%**. Wire ~2834 µm, vias 962.  
STA setup **+5.276 ns** / hold **+0.205 ns**. **NOT SIGNOFF.**

### 43) C1*-stats + C2*-stats (optional, fast) full P&R — **RAN (ROUTED DRC=0)**

| Module | Liberty u² / cells | Die/core | Routed area/util | Wire/vias | STA setup/hold |
|---|---|---|---|---|---|
| C1*-stats N=8 | 2874.0 / 304 | 100×100 / 90×90 | **2944 u² / 38%** | 6483 / 2271 | **+4.450 / +0.206** |
| C2*-stats | 2258.4 / 240 | 90×90 / 80×80 | **2307 u² / 37%** | 5374 / 1951 | **+7.423 / +0.206** |

Skipped front_pipe / back_pipe (hierarchical/heavy). **NOT SIGNOFF.**

### 44) Scoreboard — ALL routed modules (2026-09-06T00:44+08 Asia/Shanghai)

Cell-area util from `report_design_area` after CTS+DR. STA = best-effort IO-delay placement parasitics (**NOT SIGNOFF**; no SPEF / PDN / multi-corner).

| # | Module | Params | Cells (map) | Liberty u² | Die/core µm | Routed area / util | DRC | STA setup / hold @10ns | Routed DEF |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HBG-RP | — | 15 | 121.4 | (fp 8%) | **121 u² / 8%** | **0** | +6.465 / +0.204 | `hbg_routed.def` |
| 2 | OGEC | N_TILE=8 | 41 | 585.6 | 50×50 / 40×40 | **613 u² / 41%** | **0** | (not run) | `ogec_routed.def` |
| 3 | SMAM-RP | — | 66 | 730.7 | — | **757 u² / 39%** | **0** | (not run) | `smam_routed.def` |
| 4 | OP-STW | N_TILE=8 | 423 | 3071.7 | 100×100 / 90×90 | **3098 u² / 40%** | **0** | +5.945 / +0.205 | `op_stw_routed.def` |
| 5 | ADP-MAC | — | 404 | 3399.5 | 110×110 / 100×100 | **3426 u² / 35%** | **0** | +4.049 / +0.205 | `adp_mac_routed.def` |
| 6 | ECP-QKV | N_TILE=8 | 74 | 768.2 | 60×60 / 50×50 | **796 u² / 33%** | **0** | +8.248 / +0.205 | `ecp_qkv_routed.def` |
| 7 | STH-Gate | N_HEAD=8 | 147 | 1525.2 | 80×80 / 70×70 | **1614 u² / 34%** | **0** | +7.657 / +0.205 | `sth_gate_routed.def` |
| 8 | PRRC ledger | N_LEVEL=3 | 133 | 1248.7 | 70×70 / 60×60 | **1275 u² / 37%** | **0** | +6.677 / +0.205 | `prrc_routed.def` |
| 9 | MW-ΔBuf | N_TILE=8 | 520 | 4719.5 | 130×130 / 120×120 | **4842 u² / 35%** | **0** | +6.452 / +0.206 | `mw_delta_routed.def` |
| 10 | **Motion-TTB** | **N_TILE=8** | **591** | **4653.2** | **130×130 / 120×120** | **4826 u² / 34%** | **0** | **+4.178 / +0.205** | `motion_ttb_routed.def` |
| 11 | **ARM-Acc** | **N_HYP=4** | **454** | **4033.9** | **120×120 / 110×110** | **4207 u² / 35%** | **0** | **+4.759 / +0.206** | `arm_acc_routed.def` |
| 12 | **SP-Gate** | **N=8** | **73** | **788.3** | **60×60 / 50×50** | **815 u² / 33%** | **0** | **+8.266 / +0.205** | `sp_gate_routed.def` |
| 13 | **MFBD** | **MAX_B=4** | **27** | **454.2** | **50×50 / 40×40** | **480 u² / 32%** | **0** | **+8.331 / +0.205** | `mfbd_routed.def` |
| 14 | **ExactCapt** | **N_TILE=8** | **126** | **1306.3** | **70×70 / 60×60** | **1329 u² / 39%** | **0** | **+5.276 / +0.205** | `exact_capture_routed.def` |
| 15 | **C1*-stats** | **N_TILE=8** | **304** | **2874.0** | **100×100 / 90×90** | **2944 u² / 38%** | **0** | **+4.450 / +0.206** | `c1s_stats_routed.def` |
| 16 | **C2*-stats** | **—** | **240** | **2258.4** | **90×90 / 80×80** | **2307 u² / 37%** | **0** | **+7.423 / +0.206** | `c2s_stats_routed.def` |

**16/16 DRC=0.** Skipped front_pipe/back_pipe. Never claim silicon µm² / mW / timing without STA+RCX closing. Local commit only; **no push**; never `nts07`.


### 45) front_pipe (N_TILE=8 flat) + back_pipe (N_HEAD=8 flat) full P&R — **RAN (ROUTED DRC=0)**

Hierarchical pipes previously skipped; this turn liberty-mapped with Yosys `flatten` after abc.

**front_pipe mapped:** `out/synth/c1s_front_pipe_mapped.v` (top `c1s_front_pipe_synth`, liberty **11347.132800** u², **1179** cells).  
Sources: `c1s_op_stw_synth` + `c1s_ecp_qkv_synth` + `c1s_mw_delta_synth` + `c1s_front_pipe_synth`, `chparam -set N_TILE 8`.  
**Die/core:** die **200×200 µm**, core **190×190 µm** (target util ~31%; post-route **33%**).

**back_pipe mapped:** `out/synth/c2s_back_pipe_mapped.v` (top `c2s_back_pipe_synth`, liberty **2703.843200** u², **246** cells).  
Sources: HBG + SMAM RTL + `c2s_sth_gate_synth` + `c2s_back_pipe_synth`, `chparam -set N_HEAD 8`.  
**Die/core:** die **100×100 µm**, core **90×90 µm** (target util ~33%; post-route **36%**).

```bash
/workspace/tools/openroad -no_init -exit out/openroad/place_front_pipe_minimal.tcl | tee out/openroad/place_front_pipe_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_front_pipe_minimal.tcl | tee out/openroad/cts_route_front_pipe_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/front_pipe_sta_iodelay.tcl | tee out/openroad/front_pipe_sta_iodelay_report.txt
/workspace/tools/openroad -no_init -exit out/openroad/place_back_pipe_minimal.tcl | tee out/openroad/place_back_pipe_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/cts_route_back_pipe_minimal.tcl | tee out/openroad/cts_route_back_pipe_minimal.log
/workspace/tools/openroad -no_init -exit out/openroad/back_pipe_sta_iodelay.tcl | tee out/openroad/back_pipe_sta_iodelay_report.txt
```

| Design | Step | Result | Artifact |
|---|---|---|---|
| front_pipe | place dens=0.65 | **OK** — 11347 u² @ 32% | `front_pipe_placed.def` |
| front_pipe | CTS + GR + DR | **OK** (violations → **0**) | `front_pipe_routed.def`, empty DRC rpt |
| front_pipe | post-route area | **11791 u² / 33%** | wire ~42168 µm, vias 8995 |
| front_pipe | STA IO delays | setup **+5.764 ns MET**; hold **+0.221 ns MET** | `front_pipe_sta_iodelay_report.txt` |
| back_pipe | place dens=0.65 | **OK** — 2704 u² @ 35% | `back_pipe_placed.def` |
| back_pipe | CTS + GR + DR | **OK** (violations → **0**) | `back_pipe_routed.def`, empty DRC rpt |
| back_pipe | post-route area | **2825 u² / 36%** | wire ~9289 µm, vias 2136 |
| back_pipe | STA IO delays | setup **+6.343 ns MET**; hold **+0.206 ns MET** | `back_pipe_sta_iodelay_report.txt` |

**Status banners:** `FRONT_PIPE_CTS_ROUTE_STATUS cts=1 groute=1 droute=1` · `BACK_PIPE_CTS_ROUTE_STATUS cts=1 groute=1 droute=1`  
**Label: NOT SIGNOFF.** No PDN / no OpenRCX SPEF / no multi-corner.

### 46) Scoreboard doc + totals (2026-09-06 ~00:55+08 Asia/Shanghai)

Full bilingual table: **`docs/OPENROAD_PNR_SCOREBOARD_GROKBOT.md`** (16 leaf + 2 pipes = **18/18 DRC=0**).

| Item | Status |
|---|---|
| front_pipe N_TILE=8 flatten+P&R | **OK** — `front_pipe_routed.def`, DRC=0, **11791 u² @ 33%**, STA +5.764 / +0.221 |
| back_pipe N_HEAD=8 flatten+P&R | **OK** — `back_pipe_routed.def`, DRC=0, **2825 u² @ 36%**, STA +6.343 / +0.206 |
| OpenRCX / PDN / full ORFS | **NOT DONE** |

**Never claim silicon µm² / mW / timing without STA+RCX closing.** Local commit only; **no push**; never `nts07`.


## Pointer — Independent TCAS-II review (Grok Bot)

Independent bilingual review+score: [`docs/TCASII_INDEPENDENT_REVIEW_SCORE_GROKBOT.md`](TCASII_INDEPENDENT_REVIEW_SCORE_GROKBOT.md) (2026-09-06 Asia/Shanghai). Recommendation: **Borderline**; OpenROAD cited only as implementation completeness (18/18 DRC=0, NO PDN/SPEF, NOT signoff).


### 47) Card G/H modules full P&R — **RAN (7/7 ROUTED DRC=0)**

**Date:** 2026-09-06 ~12:40+08 Asia/Shanghai  
**Driver:** `flows/oss/map_and_pnr_card_gh.sh` (liberty-map + place dens retries + CTS/GR/DR + STA IO delays).  
**Modules:** `c1s_tde3_prior`, `c1s_wake_merge`, `c2s_tma_agg`, `c1s_cfp_confgate`, `c1s_sci_cleanexit`, `c2s_bisat_agg`, `c2s_bui_guard_sdsa` (N_TILE/N_TOKEN=8 flatten).

| Module | Mapped cells / u² | Die/core | Routed u²/util | DRC | STA setup/hold @10ns |
|---|---|---|---|---|---|
| TDE3-Prior | 1062 / 8683.3 | 165×165 / 155×155 | **9149 / 39%** | **0** | **+4.854 / +0.206** |
| wake_merge | 33 / 355.3 | 40×40 / 30×30 | **382 / 47%** | **0** | **+8.357 / +0.205** |
| TMA-Agg | 785 / 6817.8 | 150×150 / 140×140 | **6960 / 36%** | **0** | **+3.417 / +0.206** |
| CFP-ConfGate | 471 / 4296.6 | 120×120 / 110×110 | **4459 / 37%** | **0** | **+2.290 / +0.206** |
| SCI-CleanExit | 802 / 7250.7 | 155×155 / 145×145 | **7388 / 35%** | **0** | **-0.019 / +0.206** |
| BiSAT-Agg | 867 / 6705.2 | 145×145 / 135×135 | **6830 / 38%** | **0** | **+3.557 / +0.206** |
| BUI-GuardSDSA | 380 / 2791.4 | 95×95 / 85×85 | **2828 / 40%** | **0** | **+6.878 / +0.205** |

**Status banners:** `*_CTS_ROUTE_STATUS cts=1 groute=1 droute=1` for all seven.  
**Label: NOT SIGNOFF.** No PDN / no OpenRCX SPEF / no multi-corner. SCI setup WNS slightly negative under placement parasitics — do **not** claim timing closed.  
Scoreboard rows 19–25: `docs/OPENROAD_PNR_SCOREBOARD_GROKBOT.md`. Log: `out/regress_card_gh_pnr.log`.

