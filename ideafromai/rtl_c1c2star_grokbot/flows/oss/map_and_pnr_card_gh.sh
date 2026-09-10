#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Liberty-map + best-effort OpenROAD P&R for Card G/H modules (N_TILE=8 default).
# NO PDN, NO SPEF, NOT signoff. Mirror prior leaf-module pattern.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
LIB=/workspace/pdks/sky130hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
OUTOR=out/openroad
OUTSY=out/synth
mkdir -p "$OUTOR" "$OUTSY"

# tag short_name top rtl_path [optional extra params note]
# die sizing done after liberty area known

map_one() {
  local name="$1" top="$2" rtl="$3"
  local mapped="$OUTSY/${name}_mapped.v"
  local stat="$OUTSY/${name}_mapped.stat.txt"
  local log="$OUTSY/${name}_mapped.ys.log"
  echo "[map] $name top=$top"
  yosys -l "$log" -p "
    read_liberty -lib $LIB;
    read_verilog -sv ${rtl};
    hierarchy -check -top ${top};
    proc; opt; memory; opt; fsm; opt;
    techmap; opt;
    flatten;
    opt;
    dfflibmap -liberty $LIB;
    abc -liberty $LIB;
    clean;
    tee -o ${stat} stat -liberty $LIB;
    write_verilog -noattr ${mapped}
  "
  echo "[map] wrote $mapped"
  grep -E "Chip area|Number of cells:" "$stat" || true
}

# Parse Chip area from stat; echo float
get_area() {
  local stat="$1"
  awk '/Chip area for module/{print $(NF)}' "$stat" | head -1
}
get_cells() {
  local stat="$1"
  awk '/Number of cells:/{print $NF; exit}' "$stat"
}

# Choose die/core for ~30-40% util given liberty area A
# die side = ceil(sqrt(A/0.32)) rounded up to nice 10s, core = die-10
size_die() {
  local area="$1"
  python3 - <<PY
import math
A=float("$area")
if A < 1: A=1
side=math.ceil(math.sqrt(A/0.32)/5.0)*5
if side < 40: side=40
# keep core margin 5um each side (10 total) like prior scripts when die>=50
if side <= 50:
  die=side; core=side-10
else:
  die=side; core=side-10
if core < 30: core=die-10
print(f"{die} {core}")
PY
}

gen_place_tcl() {
  local tag="$1" top="$2" mapped="$3" die="$4" core="$5"
  local margin=$(( (die - core) / 2 ))
  local c0=$margin
  local c1=$((die - margin))
  cat > "$OUTOR/place_${tag}_minimal.tcl" << TCL
# GROKBOT NEW FILE -- iscas_ssh
# ${tag}: floorplan + place_pins + global_placement + detailed_placement
set LIB $LIB
set TLEF /workspace/pdks/sky130hd/lef/sky130_fd_sc_hd.tlef
set LEF /workspace/pdks/sky130hd/lef/sky130_fd_sc_hd_merged.lef
set NET /workspace/sdformer_c1c2star_grokbot/${mapped}
set OUT /workspace/sdformer_c1c2star_grokbot/${OUTOR}
read_liberty \$LIB
read_lef \$TLEF
read_lef \$LEF
read_verilog \$NET
link_design ${top}
initialize_floorplan -die_area {0 0 ${die} ${die}} -core_area {${c0} ${c0} ${c1} ${c1}} -site unithd
puts "FLOORPLAN_OK die=${die}x${die}um core=${core}x${core}um"
source /workspace/pdks/sky130hd/make_tracks.tcl
puts "TRACKS_OK"
if {[catch {place_pins -hor_layers met3 -ver_layers met2} err]} {
  puts "PLACE_PINS_SKIP: \$err"
} else {
  puts "PLACE_PINS_OK"
}
set gp_ok 0
foreach dens {0.65 0.55 0.45 0.35} {
  if {[catch {global_placement -density \$dens -overflow 0.2} err]} {
    puts "GLOBAL_PLACE_RETRY dens=\$dens FAIL: \$err"
  } else {
    puts "GLOBAL_PLACE_OK dens=\$dens"
    set gp_ok 1
    break
  }
}
if {!\$gp_ok} {
  if {[catch {global_placement -density 0.5 -skip_nesterov_place} err]} {
    puts "GLOBAL_PLACE_SKIP_NESTEROV_FAIL: \$err"
  } else {
    puts "GLOBAL_PLACE_OK dens=0.5 skip_nesterov"
    set gp_ok 1
  }
}
if {!\$gp_ok} {
  puts "GLOBAL_PLACE_ALL_FAILED"
  report_design_area
  write_def \$OUT/${tag}_placed.def
  exit
}
report_design_area
write_def \$OUT/${tag}_placed.def
puts "WROTE_DEF out/openroad/${tag}_placed.def"
if {[catch {detailed_placement} err]} {
  puts "DETAILED_PLACE_SKIP: \$err"
} else {
  puts "DETAILED_PLACE_OK"
  report_design_area
  write_def \$OUT/${tag}_placed.def
}
exit
TCL
}

gen_cts_tcl() {
  local tag="$1"
  local TAGU
  TAGU=$(echo "$tag" | tr 'a-z' 'A-Z')
  cat > "$OUTOR/cts_route_${tag}_minimal.tcl" << TCL
# GROKBOT NEW FILE -- iscas_ssh
# ${tag}: CTS + GR + DR from placed DEF. NOT full ORFS.
set LIB $LIB
set TLEF /workspace/pdks/sky130hd/lef/sky130_fd_sc_hd.tlef
set LEF /workspace/pdks/sky130hd/lef/sky130_fd_sc_hd_merged.lef
set DEF /workspace/sdformer_c1c2star_grokbot/${OUTOR}/${tag}_placed.def
set OUT /workspace/sdformer_c1c2star_grokbot/${OUTOR}
read_liberty \$LIB
read_lef \$TLEF
read_lef \$LEF
read_def \$DEF
puts "READ_PLACED_DEF_OK"
set block [[[ord::get_db] getChip] getBlock]
foreach net_name {one_ zero_} {
  set net [\$block findNet \$net_name]
  if {\$net != "NULL"} {
    \$net setSigType "SIGNAL"
    puts "RECLASSIFIED_NET \$net_name -> SIGNAL"
  }
}
create_clock -name clk -period 10.0 [get_ports clk]
puts "CREATE_CLOCK_OK period=10ns"
if {[catch {source /workspace/pdks/sky130hd/tech/setRC.tcl} err]} {
  puts "SETRC_SKIP: \$err"
} else { puts "SETRC_OK" }
if {[catch {place_pins -hor_layers met3 -ver_layers met2} err]} {
  puts "PLACE_PINS_SKIP: \$err"
} else { puts "PLACE_PINS_OK" }
set cts_ok 0
if {[catch {
  clock_tree_synthesis -buf_list {sky130_fd_sc_hd__clkbuf_1 sky130_fd_sc_hd__clkbuf_2 sky130_fd_sc_hd__clkbuf_4 sky130_fd_sc_hd__clkbuf_8} \
    -root_buf sky130_fd_sc_hd__clkbuf_4 \
    -sink_clustering_enable \
    -sink_clustering_size 25 \
    -sink_clustering_max_diameter 50
} err]} {
  puts "CTS_FAIL: \$err"
} else {
  puts "CTS_OK"
  set cts_ok 1
  catch {set_propagated_clock [all_clocks]}
  catch {detailed_placement}
  report_design_area
  write_def \$OUT/${tag}_cts.def
}
set groute_ok 0
catch {set_routing_layers -signal met1-met5 -clock met3-met5}
catch {set_global_routing_layer_adjustment met1-met5 0.2}
if {[catch {
  global_route -guide_file \$OUT/${tag}_route.guide -congestion_iterations 50 -verbose
} err]} {
  puts "GLOBAL_ROUTE_FAIL: \$err"
} else {
  puts "GLOBAL_ROUTE_OK"
  set groute_ok 1
  write_guides \$OUT/${tag}_route.guide
}
if {\$groute_ok} { catch {estimate_parasitics -global_routing} }
set droute_ok 0
if {\$groute_ok} {
  if {[catch {
    detailed_route -output_drc \$OUT/${tag}_route_drc.rpt \
      -output_maze \$OUT/${tag}_route_maze.log \
      -verbose 1
  } err]} {
    puts "DETAILED_ROUTE_FAIL: \$err"
  } else {
    puts "DETAILED_ROUTE_OK"
    set droute_ok 1
    write_def \$OUT/${tag}_routed.def
  }
}
puts "=== ${TAGU}_CTS_ROUTE_STATUS cts=\$cts_ok groute=\$groute_ok droute=\$droute_ok ==="
report_design_area
catch {
  set fa [open \$OUT/${tag}_routed_area.txt w]
  puts \$fa [report_design_area]
  close \$fa
}
exit
TCL
}

gen_sta_tcl() {
  local tag="$1"
  local TAGU
  TAGU=$(echo "$tag" | tr 'a-z' 'A-Z')
  cat > "$OUTOR/${tag}_sta_iodelay.tcl" << TCL
# GROKBOT NEW FILE -- iscas_ssh
# ${tag} STA with IO delays. NOT signoff.
set LIB $LIB
set TLEF /workspace/pdks/sky130hd/lef/sky130_fd_sc_hd.tlef
set LEF /workspace/pdks/sky130hd/lef/sky130_fd_sc_hd_merged.lef
set OUT /workspace/sdformer_c1c2star_grokbot/${OUTOR}
set DEF \$OUT/${tag}_routed.def
read_liberty \$LIB
read_lef \$TLEF
read_lef \$LEF
read_def \$DEF
puts "READ_DEF_OK STA_DEF ${tag}_routed.def"
create_clock -name clk -period 10.0 [get_ports clk]
puts "CREATE_CLOCK_OK period=10ns"
foreach pin [all_inputs] {
  set pname [get_property \$pin full_name]
  if {\$pname eq "clk"} { continue }
  set_input_delay -clock clk -max 1.0 \$pin
  set_input_delay -clock clk -min 0.5 \$pin
}
set_output_delay -clock clk -max 1.0 [all_outputs]
set_output_delay -clock clk -min 0.5 [all_outputs]
puts "IO_DELAY_OK in/out max=1.0ns min=0.5ns"
set para_mode "liberty_cell_delays_only_zero_wire_load"
catch {source /workspace/pdks/sky130hd/tech/setRC.tcl}
if {[catch {estimate_parasitics -placement} err]} {
  puts "EST_PARASITICS_PLACEMENT_SKIP: \$err"
} else {
  puts "EST_PARASITICS_OK (placement estimate — NOT OpenRCX SPEF)"
  set para_mode "placement_estimate_not_rcx_spef"
}
puts "=============================================="
puts "${TAGU} STA WITH IO DELAYS — NOT SIGNOFF"
puts "parasitics_mode=\$para_mode"
puts "=============================================="
puts "--- SETUP (max) top 10 ---"
catch {report_checks -path_delay max -fields {slew cap input_pins} -digits 3 -group_count 10}
puts "--- HOLD (min) top 5 ---"
catch {report_checks -path_delay min -digits 3 -group_count 5}
puts "--- WNS/TNS ---"
catch {report_wns -digits 3}
catch {report_tns -digits 3}
catch {report_worst_slack -max -digits 3}
catch {report_worst_slack -min -digits 3}
puts "=== ${TAGU}_STA_IODELAY_DONE para=\$para_mode ==="
exit
TCL
}

run_or() {
  local tcl="$1" log="$2"
  /workspace/tools/openroad -no_init -exit "$tcl" 2>&1 | tee "$log"
}

# ---- module table ----
# tag | synth_name | top | rtl | short_label
MODULES=(
  "tde3_prior|c1s_tde3_prior|c1s_tde3_prior|rtl_c1star/c1s_tde3_prior.sv|TDE3-Prior"
  "wake_merge|c1s_wake_merge|c1s_wake_merge|rtl_c1star/c1s_wake_merge.sv|wake_merge"
  "tma_agg|c2s_tma_agg|c2s_tma_agg|rtl_c2star/c2s_tma_agg.sv|TMA-Agg"
  "cfp_confgate|c1s_cfp_confgate|c1s_cfp_confgate|rtl_c1star/c1s_cfp_confgate.sv|CFP-ConfGate"
  "sci_cleanexit|c1s_sci_cleanexit|c1s_sci_cleanexit|rtl_c1star/c1s_sci_cleanexit.sv|SCI-CleanExit"
  "bisat_agg|c2s_bisat_agg|c2s_bisat_agg|rtl_c2star/c2s_bisat_agg.sv|BiSAT-Agg"
  "bui_guard_sdsa|c2s_bui_guard_sdsa|c2s_bui_guard_sdsa|rtl_c2star/c2s_bui_guard_sdsa.sv|BUI-GuardSDSA"
)

RESULTS_CSV="$OUTOR/card_gh_pnr_results.csv"
echo "tag,label,cells,mapped_u2,die,core,routed_u2,util,drc,setup,hold,status" > "$RESULTS_CSV"

for entry in "${MODULES[@]}"; do
  IFS='|' read -r tag sname top rtl label <<< "$entry"
  echo "======== MODULE $tag ($label) ========"
  map_one "$sname" "$top" "$rtl"
  area=$(get_area "$OUTSY/${sname}_mapped.stat.txt")
  cells=$(get_cells "$OUTSY/${sname}_mapped.stat.txt")
  read -r die core <<< "$(size_die "$area")"
  echo "[size] area=$area cells=$cells die=${die} core=${core}"
  gen_place_tcl "$tag" "$top" "$OUTSY/${sname}_mapped.v" "$die" "$core"
  gen_cts_tcl "$tag"
  gen_sta_tcl "$tag"

  status="FAIL"
  routed_u2="NA"; util="NA"; drc="NA"; setup="NA"; hold="NA"

  if ! run_or "$OUTOR/place_${tag}_minimal.tcl" "$OUTOR/place_${tag}_minimal.log"; then
    echo "[FAIL] place $tag"
    echo "$tag,$label,$cells,$area,$die,$core,$routed_u2,$util,$drc,$setup,$hold,PLACE_FAIL" >> "$RESULTS_CSV"
    continue
  fi
  if ! grep -q "DETAILED_PLACE_OK\|GLOBAL_PLACE_OK" "$OUTOR/place_${tag}_minimal.log"; then
    echo "[FAIL] place no OK banner $tag"
    echo "$tag,$label,$cells,$area,$die,$core,$routed_u2,$util,$drc,$setup,$hold,PLACE_NO_OK" >> "$RESULTS_CSV"
    continue
  fi

  if ! run_or "$OUTOR/cts_route_${tag}_minimal.tcl" "$OUTOR/cts_route_${tag}_minimal.log"; then
    echo "[FAIL] cts/route $tag"
    echo "$tag,$label,$cells,$area,$die,$core,$routed_u2,$util,$drc,$setup,$hold,CTS_ROUTE_FAIL" >> "$RESULTS_CSV"
    continue
  fi

  if grep -q "droute=1" "$OUTOR/cts_route_${tag}_minimal.log"; then
    status="ROUTED"
  else
    status="CTS_PARTIAL"
  fi

  # DRC: empty rpt or Complete 0
  if [[ -f "$OUTOR/${tag}_route_drc.rpt" ]]; then
    if [[ ! -s "$OUTOR/${tag}_route_drc.rpt" ]] || grep -qiE 'violation.*\b0\b|Total.*0' "$OUTOR/${tag}_route_drc.rpt" 2>/dev/null; then
      # also check log for final violations
      if grep -E 'violations = 0|Number of.*violations.*= *0' "$OUTOR/cts_route_${tag}_minimal.log" >/dev/null 2>&1 \
         || ! grep -qiE 'violations = [1-9]' "$OUTOR/cts_route_${tag}_minimal.log"; then
        # Prefer explicit end count from log
        drc_line=$(grep -E '\[INFO DRT-[0-9]+\] *Total number of violations' "$OUTOR/cts_route_${tag}_minimal.log" | tail -1 || true)
        if [[ -n "$drc_line" ]]; then
          drc=$(echo "$drc_line" | grep -oE '[0-9]+$' || echo 0)
        else
          # empty rpt historically means 0
          if [[ ! -s "$OUTOR/${tag}_route_drc.rpt" ]]; then drc=0; else drc="CHK"; fi
        fi
      else
        drc="NONZERO"
      fi
    else
      drc="NONZERO"
    fi
  else
    drc="NO_RPT"
  fi
  # better DRC parse from log (OpenROAD prints "violations = N" at end)
  vline=$(grep -E 'Complete with [0-9]+ violations|Total number of violations *= *[0-9]+' "$OUTOR/cts_route_${tag}_minimal.log" | tail -1 || true)
  if [[ -n "$vline" ]]; then
    drc=$(echo "$vline" | grep -oE '[0-9]+' | tail -1)
  elif [[ ! -s "$OUTOR/${tag}_route_drc.rpt" ]] && [[ -f "$OUTOR/${tag}_routed.def" ]]; then
    drc=0
  fi

  # area from last report_design_area in log
  al=$(grep -E 'Design area [0-9.]+ u\^2 [0-9]+% utilization' "$OUTOR/cts_route_${tag}_minimal.log" | tail -1 || true)
  if [[ -n "$al" ]]; then
    routed_u2=$(echo "$al" | grep -oE '[0-9.]+' | head -1)
    util=$(echo "$al" | grep -oE '[0-9]+%' | head -1)
  fi
  echo "$al" > "$OUTOR/${tag}_routed_area.txt" || true

  if [[ -f "$OUTOR/${tag}_routed.def" ]]; then
    run_or "$OUTOR/${tag}_sta_iodelay.tcl" "$OUTOR/${tag}_sta_iodelay_report.txt" || true
    # parse worst slack max/min
    setup=$(grep -E 'worst slack|wns' -i "$OUTOR/${tag}_sta_iodelay_report.txt" | head -5 || true)
    # OpenROAD report_worst_slack prints "worst slack  X.XXX"
    setup=$(awk '/worst slack/{print $NF; exit}' "$OUTOR/${tag}_sta_iodelay_report.txt" || echo NA)
    # two report_worst_slack: max then min — take first as setup, second as hold
    mapfile -t slacks < <(grep -E 'worst slack' "$OUTOR/${tag}_sta_iodelay_report.txt" | awk '{print $NF}')
    if ((${#slacks[@]}>=1)); then setup="${slacks[0]}"; fi
    if ((${#slacks[@]}>=2)); then hold="${slacks[1]}"; fi
    # fallback: parse from path slack lines "slack (MET)" 
    if [[ "$setup" == "NA" || -z "$setup" ]]; then
      setup=$(grep -E '^\s+slack \(' "$OUTOR/${tag}_sta_iodelay_report.txt" | head -1 | awk '{print $NF}' || echo NA)
    fi
    if [[ "$hold" == "NA" || -z "$hold" ]]; then
      hold=$(grep -E '^\s+slack \(' "$OUTOR/${tag}_sta_iodelay_report.txt" | sed -n '11,20p' | head -1 | awk '{print $NF}' || echo NA)
    fi
  fi

  echo "$tag,$label,$cells,$area,$die,$core,$routed_u2,$util,$drc,$setup,$hold,$status" >> "$RESULTS_CSV"
  echo "[done] $tag status=$status drc=$drc area=$routed_u2 $util setup=$setup hold=$hold"
done

echo "======== ALL DONE ========"
cat "$RESULTS_CSV"
