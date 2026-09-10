#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Yosys synth for Card A/B + ECP + MW + SMAM + Motion-TTB + STH + OGEC + PRRC + ADP-MAC + ARM-Acc + MFBD + SP-Gate + front_pipe + back_pipe
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/synth

run_one() {
  local name="$1"
  local top="$2"
  local rtl="$3"
  local log="out/synth/${name}.ys.log"
  local stat="out/synth/${name}.stat.txt"

  echo "[run_synth] ${name} top=${top}"
  yosys -l "$log" -p "
    read_verilog -sv ${rtl};
    synth -top ${top};
    abc;
    opt;
    tee -o ${stat} stat;
    write_verilog -noattr out/synth/${name}_netlist.v
  " || {
    echo "[run_synth] WARN: abc pass failed or unavailable; retrying synth+stat only"
    yosys -l "$log" -p "
      read_verilog -sv ${rtl};
      synth -top ${top};
      tee -o ${stat} stat;
      write_verilog -noattr out/synth/${name}_netlist.v
    "
  }
  echo "[run_synth] wrote ${stat}"
  grep -E "Number of (cells|wires|wire bits|ports|port bits)" "$stat" || true
}

# Card A: flattened synth wrapper (N_TILE=64 default)
run_one "c1s_op_stw" "c1s_op_stw_predictor_synth" "flows/oss/wrappers/c1s_op_stw_synth.sv"

# Card B: native packetizer
run_one "c2s_hbg_rp" "c2s_hbg_rp_packetizer" "rtl_c2star/c2s_hbg_rp_packetizer.sv"

# ECP-QKV flat
run_one "c1s_ecp_qkv" "c1s_ecp_qkv_predictor_synth" "flows/oss/wrappers/c1s_ecp_qkv_synth.sv"

# MW-ΔBuf flat
run_one "c1s_mw_delta" "c1s_mw_delta_buf_synth" "flows/oss/wrappers/c1s_mw_delta_synth.sv"

# SMAM-RP real dual-rail
run_one "c2s_smam_rp" "c2s_smam_rp" "rtl_c2star/c2s_smam_rp.sv"

# Motion-TTB packer flat
run_one "c2s_motion_ttb" "c2s_motion_ttb_packer_synth" "flows/oss/wrappers/c2s_motion_ttb_synth.sv"

# STH-Gate flat
run_one "c2s_sth_gate" "c2s_sth_gate_synth" "flows/oss/wrappers/c2s_sth_gate_synth.sv"

# OGEC (N_TILE=8 wrapper instantiates packed RTL)
run_one "c1s_ogec" "c1s_ogec_gate_synth" \
  "rtl_c1star/c1s_ogec_gate.sv flows/oss/wrappers/c1s_ogec_synth.sv"

# Front-pipe flat @ N_TILE=8 (instantiates OP-STW/ECP/MW synth wrappers)
run_one "c1s_front_pipe" "c1s_front_pipe_synth" \
  "flows/oss/wrappers/c1s_op_stw_synth.sv flows/oss/wrappers/c1s_ecp_qkv_synth.sv flows/oss/wrappers/c1s_mw_delta_synth.sv flows/oss/wrappers/c1s_front_pipe_synth.sv"

# Back-pipe flat @ N_HEAD=8 (HBG+SMAM RTL + STH synth wrapper)
run_one "c2s_back_pipe" "c2s_back_pipe_synth" \
  "rtl_c2star/c2s_hbg_rp_packetizer.sv rtl_c2star/c2s_smam_rp.sv flows/oss/wrappers/c2s_sth_gate_synth.sv flows/oss/wrappers/c2s_back_pipe_synth.sv"

# PRRC ledger flat (unpacked budget[])
run_one "c1s_prrc" "c1s_prrc_ledger_synth" \
  "rtl_c1star/c1s_prrc_ledger.sv flows/oss/wrappers/c1s_prrc_synth.sv"

# ADP-MAC (packed ports — synth native RTL)
run_one "c2s_adp_mac" "c2s_adp_mac" "rtl_c2star/c2s_adp_mac.sv"

# ARM-Acc (packed acc_bus — native)
run_one "c2s_arm_acc" "c2s_arm_acc" "rtl_c2star/c2s_arm_acc.sv"

# MFBD (packed I/O — native)
run_one "c2s_mfbd" "c2s_mfbd" "rtl_c2star/c2s_mfbd.sv"

# SP-Gate flat (unpacked mass[])
run_one "c2s_sp_gate" "c2s_sp_gate_synth" "flows/oss/wrappers/c2s_sp_gate_synth.sv"


# Exact-capture wrap (packed ports — native @ N_TILE=8 default)
run_one "c1s_exact_capture" "c1s_exact_capture_wrap" "rtl_c1star/c1s_exact_capture_wrap.sv"

# C1* stats (packed bitmaps — native)
run_one "c1s_stats" "c1s_stats" "rtl_c1star/c1s_stats.sv"

# C2* stats (scalar events — native)
run_one "c2s_stats" "c2s_stats" "rtl_c2star/c2s_stats.sv"



# Card G: TDE3-Prior (packed ports — native)
run_one "c1s_tde3_prior" "c1s_tde3_prior" "rtl_c1star/c1s_tde3_prior.sv"

# Card G: wake_merge
run_one "c1s_wake_merge" "c1s_wake_merge" "rtl_c1star/c1s_wake_merge.sv"

# Card G: TMA-Agg — may need flat wrapper if unpacked ports fail; try native first
run_one "c2s_tma_agg" "c2s_tma_agg" "rtl_c2star/c2s_tma_agg.sv"
run_one "c1s_cfp_confgate" "c1s_cfp_confgate" "rtl_c1star/c1s_cfp_confgate.sv"
run_one "c1s_sci_cleanexit" "c1s_sci_cleanexit" "rtl_c1star/c1s_sci_cleanexit.sv"

# Card H: BiSAT-Agg + BUI-GuardSDSA (packed native)
run_one "c2s_bisat_agg" "c2s_bisat_agg" "rtl_c2star/c2s_bisat_agg.sv"
run_one "c2s_bui_guard_sdsa" "c2s_bui_guard_sdsa" "rtl_c2star/c2s_bui_guard_sdsa.sv"

echo "[run_synth] done"
