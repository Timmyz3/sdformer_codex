#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Full OSS regression: ALL flows/oss/sim_*.sh must PASS; then synth; write reports.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/waves out/sim out/synth

FLOW="$(cd "$(dirname "$0")" && pwd)"
SUMMARY="out/SUMMARY_TCASII_OSS.md"
REPORT="out/REGRESSION_REPORT.md"
STAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP_SH="$(TZ=Asia/Shanghai date +%Y-%m-%dT%H:%M:%S%z)"

echo "=== TCAS-II OSS regression @ ${STAMP} (UTC) / ${STAMP_SH} (Asia/Shanghai) ==="

mapfile -t SIMS < <(ls -1 "$FLOW"/sim_*.sh | sort)
if [[ ${#SIMS[@]} -eq 0 ]]; then
  echo "run_all: no sim_*.sh found" >&2
  exit 1
fi

declare -a NAMES RCS RESULTS
FAIL=0
set +e
for s in "${SIMS[@]}"; do
  name="$(basename "$s" .sh)"
  echo "--- $name ---"
  bash "$s"
  rc=$?
  NAMES+=("$name")
  RCS+=("$rc")
  if [[ $rc -eq 0 ]]; then
    RESULTS+=("PASS")
  else
    RESULTS+=("FAIL")
    FAIL=1
  fi
done
set -e

bash "$FLOW/run_synth.sh"

cell_of() { grep -E 'Number of cells:' "$1" 2>/dev/null | awk '{print $NF}' | tail -1 || echo n/a; }
wbits_of() { grep -E 'Number of wire bits:' "$1" 2>/dev/null | awk '{print $NF}' | tail -1 || echo n/a; }

SIM_TABLE=""
REG_TABLE=""
for i in "${!NAMES[@]}"; do
  n="${NAMES[$i]}"
  r="${RESULTS[$i]}"
  base="${n#sim_}"
  log="out/sim/${base}_sim.log"
  case "$base" in
    op_stw) log=out/sim/c1s_op_stw_sim.log ;;
    hbg_rp) log=out/sim/c2s_hbg_rp_sim.log ;;
    ecp_qkv) log=out/sim/c1s_ecp_qkv_sim.log ;;
    mw_delta) log=out/sim/c1s_mw_delta_sim.log ;;
    smam_rp) log=out/sim/c2s_smam_rp_sim.log ;;
    motion_ttb) log=out/sim/c2s_motion_ttb_sim.log ;;
    sth_gate) log=out/sim/c2s_sth_gate_sim.log ;;
    ogec) log=out/sim/c1s_ogec_sim.log ;;
    front_pipe) log=out/sim/c1s_front_pipe_sim.log ;;
    back_pipe) log=out/sim/c2s_back_pipe_sim.log ;;
    prrc) log=out/sim/c1s_prrc_sim.log ;;
    adp_mac) log=out/sim/c2s_adp_mac_sim.log ;;
    arm_acc) log=out/sim/c2s_arm_acc_sim.log ;;
    mfbd) log=out/sim/c2s_mfbd_sim.log ;;
    sp_gate) log=out/sim/c2s_sp_gate_sim.log ;;
    exact_capture) log=out/sim/c1s_exact_capture_sim.log ;;
    c1s_stats) log=out/sim/c1s_stats_sim.log ;;
    c2s_stats) log=out/sim/c2s_stats_sim.log ;;
    ablation_c2_ladder) log=out/sim/ablation_c2_FULL.log ;;
    tde3_prior) log=out/sim/c1s_tde3_prior_sim.log ;;
    tma_agg) log=out/sim/c2s_tma_agg_sim.log ;;
    cfp_confgate) log=out/sim/c1s_cfp_confgate_sim.log ;;
    sci_cleanexit) log=out/sim/c1s_sci_cleanexit_sim.log ;;
    bisat_agg) log=out/sim/c2s_bisat_agg_sim.log ;;
    bui_guard_sdsa) log=out/sim/c2s_bui_guard_sdsa_sim.log ;;
    cfp_sci_exact_glue) log=out/sim/c1s_cfp_sci_exact_glue_sim.log ;;
    bl_veto_prior) log=out/sim/c1s_bl_veto_prior_sim.log ;;
    nl_stmfa) log=out/sim/c1s_nl_stmfa_sim.log ;;
    tid_deblur_loop) log=out/sim/c1s_tid_deblur_loop_sim.log ;;
    edc_delta_fuse) log=out/sim/c1s_edc_delta_fuse_sim.log ;;
  esac
  SIM_TABLE+="| \`${n}\` | **${r}** | \`${log}\` |"$'
'
  REG_TABLE+="| \`${n}\` | **${r}** | rc=${RCS[$i]} |"$'
'
done

SYNTH_KEYS=(
  c1s_op_stw c2s_hbg_rp c1s_ecp_qkv c1s_mw_delta c2s_smam_rp c2s_motion_ttb
  c2s_sth_gate c1s_ogec c1s_front_pipe c2s_back_pipe c1s_prrc c2s_adp_mac
  c2s_arm_acc c2s_mfbd c2s_sp_gate c1s_exact_capture c1s_stats c2s_stats
)
SYNTH_TABLE=""
for k in "${SYNTH_KEYS[@]}"; do
  f="out/synth/${k}.stat.txt"
  cells="$(cell_of "$f")"
  wb="$(wbits_of "$f")"
  SYNTH_TABLE+="| \`${k}\` | **${cells}** | ${wb} | \`${f}\` |"$'
'
done

A_PROXY=""; B_PROXY=""
if [[ -f out/waves/c1s_op_stw.vcd && -f out/synth/c1s_op_stw.stat.txt ]]; then
  A_PROXY="$(python3 "$FLOW/power_proxy.py" --label c1s_op_stw \
    --vcd out/waves/c1s_op_stw.vcd --stat out/synth/c1s_op_stw.stat.txt 2>&1 || true)"
fi
if [[ -f out/waves/c2s_hbg_rp.vcd && -f out/synth/c2s_hbg_rp.stat.txt ]]; then
  B_PROXY="$(python3 "$FLOW/power_proxy.py" --label c2s_hbg_rp \
    --vcd out/waves/c2s_hbg_rp.vcd --stat out/synth/c2s_hbg_rp.stat.txt 2>&1 || true)"
fi

OVERALL=PASS
[[ $FAIL -ne 0 ]] && OVERALL=FAIL

{
  echo "# REGRESSION REPORT (Grok Bot)"
  echo
  echo "**Generated (UTC):** ${STAMP}  "
  echo "**Local (Asia/Shanghai):** ${STAMP_SH}  "
  echo "**Tree:** \`sdformer_c1c2star_grokbot\`  "
  echo "**Branch:** \`tcasii/c1c2star-oss\`  "
  echo "**Overall:** **${OVERALL}**  "
  echo "**Sim scripts:** ${#SIMS[@]}"
  echo
  echo "## Pass/fail table"
  echo
  echo "| Script | Result | Detail |"
  echo "|---|---|---|"
  printf "%s" "$REG_TABLE"
  echo
  echo "## Notes"
  echo
  echo "- Exit non-zero if any sim fails (see \`flows/oss/regress.sh\`)."
  echo "- Yosys cell counts are generic — **NOT** µm² / NOT DC."
  echo "- No liberty → no silicon power / STA claims."
} > "$REPORT"

{
  echo "# TCAS-II OSS flow SUMMARY"
  echo
  echo "**Generated (UTC):** ${STAMP}  "
  echo "**Local (Asia/Shanghai):** ${STAMP_SH}  "
  echo "**Tree:** \`sdformer_c1c2star_grokbot\` (Grok Bot)  "
  echo "**Branch intent:** \`tcasii/c1c2star-oss\`  "
  echo "**Tools:** iverilog/vvp + yosys(+abc) + gtkwave-ready VCD + power_proxy.py  "
  echo "**Regression overall:** **${OVERALL}** (see \`out/REGRESSION_REPORT.md\`)"
  echo
  echo "## Simulation (all \`sim_*.sh\`)"
  echo
  echo "| Script | Result | Log |"
  echo "|---|---|---|"
  printf "%s" "$SIM_TABLE"
  echo
  echo "## Yosys area (generic cells — NOT µm² / NOT DC)"
  echo
  echo "| Module | Cells | Wire bits | Stat file |"
  echo "|---|---|---|---|"
  printf "%s" "$SYNTH_TABLE"
  echo
  echo "## Power proxy (NOT silicon)"
  echo
  echo "### Card A"
  echo '```'
  echo "${A_PROXY:-n/a}"
  echo '```'
  echo
  echo "### Card B"
  echo '```'
  echo "${B_PROXY:-n/a}"
  echo '```'
  echo
  echo "## Tool gaps"
  echo
  echo "- **No liberty (.lib)** → no PrimeTime-class STA, no silicon power, no µm² area."
  echo "- **OpenROAD** → see \`docs/OPENROAD_STATUS_GROKBOT.md\`."
  echo "- Yosys **abc** used as internal pass when available."
  echo "- Verilator present but primary path is iverilog."
} > "$SUMMARY"

echo "=== REGRESSION → $REPORT (overall=${OVERALL}) ==="
echo "=== SUMMARY → $SUMMARY ==="
cat "$REPORT"

if [[ $FAIL -ne 0 ]]; then
  echo "run_all: sim failure" >&2
  exit 1
fi
echo "run_all: OK"
