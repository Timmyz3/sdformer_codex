#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# C1* ablation ladder: ALWAYS vs OP-STW vs +ECP vs +MW vs +EXACT(OGEC×PRRC)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/sim out/waves

VVP=out/sim/c1s_ablation_ladder.vvp
LOGDIR=out/sim
MD=out/ABLATION_C1_LADDER.md
STAMP_SH="$(TZ=Asia/Shanghai date +%Y-%m-%dT%H:%M:%S%z)"

echo "[ablation_c1_ladder] compile"
iverilog -g2012 -o "$VVP" \
  rtl_c1star/c1s_op_stw_predictor.sv \
  rtl_c1star/c1s_ecp_qkv_predictor.sv \
  rtl_c1star/c1s_mw_delta_buf.sv \
  rtl_c1star/c1s_stats.sv \
  rtl_c1star/c1s_ogec_gate.sv \
  rtl_c1star/c1s_prrc_ledger.sv \
  rtl_c1star/c1s_exact_capture_wrap.sv \
  tb_c1star/tb_c1s_ablation_ladder.sv

declare -A WAKE SKIP DNZ EXACT
MODES=(ALWAYS OPSTW ECP MW EXACT)
FAIL=0
for m in "${MODES[@]}"; do
  log="$LOGDIR/ablation_${m}.log"
  echo "[ablation_c1_ladder] MODE=$m"
  set +e
  vvp "$VVP" +MODE="$m" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]] || ! grep -q "PASS: ablation ladder mode=$m" "$log"; then
    echo "[ablation_c1_ladder] FAIL mode=$m" >&2
    FAIL=1
    WAKE[$m]=ERR; SKIP[$m]=ERR; DNZ[$m]=ERR; EXACT[$m]=ERR
    continue
  fi
  line=$(grep '^ABLATION_LINE' "$log" | tail -1)
  WAKE[$m]=$(echo "$line" | sed -n 's/.*wake_pop=\([0-9]*\).*/\1/p')
  SKIP[$m]=$(echo "$line" | sed -n 's/.*proj_skip=\([0-9]*\).*/\1/p')
  DNZ[$m]=$(echo "$line" | sed -n 's/.*delta_nz=\([0-9]*\).*/\1/p')
  EXACT[$m]=$(echo "$line" | sed -n 's/.*exact_hit=\([0-9]*\).*/\1/p')
done

{
  echo "# C1* Ablation Ladder (Grok Bot)"
  echo
  echo "**Generated (Asia/Shanghai):** ${STAMP_SH}  "
  echo "**Tree:** \`sdformer_c1c2star_grokbot\`  "
  echo "**Harness:** \`flows/oss/ablation_c1_ladder.sh\` + \`tb_c1star/tb_c1s_ablation_ladder.sv\`  "
  echo "**Stimulus:** 8 deterministic frames, N_TILE=8 (same vectors all modes)  "
  echo "**Counters:** \`c1s_stats\` (wake_pop / proj_skip / delta_nz) + \`exact_capture\` (exact_hit)"
  echo
  echo "## Ladder"
  echo
  echo "| Rung | Mode | wake_pop | proj_skip | delta_nz | exact_hit |"
  echo "|---:|---|---:|---:|---:|---:|"
  echo "| 1 | always-ish (\`ALWAYS\`) | ${WAKE[ALWAYS]} | ${SKIP[ALWAYS]} | ${DNZ[ALWAYS]} | ${EXACT[ALWAYS]} |"
  echo "| 2 | OP-STW only (\`OPSTW\`) | ${WAKE[OPSTW]} | ${SKIP[OPSTW]} | ${DNZ[OPSTW]} | ${EXACT[OPSTW]} |"
  echo "| 3 | OP-STW + ECP (\`ECP\`) | ${WAKE[ECP]} | ${SKIP[ECP]} | ${DNZ[ECP]} | ${EXACT[ECP]} |"
  echo "| 4 | + MW-ΔBuf (\`MW\`) | ${WAKE[MW]} | ${SKIP[MW]} | ${DNZ[MW]} | ${EXACT[MW]} |"
  echo "| 5 | + OGEC×PRRC (\`EXACT\`) | ${WAKE[EXACT]} | ${SKIP[EXACT]} | ${DNZ[EXACT]} | ${EXACT[EXACT]} |"
  echo
  echo "## Interpretation (letter hygiene)"
  echo
  echo "- **ALWAYS**: force all-tile wake (+high corr) — upper bound on PE wake / exact enqueue."
  echo "- **OPSTW**: optical-flow/event wake only; corr forced 0 → proj tracks wake; no residual OR."
  echo "- **ECP**: real corr_score gates proj with OP-STW wake (no MW)."
  echo "- **MW**: residual \`delta_nz\` ORs into wake for ECP (full front-pipe style); \`allow_exact=1\`."
  echo "- **EXACT**: same wake as MW, but PRRC \`INIT_BUDGET=3\` spends 1 per OGEC beat → \`allow_exact\` drops → **exact_hit capped** vs MW."
  echo "- \`exact_hit\` = OGEC×capture; CAPACITY=128."
  echo "- Not AEE / not silicon power — RTL counter ablation only."
  echo
  echo "## Logs"
  echo
  for m in "${MODES[@]}"; do
    echo "- \`out/sim/ablation_${m}.log\`"
  done
} > "$MD"

echo "[ablation_c1_ladder] wrote $MD"
cat "$MD"
[[ $FAIL -eq 0 ]] || exit 1
echo "[ablation_c1_ladder] PASS"
