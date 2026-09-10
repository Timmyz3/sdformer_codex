#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# C2* ablation ladder: ALWAYS vs HBG vs HBG+SMAM vs HBG+SMAM+ADP
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p out/sim out/waves

VVP=out/sim/c2s_ablation_ladder.vvp
LOGDIR=out/sim
MD=out/ABLATION_C2_LADDER.md
STAMP_SH="$(TZ=Asia/Shanghai date +%Y-%m-%dT%H:%M:%S%z)"

echo "[ablation_c2_ladder] compile"
iverilog -g2012 -o "$VVP" \
  rtl_c2star/c2s_hbg_rp_packetizer.sv \
  rtl_c2star/c2s_smam_rp.sv \
  rtl_c2star/c2s_adp_mac.sv \
  rtl_c2star/c2s_stats.sv \
  tb_c2star/tb_c2s_ablation_ladder.sv

declare -A MAC SKIP GATE
MODES=(ALWAYS HBG HBGSMAM FULL)
FAIL=0
for m in "${MODES[@]}"; do
  log="$LOGDIR/ablation_c2_${m}.log"
  echo "[ablation_c2_ladder] MODE=$m"
  set +e
  vvp "$VVP" +MODE="$m" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]] || ! grep -q "PASS: ablation ladder mode=$m" "$log"; then
    echo "[ablation_c2_ladder] FAIL mode=$m" >&2
    FAIL=1
    MAC[$m]=ERR; SKIP[$m]=ERR; GATE[$m]=ERR
    continue
  fi
  line=$(grep '^ABLATION_LINE' "$log" | tail -1)
  MAC[$m]=$(echo "$line" | sed -n 's/.*mac_en=\([0-9]*\).*/\1/p')
  SKIP[$m]=$(echo "$line" | sed -n 's/.*skipped=\([0-9]*\).*/\1/p')
  GATE[$m]=$(echo "$line" | sed -n 's/.*gate_fire=\([0-9]*\).*/\1/p')
done

{
  echo "# C2* Ablation Ladder (Grok Bot)"
  echo
  echo "**Generated (Asia/Shanghai):** ${STAMP_SH}  "
  echo "**Tree:** \`sdformer_c1c2star_grokbot\`  "
  echo "**Harness:** \`flows/oss/ablation_c2_ladder.sh\` + \`tb_c2star/tb_c2s_ablation_ladder.sv\`  "
  echo "**Stimulus:** 16 deterministic amp beats (EPS=1), same vectors all modes  "
  echo "**Counters:** \`c2s_stats\` (mac_en / skipped / gate_fire)"
  echo
  echo "## Ladder"
  echo
  echo "| Rung | Mode | mac_en | skipped | gate_fire |"
  echo "|---:|---|---:|---:|---:|"
  echo "| 1 | always mac (\`ALWAYS\`) | ${MAC[ALWAYS]} | ${SKIP[ALWAYS]} | ${GATE[ALWAYS]} |"
  echo "| 2 | HBG-only EPS (\`HBG\`) | ${MAC[HBG]} | ${SKIP[HBG]} | ${GATE[HBG]} |"
  echo "| 3 | HBG+SMAM (\`HBGSMAM\`) | ${MAC[HBGSMAM]} | ${SKIP[HBGSMAM]} | ${GATE[HBGSMAM]} |"
  echo "| 4 | HBG+SMAM+ADP skip sides (\`FULL\`) | ${MAC[FULL]} | ${SKIP[FULL]} | ${GATE[FULL]} |"
  echo
  echo "## Interpretation (letter hygiene)"
  echo
  echo "- **ALWAYS**: force mac_en+gate every beat — upper bound on MAC activity."
  echo "- **HBG**: EPS gate only (\`|amp|>1\`); mac_en tracks gate; skipped=!gate."
  echo "- **HBGSMAM**: HBG→SMAM dual-rail; mac_en=SMAM gate_fire (aligned)."
  echo "- **FULL**: +ADP bilateral skip_a/skip_b; mac_en=do_mac; skipped=ADP skipped; gate_fire=HBG g."
  echo "- Not silicon power / not AEE — RTL counter ablation only."
  echo
  echo "## Logs"
  echo
  for m in "${MODES[@]}"; do
    echo "- \`out/sim/ablation_c2_${m}.log\`"
  done
} > "$MD"

echo "[ablation_c2_ladder] wrote $MD"
cat "$MD"
[[ $FAIL -eq 0 ]] || exit 1
echo "[ablation_c2_ladder] PASS"
