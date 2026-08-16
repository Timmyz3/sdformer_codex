#!/usr/bin/env bash
# HIT-Flow G1 full-flow verification (RTL skill style):
#   1) iverilog functional sim (leaf + top + medium + pow2safe scale)
#   2) Verilator lint-only
#   3) Verilator --binary --assert
#   4) Yosys synth readiness (proc/opt/memory/check/stat)
#   5) Erie static lint (GPT rtl + patch rtl)
#   6) Python reference unit tests
# Does not modify GPT sources; pow2safe lives under rtl_hitflow_patch/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/projection_g1_fullflow"
REPORT_DIR="$ROOT/results/rtl_signoff_20260715"
REPORT_JSON="$REPORT_DIR/g1_fullflow_report.json"
REPORT_MD="$REPORT_DIR/g1_fullflow_report.md"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD" "$REPORT_DIR"
cd "$ROOT"

STAGE_LOG="$BUILD/stages.log"
: > "$STAGE_LOG"
PASS_N=0
FAIL_N=0
declare -a STAGE_NAMES=()
declare -a STAGE_STATUS=()
declare -a STAGE_NOTES=()

record() {
  local name="$1" status="$2" note="${3:-}"
  STAGE_NAMES+=("$name")
  STAGE_STATUS+=("$status")
  STAGE_NOTES+=("$note")
  if [[ "$status" == "PASS" ]]; then
    PASS_N=$((PASS_N + 1))
  else
    FAIL_N=$((FAIL_N + 1))
  fi
  echo "[$status] $name ${note}" | tee -a "$STAGE_LOG"
}

run_stage() {
  local name="$1"
  shift
  echo ""
  echo "======== STAGE: $name ========"
  if "$@" >"$BUILD/${name//\//_}.log" 2>&1; then
    record "$name" "PASS" "log=$BUILD/${name//\//_}.log"
    return 0
  else
    record "$name" "FAIL" "log=$BUILD/${name//\//_}.log"
    echo "---- tail log ----"
    tail -40 "$BUILD/${name//\//_}.log" || true
    return 1
  fi
}

# Allow continuing after FAIL for report completeness unless STRICT=1
STRICT="${STRICT:-0}"
maybe_fail() {
  local rc=$1
  if [[ $rc -ne 0 && "$STRICT" == "1" ]]; then
    exit $rc
  fi
  return 0
}

# ---------- 1. Functional simulation (iverilog) ----------
run_stage "sim_leaf_nmf" bash -c '
  iverilog -g2012 -Wall -s tb_hitflow_nmf_g1_builder \
    -o "'"$BUILD"'/tb_nmf_g1.vvp" \
    rtl_hitflow/hitflow_nmf_g1_builder.sv \
    tb_hitflow/tb_hitflow_nmf_g1_builder.sv &&
  vvp "'"$BUILD"'/tb_nmf_g1.vvp"
' || maybe_fail $?

run_stage "sim_leaf_product" bash -c '
  iverilog -g2012 -Wall -s tb_hitflow_gate_product_engine \
    -o "'"$BUILD"'/tb_product.vvp" \
    rtl_hitflow/hitflow_gate_product_engine.sv \
    tb_hitflow/tb_hitflow_gate_product_engine.sv &&
  vvp "'"$BUILD"'/tb_product.vvp"
' || maybe_fail $?

run_stage "sim_leaf_multicast" bash -c '
  iverilog -g2012 -Wall -s tb_hitflow_segmented_multicast \
    -o "'"$BUILD"'/tb_multicast.vvp" \
    rtl_hitflow/hitflow_segmented_multicast.sv \
    tb_hitflow/tb_hitflow_segmented_multicast.sv &&
  vvp "'"$BUILD"'/tb_multicast.vvp"
' || maybe_fail $?

run_stage "sim_leaf_accumulator_gpt" bash -c '
  iverilog -g2012 -Wall -s tb_hitflow_banked_accumulator \
    -o "'"$BUILD"'/tb_accumulator.vvp" \
    rtl_hitflow/hitflow_banked_accumulator.sv \
    tb_hitflow/tb_hitflow_banked_accumulator.sv &&
  vvp "'"$BUILD"'/tb_accumulator.vvp"
' || maybe_fail $?

run_stage "sim_leaf_accumulator_pow2safe_T32" bash -c '
  iverilog -g2012 -Wall -s tb_hitflow_banked_accumulator_pow2safe \
    -o "'"$BUILD"'/tb_accumulator_pow2safe.vvp" \
    rtl_hitflow_patch/hitflow_banked_accumulator_pow2safe.sv \
    tb_hitflow/tb_hitflow_banked_accumulator_pow2safe.sv &&
  vvp "'"$BUILD"'/tb_accumulator_pow2safe.vvp"
' || maybe_fail $?

run_stage "sim_top_g1_gpt_small" bash -c '
  iverilog -g2012 -Wall -s tb_hitflow_g1_projection_top \
    -o "'"$BUILD"'/tb_g1_top.vvp" \
    rtl_hitflow/hitflow_nmf_g1_builder.sv \
    rtl_hitflow/hitflow_gate_product_engine.sv \
    rtl_hitflow/hitflow_segmented_multicast.sv \
    rtl_hitflow/hitflow_banked_accumulator.sv \
    rtl_hitflow/hitflow_g1_projection_top.sv \
    tb_hitflow/tb_hitflow_g1_projection_top.sv &&
  vvp "'"$BUILD"'/tb_g1_top.vvp"
' || maybe_fail $?

run_stage "sim_top_g1_medium_T24" bash -c '
  iverilog -g2012 -Wall -s tb_hitflow_g1_projection_top_medium \
    -o "'"$BUILD"'/tb_g1_medium.vvp" \
    rtl_hitflow/hitflow_nmf_g1_builder.sv \
    rtl_hitflow/hitflow_gate_product_engine.sv \
    rtl_hitflow/hitflow_segmented_multicast.sv \
    rtl_hitflow/hitflow_banked_accumulator.sv \
    rtl_hitflow/hitflow_g1_projection_top.sv \
    tb_hitflow/tb_hitflow_g1_projection_top_medium.sv &&
  vvp "'"$BUILD"'/tb_g1_medium.vvp"
' || maybe_fail $?

run_stage "sim_top_g1_pow2safe_T32" bash -c '
  iverilog -g2012 -Wall -s tb_hitflow_g1_projection_top_pow2_32 \
    -o "'"$BUILD"'/tb_g1_pow2_32.vvp" \
    rtl_hitflow/hitflow_nmf_g1_builder.sv \
    rtl_hitflow/hitflow_gate_product_engine.sv \
    rtl_hitflow/hitflow_segmented_multicast.sv \
    rtl_hitflow_patch/hitflow_banked_accumulator_pow2safe.sv \
    rtl_hitflow_patch/hitflow_g1_projection_top_pow2safe.sv \
    tb_hitflow/tb_hitflow_g1_projection_top_pow2_32.sv &&
  vvp "'"$BUILD"'/tb_g1_pow2_32.vvp"
' || maybe_fail $?

run_stage "sim_top_g1_pow2safe_T64" bash -c '
  iverilog -g2012 -Wall -s tb_hitflow_g1_projection_top_pow2_64 \
    -o "'"$BUILD"'/tb_g1_pow2_64.vvp" \
    rtl_hitflow/hitflow_nmf_g1_builder.sv \
    rtl_hitflow/hitflow_gate_product_engine.sv \
    rtl_hitflow/hitflow_segmented_multicast.sv \
    rtl_hitflow_patch/hitflow_banked_accumulator_pow2safe.sv \
    rtl_hitflow_patch/hitflow_g1_projection_top_pow2safe.sv \
    tb_hitflow/tb_hitflow_g1_projection_top_pow2_64.sv &&
  vvp "'"$BUILD"'/tb_g1_pow2_64.vvp"
' || maybe_fail $?

# ---------- 2. Verilator lint ----------
run_stage "lint_nmf" \
  verilator --lint-only --sv -Wall --Wno-fatal \
    --top-module hitflow_nmf_g1_builder \
    rtl_hitflow/hitflow_nmf_g1_builder.sv || maybe_fail $?

run_stage "lint_product" \
  verilator --lint-only --sv -Wall --Wno-fatal \
    --top-module hitflow_gate_product_engine \
    rtl_hitflow/hitflow_gate_product_engine.sv || maybe_fail $?

run_stage "lint_multicast" \
  verilator --lint-only --sv -Wall --Wno-fatal \
    --top-module hitflow_segmented_multicast \
    rtl_hitflow/hitflow_segmented_multicast.sv || maybe_fail $?

run_stage "lint_acc_gpt" \
  verilator --lint-only --sv -Wall --Wno-fatal \
    --top-module hitflow_banked_accumulator \
    rtl_hitflow/hitflow_banked_accumulator.sv || maybe_fail $?

run_stage "lint_acc_pow2safe" \
  verilator --lint-only --sv -Wall --Wno-fatal \
    --top-module hitflow_banked_accumulator_pow2safe \
    rtl_hitflow_patch/hitflow_banked_accumulator_pow2safe.sv || maybe_fail $?

run_stage "lint_top_gpt" \
  verilator --lint-only --sv -Wall --Wno-fatal --Wno-UNOPTFLAT \
    --top-module hitflow_g1_projection_top \
    rtl_hitflow/hitflow_nmf_g1_builder.sv \
    rtl_hitflow/hitflow_gate_product_engine.sv \
    rtl_hitflow/hitflow_segmented_multicast.sv \
    rtl_hitflow/hitflow_banked_accumulator.sv \
    rtl_hitflow/hitflow_g1_projection_top.sv || maybe_fail $?

run_stage "lint_top_pow2safe" \
  verilator --lint-only --sv -Wall --Wno-fatal --Wno-UNOPTFLAT \
    --top-module hitflow_g1_projection_top_pow2safe \
    rtl_hitflow/hitflow_nmf_g1_builder.sv \
    rtl_hitflow/hitflow_gate_product_engine.sv \
    rtl_hitflow/hitflow_segmented_multicast.sv \
    rtl_hitflow_patch/hitflow_banked_accumulator_pow2safe.sv \
    rtl_hitflow_patch/hitflow_g1_projection_top_pow2safe.sv || maybe_fail $?

# ---------- 3. Verilator assertions ----------
run_stage "assert_nmf" bash -c '
  rm -rf "'"$BUILD"'/va_nmf"
  verilator --binary --assert --timing --sv -Wall --Wno-fatal \
    -Wno-BLKSEQ --top-module tb_hitflow_nmf_g1_builder \
    --Mdir "'"$BUILD"'/va_nmf" \
    rtl_hitflow/hitflow_nmf_g1_builder.sv \
    verif_hitflow/hitflow_nmf_g1_assertions.sv \
    verif_hitflow/bind_hitflow_nmf_g1_assertions.sv \
    tb_hitflow/tb_hitflow_nmf_g1_builder.sv &&
  "'"$BUILD"'/va_nmf/Vtb_hitflow_nmf_g1_builder"
' || maybe_fail $?

run_stage "assert_product" bash -c '
  rm -rf "'"$BUILD"'/va_product"
  verilator --binary --assert --timing --sv -Wall --Wno-fatal \
    --top-module tb_hitflow_gate_product_engine \
    --Mdir "'"$BUILD"'/va_product" \
    rtl_hitflow/hitflow_gate_product_engine.sv \
    verif_hitflow/hitflow_gate_product_assertions.sv \
    verif_hitflow/bind_hitflow_gate_product_assertions.sv \
    tb_hitflow/tb_hitflow_gate_product_engine.sv &&
  "'"$BUILD"'/va_product/Vtb_hitflow_gate_product_engine"
' || maybe_fail $?

run_stage "assert_multicast" bash -c '
  rm -rf "'"$BUILD"'/va_mcast"
  verilator --binary --assert --timing --sv -Wall --Wno-fatal \
    --top-module tb_hitflow_segmented_multicast \
    --Mdir "'"$BUILD"'/va_mcast" \
    rtl_hitflow/hitflow_segmented_multicast.sv \
    verif_hitflow/hitflow_segmented_multicast_assertions.sv \
    verif_hitflow/bind_hitflow_segmented_multicast_assertions.sv \
    tb_hitflow/tb_hitflow_segmented_multicast.sv &&
  "'"$BUILD"'/va_mcast/Vtb_hitflow_segmented_multicast"
' || maybe_fail $?

run_stage "assert_acc_gpt" bash -c '
  rm -rf "'"$BUILD"'/va_acc"
  verilator --binary --assert --timing --sv -Wall --Wno-fatal \
    --top-module tb_hitflow_banked_accumulator \
    --Mdir "'"$BUILD"'/va_acc" \
    rtl_hitflow/hitflow_banked_accumulator.sv \
    verif_hitflow/hitflow_banked_accumulator_assertions.sv \
    verif_hitflow/bind_hitflow_banked_accumulator_assertions.sv \
    tb_hitflow/tb_hitflow_banked_accumulator.sv &&
  "'"$BUILD"'/va_acc/Vtb_hitflow_banked_accumulator"
' || maybe_fail $?

run_stage "assert_acc_pow2safe_T32" bash -c '
  rm -rf "'"$BUILD"'/va_acc_pow2"
  verilator --binary --assert --timing --sv -Wall --Wno-fatal \
    --top-module tb_hitflow_banked_accumulator_pow2safe \
    --Mdir "'"$BUILD"'/va_acc_pow2" \
    rtl_hitflow_patch/hitflow_banked_accumulator_pow2safe.sv \
    verif_hitflow/hitflow_banked_accumulator_assertions.sv \
    verif_hitflow/bind_hitflow_banked_accumulator_pow2safe_assertions.sv \
    tb_hitflow/tb_hitflow_banked_accumulator_pow2safe.sv &&
  "'"$BUILD"'/va_acc_pow2/Vtb_hitflow_banked_accumulator_pow2safe"
' || maybe_fail $?

# ---------- 4. Yosys synth readiness ----------
yosys_one() {
  local name="$1"
  local top="$2"
  shift 2
  local log="$BUILD/yosys_${name}.log"
  yosys -q -l "$log" -p "read_verilog -sv -defer $*; hierarchy -check -top $top; proc; opt; memory -nomap; opt; check; stat"
  if grep -q '^Warning:' "$log"; then
    echo "Yosys Warning in $log" >&2
    grep '^Warning:' "$log" >&2 || true
    return 1
  fi
  return 0
}

run_stage "yosys_nmf" yosys_one nmf hitflow_nmf_g1_builder \
  rtl_hitflow/hitflow_nmf_g1_builder.sv || maybe_fail $?
run_stage "yosys_product" yosys_one product hitflow_gate_product_engine \
  rtl_hitflow/hitflow_gate_product_engine.sv || maybe_fail $?
run_stage "yosys_multicast" yosys_one multicast hitflow_segmented_multicast \
  rtl_hitflow/hitflow_segmented_multicast.sv || maybe_fail $?
run_stage "yosys_acc_gpt" yosys_one acc_gpt hitflow_banked_accumulator \
  rtl_hitflow/hitflow_banked_accumulator.sv || maybe_fail $?
run_stage "yosys_acc_pow2safe" yosys_one acc_pow2safe hitflow_banked_accumulator_pow2safe \
  rtl_hitflow_patch/hitflow_banked_accumulator_pow2safe.sv || maybe_fail $?
run_stage "yosys_top_gpt" yosys_one top_gpt hitflow_g1_projection_top \
  rtl_hitflow/hitflow_nmf_g1_builder.sv \
  rtl_hitflow/hitflow_gate_product_engine.sv \
  rtl_hitflow/hitflow_segmented_multicast.sv \
  rtl_hitflow/hitflow_banked_accumulator.sv \
  rtl_hitflow/hitflow_g1_projection_top.sv || maybe_fail $?
run_stage "yosys_top_pow2safe" yosys_one top_pow2safe hitflow_g1_projection_top_pow2safe \
  rtl_hitflow/hitflow_nmf_g1_builder.sv \
  rtl_hitflow/hitflow_gate_product_engine.sv \
  rtl_hitflow/hitflow_segmented_multicast.sv \
  rtl_hitflow_patch/hitflow_banked_accumulator_pow2safe.sv \
  rtl_hitflow_patch/hitflow_g1_projection_top_pow2safe.sv || maybe_fail $?

# ---------- 5. Erie static lint ----------
if [[ -f "$LINTER" ]]; then
  run_stage "erie_lint_gpt_rtl" bash -c '
    for source in rtl_hitflow/*.sv; do
      python3 "'"$LINTER"'" "$source" --mode rtl
    done
  ' || maybe_fail $?
  run_stage "erie_lint_pow2safe_rtl" bash -c '
    for source in rtl_hitflow_patch/*.sv; do
      python3 "'"$LINTER"'" "$source" --mode rtl
    done
  ' || maybe_fail $?
else
  record "erie_lint" "SKIP" "linter missing: $LINTER"
fi

# ---------- 6. Python reference tests ----------
run_stage "py_projection_ref" bash -c '
  PYTHONPATH=scripts python3 -m unittest \
    scripts.test_class_gate_multicast_projection_reference \
    scripts.test_projection_scs_cycle_ledger -v
' || maybe_fail $?

# ---------- Emit reports ----------
export G1_FULLFLOW_STAGE_LOG="$STAGE_LOG"
export G1_FULLFLOW_REPORT_JSON="$REPORT_JSON"
export G1_FULLFLOW_REPORT_MD="$REPORT_MD"
export G1_FULLFLOW_LINTER="$LINTER"
python3 - <<'PY'
import json
import os
from pathlib import Path
from datetime import datetime, timezone

stage_log = Path(os.environ["G1_FULLFLOW_STAGE_LOG"])
out_json = Path(os.environ["G1_FULLFLOW_REPORT_JSON"])
out_md = Path(os.environ["G1_FULLFLOW_REPORT_MD"])
linter = os.environ.get("G1_FULLFLOW_LINTER", "")

stages = []
for line in stage_log.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line.startswith("["):
        continue
    status = line[1:line.index("]")]
    rest = line[line.index("]") + 1:].strip()
    parts = rest.split(" ", 1)
    stages.append({
        "stage": parts[0],
        "status": status,
        "note": parts[1] if len(parts) > 1 else "",
    })

pass_n = sum(1 for s in stages if s["status"] == "PASS")
fail_n = sum(1 for s in stages if s["status"] == "FAIL")
skip_n = sum(1 for s in stages if s["status"] == "SKIP")
overall = "PASS" if fail_n == 0 else "FAIL"

report = {
    "schema_version": 1,
    "run_id": "rtl-design_20260715_g1_fullflow",
    "design": "hitflow_g1_projection (+ pow2safe patch)",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "gpu_required": False,
    "flow": [
        "iverilog functional sim",
        "verilator lint-only",
        "verilator binary+assert",
        "yosys synth readiness",
        "erie static lint",
        "python reference unittest",
    ],
    "tools": {
        "iverilog": "local",
        "verilator": "local",
        "yosys": "local",
        "erie_lint": linter,
    },
    "summary": {
        "overall": overall,
        "pass": pass_n,
        "fail": fail_n,
        "skip": skip_n,
        "total": len(stages),
    },
    "stages": stages,
    "scope_notes": [
        "GPT RTL not modified; pow2safe is parallel patch for TOKENS=2^n",
        "CDC/RDC N/A (single clock domain clk_core)",
        "No DC/SAIF; yosys is structural readiness only",
        "Deploy dims covered: T=6/24/32/64; not full 162x32",
    ],
}
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# G1 Projection Full-Flow Verification Report",
    "",
    "- run_id: `{rid}`".format(rid=report["run_id"]),
    "- overall: **{o}**".format(o=overall),
    "- pass/fail/skip/total: **{p}/{f}/{s}/{t}**".format(
        p=pass_n, f=fail_n, s=skip_n, t=len(stages)
    ),
    "- GPU: not required",
    "",
    "## Flow (RTL skill aligned)",
    "",
    "1. Functional sim (iverilog/vvp)",
    "2. Lint (verilator --lint-only)",
    "3. Assertions (verilator --binary --assert)",
    "4. Synth readiness (yosys proc/opt/memory/check/stat)",
    "5. Erie static lint",
    "6. Python golden reference unit tests",
    "",
    "## Stage table",
    "",
    "| # | Stage | Status |",
    "|---:|---|---|",
]
for i, s in enumerate(stages, 1):
    lines.append("| {i} | `{n}` | **{st}** |".format(i=i, n=s["stage"], st=s["status"]))
lines += ["", "## Scope / N-A", ""]
for n in report["scope_notes"]:
    lines.append("- " + n)
lines += [
    "",
    "## Artifacts",
    "",
    "- logs: `build_hitflow/projection_g1_fullflow/`",
    "- report: `results/rtl_signoff_20260715/`",
    "",
]
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_json)
print(out_md)
print("OVERALL={o} pass={p} fail={f} skip={s}".format(
    o=overall, p=pass_n, f=fail_n, s=skip_n
))
if fail_n:
    raise SystemExit(1)
PY
