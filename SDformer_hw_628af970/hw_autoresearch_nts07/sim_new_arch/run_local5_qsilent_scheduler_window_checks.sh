#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/local5_qsilent_scheduler_window_20260813}"
VEC="${VECTOR_DIR:-$ROOT/tb_qfit/vectors/local5_qsilent_s0b0_window_20260813}"
BUILD="$OUT/build"
mkdir -p "$BUILD" "$OUT" "$VEC"
cd "$ROOT"

python3 scripts/generate_local5_qsilent_window_vectors.py --output-dir "$VEC"

RTL=(
  rtl_hitflow/gatestack_output_tile_scheduler.sv
  rtl_qfit/qfit_local5_encoder_job_scheduler.sv
  rtl_local5/local5_shiftmax5_q17.sv
  rtl_qfit/qfit_tagged_compactor4.sv
  rtl_qfit/qfit_xorbank_compactor4.sv
  rtl_qfit/qfit_local5_score_leaf.sv
  rtl_qfit/qfit_local5_qsilent_score_leaf.sv
)

iverilog -g2012 -Wall -Wno-timescale \
  -s tb_qfit_local5_qsilent_scheduler_window \
  -o "$BUILD/qs_sched.vvp" "${RTL[@]}" \
  tb_qfit/tb_qfit_local5_qsilent_scheduler_window.sv \
  >"$OUT/iverilog_build.log" 2>&1
vvp "$BUILD/qs_sched.vvp" "+VECTOR_DIR=$VEC" | tee "$OUT/scheduler_window_iverilog.log"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-UNUSEDSIGNAL -Wno-PINCONNECTEMPTY -Wno-PINMISSING \
  --top-module tb_qfit_local5_qsilent_scheduler_window \
  --Mdir "$BUILD/verilator_obj" \
  "${RTL[@]}" \
  verif_hitflow/gatestack_output_tile_scheduler_assertions.sv \
  verif_hitflow/bind_gatestack_output_tile_scheduler_assertions.sv \
  verif_qfit/qfit_local5_encoder_job_scheduler_assertions.sv \
  verif_qfit/qfit_score_leaf_assertions.sv \
  verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv \
  tb_qfit/tb_qfit_local5_qsilent_scheduler_window.sv \
  >"$OUT/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_qfit_local5_qsilent_scheduler_window" \
  "+VECTOR_DIR=$VEC" | tee "$OUT/scheduler_window_verilator.log"

python3 - << PY
import json, re
from pathlib import Path
text = Path("$OUT/scheduler_window_verilator.log").read_text()
if "PASS tb_qfit_local5_qsilent_scheduler_window" not in text:
    raise SystemExit("missing PASS")
m = re.search(r"QS_SCHED_SUM real_heads=(\d+) checked=(\d+) dummy=(\d+) score_cycles=(\d+) wall=(\d+) qsilent_hits=(\d+)", text)
if not m:
    raise SystemExit("missing summary")
rep = {
    "schema": "local5_qsilent_scheduler_window_v1",
    "status": "PASS",
    "evidence": "[rtl]+[scheduler+qsilent-same-top]+[window-id-remapped]",
    "real_heads": int(m.group(1)),
    "checked": int(m.group(2)),
    "dummy_jobs": int(m.group(3)),
    "score_cycles": int(m.group(4)),
    "frame_wall_cycles": int(m.group(5)),
    "qsilent_hits": int(m.group(6)),
    "identity": json.loads(Path("$VEC/manifest.json").read_text()),
    "claim_boundary": [
        "Only the first S0.B0 scheduler window runs real Q/K; other jobs are 1-cycle dummy done.",
        "Profile window 94 is remapped onto scheduler window 0. Topology matches.",
        "Not 21600-group RTL or full encoder.",
    ],
}
out = Path("$OUT")
(out / "report.json").write_text(json.dumps(rep, indent=2) + "\\n")
(out / "report.md").write_text(
    "# Local5 scheduler + Q-silent same-top window\\n\\n"
    f"- real heads {rep['real_heads']}, score/gate checks {rep['checked']}\\n"
    f"- Q-silent hits {rep['qsilent_hits']}, score cycles {rep['score_cycles']}\\n"
    f"- full-frame scheduler wall {rep['frame_wall_cycles']} (dummy jobs {rep['dummy_jobs']})\\n"
    "- window id remapped 94->0; only this window is numeric RTL.\\n"
)
print("PASS Local5 scheduler+Q-silent report", rep["checked"])
PY

echo "PASS Local5 Q-silent scheduler window flow"
