#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-results/local5_erep_formal_canary_20260811}"
PROFILE="results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
VECTOR_DIR="$OUT_DIR/vectors"
BUILD_DIR="$OUT_DIR/build"
mkdir -p "$OUT_DIR" "$VECTOR_DIR" "$BUILD_DIR"
rm -rf "$BUILD_DIR/verilator_obj"

EXPECTED_SCRIPT="scripts/local5_erep_formal_canary_expected.py"
ACTUAL_SCRIPT="scripts/local5_erep_formal_canary_actual.py"
MERGE_SCRIPT="scripts/local5_erep_formal_canary_merge.py"
VECTOR_SCRIPT="scripts/generate_local5_active_projection_postg0_vectors.py"
TB="tb_qfit/tb_qfit_local5_active_projection_postg0.sv"
RTL=(
  rtl_qfit/qfit_local5_1rw_active_projection_tile.sv
  rtl_qfit/qfit_dual_color_relation_frontier_sync.sv
  rtl_qfit/qfit_dual_color_word_skipper_index.sv
  rtl_qfit/qfit_sync_relation_bank.sv
  rtl_qfit/qfit_fakeram45_relation_bank_450.sv
  rtl_qfit/qfit_source_multicast_term_builder_fifo2.sv
  rtl_qfit/qfit_source_multicast_term_builder.sv
  rtl_qfit/qfit_local5_1rw_projection_backend.sv
  rtl_qfit/qfit_local5_color_map.sv
  rtl_qfit/qfit_direct_1rw_acc_bank.sv
  rtl_qfit/qfit_gasr2c_acc_bank.sv
  rtl_qfit/qfit_single_port_acc_memory.sv
)
ASSERTIONS=(
  verif_qfit/qfit_local5_1rw_active_projection_assertions.sv
  verif_qfit/qfit_direct_1rw_acc_bank_assertions.sv
  verif_qfit/qfit_single_port_acc_memory_assertions.sv
)
FILELIST=("$TB" "${RTL[@]}")

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  python3 --version
  iverilog -V 2>&1 | sed -n '1p'
  verilator --version
} >"$OUT_DIR/tool_versions.txt"

python3 -m py_compile \
  "$EXPECTED_SCRIPT" "$ACTUAL_SCRIPT" "$MERGE_SCRIPT" "$VECTOR_SCRIPT" \
  tests/test_local5_erep_formal_canary.py \
  >"$OUT_DIR/py_compile.log" 2>&1
python3 -m unittest tests.test_local5_erep_formal_canary \
  >"$OUT_DIR/unittest.log" 2>&1

python3 "$EXPECTED_SCRIPT" \
  --profile "$PROFILE" --sample 0 --stage 0 --block 0 \
  --output-dir "$OUT_DIR/software_expected" \
  >"$OUT_DIR/software_expected.log" 2>&1

python3 "$VECTOR_SCRIPT" \
  --input-dir "$PROFILE" \
  --output-dir "$VECTOR_DIR" \
  --out-dim 32 \
  --weight-mode checkpoint_theta_folded_dyadic_int8_head_slice \
  --task-plan "$OUT_DIR/software_expected/task_plan.json" \
  --omit-expected-acc \
  >"$OUT_DIR/vector_generation.log" 2>&1

COMMON_PARAMS=(
  -Ptb_qfit_local5_active_projection_postg0.NEW_1RW_BACKEND=1
  -Ptb_qfit_local5_active_projection_postg0.MODE=0
  -Ptb_qfit_local5_active_projection_postg0.GROUPS=9
  -Ptb_qfit_local5_active_projection_postg0.RUN_GROUPS=9
  -Ptb_qfit_local5_active_projection_postg0.OUT_DIM=32
)

iverilog -g2012 -s tb_qfit_local5_active_projection_postg0 \
  "${COMMON_PARAMS[@]}" -o "$BUILD_DIR/canary_iv" \
  "$TB" "${RTL[@]}" >"$OUT_DIR/iverilog_compile.log" 2>&1
ICARUS_COMPILE_CMD="iverilog -g2012 -s tb_qfit_local5_active_projection_postg0 ${COMMON_PARAMS[*]} -o $BUILD_DIR/canary_iv ${FILELIST[*]}"
ICARUS_CMD="vvp $BUILD_DIR/canary_iv +VECTOR_DIR=$VECTOR_DIR +CHECKPOINT_WEIGHTS +NO_ACC_CHECK +ACTUAL_ACC_FILE=$OUT_DIR/actual_icarus.memh"
vvp "$BUILD_DIR/canary_iv" \
  "+VECTOR_DIR=$VECTOR_DIR" +CHECKPOINT_WEIGHTS +NO_ACC_CHECK \
  "+ACTUAL_ACC_FILE=$OUT_DIR/actual_icarus.memh" \
  >"$OUT_DIR/icarus.log" 2>&1
python3 "$ACTUAL_SCRIPT" \
  --simulator icarus --log "$OUT_DIR/icarus.log" \
  --actual "$OUT_DIR/actual_icarus.memh" \
  --vector-manifest "$VECTOR_DIR/manifest.json" \
  --task-plan "$OUT_DIR/software_expected/task_plan.json" \
  --filelist "${FILELIST[@]}" --command "$ICARUS_CMD" \
  --compile-command "$ICARUS_COMPILE_CMD" \
  --executable "$BUILD_DIR/canary_iv" \
  --tool-versions "$OUT_DIR/tool_versions.txt" \
  --output "$OUT_DIR/actual_icarus_receipt.json" \
  >"$OUT_DIR/actual_icarus_adapter.log" 2>&1

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-BLKSEQ -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module tb_qfit_local5_active_projection_postg0 \
  --Mdir "$BUILD_DIR/verilator_obj" \
  -GNEW_1RW_BACKEND=1 -GMODE=0 -GGROUPS=9 -GRUN_GROUPS=9 -GOUT_DIM=32 \
  "$TB" "${RTL[@]}" "${ASSERTIONS[@]}" \
  >"$OUT_DIR/verilator_compile.log" 2>&1
VERILATOR_BIN="$BUILD_DIR/verilator_obj/Vtb_qfit_local5_active_projection_postg0"
VERILATOR_COMPILE_CMD="verilator --binary --timing --assert -Wall -Wno-fatal -Wno-BLKSEQ -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC --top-module tb_qfit_local5_active_projection_postg0 --Mdir $BUILD_DIR/verilator_obj -GNEW_1RW_BACKEND=1 -GMODE=0 -GGROUPS=9 -GRUN_GROUPS=9 -GOUT_DIM=32 ${FILELIST[*]} ${ASSERTIONS[*]}"
VERILATOR_CMD="$VERILATOR_BIN +VECTOR_DIR=$VECTOR_DIR +CHECKPOINT_WEIGHTS +NO_ACC_CHECK +ACTUAL_ACC_FILE=$OUT_DIR/actual_verilator.memh"
"$VERILATOR_BIN" \
  "+VECTOR_DIR=$VECTOR_DIR" +CHECKPOINT_WEIGHTS +NO_ACC_CHECK \
  "+ACTUAL_ACC_FILE=$OUT_DIR/actual_verilator.memh" \
  >"$OUT_DIR/verilator.log" 2>&1
python3 "$ACTUAL_SCRIPT" \
  --simulator verilator --log "$OUT_DIR/verilator.log" \
  --actual "$OUT_DIR/actual_verilator.memh" \
  --vector-manifest "$VECTOR_DIR/manifest.json" \
  --task-plan "$OUT_DIR/software_expected/task_plan.json" \
  --filelist "${FILELIST[@]}" "${ASSERTIONS[@]}" --command "$VERILATOR_CMD" \
  --compile-command "$VERILATOR_COMPILE_CMD" \
  --executable "$VERILATOR_BIN" \
  --tool-versions "$OUT_DIR/tool_versions.txt" \
  --output "$OUT_DIR/actual_verilator_receipt.json" \
  >"$OUT_DIR/actual_verilator_adapter.log" 2>&1

python3 "$MERGE_SCRIPT" \
  --task-plan "$OUT_DIR/software_expected/task_plan.json" \
  --expected "$OUT_DIR/software_expected/software_expected.npz" \
  --expected-receipt "$OUT_DIR/software_expected/software_expected_receipt.json" \
  --actual "$OUT_DIR/actual_icarus.memh" \
  --actual "$OUT_DIR/actual_verilator.memh" \
  --actual-receipt "$OUT_DIR/actual_icarus_receipt.json" \
  --actual-receipt "$OUT_DIR/actual_verilator_receipt.json" \
  --output "$OUT_DIR/merge_report.json" \
  >"$OUT_DIR/merge.log" 2>&1

sha256sum \
  "$EXPECTED_SCRIPT" "$ACTUAL_SCRIPT" "$MERGE_SCRIPT" "$VECTOR_SCRIPT" \
  tests/test_local5_erep_formal_canary.py "$TB" "${RTL[@]}" "${ASSERTIONS[@]}" \
  sim_qfit/run_local5_erep_formal_canary_checks.sh \
  >"$OUT_DIR/source_sha256.txt"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 \
  --numeric-owner -cf "$OUT_DIR/source_bundle.tar" \
  "$EXPECTED_SCRIPT" "$ACTUAL_SCRIPT" "$MERGE_SCRIPT" "$VECTOR_SCRIPT" \
  tests/test_local5_erep_formal_canary.py "$TB" "${RTL[@]}" \
  "${ASSERTIONS[@]}" sim_qfit/run_local5_erep_formal_canary_checks.sh
sha256sum \
  "$OUT_DIR"/{tool_versions.txt,py_compile.log,unittest.log,software_expected.log,vector_generation.log,iverilog_compile.log,icarus.log,actual_icarus_adapter.log,actual_icarus_receipt.json,verilator_compile.log,verilator.log,actual_verilator_adapter.log,actual_verilator_receipt.json,merge.log,merge_report.json,actual_icarus.memh,actual_verilator.memh,source_sha256.txt,source_bundle.tar} \
  "$OUT_DIR/software_expected"/{task_plan.json,software_expected.npz,software_expected_receipt.json} \
  "$VECTOR_DIR/manifest.json" "$VECTOR_DIR"/*.memh \
  >"$OUT_DIR/result_sha256.txt"

python3 - "$OUT_DIR/complete.json" "$OUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[2]).resolve()
report = json.loads((out / "merge_report.json").read_text(encoding="utf-8"))
value = {
    "schema": "local5_erep_formal_canary_complete_v1",
    "status": report["status"],
    "evidence": report["evidence"],
    "formal_g0": "DENY",
    "output_directory": str(out),
    "result_sha256_file_sha256": hashlib.sha256(
        (out / "result_sha256.txt").read_bytes()
    ).hexdigest(),
}
Path(sys.argv[1]).write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
sha256sum "$OUT_DIR/result_sha256.txt" "$OUT_DIR/complete.json" \
  >"$OUT_DIR/receipt_sha256.txt"

echo "PASS Local5 formal source-isolation canary formal_g0=DENY output=$OUT_DIR"
