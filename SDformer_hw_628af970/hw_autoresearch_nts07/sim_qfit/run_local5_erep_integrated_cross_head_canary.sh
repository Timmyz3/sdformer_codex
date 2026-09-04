#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-results/local5_erep_integrated_cross_head_canary_20260811}"
USE_MEMO="${USE_MEMO:-0}"
VECTOR_RESULT_MODE="${VECTOR_RESULT_MODE:-0}"
SAMPLE_ID="${SAMPLE_ID:-0}"
PROFILE="results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
SOFTWARE="$OUT_DIR/software_expected"
VECTORS="$OUT_DIR/vectors"
BUILD="$OUT_DIR/build"
mkdir -p "$OUT_DIR" "$SOFTWARE" "$VECTORS" "$BUILD"
rm -rf "$BUILD/verilator_obj"

if [[ "$USE_MEMO" != "0" && "$USE_MEMO" != "1" ]]; then
  echo "USE_MEMO must be 0 or 1" >&2
  exit 2
fi
if [[ "$VECTOR_RESULT_MODE" != "0" && "$VECTOR_RESULT_MODE" != "1" ]]; then
  echo "VECTOR_RESULT_MODE must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$SAMPLE_ID" =~ ^[0-9]+$ ]] || (( SAMPLE_ID >= 100 )); then
  echo "SAMPLE_ID must be an integer in [0,99]" >&2
  exit 2
fi

EXPECTED_SCRIPT="scripts/local5_erep_formal_canary_expected.py"
VECTOR_SCRIPT="scripts/local5_erep_integrated_cross_head_vectors.py"
ACTUAL_SCRIPT="scripts/local5_erep_integrated_cross_head_actual.py"
MERGE_SCRIPT="scripts/local5_erep_integrated_cross_head_merge.py"
TB="tb_qfit/tb_qfit_local5_memo_multitile_cross_head.sv"
RTL=(
  rtl_hitflow/gatestack_output_tile_scheduler.sv
  rtl_local5/local5_shiftmax5_q17.sv
  rtl_qfit/qfit_tagged_compactor4.sv
  rtl_qfit/qfit_xorbank_compactor4.sv
  rtl_qfit/qfit_local5_score_leaf.sv
  rtl_qfit/qfit_retirement_scheduler.sv
  rtl_qfit/qfit_sync_1r1w_bank.sv
  rtl_qfit/qfit_relation_transpose_leaf.sv
  rtl_qfit/qfit_sync_relation_bank.sv
  rtl_qfit/qfit_exposure_relation_vault.sv
  rtl_qfit/qfit_fcsr_relation_memo_top.sv
  rtl_qfit/qfit_source_multicast_term_builder.sv
  rtl_qfit/qfit_tcfm5_acc_bank.sv
  rtl_qfit/qfit_tcfm5_projection_top.sv
  rtl_qfit/qfit_fcsr_relation_memo_projection_top.sv
  rtl_qfit/qfit_local5_memo_tagged_t450_job_engine.sv
  rtl_qfit/qfit_local5_tile.sv
  rtl_qfit/qfit_local5_projection_tile.sv
  rtl_qfit/qfit_local5_tagged_t450_job_engine.sv
  rtl_qfit/qfit_single_port_acc_memory.sv
  rtl_qfit/qfit_direct_1rw_acc_bank.sv
  rtl_qfit/qfit_local5_vector_cross_head_acc.sv
  rtl_qfit/qfit_acc32_vector_serializer.sv
  rtl_qfit/qfit_local5_cross_head_tile_executor.sv
)
ASSERTIONS=(
  verif_qfit/qfit_relation_transpose_assertions.sv
  verif_qfit/qfit_exposure_relation_vault_assertions.sv
  verif_qfit/qfit_source_multicast_assertions.sv
  verif_qfit/qfit_tcfm5_assertions.sv
  verif_qfit/qfit_tcfm5_acc_bank_assertions.sv
  verif_qfit/qfit_local5_tagged_t450_job_engine_assertions.sv
  verif_qfit/qfit_local5_memo_tagged_t450_job_engine_assertions.sv
  verif_qfit/qfit_local5_cross_head_tile_executor_assertions.sv
)
FILELIST=("$TB" "${RTL[@]}")

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  python3 --version
  iverilog -V 2>&1 | sed -n '1p'
  verilator --version
  python3 -c 'import numpy; print("numpy", numpy.__version__)'
} >"$OUT_DIR/tool_versions.txt"

python3 -m py_compile \
  "$EXPECTED_SCRIPT" "$VECTOR_SCRIPT" "$ACTUAL_SCRIPT" "$MERGE_SCRIPT" \
  scripts/generate_local5_checkpoint_score_vectors.py \
  scripts/generate_local5_masked_integer_vectors.py \
  tests/test_local5_erep_integrated_cross_head_canary.py \
  >"$OUT_DIR/py_compile.log" 2>&1
python3 -m unittest tests.test_local5_erep_integrated_cross_head_canary \
  >"$OUT_DIR/unittest.log" 2>&1

python3 "$EXPECTED_SCRIPT" \
  --profile "$PROFILE" --sample "$SAMPLE_ID" --stage 0 --block 0 \
  --output-dir "$SOFTWARE" >"$OUT_DIR/software_expected.log" 2>&1
python3 "$VECTOR_SCRIPT" \
  --profile "$PROFILE" --task-plan "$SOFTWARE/task_plan.json" \
  --output-dir "$VECTORS" >"$OUT_DIR/vector_generation.log" 2>&1

read -r STAGE_ID BLOCK_ID WINDOW_ID HEADS_ID < <(
  python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["stage"],p["block"],p["window"],p["heads"])' \
    "$SOFTWARE/task_plan.json"
)
if [[ "$HEADS_ID" != "3" ]]; then
  echo "integrated cross-head canary expects H=3" >&2
  exit 1
fi

COMMON_ARGS=(
  "+INPUTS=$VECTORS/combined_head_inputs.txt"
  "+WEIGHTS=$VECTORS/projection_weights.txt"
  "+STAGE_ID=$STAGE_ID"
  "+BLOCK_ID=$BLOCK_ID"
  "+WINDOW_ID=$WINDOW_ID"
  +NO_ACC_CHECK
  +SERVICE_SEED=17717
)
COMMON_PARAMS=(
  -Ptb_qfit_local5_memo_multitile_cross_head.USE_MEMO="$USE_MEMO"
  -Ptb_qfit_local5_memo_multitile_cross_head.USE_INPLACE=0
  -Ptb_qfit_local5_memo_multitile_cross_head.VECTOR_RESULT_MODE="$VECTOR_RESULT_MODE"
  -Ptb_qfit_local5_memo_multitile_cross_head.TRANSACTION_INDEXED_SERVICE=1
  -Ptb_qfit_local5_memo_multitile_cross_head.STAGE_ID="$STAGE_ID"
  -Ptb_qfit_local5_memo_multitile_cross_head.BLOCK_ID="$BLOCK_ID"
  -Ptb_qfit_local5_memo_multitile_cross_head.WINDOW_ID="$WINDOW_ID"
)

iverilog -g2012 "${COMMON_PARAMS[@]}" \
  -s tb_qfit_local5_memo_multitile_cross_head \
  -o "$BUILD/integrated_iv" "${RTL[@]}" "$TB" \
  >"$OUT_DIR/iverilog_compile.log" 2>&1
ICARUS_CMD="$BUILD/integrated_iv ${COMMON_ARGS[*]} +ACTUAL_ACC_FILE=$OUT_DIR/actual_icarus.memh"
ICARUS_CMD="vvp $ICARUS_CMD"
ICARUS_COMPILE_CMD="iverilog -g2012 ${COMMON_PARAMS[*]} -s tb_qfit_local5_memo_multitile_cross_head -o $BUILD/integrated_iv ${RTL[*]} $TB"
vvp "$BUILD/integrated_iv" "${COMMON_ARGS[@]}" \
  "+ACTUAL_ACC_FILE=$OUT_DIR/actual_icarus.memh" \
  >"$OUT_DIR/icarus.log" 2>&1
python3 "$ACTUAL_SCRIPT" \
  --simulator icarus --log "$OUT_DIR/icarus.log" \
  --actual "$OUT_DIR/actual_icarus.memh" \
  --vector-manifest "$VECTORS/manifest.json" \
  --task-plan "$SOFTWARE/task_plan.json" \
  --filelist "${FILELIST[@]}" --command "$ICARUS_CMD" \
  --compile-command "$ICARUS_COMPILE_CMD" \
  --executable "$BUILD/integrated_iv" \
  --tool-versions "$OUT_DIR/tool_versions.txt" \
  --output "$OUT_DIR/actual_icarus_receipt.json" \
  >"$OUT_DIR/actual_icarus_adapter.log" 2>&1

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-BLKSEQ -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC \
  --top-module tb_qfit_local5_memo_multitile_cross_head \
  --Mdir "$BUILD/verilator_obj" \
  -GUSE_MEMO="$USE_MEMO" -GUSE_INPLACE=0 \
  -GVECTOR_RESULT_MODE="$VECTOR_RESULT_MODE" \
  -GTRANSACTION_INDEXED_SERVICE=1 \
  -GSTAGE_ID="$STAGE_ID" -GBLOCK_ID="$BLOCK_ID" -GWINDOW_ID="$WINDOW_ID" \
  "${RTL[@]}" "${ASSERTIONS[@]}" "$TB" \
  >"$OUT_DIR/verilator_compile.log" 2>&1
VERILATOR_BIN="$BUILD/verilator_obj/Vtb_qfit_local5_memo_multitile_cross_head"
VERILATOR_COMPILE_CMD="verilator --binary --timing --assert -Wall -Wno-fatal -Wno-BLKSEQ -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC --top-module tb_qfit_local5_memo_multitile_cross_head --Mdir $BUILD/verilator_obj -GUSE_MEMO=$USE_MEMO -GUSE_INPLACE=0 -GVECTOR_RESULT_MODE=$VECTOR_RESULT_MODE -GTRANSACTION_INDEXED_SERVICE=1 -GSTAGE_ID=$STAGE_ID -GBLOCK_ID=$BLOCK_ID -GWINDOW_ID=$WINDOW_ID ${RTL[*]} ${ASSERTIONS[*]} $TB"
VERILATOR_CMD="$VERILATOR_BIN ${COMMON_ARGS[*]} +ACTUAL_ACC_FILE=$OUT_DIR/actual_verilator.memh"
"$VERILATOR_BIN" "${COMMON_ARGS[@]}" \
  "+ACTUAL_ACC_FILE=$OUT_DIR/actual_verilator.memh" \
  >"$OUT_DIR/verilator.log" 2>&1
python3 "$ACTUAL_SCRIPT" \
  --simulator verilator --log "$OUT_DIR/verilator.log" \
  --actual "$OUT_DIR/actual_verilator.memh" \
  --vector-manifest "$VECTORS/manifest.json" \
  --task-plan "$SOFTWARE/task_plan.json" \
  --filelist "${FILELIST[@]}" "${ASSERTIONS[@]}" --command "$VERILATOR_CMD" \
  --compile-command "$VERILATOR_COMPILE_CMD" \
  --executable "$VERILATOR_BIN" \
  --tool-versions "$OUT_DIR/tool_versions.txt" \
  --output "$OUT_DIR/actual_verilator_receipt.json" \
  >"$OUT_DIR/actual_verilator_adapter.log" 2>&1

python3 "$MERGE_SCRIPT" \
  --task-plan "$SOFTWARE/task_plan.json" \
  --expected "$SOFTWARE/software_expected.npz" \
  --expected-receipt "$SOFTWARE/software_expected_receipt.json" \
  --actual "$OUT_DIR/actual_icarus.memh" \
  --actual "$OUT_DIR/actual_verilator.memh" \
  --actual-receipt "$OUT_DIR/actual_icarus_receipt.json" \
  --actual-receipt "$OUT_DIR/actual_verilator_receipt.json" \
  --use-memo "$USE_MEMO" \
  --vector-result-mode "$VECTOR_RESULT_MODE" \
  --output "$OUT_DIR/merge_report.json" >"$OUT_DIR/merge.log" 2>&1

sha256sum \
  "$EXPECTED_SCRIPT" "$VECTOR_SCRIPT" "$ACTUAL_SCRIPT" "$MERGE_SCRIPT" \
  scripts/generate_local5_checkpoint_score_vectors.py \
  scripts/generate_local5_masked_integer_vectors.py \
  tests/test_local5_erep_integrated_cross_head_canary.py "$TB" \
  "${RTL[@]}" "${ASSERTIONS[@]}" \
  sim_qfit/run_local5_erep_integrated_cross_head_canary.sh \
  >"$OUT_DIR/source_sha256.txt"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 \
  --numeric-owner -cf "$OUT_DIR/source_bundle.tar" \
  "$EXPECTED_SCRIPT" "$VECTOR_SCRIPT" "$ACTUAL_SCRIPT" "$MERGE_SCRIPT" \
  scripts/generate_local5_checkpoint_score_vectors.py \
  scripts/generate_local5_masked_integer_vectors.py \
  tests/test_local5_erep_integrated_cross_head_canary.py "$TB" \
  "${RTL[@]}" "${ASSERTIONS[@]}" \
  sim_qfit/run_local5_erep_integrated_cross_head_canary.sh
sha256sum \
  "$OUT_DIR"/{tool_versions.txt,py_compile.log,unittest.log,software_expected.log,vector_generation.log,iverilog_compile.log,icarus.log,actual_icarus_adapter.log,actual_icarus_receipt.json,verilator_compile.log,verilator.log,actual_verilator_adapter.log,actual_verilator_receipt.json,merge.log,merge_report.json,actual_icarus.memh,actual_verilator.memh,source_sha256.txt,source_bundle.tar} \
  "$BUILD/integrated_iv" "$VERILATOR_BIN" \
  "$SOFTWARE"/{task_plan.json,software_expected.npz,software_expected_receipt.json} \
  "$VECTORS"/{manifest.json,head0_inputs.txt,head1_inputs.txt,head2_inputs.txt,combined_head_inputs.txt,projection_weights.txt} \
  >"$OUT_DIR/result_sha256.txt"

python3 - "$OUT_DIR/complete.json" "$OUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[2]).resolve()
report = json.loads((out / "merge_report.json").read_text(encoding="utf-8"))
value = {
    "schema": "local5_erep_integrated_cross_head_complete_v1",
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

echo "PASS Local5 integrated cross-head formal canary sample=$SAMPLE_ID use_memo=$USE_MEMO vector_result_mode=$VECTOR_RESULT_MODE formal_g0=DENY output=$OUT_DIR"
