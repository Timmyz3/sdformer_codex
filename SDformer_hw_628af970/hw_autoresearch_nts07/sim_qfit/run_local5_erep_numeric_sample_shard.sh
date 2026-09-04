#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAMPLE="${SAMPLE:-0}"
PROFILE="${PROFILE:-results/local5_fullres_bb1e4_joint_heads_profile100_20260809}"
OUT_DIR="${OUT_DIR:-results/local5_erep_numeric_sample${SAMPLE}_shard_v3_20260811}"
RELEASE_DIR="${RELEASE_DIR:-results/local5_erep_numeric_rtl_release_v3_20260811}"
RELEASE_ONLY="${RELEASE_ONLY:-0}"
RELEASE_SERVICE_MODE="${RELEASE_SERVICE_MODE:-transaction}"
RELEASE_WEIGHT_HOLD_CYCLES="${RELEASE_WEIGHT_HOLD_CYCLES:-0}"
WINDOW_LIMIT="${WINDOW_LIMIT:-0}"
if ! [[ "$SAMPLE" =~ ^[0-9]+$ ]] || (( SAMPLE < 0 || SAMPLE >= 100 )); then
  echo "SAMPLE must be in 0..99" >&2
  exit 2
fi
if [[ "$WINDOW_LIMIT" != "0" && "$WINDOW_LIMIT" != "1" ]]; then
  echo "WINDOW_LIMIT must be 0 (full shard) or 1 (single-window canary)" >&2
  exit 2
fi
if [[ "$RELEASE_SERVICE_MODE" != "transaction" \
      && "$RELEASE_SERVICE_MODE" != "identity" ]]; then
  echo "RELEASE_SERVICE_MODE must be transaction or identity" >&2
  exit 2
fi
if ! [[ "$RELEASE_WEIGHT_HOLD_CYCLES" =~ ^[0-7]$ ]]; then
  echo "RELEASE_WEIGHT_HOLD_CYCLES must be in 0..7" >&2
  exit 2
fi
PROFILE="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$PROFILE")"
OUT_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$OUT_DIR")"
RELEASE_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$RELEASE_DIR")"

EXPECTED_SCRIPT="scripts/local5_erep_numeric_window_expected.py"
VECTOR_SCRIPT="scripts/local5_erep_integrated_cross_head_vectors.py"
ACTUAL_SCRIPT="scripts/local5_erep_integrated_cross_head_actual.py"
MERGE_SCRIPT="scripts/local5_erep_numeric_shard_merge.py"
RELEASE_SCRIPT="scripts/local5_erep_numeric_release.py"
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
SOURCE_FILES=(
  "$EXPECTED_SCRIPT" "$VECTOR_SCRIPT" "$ACTUAL_SCRIPT" "$MERGE_SCRIPT"
  "$RELEASE_SCRIPT"
  scripts/local5_erep_formal_canary_expected.py
  scripts/local5_erep_archive_replay_v4.py
  scripts/local5_erep_ledger_replay_v4.py
  scripts/local5_erep_capacity_baselines_v4.py
  scripts/local5_erep_command_schedule_v4.py
  scripts/local5_erep_identity_service_v4.py
  scripts/generate_local5_identity_service_tables_v4.py
  scripts/verify_local5_identity_service_tables_v4.py
  scripts/verify_local5_identity_service_rtl_trace_v1.py
  scripts/verify_local5_identity_service_rtl_trace_v2.py
  scripts/freeze_local5_identity_state_reference_v1.py
  scripts/generate_local5_h3_phase_template_patch_v1.py
  scripts/verify_local5_h3_phase_template_patch_v1.py
  contracts/local5_identity_service_h3_state_reference_v1.json
  scripts/local5_erep_integrated_cross_head_merge.py
  scripts/generate_local5_checkpoint_score_vectors.py
  scripts/generate_local5_masked_integer_vectors.py
  "$TB" "${RTL[@]}" "${ASSERTIONS[@]}"
  sim_qfit/run_local5_erep_numeric_sample_shard.sh
  sim_qfit/run_local5_h3_phase_template_patch_canary_v1.sh
)

write_tool_body() {
  {
  python3 --version
  python3 -c 'import numpy; print("numpy=" + numpy.__version__)'
  verilator --version
  c++ --version | sed -n '1p'
  }
}

write_tool_bindings() {
  local output="$1"
  local names=(bash c++ flock g++ make python3 sha256sum tar time verilator verilator_bin)
  local args=()
  local name path version
  for name in "${names[@]}"; do
    if [[ "$name" == "time" ]]; then
      path="/usr/bin/time"
    else
      path="$(command -v "$name")"
    fi
    path="$(readlink -f "$path")"
    version="$("$path" --version 2>&1 | sed -n '1p')"
    args+=("$name" "$path" "$version")
  done
  python3 - "$output" "${args[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
values = sys.argv[2:]
if len(values) % 3:
    raise SystemExit("tool binding 参数不是三元组")
rows = []
for index in range(0, len(values), 3):
    name, path_text, version = values[index:index + 3]
    path = Path(path_text)
    rows.append({
        "name": name,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "version": version,
    })
output.write_text(json.dumps({
    "schema": "local5_erep_numeric_tool_bindings_v1",
    "tools": rows,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

write_argv_json() {
  local output="$1"
  shift
  python3 - "$output" "$@" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(sys.argv[2:], ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

if [[ -f "$RELEASE_DIR/release_complete.json" ]]; then
  python3 "$RELEASE_DIR/source/$RELEASE_SCRIPT" verify --release-dir "$RELEASE_DIR"
else
  mkdir -p "$(dirname "$RELEASE_DIR")"
  exec 9>"${RELEASE_DIR}.lock"
  if ! flock -n 9; then
    echo "release creation is locked by another process" >&2
    exit 1
  fi
  if [[ -f "$RELEASE_DIR/release_complete.json" ]]; then
    python3 "$RELEASE_DIR/source/$RELEASE_SCRIPT" verify --release-dir "$RELEASE_DIR"
  else
    if [[ -e "$RELEASE_DIR" ]]; then
      echo "incomplete release exists; use a new RELEASE_DIR" >&2
      exit 1
    fi
    staging="${RELEASE_DIR}.staging.$$"
    if [[ -e "$staging" ]]; then
      echo "release staging path already exists" >&2
      exit 1
    fi
    trap 'rm -rf "$staging"' EXIT
    mkdir -p "$staging/build" "$staging/source"
    {
      date -u +%Y-%m-%dT%H:%M:%SZ
      write_tool_body
    } >"$staging/tool_versions.txt"
    write_tool_bindings "$staging/tool_bindings.json"
    sha256sum "${SOURCE_FILES[@]}" >"$staging/source_sha256.txt"
    tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 \
      --numeric-owner -cf "$staging/source_bundle.tar" "${SOURCE_FILES[@]}"
    tar -xf "$staging/source_bundle.tar" -C "$staging/source"
    verilator_path="$(python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); print(next(v["path"] for v in x["tools"] if v["name"] == "verilator"))' "$staging/tool_bindings.json")"

    release_rtl=()
    for source in "${RTL[@]}"; do release_rtl+=("source/$source"); done
    release_assertions=()
    for source in "${ASSERTIONS[@]}"; do release_assertions+=("source/$source"); done
    for heads in 3 6 12 24; do
      build="$staging/build/h${heads}"
      mkdir -p "$build"
      service_compile_args=(-GTRANSACTION_INDEXED_SERVICE=1)
      if [[ "$RELEASE_SERVICE_MODE" == "identity" ]]; then
        service_compile_args=(
          -GTRANSACTION_INDEXED_SERVICE=0
          -GIDENTITY_DERIVED_SERVICE=1
        )
      fi
      compile_argv=(
        "$verilator_path" --binary --timing --assert -Wall -Wno-fatal
        -Wno-BLKSEQ -Wno-PINCONNECTEMPTY -Wno-UNUSEDSIGNAL
        -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC
        --top-module tb_qfit_local5_memo_multitile_cross_head
        --Mdir "build/h${heads}/obj"
        -GUSE_MEMO=0 -GUSE_INPLACE=0 "${service_compile_args[@]}"
        -GFORCE_WEIGHT_RESPONSE_HOLD_CYCLES="$RELEASE_WEIGHT_HOLD_CYCLES"
        -GHEADS="$heads" -GOUTPUT_TILES="$heads"
        -GSTAGE_ID=0 -GBLOCK_ID=0 -GWINDOW_ID=0
        -GTIMEOUT_CYCLES=100000000
        "${release_rtl[@]}" "${release_assertions[@]}" "source/$TB"
      )
      write_argv_json "$build/compile_argv.json" "${compile_argv[@]}"
      (cd "$staging" && "${compile_argv[@]}") >"$build/compile.log" 2>&1
    done
    chmod -R a-w "$staging/source"
    python3 "$RELEASE_SCRIPT" seal --release-dir "$staging"
    mv "$staging" "$RELEASE_DIR"
    trap - EXIT
  fi
fi

RELEASE_MANIFEST="$RELEASE_DIR/release_manifest.json"
if [[ "$RELEASE_ONLY" == "1" ]]; then
  printf 'PASS Local5 sealed numeric RTL release formal_g0=DENY output=%s\n' \
    "$RELEASE_DIR"
  exit 0
fi

EXPECTED_RUNTIME="$RELEASE_DIR/source/$EXPECTED_SCRIPT"
VECTOR_RUNTIME="$RELEASE_DIR/source/$VECTOR_SCRIPT"
ACTUAL_RUNTIME="$RELEASE_DIR/source/$ACTUAL_SCRIPT"
MERGE_RUNTIME="$RELEASE_DIR/source/$MERGE_SCRIPT"

mkdir -p "$OUT_DIR/windows" "$OUT_DIR/shard"

python3 - \
  "$EXPECTED_RUNTIME" "$VECTOR_RUNTIME" "$ACTUAL_RUNTIME" "$MERGE_RUNTIME" \
  "$RELEASE_DIR/source/scripts/local5_erep_formal_canary_expected.py" \
  "$RELEASE_DIR/source/scripts/local5_erep_archive_replay_v4.py" \
  "$RELEASE_DIR/source/scripts/local5_erep_integrated_cross_head_merge.py" \
  >"$OUT_DIR/py_compile.log" 2>&1 <<'PY'
import sys
from pathlib import Path

for name in sys.argv[1:]:
    path = Path(name)
    compile(path.read_bytes(), str(path), "exec")
PY

python3 - "$OUT_DIR" "$RELEASE_MANIFEST" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2]).resolve()
complete = manifest.parent / "release_complete.json"
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
value = {
    "schema": "local5_erep_numeric_sample_release_binding_v1",
    "status": "PASS_RELEASE_BOUND_NOT_G0",
    "formal_g0": "DENY",
    "release_manifest": str(manifest),
    "release_manifest_sha256": digest(manifest),
    "release_complete": str(complete),
    "release_complete_sha256": digest(complete),
}
path = out / "release_binding.json"
if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != value:
        raise SystemExit("sample release binding changed; use a new OUT_DIR")
else:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
PY

BLOCKS=(
  "0 0 3" "0 1 3"
  "1 0 6" "1 1 6"
  "2 0 12" "2 1 12" "2 2 12" "2 3 12" "2 4 12" "2 5 12"
  "3 0 24" "3 1 24"
)

finalize_single_window_canary() {
  local stage="$1" block="$2" heads="$3"
  python3 "$MERGE_RUNTIME" --sample "$SAMPLE" \
    --window-root "$OUT_DIR/windows" --output-dir "$OUT_DIR/window_miter" \
    --release-manifest "$RELEASE_MANIFEST" \
    --single-stage "$stage" --single-block "$block" --single-heads "$heads" \
    >"$OUT_DIR/window_miter.log" 2>&1
  python3 "$RELEASE_DIR/source/$RELEASE_SCRIPT" verify \
    --release-dir "$RELEASE_DIR" >"$OUT_DIR/release_verify_after_canary.log" 2>&1
  python3 - "$OUT_DIR" "$SAMPLE" "$RELEASE_MANIFEST" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1]).resolve()
sample = int(sys.argv[2])
release = Path(sys.argv[3]).resolve()
report_path = out / "window_miter/numeric_window_miter_report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("status") != "PASS_NUMERIC_WINDOW_MITER_NOT_G0":
    raise SystemExit("single-window miter did not pass")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
value = {
    "schema": "local5_erep_numeric_window_canary_complete_v1",
    "status": "PASS_NUMERIC_WINDOW_CANARY_NOT_G0",
    "evidence": "[rtl]+[软件整数金参考]+[rtl-build-provenance]",
    "formal_g0": "DENY",
    "sample": sample,
    "identity": report["identity"],
    "mismatch_count": report["mismatch_count"],
    "scalar_count": report["scalar_count"],
    "release_manifest_sha256": digest(release),
    "window_miter_report_sha256": digest(report_path),
    "release_verify_after_canary_sha256": digest(
        out / "release_verify_after_canary.log"
    ),
    "boundary": "单个真实 H3 窗口；不是 12-window shard 或 formal G0",
}
temporary = out / "canary_complete.json.tmp"
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, out / "canary_complete.json")
PY
  sha256sum \
    "$OUT_DIR/window_miter/acc32_window_miter.npz" \
    "$OUT_DIR/window_miter/numeric_window_miter_report.json" \
    "$OUT_DIR/release_verify_after_canary.log" \
    "$OUT_DIR/canary_complete.json" >"$OUT_DIR/canary_receipt_sha256.txt"
  printf 'PASS Local5 numeric single-window canary sample=%s formal_g0=DENY output=%s\n' \
    "$SAMPLE" "$OUT_DIR"
  exit 0
}

completed_windows=0
for spec in "${BLOCKS[@]}"; do
  read -r stage block heads <<<"$spec"
  window_dir="$OUT_DIR/windows/s${stage}_b${block}"
  if [[ -f "$window_dir/window_complete.json" ]]; then
    printf 'RESUME sample=%s stage=%s block=%s heads=%s\n' \
      "$SAMPLE" "$stage" "$block" "$heads"
    if [[ "$WINDOW_LIMIT" == "1" ]]; then
      finalize_single_window_canary "$stage" "$block" "$heads"
    fi
    continue
  fi
  rm -rf "$window_dir"
  mkdir -p "$window_dir/software_expected" "$window_dir/vectors"

  python3 "$EXPECTED_RUNTIME" \
    --profile "$PROFILE" --sample "$SAMPLE" --stage "$stage" --block "$block" \
    --output-dir "$window_dir/software_expected" \
    >"$window_dir/software_expected.log" 2>&1
  python3 "$VECTOR_RUNTIME" \
    --profile "$PROFILE" \
    --task-plan "$window_dir/software_expected/task_plan.json" \
    --output-dir "$window_dir/vectors" \
    >"$window_dir/vector_generation.log" 2>&1
  read -r plan_stage plan_block plan_window plan_heads < <(
    python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["stage"],p["block"],p["window"],p["heads"])' \
      "$window_dir/software_expected/task_plan.json"
  )
  if [[ "$plan_stage $plan_block $plan_heads" != "$stage $block $heads" ]]; then
    echo "task plan coordinate mismatch" >&2
    exit 1
  fi

  bin="$RELEASE_DIR/build/h${heads}/obj/Vtb_qfit_local5_memo_multitile_cross_head"
  compile_argv_file="$RELEASE_DIR/build/h${heads}/compile_argv.json"
  seed=$((17717 + SAMPLE * 37 + stage * 7 + block))
  run_args=(
    "+INPUTS=$window_dir/vectors/combined_head_inputs.txt"
    "+WEIGHTS=$window_dir/vectors/projection_weights.txt"
    "+STAGE_ID=$stage" "+BLOCK_ID=$block" "+WINDOW_ID=$plan_window"
    +NO_ACC_CHECK "+SERVICE_SEED=$seed"
    "+ACTUAL_ACC_FILE=$window_dir/actual.memh"
  )
  run_argv=("$bin" "${run_args[@]}")
  write_argv_json "$window_dir/run_argv.json" "${run_argv[@]}"
  /usr/bin/time -f 'wall_seconds=%e\nmax_rss_kb=%M' \
    -o "$window_dir/verilator_time.txt" \
    "${run_argv[@]}" >"$window_dir/verilator.log" 2>&1

  python3 "$ACTUAL_RUNTIME" \
    --simulator verilator --log "$window_dir/verilator.log" \
    --actual "$window_dir/actual.memh" \
    --vector-manifest "$window_dir/vectors/manifest.json" \
    --task-plan "$window_dir/software_expected/task_plan.json" \
    --filelist "$TB" "${RTL[@]}" "${ASSERTIONS[@]}" \
    --run-argv "$window_dir/run_argv.json" \
    --compile-argv "$compile_argv_file" \
    --release-manifest "$RELEASE_MANIFEST" \
    --executable "$bin" --tool-versions "$RELEASE_DIR/tool_versions.txt" \
    --output "$window_dir/actual_receipt.json" \
    >"$window_dir/actual_adapter.log" 2>&1

  python3 - "$window_dir" "$RELEASE_MANIFEST" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
plan = json.loads((root / "software_expected/task_plan.json").read_text())
artifacts = {
    "task_plan": root / "software_expected/task_plan.json",
    "software_expected": root / "software_expected/software_expected.npz",
    "software_expected_receipt": root / "software_expected/software_expected_receipt.json",
    "vector_manifest": root / "vectors/manifest.json",
    "actual": root / "actual.memh",
    "actual_receipt": root / "actual_receipt.json",
    "raw_log": root / "verilator.log",
    "run_argv": root / "run_argv.json",
    "release_manifest": Path(sys.argv[2]).resolve(),
}
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
value = {
    "schema": "local5_erep_numeric_window_complete_v1",
    "status": "SEALED_READY_FOR_MITER_NOT_G0",
    "evidence": "[rtl-provenance]+[待miter]",
    "formal_g0": "DENY",
    "identity": {key: int(plan[key]) for key in ("sample", "stage", "block", "window", "heads")},
    "artifact_sha256": {name: digest(path) for name, path in artifacts.items()},
}
temporary = root / "window_complete.json.tmp"
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, root / "window_complete.json")
PY
  printf 'PASS_WINDOW sample=%s stage=%s block=%s heads=%s window=%s\n' \
    "$SAMPLE" "$stage" "$block" "$heads" "$plan_window"
  completed_windows=$((completed_windows + 1))
  if [[ "$WINDOW_LIMIT" == "1" && "$completed_windows" == "1" ]]; then
    finalize_single_window_canary "$stage" "$block" "$heads"
  fi
done

python3 "$MERGE_RUNTIME" --sample "$SAMPLE" \
  --window-root "$OUT_DIR/windows" --output-dir "$OUT_DIR/shard" \
  --release-manifest "$RELEASE_MANIFEST" \
  >"$OUT_DIR/shard_merge.log" 2>&1

find "$OUT_DIR/windows" -name 'window_complete.json' -print0 | sort -z | xargs -0 sha256sum \
  >"$OUT_DIR/window_receipt_sha256.txt"
sha256sum \
  "$OUT_DIR"/{py_compile.log,release_binding.json,window_receipt_sha256.txt,shard_merge.log} \
  "$OUT_DIR/shard"/{acc32_miter_shard.npz,numeric_shard_report.json} \
  >"$OUT_DIR/result_sha256.txt"

python3 - "$OUT_DIR" "$SAMPLE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1]).resolve()
sample = int(sys.argv[2])
report = json.loads((out / "shard/numeric_shard_report.json").read_text())
if report.get("status") != "PASS_NUMERIC_SAMPLE_SHARD_NOT_G0" or report.get("sample") != sample:
    raise SystemExit("numeric shard report is not admissible")
value = {
    "schema": "local5_erep_numeric_sample_shard_complete_v1",
    "status": "PASS_NUMERIC_SAMPLE_SHARD_NOT_G0",
    "evidence": "[rtl]+[软件整数金参考]",
    "formal_g0": "DENY",
    "sample": sample,
    "output_directory": str(out),
    "result_sha256_file_sha256": hashlib.sha256((out / "result_sha256.txt").read_bytes()).hexdigest(),
    "release_binding_sha256": hashlib.sha256((out / "release_binding.json").read_bytes()).hexdigest(),
    "boundary": "pre-bias/pre-BN/pre-requant/pre-residual Acc32；无phase ledger",
}
temporary = out / "complete.json.tmp"
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, out / "complete.json")
PY
sha256sum "$OUT_DIR/result_sha256.txt" "$OUT_DIR/complete.json" \
  >"$OUT_DIR/receipt_sha256.txt"

printf 'PASS Local5 numeric sample shard sample=%s formal_g0=DENY output=%s\n' \
  "$SAMPLE" "$OUT_DIR"
