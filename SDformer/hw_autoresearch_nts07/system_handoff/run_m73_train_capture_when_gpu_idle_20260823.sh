#!/usr/bin/env bash
set -euo pipefail

repo=/root/private_data/work/sdformer_codex/SDformer
python=/opt/conda/envs/sdformerflow/bin/python
config="$repo/neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml"
checkpoint="$repo/hw_autoresearch_nts07/system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth"
data_root="$repo/data/Datasets/DSEC/saved_flow_data"
tracer="$repo/hw_autoresearch_nts07/system_simulator/scripts/trace_m73_train_calibration_bottleneck_sources.py"
m40="$repo/hw_autoresearch_nts07/system_simulator/scripts/trace_m40_bottleneck_packed_sources.py"
profile="$repo/neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py"
train_list="$data_root/sequence_lists/train_split_seq.csv"
valid_list="$data_root/sequence_lists/valid_split_seq.csv"
output="$repo/hw_autoresearch_nts07/results/m73_h67_ep35_train_calibration_sources_s32_r1_20260823"
receipt="$repo/hw_autoresearch_nts07/results/m73_h67_ep35_train_calibration_sources_s32_r1_20260823.queue_receipt"
output_stage="${output}.partial.$$.${RANDOM}"
stage=preflight
success=0

write_failure_receipt() {
  local rc="$1"
  local failed="${receipt}.FAILED.$(date -u +%Y%m%dT%H%M%SZ).$$"
  local temporary="${failed}.tmp"
  {
    echo "status=FAILED_M73_DO_NOT_USE"
    echo "stage=$stage"
    echo "exit_code=$rc"
    echo "failure_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "retained_partial=$output_stage"
  } > "$temporary"
  mv "$temporary" "$failed"
}
on_exit() {
  local rc="$?"
  if [[ "$success" -ne 1 ]]; then
    write_failure_receipt "$rc"
  fi
}
trap on_exit EXIT

lock_file="$repo/hw_autoresearch_nts07/results/.m73_train_capture.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "REFUSE: another M73 queue/capture owns $lock_file" >&2
  exit 4
fi

check_sha() {
  local path="$1"
  local expected="$2"
  local observed
  [[ -f "$path" ]] || { echo "M73 missing pinned input: $path" >&2; return 1; }
  observed="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    echo "M73 SHA drift path=$path expected=$expected observed=$observed" >&2
    return 1
  }
}
check_all_pins() {
  check_sha "$tracer" 9d79f7198ba1ac221f6e58428480c9d59e3deafff0775d2ae3aaa0da75f693bb
  check_sha "$m40" b02ac10fb95e68fa2871b74330d6f39d7d3d8cbfa6440990d43ec832e943bf19
  check_sha "$profile" 04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684
  check_sha "$config" 86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc
  check_sha "$checkpoint" 4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158
  check_sha "$train_list" 919c79c61535eb499364ffe28fad3000441e25d1bddbf4fa9a0c27a78d4fdc10
  check_sha "$valid_list" 7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0
}

if [[ -e "$output" || -e "$receipt" || -e "$output_stage" ]]; then
  echo "REFUSE: M73 final/stage output already exists" >&2
  exit 2
fi
check_all_pins

stage=wait_for_gpu
echo "M73_QUEUE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "M73 uses the exact M87 base forward config and never kills or preempts a GPU process."
idle=0
while (( idle < 4 )); do
  if ! gpu_rows="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"; then
    idle=0
    echo "M73_IDLE_PROBE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ) active_compute_pids=UNKNOWN_NVIDIA_SMI_FAILURE consecutive_idle=0"
    sleep 30
    continue
  fi
  active="$(printf '%s\n' "$gpu_rows" | sed '/^[[:space:]]*$/d' | wc -l)"
  if [[ "$active" == "0" ]]; then idle=$((idle + 1)); else idle=0; fi
  echo "M73_IDLE_PROBE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ) active_compute_pids=$active consecutive_idle=$idle"
  if (( idle < 4 )); then sleep 30; fi
done

stage=post_wait_sha_gate
check_all_pins
stage=capture
cd "$repo"
echo "M73_CAPTURE_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$python" "$tracer" \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --data-root "$data_root" \
  --output-dir "$output_stage"
stage=post_capture_sha_gate
check_all_pins
manifest_stage="$output_stage/m73_train_calibration_source_manifest.json"
test -f "$manifest_stage"

stage=atomic_publish
mv "$output_stage" "$output"
manifest="$output/m73_train_calibration_source_manifest.json"
manifest_sha="$(sha256sum "$manifest" | awk '{print $1}')"
receipt_tmp="${receipt}.tmp.$$"
{
  echo "status=PASS_M73_CAPTURE"
  echo "completion_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "manifest_sha256=$manifest_sha"
  echo "forward_base_config_sha256=86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc"
  echo "tracer_sha256=9d79f7198ba1ac221f6e58428480c9d59e3deafff0775d2ae3aaa0da75f693bb"
  echo "checkpoint_sha256=4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158"
} > "$receipt_tmp"
mv "$receipt_tmp" "$receipt"
success=1
cat "$receipt"
