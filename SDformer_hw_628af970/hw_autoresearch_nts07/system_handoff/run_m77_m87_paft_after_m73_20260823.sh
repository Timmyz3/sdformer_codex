#!/usr/bin/env bash
set -euo pipefail

repo=/root/private_data/work/sdformer_codex/SDformer
python=/opt/conda/envs/sdformerflow/bin/python
m73_dir="$repo/hw_autoresearch_nts07/results/m73_h67_ep35_train_calibration_sources_s32_r1_20260823"
m73_manifest="$m73_dir/m73_train_calibration_source_manifest.json"
m73_receipt="$repo/hw_autoresearch_nts07/results/m73_h67_ep35_train_calibration_sources_s32_r1_20260823.queue_receipt"
m77_dir="$repo/hw_autoresearch_nts07/results/m77_h67_trainonly_phi_kmeans_paft_catalog_r1_20260823"
catalog="$m77_dir/m77_h67_k16_q16_trainonly_paft_catalog.json"
contract="$m77_dir/m77_pattern_paft_catalog_admission_contract.json"
builder="$repo/hw_autoresearch_nts07/system_simulator/scripts/build_m77_train_only_phi_kmeans_paft_catalog.py"
materializer="$repo/neuron_experiments/H9_bipolar_self_attention/entrypoints/materialize_m87_h67_trainonly_paft_configs.py"
source_config="$repo/neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml"
pattern_paft="$repo/neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/pattern_paft.py"
train_py="$repo/neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py"
tracer="$repo/hw_autoresearch_nts07/system_simulator/scripts/trace_m73_train_calibration_bottleneck_sources.py"
checkpoint="$repo/hw_autoresearch_nts07/system_handoff/received/h67_ep35_system_trace_handoff_20260821/h67_ep35_system_trace_handoff_20260821/checkpoint/checkpoint_epoch35.pth"
config_bundle="$repo/hw_autoresearch_nts07/results/m87_h67_trainonly_paft_config_bundle_r2_20260823"
full_config="$config_bundle/paft_full5.yml"
smoke_config="$config_bundle/paft_smoke1.yml"
control_config="$config_bundle/no_paft_control_full5.yml"
config_receipt="$config_bundle/CONFIG_BUNDLE_RECEIPT.txt"
run_root="$repo/neuron_experiments/H9_bipolar_self_attention/results/m87_h67_trainonly_paft_paired_20260823"
smoke_run="$run_root/smoke1"
control_run="$run_root/no_paft_control_full5"
full_run="$run_root/paft_full5"
chain_receipt="$run_root/M87_CHAIN_COMPLETE.txt"
stage=preflight
stage_output=none
success=0

write_failure_receipt() {
  local rc="$1"
  mkdir -p "$run_root"
  local failed="$run_root/M87_CHAIN_FAILED_$(date -u +%Y%m%dT%H%M%SZ)_$$.txt"
  local temporary="${failed}.tmp"
  {
    echo "status=FAILED_M87_CHAIN_DO_NOT_CITE"
    echo "stage=$stage"
    echo "exit_code=$rc"
    echo "failure_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "retained_partial=$stage_output"
    echo "valid825_accuracy=false"
    echo "cycle_speedup=false"
    echo "system_speedup=false"
  } > "$temporary"
  mv "$temporary" "$failed"
}
on_exit() {
  local rc="$?"
  if [[ "$success" -ne 1 ]]; then write_failure_receipt "$rc"; fi
}
trap on_exit EXIT

lock_file="$repo/hw_autoresearch_nts07/results/.m77_m87_paired_chain.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "REFUSE: another M77/M87 successor owns $lock_file" >&2
  exit 4
fi

check_sha() {
  local path="$1"
  local expected="$2"
  local observed
  [[ -f "$path" ]] || { echo "M87 missing pinned input: $path" >&2; return 1; }
  observed="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    echo "M87 SHA drift path=$path expected=$expected observed=$observed" >&2
    return 1
  }
}
check_static_pins() {
  check_sha "$builder" c760e21eac16c4e7d5112b1335c0b121762f47175f48b92e9393391b1b33e6c6
  check_sha "$materializer" d6f80180de911edf0a13a55f2ca2a96b474956d15c6441fd50168cf5eb71375f
  check_sha "$source_config" 86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc
  check_sha "$pattern_paft" 47e6d80fa5fd50604f0d9adce1fb7ac34a741da492ac19f2ef945cfba46c7bd2
  check_sha "$train_py" 49c77538f2de2c54b709b05ae246da4cf7f36a147da990a03acb9e94a917446b
  check_sha "$tracer" 9d79f7198ba1ac221f6e58428480c9d59e3deafff0775d2ae3aaa0da75f693bb
  check_sha "$checkpoint" 4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158
}
check_config_bundle() {
  [[ -f "$config_receipt" ]] || { echo "M87 missing config receipt" >&2; return 1; }
  local full_expected smoke_expected control_expected catalog_expected contract_expected m73_bundle_expected
  full_expected="$(awk -F= '$1=="paft_full5_sha256" {print $2}' "$config_receipt")"
  smoke_expected="$(awk -F= '$1=="paft_smoke1_sha256" {print $2}' "$config_receipt")"
  control_expected="$(awk -F= '$1=="no_paft_control_full5_sha256" {print $2}' "$config_receipt")"
  catalog_expected="$(awk -F= '$1=="catalog_sha256" {print $2}' "$config_receipt")"
  contract_expected="$(awk -F= '$1=="contract_sha256" {print $2}' "$config_receipt")"
  m73_bundle_expected="$(awk -F= '$1=="m73_manifest_sha256" {print $2}' "$config_receipt")"
  [[ ${#full_expected} -eq 64 && ${#smoke_expected} -eq 64 && ${#control_expected} -eq 64 \
      && ${#catalog_expected} -eq 64 && ${#contract_expected} -eq 64 \
      && ${#m73_bundle_expected} -eq 64 ]]
  check_sha "$full_config" "$full_expected"
  check_sha "$smoke_config" "$smoke_expected"
  check_sha "$control_config" "$control_expected"
  check_sha "$catalog" "$catalog_expected"
  check_sha "$contract" "$contract_expected"
  check_sha "$m73_manifest" "$m73_bundle_expected"
  grep -qx 'forward_base_config_sha256=86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc' "$config_receipt"
}
wait_for_idle() {
  local idle=0 active
  while (( idle < 4 )); do
    if ! gpu_rows="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"; then
      idle=0
      echo "M87_IDLE_PROBE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ) active_compute_pids=UNKNOWN_NVIDIA_SMI_FAILURE consecutive_idle=0"
      sleep 30
      continue
    fi
    active="$(printf '%s\n' "$gpu_rows" | sed '/^[[:space:]]*$/d' | wc -l)"
    if [[ "$active" == "0" ]]; then idle=$((idle + 1)); else idle=0; fi
    echo "M87_IDLE_PROBE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ) active_compute_pids=$active consecutive_idle=$idle"
    if (( idle < 4 )); then sleep 30; fi
  done
}

if [[ -f "$chain_receipt" ]]; then
  grep -qx 'status=PASS_M87_H67_TRAINONLY_PAFT_PAIRED_FULL5' "$chain_receipt"
  check_static_pins
  check_config_bundle
  m73_chain_expected="$(awk -F= '$1=="m73_manifest_sha256" {print $2}' "$chain_receipt")"
  catalog_chain_expected="$(awk -F= '$1=="catalog_sha256" {print $2}' "$chain_receipt")"
  contract_chain_expected="$(awk -F= '$1=="contract_sha256" {print $2}' "$chain_receipt")"
  control_expected="$(awk -F= '$1=="control_checkpoint_epoch4_sha256" {print $2}' "$chain_receipt")"
  paft_expected="$(awk -F= '$1=="paft_checkpoint_epoch4_sha256" {print $2}' "$chain_receipt")"
  [[ ${#m73_chain_expected} -eq 64 && ${#catalog_chain_expected} -eq 64 \
      && ${#contract_chain_expected} -eq 64 && ${#control_expected} -eq 64 \
      && ${#paft_expected} -eq 64 ]]
  check_sha "$m73_manifest" "$m73_chain_expected"
  check_sha "$catalog" "$catalog_chain_expected"
  check_sha "$contract" "$contract_chain_expected"
  check_sha "$control_run/checkpoint_epoch4.pth" "$control_expected"
  check_sha "$full_run/checkpoint_epoch4.pth" "$paft_expected"
  success=1
  cat "$chain_receipt"
  exit 0
fi
check_static_pins

stage=wait_for_m73
echo "M87_SUCCESSOR_WAIT_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
while [[ ! -f "$m73_receipt" || ! -f "$m73_manifest" ]]; do sleep 30; done
stage=admit_m73
check_static_pins
grep -qx 'status=PASS_M73_CAPTURE' "$m73_receipt"
grep -qx 'forward_base_config_sha256=86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc' "$m73_receipt"
m73_expected="$(awk -F= '$1=="manifest_sha256" {print $2}' "$m73_receipt")"
[[ ${#m73_expected} -eq 64 ]]
check_sha "$m73_manifest" "$m73_expected"
echo "M87_M73_ADMITTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ ! -d "$m77_dir" ]]; then
  stage=build_m77
  m77_stage="${m77_dir}.partial.$$.${RANDOM}"
  stage_output="$m77_stage"
  mkdir "$m77_stage"
  "$python" "$builder" \
    --train-trace-manifest "$m73_manifest" \
    --output-catalog "$m77_stage/m77_h67_k16_q16_trainonly_paft_catalog.json" \
    --output-admission-contract "$m77_stage/m77_pattern_paft_catalog_admission_contract.json"
  test -s "$m77_stage/m77_h67_k16_q16_trainonly_paft_catalog.json"
  test -s "$m77_stage/m77_pattern_paft_catalog_admission_contract.json"
  check_static_pins
  mv "$m77_stage" "$m77_dir"
fi
test -s "$catalog"
test -s "$contract"

if [[ ! -d "$config_bundle" ]]; then
  stage=materialize_paired_configs
  config_stage="${config_bundle}.partial.$$.${RANDOM}"
  stage_output="$config_stage"
  mkdir "$config_stage"
  "$python" "$materializer" \
    --catalog "$catalog" \
    --admission-contract "$contract" \
    --train-trace-manifest "$m73_manifest" \
    --full-output "$config_stage/paft_full5.yml" \
    --smoke-output "$config_stage/paft_smoke1.yml" \
    --control-output "$config_stage/no_paft_control_full5.yml"
  {
    echo "status=PASS_M87_CONFIG_BUNDLE"
    echo "forward_base_config_sha256=86db3960c7d12ce5c7365e82e24b1f3aef6961b79d12317da32fc41b15d1cbcc"
    echo "paft_full5_sha256=$(sha256sum "$config_stage/paft_full5.yml" | awk '{print $1}')"
    echo "paft_smoke1_sha256=$(sha256sum "$config_stage/paft_smoke1.yml" | awk '{print $1}')"
    echo "no_paft_control_full5_sha256=$(sha256sum "$config_stage/no_paft_control_full5.yml" | awk '{print $1}')"
    echo "catalog_sha256=$(sha256sum "$catalog" | awk '{print $1}')"
    echo "contract_sha256=$(sha256sum "$contract" | awk '{print $1}')"
    echo "m73_manifest_sha256=$m73_expected"
  } > "$config_stage/CONFIG_BUNDLE_RECEIPT.txt"
  check_static_pins
  mv "$config_stage" "$config_bundle"
fi
check_config_bundle

run_arm() {
  local label="$1" cfg="$2" final_dir="$3" require_paft="$4" require_checkpoint="$5"
  if [[ -d "$final_dir" ]]; then
    test -s "$final_dir/train.log"
    if [[ "$require_checkpoint" == 1 ]]; then test -s "$final_dir/checkpoint_epoch4.pth"; fi
    if [[ "$require_paft" == 1 ]]; then
      grep -q '\[M71\] installed hardware-weighted PAFT hooks:' "$final_dir/train.log"
      grep -q '\[M71\] PAFT summary:' "$final_dir/train.log"
    fi
    return
  fi
  stage="wait_gpu_$label"
  stage_output=none
  wait_for_idle
  check_static_pins
  check_config_bundle
  stage="train_$label"
  local partial="${final_dir}.partial.$$.${RANDOM}"
  stage_output="$partial"
  mkdir -p "$partial"
  cd "$repo/neuron_experiments/H9_bipolar_self_attention"
  "$python" -u "$train_py" \
    --config "$cfg" \
    --prev_runid "$checkpoint" \
    --save_path "$partial/checkpoint_epoch{}.pth" \
    --finetune 1 2>&1 | tee "$partial/train.log"
  if grep -Eiq 'Traceback|RuntimeError|out of memory|CUDNN_STATUS' "$partial/train.log"; then
    echo "M87 $label failure signature" >&2
    return 20
  fi
  if [[ "$require_paft" == 1 ]]; then
    grep -q '\[M71\] installed hardware-weighted PAFT hooks:' "$partial/train.log"
    grep -q '\[M71\] PAFT summary:' "$partial/train.log"
  fi
  if [[ "$require_checkpoint" == 1 ]]; then test -s "$partial/checkpoint_epoch4.pth"; fi
  check_static_pins
  check_config_bundle
  mv "$partial" "$final_dir"
}

mkdir -p "$run_root"
run_arm smoke1 "$smoke_config" "$smoke_run" 1 0
run_arm no_paft_control_full5 "$control_config" "$control_run" 0 1
run_arm paft_full5 "$full_config" "$full_run" 1 1

stage=publish_chain_receipt
stage_output=none
receipt_tmp="${chain_receipt}.tmp.$$"
{
  echo "status=PASS_M87_H67_TRAINONLY_PAFT_PAIRED_FULL5"
  echo "completion_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "m73_manifest_sha256=$(sha256sum "$m73_manifest" | awk '{print $1}')"
  echo "catalog_sha256=$(sha256sum "$catalog" | awk '{print $1}')"
  echo "contract_sha256=$(sha256sum "$contract" | awk '{print $1}')"
  echo "control_checkpoint_epoch4_sha256=$(sha256sum "$control_run/checkpoint_epoch4.pth" | awk '{print $1}')"
  echo "paft_checkpoint_epoch4_sha256=$(sha256sum "$full_run/checkpoint_epoch4.pth" | awk '{print $1}')"
  echo "paired_control_complete=true"
  echo "valid825_accuracy=false"
  echo "cycle_speedup=false"
  echo "system_speedup=false"
  echo "headline=false"
} > "$receipt_tmp"
mv "$receipt_tmp" "$chain_receipt"
success=1
cat "$chain_receipt"
