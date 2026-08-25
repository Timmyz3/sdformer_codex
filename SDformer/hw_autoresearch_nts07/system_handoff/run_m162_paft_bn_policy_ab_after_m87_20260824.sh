#!/usr/bin/env bash
set -euo pipefail

repo=/root/private_data/work/sdformer_codex/SDformer
python_bin=/opt/conda/envs/sdformerflow/bin/python
chain="$repo/neuron_experiments/H9_bipolar_self_attention/results/m87_h67_trainonly_paft_paired_20260823/M87_CHAIN_COMPLETE.txt"
checkpoint="$repo/neuron_experiments/H9_bipolar_self_attention/results/m87_h67_trainonly_paft_paired_20260823/paft_full5/checkpoint_epoch4.pth"
config="$repo/hw_autoresearch_nts07/results/m87_h67_trainonly_paft_config_bundle_r2_20260823/paft_full5.yml"
evaluator="$repo/third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
output_root="$repo/neuron_experiments/H9_bipolar_self_attention/results/m162_paft_ep4_bn_policy_ab_valid825_20260824"
receipt="$output_root/M162_COMPLETE.txt"
log="$repo/hw_autoresearch_nts07/system_handoff/logs/m162_paft_bn_policy_ab_valid825_20260824.log"

mkdir -p "$(dirname "$log")"
mkdir -p "$output_root"
exec > >(tee -a "$log") 2>&1

if [[ -f "$receipt" ]]; then
  grep -qx 'status=PASS_M162_PAFT_EP4_BN_POLICY_AB_VALID825' "$receipt"
  cat "$receipt"
  exit 0
fi

echo "M162_WAIT_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
while [[ ! -f "$chain" ]]; do sleep 30; done
grep -qx 'status=PASS_M87_H67_TRAINONLY_PAFT_PAIRED_FULL5' "$chain"
test -s "$checkpoint"
test -s "$config"
test -s "$evaluator"

expected_checkpoint="$(awk -F= '$1=="paft_checkpoint_epoch4_sha256" {print $2}' "$chain")"
observed_checkpoint="$(sha256sum "$checkpoint" | awk '{print $1}')"
[[ ${#expected_checkpoint} -eq 64 && "$observed_checkpoint" == "$expected_checkpoint" ]]
config_sha="$(sha256sum "$config" | awk '{print $1}')"
evaluator_sha="$(sha256sum "$evaluator" | awk '{print $1}')"

idle=0
while (( idle < 4 )); do
  active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)"
  if [[ "$active" == "0" ]]; then idle=$((idle + 1)); else idle=0; fi
  echo "M162_IDLE_PROBE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ) active_compute_pids=$active consecutive_idle=$idle"
  if (( idle < 4 )); then sleep 30; fi
done

run_policy() {
  local policy="$1"
  local final_dir="$output_root/$policy"
  if [[ -d "$final_dir" ]]; then
    test -s "$final_dir/eval.log"
    return
  fi
  local partial="${final_dir}.partial.$$.${RANDOM}"
  mkdir "$partial"
  cd "$repo/neuron_experiments/H9_bipolar_self_attention"
  echo "M162_POLICY_START policy=$policy utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  SDFORMER_USE_MLFLOW=0 \
    SDFORMER_MLFLOW_MODEL_LOGGING=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=overlay:../../third_party/SDformerFlow \
    "$python_bin" -u "$evaluator" \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --path_results "$partial" \
      --mode valid \
      --bn-policy "$policy" \
      --dump-per-frame "$partial/per_frame.csv" \
      2>&1 | tee "$partial/eval.log"
  if grep -Eiq 'Traceback|RuntimeError|out of memory|CUDNN_STATUS|NaN|nan' "$partial/eval.log"; then
    echo "M162 failure signature policy=$policy" >&2
    return 20
  fi
  test -s "$partial/eval.log"
  test -s "$partial/per_frame.csv"
  mv "$partial" "$final_dir"
  echo "M162_POLICY_DONE policy=$policy utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_policy no_running
run_policy running

receipt_tmp="${receipt}.tmp.$$"
{
  echo 'status=PASS_M162_PAFT_EP4_BN_POLICY_AB_VALID825'
  echo "completion_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "checkpoint_sha256=$observed_checkpoint"
  echo "config_sha256=$config_sha"
  echo "evaluator_sha256=$evaluator_sha"
  echo 'no_running_valid825=true'
  echo 'running_valid825=true'
  echo 'accuracy_comparison_pending_parse=true'
  echo 'cycle_speedup=false'
  echo 'system_speedup=false'
  echo 'headline=false'
} > "$receipt_tmp"
mv "$receipt_tmp" "$receipt"
cat "$receipt"
