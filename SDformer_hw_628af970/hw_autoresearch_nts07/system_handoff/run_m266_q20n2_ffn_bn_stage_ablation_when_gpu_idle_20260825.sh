#!/usr/bin/env bash
set -euo pipefail

repo="/root/private_data/work/sdformer_codex/SDformer"
python_bin="/opt/conda/envs/sdformerflow/bin/python"
evaluator_rel="hw_autoresearch_nts07/system_handoff/scripts/eval_m266_q20n2_ffn_bn_stage_ablation_DSEC.py"
config_rel="neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml"
checkpoint_rel="neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
samples="${M266_SAMPLES:-10}"
output_rel="hw_autoresearch_nts07/results/m266_q20n2_ffn_bn_stage_ablation_s${samples}_r1_20260825"
evaluator="$repo/$evaluator_rel"
config="$repo/$config_rel"
checkpoint="$repo/$checkpoint_rel"
output="$repo/$output_rel"

cd "$repo"
if [[ ! "$samples" =~ ^[1-9][0-9]*$ ]]; then
  echo "M266_SAMPLES must be a positive integer" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 3
fi

sha256sum -c <<'SHA256'
7d0bfdd1adc2c0db44f19130a49aada212c733797cfb12e6d6a09a046b0ced6f  hw_autoresearch_nts07/system_handoff/scripts/eval_m266_q20n2_ffn_bn_stage_ablation_DSEC.py
bd5c8587c85f96e93b7dea18e6ca0e9c01898355abceea462fd89e1159737e32  hw_autoresearch_nts07/system_simulator/scripts/analyze_m263_dynamic_bn_precision_cost_dse.py
e5e2811583e99045bb41862b9e2b5a96ccec2ab6a938f31f7c11e8d7b4251094  hw_autoresearch_nts07/results/m263_dynamic_bn_precision_cost_dse_r1_20260825/SHA256SUMS
c82c50f886c5af45cbc8c185abd33e23f3a5e30733dcac17da8e9a7c4ef78f16  hw_autoresearch_nts07/results/m263_balanced_q20n2_ffn_bn_paired_s10_r1_20260825/LOCAL_SHA256SUMS
ba40b42c7395fd703c59a183a19b6a4fd38fa08ed75201008f03fd71b82aaef1  third_party/SDformerFlow/eval_DSEC_flow_SNN.py
8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49  neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml
4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158  neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth
dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4  hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md
SHA256

mkdir -p "$output"
sha256sum "$evaluator_rel" "$config_rel" "$checkpoint_rel" \
  > "$output/input_identity.sha256"
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$output/gpu_before.csv"

cd "$repo/third_party/SDformerFlow"
for target in stage0 stage1 stage2 stage3 bn1 bn2; do
  target_output="$output/$target"
  mkdir -p "$target_output"
  SDFORMER_USE_MLFLOW=0 PYTHONPATH="$repo/third_party/SDformerFlow" \
    "$python_bin" -u "$evaluator" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --bn-policy no_running \
    --max-samples "$samples" \
    --mode valid \
    --m266-mode balanced_q20n2 \
    --m266-target "$target" \
    --path_results "$target_output" \
    --dump-per-frame "$target_output/per_frame.csv" \
    --m266-receipt "$target_output/m266_runtime_receipt.json" \
    2>&1 | tee "$target_output/run.log"
done

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$output/gpu_after.csv"
find "$output" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$output/SHA256SUMS"
sha256sum "$output/SHA256SUMS"
