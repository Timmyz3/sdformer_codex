#!/usr/bin/env bash
set -euo pipefail

repo="/root/private_data/work/sdformer_codex/SDformer"
python_bin="/opt/conda/envs/sdformerflow/bin/python"
evaluator_rel="hw_autoresearch_nts07/system_handoff/scripts/eval_m260_m235_approx_ffn_bn_DSEC.py"
config_rel="neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml"
checkpoint_rel="neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
output_rel="hw_autoresearch_nts07/results/m260_m235_ffn_bn_paired_s${M260_SAMPLES:-10}_r1_20260825"
samples="${M260_SAMPLES:-10}"
evaluator="$repo/$evaluator_rel"
config="$repo/$config_rel"
checkpoint="$repo/$checkpoint_rel"
output="$repo/$output_rel"

cd "$repo"

if [[ ! "$samples" =~ ^[1-9][0-9]*$ ]]; then
  echo "M260_SAMPLES must be a positive integer" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 3
fi

sha256sum -c <<'SHA256'
5716068204442df606f5bec785ed14e5d3a2628e6152e75c2227efb86c23b37b  hw_autoresearch_nts07/system_handoff/scripts/eval_m260_m235_approx_ffn_bn_DSEC.py
ba40b42c7395fd703c59a183a19b6a4fd38fa08ed75201008f03fd71b82aaef1  third_party/SDformerFlow/eval_DSEC_flow_SNN.py
8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49  neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml
4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158  neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth
8ec3b3ca594962c5f7a5a050df030a4a1dddccc768d791975148a2d895985430  hw_autoresearch_nts07/system_simulator/scripts/analyze_m234_h67_dynamic_bn_lut_newton_coefficients.py
a10da0a8ffe7b30665cb8fb3270603448166f8ac3f6e51d4831765a210b35272  hw_autoresearch_nts07/results/m245_m235_full220800_vcs_r1_exact_20260825/SHA256SUMS
8a0f07a74d49229019dde0ae7c69ea2fdc1040d4723d82d5ccaefe49790795eb  hw_autoresearch_nts07/results/m246_m245_full220800_independent_hammer_r1_20260825/SHA256SUMS
dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4  hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md
SHA256

mkdir -p "$output/reference" "$output/m235_approx"
sha256sum \
  "$evaluator_rel" "$config_rel" "$checkpoint_rel" \
  > "$output/input_identity.sha256"
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$output/gpu_before.csv"

# The frozen upstream evaluator resolves DSEC sequence-list paths from its own
# repository root.  Keep all M260 identities and outputs absolute, then launch
# from that expected working directory.
cd "$repo/third_party/SDformerFlow"

common=(
  "$python_bin" -u "$evaluator"
  --config "$config"
  --checkpoint "$checkpoint"
  --bn-policy no_running
  --max-samples "$samples"
  --mode valid
)

SDFORMER_USE_MLFLOW=0 PYTHONPATH="$repo/third_party/SDformerFlow" \
  "${common[@]}" \
  --m260-mode reference \
  --path_results "$output/reference" \
  --dump-per-frame "$output/reference/per_frame.csv" \
  --m260-receipt "$output/reference/m260_runtime_receipt.json" \
  2>&1 | tee "$output/reference/run.log"

SDFORMER_USE_MLFLOW=0 PYTHONPATH="$repo/third_party/SDformerFlow" \
  "${common[@]}" \
  --m260-mode m235_approx \
  --path_results "$output/m235_approx" \
  --dump-per-frame "$output/m235_approx/per_frame.csv" \
  --m260-receipt "$output/m235_approx/m260_runtime_receipt.json" \
  2>&1 | tee "$output/m235_approx/run.log"

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$output/gpu_after.csv"
find "$output" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$output/SHA256SUMS"
sha256sum "$output/SHA256SUMS"
