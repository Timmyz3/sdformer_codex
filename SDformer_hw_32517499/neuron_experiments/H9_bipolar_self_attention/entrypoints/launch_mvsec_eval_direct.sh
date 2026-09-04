#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
cd "$REPO"

export SDFORMER_USE_MLFLOW=0
export SDFORMER_SNN_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=TRUE

NB0_CKPT="experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
NTS07B_CKPT="neuron_experiments/H9_bipolar_self_attention/results/nts07b_hw_h60_ffn_update0_act0_s1224_steps1224_auto_full_bs6_20260608_042113_setsid/checkpoint_epoch29.pth"
BASE_CFG="neuron_experiments/H9_bipolar_self_attention/configs/generated/eval_mvsec_dt1_baseline.yml"
NTS_CFG="neuron_experiments/H9_bipolar_self_attention/configs/generated/eval_mvsec_dt1_nts07b.yml"

python3 neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_mvsec_eval.py \
  --config "$BASE_CFG" \
  --checkpoint "$NB0_CKPT" \
  --out-dir results_inference/mvsec_nb0_ep59_dt1 \
  --sequence indoor_flying3

python3 neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_mvsec_eval.py \
  --config "$NTS_CFG" \
  --checkpoint "$NTS07B_CKPT" \
  --out-dir results_inference/mvsec_nts07b_ep29_dt1 \
  --sequence indoor_flying3