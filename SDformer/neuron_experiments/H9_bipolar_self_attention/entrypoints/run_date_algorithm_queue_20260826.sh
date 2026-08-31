#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/private_data/work/sdformer_codex/SDformer"
ENTRY="$ROOT/neuron_experiments/H9_bipolar_self_attention/entrypoints"
RESULT="$ROOT/neuron_experiments/H9_bipolar_self_attention/results/date_algorithm_queue_20260826"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/sdformerflow/bin/python}"
mkdir -p "$RESULT"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
record() { printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$RESULT/status.log"; }

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_MDR_USE_MLFLOW=0
export SDFORMER_MDR_SKIP_MLFLOW_MODEL_LOG=1
export SDFORMER_MDR_SKIP_MLFLOW_STATE_LOG=1
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

record "START MVSEC validation-gated Complete-TTX alpha screen"
"$PYTHON_BIN" -u "$ENTRY/run_mvsec_ttx_alpha_screen_20260826.py" \
  >>"$RESULT/mvsec_alpha_screen.log" 2>&1
record "END MVSEC validation-gated Complete-TTX alpha screen"

record "START DSEC same-parent full30 two-contribution controls"
"$PYTHON_BIN" -u "$ENTRY/run_date_two_contribution_full30_20260826.py" \
  >>"$RESULT/dsec_full30.log" 2>&1
record "END DSEC same-parent full30 two-contribution controls"

record "ALL COMPLETE DATE algorithm queue"
