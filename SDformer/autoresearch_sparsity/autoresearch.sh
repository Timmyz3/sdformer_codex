#!/bin/bash
# Autoresearch benchmark script for sparse pipeline optimization.
# Mode 1 (eval): Profile SOPs/AEE for a config using existing checkpoint.
# Mode 2 (train): Train with sparsity config, then profile the result.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-eval}"           # eval | train
CONFIG="${2:-autoresearch_sparsity/configs/baseline_upstream.yml}"
CHECKPOINT="${3:-experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth}"
NUM_SAMPLES="${4:-40}"

STAMP=$(date +%Y%m%d_%H%M%S)

# ── GPU check ──────────────────────────────────────────────────────────
gpu_free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [ -z "$gpu_free_mb" ]; then
    echo "ERROR: Cannot query GPU"
    exit 1
fi
# Need at least 10GB free for training
if [ "$MODE" = "train" ] && [ "$gpu_free_mb" -lt 10000 ]; then
    echo "BLOCKED: GPU has only ${gpu_free_mb}MB free, need 10000MB for training"
    exit 2
fi

# ── Eval mode: profile existing checkpoint ─────────────────────────────
if [ "$MODE" = "eval" ]; then
    OUTPUT_DIR="autoresearch_sparsity/results/profile_${STAMP}"
    python -m autoresearch_sparsity.entrypoints.profile_upstream_sparse \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --num-samples "$NUM_SAMPLES" \
        --batch-size 1 \
        --num-workers 0 \
        --output-dir "$OUTPUT_DIR" \
        --split valid \
        --snn-backend torch \
        --metric AEE --metric AAE \
        --module-pattern Spiking_neuron \
        | tee "/tmp/ar_output_$$.txt"

    AEE=$(grep -oP 'AEE:\s*\K[\d.]+' "/tmp/ar_output_$$.txt" | head -1)
    AAE=$(grep -oP 'AAE:\s*\K[\d.]+' "/tmp/ar_output_$$.txt" | head -1)
    SOPS=$(grep -oP 'estimated_total_sops:\s*\K[\d.]+[GMK]?' "/tmp/ar_output_$$.txt" | head -1)
    FIRING=$(grep -oP 'global_firing_rate:\s*\K[\d.]+' "/tmp/ar_output_$$.txt" | head -1)

    echo "METRIC aee=${AEE:-nan}"
    echo "METRIC aae=${AAE:-nan}"
    echo "METRIC sops=${SOPS:-nan}"
    echo "METRIC firing_rate=${FIRING:-nan}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"

# ── Train mode: run training then profile ──────────────────────────────
elif [ "$MODE" = "train" ]; then
    TRAIN_OUTPUT_DIR="autoresearch_sparsity/results/train_${STAMP}"
    mkdir -p "$TRAIN_OUTPUT_DIR"

    echo "=== Training with config: $CONFIG ==="
    echo "Checkpoint: $CHECKPOINT"
    echo "Output: $TRAIN_OUTPUT_DIR"

    # Run training (limited epochs for rapid iteration)
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    SDFORMER_USE_MLFLOW=0 python -m autoresearch_sparsity.entrypoints.train \
        --config "$CONFIG" \
        --prev_runid "$CHECKPOINT" \
        --save_path "${TRAIN_OUTPUT_DIR}/checkpoint_epoch{}.pth" \
        2>&1 | tee "${TRAIN_OUTPUT_DIR}/train.log"

    TRAIN_EXIT="${PIPESTATUS[0]}"
    if [ "$TRAIN_EXIT" -ne 0 ]; then
        echo "METRIC aee=0"
        echo "METRIC sops=0"
        echo "STATUS=crash"
        exit 0
    fi

    # Find the latest checkpoint
    LATEST_CKPT=$(ls -t "${TRAIN_OUTPUT_DIR}"/checkpoint_epoch*.pth 2>/dev/null | head -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "ERROR: No checkpoint found after training"
        echo "METRIC aee=0"
        echo "METRIC sops=0"
        echo "STATUS=crash"
        exit 0
    fi

    echo "=== Profiling trained model: $LATEST_CKPT ==="
    python -m autoresearch_sparsity.entrypoints.profile_upstream_sparse \
        --config "$CONFIG" \
        --checkpoint "$LATEST_CKPT" \
        --num-samples "$NUM_SAMPLES" \
        --batch-size 1 \
        --num-workers 0 \
        --output-dir "${TRAIN_OUTPUT_DIR}/profile" \
        --split valid \
        --snn-backend torch \
        --metric AEE --metric AAE \
        --module-pattern Spiking_neuron \
        | tee "/tmp/ar_output_$$.txt"

    AEE=$(grep -oP 'AEE:\s*\K[\d.]+' "/tmp/ar_output_$$.txt" | head -1)
    AAE=$(grep -oP 'AAE:\s*\K[\d.]+' "/tmp/ar_output_$$.txt" | head -1)
    SOPS=$(grep -oP 'estimated_total_sops:\s*\K[\d.]+[GMK]?' "/tmp/ar_output_$$.txt" | head -1)
    FIRING=$(grep -oP 'global_firing_rate:\s*\K[\d.]+' "/tmp/ar_output_$$.txt" | head -1)

    echo "METRIC aee=${AEE:-nan}"
    echo "METRIC aae=${AAE:-nan}"
    echo "METRIC sops=${SOPS:-nan}"
    echo "METRIC firing_rate=${FIRING:-nan}"
    echo "CHECKPOINT=${LATEST_CKPT}"
    echo "OUTPUT_DIR=${TRAIN_OUTPUT_DIR}"
fi
