#!/usr/bin/env bash
# Autoresearch monitor: detects H1 completion, launches A5 smoke → full.
set -euo pipefail

REPO_ROOT="/root/private_data/work/sdformer_codex/SDformer"
MONITOR_LOG="$REPO_ROOT/neuron_autoresearch/monitor.log"
H1_LOG="$REPO_ROOT/neuron_experiments/H1_hw_sparse/results/h1_full_20260507_v2.log"
STATE_FILE="$REPO_ROOT/neuron_autoresearch/.monitor_state"
BASELINE_CKPT="experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"
A5_SMOKE_CONFIG="neuron_autoresearch/experiments/a5_refractory/configs/smoke.yml"
A5_FULL_CONFIG="neuron_autoresearch/experiments/a5_refractory/configs/full.yml"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"; }

cd "$REPO_ROOT/.."

# --- Determine current state ---
H1_PID=$(pgrep -f "H1_hw_sparse/entrypoints/train.py" 2>/dev/null | head -1 || true)
H1_STILL_RUNNING=false
if [ -n "$H1_PID" ]; then
    H1_STILL_RUNNING=true
fi

# Read previous state
PREV_STATE="unknown"
if [ -f "$STATE_FILE" ]; then
    PREV_STATE=$(cat "$STATE_FILE")
fi

# --- State machine ---
if $H1_STILL_RUNNING; then
    if [ "$PREV_STATE" != "h1_running" ]; then
        echo "h1_running" > "$STATE_FILE"
        log "H1 still running (PID=$H1_PID), waiting..."
    fi
    # Check H1 epoch progress from log
    if [ -f "$H1_LOG" ]; then
        LATEST_EPOCH=$(grep -oP "Epoch \d+" "$H1_LOG" 2>/dev/null | tail -1 || echo "unknown")
        LATEST_GATE=$(grep -oP "open_gates.: \d+" "$H1_LOG" 2>/dev/null | tail -1 || echo "unknown")
        log "H1 progress: $LATEST_EPOCH, gates: $LATEST_GATE"
    fi
    exit 0
fi

CURRENT_STATE=$(cat "$STATE_FILE" 2>/dev/null || echo "unknown")

# H1 done tracking (backward compat)
if [ "$CURRENT_STATE" = "h1_running" ] && ! $H1_STILL_RUNNING; then
    log "H1 training completed (detected retrospectively)"
    CURRENT_STATE="h1_done"
fi

case "$CURRENT_STATE" in
    a5_running)
        A5_PID=$(pgrep -f "a5_refractory.*train.py" 2>/dev/null | head -1 || true)
        if [ -z "$A5_PID" ]; then
            A5_LOG=$(ls -t "$REPO_ROOT/neuron_autoresearch/experiments/a5_refractory/results/a5_full_"*.log 2>/dev/null | head -1 || true)
            if [ -f "$A5_LOG" ]; then
                LATEST=$(grep -oP "Epoch stats.*epoch_time" "$A5_LOG" 2>/dev/null | tail -1 || echo "unknown")
                if echo "$LATEST" | grep -q "lr=0.000000"; then
                    log "A5 full training COMPLETED! Latest: $LATEST"
                    echo "a5_full_done" > "$STATE_FILE"
                else
                    log "A5 process not found but may have crashed. Latest: $LATEST"
                    echo "a5_crashed" > "$STATE_FILE"
                fi
            fi
        else
            A5_LOG=$(ls -t "$REPO_ROOT/neuron_autoresearch/experiments/a5_refractory/results/a5_full_"*.log 2>/dev/null | head -1 || true)
            if [ -f "$A5_LOG" ]; then
                LATEST=$(grep -oP "Epoch \d+" "$A5_LOG" 2>/dev/null | tail -1 || echo "unknown")
                GATE_INFO=$(grep -oP "\Q[AR]\E.*" "$A5_LOG" 2>/dev/null | tail -1 || echo "")
                log "A5 running: $LATEST"
            fi
        fi
        ;;
    a5_full_done)
        log "A5 complete! Time to profile results."
        # Profile will be done manually for now
        ;;
esac
