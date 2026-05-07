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

# H1 is NOT running
if [ "$PREV_STATE" = "h1_running" ]; then
    log "H1 TRAINING COMPLETED! Transitioning to A5..."
    echo "h1_done" > "$STATE_FILE"
fi

CURRENT_STATE=$(cat "$STATE_FILE" 2>/dev/null || echo "unknown")

case "$CURRENT_STATE" in
    h1_done)
        log "=== Starting A5 smoke test ==="
        SDFORMER_USE_MLFLOW=0 python SDformer/neuron_autoresearch/entrypoints/train.py \
            --config "$A5_SMOKE_CONFIG" \
            --prev_runid "$BASELINE_CKPT" \
            > "$REPO_ROOT/neuron_autoresearch/experiments/a5_refractory/results/smoke_$(date +%Y%m%d_%H%M%S).log" 2>&1
        SMOKE_EXIT=$?
        if [ $SMOKE_EXIT -eq 0 ]; then
            log "A5 smoke PASSED, launching full training..."
            echo "a5_smoke_ok" > "$STATE_FILE"
            # Launch full training in background
            nohup bash -c "
                cd /root/private_data/work/SDformer && \
                SDFORMER_USE_MLFLOW=0 python neuron_autoresearch/entrypoints/train.py \
                    --config $A5_FULL_CONFIG \
                    --prev_runid $BASELINE_CKPT \
                    > neuron_autoresearch/experiments/a5_refractory/results/a5_full_\$(date +%Y%m%d_%H%M%S).log 2>&1
                echo \"a5_full_done\" > neuron_autoresearch/.monitor_state
            " &
            A5_PID=$!
            log "A5 full training launched (PID=$A5_PID)"
        else
            log "A5 smoke FAILED (exit=$SMOKE_EXIT), check logs"
            echo "a5_smoke_failed" > "$STATE_FILE"
        fi
        ;;
    a5_smoke_ok)
        A5_PID=$(pgrep -f "a5_refractory" 2>/dev/null | head -1 || true)
        if [ -n "$A5_PID" ]; then
            log "A5 full training running (PID=$A5_PID)"
        else
            A5_LOG=$(ls -t "$REPO_ROOT/neuron_autoresearch/experiments/a5_refractory/results/a5_full_"*.log 2>/dev/null | head -1 || true)
            if [ -f "$A5_LOG" ]; then
                LATEST=$(grep -oP 'Epoch \d+' "$A5_LOG" 2>/dev/null | tail -1 || echo "unknown")
                log "A5 full possibly done, latest: $LATEST. Check: $A5_LOG"
            fi
        fi
        ;;
    a5_full_done)
        log "=== A5 complete! Next: A1 FSN on G1 ==="
        echo "ready_for_a1" > "$STATE_FILE"
        ;;
    *)
        log "Unknown state: $CURRENT_STATE"
        ;;
esac
