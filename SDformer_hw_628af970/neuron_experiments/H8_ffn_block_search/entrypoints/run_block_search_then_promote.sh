#!/usr/bin/env bash
set -u

BASE=${BASE:-/root/private_data/work/sdformer_codex/SDformer}
EXP=${EXP:-neuron_experiments/H8_ffn_block_search}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}
TOP_K=${TOP_K:-1}
AEE_MAX=${AEE_MAX:-1.07}
AAE_MAX=${AAE_MAX:-6.35}
SOPS_MAX_G=${SOPS_MAX_G:-3.60}
EXTRA_STAMP=${EXTRA_STAMP:-20260511_165537}
RUN_FULL=${RUN_FULL:-1}

log="$BASE/$EXP/results/orchestrator_${STAMP}.log"
mkdir -p "$BASE/$EXP/results"

{
  echo "[H8 orchestrator] stamp=$STAMP"
  echo "[H8 orchestrator] top_k=$TOP_K aee_max=$AEE_MAX aae_max=$AAE_MAX sops_max_g=$SOPS_MAX_G"
  if [[ -n "${WAIT_PID:-}" ]]; then
    echo "[H8 orchestrator] waiting for upstream pid $WAIT_PID"
  fi

  STAMP="$STAMP" WAIT_PID="${WAIT_PID:-}" "$BASE/$EXP/entrypoints/run_block_search_queue.sh"

  promote_args=(
    --base "$BASE"
    --exp "$EXP"
    --stamp "$STAMP"
    --top-k "$TOP_K"
    --aee-max "$AEE_MAX"
    --aae-max "$AAE_MAX"
    --sops-max-g "$SOPS_MAX_G"
  )
  if [[ -n "$EXTRA_STAMP" ]]; then
    promote_args+=(--extra-stamp "$EXTRA_STAMP")
  fi

  /opt/conda/envs/sdformerflow/bin/python "$BASE/$EXP/entrypoints/promote_best_short.py" "${promote_args[@]}"

  full_script="$BASE/$EXP/results/run_promoted_full_${STAMP}.sh"
  if [[ "$RUN_FULL" == "1" && -x "$full_script" ]]; then
    echo "[H8 orchestrator] launching promoted full run(s): $full_script"
    "$full_script"
  else
    echo "[H8 orchestrator] RUN_FULL=$RUN_FULL, full run script not launched"
  fi
} >> "$log" 2>&1
