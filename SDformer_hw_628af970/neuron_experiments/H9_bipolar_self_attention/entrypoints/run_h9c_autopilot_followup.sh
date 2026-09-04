#!/usr/bin/env bash
set -euo pipefail

cd /root/private_data/work/sdformer_codex/SDformer

PY=${PY:-/opt/conda/envs/sdformerflow/bin/python}
EXP=neuron_experiments/H9_bipolar_self_attention
BASE_CKPT=experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth

CURRENT_PID=${CURRENT_PID:-1253020}
CURRENT_RUN=${CURRENT_RUN:-$EXP/results/h9c_layers2_all6_ffn_no_down_full_20260513_172341_setsid}
CURRENT_CONFIG=${CURRENT_CONFIG:-$EXP/configs/h9c_layers2_all6_ffn_no_down_full.yml}

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
AUTOPILOT_LOG=$EXP/results/h9c_autopilot_followup_${STAMP}.log

mkdir -p "$EXP/results"
exec > >(tee -a "$AUTOPILOT_LOG") 2>&1

echo "[autopilot] started at $(date)"
echo "[autopilot] watching current PID=$CURRENT_PID run=$CURRENT_RUN"
echo "[autopilot] log=$AUTOPILOT_LOG"

wait_for_pid() {
  local pid="$1"
  while kill -0 "$pid" >/dev/null 2>&1; do
    echo "[autopilot] $(date) PID $pid still running"
    sleep 300
  done
  echo "[autopilot] $(date) PID $pid finished"
}

latest_checkpoint() {
  local run_dir="$1"
  local ckpt
  ckpt=$(ls -1t "$run_dir"/checkpoint_epoch*.pth 2>/dev/null | head -n 1 || true)
  if [[ -z "$ckpt" ]]; then
    echo "[autopilot] ERROR: no checkpoint found under $run_dir" >&2
    return 1
  fi
  printf '%s\n' "$ckpt"
}

profile_checkpoint() {
  local tag="$1"
  local config="$2"
  local ckpt="$3"
  local out_dir="$EXP/results/profile_${tag}_${STAMP}"

  echo "[autopilot] profiling tag=$tag checkpoint=$ckpt" >&2
  "$PY" -u "$EXP/entrypoints/profile_sops.py" \
    --config "$config" \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
    --split valid \
    --num-samples 40 \
    --batch-size 8 \
    --num-workers 8 \
    --device cuda:0 \
    --fallback-dense-ops 42.63G \
    --metric AEE \
    --metric AAE >&2

  echo "[autopilot] profile summary: $out_dir/sops_summary.json" >&2
  "$PY" - "$out_dir/sops_summary.json" <<'PY' >&2
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)
metrics = d.get("metrics", {})
print(
    "[autopilot] metrics "
    f"AEE={metrics.get('AEE')} "
    f"AAE={metrics.get('AAE')} "
    f"SOPs={d.get('estimated_total_sops_human')} "
    f"firing={d.get('global_firing_rate')}"
)
PY
  printf '%s\n' "$out_dir/sops_summary.json"
}

choose_followup() {
  local summary="$1"
  "$PY" - "$summary" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)

metrics = d.get("metrics", {})
aee = float(metrics.get("AEE", 999.0))
aae = float(metrics.get("AAE", 999.0))
sops = float(d.get("estimated_total_sops", 9e99))

# Keep this gate intentionally conservative: H9b showed AAE explosion even with low SOPs.
stable = (aee <= 1.65) and (aae <= 10.0) and (sops <= 3.30e9)
if stable:
    print("h9d_layers2_all6_plus_layer0_mlp_no_down_full")
else:
    print("h9c_layers2_b025_ffn_no_down_full")
PY
}

run_training() {
  local name="$1"
  local config="$EXP/configs/${name}.yml"
  local run_dir="$EXP/results/${name}_${STAMP}_setsid"

  if [[ ! -f "$config" ]]; then
    echo "[autopilot] ERROR: missing config $config" >&2
    return 1
  fi

  mkdir -p "$run_dir"
  echo "$PY -u $EXP/entrypoints/train.py --config $config --prev_runid $BASE_CKPT --save_path $run_dir/checkpoint_epoch{}.pth" > "$run_dir/command.txt"
  echo "[autopilot] launching follow-up training name=$name run_dir=$run_dir" >&2
  "$PY" -u "$EXP/entrypoints/train.py" \
    --config "$config" \
    --prev_runid "$BASE_CKPT" \
    --save_path "$run_dir/checkpoint_epoch{}.pth" >&2
  echo "[autopilot] follow-up training finished name=$name" >&2
  printf '%s\n' "$run_dir"
}

append_runs_note() {
  local text="$1"
  {
    echo
    echo "### Autopilot ${STAMP}"
    echo "$text"
  } >> "$EXP/RUNS.md"
}

wait_for_pid "$CURRENT_PID"

current_ckpt=$(latest_checkpoint "$CURRENT_RUN")
current_summary=$(profile_checkpoint "h9c_all6_after_full" "$CURRENT_CONFIG" "$current_ckpt")
next_name=$(choose_followup "$current_summary")
echo "[autopilot] selected next experiment: $next_name"

append_runs_note "- H9c all6 completed and profiled at \`$current_summary\`; selected next: \`$next_name\`."

next_run=$(run_training "$next_name")
next_ckpt=$(latest_checkpoint "$next_run")
next_summary=$(profile_checkpoint "${next_name}_after_full" "$EXP/configs/${next_name}.yml" "$next_ckpt")
append_runs_note "- \`$next_name\` completed at \`$next_run\`; profile: \`$next_summary\`."

if [[ "$next_name" == "h9c_layers2_b025_ffn_no_down_full" ]]; then
  second_choice=$("$PY" - "$next_summary" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)
metrics = d.get("metrics", {})
aee = float(metrics.get("AEE", 999.0))
aae = float(metrics.get("AAE", 999.0))
sops = float(d.get("estimated_total_sops", 9e99))
print("yes" if (aee <= 1.65 and aae <= 10.0 and sops <= 3.50e9) else "no")
PY
)
  if [[ "$second_choice" == "yes" ]]; then
    echo "[autopilot] gentler B025 is stable; launching stronger H9d as second follow-up"
    second_run=$(run_training "h9d_layers2_all6_plus_layer0_mlp_no_down_full")
    second_ckpt=$(latest_checkpoint "$second_run")
    second_summary=$(profile_checkpoint "h9d_after_b025_full" "$EXP/configs/h9d_layers2_all6_plus_layer0_mlp_no_down_full.yml" "$second_ckpt")
    append_runs_note "- B025 was stable, so H9d second follow-up ran at \`$second_run\`; profile: \`$second_summary\`."
  else
    echo "[autopilot] B025 did not pass the stronger gate; stopping queue to avoid burning time on bad variants"
  fi
fi

echo "[autopilot] finished at $(date)"
