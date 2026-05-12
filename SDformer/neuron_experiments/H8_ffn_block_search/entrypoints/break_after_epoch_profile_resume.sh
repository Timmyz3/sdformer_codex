#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-/root/private_data/work/sdformer_codex/SDformer}
EXP="$BASE/neuron_experiments/H8_ffn_block_search"
RUN_DIR=${RUN_DIR:-$EXP/results/h8m_stage3_block0_all_120_full_from_20260511_180615_setsid}
CONFIG=${CONFIG:-$EXP/configs/generated_full/h8m_stage3_block0_all_120_full_from_20260511_180615.yml}
TRAIN_PID=${TRAIN_PID:-976513}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}
NUM_SAMPLES=${NUM_SAMPLES:-40}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
SNN_BACKEND=${SNN_BACKEND:-auto}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-30}

latest_ckpt() {
  ls -1v "$RUN_DIR"/checkpoint_epoch*.pth 2>/dev/null | tail -1
}

epoch_from_ckpt() {
  basename "$1" .pth | sed -E 's/checkpoint_epoch([0-9]+)/\1/'
}

wait_stable_file() {
  local path="$1"
  local last_size current_size
  last_size=$(stat -c %s "$path")
  sleep 10
  current_size=$(stat -c %s "$path")
  while [[ "$last_size" != "$current_size" ]]; do
    last_size="$current_size"
    sleep 10
    current_size=$(stat -c %s "$path")
  done
}

echo "[break-profile $STAMP] started at $(date -u)"
start_ckpt=$(latest_ckpt)
start_epoch=$(epoch_from_ckpt "$start_ckpt")
echo "[break-profile $STAMP] current latest checkpoint: $start_ckpt"
echo "[break-profile $STAMP] waiting for next epoch after $start_epoch"

while true; do
  ckpt=$(latest_ckpt)
  epoch=$(epoch_from_ckpt "$ckpt")
  if (( epoch > start_epoch )); then
    wait_stable_file "$ckpt"
    break
  fi
  sleep 20
done

echo "[break-profile $STAMP] new checkpoint ready: $ckpt"

if ps -p "$TRAIN_PID" >/dev/null 2>&1; then
  echo "[break-profile $STAMP] stopping train pid $TRAIN_PID"
  kill -TERM "$TRAIN_PID" || true
  for _ in $(seq 1 60); do
    if ! ps -p "$TRAIN_PID" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  if ps -p "$TRAIN_PID" >/dev/null 2>&1; then
    echo "[break-profile $STAMP] train pid still alive, sending KILL"
    kill -KILL "$TRAIN_PID" || true
  fi
fi

sleep 20
echo "[break-profile $STAMP] GPU before profile:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

profile_out="$EXP/results/profile_h8m_interrupt_epoch${epoch}_valid${NUM_SAMPLES}_${STAMP}"
echo "[break-profile $STAMP] profiling $ckpt -> $profile_out"
/opt/conda/envs/sdformerflow/bin/python -u "$EXP/entrypoints/profile_sops.py" \
  --config "$CONFIG" \
  --checkpoint "$ckpt" \
  --output-dir "$profile_out" \
  --split valid \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --snn-backend "$SNN_BACKEND" \
  --metric AEE \
  --metric AAE

remaining=$(( TOTAL_EPOCHS - epoch - 1 ))
if (( remaining <= 0 )); then
  echo "[break-profile $STAMP] no remaining epochs; not restarting"
  exit 0
fi

continue_cfg="$EXP/configs/generated_full/h8m_stage3_block0_continue_from_epoch${epoch}_${STAMP}.yml"
continue_dir="$EXP/results/h8m_stage3_block0_continue_from_epoch${epoch}_${STAMP}_setsid"
mkdir -p "$continue_dir"

python - "$CONFIG" "$continue_cfg" "$remaining" "$epoch" "$STAMP" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
remaining = int(sys.argv[3])
epoch = int(sys.argv[4])
stamp = sys.argv[5]

cfg = yaml.safe_load(src.read_text())
cfg["experiment"] = f"h8m_stage3_block0_continue_from_epoch{epoch}_{stamp}"
cfg.setdefault("loader", {})["n_epochs"] = remaining
cfg.setdefault("runtime", {})["continued_from_epoch"] = epoch
cfg["runtime"]["continued_from_checkpoint"] = None
cfg.setdefault("optimizer", {})
cfg["optimizer"]["milestones"] = []
cfg.setdefault("note", "")
cfg["note"] = (cfg["note"] + f" Continued after interrupt/profile from epoch {epoch}.").strip()
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

python - "$continue_cfg" "$ckpt" <<'PY'
from pathlib import Path
import sys
import yaml

cfg_path = Path(sys.argv[1])
ckpt = str(Path(sys.argv[2]).resolve())
cfg = yaml.safe_load(cfg_path.read_text())
cfg.setdefault("runtime", {})["continued_from_checkpoint"] = ckpt
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "[break-profile $STAMP] restarting remaining $remaining epoch(s)"
setsid /opt/conda/envs/sdformerflow/bin/python -u "$EXP/entrypoints/train.py" \
  --config "$continue_cfg" \
  --prev_runid "$ckpt" \
  --save_path "$continue_dir/checkpoint_epoch{}.pth" \
  > "$continue_dir/train.log" 2>&1 &
echo $! > "$continue_dir/pid.txt"
printf '%s\n' \
  "config=$continue_cfg" \
  "prev_runid=$ckpt" \
  "save_path=$continue_dir/checkpoint_epoch{}.pth" \
  "pid=$(cat "$continue_dir/pid.txt")" \
  > "$continue_dir/run_command.txt"

echo "[break-profile $STAMP] resumed pid $(cat "$continue_dir/pid.txt")"
echo "[break-profile $STAMP] done at $(date -u)"
