#!/usr/bin/env bash
set -euo pipefail

BASE=/root/private_data/work/sdformer_codex/SDformer
EXP=neuron_experiments/H9_bipolar_self_attention
PY=/opt/conda/envs/sdformerflow/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
H9A_RUN=${H9A_RUN:-$BASE/$EXP/results/h9a_shiftmax_compat_h8m_full_bs8_20260512_200523_setsid}
BASE_CKPT=${BASE_CKPT:-$BASE/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth}
LOG=$BASE/$EXP/results/autopilot_${STAMP}.log

cd "$BASE"

log() {
  echo "[$(date +%F_%T)] $*" | tee -a "$LOG"
}

latest_ckpt() {
  local dir=$1
  ls -1v "$dir"/checkpoint_epoch*.pth 2>/dev/null | tail -1
}

profile_ckpt() {
  local config=$1
  local ckpt=$2
  local outdir=$3
  local samples=$4
  mkdir -p "$outdir"
  SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$EXP/entrypoints/profile_sops.py" \
      --config "$config" \
      --checkpoint "$ckpt" \
      --split valid \
      --num-samples "$samples" \
      --batch-size 1 \
      --num-workers 4 \
      --metric AEE \
      --metric AAE \
      --output-dir "$outdir" 2>&1 | tee -a "$LOG"
}

wait_for_run() {
  local run_dir=$1
  local pid_file=$run_dir/pid.txt
  if [[ -f "$pid_file" ]]; then
    local shell_pid
    shell_pid=$(cat "$pid_file")
    if ps -p "$shell_pid" >/dev/null 2>&1; then
      log "waiting for run pid=$shell_pid dir=$run_dir"
      while ps -p "$shell_pid" >/dev/null 2>&1; do
        sleep 300
        local ckpt
        ckpt=$(latest_ckpt "$run_dir" || true)
        log "still running; latest checkpoint=${ckpt:-none}"
      done
    fi
  fi
}

select_best_profile() {
  "$PY" - "$BASE/$EXP/results" "$STAMP" <<'PY'
import glob
import json
import math
import pathlib
import sys

results_dir = pathlib.Path(sys.argv[1])
stamp = sys.argv[2]
rows = []
for path in glob.glob(str(results_dir / f"profile_h9b_*_valid10_{stamp}" / "sops_summary.json")):
    data = json.load(open(path))
    metrics = data.get("metrics", {})
    aee = float(metrics.get("AEE", math.inf))
    aae = float(metrics.get("AAE", math.inf))
    sops = float(data.get("estimated_total_sops", math.inf))
    firing = float(data.get("global_firing_rate", math.inf))
    name = pathlib.Path(path).parent.name.removeprefix("profile_").removesuffix(f"_valid10_{stamp}")
    score = aee + 0.025 * aae + 0.03 * max(0.0, sops / 1e9 - 3.60)
    rows.append((score, name, aee, aae, sops, firing, path))
rows.sort()
summary_path = results_dir / f"h9b_selection_{stamp}.md"
with summary_path.open("w") as f:
    f.write("# H9b Selection\\n\\n")
    f.write("| rank | name | AEE | AAE | SOPs(G) | firing | score | promote |\\n")
    f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\\n")
    for rank, (score, name, aee, aae, sops, firing, path) in enumerate(rows, 1):
        promote = aee <= 1.15 and aae <= 7.0 and sops <= 3.75e9
        f.write(f"| {rank} | {name} | {aee:.6f} | {aae:.6f} | {sops/1e9:.4f} | {firing:.6f} | {score:.6f} | {promote} |\\n")
if not rows:
    print("")
    sys.exit(0)
best = rows[0]
promote = best[2] <= 1.15 and best[3] <= 7.0 and best[4] <= 3.75e9
print(best[1] if promote else "")
PY
}

log "H9 autopilot started"
wait_for_run "$H9A_RUN"

H9A_CKPT=$(latest_ckpt "$H9A_RUN")
log "H9a finished/latest checkpoint: $H9A_CKPT"
profile_ckpt "$EXP/configs/h9a_shiftmax_compat_h8m_full.yml" "$H9A_CKPT" "$EXP/results/profile_h9a_full_$(basename "$H9A_CKPT" .pth)_valid40_${STAMP}" 40

log "generating H9b configs"
"$PY" "$EXP/entrypoints/generate_h9b_configs.py" \
  --base "$EXP/configs/h9a_shiftmax_compat_h8m_smoke.yml" \
  --out-dir "$EXP/configs/generated_h9b_${STAMP}"

for config in "$EXP"/configs/generated_h9b_${STAMP}/h9b_*_120.yml; do
  name=$(basename "$config" .yml)
  run_dir="$EXP/results/${name}_${STAMP}"
  mkdir -p "$run_dir"
  log "short training $name"
  SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$EXP/entrypoints/train.py" \
      --config "$config" \
      --prev_runid "$BASE_CKPT" \
      --save_path "$run_dir/checkpoint_epoch{}.pth" > "$run_dir/train.log" 2>&1
  ckpt=$(latest_ckpt "$run_dir")
  log "profiling $name checkpoint=$ckpt"
  profile_ckpt "$config" "$ckpt" "$EXP/results/profile_${name}_valid10_${STAMP}" 10
done

best=$(select_best_profile)
log "H9b best promoted candidate: ${best:-none}"
if [[ -z "$best" ]]; then
  log "No H9b short run met promotion criteria; autopilot stops to avoid wasting GPU."
  exit 0
fi

short_config="$EXP/configs/generated_h9b_${STAMP}/${best}.yml"
full_name="${best}_full"
"$PY" "$EXP/entrypoints/generate_h9b_configs.py" \
  --base "$EXP/configs/h9a_shiftmax_compat_h8m_smoke.yml" \
  --out-dir "$EXP/configs/generated_h9b_${STAMP}" \
  --full-from "$short_config" \
  --full-name "$full_name"
full_config="$EXP/configs/generated_h9b_${STAMP}/${full_name}.yml"
full_run="$EXP/results/${full_name}_${STAMP}_setsid"
mkdir -p "$full_run"
log "starting promoted full run $full_name"
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u "$EXP/entrypoints/train.py" \
    --config "$full_config" \
    --prev_runid "$BASE_CKPT" \
    --save_path "$full_run/checkpoint_epoch{}.pth" > "$full_run/train.log" 2>&1
full_ckpt=$(latest_ckpt "$full_run")
log "promoted full finished checkpoint=$full_ckpt"
profile_ckpt "$full_config" "$full_ckpt" "$EXP/results/profile_${full_name}_valid40_${STAMP}" 40
log "H9 autopilot complete"
