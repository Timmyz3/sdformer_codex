#!/usr/bin/env bash
set -u

BASE=${BASE:-/root/private_data/work/sdformer_codex/SDformer}
EXP=${EXP:-neuron_experiments/H8_ffn_block_search}
CKPT=${CKPT:-$BASE/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth}
STAMP=${STAMP:-$(date -u +%Y%m%d_%H%M%S)}
DENSE_OPS=${DENSE_OPS:-42.63G}

configs=(
  h8e_stage0_block0_all_120
  h8f_stage0_block1_all_120
  h8g_stage2_block0_all_120
  h8h_stage2_block1_all_120
  h8i_stage2_block2_all_120
  h8j_stage2_block3_all_120
  h8k_stage2_block4_all_120
  h8l_stage2_block5_all_120
  h8m_stage3_block0_all_120
  h8n_stage3_block1_all_120
  h8o_stage3_all_all_120
  h8p_stage1b0_stage2b4_all_120
  h8q_stage1b0_stage3b0_all_120
  h8r_stage2_mid_all_120
)

if [[ -n "${WAIT_PID:-}" ]]; then
  echo "[H8 queue] waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 300
  done
fi

echo "[H8 queue] stamp=$STAMP"
echo "[H8 queue] checkpoint=$CKPT"

for name in "${configs[@]}"; do
  cfg="$BASE/$EXP/configs/${name}.yml"
  outprefix="$BASE/$EXP/results/${name}_${STAMP}"
  trainlog="${outprefix}_train.log"
  echo "===== TRAIN $name $STAMP =====" | tee -a "$trainlog"
  SDFORMER_USE_MLFLOW=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /opt/conda/envs/sdformerflow/bin/python -u "$BASE/$EXP/entrypoints/train.py" \
      --config "$cfg" \
      --prev_runid "$CKPT" \
      --save_path "${outprefix}_checkpoint_epoch{}.pth" 2>&1 | tee -a "$trainlog"
  ckpt="${outprefix}_checkpoint_epoch0.pth"
  if [[ ! -f "$ckpt" ]]; then
    echo "[H8 queue] skip profile for $name: checkpoint missing" | tee -a "$trainlog"
    continue
  fi
  profdir="$BASE/$EXP/results/profile_${name}_valid10_${STAMP}"
  echo "===== PROFILE $name $STAMP =====" | tee -a "${profdir}.log"
  /opt/conda/envs/sdformerflow/bin/python -u "$BASE/$EXP/entrypoints/profile_sops.py" \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --output-dir "$profdir" \
    --split valid \
    --num-samples 10 \
    --batch-size 1 \
    --num-workers 4 \
    --dense-ops "$DENSE_OPS" \
    --metric AEE \
    --metric AAE \
    --module-pattern Spiking_neuron 2>&1 | tee -a "${profdir}.log"
done
