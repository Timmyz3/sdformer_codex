#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CKPT_DIR="${1:-experiments/logs/train}"

CONFIGS=(
  "configs/sdformer_baseline.yaml"
  "configs/model_variants/variant_a.yaml"
  "configs/model_variants/variant_b.yaml"
  "configs/model_variants/variant_c.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  variant="$(basename "${cfg}" .yaml)"
  if [[ "${variant}" == "sdformer_baseline" ]]; then
    ckpt_name="sdformer_baseline_best.pth"
  else
    ckpt_name="${variant}_best.pth"
  fi
  python -m src.trainers.eval --config "${cfg}" --checkpoint "${CKPT_DIR}/${ckpt_name}" --write-summary
done

python -m src.trainers.eval \
  --config configs/model_variants/variant_c.yaml \
  --checkpoint "${CKPT_DIR}/variant_c_best.pth" \
  --dataset mvsec \
  --write-summary

echo "ablation summary written to experiments/results/tables"
