#!/bin/bash
# T51 short LR sweep: three 600-step ternary warm-start probes from the same ep34
# checkpoint, differing only in backbone LR. Each writes results/t51_lrprobe_<tag>/train.log.
PY=/opt/conda/envs/sdformerflow/bin/python
for spec in "1e4:1e-4" "3e4:3e-4" "1e3:1e-3"; do
  tag=${spec%%:*}
  LR=${spec##*:}
  echo "=== START t51_lrprobe_${tag} lr=${LR} $(date -Is) ==="
  "${PY}" -u /root/t49_ternary_retrain.py \
      --name "t51_lrprobe_${tag}" --epochs 1 --max-steps 600 --backbone-lr "${LR}"
  echo "=== DONE t51_lrprobe_${tag} rc=$? $(date -Is) ==="
done
echo "=== SWEEP COMPLETE $(date -Is) ==="
