#!/usr/bin/env bash
# Resume interrupted 11lite qkonly full30 from latest good checkpoint, then valid825.
set -euo pipefail

REPO=/root/private_data/work/sdformer_codex/SDformer
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
PY=/opt/conda/bin/python3
RUN_DIR="${1:-${EXP}/results/nts11lite_u12_qkonly_w720_fastlr_full30_bs8_20260615_052324_setsid}"
CONFIG_SRC="${EXP}/configs/generated/nts11lite_u12_qkonly_w720_fastlr_full30.yml"
CONFIG_RESUME="${RUN_DIR}/config_resume_skip_state.yml"
LOG="${RUN_DIR}/pipeline_resume.log"

mkdir -p "${RUN_DIR}"
exec >>"${LOG}" 2>&1
echo "=== 11lite qkonly resume start $(date -Is) ==="
echo "run_dir=${RUN_DIR}"

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"

RESUME=$("${PY}" - <<'PY' "${RUN_DIR}"
import sys
from pathlib import Path
run = Path(sys.argv[1])
ckpts = sorted(
    int(p.stem.replace("checkpoint_epoch", ""))
    for p in run.glob("checkpoint_epoch*.pth")
    if "state_dict" not in p.name
)
if not ckpts:
    raise SystemExit("no checkpoint to resume")
print(run / f"checkpoint_epoch{ckpts[-1]}.pth")
PY
)
echo "resume=${RESUME}"

"${PY}" - <<'PY' "${CONFIG_SRC}" "${CONFIG_RESUME}"
import sys
from pathlib import Path
import yaml
src, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg.setdefault("runtime", {})["skip_state_save"] = True
cfg["note"] = str(cfg.get("note", "")) + "\nresume: skip_state_save to reduce disk."
out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"config_resume={out}")
PY

echo "=== resume train full30 ==="
"${PY}" -u "${EXP}/entrypoints/train.py" \
  --config "${CONFIG_RESUME}" \
  --prev_runid "${RESUME}" \
  --resume True \
  --save_path "${RUN_DIR}/checkpoint_epoch{}.pth"

echo "=== train done $(date -Is) ==="
ls -la "${RUN_DIR}"/checkpoint_epoch*.pth || true

echo "=== standard valid825 ==="
"${PY}" -u "${EXP}/entrypoints/run_h9_standard_valid825_eval.py" \
  --config "${CONFIG_SRC}" \
  --run-dir "${RUN_DIR}" \
  --epoch 9 --epoch 14 --epoch 19 --epoch 24 --epoch 28 --epoch 29

echo "=== valid825 done $(date -Is) ==="
echo "=== 11lite qkonly resume pipeline complete ==="