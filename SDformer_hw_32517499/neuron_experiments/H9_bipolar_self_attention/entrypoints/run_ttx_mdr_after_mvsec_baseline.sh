#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
BASELINE_EVAL_ROOT="${BASELINE_EVAL_ROOT:-results_inference}"
CONFIG="${CONFIG:-configs/generated/train_mdr_ttx_mvsec_route_fast.yml}"
MLFLOW_URI="${MLFLOW_URI:-file:///root/private_data/sdformer_mlflow}"
ORCH_PID="${ORCH_PID:-1634623}"
SNN_BACKEND="${SNN_BACKEND:-cupy}"
LABEL="${LABEL:-mdr_ttx_from_best_baseline_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="neuron_experiments/H9_bipolar_self_attention/results/${LABEL}"
LOCAL_CKPT_DIR="${LOCAL_CKPT_DIR:-${RESULT_ROOT}/local_ckpts}"

mkdir -p "$RESULT_ROOT" "$LOCAL_CKPT_DIR"

echo "[ttx-mdr] start $(date -Iseconds)"
echo "[ttx-mdr] waiting for baseline MVSEC orchestrator pid=${ORCH_PID}"
while ps -p "$ORCH_PID" >/dev/null 2>&1; do
  ps -p "$ORCH_PID" -o pid,stat,etime,cmd || true
  sleep 300
done

echo "[ttx-mdr] selecting baseline checkpoint from MVSEC rankings"
BEST_INFO="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import math

repo = Path("/root/private_data/work/sdformer_codex/SDformer")
candidates = [
    (41, repo / "results_inference/mvsec_mdr_baseline_epoch41_dt1_full4_20260629_235858/mvsec_ranking.md"),
    (47, repo / "results_inference/mvsec_mdr_baseline_epoch47_dt1_full4_20260629_235858/mvsec_ranking.md"),
]
best = None
for epoch, path in candidates:
    if not path.exists():
        continue
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if not parts or parts[0] in {"sequence", "---"}:
            continue
        try:
            values.append(float(parts[1]))
        except Exception:
            pass
    if len(values) < 4:
        continue
    mean = sum(values) / len(values)
    if best is None or mean < best[1]:
        best = (epoch, mean, path)
if best is None:
    raise SystemExit("no complete epoch41/47 MVSEC ranking found")
ckpt = repo / f"neuron_experiments/H9_bipolar_self_attention/results/mdr_valid_resume_local_ckpts_20260625_164239/checkpoint_epoch{best[0]}.pth"
print(f"{best[0]} {best[1]:.6f} {ckpt}")
PY
)"
read -r BEST_EPOCH BEST_AEE BEST_CKPT <<<"$BEST_INFO"
echo "[ttx-mdr] selected baseline epoch=${BEST_EPOCH} mean_aee=${BEST_AEE} checkpoint=${BEST_CKPT}"

SMOKE_CONFIG="${RESULT_ROOT}/smoke_config.yml"
"$PYTHON_BIN" - <<'PY' "$CONFIG" "$SMOKE_CONFIG"
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg.setdefault("loader", {})["n_epochs"] = 1
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

echo "[ttx-mdr] smoke test one MDR batch $(date -Iseconds)"
(
  cd third_party/SDformerFlow
  SDFORMER_SNN_BACKEND="$SNN_BACKEND" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  KMP_DUPLICATE_LIB_OK=TRUE \
  SDFORMER_MDR_DETECT_ANOMALY=0 \
  SDFORMER_MDR_MAX_TRAIN_BATCHES=1 \
  SDFORMER_MDR_MAX_VALID_BATCHES=1 \
  SDFORMER_MDR_SKIP_MLFLOW_MODEL_LOG=1 \
  SDFORMER_MDR_LOCAL_CHECKPOINT_DIR="$REPO/${RESULT_ROOT}/smoke_ckpts" \
  "$PYTHON_BIN" - <<'PY' "../../${SMOKE_CONFIG}" "$BEST_CKPT" "$MLFLOW_URI"
import runpy
import sys
import torch.multiprocessing as mp

mp.set_forkserver_preload(["torch", "torchvision.extension", "torchvision"])
mp.set_start_method("forkserver", force=True)
import torch
import torchvision.extension
import torchvision

config, checkpoint, mlflow_uri = sys.argv[1:4]
sys.argv = [
    "train_mdr_supervised_SNN.py",
    "--config",
    config,
    "--prev_runid",
    checkpoint,
    "--path_mlflow",
    mlflow_uri,
]
runpy.run_path("train_mdr_supervised_SNN.py", run_name="__main__")
PY
) 2>&1 | tee "${RESULT_ROOT}/smoke.log"

grep -q "\\[H9-MDR\\] installed ATLIFTernaryPSN" "${RESULT_ROOT}/smoke.log"
grep -q "\\[H9-MDR\\] installed Shiftmax attention" "${RESULT_ROOT}/smoke.log"
grep -q "\\[H9-MDR\\] load audit" "${RESULT_ROOT}/smoke.log"

echo "[ttx-mdr] full training $(date -Iseconds)"
(
  cd third_party/SDformerFlow
  SDFORMER_SNN_BACKEND="$SNN_BACKEND" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  KMP_DUPLICATE_LIB_OK=TRUE \
  SDFORMER_MDR_DETECT_ANOMALY=0 \
  SDFORMER_MDR_SKIP_MLFLOW_MODEL_LOG=1 \
  SDFORMER_MDR_LOCAL_CHECKPOINT_DIR="$REPO/${LOCAL_CKPT_DIR}" \
  "$PYTHON_BIN" - <<'PY' "../../${CONFIG}" "$BEST_CKPT" "$MLFLOW_URI"
import runpy
import sys
import torch.multiprocessing as mp

mp.set_forkserver_preload(["torch", "torchvision.extension", "torchvision"])
mp.set_start_method("forkserver", force=True)
import torch
import torchvision.extension
import torchvision

config, checkpoint, mlflow_uri = sys.argv[1:4]
sys.argv = [
    "train_mdr_supervised_SNN.py",
    "--config",
    config,
    "--prev_runid",
    checkpoint,
    "--path_mlflow",
    mlflow_uri,
]
runpy.run_path("train_mdr_supervised_SNN.py", run_name="__main__")
PY
) 2>&1 | tee "${RESULT_ROOT}/train.log"

echo "[ttx-mdr] complete $(date -Iseconds)"
