# Full DSEC Training Record

## Summary

- Start time (UTC): `2026-04-23 16:31:15`
- Environment: `conda activate sdformerflow`
- Backend: `SDFORMER_SNN_BACKEND=cupy`
- Training config:
  `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml`
- Main process PID: `16441`
- MLflow run id: `811c705677b34ae9ae9eb34ca278b48d`
- Log file:
  `/root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_full_dsec_cupy_20260423_163115.log`

## Launch Command

```bash
cd /root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow
source /opt/conda/etc/profile.d/conda.sh
conda activate sdformerflow
export PYTHONPATH=.
export SDFORMER_SNN_BACKEND=cupy
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_amp_lr5e5.yml \
  --path_mlflow file:///root/private_data/work/SDformer/experiments/mlruns
```

## How To Watch Training

```bash
tail -f /root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_full_dsec_cupy_20260423_163115.log
```

To show the most recent 80 lines first:

```bash
tail -n 80 -f /root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_full_dsec_cupy_20260423_163115.log
```

## What We Verified

- GPU training process is attached to PID `16441`
- Process parent is `1`, so it is detached from the current terminal
- Training entered `Epoch 0`
- DataLoader worker processes are running
- GPU memory is in use during training

## Notes

- The config name still contains `torch`, but runtime backend was overridden to `cupy` via environment variable.
- Early startup prints the full model structure, so the log grows quickly at the beginning. This is normal.
