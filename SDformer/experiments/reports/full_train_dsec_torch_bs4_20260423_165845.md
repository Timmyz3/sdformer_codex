# Full DSEC Training Record (BS4)

## Summary

- Start time (UTC): `2026-04-23 16:58:45`
- Environment: `conda activate sdformerflow`
- Training config:
  `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_bs4_fast.yml`
- Main process PID: `26533`
- MLflow run id: `455ad1898a8f47669c3f902c57fda2fe`
- Log file:
  `/root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_full_dsec_torch_bs4_20260423_165845.log`

## What Changed

- switched full training from old `batch_size=1` run to `batch_size=4`
- backend: `torch`
- anomaly detection: off
- `cudnn_benchmark`: on
- `allow_tf32`: on
- dataloader workers: `4`
- `persistent_workers`: on
- `prefetch_factor`: `4`
- `non_blocking`: on

## Launch Command

```bash
cd /root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow
source /opt/conda/etc/profile.d/conda.sh
conda activate sdformerflow
export PYTHONPATH=.
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_full_torch_bs4_fast.yml \
  --path_mlflow file:///root/private_data/work/SDformer/experiments/mlruns
```

## How To Watch Training

```bash
tail -n 80 -f /root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_full_dsec_torch_bs4_20260423_165845.log
```

## Early Runtime Check

- runtime flags printed correctly
- model entered `Epoch 0`
- train loop size became `1838` steps per epoch because `7354 / 4 -> 1838`
- observed GPU state shortly after launch:
  - power draw about `270-282 W`
  - memory about `21.5 GiB`
  - utilization about `73-94%`

## Notes

- The previous full-training process group was terminated before this run started.
- This `bs=4` run is a new training recipe, not a pure drop-in acceleration of the old `bs=1` recipe.
