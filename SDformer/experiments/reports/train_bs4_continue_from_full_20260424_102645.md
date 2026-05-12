# BS4 Continue-Training Record

## Summary

- Start time (UTC): `2026-04-24 10:26:45`
- Base checkpoint run id: `66d1fc5322004d59a03c8ab132b11830`
- Config:
  `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_bs4_continue_from_full.yml`
- Main process PID: `80661`
- Log file:
  `/root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_bs4_continue_from_full_20260424_102645.log`

## What This Run Does

- loads the first full-training checkpoint as initialization
- continues training with `batch_size=4`
- uses lower LR `1e-5`
- disables MLflow artifact/metric logging to avoid the previous file-store stall
- still keeps the MLflow tracking URI only for resolving the old checkpoint

## Confirmed At Startup

- `MLflow tracking URI: file:///root/private_data/work/SDformer/experiments/mlruns`
- `[runtime] MLflow logging disabled via SDFORMER_USE_MLFLOW`
- `Model restored from 66d1fc5322004d59a03c8ab132b11830`
- entered `Epoch 0`

## How To Watch

```bash
tail -n 80 -f /root/private_data/work/sdformer_codex/SDformer/experiments/logs/train_bs4_continue_from_full_20260424_102645.log
```

## Notes

- This is not a strict resume with old optimizer state.
- It is a continued optimization run that starts from the old trained weights and uses a new `bs=4` training recipe.
