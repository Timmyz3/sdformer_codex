# Run Commands

## full_bs8w8_20260425_024413

```bash
CONDA_PREFIX=/opt/conda/envs/sdformerflow \
PATH=/opt/conda/envs/sdformerflow/bin:$PATH \
SDFORMER_USE_MLFLOW=0 \
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/E2_exp_atlif/entrypoints/train.py \
  --config neuron_experiments/E2_exp_atlif/configs/full_bs8w8.yml \
  --save_path ../../neuron_experiments/E2_exp_atlif/results/full_bs8w8_checkpoint_epoch{}.pth
```

- PID: `215441`
- Log: `neuron_experiments/E2_exp_atlif/results/full_bs8w8_train_20260425_024413.log`
- Config: `neuron_experiments/E2_exp_atlif/configs/full_bs8w8.yml`

Stopped after loader benchmark showed `bs12w8` was faster.

## full_bs12w8_20260425_025854

```bash
CONDA_PREFIX=/opt/conda/envs/sdformerflow \
PATH=/opt/conda/envs/sdformerflow/bin:$PATH \
SDFORMER_USE_MLFLOW=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/sdformerflow/bin/python \
  neuron_experiments/E2_exp_atlif/entrypoints/train.py \
  --config neuron_experiments/E2_exp_atlif/configs/full_bs12w8.yml \
  --save_path ../../neuron_experiments/E2_exp_atlif/results/full_bs12w8_checkpoint_epoch{}.pth
```

- PID: `224250`
- Log: `neuron_experiments/E2_exp_atlif/results/full_bs12w8_train_20260425_025854.log`
- Config: `neuron_experiments/E2_exp_atlif/configs/full_bs12w8.yml`
- Loader benchmark: `neuron_experiments/E2_exp_atlif/results/loader_benchmark_20260425.md`
