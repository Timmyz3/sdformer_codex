# Corrected ATLIF Loader Benchmark

This benchmark uses the corrected ATLIF implementation with learnable activity threshold updates enabled. It runs one short training epoch on `train_backend_benchmark_split_seq.csv` and a tiny validation pass on `valid_backend_benchmark_split_seq.csv`.

## Result Table

| config | status | train samples/s | step time s | max GPU GiB | epoch time s |
| --- | ---: | ---: | ---: | ---: | ---: |
| bs8w4 | 0 | 5.0085 | 1.5973 | 37.077 | 25.56 |
| bs8w8 | 0 | 5.0360 | 1.5886 | 37.077 | 25.42 |
| bs12w4 | 0 | 5.4217 | 2.2133 | 55.399 | 22.13 |
| bs12w8 | 0 | 5.3356 | 2.2490 | 55.399 | 22.49 |
| bs14w4 | 0 | 5.4843 | 2.5528 | 64.560 | 22.97 |
| bs14w8 | 0 | 5.3661 | 2.6090 | 64.560 | 23.48 |
| bs16w4 | 0 | 5.5103 | 2.9037 | 73.721 | 23.23 |
| bs16w8 | 0 | 5.2980 | 3.0200 | 73.721 | 24.16 |
| bs16w2 | 0 | 5.5360 | 2.8902 | 73.721 | 23.12 |
| bs17w4 | 1 | OOM |  |  |  |
| bs18w4 | 1 | OOM |  |  |  |

## Selected Full Training Parameters

Chosen config: `neuron_experiments/E2_exp_atlif/configs/full_corrected_bs16w2.yml`

Reason: `bs16w2` is the fastest successful short-sequence run at 5.5360 samples/s. `bs17w4` and `bs18w4` are not usable because they OOM. `bs16w2` and `bs16w4` have the same peak memory in the benchmark, while `bs16w2` is slightly faster and uses fewer dataloader workers.

Full training command template:

```bash
SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
CONDA_PREFIX=/opt/conda/envs/sdformerflow \
PATH=/opt/conda/envs/sdformerflow/bin:$PATH \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/opt/conda/envs/sdformerflow/bin/python neuron_experiments/E2_exp_atlif/entrypoints/train.py \
  --config neuron_experiments/E2_exp_atlif/configs/full_corrected_bs16w2.yml \
  --path_mlflow '' \
  --save_path 'neuron_experiments/E2_exp_atlif/results/full_corrected_bs16w2_checkpoint_epoch{}.pth'
```
