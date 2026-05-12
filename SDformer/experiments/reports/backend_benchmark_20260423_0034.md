# Backend Benchmark Record

## Goal

Compare `cupy` and `torch` SNN backends on the same machine, model, data slice, and training script.

## Environment

- GPU: `NVIDIA A800 80GB PCIe`
- Conda env: `sdformerflow`
- Script:
  `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py`
- Benchmark config:
  `third_party/SDformerFlow/configs/train_DSEC_supervised_SDformerFlow_en4_backend_benchmark.yml`

## Dataset Slice

- Train list: first `128` samples from `train_split_seq.csv`
- Valid list: first `8` samples from `valid_split_seq.csv`

Generated files:

- `data/Datasets/DSEC/saved_flow_data/sequence_lists/train_backend_benchmark_split_seq.csv`
- `data/Datasets/DSEC/saved_flow_data/sequence_lists/valid_backend_benchmark_split_seq.csv`

## Commands

### CuPy

```bash
cd /root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow
source /opt/conda/etc/profile.d/conda.sh
conda activate sdformerflow
export PYTHONPATH=.
export SDFORMER_SNN_BACKEND=cupy
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_backend_benchmark.yml \
  --path_mlflow file:///root/private_data/work/SDformer/experiments/mlruns
```

Log:

- `/root/private_data/work/sdformer_codex/SDformer/experiments/logs/backend_benchmark_cupy_20260423_0034.log`

### Torch

```bash
cd /root/private_data/work/sdformer_codex/SDformer/third_party/SDformerFlow
source /opt/conda/etc/profile.d/conda.sh
conda activate sdformerflow
export PYTHONPATH=.
export SDFORMER_SNN_BACKEND=torch
python train_flow_parallel_supervised_SNN.py \
  --config configs/train_DSEC_supervised_SDformerFlow_en4_backend_benchmark.yml \
  --path_mlflow file:///root/private_data/work/SDformer/experiments/mlruns
```

Log:

- `/root/private_data/work/sdformer_codex/SDformer/experiments/logs/backend_benchmark_torch_20260423_0034.log`

## Results

### CuPy

- backend: `cupy`
- wall time: `196.580 s`
- epoch_time_sec: `174.91`
- train_step_time_sec: `1.3665`
- train_samples_per_sec: `0.7318`
- valid_time_sec: `7.55`
- valid_step_time_sec: `0.9432`
- max_gpu_mem_gib: `5.039`

### Torch

- backend: `torch`
- wall time: `192.472 s`
- epoch_time_sec: `170.98`
- train_step_time_sec: `1.3358`
- train_samples_per_sec: `0.7486`
- valid_time_sec: `6.91`
- valid_step_time_sec: `0.8636`
- max_gpu_mem_gib: `5.032`

## Simple Comparison

- `torch` train throughput vs `cupy`: about `+2.3%`
- `torch` validation step time vs `cupy`: about `-8.4%`
- GPU memory difference is negligible

## Interpretation

On this server and this code path, `torch` is slightly faster than `cupy`.
That means switching the full training backend from `cupy` to `torch` is reasonable if the goal is shorter wall-clock time.
