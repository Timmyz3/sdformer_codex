# Throughput Tuning Record

## Goal

Check whether the DSEC full-training baseline can better utilize the A800 by tuning:

- SNN backend
- anomaly detection
- dataloader workers
- non-blocking H2D copies
- cuDNN autotune / TF32
- batch size

## Main Findings

### Biggest hidden slowdown

`train_flow_parallel_supervised_SNN.py` had `torch.autograd.set_detect_anomaly(True)` inside every training step.

That is useful for debugging gradients, but it is expensive and should not stay enabled for normal full training.

### Backend finding

On this server, `torch` backend is slightly faster than `cupy` for this code path.

### Batch / loader finding

With runtime tuning enabled, throughput improved a lot.

## Benchmark Setup

- GPU: `NVIDIA A800 80GB PCIe`
- Dataset slice:
  - train first `128` samples
  - valid first `8` samples
- Script:
  - `third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py`

## Results

| config | backend | batch | workers | epoch_time_sec | train_step_time_sec | train_samples_per_sec | max_gpu_mem_gib |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline rerun | cupy | 1 | 0 | 120.40 | 0.9406 | 1.0631 | 5.039 |
| tuned loader/runtime | torch | 1 | 4 | 38.95 | 0.3043 | 3.2858 | 10.790 |
| tuned + larger batch | torch | 4 | 4 | 17.01 | 0.5314 | 7.5272 | 18.553 |
| tuned + larger batch | torch | 8 | 4 | 16.64 | 1.0400 | 7.6922 | 36.722 |
| tuned + larger batch | torch | 16 | 4 | 19.51 | 2.4383 | 6.5619 | 66.862 |

## Practical Interpretation

- `batch=1 -> batch=1, workers=4, runtime tuned`
  - about `3.09x` faster in train samples/sec
- `batch=1 baseline -> batch=4 tuned`
  - about `7.08x` faster
- `batch=1 baseline -> batch=8 tuned`
  - about `7.24x` faster
- `batch=16` is too large for efficiency on this workload

## Recommended Sweet Spots

### Safer path

- `backend=torch`
- `detect_anomaly=False`
- `n_workers=4`
- `persistent_workers=True`
- `prefetch_factor=4`
- `non_blocking=True`
- `cudnn_benchmark=True`
- `allow_tf32=True`
- keep `batch_size=1`

This preserves batch semantics and still gives a large speedup.

### Aggressive throughput path

- same runtime tuning as above
- `batch_size=4` or `batch_size=8`

`batch_size=8` was the fastest in this benchmark, but only slightly above `batch_size=4`, while using much more memory.

## Important Caution

Increasing batch size does not only change speed. It also changes optimization behavior:

- fewer optimizer steps per epoch
- different gradient noise scale
- same LR schedule now corresponds to fewer parameter updates

So `batch_size=4/8` should be treated as a new training recipe, not as a pure engineering acceleration.
