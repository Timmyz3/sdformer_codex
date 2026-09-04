# REPORT

## Phase 0

### Completed

- Rebuilt the repository around the optical-flow `SDformerFlow` baseline.
- Added `third_party/SDformerFlow` as a git submodule.
- Added unified configs, scripts, Python adapters, profiling hooks, and RTL skeleton directories.

### Baseline Reference

- Upstream baseline: `SDformerFlow`
- Commit: `13088516440ab3faba4142c986d162cf5dd7c299`
- Primary dataset target: `DSEC-Flow`
- Generalization dataset target: `MVSEC`

## Phase A

### SDformerFlow Topology Summary

- Input event representation: voxel or count volumes over `num_bins`
- Temporal semantics: `T = num_steps` in spiking blocks, configured through `spiking_neuron.num_steps`
- Encoder: spatiotemporal Swin-based spiking transformer with multi-stage feature hierarchy
- Decoder: multi-resolution U-Net style convolutional decoder
- Output: multi-scale optical-flow predictions summed over time and upsampled to image resolution

### Key Tensor Shapes

- Raw event voxel batch: `[B, num_bins, H, W]` before polarity split in upstream scripts
- Polarity-expanded voxel batch: `[B, num_bins, 2, H, W]`
- Spiking encoder internal state: `[T, B, C, H, W]`
- Tokenized Swin feature maps: stage dependent 3D windows over `[T, H, W]`
- Final flow prediction: `[B, 2, H, W]`

### Reproduction Path

Run:

```bash
bash scripts/run_train.sh configs/sdformer_baseline.yaml --output-dir experiments/logs/train
bash scripts/run_eval.sh configs/sdformer_baseline.yaml --checkpoint experiments/logs/train/sdformer_baseline_best.pth
```

The local adapter stack builds the upstream model inside `src/models/sdformer/backbone.py`, runs local train/eval entrypoints, and stores summaries under `experiments/results/`.

### Current Validation Status

- Code integration complete
- Local runtime validation blocked because this machine currently has no working `python`/`pip`
- RTL validation blocked because this machine currently has no `iverilog`/`yosys`

## Phase B

### Pluggable Module Inventory

| Module family | Interface key | Replace point | Hardware rationale |
| --- | --- | --- | --- |
| Window/local spike attention | `model.attention.type` | Swin attention/window config | Bounded token count and SRAM-friendly tiling |
| Spike encoding | `model.spike_encoder.type` | Event voxel preprocessing | Reduces inactive timesteps and input bandwidth |
| Normalization / neuron | `model.norm.type`, `model.neuron.type` | Spiking blocks | Low-cost normalization and tunable threshold dynamics |
| Structured sparsity | `model.sparsity.*` | Token/head/channel gating | Exportable masks and deterministic skip behavior |

### Next Steps

- Connect `variant_a/b/c` to deeper upstream module replacement points
- Add experiment runners that emit DSEC and MVSEC tables from real checkpoints
- Implement fixed-point export and RTL golden-vector flow
