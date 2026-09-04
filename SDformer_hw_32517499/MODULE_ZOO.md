# SDformerFlow Plug-In Module Zoo

This file tracks small, pluggable research modules that can be toggled independently for ablation.

## Current Hook

The current adapter executes `model.plug_in_modules` after spike encoding and before the upstream SDformerFlow backbone.

Tensor contract for the current hook:

- input: `[B, T, C, H, W]`
- output: `[B, T, C, H, W]`
- optional metadata: `timestep_mask`, `token_mask`, `window_mask`, `head_mask`

## Implemented Modules

### `token_mixer.temporal_shift`

- File: `src/models/modules/token_mixer/temporal_shift.py`
- Inspiration: [TokShift](https://arxiv.org/abs/2208.13273) and temporal shift style video mixing
- Purpose: inject low-cost temporal mixing without adding parameters
- Best use: as a cheap front-end temporal enhancer before sparse gating
- Hardware note: maps to address remap rather than MAC-heavy compute

### `sparse_ops.timestep_budget`

- File: `src/models/modules/sparse_ops/timestep_budget.py`
- Inspiration: adaptive token or timestep budgeting such as [A-ViT](https://arxiv.org/abs/2112.07658)
- Purpose: keep only informative event bins
- Best use: replace fixed `adaptive_t` logic with an explicit, reusable module
- Hardware note: clean fit for block-level skip scheduling

### `sparse_ops.window_topk`

- File: `src/models/modules/sparse_ops/window_pruning.py`
- Inspiration: [HeatViT](https://arxiv.org/abs/2211.08110)
- Purpose: prune inactive spatial windows in a hardware-friendly structured way
- Best use: before token-level pruning, so zero-work windows disappear early
- Hardware note: strong fit for SRAM-friendly local window scheduling

### `sparse_ops.head_group`

- File: `src/models/modules/sparse_ops/head_pruning.py`
- Inspiration: [SpAtten](https://arxiv.org/abs/2012.09852)
- Purpose: approximate head pruning by gating channel groups
- Best use: after patch embedding or inside stage-level feature hooks once deeper backbone injection is added
- Hardware note: natural fit for head-group masking and controller metadata

### `sparse_ops.structured_token`

- File: `src/models/modules/sparse_ops/token_pruning.py`
- Purpose: retain top-k spatial tokens per timestep
- Best use: after window pruning
- Hardware note: already available, now standardized to emit `token_mask`

## Recommended Module Bundles

### Low-Risk Speed Bundle

```yaml
model:
  plug_in_modules:
    - kind: token_mixer
      name: temporal_shift
      shift_div: 2
    - kind: sparse_ops
      name: timestep_budget
      threshold: 0.02
    - kind: sparse_ops
      name: window_topk
      keep_ratio: 0.75
      window_size: [8, 8]
    - kind: sparse_ops
      name: structured_token
      keep_ratio: 0.75
```

### Paper-Oriented Sparse Bundle

```yaml
model:
  plug_in_modules:
    - kind: token_mixer
      name: temporal_shift
      shift_div: 2
    - kind: sparse_ops
      name: timestep_budget
      threshold: 0.02
    - kind: sparse_ops
      name: window_topk
      keep_ratio: 0.75
      window_size: [8, 8]
    - kind: sparse_ops
      name: structured_token
      keep_ratio: 0.75
    - kind: sparse_ops
      name: head_group
      keep_ratio: 0.5
      num_groups: 2
```

The second bundle is mainly a staging point for future deeper feature-hook injection, because the current adapter hook still operates on low-channel event tensors.
