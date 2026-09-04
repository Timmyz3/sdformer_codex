# Performance Model

## Cycle Model

- Current model:
  - `load_cycles = ceil(tokens / LANES)`
  - `attention_cycles = load_cycles`
  - `token_mixer_cycles = load_cycles`
  - `spike_cycles = load_cycles`
  - `store_cycles = ceil(outputs / LANES)`
  - `total_cycles = load + attention + token_mixer + spike + store + controller_overhead`

## Target Sparse Cycle Model

For the paper-facing accelerator, the model should expand to:

```text
active_timesteps = total_timesteps * active_timestep_ratio
active_windows = total_windows * active_window_ratio
active_heads = total_heads * head_keep_ratio
active_tokens = window_tokens * token_keep_ratio

load_cycles =
  ceil(active_timesteps * active_windows * active_tokens / LANES)

qk_cycles =
  ceil(active_timesteps * active_windows * active_heads * active_tokens * head_dim / QK_LANES)

proj_cycles =
  ceil(active_timesteps * active_windows * active_tokens * embed_dim / MAC_LANES)

mlp_cycles =
  ceil(active_timesteps * active_windows * active_tokens * mlp_dim / MAC_LANES)

mask_cycles =
  ceil(active_timesteps * active_windows / MASK_LANES)

refine_cycles =
  refinement_iters * ceil(refinement_tokens * refinement_dim / MAC_LANES)

total_cycles =
  load_cycles + qk_cycles + proj_cycles + mlp_cycles + spike_cycles +
  mask_cycles + refine_cycles + store_cycles + controller_overhead
```

## Sparsity Terms

- `active_timestep_ratio`: fraction of timesteps not early-exited
- `active_window_ratio`: fraction of windows not skipped
- `token_keep_ratio`: fraction of tokens surviving structured pruning inside active windows
- `head_keep_ratio`: fraction of active heads per stage
- Effective cycles:

```text
effective_cycles =
  total_cycles *
  active_timestep_ratio *
  active_window_ratio *
  token_keep_ratio *
  head_keep_ratio
```

## Energy Model

- `E_total = E_mac + E_sram + E_dram + E_ctrl`
- `E_mac` uses lane MAC/popcount counts
- `E_sram` uses read/write bytes from profiling CSV and metadata SRAM accesses
- `E_dram` is reported separately and can be omitted for fully on-chip sweeps
- `E_ctrl` must include sparse scheduler overhead once masks become explicit metadata

## Area Model

- Logic area: from `yosys` cell report
- SRAM area: estimated from feature, metadata, and membrane buffers
- Total area = logic + SRAM estimate

## What The Profiler Should Export Next

The Python profiler should eventually export at least:

- `active_timestep_ratio`
- `active_window_ratio`
- `token_keep_ratio`
- `head_keep_ratio`
- `feature_bytes`
- `metadata_bytes`
- `estimated_qk_ops`
- `estimated_proj_ops`
- `estimated_refine_ops`

Without those fields, the hardware model will keep overstating work reduction from sparsity.
