# Interfaces

## Stream Interface

- `in_valid`, `in_ready`, `in_last`
- `in_data[LANES*DATA_W-1:0]`
- `out_valid`, `out_ready`, `out_last`
- `out_data[LANES-1:0]`

## Metadata Contract

- Tokens are packed lane-major in little-endian byte order.
- Each lane is a signed fixed-point integer.
- Current RTL still treats masks as implicit zeroed inputs.
- Target accelerator interface should make sparse execution explicit rather than implicit.

## Target Sparse Metadata

The hardware-facing contract should export or stream the following fields per active work item:

- `stage_id`
- `block_id`
- `timestep_id`
- `window_id`
- `head_group_id`
- `token_base`
- `channel_base`
- `timestep_enable`
- `window_enable`
- `head_mask`
- `token_mask`

Recommended semantics:

- `timestep_enable`: skip the entire timestep group when `0`
- `window_enable`: skip the entire local window when `0`
- `head_mask`: packed bits for active heads in the current group
- `token_mask`: packed bits for active tokens in the current local window

This makes the controller responsible for sparse scheduling instead of forcing the datapath to consume zeros.

## Quantization Contract

- Weights/activations: signed `8-bit`
- Membrane/threshold: signed `12-bit`
- Accumulator: signed `24-bit`
- Rounding: nearest
- Clamp: saturating

## Error Budget

- Golden-vs-RTL element error: `<= 1 LSB`
- End-to-end dequantized output drift target: `<= 0.02 EPE`

## Target Golden Export Additions

The Python export path should eventually emit:

- feature tiles
- per-stage quantization scales
- timestep masks
- window masks
- head masks
- token masks

That is required for cycle-accurate or near-cycle-accurate validation of sparse execution.
