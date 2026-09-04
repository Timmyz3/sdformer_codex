# Architecture

## Target

- RTL target variant: `variant_c`
- Datapath granularity: `8 lanes x 8-bit`
- Control granularity: `stage -> block -> timestep -> window -> head -> token`

## Status

- Current RTL is a minimum synthesizable skeleton.
- The target architecture described below is the paper-facing accelerator plan.
- See [`../../PAPER_CO_DESIGN_PROPOSAL.md`](../../PAPER_CO_DESIGN_PROPOSAL.md) for the full co-design rationale and paper references.

## Modules

- `top.v`: stream wrapper, parameter exposure, module integration
- `controller.v`: valid/ready orchestration, sparse schedule control, perf counters
- `attention_unit.v`: target block for spike-QK, mask-aware window scheduling, and projection fusion
- `token_mixer.v`: post-attention lane-wise linear mixing or stage-local MLP mapping
- `spike_unit.v`: membrane accumulation, threshold compare, reset
- `pe_array.v`: reusable signed MAC lanes for projection, MLP, patch merge, and refinement
- `sram_if.v`: ping-pong buffer shell for on-chip storage

## Target Accelerator Blocks

1. Event and feature DMA
2. Activity reduction and mask generation
3. Window scheduler
4. Head selector
5. Spike-QK engine
6. Projection and MLP PE array
7. Membrane and spike update unit
8. Optional low-resolution flow refinement engine
9. Scratchpad and metadata SRAMs

## Dataflow

1. Input tokens or feature tiles arrive on the packed data bus.
2. `controller.v` consumes explicit timestep/window/head metadata and issues sparse work descriptors.
3. `attention_unit.v` computes only active local windows and active heads.
4. `token_mixer.v` applies stage-local projection or MLP mixing.
5. `spike_unit.v` accumulates membrane state and emits spike bits.
6. `sram_if.v` stores feature, mask, and membrane tiles.
7. Optional refinement reuses `pe_array.v` in convolution mode on low-resolution flow features.

## Scheduling Order

```text
stage
  -> timestep group
  -> window group
  -> head group
  -> token tile
  -> channel tile
```

This order makes sparse skip decisions early and preserves scratchpad reuse for local windows.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `DATA_W` | `8` | Scalar bitwidth |
| `LANES` | `8` | Packed SIMD lanes |
| `ACC_W` | `24` | Accumulator width |
| `THRESHOLD` | `4` | Spike threshold |
| `WINDOW_TOKENS` | `64` | Logical window size for attention tiling |

## Memory Hierarchy

- `SRAM_A`: active window feature tiles
- `SRAM_B`: key/projection weight tiles
- `SRAM_C`: timestep/window/head metadata and masks
- `SRAM_D`: membrane state and optional refinement state

Recommended buffering policy:

- ping-pong feature buffering between DMA and compute
- narrow dedicated metadata SRAM
- stage-local mask caching so the controller does not refetch skip information
