# Paper-Driven HW/SW Co-Design Proposal for SDformerFlow

## Scope

This note translates paper ideas into a concrete upgrade path for this repository.
It focuses on the real bottlenecks in the current codebase rather than proposing a new model from scratch.

## Current Reality Check

- The software wrapper already supports input encoding, timestep masking, token pruning, and LayerNorm-to-RMSNorm replacement in [`src/models/sdformer/backbone.py`](src/models/sdformer/backbone.py).
- The upstream baseline selected by default is `MS_SpikingformerFlowNet_en4`, which already uses a spike-driven Q-K style attention path in [`third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py`](third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py).
- The local attention registry is still descriptor-only; it does not yet replace the internal upstream attention kernel.
- The hardware side is still a skeleton: [`hw/rtl/attention_unit.v`](hw/rtl/attention_unit.v) is a passthrough and the current performance model is still FLOPs-proxy based.

The consequence is simple: the next useful gains must come from deep block-level replacement and explicit sparse scheduling, not from adding more outer-wrapper options.

## Papers Worth Extracting From

| Paper | Transferable idea | Why it matters here | Expected risk |
| --- | --- | --- | --- |
| [SDformerFlow](https://arxiv.org/abs/2407.15801) | Spike-driven Swin-style event optical flow | Confirms the baseline should stay event-native and spike-driven | Low |
| [QKFormer](https://arxiv.org/abs/2403.16552) | Q-K only attention for SNNs | Supports keeping a multiplication-light attention core instead of dense softmax attention | Low |
| [STAtten](https://arxiv.org/abs/2409.19764) | Joint spatiotemporal attention in SNNs | Strong fit for event optical flow, where temporal ordering carries motion cues | Medium |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | IO-aware tiling and SRAM reuse | Necessary for turning window attention into a bandwidth-efficient kernel | Low |
| [HeatViT](https://arxiv.org/abs/2211.08110) | Hardware-friendly token pruning with predictable structure | Better fit than arbitrary sparsity for on-chip scheduling | Low |
| [DynamicViT](https://arxiv.org/abs/2106.02034) | Hierarchical token sparsification | Good software baseline for learning which windows/tokens to skip | Medium |
| [SpAtten](https://arxiv.org/abs/2012.09852) | Cascade token/head pruning plus attention accelerator design | Directly relevant to head/window/token scheduling in hardware | Low |
| [FlowFormer](https://arxiv.org/abs/2203.16194) | Cost-memory based flow refinement | Useful as a lightweight accuracy recovery head after aggressive sparsification | Medium |
| [SEA-RAFT](https://arxiv.org/abs/2405.14793) | Efficient RAFT-style refinement for speed/accuracy balance | Strong candidate for a small refinement head on top of sparse encoder features | Medium |
| [FireFly-T](https://arxiv.org/abs/2503.18018) | Transformer accelerator with 3D-stacked memory and token/head optimizations | Useful reference for system-level memory hierarchy design | Medium |

## Recommended Upgrade Stack

### 1. Deep Attention Injection

Replace the internal attention module of the upstream encoder blocks instead of stopping at config descriptors.

Recommended direction:

- Keep the current spike-driven Q-K path as the base.
- Add a new `temporal_spike_qk` variant that injects local temporal mixing and relative temporal bias inspired by STAtten.
- Apply pruning and skip decisions inside the block, before window partition and projection, not only as zero-masking in the wrapper.

Why this is the right next step:

- It preserves the spike-native nature of SDformerFlow.
- It keeps the hardware mapping simple enough for a real accelerator.
- It gives a publishable difference over the current repo, where local `attention` is only a spec layer.

### 2. Three-Level Sparse Scheduling

Use explicit structured sparsity at three levels:

1. Timestep-level gating
2. Window/token-level gating
3. Head-level gating

Recommended policy:

- `timestep_mask`: computed from event activity or membrane variance
- `window_mask`: computed from pooled activity per local window
- `token_mask`: top-k within active windows only
- `head_mask`: top-k heads per stage, updated less frequently than token masks

This is a better fit than pure unstructured sparsity because it maps cleanly to SRAM banking, DMA bursts, and deterministic controller logic.

### 3. IO-Aware Sparse Window Kernel

Adopt FlashAttention-style tiling ideas, but apply them to spike-driven local windows:

- Keep `K` and positional terms resident in scratchpad for the current window tile
- Stream `Q` tiles and accumulate token scores in-place
- Fuse score generation, masking, projection, and optional membrane update when possible
- Skip zero-work windows entirely instead of feeding zero tokens through the same datapath

This is the main path to a real latency reduction. The current repo already exposes `token_mask` and `timestep_mask`; the missing part is turning them into execution control rather than arithmetic on zeros.

### 4. Accuracy Recovery Head

Aggressive sparsity usually hurts motion boundaries and large displacement.
Add one of the following lightweight heads on top of sparse encoder features:

- a 1-2 iteration recurrent update head inspired by SEA-RAFT
- a small cost-memory refinement head inspired by FlowFormer

Recommendation:

- Use the sparse spike transformer only for feature extraction and coarse flow.
- Run the refinement head at `1/8` or `1/4` resolution.
- Reuse the same PE array for the refinement head if the hardware target is edge or embedded.

### 5. Quantization Strategy

The current `8/8/12` bit setting is a good baseline, but the proposed architecture should move from static export to real QAT/PTQ flow:

- `Q`, `K`, gating scores, and projection weights: signed 8-bit
- membrane and threshold: signed 12-bit
- accumulators: signed 24-bit or wider only where proven necessary
- mask thresholds: per-stage calibrated offline

The key point is that sparsity thresholds and quantization scales must be calibrated together, because pruning decisions are sensitive to activation scale drift.

## Proposed Hardware Accelerator

Name: `SWSA-Flow`

Expansion: Sparse Window Spike Attention accelerator for optical flow.

### Targeted Compute Dominance

The accelerator should primarily target:

- patch embedding / projection layers
- spike-driven window attention blocks
- patch merging and token mixing
- optional lightweight refinement head

The decoder and prediction heads can reuse the same PE array rather than requiring a separate accelerator family.

### Top-Level Blocks

1. `Event/Feature DMA`
2. `Activity and Mask Generator`
3. `Window Scheduler`
4. `Head Selector`
5. `Spike-QK Engine`
6. `Projection and MLP PE Array`
7. `Membrane/Spike Update Unit`
8. `Flow Refinement Engine`
9. `Scratchpad and Metadata SRAM`

### Dataflow

```text
off-chip event tensor / feature maps
  -> activity reduction
  -> timestep/window/head mask generation
  -> active-window scheduler
  -> spike-QK engine
  -> projection / MLP PE array
  -> membrane update
  -> optional low-res flow refinement
  -> off-chip flow pyramid / final flow
```

### Core Scheduling Order

Recommended loop nest:

```text
stage
  -> timestep group
  -> window group
  -> head group
  -> token tile
  -> channel tile
```

Why this order:

- it exposes coarse skip opportunities early
- it minimizes wasted reads for pruned timesteps/windows/heads
- it reuses local `K`/positional state across token tiles

### On-Chip Memory Layout

- `SRAM_A`: active feature window buffer
- `SRAM_B`: key/projection weight tile buffer
- `SRAM_C`: metadata buffer for timestep/window/head masks
- `SRAM_D`: membrane state / refinement state

Recommended buffering policy:

- ping-pong buffers between load and compute
- metadata SRAM sized for one stage worth of masks
- separate narrow SRAM for masks so metadata fetch does not contend with feature fetch

### Arithmetic Mapping

For the spike-driven Q-K path:

- binary or low-bit spike activations should map to AND-popcount or low-cost gated accumulate
- positional terms are added before the spike gate
- projection and MLP stay on signed MAC lanes

For the refinement head:

- use the same MAC array in convolution mode
- keep iteration count fixed and small to preserve deterministic scheduling

## Software Changes Required in This Repo

### Mandatory

1. Replace descriptor-only attention selection with real upstream block injection.
2. Move `timestep_mask` and `token_mask` from preprocessing-only logic into block execution control.
3. Add `window_mask` and `head_mask` to the model output contract and hardware export path.
4. Upgrade profiling from FLOPs proxies to window/head-aware memory and cycle estimates.

### Recommended

1. Add a `variant_d` or `variant_hwaware` config for the paper-ready path.
2. Add calibration scripts that jointly dump quant scales and sparse thresholds.
3. Extend the golden simulator to consume explicit mask metadata rather than implicit zeroed inputs.

## Paper-Ready Contribution Packaging

A coherent paper story can be built around three contributions:

1. A sparse spatiotemporal spike-driven attention block for event optical flow.
2. A structured multi-level scheduler across timestep, window, token, and head dimensions.
3. A hardware accelerator that exploits the same metadata contract end-to-end.

That is stronger than presenting isolated tricks because the software changes and the accelerator reuse the same sparse execution semantics.

## Suggested Experimental Order

1. Reproduce the current baseline exactly.
2. Replace block internals to support explicit window/head/timestep skipping.
3. Add a lightweight refinement head and measure accuracy recovery.
4. Calibrate QAT/PTQ together with sparsity thresholds.
5. Update the performance model and RTL interfaces to consume explicit masks.
6. Implement the first real `attention_unit` around the spike-QK datapath.

## What To Avoid

- Do not replace the whole model with dense FlashAttention-style softmax attention; it breaks the spike-native hardware story.
- Do not use unstructured token sparsity as the main contribution; it is much harder to schedule efficiently in RTL.
- Do not push refinement to full resolution; that will erase most of the hardware savings.
- Do not keep sparsity only as zero-masking in Python; it will not translate into cycle savings.
