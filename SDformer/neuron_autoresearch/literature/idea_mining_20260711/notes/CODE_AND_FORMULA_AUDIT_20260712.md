# Code and Formula Audit Update (2026-07-12)

## Scope

This note records mechanisms checked against full papers or official repositories after the H67-H71 queue was frozen. It separates what the cited method actually implements from a possible TTX adaptation. A proposed adaptation is not described as a reproduction.

## 1. EEMFlow (CVPR 2024): useful motion prior, expensive direct transplant

Primary assets:

- Paper: `papers/pdfs/Luo_EEMFlow_CVPR_2024_paper.pdf`
- Official repository: `repos/EEMFlow/`
- Core code: `repos/EEMFlow/model/EEMFlow/EEMFlow.py`
- Dense-flow/CDC code: `repos/EEMFlow/model/EEMFlow/EEMFlow+.py` and `cdc_utils.py`

### What the method actually does

`Correlation(max_displacement=4)` first computes a full 9x9 spatial correlation for each location. `EEMFlow.py` then applies a hard-coded 49-index mask; `EEMFlow+.py` uses 53 indices. The 49-offset set keeps Manhattan radii 0/1/2/3/5/7 and removes radii 4/6/8; the 53-offset variant uses a different checkerboard-like set. The base network builds three feature scales, average-pools all of them to a common coarse resolution, computes three masked cost volumes, decodes three flows, then fuses them. The dense-flow variant performs coarse-to-fine flow prediction, feature warping and CDC at multiple scales.

CDC is not a scalar confidence gate. Its official code concatenates the first feature with a warped second feature, runs a dense convolutional estimator, predicts a 2-D intermediate flow and sigmoid mask, and blends a warped initial flow with the initial flow. This requires warp/grid-sample, several convolutions and dense flow state.

### TTX transfer verdict

Directly copying EEMFlow correlation into all12 TTX is rejected as the next experiment: 49/53 QK comparisons per token plus halo reads are far beyond H60 and would invalidate the current low-cost attention story. Copying CDC is also rejected because it changes decoder/refinement structure rather than the unified attention formula.

The transferable idea is narrower: a **Fixed Sparse-Offset Match TTX** may use a pre-registered small set such as center plus four axial offsets, with one identical set in all 12 blocks. This is inspired by EEMFlow's near/far sparse matching but is a new adaptation, not EEMFlow reproduction. Before implementation it needs an exact aggregation rule, Swin shift/boundary semantics, K-halo traffic model, and a comparison count that remains competitive with H60.

Status: incubation only; no H number and no training slot yet.

## 2. DAR-TR-PEFT (CVPR 2025): not a deployment-free attention regularizer

Primary assets:

- Official repository: `repos/DAR-TR-PEFT/`
- Block implementation: `src/Models/backbones/dinov2_vit_l_ft_dar.py`
- Router: `src/Models/finetunes/msk_gen.py`
- Adapter/compensator: `src/Models/finetunes/dar.py`
- Ratio loss: `src/Models/losses/reg_loss.py`

### What the method actually does

Each transformer block still executes attention for the full token sequence. A learned 3x3 convolutional router uses Gumbel-sigmoid to select tokens for the MLP. Training masks MLP outputs; inference gathers selected tokens, runs the MLP, and scatters results back. A parallel adapter plus spatial depthwise-convolution compensator updates the full patch sequence. `AdaLoss` only pushes the average token selection ratio toward a target; it is not an attention-diversity objective by itself.

### TTX transfer verdict

The direct method adds a router, Gumbel/threshold control, adapter parameters, depthwise convolution and irregular gather/scatter. It targets MLP token reduction, not the attention-score deficiency of H60. It therefore conflicts with the current goal of finding a cleaner unified attention replacement and duplicates the broad density-bypass direction already represented more simply by H70/Delta-locality work.

Status: not queued. It remains a citation for why pixel-level tasks should preserve output token topology, but it is not currently a software-mainline candidate.

## 3. Updated candidate priority

1. Finish H67-H71 full30 and valid825 before adding another long run.
2. Keep Fixed Sparse-Offset Match TTX in incubation, conditional on an operation/bandwidth model showing a bounded attention-block redesign.
3. Do not queue full EEMFlow correlation, CDC, or DAR-TR as TTX replacements.
4. Continue deep reading for a training-time mechanism that improves H60 attention selectivity while leaving deployment arithmetic dyadic; such a mechanism would have better hardware fit than an additional dynamic router.
