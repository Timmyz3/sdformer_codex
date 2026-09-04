# H9 Design: PSN-ATLIF-Ternary Attention with Shiftmax

## Problem

H6 and H8 introduced ternary Q/K neurons and binary ATLIF in selected high-SOP
layers. They reduced SOPs while keeping AEE close to baseline, but all-parameter
training produced severe AAE degradation. The current leading hypothesis is that
ternary Q/K introduces signed polarity interactions, but the original attention
path does not restore softmax-like row normalization. The result is unstable
attention allocation, which hurts direction-sensitive optical flow.

## Primary Design

H9 should not abandon the current PSN-based baseline. It should continue the
working H-series neuron stack and fuse it with the attention ideas from Bipolar
Self-attention for Spiking Transformers:

1. Keep PSN temporal mixing by preserving each replaced PSN neuron's learned
   weight and bias.
2. Keep ATLIF adaptive thresholds as learnable parameters and keep the
   threshold-scaled spike output.
3. Use ternary Q/K activations to represent negative, zero, and positive membrane
   states in attention.
4. Compute ternary matrix product scores for signed polarity interactions.
5. Apply a Shiftmax-style approximation to recover bounded, low-entropy
   attention allocation without full softmax.
6. Multiply/shift V by the normalized scores.

The implementation should begin as an overlay around SDFormerFlow attention
modules, not as an invasive baseline rewrite.

## Variants

### H9a Shiftmax Compatibility

Start from the H8m neuron stack and add a Shiftmax compatibility gate to the
existing SDFormerFlow QK attention path. This is the first non-redundant H9
experiment: it directly tests whether the H6/H8 AAE collapse comes from missing
attention normalization rather than from the PSN+ATLIF ternary neuron itself.

### H9b Attention Stage/Block Search

Use the H9a compatibility mechanism on selected attention stages or blocks. This
search identifies whether early, middle, or late attention blocks tolerate
ternary+Shiftmax best.

### H9c BSA + H6 Sparse FFN

Start from the best H9b attention subset and search high-SOP sparse modules:

- stage/block FFN binary ATLIF
- downsample binary ATLIF

This tests whether the final sparse story works once attention normalization is
fixed. Attention keeps ternary PSN+ATLIF, while FFN/downsample use binary
PSN+ATLIF for a cleaner hardware story.

### H9d Optional A2OS2A Reference

Use the CVPR 2025 A2OS2A idea as a reference branch:

- Q binary
- K ReLU/nonnegative
- V ternary
- no softmax/scale

This should not be mixed into H9a-H9c. It is a separate attention paradigm used
only for comparison.

## Integration Plan

- `overlay/models/STSwinNet_SNN/atlif_ternary_psn/`: copy of the H8 PSN+ATLIF
  neuron stack, kept local so H9 is an independent experiment.
- `overlay/models/STSwinNet_SNN/bsa_attention.py`: Shiftmax compatibility gate
  and later TMP attention helpers.
- `overlay/models/STSwinNet_SNN/bsa_installer.py`: module installer or forward
  patcher for selected Swin attention blocks.
- `entrypoints/train.py`: baseline training entrypoint with H9 install hook.
- `entrypoints/profile_sops.py`: baseline profiling entrypoint with H9 install
  hook and existing SOP metrics.
- `configs/h9a_shiftmax_compat_*.yml`: short/full configs for H9a.
- `configs/h9b_*stage*_*.yml`: stage/block search configs for H9b.
- `configs/h9c_*ffn*_*.yml`: combined attention plus FFN sparse configs.

## Success Criteria

Primary:

- AAE should be much closer to baseline than H6/H8 all-parameter runs.
- AEE should stay near baseline.
- SOPs should be below the PSN baseline.

Suggested thresholds for a promising valid40 result:

- AEE <= 1.62
- AAE <= 9.0
- SOPs <= 3.30G

## Risks

- SDFormerFlow already uses a custom spiking attention variant, so a literal BSA
  transplant may need shape adaptation.
- Shiftmax may initially increase dense operations in PyTorch even if the
  algorithmic hardware story is efficient.
- If V remains binary PSN while only Q/K are ternary, the implementation may not
  match the paper well enough; this should be tracked explicitly per config.
- If no official BSA source code is available, the H9 implementation is a
  paper-formula reproduction. The exact Shiftmax formula and row-sum bounds must
  be unit-tested before any full training.
