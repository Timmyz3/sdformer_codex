# H9 PSN-ATLIF-Ternary Bipolar Attention

H9 keeps the H-series mainline: PSN temporal mixing + ATLIF adaptive threshold +
ternary/binary spiking output. The new part is a better attention-side fusion
inspired by Bipolar Self-attention for Spiking Transformers, especially TMP and
Shiftmax.

## Motivation

H6/H8 results show a repeated pattern:

- AEE remains close to the PSN baseline.
- SOPs and firing rate are reduced.
- AAE becomes much worse in all-parameter Q/K ternary runs.

This suggests the model can preserve endpoint magnitude but loses direction
consistency. H9 treats this as an integration problem rather than a neuron-only
problem: ternary Q/K gives polarity, ATLIF gives adaptive sparsity, but attention
scores still need a softmax-like row constraint. BSA proposes:

- ternary matrix product (TMP) for Q/K polarity interactions
- Shiftmax for low-cost attention normalization

## Paper References

Primary:

- Bipolar Self-attention for Spiking Transformers, NeurIPS 2025 spotlight.
  OpenReview: https://openreview.net/forum?id=nG45z7lJ7D

Reference branch:

- Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention
  for Transformer, CVPR 2025.
  CVF: https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Spiking_Transformer_Introducing_Accurate_Addition-Only_Spiking_Self-Attention_for_Transformer_CVPR_2025_paper.html

## Planned Experiments

| Experiment | Attention change | FFN/downsample change | Purpose |
| --- | --- | --- | --- |
| H9a | PSN+ATLIF ternary Q/K + TMP only | baseline PSN | isolate polarity without Shiftmax |
| H9b | PSN+ATLIF ternary Q/K + TMP + Shiftmax | baseline PSN | test whether attention normalization fixes AAE |
| H9c | H9b attention | H6-style PSN+ATLIF binary FFN/downsample | combine attention fix with high-SOP sparse story |
| H9d optional | A2OS2A-style Q binary, K ReLU, V ternary | baseline PSN | separate reference branch, not the H9 mainline |

The mainline H9a-H9c must keep PSN weights/biases and ATLIF learnable
thresholds. It should not replace the neuron with a pure TSN unless that branch
is explicitly labeled as an ablation.

## Isolation Rules

- Do not modify `third_party/SDformerFlow`.
- H9 code lives under `neuron_experiments/H9_bipolar_self_attention`.
- Training/inference still reuse baseline entry semantics through H9 entrypoints.
- All modified modules are installed through an overlay or monkey-patched entrypoint.
- If no official BSA code is available, the implementation must be labeled as a
  paper-formula reproduction and checked with shape/statistics tests before
  full training.

## Metrics

Every promoted run must record:

- AEE
- AAE
- global firing rate
- estimated total SOPs
- layer firing rates
- attention-specific firing/score diagnostics where possible

Baseline comparison:

- PSN baseline epoch59 valid40: AEE 1.584776, AAE 7.501204, firing 0.084961, SOPs 3.6219G.
