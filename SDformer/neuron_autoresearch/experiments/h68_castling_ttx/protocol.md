# H68 Castling-TTX protocol

## Hypothesis

H66a full matrix carries useful training-time pairwise context but is unsuitable
for deployment. A Castling-ViT-style auxiliary can guide H60 and disappear before
inference, retaining the existing all12 TTX hardware.

## Frozen design

- Start: all-binary TTX epoch2.
- Neuron: 105 one-sided binary official ATLIF wrappers.
- Deployed attention: 12 identical H60 dyadic TTX blocks.
- Training-only auxiliary: binary alpha-XNOR `N x N`, Shiftmax, `weights@K`.
- Output blend coefficient: linear `0.5` at step 0 to `0` at step 360.
- No new trainable parameter; no coefficient sweep.
- Motion XOR, SC, Kmag, target-rate, and native QKFormer carrier are disabled.

## Gates

1. Run 360 steps once from TTX epoch2.
2. Evaluate the same checkpoint with the explicit deployment config, batch1.
3. Require `checkpoint_overlay_keys=210`, missing=0, unexpected=0, 105 ATLIF,
   12 attention modules, and reported Castling auxiliary weight 0 in eval.
4. Promote valid40 only if AEE <=1.65 and AAE <=20.
5. Standard valid825 is required before any paper claim.

The profiler's neuron SOP estimate does not include training-only matrix operations.
Those operations are excluded from deployment energy only after the deployment audit
proves that the auxiliary branch is disabled.

