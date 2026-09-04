# H65: Uniform Signed Hamming Linear Attention

Status: completed negative at step20; no promotion
Scope: DSEC, all 105 symmetric ternary ATLIF wrappers, all 12 attention blocks

## Motivation

H63 direct Shiftmax and H64 centered ATLIF failed their activity gates. The remaining literature-backed no-gate-K candidate is SpikeVideoFormer's Hamming linear attention, for which this repository previously observed partial-scope valid40 AEE around 1.63. That evidence does not cover the current full-network constraint, so H65 is a new controlled test rather than a claimed positive result.

## Operator

```text
Q,K in {-1,0,+1}, shape [N,D]
M = K^T K_value                 # [D,D]
Y = Q M / (2D)                  # [N,D]
```

All dynamic products involving signed events map to conditional add/subtract. There is no Shiftmax, gate×K, SC branch, stage mixture, Kmag, or target-rate controller. K is reused as the value stream because the baseline block has no live V projection.

## Gate

Warm start: all-ternary + TX epoch29. Run exactly 20 train steps and valid1. Stop unless step20 activity is below 20% and valid1 AEE is below 2.2 with finite loss. No LR sweep, G sweep, or full training is allowed before that gate.

## Hardware cost

The price for preserving full channel rank is a `D x D` signed accumulator matrix per active head context (`D=32`, 1024 accumulators if fully parallel). A serialized implementation can reduce area but costs `O(ND^2)` add/sub operations. H65 is therefore accuracy-first and higher-risk than TTX/STC; it is acceptable only if the DSEC result materially exceeds direct Shiftmax.
