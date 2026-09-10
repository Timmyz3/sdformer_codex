# 03 — Algorithm-native hardware (our modifications, not generic SNN)

Two idea paths were allowed. This file is path (1): mechanisms that exist **because we changed the algorithm**.

## A. Motion-XOR is not QK GEMM

Frozen deploy leaf: `rtl` `h67_motionxor_score_q7` (also synthesized as `H67_MOTIONXOR_SCORE_Q7` in the hw tree).

```
score = round_even( 128 * (overlap + same_zero/64 + motion_xor/4) / 32 )
overlap    = popcount(Q AND K)
motion_xor = popcount(K XOR K_peer)     # temporal peer, not Q
same_zero  = co-silence
attn       = gate ⊙ K                    # K as V
α          = 0.125                       # frozen H67, not α-XNOR’s 0.3/0.5
```

Hardware translation:

1. Three masks on the same aligned Q/K/K_peer bits.
2. `/64` and `/4` are shifts.
3. No MAC in the score leaf.
4. K_peer must be a **shadow bank** from t0, not a second DRAM fetch of a new tensor.
5. C1’s XOR (conv source-mask residual) is a **different XOR**. Do not conflate.

Related-work knives that must appear in any paper:

- FireFly-T: AND-PopCount of Q and K. No K_peer XOR.
- Bishop AAC: AND+accumulate of QK^T. No motion term.
- α-XNOR: co-silence α, classification, no temporal-peer XOR, no RTL. Our α=0.125 ≠ their peak.
- Comperity: XOR for GeMM product reuse.
- SDformerFlow published: SDSA or linear QK, **not** this formula.

## B. ATLIF output is threshold/binary, not T=10 analog

Inventory (do not collapse the two twelves):

- 105 installed ATLIF wrappers.
- 12 `sn2_q` never called.
- 93 invoked (48 T=2, 45 T=10).
- 12 invoked `attn_sn` unused by proj under fixed normal inference → 81 graph-live (36 T=2, 45 T=10).
- All 93 captured outputs **binary**.

Hardware translation:

- Membrane, leak, threshold compare, reset stay **inside** the neuron island (T=10 INT state).
- Island boundary = 1-bit fire (+ optional sign protocol downstream, not analog payload).
- Post-neuron Conv/FC/attention = spike × weight = **add/sub**.
- Online homeostatic θ update is **off** at inference (checkpoint-static threshold). Do not build a runtime θ datapath for the frozen identity.

Grok Bot HBG-RP `{g, int8 p}` **contradicts** this capture. If someone later trains a non-absorbable amplitude, that is a **new algorithm Pareto**, not ep34.

## C. Mixed horizon T_w=2 vs T_snn=10

Attention windows are T=2. Neurons that need membrane recurrence are T=10. LoAS FTP and Bishop TTB assume **uniform T**.

Hardware translation:

- Do not pack T=10 analog (or 10 spike bits) into the attention fabric.
- Attention fiber width 2; membrane engine width 10; rate-match on the fire bit.
- Chen 28 nm already reconfigures T=4/2/1 for classification Spikformer. Our claim is **asymmetric roles on one event-OF encoder**, not “we invented mixed T”.

## D. Event-OF data properties that are not generic ImageNet SNN

From P0 (label as directional; re-census on ep34):

| Property | Why it is OF / event, not ImageNet SNN |
|---|---|
| K-motion zero ~76% | Temporal peer of optical flow; XOR term often constant 0 |
| Q∨K toggle 3.70% | Scene mostly static between T=2 slices |
| Paired scores equal ~97.5% | Score memo, not recompute |
| Dirty run length ~4.14 | Motion spatially clustered |
| K-zero token ~84% | K-as-V gating |
| No correlation volume | C12 forbids RAFT cost-volume island |

## E. What “hardware for our algorithm” is **not**

- Not wrapping Prosperity around bottleneck conv and calling it co-design.
- Not renaming TSBG after noticing K is reused as V.
- Not empty-window skip (camera already sparse).
- Not claiming Shiftmax removes all multiplies in gating (`gate × weight` still exists in deploy subset).
