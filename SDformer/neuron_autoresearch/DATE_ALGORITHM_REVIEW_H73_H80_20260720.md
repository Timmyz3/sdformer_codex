# DATE Algorithm Review: H73-H80

Date: 2026-07-20

## Reviewer Summary

The eight labels do not represent eight paper-strength ideas. They form one carrier-free local
binary Match-Code operator plus support, cost-aggregation, channel-grouping, and assignment
variants. DATE needs one co-designed mechanism with measured hardware benefit, not an exhaustive
collection of attention formulas. Full30 is therefore retained only for H73, the smallest
representative that isolates the central Match-Code claim.

## Candidate Decisions

| ID | Mechanism | Novelty risk | Hardware fit | DATE decision |
|---|---|---|---|---|
| H73 DE9 | Separate active/silent alpha-XNOR evidence over cross-time 3x3 offsets, two Shiftmax9 descriptors, static per-head codebook output, no K/V carrier | Medium | Strong, bounded 9-offset engine | **Finish full30** as the sole Match-Code representative |
| H74 MC49 | One Shiftmax over 49 selected 9x9 offsets followed by a 49-row codebook | High: support-size expansion, close to ordinary local correlation-volume search | Weak: address/halo/codebook cost grows sharply | **Do not train**; analytical support-size ablation only |
| H75 AX17 | Horizontal/vertical 17-offset support instead of 3x3/49 | High: support shape is an engineering choice | Medium | **Do not train**; analytical support-shape ablation only |
| H76 PC9 | Fixed 4/2/1 same-time 3x3 smoothing of each displacement score plane | Very high: standard local cost aggregation | Medium | **Do not train** |
| H77 LC4 | Per-head dyadic weights over n11/n10/n01/n00 binary contingencies | Medium-high: a learnable Boolean matching metric, but algebraically reducible to match/activity terms | Strong | **Do not full30 now**; retain only as a future mechanism ablation if H73 becomes the mainline |
| H78 G4 | Split D=32 into four fixed 8-lane groups, produce four Shiftmax9 distributions and project a 36-D descriptor | High: grouped correlation/multi-head refinement is established | Weak-medium: four normalizers and larger descriptor | **Do not train** |
| H79 CF10 | Add a null/dustbin candidate from top-2 margin and query activity, with fixed-zero output codeword | Medium-high: dustbin and matchability are established in RCM/LightGlue | Medium | **Do not train** as a standalone contribution |
| H80 DN9 | Product of source-row and destination-incoming Shiftmax assignments | Very high: bounded local dual-softmax adaptation of LightGlue/Efficient LoFTR | Weak: second normalization, scatter/banking, gate product | **Do not train** |

## What H73 Can Claim

H73 is defensible only as a carrier-free binary correspondence operator: it converts cross-time
binary matching evidence into a fixed-size displacement descriptor and reads out static codewords,
removing the dynamic K/V carrier. It should not be presented as inventing local correlation or a
cost volume. A positive DATE claim requires all of the following:

1. beat or match H67 under standard valid825 and MVSEC;
2. report attention-inclusive operations, SRAM traffic, latency, area, and energy;
3. compare H60/H67/H73 under the same all12 ATLIF105 topology;
4. demonstrate strict quantized/RTL-equivalent inference;
5. explain why separate active/silent evidence and static codewords matter through ablation.

If H73 does not beat H67 or deliver a clear PPA advantage, it remains a negative design-space
result. The safer DATE algorithm line is H67 Motion-XOR TTX combined with the TTB/Exact-Delta
hardware schedule.

## Resolution Review

The current 288x384 experiments are controlled architecture search, not the final DSEC protocol.
They share the same NB0/TTX checkpoint, split, crop, and window `[2,9,9]`, so differences isolate
the attention mechanism. The current hardware and several Match-Code helpers assume
`T=2,H=W=9,N=162` explicitly.

The SDformerFlow paper uses 480x640 full-resolution fine-tuning with `[2,15,15]`, giving 450 tokens
per window. Moving every candidate to that protocol would change the attention storage, address
generation, Shiftmax width, positional encoding, batch size, and hardware tile. It is not a fair
or economical search protocol.

After the final architecture freezes, run full-resolution fine-tuning only for NB0 and the final
candidate. Two claims must remain separate:

- hardware-consistent result: 480x640 with window9, preserving the 162-token tile;
- SDformerFlow-protocol result: 480x640 with window15 plus official test submission, requiring a
  parameterized 450-token hardware configuration.

Do not compare either local validation result directly with the official hidden-test AEE/AE.

## Author Override: H79/H80 Exploratory Runs

After the novelty review, the author explicitly requested empirical H79 and H80 runs. The review
assessment is unchanged: neither mechanism is pre-accepted as a standalone contribution. They are
restored only as controlled assignment-mechanism candidates after H73, with the following rules:

- H79 and H80 independently warm-start from the same frozen TTX epoch2 checkpoint; they are not
  sequential fine-tunes of one another.
- Both use batch4 with gradient accumulation2, full30, trained strict-load audit, and the same eight
  pre-registered valid825 checkpoints.
- H79 is interpreted as a confidence/null-assignment ablation and H80 as a bidirectional assignment
  ablation. A favorable metric alone does not remove their prior-art/complexity concerns.
- No H74-H78 training is restored. Full-resolution training remains gated on selection of the final
  architecture rather than being applied to both exploratory variants.
