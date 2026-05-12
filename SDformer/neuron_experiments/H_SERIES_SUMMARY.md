# H-series neuron fusion experiment summary

Updated: 2026-05-12

Baseline reference:

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 PSN baseline epoch59 | valid40 | 0.084961 | 3.6219G | 1.584776 | 7.501204 |

## H1 Hardware sparse

Goal: early hardware-friendly sparse neuron/gating prototype.

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 epoch19 | valid40 | 0.063626 | 2.7124G | 2.666143 | n/a |

Read: strong SOP reduction but accuracy drop is too large. Not the current main path.

## H2 Adaptive ternary PSN

Goal: first direct adaptive ternary PSN fusion attempt.

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2 epoch19 | valid40 | 0.203727 | 8.6849G | 1.794888 | 8.848582 |

Read: ternary expansion made activity/SOPs much denser. Abandoned as a direct path.

## H3 Official ATLIF-PSN

Goal: isolate official ATLIF-style adaptive threshold with PSN, mostly Q/K attention replacement.

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H3f epoch29 | valid40 | 0.081477 | 3.4734G | 1.585315 | 8.433778 |
| H3h Q/K high-SOP short | valid40 | 0.082442 | 3.5145G | 1.706355 | 8.907388 |

Read: official ATLIF path is viable, modest SOP reduction, AEE close to baseline in H3f. This is the foundation for later H4-H8.

## H4 ATLIF ternary PSN and controls

Goal: fuse official ATLIF threshold update with ternary output in attention Q/K. H4 controls also tested whether simply turning off Q/K explains the result.

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H4h fusion epoch29 | valid40 | 0.080117 | 3.4154G | 1.560816 | 8.364778 |
| Q/K-off control | valid40 | 0.076818 | 3.2748G | 1.622467 | 7.875712 |

Read: H4 gives the first useful fusion proof. It can reduce SOPs with good AEE. Q/K-off control shows attention sparsity alone can save SOPs, but H4 is not merely a hard deletion story.

## H5 High-SOPs ternary expansion

Goal: expand H4 beyond Q/K to high-SOP layers such as proj, stage0 MLP, downsample.

Short probes only:

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H5a Q/K + proj ternary | valid10 | 0.108943 | 4.6442G | 1.061316 | 6.269898 |
| H5b + stage0 MLP ternary | valid10 | 0.171391 | 7.3064G | 1.064775 | 6.315249 |
| H5c + downsample ternary | valid10 | 0.180286 | 7.6856G | 1.137244 | 6.704098 |

Read: signed ternary in non-attention high-SOP layers makes activity much denser. Not suitable for full training.

## H6 Attention ternary + binary high-SOPs

Goal: keep ternary only where signed information may help, and use binary ATLIF for FFN/downsample to avoid H5 densification.

Mechanism:

- Q/K attention: PSN + ATLIF + ternary
- stage0 FFN + stage0/stage2 downsample: PSN + ATLIF + binary

Short probe:

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H6a short | valid10 | 0.086604 | 3.6919G | 1.049870 | 6.106936 |
| H6b includes proj ternary | valid10 | 0.106058 | 4.5213G | 1.061870 | 6.567753 |

Full frozen/ATLIF-only run:

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H6a frozen epoch11 | valid40 | 0.077434 | 3.3010G | 1.553494 | 8.200176 |
| H6a frozen epoch19 | valid40 | 0.075049 | 3.1994G | 1.614310 | 8.376308 |
| H6a frozen epoch29 | valid40 | 0.071159 | 3.0335G | 1.628533 | 8.709481 |

Full all-params run:

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H6a all-params epoch29 | valid40 | 0.064930 | 2.7680G | 1.594698 | 21.636660 |

Read: H6 is the best conceptual direction so far. Frozen epoch11 is a clean trade-off. All-params gives strong SOP reduction and AEE close to baseline, but AAE explodes, likely because full fine-tuning over-sparsifies early/downsample paths with no angular loss.

## H7 FFN stage expansion

Goal: keep H6a core and add one more FFN stage with binary ATLIF.

Short probes:

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H7 stage01 FFN | valid10 | 0.083854 | 3.5747G | 1.074384 | 6.388056 |
| H7 stage02 FFN | valid10 | 0.082757 | 3.5279G | 1.100928 | 6.760951 |
| H7 stage03 FFN | valid10 | 0.084142 | 3.5870G | 1.102599 | 6.741123 |

Read: stage1 is least damaging, but frozen-backbone expansion loses accuracy versus H6a. Useful as sensitivity, not final.

## H8 FFN block search

Goal: search individual FFN blocks/stages with H6-like core, weaker FFN sparse pressure, and all-parameter training.

Fixed core:

- Attention Q/K: PSN + ATLIF + ternary
- Selected FFN/downsample/high-SOP modules: PSN + ATLIF + binary
- Trainable: all

Short probe selection threshold: AEE <= 1.07, AAE <= 6.35, SOPs <= 3.60G.

Key short probes:

| run | split | firing | SOPs | AEE | AAE | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H8a stage1 block0 | valid10 | 0.082783 | 3.5290G | 1.050845 | 6.160332 | candidate |
| H8g stage2 block0 | valid10 | 0.083428 | 3.5565G | 1.067276 | 6.021865 | candidate |
| H8m stage3 block0 | valid10 | 0.084160 | 3.5877G | 1.040409 | 6.133985 | promoted |
| H8p stage1b0 + stage2b4 | valid10 | 0.083978 | 3.5800G | 1.040574 | 6.099212 | candidate |
| H8r stage2 mid | valid10 | 0.083629 | 3.5651G | 1.052276 | 6.067079 | candidate |

Mid-run H8m full check:

| run | split | firing | SOPs | AEE | AAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| H8m full interrupted epoch12 | valid40 | 0.068976 | 2.9405G | 1.579596 | 20.715118 |
| H8m continuation epoch16/global29 | valid40 | 0.069890 | 2.9794G | 1.596608 | 22.794633 |

Current status:

- H8m full/continuation finished from original `checkpoint_epoch12`.
- Continuation directory: `neuron_experiments/H8_ffn_block_search/results/h8m_stage3_block0_continue_from_epoch12_20260512_1210_nomlflowmodel_setsid`
- Local epoch numbers in that directory start from 0. Local epoch0 equals global epoch13.

Read: H8 short probes looked promising, but valid40 checks show the same AAE risk as H6 all-params. The final H8m checkpoint keeps AEE close and saves SOPs, but AAE is worse than H6 all-params. This supports the newer BSA hypothesis: Q/K ternary replacement needs the full attention-side normalization path, such as Shiftmax, not just a neuron swap. H8p or H8a may still be worth a later run after the attention operator is fixed.

## Current best story

Most defensible current path:

1. Base mechanism: H6-style split output.
   - ternary only in attention Q/K
   - binary in FFN/downsample/high-SOP layers
2. Training should avoid unconstrained all-parameter drift.
   - start with ATLIF/threshold-only or mostly frozen training
   - then carefully unfreeze late/selected modules
3. Add angular/cosine consistency or baseline direction distillation before claiming final improvement.
4. Avoid H5-style ternary expansion into FFN/downsample.

Best currently reportable trade-off:

| run | split | firing | SOPs | AEE | AAE | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H6a frozen epoch11 | valid40 | 0.077434 | 3.3010G | 1.553494 | 8.200176 | cleanest sparse trade-off so far |
| H6a all-params epoch29 | valid40 | 0.064930 | 2.7680G | 1.594698 | 21.636660 | strong SOP but angular failure |
| H8m continuation epoch16/global29 | valid40 | 0.069890 | 2.9794G | 1.596608 | 22.794633 | confirms all-params Q/K ternary angular failure |
