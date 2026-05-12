# Neuron Experiment Results Summary

Date: 2026-05-06

All experiment code is under `neuron_experiments/` and does not edit the baseline `third_party/SDformerFlow` tree. The main comparable evaluation protocol is `tools/profile_sops.py` on `valid40`, reporting AEE, AAE, global firing rate, and estimated SOPs.

## Baseline

| run | checkpoint | samples | AEE | AAE | firing | SOPs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PSN baseline | `experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth` | 40 | 1.5848 | 7.5012 | 0.08496 | 3.6219G |
| PSN baseline | same | 8 | 1.0116 | 6.2213 | 0.08548 | 3.6439G |

## Comparable Full/Valid40 Results

| rank | experiment | neuron/protocol | checkpoint | AEE | AAE | firing | SOPs | verdict |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | E0 | PSN baseline | epoch59 | 1.5848 | 7.5012 | 0.08496 | 3.6219G | current best overall |
| 1 | G1 | local PSN sparse gate, 6 layer0 nodes | smoke epoch0 | 1.6056 | 7.2452 | 0.06365 | 2.7134G | best sparse/accuracy tradeoff; 25.1% SOP reduction |
| 2 | G1 | local PSN sparse gate, BN-eval short | epoch4 | 1.6248 | 7.2609 | 0.06434 | 2.7426G | still good; gate-only training did not reopen gates |
| 3 | E6a | NASN blanket replacement | epoch59 | 2.1676 | 8.3613 | 0.78138 | 33.3102G | accuracy second among replacements, but almost dense and not viable |
| 4 | E4 | official TS-LIF transplant | epoch59 | 2.1816 | 9.8193 | 0.09417 | 4.0146G | closest balanced full replacement, but worse than PSN |
| 5 | E6a | NASN blanket replacement | epoch30 | 2.2866 | 9.5194 | 0.85279 | 36.3546G | best validation-loss checkpoint, even denser |
| 6 | E2 | ATLIF full-pretrained | epoch59 | 2.5128 | 12.5417 | 0.12212 | 5.2062G | accuracy worse and less sparse |
| 7 | E3 | official LMHT | epoch54 | 2.5621 | 9.6492 | 0.22770 | 9.7070G | AAE moderate, SOPs too high |
| 8 | E3 | official LMHT | epoch59 | 2.7290 | 10.1696 | 0.23083 | 9.8404G | slightly worse than epoch54 |
| 9 | E2 | ATLIF official-copy low-SOP | epoch30 | 3.6035 | 19.4891 | 0.07051 | 3.0059G | sparse but accuracy poor |
| 10 | E2 | ATLIF official-copy low-SOP | epoch59 | 3.7574 | 18.6163 | 0.06730 | 2.8692G | best full-replacement sparsity, unusable accuracy hit |
| 11 | E2 | ATLIF official-copy low-SOP | epoch49 | 3.8743 | 20.7177 | 0.06626 | 2.8245G | lowest full-replacement SOPs observed, accuracy poor |
| 12 | E2 | first ATLIF full | epoch59 | 4.0057 | 21.4918 | 0.38560 | 16.4381G | broken/incorrect ATLIF scale |
| 13 | E2 | Plan A conservative ATLIF | epoch10 | 5.6600 | 27.6559 | 0.16096 | 6.8619G | not worth continuing |
| 14 | E2 | Plan A conservative ATLIF | epoch19 | 5.6760 | 29.2109 | 0.16163 | 6.8903G | not worth continuing |
| 15 | E2 | corrected ATLIF bs16w2 | epoch59 | 8.6602 | 67.8866 | 0.37876 | 16.1464G | still bad |
| 16 | E5b | official-style ternary spike | epoch59 | 29.7720 | 98.3742 | 0.60730 | 25.8892G | failed badly |

## ATLIF Branch Details

Official source:

`/root/private_data/work/optimization_sources/neuron_optimization/ATLIF_Activity-Pruning-SNN`

Key implementation path:

`neuron_experiments/E2_exp_atlif/overlay/models/STSwinNet_SNN/experimental_neurons/single/atlif.py`

| branch | setup | result |
| --- | --- | --- |
| early E2 full | first ATLIF replacement, `full_bs12w8.yml` | AEE 4.0057, firing 0.38560, SOPs 16.4381G; worse than baseline |
| corrected E2 bs16w2 | fixed part of ATLIF dynamics but still wrong training scale | AEE 8.6602, SOPs 16.1464G; worse |
| official-copy low-SOP | upstream ATLIF blocks copied, threshold update active, no AMP, `threshold_eta=1e-3`, `threshold_lr_scale=1000`, `activity_eta=1e-4` | thresholds grew and SOPs dropped to 2.8692G at epoch59, but AEE rose to 3.7574 |
| full-pretrained ATLIF | initialized from PSN baseline, tuned with `eta3e-4`, `eta2=3e-5`, AMP, batch 12 | AEE 2.5128, SOPs 5.2062G; better accuracy than low-SOP ATLIF but worse than PSN |
| Plan A | conservative baseline-like transfer, lr `1e-5`, weak activity penalty | AEE about 5.66 and SOPs about 6.89G; failed |
| freeze threshold-only | froze 54.9M parameters, trained only 105 ATLIF thresholds for 5 epochs | slight sparsity improvement versus ATLIF full-pretrained, but still worse than PSN |

ATLIF freeze results:

| run | AEE | AAE | firing | SOPs |
| --- | ---: | ---: | ---: | ---: |
| ATLIF full-pretrained epoch59 | 2.5128 | 12.5417 | 0.12212 | 5.2062G |
| freeze threshold-only epoch2 | 2.5498 | 13.4973 | 0.11745 | 5.0068G |
| freeze threshold-only epoch4 | 2.5837 | 13.5899 | 0.11563 | 4.9292G |

ATLIF conclusion:

The only branch that clearly reduced SOPs below PSN was the official-copy low-SOP branch, but it lost too much optical-flow accuracy. The threshold growth mechanism is working, but it currently over-prunes or disrupts the SDFormerFlow feature path.

## LMHT Branch Details

Official source:

`/root/private_data/work/optimization_sources/neuron_optimization/LMH_LMHT_SNN`

Remote:

`https://github.com/hzc1208/LMHT_SNN`

Commit:

`d9e0db3ce917c4c93acc46d8a63e4d4919e7eb2c`

Key implementation path:

`neuron_experiments/E3_exp_lmh/overlay/models/STSwinNet_SNN/experimental_neurons/single/lmh.py`

Full setting:

| item | value |
| --- | --- |
| config | `neuron_experiments/E3_exp_lmh/configs/full.yml` |
| init | PSN baseline epoch59 |
| batch/workers | 9 / 8 |
| AMP | true |
| pin_memory | false |
| selected profile | epoch54 was better than epoch59 |

Result:

LMHT trained through epoch59 without runtime failure. It follows the official training-time LMHT neuron and threshold rule `v_th = 2 / L` with `L=2`, but it does not run the official post-training `LMHT_Inference_Neuron` and `L*T` temporal expansion path because SDFormerFlow is built around fixed 10-bin event input. It did not beat PSN.

## TS-LIF Branch Details

Official source:

`/root/private_data/work/optimization_sources/neuron_optimization/TSLIF_TS-LIF`

Remote:

`https://github.com/kkking-kk/TS-LIF`

Commit:

`a59826a6c7f62d0f16edbafdbb28db65bebd9f69`

Key implementation path:

`neuron_experiments/E4_exp_tslif/overlay/models/STSwinNet_SNN/experimental_neurons/single/official_tslif.py`

Full setting:

| item | value |
| --- | --- |
| config | `neuron_experiments/E4_exp_tslif/configs/full.yml` |
| init | PSN baseline epoch59 |
| batch/workers | 4 / 8 |
| AMP | true |
| pin_memory | false |
| TF32 | true |
| final train loss | 1.4070 |

Result:

E4 official TS-LIF is the closest full replacement result after PSN: AEE 2.1816, AAE 9.8193, firing 0.09417, SOPs 4.0146G. It is still worse than baseline, but the failure is much less severe than ATLIF, LMHT, or TSN.

## E4b Official-Style TS-LIF Short Runs

Purpose:

Use official-style optimizer behavior: Adam, gradient clipping, low LR for PSN-pretrained backbone, higher LR for TS-LIF parameters.

| run | samples | AEE | AAE | firing | SOPs | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E4b aggressive short | 8 | 6.9871 | 83.8741 | 0.05075 | 2.1633G | sparse but accuracy collapsed |
| E4b stable short | 8 | 7.0631 | 85.3213 | 0.05525 | 2.3555G | sparse but accuracy collapsed |

Decision:

Do not launch full E4b from these settings. Optimizer groups alone are not enough; the TS-LIF integration likely needs a better non-scalar alpha/channel adaptation or partial insertion strategy.

## Ternary Spike Branch Details

Official source:

`/root/private_data/work/optimization_sources/neuron_optimization/TSN_Ternary-Spike`

Remote:

`https://github.com/yfguo91/Ternary-Spike`

Commit:

`2aca58747f01d7960cb6f0284665bbb353d35aab`

Key implementation path:

`neuron_experiments/E5b_exp_tsn_officialstyle/overlay/models/STSwinNet_SNN/experimental_neurons/single/tsn.py`

| run | samples | AEE | AAE | firing | SOPs | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E5b short epoch2 | 8 | 38.1304 | 102.8358 | 0.57581 | 24.5468G | collapsed immediately |
| E5b full epoch59 | 40 | 29.7720 | 98.3742 | 0.60730 | 25.8892G | failed after full run |

Decision:

Do not continue this TSN setting. The official ternary spike dynamics are ported, but the official repo is closer to an ANN-to-SNN image-classification conversion pipeline with `SpikeModel`/`SpikeConv`, while SDFormerFlow uses event bins and blanket `Spiking_neuron` replacement. The paradigm mismatch is too large for this transplant.

## NASN Branch Details

Source:

`Adaptive Spiking Neurons for Vision and Language Modeling`, arXiv `2604.12365`

Implementation path:

`neuron_experiments/E6_exp_asn/overlay/models/STSwinNet_SNN/experimental_neurons/single/asn.py`

Training setup:

| item | value |
| --- | --- |
| config | `neuron_experiments/E6_exp_asn/configs/full_resume_bs16w8_amp.yml` |
| init | PSN baseline epoch59, then resumed from E6a bs4 epoch5 |
| batch/workers | 16 / 8 |
| AMP | true |
| NASN params | `D=4`, `N=4`, `beta=0.5`, scalar learnable alpha |
| final train loss | 1.3075 |

Result:

| checkpoint | AEE | AAE | firing | SOPs | result |
| --- | ---: | ---: | ---: | ---: | --- |
| epoch30 | 2.2866 | 9.5194 | 0.85279 | 36.3546G | best validation-loss checkpoint, too dense |
| epoch59 | 2.1676 | 8.3613 | 0.78138 | 33.3102G | better accuracy than E4, but about 9.2x PSN SOPs |

Decision:

Do not use E6a NASN as a blanket replacement in this form. It can recover some optical-flow accuracy from the PSN checkpoint, but it opens far too many spikes. Any next NASN attempt should be partial insertion or add explicit firing regularization.

## G1 Partial Sparse Gate Details

Key implementation path:

`neuron_experiments/G1_partial_sparse_gate/overlay/models/STSwinNet_SNN/sparse_gate.py`

Target:

Six sensitivity-selected layer0 Swin nodes:

- `layers.0.swin_blocks.0.attn.proj_sn`
- `layers.0.swin_blocks.0.mlp.sn1`
- `layers.0.swin_blocks.0.mlp.sn2`
- `layers.0.swin_blocks.1.attn.proj_sn`
- `layers.0.swin_blocks.1.mlp.sn1`
- `layers.0.swin_blocks.1.mlp.sn2`

Result:

| run | AEE | AAE | firing | SOPs | result |
| --- | ---: | ---: | ---: | ---: | --- |
| smoke epoch0, gates closed | 1.6056 | 7.2452 | 0.06365 | 2.7134G | best sparse/accuracy tradeoff so far |
| BN-eval short epoch4 | 1.6248 | 7.2609 | 0.06434 | 2.7426G | gate-only short kept all gates closed |

Decision:

G1 is currently the strongest result for the sparse hardware-friendly story. It reduces SOPs by about 25% while keeping AEE within 1.3-2.5% of PSN baseline. The important implementation detail is that gate-only training must keep the frozen backbone in eval mode; otherwise BatchNorm statistics drift even with frozen weights.

## Smoke-Only / Scaffolded Experiments

These ran as wiring checks only and do not have valid40 AEE/SOPs profiles.

| experiment | neuron | train loss | validation loss | samples/sec | max GPU GiB | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| E1 | SN | 6.7747 | 6.3207 | 0.3077 | 4.878 | smoke passed |
| E5 | early TSN scaffold | 10.8713 | 8.5419 | 0.3896 | 4.870 | smoke passed only; replaced by E5b official-style |
| E6a | NASN | 10.2807 | 6.6645 | 0.7509 | 4.341 | smoke passed; valid1 firing 0.19285, SOPs 8.2211G |
| F1 | fused adaptive PSN | 8.1597 | 6.2164 | 0.3709 | 5.599 | smoke passed |
| F2 | fused LMH + ATLIF | 6.9589 | 6.4206 | 0.3133 | 7.343 | smoke passed |
| F3 | fused adaptive TS-LIF | 6.0360 | 6.3523 | 0.2639 | 10.759 | smoke passed |
| F4 | fused LMH + TS-LIF | 14.8995 | 15.3773 | 0.2954 | 9.484 | smoke passed, quality bad |
| F5 | fused signed hybrid | 9.0083 | 6.8135 | 0.3060 | 9.162 | smoke passed |

## Overall Takeaways

1. PSN baseline is still the best complete result.
2. G1 local sparse gating is the first result that supports the desired sparse story: about 25% lower SOPs with only about 1.3% AEE increase.
3. E6a NASN reaches the second-best AEE/AAE among blanket replacements, but its firing and SOPs are about 9.2x PSN, so it is not viable for sparse inference.
4. E4 official TS-LIF is the most promising balanced single-neuron replacement by accuracy and sparsity, but it is not sparse enough and still loses to PSN.
5. E2 ATLIF official-copy proves that adaptive threshold pruning can reduce SOPs below baseline, but the current SDFormerFlow integration trades away too much accuracy.
6. E3 LMHT trains cleanly and is source-aligned for training, but missing official direct-inference reparameterization and high SOPs make it noncompetitive.
7. E5b TSN is not viable in the current blanket-replacement form.
8. Fusion folders F1-F5 are only smoke-tested scaffolds; they need proper short/full experiments before being compared.

## Recommended Next Step

The most rational next branch is not another blanket full run. Use E4 as the accuracy-preserving base and borrow only the useful ATLIF idea as a controlled sparsity mechanism:

| proposal | base | change | reason |
| --- | --- | --- | --- |
| G1 partial ATLIF gating on E4/PSN | PSN or E4 | apply adaptive threshold only to high-firing MLP/attention spike nodes | target SOPs without destabilizing every neuron |
| G2 E4 staged fine-tune | E4 | freeze most backbone first, tune TS-LIF parameters, then unfreeze with tiny LR | avoid random TS-LIF params destroying pretrained PSN features |
| G3 ATLIF low-SOP recovery | E2 official-copy epoch30/49 | reduce or decay `activity_eta` and threshold update after SOPs reach baseline level | keep sparsity but stop over-pruning |
| G4 E4 channel-aware alpha | E4/E4b | replace scalar `alpha_s/alpha_l` with lazy/channel-shaped parameters where feature dims are known | closer to official TS-LIF usage |
