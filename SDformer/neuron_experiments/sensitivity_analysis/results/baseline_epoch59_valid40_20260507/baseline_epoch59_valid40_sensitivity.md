# PSN Baseline Layer Sensitivity

This report ranks baseline spiking layers by spike/SOP contribution using the same global SOP proxy as `tools/profile_sops.py`:

`layer_sops_proxy = dense_ops * layer_spikes / total_elements`

It is a contribution sensitivity report, not an accuracy ablation yet.

## Baseline

| item | value |
| --- | ---: |
| samples | 40 |
| AEE | 1.5848 |
| AAE | 7.5012 |
| firing | 0.08496 |
| total SOPs | 3.6219G |
| profiled layers | 105 |

## SOP Concentration

| target set | SOPs proxy | share |
| --- | ---: | ---: |
| top 10 layers | 1.7278G | 47.71% |
| top 20 layers | 2.5659G | 70.85% |
| top 40 layers | 3.2366G | 89.36% |

## Stage Summary

| group | layers | firing | SOPs proxy | SOPs % |
| --- | ---: | ---: | ---: | ---: |
| `encoder` | 93 | 0.07366 | 2.7683G | 76.43% |
| `decoder` | 4 | 0.24879 | 0.4730G | 13.06% |
| `sttmultires_unet_other` | 4 | 0.11871 | 0.3436G | 9.49% |
| `transformer_block` | 4 | 0.14687 | 0.0370G | 1.02% |

## Substage Summary

| group | layers | firing | SOPs proxy | SOPs % |
| --- | ---: | ---: | ---: | ---: |
| `encoder.swin3d` | 93 | 0.07366 | 2.7683G | 76.43% |
| `sttmultires_unet_other` | 4 | 0.11871 | 0.3436G | 9.49% |
| `decoder.3` | 1 | 0.28186 | 0.2867G | 7.92% |
| `decoder.2` | 1 | 0.19899 | 0.1007G | 2.78% |
| `decoder.1` | 1 | 0.21786 | 0.0550G | 1.52% |
| `transformer_block` | 4 | 0.14687 | 0.0370G | 1.02% |
| `decoder.0` | 1 | 0.24374 | 0.0307G | 0.85% |

## Top Layers

| rank | layer | firing | SOPs proxy | SOPs % | cumulative % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `sttmultires_unet.decoders.3.sn` | 0.28186 | 0.2867G | 7.92% | 7.92% |
| 2 | `sttmultires_unet.encoders.swin3d.patch_embed.head.sn` | 0.05527 | 0.2226G | 6.14% | 14.06% |
| 3 | `sttmultires_unet.preds.3.sn` | 0.10495 | 0.2113G | 5.83% | 19.90% |
| 4 | `sttmultires_unet.encoders.swin3d.patch_embed.proj.sn` | 0.08795 | 0.1771G | 4.89% | 24.78% |
| 5 | `sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.1.sn1` | 0.08428 | 0.1697G | 4.69% | 29.47% |
| 6 | `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn2` | 0.07328 | 0.1476G | 4.07% | 33.54% |
| 7 | `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn2` | 0.07207 | 0.1451G | 4.01% | 37.55% |
| 8 | `sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.1.sn2` | 0.06589 | 0.1327G | 3.66% | 41.21% |
| 9 | `sttmultires_unet.encoders.swin3d.layers.0.downsample.sn` | 0.23622 | 0.1189G | 3.28% | 44.50% |
| 10 | `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn1` | 0.23101 | 0.1163G | 3.21% | 47.71% |
| 11 | `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1` | 0.22197 | 0.1117G | 3.08% | 50.79% |
| 12 | `sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.sn1` | 0.05520 | 0.1111G | 3.07% | 53.86% |
| 13 | `sttmultires_unet.decoders.2.sn` | 0.19899 | 0.1007G | 2.78% | 56.64% |
| 14 | `sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.sn2` | 0.04931 | 0.0993G | 2.74% | 59.38% |
| 15 | `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.attn.proj_sn` | 0.18247 | 0.0947G | 2.62% | 62.00% |
| 16 | `sttmultires_unet.preds.2.sn` | 0.17241 | 0.0868G | 2.40% | 64.39% |
| 17 | `sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.1.mlp.sn2` | 0.06627 | 0.0667G | 1.84% | 66.23% |
| 18 | `sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.0.mlp.sn2` | 0.05595 | 0.0563G | 1.56% | 67.79% |
| 19 | `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.attn.proj_sn` | 0.10732 | 0.0557G | 1.54% | 69.33% |
| 20 | `sttmultires_unet.decoders.1.sn` | 0.21786 | 0.0550G | 1.52% | 70.85% |
| 21 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp.sn2` | 0.10314 | 0.0519G | 1.43% | 72.28% |
| 22 | `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.attn.sn_k` | 0.08386 | 0.0435G | 1.20% | 73.48% |
| 23 | `sttmultires_unet.encoders.swin3d.layers.1.downsample.sn` | 0.17007 | 0.0428G | 1.18% | 74.66% |
| 24 | `sttmultires_unet.encoders.swin3d.layers.2.downsample.sn` | 0.29898 | 0.0376G | 1.04% | 75.70% |
| 25 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.5.mlp.sn1` | 0.28112 | 0.0354G | 0.98% | 76.68% |
| 26 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp.sn2` | 0.06691 | 0.0337G | 0.93% | 77.61% |
| 27 | `sttmultires_unet.encoders.swin3d.layers.1.swin_blocks.1.mlp.sn1` | 0.13358 | 0.0336G | 0.93% | 78.54% |
| 28 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.5.attn.proj_sn` | 0.23607 | 0.0334G | 0.92% | 79.46% |
| 29 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.4.attn.proj_sn` | 0.23017 | 0.0326G | 0.90% | 80.36% |
| 30 | `sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.3.mlp.sn2` | 0.06451 | 0.0325G | 0.90% | 81.25% |

## Candidate Target Sets

For the first partial sparsity experiment, use target layers with high SOP share and avoid changing every spiking node.

| set | layers | reason |
| --- | ---: | --- |
| G1-top10 | 10 | smallest intervention; tests whether the hottest layers are compressible |
| G1-top20 | 20 | stronger SOP target while still avoiding blanket replacement |
| G1-decoder-hot | variable | decoder layers have high firing and direct reconstruction impact, so use after top10/top20 probe |

## Accuracy Ablation

Method:

Selected layer outputs were temporarily set to zero with forward hooks, then evaluated on PSN baseline checkpoint `epoch59`. This is a stress test for accuracy sensitivity. It is not the final sparse-gate implementation.

Fast valid8 sweep:

| set | layers | removable SOPs proxy | share | AEE | dAEE | AAE | dAAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0 | 0.0000G | 0.00% | 1.0116 | +0.0000 | 6.2213 | +0.0000 |
| top1 | 1 | 0.2867G | 7.92% | 7.0368 | +6.0252 | 99.7679 | +93.5466 |
| top3 | 3 | 0.7206G | 19.90% | 6.8008 | +5.7892 | 81.6197 | +75.3984 |
| top5 | 5 | 1.0673G | 29.47% | 6.8008 | +5.7892 | 81.6197 | +75.3984 |
| top10 | 10 | 1.7278G | 47.71% | 6.8008 | +5.7892 | 81.6197 | +75.3984 |
| top20 | 20 | 2.5659G | 70.85% | 6.8008 | +5.7892 | 81.6197 | +75.3984 |
| decoder_all | 4 | 0.4730G | 13.06% | 7.0368 | +6.0252 | 99.7679 | +93.5466 |
| patch_embed_hot | 6 | 0.9124G | 25.19% | 6.0800 | +5.0685 | 37.2835 | +31.0623 |
| layer0_mlp_hot | 4 | 0.5207G | 14.38% | 1.0013 | -0.0102 | 5.6287 | -0.5925 |
| layer0_attn_proj | 2 | 0.1504G | 4.15% | 1.0290 | +0.0175 | 6.2886 | +0.0673 |
| pred_hot | 2 | 0.2981G | 8.23% | 6.8008 | +5.7892 | 81.6197 | +75.3984 |

Valid40 confirmation for the promising layer0 sets:

| set | layers | removable SOPs proxy | estimated remaining SOPs | AEE | dAEE | AEE change | AAE | dAAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0 | 0.0000G | 3.6219G | 1.5848 | +0.0000 | +0.00% | 7.5012 | +0.0000 |
| layer0_attn_proj | 2 | 0.1504G | 3.4714G | 1.5583 | -0.0265 | -1.67% | 7.4247 | -0.0765 |
| layer0_mlp_hot | 4 | 0.5207G | 3.1012G | 1.6327 | +0.0479 | +3.02% | 7.3547 | -0.1465 |
| layer0_mlp_attn | 6 | 0.6711G | 2.9508G | 1.6248 | +0.0401 | +2.53% | 7.2609 | -0.2403 |

Interpretation:

The decoder, prediction heads, and early patch embedding spike outputs are accuracy-critical even when their SOP contribution is high. They are poor first targets for hard sparsity.

The first two layer0 Swin blocks are the opposite: their MLP and attention projection spike nodes contribute about 18.5% of baseline SOPs, but zeroing them on valid40 only raises AEE by about 2.5% and improves AAE. These six nodes are the best first target for a trainable partial sparse gate.

Artifacts:

- ranked layers: `neuron_experiments/sensitivity_analysis/results/baseline_epoch59_valid40_20260507/baseline_epoch59_valid40_ranked_layers.csv`
- stage summary: `neuron_experiments/sensitivity_analysis/results/baseline_epoch59_valid40_20260507/baseline_epoch59_valid40_stage_summary.csv`
- substage summary: `neuron_experiments/sensitivity_analysis/results/baseline_epoch59_valid40_20260507/baseline_epoch59_valid40_substage_summary.csv`
- ablation target sets: `neuron_experiments/sensitivity_analysis/results/baseline_epoch59_valid40_20260507/ablation_target_sets.csv`
- ablation JSON outputs: `neuron_experiments/sensitivity_analysis/results/baseline_epoch59_valid40_20260507/ablation_*_valid*.json`
