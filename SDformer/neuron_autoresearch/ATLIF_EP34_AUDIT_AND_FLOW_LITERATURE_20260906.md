# ATLIF ep34 audit and SNN optical-flow literature refresh

Audit date: 2026-09-06. Scope: algorithm source, checkpoint tensors, existing
capture receipts, and public primary sources. No training, model changes, or
hardware code changes were made.

## 1. Installed, called, and functionally live are different counts

| Quantity | ep34 count | Meaning |
|---|---:|---|
| Installed ATLIFTernaryPSN modules | 105 | Converted modules retained in the model/checkpoint |
| Not called by current attention | 12 | One `attn.sn2_q.spiking_neuron` per block; original QK carrier path |
| Dynamically called modules | 93 | Observed in the 40-sample ep34 capture |
| Called but output unused by normal flow inference | 12 | One `attn.attn_sn.spiking_neuron` per block; diagnostic attention return |
| Functionally live modules for normal flow inference | 81 | 45 at temporal T=10; 36 at T=2 |

This is not neuron pruning or hardware disabling working layers. The H60-family
no-carrier branch uses `attn = k_orig.mul(gate)` and never calls `sn2_q`.
At the shared tail, `attn = self.attn_sn(x)` is followed by `x = self.proj(x)`,
not `self.proj(attn)`. The caller consumes the projected first return value;
the second value is consumed only when `return_attention=True`.

The capture name `live93` uses runtime-call liveness. Its own activity records
mark 12 entries `deployment_dead_result=true`. Do not relabel all 93 as
functionally necessary, or call 81 the total installed neuron count. These
counts also do not specify physical hardware instance counts.

Verified capture:
`hw_autoresearch_nts07/results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831/`.
Its `manifest.json` records static_atlif=105, live_atlif=93 and 12 bypassed
`sn2_q` names. `forensic_samples/sample_39/atlif_activity_cumulative.json`
contains 93 rows, 40 calls per row, 12 dead-result rows, 81 functional rows,
all binary modes, and zero summed sampled recomputation mismatch. The dead-result
rows emitted zero active events in this particular capture; lack of output
consumption, not zero activity, establishes their functional status.

Source locations:

- H9 overlay `models/STSwinNet_SNN/bsa_attention.py:6083`: H60 no-carrier branch.
- Same file at line 6620: separate diagnostic `attn_sn` and projection inputs.
- `third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_swin_transformer3D.py:805`:
  first/second return consumption and `return_attention` branch.

## 2. Binary ATLIF still emits the threshold amplitude

The implemented temporal neuron is a PSN temporal matrix with ATLIF-style firing
and surrogate gradients, not a newly implemented recurrent LIF membrane update:

```text
h = A_time @ x + b_time
event = 1[h >= theta]
output = theta * event
```

`OfficialATLIFSurrogate.forward` returns `out * thre` in
`neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py:104`.
Binary means two levels `{0, theta}`. Ternary would mean three levels
`{-theta, 0, +theta}`. The historical class name `ATLIFTernaryPSN` supports both;
the final configuration selects `output_mode: binary` and
`threshold_mode: official_atlif` for all sites.

Hardware may store `event` as one bit with one shared theta per logical site.
The amplitude must be restored or accounted for in downstream scales. For a
linear consumer, `W(theta*event) = (theta*W)event` in real arithmetic, but folding
requires checking the deployed rounding and saturation order. Changing both the
threshold comparator and output amplitude to one is not automatically exact.
Residuals, membrane values, and gated-K remain multi-bit where required.

## 3. Actual final checkpoint thresholds

Read directly on CPU from `checkpoint_epoch34.pth`, key `model_state_dict`:

- 105 scalar thresholds present; 95 exactly equal to float32 1.0.
- 10 non-unit thresholds, all between 0.9998828172683716 and 1.0.
- All 24 Q/K thresholds exactly equal to 1.0; their outputs really are `{0,1}`.
- All 105 thresholds are bit-identical between strict C12 ep29 and resumed ep34.
- Of the 81 functional sites, 71 have unit thresholds and 10 have non-unit thresholds.

| Non-unit site (prefix `sttmultires_unet.` omitted) | theta |
|---|---:|
| resblocks.0.sn1.spiking_neuron | 0.9999106526374817 |
| resblocks.0.sn2.spiking_neuron | 0.9998828172683716 |
| resblocks.1.sn1.spiking_neuron | 0.9999173283576965 |
| resblocks.1.sn2.spiking_neuron | 0.999895453453064 |
| decoders.0.sn.spiking_neuron | 0.9999606013298035 |
| decoders.1.sn.spiking_neuron | 0.9999942779541016 |
| preds.0.sn.spiking_neuron | 0.9999988079071045 |
| preds.1.sn.spiking_neuron | 0.999998152256012 |
| encoders.swin3d.layers.2.downsample.sn.spiking_neuron | 0.9999970197677612 |
| encoders.swin3d.layers.3.swin_blocks.1.mlp.sn1.spiking_neuron | 0.9999987483024597 |

Checkpoint: `neuron_experiments/H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830/checkpoint_epoch34.pth`.
Capture-bound SHA256: `4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48`.
Standard ep34 valid825 receipt: 105 ATLIF, 12 attention, 210 overlay keys,
missing=0, unexpected=0. Module absence is therefore not explained by missing weights.
The adjacent `checkpoint_epoch34_state_dict.pth` is optimizer/scheduler/scaler
resume state, despite its name; neuron tensors are in `checkpoint_epoch34.pth`.

## 4. What is currently adaptive, and what is disabled

The final config has `threshold_eta=0`, `activity_eta=0`, `target_rate=null`, and
`target_rate_eta=0`. Activity-driven manual threshold growth and rate feedback
are disabled. Threshold parameters and a nonzero optimizer threshold LR remain
configured; this is not evidence of an explicit `requires_grad=False` freeze.
The direct ep29/ep34 tensor comparison establishes no net threshold change in
the five-epoch resume. The cause of every zero/sub-ULP optimizer update was not
reconstructed in this audit.

Consequently, do not attribute the final sparsity improvement to substantial
learned threshold growth or an active target-rate controller. C00/C10 isolates
the complete neuron/training change, not threshold adaptation by itself.
The safe description is an all-site binary PSN/ATLIF-style neuron with nearly
unit deployment thresholds. Proving adaptation as a separate contribution would
require a matched fixed-threshold control and threshold-trajectory evidence.

The configured min/max threshold fields also are not evidence of active clamping:
`installer.py:710` skips that clamp for `official_atlif` mode.

## 5. Literature refresh: what appeared after SDformerFlow

Searched public web results for spiking transformer/spikeformer + optical flow,
2025/2026, including targeted CVF and OpenReview queries; inspected primary
paper method sections and author repositories where accessible. A negative
search result is not proof that no other work exists.

| Work | Verified architecture/status | Relevance |
|---|---|---|
| SDformerFlow extended version | Author repository now reports TETCI acceptance; same family as ICPR 2024, with its improved model | Still the closest verified spiking-Transformer flow parent; do not count the two versions as independent competitor families |
| SNN-Driven Event-Based Flow and Rotation Estimation with SO(3) Refinement, AAAI 2026 | TIDNet-derived convolutional architecture with Spike GRU and membrane carryover | New direct SNN optical-flow competitor, not a spiking Transformer; should enter related work and protocol-aware external comparison |
| ST-FlowNet, 2025 | FlowNet encoder/decoder plus ConvGRU; ANN-to-SNN/BISNN training | New SNN flow competitor, not Transformer; architecture verified in full-text section 3 |
| Aquatic Neuromorphic Optical Flow, May 2026 preprint | Aq-FireNet: multi-scale convolution, lightweight spiking attention, LIF/ConvGRU temporal processing | SNN + attention in underwater flow; not established as a Transformer backbone or a DSEC/MVSEC replacement |
| STIRFlow, author-announced IEEE TIP acceptance, 2026 | SNN feature encoders plus temporal iterative refinement; full architecture not independently inspected | Important recent SNN flow lead. Author GitHub currently contains only README saying code is forthcoming; no runnable release verified |

The search did not establish another independent, released **spiking Transformer
optical-flow backbone** beyond the SDformerFlow family. It did establish newer
SNN optical-flow competitors. Do not write "no new SNN flow models" or current
SNN accuracy SOTA based only on comparison with SDformerFlow.

AAAI 2026 Table 1 reports EPE 0.87 and 3PE 2.8 for the Spike-GRU model, versus
0.85 for its conventional-ConvGRU variant. The paper separately says official
DSEC test results are in its appendix; Figure 6 is validation. These published
numbers must not be inserted as locally reproduced valid825 scores or silently
treated as official hidden-test scores. Its energy table has a unit issue
(G operations times pJ/operation has mJ scale); do not copy its `pJ` label into
our hardware energy table without resolving it.

## 6. Primary sources and next use

- [SDformerFlow author repository](https://github.com/yitian97/SDformerFlow)
- [AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/view/37877)
- [AAAI 2026 full PDF, architecture and Tables 1-2](https://ojs.aaai.org/index.php/AAAI/article/download/37877/41839)
- [ST-FlowNet full text, section 3](https://arxiv.org/html/2503.10195v1)
- [Aq-FireNet full text, section II-B](https://arxiv.org/html/2605.07653v1)
- [STIRFlow author acceptance announcement](https://www.linkedin.com/posts/sajid-javed-a531bb63_computervision-eventcameras-neuromorphicvision-activity-7493789060147122176-LYey)
- [STIRFlow author repository](https://github.com/AhmedHumais/STIRFlow)

Prioritize a protocol audit of AAAI 2026 and tracking STIRFlow's paper/code
release. No need to change the final model merely because these papers exist.
The algorithm-side description must distinguish event encoding, amplitude,
temporal mixing, and disabled training regularizers. Hardware-side integration
and implementation decisions remain with the hardware owner.
