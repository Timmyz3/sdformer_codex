# Algorithm refresh: frozen-graph candidates for Complete TTX

Date: 2026-09-06. Status: literature and implementation review, NOT launched.
This is a candidate protocol, not evidence of an accuracy improvement. No model,
training, queue, checkpoint, or hardware implementation was changed.

## 1. Decision and frozen reference

Keep C12 ep34, alpha=1/8, as the paper/deployment reference. First investigate
training-only temporal robustness/distillation, then event-structure supervision.
Do not reopen architecture search merely because newer papers exist.

- DSEC local valid825: AEE 1.199514, AAE-2D 5.400641, AE-3D 5.106363,
  Fl 5.3138%, spikes 72.8912G. Source: `DATE_PAPER_RESULT_TABLES_20260901.md`.
- Parent: `neuron_experiments/H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830/checkpoint_epoch34.pth`.
- Retain all12 Complete TTX, one-sided binary ATLIF, no carrier, full resolution,
  existing temporal dimensions and window 2x15x15. No partial attention replacement.
- Binary is a two-level output `{0, theta_module}`, not arbitrary per-event
  amplitude or a requirement that every module share one global theta.
- Installed/called/functional ATLIF counts are 105/93/81. See the separate ep34
  audit. Changes to dead-result diagnostic neurons cannot improve flow accuracy.
- C12 ep34 includes five additional epochs. It is not a same-budget replacement
  for the C12 ep29 row in the C00/C10/C12 factorial table.

Most papers below appeared in early/mid-2026, before the August 30 numerical
closure. They are additions to our reviewed literature, not all publications
that appeared after that closure. This search does not establish exhaustive
coverage or any new algorithmic SOTA claim.

## 2. Sources actually read

| Source | Evidence inspected | Reusable principle and limitation |
|---|---|---|
| [MEOM, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/f04957cc30544d62386f402e1da0b001-Paper-Conference.pdf) | Section 3.2, equations 7-12; author code below | Mask-weighted teacher perspectives and progressive supervision. Classification logits and cumulative time averages are not flow fields for different intervals. |
| [FlowAnyTime, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38019/41981) | Methodology, paired-input and intra/inter-frame distillation | Clean/degraded pairs with identical motion support robustness fine-tuning without replacing the backbone. Its RGB/CroCo recipe is not a ready event-SNN recipe. |
| [STSC-Flow, CVPR 2026](https://arxiv.org/html/2605.25570v1) | Section 3, equations 1-16, implementation and dataset protocol | Local structure and trajectory consistency on warped event volumes, introduced through a supervised curriculum. Original model predicts continuous trajectories and uses bidirectional recurrent refinement. |
| [MD-Mixer, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Shi_Temporal_Interaction_in_Spiking_Transformers_with_Multi-Delay_Mixer_CVPR_2026_paper.pdf) | Sections 4.1-4.4, equations 5-12 | Channel-specific weighted delays in K/V. The original channel-specific operator does not generally fold into our channel-shared PSN temporal matrix. |
| [TRE, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_Temporal_Representation_Enhancement_TRE_Learning_to_Forget_Dominant_Patterns_for_CVPR_2026_paper.pdf) | Section 3.2, equations 3-10, Figure 2 | Training-only suppression of repeatedly dominant semantic channels; inference has no mask. Classifier-based relevance must be replaced for dense flow. |

MEOM code inspected at commit `f66347a06910095f3b64be7f244a088b31444e18`:

- [ResNet ImageNet training](https://github.com/KaiSUN1/MEOM/blob/f66347a06910095f3b64be7f244a088b31444e18/MEOM_Resnet/experiment/imagenet/main.py):
  `tpd` compares cumulative mean logits; `train` dispatches timestep masks to the
  ANN teacher. This is the relevant implementation evidence, not just the README.
- [Transformer loss](https://github.com/KaiSUN1/MEOM/blob/f66347a06910095f3b64be7f244a088b31444e18/MEOM_SpikingTransformer/imagenet/loss.py):
  `meom` uses class-logit KL and label-derived dynamic teachers. A function name
  alone does not establish implementation of the full TMPD/TPD paper recipe.
- The inspected `tpd` computes the later target from the same graph without
  detaching it. A frozen/EMA teacher in our proposed variant is an explicit
  design difference, not a verbatim reproduction.

CVF PDF method text for MD-Mixer and TRE was accessible with curl/pdftotext even
where browser-tool PDF retrieval failed. Neither author repository was verified
in this pass; do not claim their implementation was reproduced or tested.

## 3. Ranked candidates

### R1: Same-interval temporal-view distillation (P0, accuracy first)

Inspiration: MEOM and FlowAnyTime. Proposed adaptation, not an established result.

Use the uncorrupted training event interval as the frozen ep34 teacher input.
Give the student a lightly corrupted view of the SAME interval, initially by
dropping a small fraction of input voxel entries while retaining shape, polarity
semantics, temporal-bin order and flow endpoints. This is voxel-view corruption,
not exact raw-event thinning. If exact event thinning is required, construct it
before voxelization and cache the new train-only views.

Keep full-input supervised training in the mix. Add a confidence-masked robust
flow loss between clean teacher and student, in a single documented flow unit.
Teacher predictions are detached. Use only train GT to decide teacher confidence;
never use valid825 GT to build masks, pseudo-labels or schedules. Start with a
frozen teacher for a clean ablation; EMA is a separate later experiment.

Why it differs from H55: H55 uses a baseline PSN teacher on ordinary inputs. This
proposal tests clean-to-corrupted consistency using the current model, not a
claim that an old NB0 or Local5 checkpoint is the stronger teacher. Ep34 is only
expected to be more reliable on clean inputs than on the matched corrupted view;
check that premise on train-only probes before training.

Do NOT average different physical flow intervals or supervise each raw SNN
timestep with the full-interval displacement. Do not interpret the flow decoder's
multi-scale output list as a sequence of classification logits.

Inference change: none. Training cost: an extra teacher forward, or a properly
bound cache. Minimum attribution: continuation control, augmentation-only,
augmentation+distillation. Teacher state reset, BN semantics and transform
alignment must match the existing entrypoint.

### R2: GT-anchored event-structure loss (P1; strongest task-specific candidate)

Inspiration: STSC-Flow. Its warped volume preserves within-bin relative time;
the losses constrain local volume structure and cross-bin gradient consistency.
Its original curriculum eventually reduces GT weight to zero. We should NOT
copy that schedule for short fine-tuning of a supervised endpoint-flow model.

Adaptation: retain the existing supervised objective throughout and introduce a
small event-structure auxiliary term after a warm-up. First audit availability
of train events, timestamps, rectification and endpoint duration. Our cached
voxel tensors do not retain every event's within-bin timestamp. The original
continuous-trajectory VWE loss therefore cannot be reconstructed exactly from
those tensors or obtained by simply calling the current endpoint decoder.

Two distinct options must not be conflated:

1. A coarse voxel-bin warp loss using bin-center times and a stated local
   constant-velocity approximation. This is our discrete approximation.
2. Original-style trajectory consistency, requiring temporal supervision/output
   support that the current architecture does not already provide. Defer this
   rather than silently add a recurrent decoder or arbitrary-time output head.

Use boundary/occlusion masks; keep signed polarities separate where needed to
avoid cancellation. Check identity, known translation, zero-event input, flow
units, timestamp scaling and gradients. Event images are not brightness images:
ordinary RGB photometric equality is not automatically a valid event loss.

Inference change for option 1: none. Potential benefit: supervision beyond
sparse GT locations and improved motion boundaries. Main risks: incorrect warp
semantics, aperture ambiguity and collapsed/over-smoothed flow. First compare
against continuation only; introduce LSC-like and temporal terms separately.

### R3: Training-only dominant-feature suppression (P1, cheap but conflicting)

Inspiration: TRE. Its class-specific contribution uses classifier weights and
accumulated features, which our dense regressor does not have in that form.

Adaptation hypothesis: estimate channel relevance using detached flow-loss
attribution, or a explicitly named cheaper activity proxy, then mildly mask
persistently dominant channels during training. Apply one uniform rule across
all eligible live blocks, not an S2-only attention swap. Disable it at inference.
Use a random-mask control at matched mask rate to establish whether relevance
selection contributes beyond ordinary dropout.

Do not implement real-valued modulation on the exported spike path. Training
binary masks keep retained spikes at their original threshold amplitude.

Important conflict: encouraging temporal diversity can lower score equality and
pattern reuse. Track clean-input accuracy AND both-active pair equality, occupied
classes, spike count and actual hardware-owner work counters. More diverse
features are not inherently better for our hardware. Do not simultaneously add
a score-equality penalty in the first experiment: it would obscure the test.

### R4: Foldable temporal-delay parameterization (P2, constrained adaptation)

Inspiration: MD-Mixer, whose actual equation 7 uses channel-specific delays and
coefficients. Directly importing that K/V operator changes buffering and temporal
addressing; it is not admitted to the frozen DATE graph.

Our existing neuron already performs `h = A_time @ x + b_time`. A restricted
training parameterization `A_eff = A_base + sum(a_d * D_d)` can be exported into
the SAME dense matrix if the delays and weights are channel-shared, linear,
within the existing temporal extent, and at the same side of nonlinearities.
It adds an optimization prior, not new representational capacity relative to an
unconstrained full A_time. Original MD-Mixer is more general than this adaptation.

This is also not M29 rank-3 factorization: there is no rank constraint or claimed
runtime low-rank saving. Start from an exactly matching effective matrix. Verify
materialized float output first, then fixed-point rounding/order. A new tensor
of weights requires recapture even when the inference graph is unchanged.

## 4. Useful, but not newly discovered experiments

| Existing direction | Current audit implication |
|---|---|
| Alpha search | Already performed; final alpha=1/8 +5ep. Do not relabel another alpha scan as a new literature idea. |
| Weak baseline-teacher/angular loss | Already explored. Existing H55 teacher builder deliberately constructs baseline PSN, not the current overlay teacher. |
| PAFT | Existing train-only pattern-codebook regularization. Coordinate with its owner; do not duplicate the queue or reuse valid-derived codebooks. |
| Local5 source gate cardinality | Existing `mean_collapse` and `tail_gap_c2` proxies and QAT/calibration entrypoints. Not a fresh proposal. |
| Class stability | Existing optional class-major paths. Their normalization semantics differ from token-wise multiplicity-preserving Shiftmax; never silently enable them as an exact implementation of C12. |
| ATLIF threshold learning | Ep29/34 theta tensors did not change. Audit optimizer membership, grad scale, clipping and update magnitude before proposing adaptive-threshold accuracy claims. Do not blindly raise target-rate penalties again. |

An exact-code cost regularizer can still be a co-design follow-up, but it must
target the hardware owner's CURRENT operation counts and fixed-point contract.
No claim that Motion itself increases score equality is supported by the old
H67/H81 controls. Reuse and accuracy should be measured separately.

Implementation pointers, all under the H9 overlay:

- `h55_teacher.py`: existing builder omits overlay installation. Not suitable
  for loading C12 ep34 without an explicit new, audited teacher-loading branch.
- `h9_losses.py`: existing flow-distillation wrapper is a possible integration
  point, but flow scaling and mask application require tests. Its EPE term uses
  the returned valid mask, whereas teacher confidence affects the separate
  direction weight. Do not assume enabling confidence masks its EPE term.
- `bsa_attention.py`: Local5 source-cardinality proxies and class-major variants
  already exist. Neither should be changed as part of this literature report.
- `atlif_ternary_psn/installer.py`: existing temporal factor materialization is
  relevant testing infrastructure, not permission to alter M29 experiments.

## 5. Minimal, paired protocol before any run is launched

First batch: R0 continuation, R1a augmentation-only, R1b augmentation+distillation.
All start from the same frozen ep34 and use matched sample order, optimizer
initialization, LR, number of updates and checkpoint-selection schedule.
Pre-register one mild corruption level and one normalized loss weight; do not
begin with a broad multi-dimensional sweep. Establish teacher reliability and
units on train probes first. Keep alpha, BN mode, resolution and neuron output
semantics fixed. A new fine-tuning optimizer is permissible only if all branches
use it and are described as fine-tuning, not exact optimizer-state continuation.

Use a small smoke test only for loading/gradients/OOM, then an initial five full
epochs per branch. This is a screen, not proof of convergence or permanent
failure. If both control and candidate are still improving, extend the paired
runs under the same budget. Advance a stable winner to two additional seeds.
Record extra teacher compute separately from matched student update count.

Keep valid825 unchanged, no official DSEC submission requested. Because valid825
has been used repeatedly, it is a development set, not independent evidence of
generalization. Freeze the recipe before applying it to the established MVSEC
day2 training protocol; avoid test-sequence-driven hyperparameter selection.

Report AEE, AAE-2D, AE-3D, Fl, per-scene error and spikes at every selected
checkpoint. Add profile counters under the SAME deployed numeric configuration.
Use sequence-level paired uncertainty estimates, not independent-pixel error
bars. An illustrative internal promotion target is at least 0.5% relative AEE
gain over the matched continuation control, without a material angular/outlier
regression or loss of the spike budget; this is not a conference requirement.

Load audit remains mandatory: 105 installed ATLIF / 12 replaced attention,
expected overlay keys and missing/unexpected counts for the actual model variant.
Training-only teacher/helper keys must not leak into deploy state. Any replacement
paper checkpoint needs a new SHA-bound standard evaluation and hardware capture;
old ep34 RTL/profile receipts do not transfer to newly fine-tuned weights.

## 6. Watchlist and non-admitted replacements

- [STIRFlow author repository](https://github.com/AhmedHumais/STIRFlow) still
  contains only a forthcoming-code README at this check. Track the recent work;
  do not claim code-level replication or import recurrent refinement blindly.
- [ICPR 2026 directional-filter SNN flow publication](https://www.ibisc.univ-evry.fr/en/2026-publications/)
  is confirmed by the authors' lab, HAL identifier hal-05625805. Full method not
  retrieved in this pass; not promoted to an executable candidate based on title.
- [EmFlow August 2026 thesis announcement](https://carleton.ca/share/2026/thesis-defense-2/)
  describes a hardware-oriented convolutional SNN flow system. It is a recent
  related-system lead, not a verified top-tier paper or a new Transformer parent.
- STSC recurrent architecture, MD-Mixer channel-specific delay buffers, dynamic
  neighborhoods and extra persistent state are deferred. This is a hardware
  compatibility decision, not an empirical claim that these methods fail.

Borrowed methods must be cited. An adaptation only becomes our contribution
after a specific methodological difference and controlled benefit are shown;
renaming an existing method is not new research.
