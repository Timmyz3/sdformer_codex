# MVSEC Train-to-Test Protocol (Revised from Full Text, 2026-08-01)

## 2026-08-01 CICC/Spike-FlowNet Correction

The full CICC 2026 paper is now locally available. It deploys an INT8 Hybrid
U-Net based on Spike-FlowNet and evaluates `800` inputs from each of
`indoor1/2/3` and `outdoor1`. The cited Spike-FlowNet training protocol uses
`outdoor_day2` only for training. Therefore the paper-facing direct-MVSEC route
is corrected to:

- train: `outdoor_day2` only;
- test: `indoor_flying1/2/3` and `outdoor_day1`, never used for training;
- train geometry: random `256x256` crop with horizontal and vertical flips,
  each at probability `0.5`;
- test geometry: center-cropped `256x256` left-camera input;
- primary protocol: `dt=1`, event-masked AEE over pixels containing both events
  and valid ground truth;
- paper audit subset: exactly `800` inputs per sequence, plus a full-sequence
  result reported separately.

The Spike-FlowNet reference trains self-supervised for `100` epochs with Adam,
batch `8`, initial LR `5e-5`, LR multiplied by `0.7` every five epochs through
epoch 10 and every ten epochs thereafter. For `dt=1`, it uses `N=5`, smoothness
weight `10`, and IF threshold `0.75`. These are reference-model settings, not
settings that may be silently applied to SDformer. An SDformer direct-MVSEC run
must use the same `outdoor_day2 -> four test sequences` split for NB0 and the
candidate, disclose its supervised loss and architecture-specific settings, and
must not claim exact reproduction of the Spike-FlowNet optimizer/model.

The `outdoor_day1 -> indoor1/2/3` split below is retained only as a legacy
in-domain fallback. It is not CICC-comparable and is no longer the preferred
paper-facing route.

## Purpose

This is the fallback/second-dataset validation when an official DSEC test submission is unavailable or when DATE needs an additional dataset. It is not a reproduction of SDformerFlow's MDR-to-MVSEC experiment.

## Data Readiness

The local dt1 preprocessing is complete for:

| sequence | event files | flow files | role |
|---|---:|---:|---|
| outdoor_day2 | 14977 | 2629 | paper-facing training source; required train/validation range audited |
| outdoor_day1 | 11936 | 11908 | primary test; legacy fallback training only |
| indoor_flying1 | 2205 | 2177 | unseen test |
| indoor_flying2 | 2664 | 2636 | unseen test |
| indoor_flying3 | 2951 | 2923 | unseen test |

The loader's valid timestamp ranges remain the upstream ranges. Results must use valid GT pixels and, for comparison with prior MVSEC work, also report the event-masked metric.

## Legacy In-Domain Split

- Train: `outdoor_day1` only.
- Validation for checkpoint selection: a fixed tail block from `outdoor_day1`, removed from training before any run.
- Test: `indoor_flying1`, `indoor_flying2`, and `indoor_flying3` only.
- Do not report `outdoor_day1` as test after training on it.
- Do not initialize from DSEC or MDR checkpoints. First train NB0 on this MVSEC split, then initialize the compressed H67/final candidate from the MVSEC NB0 checkpoint.

This split is useful for an internal equal-budget comparison only. Do not place
its absolute AEE beside CICC 2026 or Spike-FlowNet results because `outdoor_day1`
is part of their held-out evaluation set.

## Legacy Hardware-Compatible Geometry

H67 hardware is frozen around `window=[2,9,9]`. The upstream MVSEC route commonly uses `256x256` with an 8x8 deepest feature map, which changes the attention tile and is not hardware-equivalent.

For the hardware-consistent comparison:

- use `288x288` model input and `window=[2,9,9]`;
- center-crop width and zero-pad height from `260x346` to `288x288`;
- set the valid mask to zero over padded rows;
- apply the identical geometry to NB0 and the final candidate.

This route supports a fair internal NB0-vs-candidate comparison on MVSEC, but it must not be described as the paper's original `256x256/window=8` protocol.

## Primary Paper-Facing Required Runs

1. Download and preprocess `outdoor_day2`, including event stream, grayscale
   frames, calibration, timestamps, and flow ground truth needed by the selected
   training objective.
2. Train MVSEC-NB0 from random initialization on `outdoor_day2` only.
3. Train the one frozen final candidate from the MVSEC-NB0 checkpoint using the
   identical samples, objective, geometry, optimizer budget, and seed.
4. Evaluate both checkpoints on `indoor_flying1/2/3 + outdoor_day1` using the
   same event mask. Report the CICC-style fixed `800` inputs per sequence and
   full-sequence results in separate columns.
5. Report AEE/outlier, spikes, executed operators, attention-inclusive energy,
   and the loading audit. Do not use AAE as a primary MVSEC metric.
6. Add seeds 1 and 2 only after the candidate passes the seed-0 AEE/spike gate.

## Legacy In-Domain Required Runs

1. MVSEC-NB0 supervised training from random initialization.
2. MVSEC-final-candidate training from the frozen MVSEC-NB0 checkpoint, preserving all12 attention and neuron coverage.
3. Standard inference on all three indoor sequences with AEE, AE-3D/Barron, outlier rate, spikes, and attention-inclusive cost. Legacy AAE-2D may be retained in artifacts for historical debugging but must not be used for paper comparison.
4. At least three seeds for the final candidate and its equal-budget no-motion control if MVSEC is used as a main paper table.

## Candidate Gate and Minimum Matrix

A DSEC candidate is eligible for MVSEC only if its standard valid825 result satisfies all of the following:

- AEE no worse than NB0 by 5% (`AEE <= 1.5616` using NB0 `1.4872`);
- total spikes reduced by at least 20% (`<= 35.24G` using NB0 `44.05G`);
- one uniform all12 attention formula, no native carrier, and hardware-countable operators;
- `ATLIFTernaryPSN=105`, attention modules=12, and trained checkpoint `missing=0/unexpected=0`.

Do not train every passing candidate on MVSEC. Freeze at most one new DSEC winner, then run the minimum matrix:

1. MVSEC-NB0, seed 0, from scratch;
2. current deploy reference H67/TTX, seed 0, initialized from the frozen MVSEC-NB0 checkpoint;
3. final new winner, seed 0, initialized from the same MVSEC-NB0 checkpoint;
4. only if the new winner beats H67 on MVSEC AEE while retaining the spike target, add seeds 1 and 2 for NB0 and the winner.

The primary MVSEC table uses AEE/outlier plus spikes and attention-inclusive cost. AE-3D is secondary because the original SDformerFlow MVSEC table is sparse event-masked evaluation, whereas the DSEC `4.871` value is an official hidden-test benchmark result.

## Launch Gate

Do not launch candidate training until the DSEC rescue freezes the final attention
candidate. `outdoor_day2` download and preprocessing may proceed independently,
but the generated sample count, timestamps, grayscale frames, event tensors, flow
ground truth, and train/test non-overlap must be audited before training.

The CICC-style deployment ablation does not require waiting for a new MVSEC-trained
checkpoint. It may first run on the existing frozen MDR-to-MVSEC NB0 and TTX
checkpoints to validate counters and hardware trends. `outdoor_day2` is required
only for the separate paper-facing direct-MVSEC training table and for comparing
absolute AEE under the Spike-FlowNet split.

## CICC 2026 Deployment Evaluation (Corrected from Full Text, 2026-08-01)

Reference: Tao Zhang et al., "A 28-nm Optical Flow Estimation Accelerator with
Redundancy Speculation, Bit-Width-Aware Compression and Similarity Detection,"
CICC 2026, DOI `10.1109/CICC65509.2026.11509564`.

The full text verifies that the accelerator deploys a Spike-FlowNet-derived Hybrid
U-Net with SNN encoder levels and ANN flow-estimation/decoder levels. Its weights
are quantized to INT8. The three actual mechanisms are:

1. `BWAC`: group `16` adjacent INT8 weights, store the minimum signed bit width
   plus a non-zero bitmap, and decompress losslessly on chip. This reduces weight
   EMA by `71.4%`.
2. `Dense-Channel-First Speculation`: sort input channels by measured channel
   density, evaluate MaxPool/ReLU after a channel threshold, speculate output
   addresses that will be non-maximum or negative, and clock-gate their remaining
   channel computations. The paper reports up to `73.8%` redundancy reduction.
3. `DLSS`: always execute U-Net Level 0; compare `FM_i` and `FM_(i+delta)` using
   average L1 similarity, and execute deeper levels only when similarity is below
   threshold `theta_s`. The similarity check is mapped with concatenated feature
   maps, `+1/-1` weights, accumulation, and a threshold comparator.

The paper does not disclose numerical values for the channel threshold or
`theta_s`. They must be selected on a validation set, never on the four test
sequences. TTB is not a mechanism in this CICC paper and must be presented as a
separate project idea rather than attributed to it.

Use the paper's cumulative ablation order on one frozen checkpoint:

| ID | cumulative execution policy | exactness/accuracy contract |
|---|---|---|
| C0 | INT8 dense baseline, no BWAC/speculation/DLSS | numerical reference |
| C1 | C0 + BWAC | lossless weights; output must equal C0 |
| C2 | C1 + Dense-Channel-First Speculation | report AEE delta and speculative-stop statistics |
| C3 | C2 + DLSS | report AEE delta, skip interval, deep-level execution rate, and fallback |

For the original Hybrid U-Net, the paper reports the following `800`-sample
per-sequence results:

| metric | indoor1 | indoor2 | indoor3 | outdoor1 | mean |
|---|---:|---:|---:|---:|---:|
| baseline AEE, INT8 | 0.84 | 1.32 | 1.14 | 0.52 | 0.96 |
| AEE with all features, INT8 | 0.87 | 1.35 | 1.17 | 0.56 | 0.99 |
| channel proportion for speculation | 33.41% | 33.41% | 59.68% | 26.23% | 38.18% |
| remaining MP/ReLU redundancy | 0.53x | 0.57x | 0.82x | 0.26x | 0.54x |
| mean DLSS interval, delta | 15.77 | 9.38 | 6.71 | 11.96 | 10.95 |
| overall operations | 0.18x | 0.20x | 0.25x | 0.17x | 0.20x |
| EMA | 0.08x | 0.09x | 0.10x | 0.08x | 0.08x |
| energy | 0.10x | 0.12x | 0.14x | 0.11x | 0.12x |
| latency | 0.17x | 0.21x | 0.24x | 0.19x | 0.19x |

For TTX, adapt mechanisms only where operator semantics remain auditable:

- BWAC applies to INT8 convolution/projection weights; binary/ternary activation
  packing remains a separate exact storage optimization.
- DLSS may use binary-feature XOR/popcount as the L1 similarity detector, but it
  needs a valid shallow flow output and validation-selected threshold.
- Dense-channel speculation cannot be claimed exact for Shiftmax/TTX scores.
  It requires a bound or an empirical error-controlled stop rule before inclusion.
- TTB density scheduling remains an orthogonal DATE mechanism and needs its own
  ablation rather than being renamed as CICC speculation.

For each C0-C3 row report:

- per-sequence and aggregate AEE, outlier rate, and valid-pixel count;
- total and executed TTX/neuron/decoder operations, not only spike count;
- on-chip read/write bytes, off-chip read/write bytes, and metadata traffic;
- measured or cycle-model latency, detector/dispatcher/pack overhead, and total
  energy including compute, SRAM, DRAM, and control;
- ratios normalized to C0 on the same checkpoint and identical input traces.

Use both the paper's unweighted four-sequence mean and a valid-pixel-weighted
aggregate. Do not average ratios produced from different checkpoints. Permit one
preregistered similarity operating point and one validation-only sensitivity curve;
do not sweep thresholds against the test set to select the reported point.

This CICC-inspired matrix is a hardware/software co-design result and does not replace
the required NB0-versus-candidate training and standard inference table.

## Project Matrix Following the CICC Experimental Organization

The project follows the paper's experimental organization, not its model-specific
MaxPool/ReLU mechanism. Freeze one deterministic manifest of `800` dt1 inputs from
each of `indoor_flying1/2/3` and `outdoor_day1`, and reuse exactly that manifest for
every row.

First report the model-level table:

| row | checkpoint/numerics | purpose |
|---|---|---|
| M0 | NB0 floating point | accuracy baseline |
| M1 | NB0 INT8/hardware order | baseline quantization loss |
| M2 | final TTX floating point | algorithm and spike improvement |
| M3 | final TTX INT8/hardware order | deployable main result |

Then freeze M3 and report the cumulative deployment table:

| row | cumulative feature | relation to CICC experiment |
|---|---|---|
| D0 | fixed-width, all bundles executed, no temporal skip | dense reference |
| D1 | + lossless weight BWAC and binary/ternary activation packing | compression counterpart |
| D2 | + exact-empty TTB skip and density dispatch | project counterpart of redundancy elimination |
| D3 | + validation-selected feature-similarity deep-level skip | DLSS counterpart |

Do not replace D2 with the paper's MaxPool/ReLU speculation because that operator
contract does not match TTX. D2 is a project mechanism and must be named TTB, while
the CICC paper is cited only as the experimental methodology and hardware motivation.

For every model row report AEE/outlier/spikes/energy. For every deployment row report
per-sequence and mean AEE, AEE degradation from D0, active TTB proportion, remaining
executed-operation ratio, average deep-level interval, EMA ratio, energy ratio,
latency ratio, detector/metadata overhead, and area overhead. Add cumulative waterfall
plots for EMA, energy, and latency, matching the structure of CICC Fig. 9.

Unlike the fabricated CICC chip, this project must label synthesis or cycle-model
numbers as estimates. Do not present voltage-sweep, measured power, or silicon
TOPS/W until measured hardware exists. Use one declared DRAM energy/bandwidth model
for D0-D3 and include all feature/control metadata in the traffic count.

Current implementation status (2026-08-11):

- deterministic train/validation and four-sequence fixed800 manifests now exist;
  every referenced flow/event pair has passed the dataset audit, and full-sequence
  evaluation remains a separate output;
- direct training now has random-256 crop, source-frame valid-FOV handling,
  workers8 seeded data order, local-only checkpointing, validation-loss selection,
  and NB0-to-candidate loading audits; CuPy/AMP is seeded but not claimed bit-exact;
- `run_h9_standard_mvsec_eval.py` persists fixed800/full summaries with AEE,
  valid-pixel counts, GT-magnitude Fl, legacy prediction-magnitude outlier,
  spikes and spike-proxy energy, but still has no EMA,
  cycle, compression-metadata, or detector-overhead columns;
- TTB profile/cycle tools exist for profile100 traces, but are not yet wired
  to the same MVSEC samples or checkpoint fingerprints;
- the algorithm-level NB0/H67/Local5 train-and-infer queue is active; the full
  CICC-style hardware table still requires a trace-to-cycle/traffic summarizer and
  M0-M3 INT8/hardware-order evaluation after the floating-point winner is frozen.
