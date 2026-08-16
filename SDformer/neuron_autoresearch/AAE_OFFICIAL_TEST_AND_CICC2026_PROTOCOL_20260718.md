# AAE Official-Test Gap and CICC 2026 Emulation Protocol

Date: 2026-07-18

## Decision

The corrected DSEC/Barron AE-3D is a trustworthy metric for comparisons made on the
same local split. It cannot reproduce the paper's `AE=4.871` from the current NB0
checkpoint because the model, resolution, split, inference state, and aggregation
protocol are not the paper's official-test route.

Do not replace DSEC with MVSEC merely because local AE-3D is about 9 degrees. First
complete the DSEC queue, then perform a full-resolution control and generate an
official DSEC submission. MVSEC remains a second-dataset experiment.

## Why Local AE-3D Does Not Reach 4.871

| dimension | paper / official result | current local audit |
|---|---|---|
| reported set | seven hidden DSEC test sequences | 825 samples held out from DSEC training sequences |
| checkpoint training | 80 crop epochs plus 30 full-resolution fine-tune epochs | NB0 epoch59, 60 crop epochs |
| spatial evaluation | full `480x640` | center crop `288x384` |
| optimizer headline | AdamW, initial LR `1e-3`, WD `0.01`, LR halved every 10 epochs | NB0 initial LR `1e-4`, different milestone list |
| inference normalization | paper disables BN running-state tracking at test | ordinary `model.eval()` with checkpoint running statistics |
| timestamps | official submission references at 2 Hz; each flow spans about 100 ms | local preprocessed validation list |
| aggregation | official server all-sequence result | unweighted mean of per-sample valid-pixel means |
| model selection | final official submission checkpoint | local crop baseline/candidate checkpoint |

Primary evidence:

- The SDformerFlow paper states 80 crop epochs followed by 30 full-resolution
  fine-tune epochs and disabled BN running-state tracking during test:
  https://dsec.ifi.uzh.ch/wp-content/uploads/sourcenova/uni-comp/optical-flow-benchmark-v1-0/submissions/269/details.pdf
- The DSEC submission protocol evaluates seven specified test sequences at reference
  timestamps sampled at 2 Hz, with approximately 100 ms flow intervals:
  https://dsec.ifi.uzh.ch/optical-flow-submission-format/
- The official SDformerFlow page reports all-sequence `EPE=1.602`, `AE=4.871` and
  strongly sequence-dependent AE values from `3.318` to `7.248`:
  https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/sdformerflow/

The earlier metric fix remains necessary: legacy local AAE uses the 2-D angle of
`(u,v)`, whereas benchmark-facing AE uses the Barron `(u,v,1)` definition. The
remaining gap is not evidence that the new formula is wrong.

## DSEC Closure Protocol After the Current Queue

1. Evaluate NB0 and the frozen final candidate on the same local valid825 at full
   `480x640`, using the upstream `v1` positional interpolation path. Run two BN
   controls: normal eval running statistics and paper-style no-running-state mode.
2. Fine-tune both NB0 and the final candidate at full resolution with equal budgets.
   The paper-faithful target is 30 epochs; use a short health check only to validate
   memory and loading, not to rank the final result.
3. Add a DSEC-test submission writer. The current `eval_DSEC_flow_SNN.py` implements
   only `mode=valid`; passing another mode performs no inference. The new writer must
   consume the 416 specified test samples, preserve official filenames and encode
   flow PNGs in DSEC format.
4. Audit ATLIF count, attention count, overlay keys, missing/unexpected keys, full
   image geometry, relative-position interpolation, and BN policy before submission.
5. Report local valid825 and official test in separate table columns. Never substitute
   one for the other.

## MVSEC Fallback and Second-Dataset Rule

All four local dt1 routes are ready: `outdoor_day1` and `indoor_flying1/2/3`.

- Paper-comparable line: train from scratch on MDR (`256x256`, window 8, 50 epochs)
  and evaluate event-masked sparse flow on all four MVSEC sequences. The SDformerFlow
  paper reports AEE and outlier only, not AAE, for this table.
- Additional train-to-test line: train MVSEC-NB0 on `outdoor_day1`, hold out a fixed
  tail validation block, and test on indoor1/2/3. Initialize the final compressed
  model from that MVSEC-NB0 checkpoint. Label this `MVSEC in-domain split`; it is not
  an MDR reproduction.
- If DATE space permits only one MVSEC table, prefer the MDR reproduction because it
  matches the published SDformerFlow comparison. Use MVSEC train-to-test as a
  robustness appendix or fallback when MDR convergence cannot be reproduced.

Full-text correction: the CICC 2026 hardware comparison provides a second defensible
paper-facing convention. For that convention, train only on `outdoor_day2` and test
`indoor1/2/3 + outdoor1` with event-masked dt1 AEE. This can replace the expensive MDR
route when the claim is direct-MVSEC robustness/hardware evaluation, but its absolute
AEE must not be compared to SDformerFlow's MDR-to-MVSEC row. If the SDformerFlow
external baseline row is retained, keep the MDR route as a separately labeled table.

## CICC 2026 Paper and What Can Be Emulated (Full-Text Correction 2026-08-01)

Paper: Tao Zhang et al., "A 28-nm Optical Flow Estimation Accelerator with
Redundancy Speculation, Bit-Width-Aware Compression and Similarity Detection,"
CICC 2026, DOI `10.1109/CICC65509.2026.11509564`.

The four-page full text is now available at
`hw_autoresearch_nts07/docs/Zhang 等 - 2026 - A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression.pdf`.
It verifies the following quantitative claims:

- operations reduced to `0.20x`;
- external-memory access reduced to `0.08x`;
- corresponding energy and latency reduced to `0.12x` and `0.19x`;
- EMA-included efficiency of `14.07 TOPS/W`.

It uses an INT8 Hybrid U-Net referring to Spike-FlowNet and evaluates `800` inputs
from each of `indoor1/2/3` and `outdoor1`. Baseline AEE is
`0.84/1.32/1.14/0.52` (mean `0.96`); all three features produce
`0.87/1.35/1.17/0.56` (mean `0.99`). The cited Spike-FlowNet protocol trains on
`outdoor_day2`, not `outdoor_day1`, and tests center-cropped `256x256` dt1 inputs
with event-masked AEE. The CICC paper does not disclose its numerical channel or
similarity thresholds.

### Software Experiments

Use one frozen model/checkpoint and follow the paper's cumulative feature order:

| ID | execution policy | measured purpose |
|---|---|---|
| C0 | INT8 dense, no BWAC/speculation/DLSS | same-model reference |
| C1 | C0 + lossless group-16 BWAC | weight EMA reduction |
| C2 | C1 + dense-channel-first speculation | MaxPool/ReLU redundant-operation elimination |
| C3 | C2 + DLSS | dynamic deep-level skipping from feature similarity |

For every row report MVSEC event-masked AEE, skipped operations, executed operations,
on-chip bytes, off-chip bytes, estimated cycles, and total energy including SRAM,
control, and DRAM. Accuracy loss must be measured against C0 on the identical
checkpoint. Threshold sweeps are allowed only on validation data. TTB is orthogonal
project work and is not attributed to this CICC paper.

### Hardware Experiments

Map the software rows to auditable hardware blocks:

- group-16 minimum-bit-width/non-zero-map storage and BWADU;
- channel-density ordering, speculative address buffer, and load/skip detector;
- feature-map similarity detector and shallow/deep-level dispatch;
- binary-feature XOR/popcount specialization for the TTX model;
- counters for speculated addresses, deep-level skips, false skips, bytes avoided,
  and control/decompression overhead.

Report synthesized area/frequency/power for C0 through C3, SRAM/DRAM energy included,
and a breakdown showing that detector/metadata overhead is smaller than the saved
compute and traffic. Normalize all ratios to C0 on the same trace. This is the useful
CICC-style lesson for DATE: do not report only peak TOPS/W or spike proxy energy.

## Launch Order

1. Finish H66/H81/H73-H80 queue.
2. Freeze final candidate by valid825 AEE plus hardware cost.
3. Run full-resolution DSEC controls and build official submission.
4. Acquire and preprocess `outdoor_day2`; use it as the only training sequence for
   the paper-facing direct-MVSEC route.
5. Run C0-C3 trace/cycle/traffic ablation on NB0 and the final candidate over the
   four held-out sequences. Keep MDR-to-MVSEC as a separate SDformerFlow comparison.
