# M162 PAFT ep4 BN-policy valid825 independent hammer review

Verdict: **90/100, `PASS_PAIRED_EVALUATION__REJECT_CURRENT_HARDWARE_ACCURACY_PROMOTION`**.
The two 825-frame evaluations are complete, paired to the same PAFT checkpoint,
and internally consistent.  The lower `no_running` AEE of `1.309925` is an
algorithm/evaluation result under current-sample BatchNorm; it is **not** a
deployable accuracy result for the present statically folded hardware and is
not a speedup.  `running` AEE `1.469151` is the only result in this pair whose
BN parameters are eligible to become inference constants, but eligibility is
not yet a folded/quantized hardware proof.

## Independent evidence checks

- The source `manifest.sha256` passes for all 9 listed artifacts.  Its own
  SHA-256 is
  `34e9303d46d678fa424583136131541dba1180488c6fb63e26d21bf968e1359d`.
- Both logs name the same checkpoint and each independently reports strict
  overlay load `missing=0`, `unexpected=0`, 105 installed ATLIF modules and
  12 installed Shiftmax-attention modules.  The launcher receipt checkpoint
  SHA is
  `cf4833b2a53e088ce698d4677822d60539126e8d89dfe239469181ba362e9cca`.
- The `no_running` log explicitly reports 78 BatchNorm modules.  The `running`
  log does not repeat the count, but the complete pre-policy model dumps are
  byte-identical and both policies use the same evaluator/config/checkpoint;
  therefore 78 is a sound paired-architecture inference, not an independently
  printed running-policy count.
- Each CSV has exactly 825 data rows, 825 unique file identities and 18
  sequences.  File order, sequence, valid-pixel count, ground-truth magnitude
  and element count match row-for-row.  Each profile also records 825 samples,
  18 sequences, 93 profiled spike layers and validation-list SHA
  `7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0`.
- Independent sums recover 48,152,523 valid pixels and
  1,285,356,600,000 profiled elements for each policy.  They recover exactly
  83,939,988,024 and 82,728,331,647 spikes, hence firing rates
  `0.06530482515435794` and `0.0643621635015528`.
- From the 10-decimal CSV rows, the independent frame-equal AEE means are
  `1.309925156899635` and `1.469150668812973`; their delta is
  `-0.159225511913338`.  This agrees with the full-precision profile delta
  `-0.15922551364609694`; the few-nanounit difference is explained by CSV
  decimal truncation.  `no_running` is lower on 657/825 paired frames and
  higher on 168/825.
- The exact spike delta is +1,211,656,377 for `no_running`, or +1.464620829%
  versus `running`.  This is activity, not measured cycles or energy.
- `docs/359_DATE终局冻结_20260813.md` remains unchanged at SHA-256
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## Hardware semantic ruling

With evaluation batch size 1, `no_running` still computes per-channel moments
from the current sample's spatial/temporal population.  Its output therefore
depends on input data.  It cannot be replaced by one checkpoint-time affine
constant.  It could become a hardware accuracy result only after the design
implements all 78 current-sample moment reductions, variance/epsilon,
reciprocal-square-root and affine transforms with a verified fixed-point
contract, buffering/barriers, cycle cost and accuracy replay.  The present
receipt proves none of that end-to-end path.

`running` uses frozen population mean/variance at inference and is therefore
eligible for ordinary BN folding.  Before calling it hardware accuracy, the
actual fold must be exported (including any induced bias), quantized, replayed
against the software running-policy graph and connected to the RTL numeric
contract.  M162 establishes the correct deployment direction, not that final
bridge.

The `1.309925` number is AEE (lower is better).  It is neither `1.309925x`
speedup nor a hardware headline.  Likewise the 1.4646% spike change is only a
profiled activity delta and cannot be promoted to cycle or energy savings.

## Algorithm feedback ruling

BN running-stat recalibration with frozen weights, followed by another
`running` valid825, is the right next low-risk experiment.  It must reset the
running moments, use only a representative training/calibration population
(no validation leakage), keep weights/PAFT masks/ATLIF thresholds fixed, put
only BN moment updates in calibration mode, and record sample order, count,
momentum/cumulative policy and final state SHA.  The post-calibration
checkpoint then needs the same strict-load and running-policy valid825 plus an
actual fold/quantization replay.

Recalibration is plausible but not guaranteed: per-sample normalization can
exploit input-dependent moments that a single population estimate cannot
represent.  If the `0.159226` AEE gap remains, retraining should make the
deployment graph explicit through running-stat/fold-consistency supervision
or fold-aware quantization-aware training.  M162 alone also does not prove a
PAFT benefit over the non-PAFT control checkpoint; that requires same-policy,
same-825 control evaluation.

## P0

1. Do not promote `no_running` AEE `1.309925` as present-hardware accuracy,
   cycle speedup, system speedup or headline evidence.
2. Complete BN recalibration, rerun `running` valid825, and prove exported
   folded/quantized graph equivalence before any statically folded hardware
   accuracy claim.
3. Run the paired non-PAFT control checkpoint on the identical 825 population
   and BN policy before claiming PAFT accuracy improvement.

## P1

1. Make the evaluator print `modules=78` for `running` as well as
   `no_running`; the current count is valid by paired model identity but is
   asymmetric in the logs.
2. Seal the calibration dataset identity, order, BN update rule, number of
   passes and final checkpoint/state hashes, and prohibit validation leakage.
3. Keep the full-precision profile metric as canonical; CSV values are rounded
   to 10 decimals and reproduce only within about `2.3e-9` AEE.
4. Treat spike/energy fields as activity proxies only.  They exclude the
   dynamic-BN or folded-BN implementation, memory, control and physical power.
5. If dynamic BN is retained as a fallback, finish 78-channel moment/fixed-
   point RTL, VCS numeric replay and cycle/physical accounting rather than
   assuming `no_running` is free.

Machine-readable findings are in
`m162_paft_independent_hammer_review_r1.json`.  This review only creates files
inside its own review directory and does not modify source receipts or frozen
documents.
