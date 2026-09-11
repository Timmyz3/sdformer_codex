# Frozen observations (independent generation MUST use only this)

Date of this freeze: 2026-09-11. These are reported measurements, not claims of novelty.

## System
- Task: event-camera 2D optical flow, DSEC valid825 AEE.
- Frozen student family: Motion C12 / H67 / ep34.
- Neuron: official AT-LIF. Output is two-level \(\{0,\theta\}\). At inference, layer-shared \(\theta\) is absorbed into the next \(W\); transmitted spikes are then \(\{0,1\}\). This is the locked identity (user 2026-09-11). It is not a per-event analog payload and not a non-absorbable int8 contract.
- Expensive region historically: patch embed residual r1 (two convs) plus T10 PSN; old activity-weighted-dot ledger put whole patch ~35% of that proxy. Proxy ≠ new student's cycle share.
- Dual consumers after source: spike/gate path and continuous residual/PED path. Native projection convolution + BN + residual add also exist.
- Time: noncausal T10 at this PSN.

## Reported numbers
- Ordinary dense-source/raw valid825 AEE = 1.219801338
- Learnable 40-coeff lifting T10 raw AEE = 1.232979368 (delta +0.013178 vs ordinary; absolute budget 1.259 passes, relative +0.005 fails)
- Source graph after official CSE: ordinary 260 add/sub per T10 vector; lifting 159 add/sub + 35 intermediate RNE/sat
- Same two-stage SIMD source resource: always-ready slots 6938 → 5354 (−22.83%); long backpressure both 8088 (gain absorbed). Part of −22.83% is generic round→sat fusion.
- Separate integer consumer model (different resource point): 758777 → 714889 (−5.78%). Do not add the two tables.
- Native projection BN on captured students uses **actual batch statistics over the full 10×96×120×160 domain**, not frozen running stats. Local-window replay given free mean/var undercharges wait/storage.
- Integer gates / I24 / PED q24 on two captured windows: 0 difference vs model. FP32 has CUDA rounding differences.

## Venue
IEEE TCAS-II Express Briefs: 5 pages, one mechanism, circuits+systems co-design, reviewers punish missing **relative prior** and missing **measured advantage**. Binary accept/reject. Need novelty that is not a rename, AND performance that survives same-port/same-state/same-backpressure.

## Hard identity
AT-LIF is \(\{0,\theta\}\) with \(\theta\) absorbed into the next \(W\) at inference. Do not sell non-absorbable int8 θg as the frozen identity. Do not sell analog CIM. Do not quote OpenROAD/Yosys as foundry PPA. Do not multiply component speedups into FPS. See `IDENTITY_ATLIF.md`.

## What independent ideation should produce
For each idea: ID, one sentence, A (complete prior to copy), B (hole in THIS net using the observations above), X (why not a reskin), strongest controls, kill-gate number, two-sentence TCAS-II pitch, biggest objection.
Do not assume any existing idea catalog is correct. Do not propose “add 18 modules.”
