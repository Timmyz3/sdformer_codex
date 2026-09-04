# DATE Paper Result Tables

Frozen on 2026-09-01. This document separates local controlled experiments,
cross-dataset experiments, and official hidden-test results. Lower is better for
all error, spike, and energy columns.

## Claim and protocol rules

- `DSEC local valid825` is an internal validation population. It must not be mixed
  with DSEC official hidden-test rows.
- The strict DSEC ablation uses the same NB0 ep29 parent, seed 0, fresh optimizer,
  30 full-resolution epochs, `480x640`, and `T2x15x15` windows.
- The strict MVSEC ablation uses the same MVSEC NB0 ep11 parent, seed 0, fresh
  optimizer, day2-only training/held-out validation, and 30 epochs.
- Spike energy is an activity proxy. It does not include all attention control,
  memory, score, or reduction costs and must not replace the hardware energy table.
- `AE-3D` is the standard Barron/Middlebury angular-error formula, not a metric
  proposed by this work.

## Main Table 1: DSEC strict two-contribution ablation

Use this table for causal claims. Do not replace its C12 row with the optimized
ep34 row because ep34 has five additional epochs.

| Model | Binary ATLIF | Complete TTX | alpha | Epoch | AEE | AAE-2D | AE-3D | Fl (%) | Spikes (G) | Energy proxy (uJ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C00: PSN + original SDSA | 0 | 0 | 0 | 29 | 1.260341 | 5.951858 | 5.552478 | 5.8376 | 138.4041 | 117260.96 |
| C10: binary ATLIF + original SDSA | 1 | 0 | 0 | 29 | **1.206569** | 5.459142 | 5.198405 | 5.4074 | 72.2082 | **62711.96** |
| C12: binary ATLIF + Complete TTX | 1 | 1 | 1/4 | 29 | 1.209877 | **5.406798** | **5.148612** | **5.4029** | **72.0363** | 63130.12 |

Paper deltas:

- C00 to C10: AEE -4.27%, AE-3D -6.38%, Fl -7.37%, spikes -47.83%.
- C00 to C12: AEE -4.00%, AE-3D -7.27%, Fl -7.45%, spikes -47.95%.
- C10 to C12: AEE +0.27%, AE-3D -0.96%, Fl -0.08%, spikes -0.24%.

Recommended statement: ATLIF supplies the main sparsity and accuracy gain. Complete
TTX preserves the same low-activity regime and improves angular/outlier quality
while giving the encoder one uniform attention datapath.

## Main Table 2: DSEC final deployment model

This table identifies the final model used by the ep34 hardware capture. The strict
C00/C10 rows are references; the final C12 row is an optimized deployment result,
not a same-budget causal row.

| Model | Training status | AEE | AAE-2D | AE-3D | Fl (%) | Spikes (G) | Energy proxy (uJ) |
|---|---|---:|---:|---:|---:|---:|---:|
| C00 ep29 | strict full30 | 1.260341 | 5.951858 | 5.552478 | 5.8376 | 138.4041 | 117260.96 |
| C10 ep29 | strict full30 | 1.206569 | 5.459142 | 5.198405 | 5.4074 | 72.2082 | 62711.96 |
| **C12 ep34, alpha=1/8** | C12 ep29 + true-resume5 | **1.199514** | **5.400641** | **5.106363** | **5.3138** | 72.8912 | 63865.91 |

Final C12 versus C00: AEE -4.83%, AE-3D -8.03%, Fl -8.97%, spikes -47.33%.
Final C12 versus C10: AEE -0.58%, AE-3D -1.77%, Fl -1.73%, spikes +0.95%.

## Main Table 3: MVSEC strict same-parent full-sequence ablation

| Model | Macro AEE | Weighted AEE | Macro Fl (%) | Spikes (G) | Energy proxy (uJ) |
|---|---:|---:|---:|---:|---:|
| C00: PSN + original SDSA | 1.889594 | 2.417774 | 19.5309 | 258.0976 | 220025.48 |
| C10: binary ATLIF + original SDSA | 1.799195 | 2.259916 | 17.9547 | **125.1181** | **107618.78** |
| **C12: binary ATLIF + Complete TTX** | **1.767113** | **2.230047** | **17.1276** | 140.6647 | 121555.15 |

Paper deltas:

- C00 to C10: macro AEE -4.78%, weighted AEE -6.53%, Fl -8.07%, spikes -51.52%.
- C00 to C12: macro AEE -6.48%, weighted AEE -7.76%, Fl -12.31%, spikes -45.50%.
- C10 to C12: macro AEE -1.78%, weighted AEE -1.32%, Fl -4.61%, spikes +12.43%.

Recommended statement: ATLIF is the principal activity-reduction mechanism. TTX
spends part of that activity margin to improve aggregate endpoint and outlier quality.

## Main Table 4: MVSEC fixed800 per-sequence comparison

| Model | OD1 AEE | IF1 AEE | IF2 AEE | IF3 AEE | Macro AEE | Weighted AEE | Macro Fl (%) | Spikes (G) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C00 | 0.8481 | 1.6901 | 2.8169 | 2.1979 | 1.8883 | 2.5257 | 19.5239 | 99.8024 |
| C10 | **0.7940** | 1.6525 | 2.6564 | 2.0912 | 1.7985 | 2.3661 | 17.9456 | **49.1790** |
| **C12** | 0.8181 | **1.5850** | **2.6212** | **2.0352** | **1.7649** | **2.3287** | **17.0631** | 55.1700 |

Claim boundary: C12 improves all three indoor sequences and the aggregate metrics,
but does not beat C10 on OD1. Do not claim per-sequence universal dominance.

## Main Table 5: DSEC official hidden-test context

Published/server results must be rechecked at manuscript freeze. The proposed row
must remain empty until an official DSEC submission is completed.

| Method | Type | EPE | AE | 3PE (%) | Source/status |
|---|---|---:|---:|---:|---|
| BAT | ANN | 0.655 | 2.439 | 1.773 | AAAI 2026 |
| IDNet | ANN | 0.719 | 2.723 | 2.036 | DSEC server |
| EDCFlow | ANN | 0.720 | 2.650 | 2.100 | CVPR 2025 |
| TMA | ANN | 0.743 | 2.684 | 2.301 | DSEC server / ICCV 2023 |
| E-FlowFormer | ANN | 0.759 | 2.676 | 2.446 | DSEC server |
| E-RAFT | ANN | 0.788 | 2.851 | 2.684 | DSEC server |
| SDformerFlow-v2 | SNN | 1.602 | 4.871 | 10.051 | nearest SNN parent |
| OF_EV_SNN | SNN | 1.707 | 6.338 | 10.308 | prior SNN |
| **Proposed ATLIF + Complete TTX** | SNN | **pending** | **pending** | **pending** | official submission required |

This table provides external context only. The paper's valid current claim is
accuracy-preserving efficiency relative to the matched SNN parent, not optical-flow
accuracy SOTA.

## Main or Appendix Table 6: MVSEC public literature context

Published rows use different training datasets and cannot be treated as a strict
ablation against the direct-day2 proposed row.

| Method | Training | OD1 | IF1 | IF2 | IF3 | Macro AEE |
|---|---|---:|---:|---:|---:|---:|
| EV-FlowNet2 | MVSEC | 0.32 | 0.58 | 1.02 | 0.87 | 0.69 |
| OF_EV_SNN | MVSEC | 0.85 | 0.58 | 0.72 | 0.67 | 0.71 |
| Spatiotemporal SNN | MVSEC | 0.45 | 0.76 | 1.13 | 0.95 | 0.82 |
| Spike-FlowNet | MVSEC | 0.49 | 0.84 | 1.28 | 1.11 | 0.93 |
| SDformerFlow-v2 | MDR | 0.61 | 0.54 | 0.81 | 0.69 | 0.66 |
| Proposed C12 | MVSEC day2 internal | 0.8181 | 1.5850 | 2.6212 | 2.0352 | 1.7649 |
| Earlier local TTX ep20 | MDR local reproduction | 0.9779 | 1.1414 | 1.4266 | 1.2617 | 1.2019 |

Recommended placement: appendix or protocol discussion. The direct-day2 row is a
same-parent module ablation, not a competitive reproduction of the parent paper's
MDR protocol.

## Appendix Table A1: Dyadic alpha sensitivity

Frozen C12 ep29 checkpoint; inference-time constant sensitivity only.

| alpha | AEE | AAE-2D | AE-3D | Fl (%) | Spikes (G) |
|---:|---:|---:|---:|---:|---:|
| **1/8** | **1.205595** | **5.379035** | **5.117846** | 5.3761 | **72.0355** |
| 1/4 | 1.209877 | 5.406798 | 5.148612 | 5.4029 | 72.0363 |
| 1/2 | 1.206061 | 5.384223 | 5.118100 | **5.3506** | 72.0420 |

## Appendix Table A2: Final C12 checkpoint selection

Only the preregistered ep30/32/34 checkpoints were evaluated.

| Epoch | AEE | AAE-2D | AE-3D | Fl (%) | Spikes (G) |
|---:|---:|---:|---:|---:|---:|
| 30 | 1.207285 | 5.405067 | 5.139178 | 5.3887 | 72.3407 |
| 32 | 1.217259 | **5.399178** | 5.126907 | 5.6212 | 72.6276 |
| **34** | **1.199514** | 5.400641 | **5.106363** | **5.3138** | 72.8912 |

## Appendix Table A3: Score precision sensitivity

This is an older H67 ep35 algorithm-sensitivity table plus an idealized dual-slot
proxy. QF7 is the historical H67 hardware point; none of these rows inherits the
final C12 ep34 live93 capture identity. The other precision rows are not RTL results.

| Score | AEE | AAE-2D | AE-3D | Fl (%) | Spikes (G) | Pair equal | Ideal dual-slot reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| QF5 | 1.332377 | 5.908379 | 5.665975 | 6.4666 | 82.1065 | 98.4138% | 49.2069% |
| QF6 | 1.331083 | 5.925380 | 5.676300 | 6.4542 | 82.1075 | 92.0596% | 46.0298% |
| **QF7** | **1.327912** | **5.914105** | **5.666098** | **6.3915** | **82.1065** | 97.5198% | 48.7599% |
| QF8 | 1.330811 | 5.926746 | 5.679027 | 6.4539 | 82.1074 | 92.9069% | 46.4535% |

## Appendix Table A4: Motion/no-motion reviewer control

This is a same-parent, seed-matched recipe-level control, not step-paired causal
training. It should not be used to claim a large Motion accuracy gain.

| Route | AEE | AAE-2D | AE-3D | Spikes (G) |
|---|---:|---:|---:|---:|
| H81 no-motion TTX ep29 | 1.330597 | 5.969235 | 5.672632 | **80.9024** |
| H67 Motion-TTX ep35 | **1.329678** | **5.900353** | **5.650878** | 82.1107 |

Motion changes AEE by only -0.069% and increases spikes by 1.49%. Its paper value is
the complete TTX operator/dataflow and hardware evidence, not a large isolated AEE gain.

## Appendix Table A5: Alternative Local5 accuracy line

Local5 is not the final DATE deployment mainline. Keep this table only for an
alternative-topology appendix or reviewer response.

| Route | DSEC AEE | DSEC AE-3D | DSEC Spikes (G) | MVSEC full Macro AEE | MVSEC full Spikes (G) |
|---|---:|---:|---:|---:|---:|
| H67 Motion-TTX | 1.329678 | 5.650878 | **82.1107** | **1.767113** | **140.6647** |
| Local5 rank-1 ep44 | **1.281893** | **5.508685** | 85.2376 | 1.801101 | 141.3613 |

Local5 is the DSEC accuracy upper bound, while the Complete-TTX family generalizes
better to MVSEC. The final deployment identity is C12 ep34; the H67 row in this
historical comparison has its own older checkpoint provenance.

## Hardware-side tables required by the paper

The hardware team owns these values. Do not populate them from spike-energy proxies.

1. Component RTL comparison: Fixed2S versus exact quotient/coalesced datapath;
   columns should include cycles, speedup, slots, normalization transactions,
   gated-K/projection transactions, area, Fmax, dynamic power, and energy/head-row.
2. Full-encoder system table: latency/flow, throughput, energy/flow, SRAM/DRAM traffic,
   area, and end-to-end speedup under matched memory and frequency constraints.
3. Hardware ablation: baseline, generic zero-skip, quotient/coalescing, scheduling,
   and the final combined system, with identical workload and synthesis constraints.
4. Correctness/coverage table: checkpoint SHA, sample/window/head-row coverage,
   real projection weights, backpressure tests, mismatches, RTL/formal evidence level,
   and the explicit boundary between component exact and full-network exact.

## Missing before manuscript freeze

- Official DSEC hidden-test submission for the proposed row.
- Final hardware PPA/SAIF and full-encoder Amdahl/system table from the hardware team.
- Parameter count and deployment memory footprint for C00/C10/C12 under one counting script.
- At least one additional training seed if the paper claims statistical robustness.
- Recheck all public leaderboard values and paper citations on the final submission date.

## Authoritative local receipts

- `neuron_experiments/H9_bipolar_self_attention/results/date_two_contribution_full30_20260826/summary.json`
- `neuron_experiments/H9_bipolar_self_attention/results/dsec_c12_alpha0125_ep29_resume5_20260830/summary.json`
- `neuron_autoresearch/MVSEC_STRICT_SAME_PARENT_ABLATION_20260830.json`
- `neuron_autoresearch/H67_H81_NOMOTION_RESULT_20260812.json`
- `neuron_experiments/H9_bipolar_self_attention/results/h67_ep35_score_precision_qf5_qf8_20260813/summary.json`
- `neuron_autoresearch/DATE_PUBLIC_BENCHMARK_COMPARISON_20260826.md`

## Public primary sources

- DSEC-Flow benchmark and metric definitions:
  `https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/`
- BAT, AAAI 2026: `https://ojs.aaai.org/index.php/AAAI/article/view/38100`
- EDCFlow, CVPR 2025:
  `https://openaccess.thecvf.com/content/CVPR2025/html/Liu_EDCFlow_Exploring_Temporally_Dense_Difference_Maps_for_Event-based_Optical_Flow_CVPR_2025_paper.html`
- SDformerFlow parent paper and MVSEC table: `https://arxiv.org/abs/2409.04082`
