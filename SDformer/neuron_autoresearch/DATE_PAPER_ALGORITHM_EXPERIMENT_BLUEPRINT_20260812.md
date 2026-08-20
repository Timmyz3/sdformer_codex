# DATE Paper Algorithm Experiment Blueprint

Date: 2026-08-12

## 1. MVSEC protocol families in prior optical-flow work

MVSEC does not define one universal train/test protocol. Papers must be grouped by
training source and supervision before comparing absolute AEE.

| family | representative work | training | evaluation | comparison boundary |
|---|---|---|---|---|
| MVSEC in-domain | EV-FlowNet, Spike-FlowNet, STE-FlowNet, SA-FlowNet | usually `outdoor_day2` only | `indoor_flying1/2/3` and `outdoor_day1`; many use only 800 outdoor1 frames | common historical four-sequence protocol |
| correlation/recurrent | E-RAFT and descendants | `outdoor_day2`; E-RAFT uses differentiable warm-start and temporally upsampled approximately 45-Hz GT | primarily `outdoor_day1`, often at 20/45 Hz and dt1/dt4 | not directly equal to original sparse-GT training |
| DSEC pretrain plus MVSEC fine-tune | lightweight iterative methods | DSEC pretrain, then `outdoor_day2` fine-tune | `outdoor_day1` | transfer setting; must be labeled separately |
| cross-domain real data | FireNet, ConvGRU-EV-FlowNet, ET-FlowNet | UZH-FPV or other drone data | four MVSEC sequences | stronger domain-generalization test, different training data |
| rendered-data generalization | ADMFlow/MDR, SDformerFlow | MDR rendered event-flow data | four MVSEC sequences | no MVSEC training; do not mix with day2-trained rows |
| model-based | Secrets of Event-Based Optical Flow | no learning | four MVSEC sequences | no training data; not a neural-network training baseline |

Primary sources:

- STE-FlowNet explicitly trains only on `outdoor_day2`, evaluates the other four
  sequences, and trains dt1/dt4 models separately:
  https://yuzhaofei.github.io/papers/22-AAAI-Spatio-Temporal%20Recurrent%20Networks%20for%20Event-Based%20Optical%20Flow%20Estimation.pdf
- E-RAFT trains on `outdoor_day2` with differentiable warm-start and reports both
  temporally upsampled and original GT rates:
  https://dsec.ifi.uzh.ch/wp-content/uploads/2021/10/eraft_3dv.pdf
- ET-FlowNet intentionally trains on UZH-FPV instead of day2 to test domain
  generalization:
  https://bmvc2022.mpi-inf.mpg.de/0577.pdf
- ADMFlow trains on MDR and evaluates on MVSEC:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Luo_Learning_Optical_Flow_from_Event_Camera_with_Rendered_Dataset_ICCV_2023_paper.pdf
- SDformerFlow uses MDR training for its reported MVSEC SNN rows:
  https://www.iri.upc.edu/files/scidoc/2930-SDformerFlow%3A-Spiking-neural-network-transformer-for-event-based-optical-flow.pdf
- Secrets of Event-Based Optical Flow documents that learning methods often reserve
  day2 for training, but also shows no-training and indoor-to-outdoor alternatives:
  https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780616.pdf

### Current project protocol and its limit

The direct-MVSEC run uses dt1, center/random `256x256`, `outdoor_day2`, a
chronological 2363/263 train/held-out-validation split, and deterministic fixed800
evaluation on all four test sequences. This is a fair internal NB0/H67/Local5
comparison and prevents test-set checkpoint selection.

It is not numerically identical to E-RAFT's temporally upsampled approximately
45-Hz training or Spike-FlowNet's self-supervised use of many grayscale-frame
intervals. Papers quote roughly 26k frames or 28k generated pairs in some variants,
whereas this supervised manifest has 2363 train pairs. Therefore the current direct
MVSEC table is a cross-model generalization table, not a claimed reproduction of
the best published absolute AEE.

### Frozen paper decision: day2-only MVSEC

The DATE paper adopts the most common learning-based MVSEC family as its primary
secondary-dataset protocol: use only `outdoor_day2` for model development and reserve
`outdoor_day1 + indoor_flying1/2/3` for testing. The current chronological 90/10 split
within day2 remains frozen for train/validation checkpoint selection; no test sequence
is used to choose an epoch or tune a threshold. Report this explicitly as
`day2-only (2363 train / 263 held-out validation)`, rather than claiming that all day2
pairs were used for gradient updates.

MDR-to-MVSEC and FPV-to-MVSEC results may appear only in a separately labeled related-work
comparison. They are not the primary project result and must not be pooled with the
day2-trained NB0/H67/Local5 rows. No retraining is triggered by this protocol decision:
the three existing direct-MVSEC runs already share the frozen day2-only manifest.

## 2. Mainline decision: H67 Motion-TTX is frozen

Freeze **H67 Motion-TTX ep35** as the only DATE paper mainline. Keep Local5 alive as
an accuracy/topology extension and appendix candidate, but do not let it change the
main model identity or create a mixed Motion+Local5 network:

- **H67 Motion-TTX:** safer narrow mechanism, current stronger cycle evidence and
  better direct-MVSEC generalization.
- **Local5 TTX:** current best DSEC accuracy and a thicker topology-stationary
  accelerator story.

Current evidence:

| evidence | H67 Motion-TTX | Local5 TTX |
|---|---:|---:|
| DSEC fullres AEE | 1.3297 | **1.3153** |
| DSEC spikes (G) | **82.1107** | 84.4197 |
| MVSEC fixed800 macro AEE | **1.7649** | 1.7984 |
| MVSEC fixed800 spikes (G) | **55.1700** | 55.4902 |
| MVSEC full macro AEE | **1.7671** | 1.8011 |
| MVSEC full spikes (G) | **140.6647** | 141.3613 |
| checkpoint-bound component RTL | PASS, ep35 | PASS, ep29 only |
| current fair cycle evidence | about 1.185x single-window | direct GASR 0.995x; selector 1.022x |
| Local5 SRAM-transaction reduction | N/A | about 80.0% versus direct backend |

The other agent's assessment is directionally correct: Local5 has higher architectural
ceiling because the five-relation topology can be tied to source-major multicast,
relation residency, five-bank accumulation, and exact miss-recompute. However, the
current evidence proves memory-traffic potential, not yet an end-to-end speed/energy
win. Avoid presenting memo, transpose, banking, and replay as separate innovations.
Use one contribution sentence:

> A topology-stationary Local-TTX dataflow converts destination-major five-neighbor
> matching into source-major multicast, retaining exact relation semantics while
> reducing relation/accumulator traffic.

### Local5 extension gate

Local5 may be promoted from appendix/extension to a secondary paper contribution only
after all of the following are available:

1. Local5 ep39/44/49 convergence result and final checkpoint selection.
2. Full-sequence MVSEC comparison. This is complete: Local5 is 1.92% above H67 in
   macro AEE, but it fails the stricter all-four-sequences-better-than-NB0 gate.
3. Same-checkpoint Local5 profile/RTL if the winner is not ep29.
4. Equal-area system model showing at least 10% energy or EDP gain versus H67, with
   throughput regression no worse than 5% and area overhead no more than 15%.
5. A direct-backend, always-Local5, and adaptive Local5 comparison that reports both
   cycles and SRAM/DRAM traffic; an oracle-only result is not sufficient.

Regardless of that extension gate, write H67 as the paper mainline and Local5 as the
accuracy/system-architecture challenger. Do not combine Motion and Local5 in the final
network merely to improve a number.

## 3. Required algorithm tables for the DATE paper

### Table A: Dataset and protocol

| Dataset | Train source/split | Validation | Test population | Input | Resolution/crop | Window | Checkpoint rule | Metrics |
|---|---|---|---|---|---|---|---|---|
| DSEC | official local train split | frozen local validation list | 18-sequence valid825 | 10-bin voxel, T=2 | 480x640, no crop | T2x15x15 | AEE rank-1, convergence audited | AEE, AAE-2D, AE-3D, Fl-all, spikes |
| MVSEC direct | outdoor_day2 2363 pairs | chronological tail 263 pairs | OD1 + IF1/2/3 | dt1 voxel | 256x256 crop | frozen model window | held-out loss rank-1 | event-masked AEE, GT Fl, spikes |

The paper must state that DSEC valid825 is local validation, not official hidden test.

### Table B: DSEC primary algorithm result

Use this exact header:

| Method | Neuron | Attention | Temporal prior | Spatial candidates | AEE down | AAE-2D down | AE-3D down | Fl-all (%) down | Spikes (G) down | Delta AEE | Delta spikes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NB0 | PSN | original | none | original | 1.4454 | 6.5128 | 6.1803 | 7.9323 | 126.1156 | reference | reference |
| H81 TTX no-motion | binary ATLIF | all12 TTX | none | self | 1.3306 | 5.9692 | 5.6726 | 6.4310 | **80.9024** | -7.94% | -35.85% |
| H67 Motion-TTX | binary ATLIF | all12 TTX | Motion-XOR | self | 1.3297 | 5.9004 | 5.6509 | 6.4279 | 82.1107 | -8.00% | -34.89% |
| Local5 TTX | binary ATLIF | all12 Local-TTX | none | self+4 axial | **1.2819** | 5.8498 | **5.5087** | **6.0210** | 85.2376 | -11.31% | -32.41% |

Bold only the best measured value. Do not bold pending or proxy values.

### Table C: Equal-budget convergence

| Method | Budget30 AEE | Budget35 AEE | Budget40 AEE | Budget50 AEE | Best checkpoint | Last-minus-best (%) | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| NB0 | 1.4454 | 1.4584 | 1.4549 | N/A | ep29 | +0.66 | plateau/overfit |
| H81 | **1.3306** | 1.3475 | 1.3438 | N/A | ep29 | +0.99 | passed optimum |
| H67 | 1.3387 | **1.3297** | 1.3434 | N/A | ep35 | +1.03 | passed optimum |
| Local5 | 1.3286 | 1.3355 | 1.3153 | 1.2982 | **ep44 / 1.2819** | +1.27 | passed optimum |

This table prevents the reviewer from attributing the gain to unequal training budgets.

### Table D: MVSEC direct fixed800 generalization

Use per-sequence AEE/GT-Fl plus macro and cost columns:

| Method | Init | Ckpt | OD1 AEE/Fl | IF1 AEE/Fl | IF2 AEE/Fl | IF3 AEE/Fl | Macro AEE down | Pixel-weighted AEE down | Macro Fl (%) down | Spikes (G) down | Energy proxy (uJ) down |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NB0 | scratch | ep11 | 0.8379/3.735 | 1.5977/11.699 | 2.7469/35.151 | 2.1102/22.712 | 1.8231 | 2.4429 | 18.324 | 97.3392 | 82900.09 |
| H67 | same NB0 ep11 | ep12 | **0.8181/3.710** | **1.5850/11.465** | **2.6212/32.673** | **2.0352/20.404** | **1.7649** | **2.3287** | **17.063** | **55.1700** | **47654.90** |
| Local5 | same NB0 ep11 | ep12 | 0.8380/3.940 | 1.6259/12.074 | 2.6646/33.649 | 2.0650/21.203 | 1.7984 | 2.3694 | 17.716 | 55.4902 | 47888.89 |

Add a footnote: all three rows use the same manifest, crop, fixed800 indices, metric,
and seed; candidates use the same NB0 initialization. Energy is a software proxy and
must not be called measured chip energy.

### Table D2: MVSEC direct full-sequence confirmation

| Method | Macro AEE down | Pixel-weighted AEE down | Macro Fl (%) down | Spikes (G) down | Energy proxy (uJ) down | Versus NB0 macro AEE | Versus NB0 spikes |
|---|---:|---:|---:|---:|---:|---:|---:|
| NB0 | 1.8273 | 2.3435 | 18.354 | 251.4680 | 214151.98 | reference | reference |
| H67 | **1.7671** | **2.2300** | **17.128** | **140.6647** | **121555.15** | -3.29% | -44.06% |
| Local5 | 1.8011 | 2.2696 | 17.792 | 141.3613 | 122047.50 | -1.43% | -43.79% |

The fail-closed audit is PASS. H67 improves AEE on all four sequences; Local5 improves
aggregate AEE and cost but does not improve every sequence, so only H67 qualifies under
the pre-registered MVSEC algorithm gate.

### Table E: Mechanism ablation

| ID | Binary ATLIF | all12 unified | Motion-XOR | Local5 topology | AEE | AE-3D | Fl | Spikes | Purpose |
|---|---|---|---|---|---:|---:|---:|---:|---|
| A0 NB0 | no | no | no | no | 1.4454 | 6.1803 | 7.9323 | 126.12 | baseline |
| A1 H81 | yes | yes | no | no | 1.3306 | 5.6726 | 6.4310 | 80.90 | unified TTX effect |
| A2 H67 | yes | yes | yes | no | 1.3297 | 5.6509 | 6.4279 | 82.11 | temporal prior effect |
| A3 Local5 | yes | yes | no | yes | 1.2819 | 5.5087 | 6.0210 | 85.24 | spatial topology effect |

This is the minimum causal ablation. Do not add partial-stage or mixed-attention rows.

### Table F: Seed robustness

| Method | Seeds | AEE mean +/- std | AE-3D mean +/- std | Fl mean +/- std | Spikes mean +/- std | Runs meeting 5%/20% gate |
|---|---:|---:|---:|---:|---:|---:|
| NB0 | 0 only | 1.4454 | 6.1803 | 7.9323 | 126.1156 | seed0 yes |
| H67 | 0 only | 1.3297 | 5.6509 | 6.4279 | 82.1107 | seed0 yes |
| Local5 | 0 only | 1.3153 | 5.5379 | 6.3815 | 84.4197 | seed0 yes |

Seed1/2 configs are registered but not launched. The DATE claim is explicitly limited
to seed0. Do not manufacture a standard deviation from checkpoints of one seed.
See `DSEC_SEED12_REGISTRY_20260813.json`.

### Table G: Event-density/workload stratification

| Dataset | Method | Density quartile | Frames | AEE | Fl | Spikes/frame | Active relations | Memo hit rate | Cycles/frame |
|---|---|---|---:|---:|---:|---:|---|---|---|
| DSEC valid825 | NB0 | Q1 voxel-L1 <= 556401.6 | 207 | 1.3741 | 6.2213 | 1.423e8 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | H81 | Q1 voxel-L1 <= 556401.6 | 207 | 1.2623 | 4.9927 | 9.163e7 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | H67 | Q1 voxel-L1 <= 556401.6 | 207 | 1.2886 | 5.0558 | 9.306e7 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | Local5 | Q1 voxel-L1 <= 556401.6 | 207 | **1.1757** | **4.5174** | 9.649e7 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | NB0 | Q2 voxel-L1 <= 719957.9 | 206 | 1.3612 | 6.5766 | 1.507e8 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | H81 | Q2 voxel-L1 <= 719957.9 | 206 | 1.2622 | 5.5715 | 9.684e7 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | H67 | Q2 voxel-L1 <= 719957.9 | 206 | 1.2475 | 5.4153 | 9.830e7 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | Local5 | Q2 voxel-L1 <= 719957.9 | 206 | **1.2386** | **5.3216** | 1.021e8 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | NB0 | Q3 voxel-L1 <= 891402.7 | 206 | 1.3799 | 7.1168 | 1.548e8 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | H81 | Q3 voxel-L1 <= 891402.7 | 206 | 1.2839 | 6.0175 | 9.944e7 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | H67 | Q3 voxel-L1 <= 891402.7 | 206 | 1.2723 | 5.8833 | 1.009e8 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | Local5 | Q3 voxel-L1 <= 891402.7 | 206 | **1.2432** | **5.6194** | 1.048e8 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | NB0 | Q4 voxel-L1 > 891402.7 | 206 | 1.6664 | 11.8226 | 1.637e8 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | H81 | Q4 voxel-L1 > 891402.7 | 206 | 1.5144 | 9.1492 | 1.044e8 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | H67 | Q4 voxel-L1 > 891402.7 | 206 | 1.5106 | 9.3638 | 1.059e8 | hardware read-only | hardware read-only | hardware read-only |
| DSEC valid825 | Local5 | Q4 voxel-L1 > 891402.7 | 206 | **1.4705** | **8.6331** | 1.099e8 | hardware read-only | hardware read-only | hardware read-only |

Cuts stay frozen in `DSEC_VALID825_DENSITY_POPULATION_20260813.json`.
Attached AEE/Fl/spikes are in `DSEC_DENSITY_QUARTILE_TABLE_G_20260817.json`.
Re-eval AEE matches rank-1 profiles to ~1e-9. Hardware columns stay read-only.
Local5 wins every quartile; the extra gain vs H81 is largest on Q1.
H67 is slightly worse than H81 on Q1 and slightly better on Q2–Q4.

### Table H: Algorithm-to-hardware complexity bridge

| Method | Candidates/token | Score/gate bits | Params | XNOR/popcount | Projection ops | Activation bytes | Weight bytes | SRAM transactions | Cycles | Area | Energy | EDP |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Use measured or post-synthesis values where available. Mark analytical, RTL-cycle,
post-synthesis, and proxy values explicitly. Never mix them without a provenance column.

## 4. Required figures and appendix evidence

1. DSEC AEE versus spikes Pareto plot: NB0, H81, H67, Local5.
2. Accuracy versus epoch/budget plot for 30/35/40 and Local5 50.
3. MVSEC four-sequence bar plot, grouped by method, with a separate spikes axis/plot.
4. Qualitative DSEC/MVSEC flow and endpoint-error maps: low density, high density,
   motion boundary, and large-flow cases selected by frozen rules.
5. Local5 topology/dataflow diagram with one terminology set only.
6. Appendix loading table: ATLIF count, Shiftmax count, overlay keys,
   missing/unexpected keys, checkpoint SHA, config SHA, and RTL claim scope.

## 5. Minimum remaining experiment list

1. H81 no-motion DSEC fullres40 is done. Rank-1 is ep29: AEE 1.3306, AE-3D 5.6726,
   Fl 6.4310, spikes 80.9024G. Versus H67 ep35 the AEE gap is only -0.069%.
   Treat Motion-XOR as a small temporal add-on, not the main DSEC gain.
2. H67 QF5-QF8 is done (algorithm sensitivity only; QF5/6/8 are not RTL). Versus QF7:
   QF5 AEE +0.0045, QF6 +0.0032, QF8 +0.0029. Spikes stay ~82.11G.
3. Local5 40-to-50 is done. Rank-1 is ep44: AEE 1.2819, AE-3D 5.5087, Fl 6.0210,
   spikes 85.2376G. ep49 is worse. Hardware remains bound to ep29.
4. Final audit PASS: H67 stays the frozen mainline; Local5 is reported as an
   extension. Local5 rank1 and RTL checkpoint are not the same (ep44 vs ep29).
5. Seed1/2 configs are registered and not launched. The paper claim is limited to seed0.
8. MVSEC completion finished 2026-08-16. H81 same-protocol full AEE 1.7926 fails the
   all-four-sequences gate on indoor_flying1. Local5 DSEC-ep44 day2 FT full AEE 1.6686
   beats NB0 on all four sequences, but it is a transfer protocol and must stay a
   separate row. Scratch Local5 remains the official same-protocol Local5 line.
6. Density-quartile Table G is attached (2026-08-17). Four rank-1 re-evals match
   the existing valid825 AEE to ~1e-9. Hardware columns stay read-only.
7. If a later paper promotes Local5 and uses ep39/44/49, regenerate same-checkpoint
   hardware evidence in the hardware workstream; it cannot inherit ep29 provenance.

## 6. Grok takeover 2026-08-13

Codex session `019ec76b-ea14-7862-be41-45ea956713db` lost quota. Grok continues the
same GPU queue and does not start new training:

H81 train -> H81 valid825 ep29/34/39 -> H67 QF5-QF8 -> Local5 40-50 -> final H67 audit.

Algorithm waiters no longer write hardware docs. Hardware evidence remains read-only.

## 7. Integrity notes after source-file audit 2026-08-13

Machine receipt: `DATE_ALGORITHM_INTEGRITY_AUDIT_20260813.json`.

- Table B/C numbers match the plus10 `spike_profile.json` files. Identity SHAs still bind.
- Write H81 as a recipe-level no-motion control, not a step-paired Motion ablation.
  H67 `ep35` includes a plus10 continuation that H81 does not pair.
- Write DSEC as local valid825. Do not compare NB0 AE-3D 6.18 to hidden-test 4.871.
- Write MVSEC as separate day2-trained models. Local5 fails the all-sequence gate on
  `indoor_flying1`.
- Paper claim stays seed0. The August 5 closure Local5 row (ep29 / 1.3286) is stale
  relative to Table B (ep39 / 1.3153).
