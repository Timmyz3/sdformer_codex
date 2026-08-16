# CICC 2026 Optical-Flow Accelerator: MVSEC Protocol Audit

Date: 2026-08-01

## Source

Tao Zhang, Yi Zhong, Youming Yang, Jinhao Ruan, Yipeng Gao, Zhaoxiang Zhang,
and Yuan Wang, "A 28-nm Optical Flow Estimation Accelerator with Redundancy
Speculation, Bit-Width-Aware Compression and Similarity Detection," CICC 2026,
pp. 1-4, DOI `10.1109/CICC65509.2026.11509564`.

- IEEE record: https://ieeexplore.ieee.org/document/11509564
- DBLP record: https://dblp.org/rec/conf/cicc/ZhangZYRGZW26
- Local full paper:
  `hw_autoresearch_nts07/docs/Zhang 等 - 2026 - A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression.pdf`

## Verified Claims

The accessible abstract states that the workload is event-camera optical flow and
that redundant operations and external-memory accesses are the target bottlenecks.
It reports:

| quantity | reported normalized/result value |
|---|---:|
| overall operations | 0.20x |
| external-memory accesses | 0.08x |
| corresponding energy | 0.12x |
| corresponding latency | 0.19x |
| EMA-included efficiency | 14.07 TOPS/W |

The title and abstract also verify three named mechanisms: redundancy speculation,
bit-width-aware compression, and similarity detection.

## Full-Text Findings

The deployed model is an INT8 SNN-ANN Hybrid U-Net referring to Spike-FlowNet.
The CICC evaluation uses `800` inputs from each of `indoor1/2/3` and `outdoor1`.
Its cited Spike-FlowNet protocol trains on `outdoor_day2` only and evaluates
event-masked AEE on center-cropped `256x256` dt1 inputs. CICC does not disclose
numerical values for `theta_ch` or `theta_s`.

| metric | indoor1 | indoor2 | indoor3 | outdoor1 | mean |
|---|---:|---:|---:|---:|---:|
| INT8 baseline AEE | 0.84 | 1.32 | 1.14 | 0.52 | 0.96 |
| all-feature AEE | 0.87 | 1.35 | 1.17 | 0.56 | 0.99 |
| operations | 0.18x | 0.20x | 0.25x | 0.17x | 0.20x |
| EMA | 0.08x | 0.09x | 0.10x | 0.08x | 0.08x |
| energy | 0.10x | 0.12x | 0.14x | 0.11x | 0.12x |
| latency | 0.17x | 0.21x | 0.24x | 0.19x | 0.19x |

The mechanisms are group-16 lossless BWAC, density-ordered early MaxPool/ReLU
speculation, and feature-similarity-controlled deep-level skipping. TTB is not
part of this paper.

Hardware details relevant to comparison are a 28-nm CMOS chip, `0.80 mm2` core
(`1.45 mm2` die), INT8 weights, 16 PE lines with four PEs per line, and eight
single-bit-input by INT8-weight operations per PE. The on-chip buffers total
`162 KB`: 72-KB compressed weights, 21-KB compression metadata, 50-KB feature
maps, 18-KB partial sums, and 1-KB speculative addresses. The paper estimates
LPDDR3 EMA at `3.7 pJ/bit` and `6.4 GB/s`; four-sequence inference costs are
`2.07-2.88 mJ` and `58.53-79.95 ms`.

Figure 9 reports cumulative reductions. BWAC reduces EMA to `0.35x`, and DLSS
then reduces the remaining EMA to `0.24x`, consistent with approximately `0.08x`
overall. The energy path is shown cumulatively as `0.54x` after BWAC, `0.88x`
after speculation, and `0.23x` after DLSS. These intermediate ratios should not
be mistaken for independent ablations from the dense baseline.

## Adopted Experiment Pattern

For a frozen MVSEC checkpoint and identical traces, use the paper's cumulative order:

| ID | policy | exactness contract |
|---|---|---|
| C0 | INT8 dense/no BWAC/speculation/DLSS | numerical reference |
| C1 | C0 + group-16 BWAC | lossless and exact versus C0 |
| C2 | C1 + dense-channel-first speculation | approximate, report AEE delta |
| C3 | C2 + feature-similarity DLSS | approximate, report AEE delta and skip interval |

The table must include per-sequence event-masked AEE, operation counts, SRAM/DRAM traffic,
cycles, control overhead, and total energy. Ratios are normalized to C0 for the same
checkpoint and trace. This mirrors the paper's emphasis on operations, EMA, energy,
and latency.

## Project Decision

The paper-facing direct-MVSEC route uses `outdoor_day2` for training and the other four
sequences for testing. The old `outdoor_day1 -> indoor1/2/3` split remains only an
internal fallback. The C0-C3 matrix runs after MVSEC-NB0 and one final candidate are
trained and frozen. It supplements, rather than replaces, the standard NB0-versus-
candidate accuracy and spike/cost comparison.

## What Is Actually Copied from the CICC Experiment

The intended reference is its experiment structure:

1. Freeze an equal-size workload: `800` inputs from each of four MVSEC sequences.
2. Keep the INT8 model and input traces fixed while enabling features cumulatively.
3. Put accuracy degradation and hardware benefits in the same per-sequence table.
4. Report application-level operations, EMA, energy, and latency normalized to a
   feature-disabled baseline.
5. Break cumulative contributions into EMA/energy/latency waterfall charts.
6. Characterize implementation overhead and compare against prior accelerators.

For this project, use a two-table design. The model table compares NB0 float,
NB0 hardware-order, TTX float, and TTX hardware-order. The deployment table then
freezes the TTX hardware-order checkpoint and cumulatively enables lossless
compression, TTB skip/dispatch, and similarity-controlled deep-level skip.

The deployment experiment can run immediately on existing MDR-trained frozen
checkpoints because it is a same-checkpoint trace experiment. Training on
`outdoor_day2` is a separate experiment needed only for the published direct-MVSEC
split; it is not a prerequisite for validating the CICC-style hardware ablation.
