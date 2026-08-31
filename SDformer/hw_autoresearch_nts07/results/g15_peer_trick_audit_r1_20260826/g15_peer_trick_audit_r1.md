# G15 peer-trick and novelty audit (research-only)

Date: 2026-08-26

Scope: primary papers and official repositories only. This is a design-screening artifact, not a paper claim, cycle result, or admission receipt. `docs/359_DATE终局冻结_20260813.md` was not modified.

## Bottom line

The generic ideas "bitmap first", "zero / very-sparse / dense routing", "multi-lane sparse decode", and "selectively fetch only used precomputed products" are already occupied. A G15 contribution cannot be framed as any of those in isolation.

The defensible candidate is narrower: for exact, analog-valued H67 Conv3x3 rows, make a cost-aware choice among (1) empty skip, (2) direct signed-weight issue for cheap rows, and (3) q32 parent-PWP plus signed correction only when its measured benefit exceeds matcher and PWP-miss cost; optionally replace expanded-PWP DRAM traffic with an explicitly charged, bounded on-demand exact PWP generator/cache that stores INT8 weights rather than all derived PWPs. This combination must remain a candidate until the generator ports/cycles/area and cache behavior are modeled and synthesized.

## Primary-source findings

| Work | Actual trick | Performance-claim scope and important caveat | Conflict/reuse boundary for G15 |
|---|---|---|---|
| Prosperity, HPCA'25 | One-prefix exact product reuse; TCAM subset detection; popcount ordering; preprocessing of tile *n* overlapped with compute of tile *n-1*; 256x16 spike tile, 128 PEs; 8KB spike + 32KB weight + 96KB output; 64GB/s DDR. | Paper reports 7.4x over PTB and 1.8x over A100. Official simulator charges compute as `max(compute, preprocess)`, memory stalls from traffic / 1024-bit-per-cycle interface, and overlaps LIF with preceding Conv/FC under stated assumptions. Transformer accelerator baselines cover linear layers rather than unsupported attention/LN. | One-prefix and hidden preprocessing are prior art. Reusable: matcher should be conditional and pipelined. G15 must charge first-tile preprocessing and cannot call an overlap "free" without proving producer/consumer timing. |
| Phi, ISCA'25 | Offline k=16, q=128 patterns; precompute `PWP = pattern x weight`; systolic matcher; packed signed residual; 16-bank PWP buffer; prefetch only PWPs referenced by the next pattern-index tile. | 3.45x over Stellar across SNN models in a component simulator. 240KB SRAM, 64GB/s. Even after selective prefetch, PWP/weight traffic is about 3x regular weight traffic; buffer is 0.452/0.662 mm2 and 220.8/346.6mW. PAFT is a separate lossy +1.26x. | Selective PWP prefetch is prior art. Phi explicitly filters zero and one-hot vectors from pattern calibration and notes one-hot PWP equals a weight row. Potential difference: exact on-demand synthesis from resident INT8 weights instead of fetching expanded PWPs, but only if generator/cache cost is paid. |
| Bishop, 2025 | Count active token-time bundles per feature, route dense and sparse features (and coordinated weights) to equal-area heterogeneous cores; tune per-layer threshold to balance both cores. | 5.91x average end-to-end over previous accelerators is analytic/cycle modeling plus BSA/ECP. Hardware-only dense/sparse heterogeneity is 1.39x on ImageNet-100. Attention core and ECP local gains are much larger and must not be mixed with end-to-end. 144KB weight + two 12KB TTB GLBs, 512-bit ports, 76.8GB/s. | Density stratification is prior art. Reusable: threshold is a load-balancing/resource DSE axis, not a sparsity-rate shortcut. G15 needs per-path occupancy, tail, and imbalance statistics. |
| FireFly-T, 2025 | Grouped carry-style bitmap decoder emits M nonzero indices/cycle; popcount tracker controls bitmap replacement; wide broadcast weight memory plus out-of-order workers avoids multi-bank crossbar conflicts. | FPGA deployment. Headline comparisons are mainly energy efficiency and DSP efficiency, not an ASIC end-to-end speedup. Fixed-throughput decoder DSE shows lane/worker tradeoff; wide-bank bandwidth is explicit. | Multi-lane decoding is prior art and can be cited/reused. Must charge decoder logic depth, popcount/tracker, wide weight ports, workers, extraction network, and bank conflicts. Analog H67 also needs signed payload fetch after bitmap decode. |
| DeltaCNN, CVPR'22 | Hybrid convolution has exactly three modes: empty tile skip; 1--4 active pixels use a special direct sparse kernel; >=5 active pixels use dense code. | GPU execution; up to 7x vs cuDNN with thresholded delta updates. Supplement reports very-sparse mode for about 20% of nonempty tiles and up to 2x local benefit. It stresses that one active pixel still causes most memory transfers. | This is the closest novelty collision: EMPTY/DIRECT/DENSE and low-popcount direct are not new. G15 must center the third path on exact parent-result reuse and cost-aware PWP generation, not claim generic three-way routing. |
| ESDA, FPGA'24 | Bitmap creates ordered coordinate tokens before feature payload; sparse line buffer uses bitmap plus kernel-offset stream; submanifold convolution prevents dilation; weights are all-on-chip and layers are spatially pipelined. | At 10% nonzero, individual blocks show 4.5--11x over an equal-PF dense block; up to 54.8x vs embedded GPU uses customized model/all-on-chip FPGA. Submanifold convolution changes the operator/model and needs training/accuracy comparison. | Bitmap/coordinate-first and kernel-offset streams are prior art. Frozen standard H67 Conv cannot inherit submanifold output suppression without changing the model. Ordered bitmap generation can be reused exactly. |
| EvConv, 2023 | Cross-inference sparse increments, masks, explicit sparsification layers, delayed U-Net, periodic refresh. | Up to 98% FLOP reduction but only up to 1.6x latency; optical-flow RecEVFlowNet is about 1.05x slower despite >90% FLOP reduction. Thresholding is lossy and stateful. | Strong warning: irregular memory/control must be charged. This is not an exact frozen-checkpoint shortcut and should remain separate from G15. |
| DeltaCNN official repo | Full-network delta tensors plus update masks; nonlinear layers cache feature maps; BN can be folded into convolution. | Threshold zero is supported, but useful published speedups generally use nonzero tuned thresholds and cached state. | Cross-frame state cache and refresh belong to a separate lossy/stateful axis, not the exact G15 table. |

## Local frozen evidence relevant to G15

Sources:

- `results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/m430b_h67_dualaware_q32_heldout_r1.json`
- `results/m430a_trainonly_dualaware_q32_catalog_r1_20260826/m430_trainonly_dualaware_q32_catalog_r1.json`
- `results/m468r2_h67_peer_budget_rowtile_r1_20260826/m468_h67_peer_budget_rowtile_result_r1.json`

For the four frozen H67 bottleneck Conv3x3 traces:

- 51,840,000 source rows: 24,534,432 zero (47.33%), 7,516,420 popcount-one (14.50%), and 19,789,148 popcount>=2 (38.17%). Therefore zero+one-hot is a guaranteed exact pre-matcher bypass for 61.83% of rows; it is not a new idea, but it is a valid implementation optimization.
- q32 parent-PWP is selected for 15,909,646 rows. Among popcount>=2 rows this is about 80.40%, so indiscriminately routing every low-density nonzero row direct can destroy useful parent reuse; the threshold must be trace-DSE'd.
- M430 matcher cost is 67,912,100 cycles, 13.13% of the 517,041,352-cycle four-Conv candidate. Removing the matcher entirely has only a 1.151x ideal upper bound; zero+one-hot gating removes fewer cycles and cannot by itself provide a 2--3x claim.
- The q32 catalog contains 55,296 patterns with mean popcount 4.027; most are popcount 2 or 3. This makes exact on-demand generation plausible only with a sufficiently wide source-fold engine; it does not make generation free.
- At M468 row-tile 96, 128 B/cycle, the current model loads 6,148,261 used pattern records, 7,869,774,080 PWP bytes and 6,311,645,184 weight bytes. PWP traffic is therefore a real target. But the current 1.0424x point is not an admitted G15 result, and an equal empty-gate baseline plus physical SRAM overhead still need correction.

## Tricks that can be legally reused

1. **Metadata-before-payload gate.** Read a compact exact bitmap/popcount first. Empty rows cause no payload or matcher request. Cite ESDA/FireFly-T/DeltaCNN; do not claim this alone as novel.
2. **Grouped multi-lane direct decoder.** Use a FireFly-T-style grouped carry decoder for the DIRECT path and reuse decoded indices across all output lanes/blocks. Charge tracker, compaction, signed-value fetch, weight-port width and arbitration.
3. **Cost-aware heterogeneous routing.** Borrow Bishop's threshold/load-balance discipline and DeltaCNN's hybrid-path lesson, but route H67 rows to EMPTY, DIRECT, or exact PARENT-REUSE. The selection objective must be measured cycles/energy, not only popcount.
4. **Single-parent bounded search.** Keep Prosperity's one-prefix cost discipline: matcher runs only where the best possible saved vector issues can cover matching plus PWP miss/generation. Pipeline it with previous work, but expose first/final tile overhead.
5. **Selective product materialization.** Phi proves expanded PWP traffic can dominate and that only used PWPs should move. G15 may instead generate only referenced exact PWPs from resident INT8 weights and retain them in a bounded cache. This is a hypothesis until miss rate, generation service cycles, cache evictions, ports and synthesis area are charged.

## Mandatory costs for an exact lazy-PWP DSE

- Bitmap storage/read/transport and popcount/threshold latency, including bank/macro rounding.
- DIRECT decoder lane count, residual iterations, signed payload reads, weight-bank reads, conflicts, crossbar/extractor, and partial-sum RMW.
- Matcher cycles only for eligible rows, pattern SRAM reads, descriptor writes, and first/final pipeline bubbles.
- Per PWP miss: pattern popcount, number of 96-lane output blocks, available source-fold width, weight read ports, accumulation width, cache write, and any stall against correction service.
- PWP cache: exact record width, tag/valid/LRU or deterministic replacement, banks/ports, capacity, cold misses, evictions, and partition/row-tile schedule. No unbounded or prewarmed cache.
- All traffic: weights, generated/fetched PWP, bitmap, source payload, descriptors, psum reads/writes/spills, configuration and commit. DMA command/setup and fill/drain must be explicit.
- Same-resource baselines must receive the same exact empty gate, bitmap front end, SRAM budget, bandwidth, ports and schedule. Compare against both strong-zero and M430, never against dense16 alone.
- Report mean, per-sample, p95 and worst-case cycles/path occupancy. Averages can hide dense-motion windows where the sparse path loses.

## Minimum novelty wording allowed by this audit

Safe candidate description (not yet an admitted claim):

> An exact, cost-aware Conv3x3 execution policy for analog event-flow activations that chooses direct signed-weight issue or parent-result reuse per source row, and trades expanded PWP traffic for bounded on-demand PWP materialization from resident INT8 weights.

Do not claim: first bitmap-first accelerator; first density-stratified sparse engine; first three-mode sparse convolution; first multi-lane decoder; first selective PWP transfer; or first product-reuse accelerator.

## Comparison-table fields required for review

1. Work/venue/year and evaluation platform (analytic simulator, cycle simulator, RTL+DC, FPGA measurement).
2. Claim scope (local operator, supported network subset, full model/end-to-end) and exact model/dataset/checkpoint.
3. Exact vs lossy, accuracy/AEE/EPE delta, threshold/training/state-refresh policy.
4. Baseline name and equality axes: technology, frequency, PE/lane count, area, precision, SRAM, ports, bandwidth, supported operators.
5. Sparse granularity and modes; density/path histogram per layer/sample, p95 and worst case.
6. Decoder/matcher/generator throughput and stalls; first/final tile overhead; load imbalance.
7. SRAM bytes by payload, banks, ports, double buffering and physical/macro status.
8. DRAM bandwidth and bytes by weight/PWP/activation/psum/metadata plus command/setup model.
9. Raw cycles/latency/FPS and speedup with numerator/denominator named; local gains kept separate from end-to-end.
10. Area, power and energy source (DC/PTPX/CACTI/DRAMsim3/Vivado/board measurement) and exclusions.

## Primary URLs

- Prosperity paper: https://arxiv.org/abs/2503.03379
- Prosperity official simulator: https://github.com/dubcyfor3/Prosperity
- Phi: https://arxiv.org/abs/2505.10909
- Bishop: https://arxiv.org/abs/2505.12281
- FireFly-T: https://arxiv.org/abs/2505.12771
- ESDA: https://arxiv.org/abs/2401.05626
- EvConv: https://arxiv.org/abs/2303.04670
- DeltaCNN paper: https://openaccess.thecvf.com/content/CVPR2022/html/Parger_DeltaCNN_End-to-End_CNN_Inference_of_Sparse_Frame_Differences_in_Videos_CVPR_2022_paper.html
- DeltaCNN supplement: https://openaccess.thecvf.com/content/CVPR2022/supplemental/Parger_DeltaCNN_End-to-End_CNN_CVPR_2022_supplemental.pdf
- DeltaCNN official repo: https://github.com/facebookresearch/DeltaCNN
