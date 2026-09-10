# 01 — Kill list (do not rebuild as novelty)

## Dead as contribution bullets

| Object | Owner / why dead |
|---|---|
| C1 exact-subset product capture / ProSparsity forest | Prosperity, HPCA 2025, arXiv:2503.03379. Math is `P_r = P_p + sum_{s in M_r \ M_p} p_s`. Our remainder is 1RW tax + bottleneck-Conv hang. 1.6945× includes inherited product sparsity. |
| C2 TSBG token-major vs group-major | Ordinary weight broadcast / Gustavson. Priors: ELSA ISCA 2026 mini-batch Gustavson + bundled AER; Eyeriss row-stationary; SpikeX multi-bit weight sharing; FireFly-T weight dispatch. Equal-bandwidth K8 vs K1×8 cycles ≈1.017×; throughput/logic-area 4.54× is PE merge, not a new skip. |
| Empty-tile / inactive-TTB / zero-source skip ASIC | Event camera already omits empty events. FireFly-T / Bishop inactive bundle / ordinary `row_live` masks. Fudan ISSCC butterfly skipper is prior. |
| Bit-skip vs strongest-zero as a win | Already measured: bit skip ≈1.003× vs strongest-zero. |
| Shiftmax integer-power gating | Deploy score is Q7; exp LUT on fractional exponents; Q1.7 gate codes ≈28 values, not integer 2^k. Low-exp zeroing is **lossy** and needs AEE Pareto. |
| ASNA-Flow “optical-flow spatial locality” as our primitive | TVLSI 2025, TSMC 28 nm, 104 FPS, 7.9 mW, 0.3 pJ/SOP already claims it. |
| FireFly-T AND-PopCount as CMOS novelty | FPGA LUT6 6:2/6:3 overlay, Zynq UltraScale+, systolic binary QK^T. Use as **baseline** for MX3P, not as our engine. |
| SDSA / Spike-driven Transformer mask+add as first HW | FPGA mappings exist; Chen 28 nm SRAM Spikformer with T=4/2/1 already maps addition-only attention-class work. |
| Analog CIM / ASTER PIM / Xpikeformer AIMC | Leaves digital 28 nm 1RW island path. |
| SpiDR unstructured zero-skip | 65 nm digital CIM, dual-port IFspad, not 1RW foundry SRAM. |
| Comperity DSE as Motion-XOR | TACO: AND trees extract shared base, XOR encodes GeMM residual. Different object than a score term. |
| FireFly-S dual-side sparsity | Requires pruning (~85% W) + 4-bit quant; unfreezes ep34; lossy AEE Pareto. |
| LoAS dual-sparse weights | LTH ~97–98% W sparsity; not frozen identity. Inner-join may be reused only as a **fiber primitive**, not a dual-sparse GEMM island. |
| ELSA elastic first-response pipeline | Dense OF has no first-correct classification token; 24 dynamic-BN barriers; documented NO-GO. Membrane-stationary Gustavson compared at 1.0× cycles. |
| Bishop BSA/ECP training | Changes the frozen model. Inactive-TTB skip = empty-tile. Keep only “bound without computing S” as an optional gate. |
| CICC 2026 MaxPool/ReLU redundancy speculation | Invalid on Shiftmax: silent tokens still affect the row denominator. |
| CICC 2026 BWAC | On Motion INT8 projection weights, bitmap/INT8 = 105.858% at group 16; weight zero-rate 1.514%; expands vs adaptive 94.04% encoding. |
| CICC 2026 DLSS as lossless | Approximate deep U-Net skip, AEE 0.96→0.99. No AEE budget if applied lossily to our decoder. Retarget only as **temporal-peer similarity probe** in front of Motion-XOR (see Rank 4). |
| Correlation volume / RAFT pyramid / residual Δf island | Unfreezes C12. 49-offset / 9×9 cost-volume already rejected. Decoder is U-Net skip-concat, not E-RAFT. |
| GANAX/Chang ConvTranspose skip-inserted-zeros as new decoder island | All four ConvTranspose C_out in {384,192,96,96}; 96-lane slice already fills 96/96; EPD/A1 ≤1.0×. |
| Grok Bot HBG-RP int8 ATLIF payload | Not the frozen capture. 93/93 binary. |
| Component-ratio product as system speedup | Reviewer-illegal. |
| C3 as speedup | Coverage of Fixed-T10, 17 cycles/tile, 63,756 µm². |

## Still legal as **execution context** (not headline)

- C1 1RW forest as “how bottleneck conv is served”.
- C2 K8 typed-signed Acc24 + TSBG as “how FC weight rows are delivered to dirty contexts”.
- Ordinary live-mask skip of inactive sources.
- Foundry 1RW TS1N28 density story (port constraint, not a new algorithm).
