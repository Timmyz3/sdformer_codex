# 04 — Paper survey (steal / modify / do not copy)

Path (2): published mechanisms, slightly modified. Citations are for Codex to open; numbers below are from abstracts/HTML/local PDFs, not a new PPA table.

## SNN accelerators

| Paper | Venue | Primitive | Steal? | How to modify | Kill if copied as-is |
|---|---|---|---|---|---|
| Prosperity | HPCA 2025 | Product sparsity forest, TCAM prefix | No as novelty | — | C1 |
| LoAS | MICRO 2024 | FTP, CSF fibers, fast/laggy inner-join | Yes, fiber/join | Mixed T=2 vs T=10; join XOR not GEMM | Dual-sparse W (not frozen) |
| APEX | arXiv 2608.19046 | PASC-IF on LoAS | Organization only | Do not replace ATLIF with PASC-IF | Preprint; 40 nm |
| FireFly-T | TC 2026 | Dual engine; LUT6 AND-PopCount; byte-write SRAM permute | AND-PopCount as **baseline** | Extend to AND+XOR+co-silence on 1RW CMOS | FPGA overlay as our ASIC |
| FireFly-S | TVLSI/arXiv | Dual-side sparsity, 4-bit, 85%+ W prune | No under freeze | — | Unfreeze |
| Bishop | ISCA 2025 | TTB, dense/sparse stratifier, ECP, AAC | ECP bound-without-S | Motion-XOR bound; mixed T; no BSA training | Inactive-TTB; 2.96 mm² own ViTs |
| SpikeX | TCAD 2025 | Spatiotemporal unstructured sparsity, weight reuse, HAS | Tags only | Gate Motion-XOR tiles | Bundle reuse = C2 family |
| ELSA | ISCA 2026 | Elastic token pipeline, BAER, mini-batch Gustavson | BAER packing (weak) | Binary ATLIF packing | Gustavson = TSBG family; elastic NO-GO |
| GustavSNN | HPCA 2026 | Column-parallel tick-batch Gustavson | No | — | Broadcast family |
| ASTER | arXiv 2511.06770 | Hybrid analog-digital PIM, membrane buffers | Membrane stay-put | Digital SRAM firewall | Analog CIM |
| Chen/Chang Spike-IAND-Former | arXiv 2503.19643 | 28 nm mux-unroll T=4/2/1, no Vmem SRAM | Unroll template | T=10 ATLIF may keep Vmem | “First mixed-T” claim |
| ITA | arXiv 2307.03493 | Integer softmax: shift normalize, no exp/mul | Shift/round leaf | No softmax unit | ANN softmax |
| SpinalFlow | ISCA 2020 | Timestamp-sorted temporal code, drop Vmem | No | Rate-coded ATLIF/PSN+threshold ≠ temporal code | Fit fail |
| ExSpike | arXiv 2606.20414 | Adjacent-position event compression, attention core | Spatial compress idea | Dirty-run on Motion-XOR tokens | FPGA vs 1RW |
| Comperity | TACO | AND base + XOR diff spike encoding | Related-work knife | XOR is GeMM reuse | Calling it Motion-XOR |
| Sparse HW for Spike-driven Transformer | arXiv 2501.07825 | Dual-spike encode, skip zeros, add-only SDSA | Downstream add-only | Attach to binary ATLIF export | First-SDSA-HW claim |
| Phi | ISCA 2025 | Pattern-wise products L1 | No | Too close to Prosperity | Product family |

## ANN / sparsity / delta

| Paper | Primitive | Steal? | Modify |
|---|---|---|---|
| CBinfer (Cavigelli et al.) | Change-based conv, recompute dirty pixels | Yes | Score-lane dirty (Rank 4); event dirty tile (Rank 8) |
| Delta networks / DeltaCNN | Transmit/compute only Δ above threshold | Yes | Lossless ε=0 first; lossy needs AEE |
| SparTen MICRO’19 | Bitmask inner-join, prefix-sum | Indirect | LoAS already SNN-ified; mixed-T fibers only |
| GoSPA ISCA’21 | Implicit on-the-fly intersection | Indirect | Same |
| SCNN | Cartesian product two-sided sparse | No | Overhead; not our leaf |
| Eyeriss | Row-stationary reuse | No | TSBG family |
| GANAX / Chang dilated-transpose | Skip inserted zeros in ConvTranspose | No new island | 96-lane already full; ≤1.0× |
| FEATHER | Reorder-in-reduction | C2 pipeline only | Not a novelty column |
| OpenEye | Sparse stream / varlen FIFO | Decoder support | Not typed-signed FC claim |
| SpAtten HPCA’21 | Attention-mass cascade | Weak | Score-stationary already nearby |

## Optical flow accelerators / event-OF

| Paper | Primitive | Steal? | Do not |
|---|---|---|---|
| ASNA-Flow TVLSI 2025 | Event-driven async neuromorphic OF; spatial locality; 28 nm 104 FPS 7.9 mW | Cite as prior | Restory spatial locality |
| Zhang et al. CICC 2026 | 28 nm OF: DLSS similarity, BWAC, MaxPool/ReLU speculation; die 1.45 mm² | DLSS **temporal** similarity only | Speculation on Shiftmax; BWAC on our W; ranking their silicon vs our prelayout |
| SpiDR | 65 nm digital CIM SNN, DSEC-flow, unstructured zero-skip | No | Dual-port CIM; generic skip |
| Michigan 28 nm dense OF | 1920×1080 25 fps, 760 mW | Eval template | Dense frame OF |
| Ultra-Flow / Liu TCAS-I 2025 | FPGA OF, direction prediction, 405 FPS | Grok Bot OP-STW uses this | Unfreezes C12 if it becomes residual-flow island |
| ERAFT ISCAS 2025 | FPGA RAFT-like + prediction | Same | Correlation volume forbidden |
| SENECA event OF | Generic spike sparsity on neuromorphic CPU | No | Not a 1RW island |

## Algorithm-only (no matching ASIC for **our** operators)

| Paper | Operator | First-HW opportunity? |
|---|---|---|
| SDformerFlow arXiv:2409.04082 / ICPR 2024 | Swin spikeformer event OF; v1 LIF T=5 SDSA; v2 PSN T=10 linear QK | First HW of **H67 Motion-XOR**, not of published SDSA. 45 nm E_MAC=4.6 pJ / E_AC=0.9 pJ is theoretical. |
| Spike-driven Transformer NeurIPS 2023 / V2 ICLR 2024 | SDSA mask+add | HW already exists (FPGA + 28 nm class). Use as downstream contract. |
| A²OS²A arXiv:2503.00226 | Binary Q, ReLU K, ternary V; no softmax | Opposite encoding to our all-binary. Cover for no-softmax path only. |
| α-XNOR SSA CVPR 2025 | Co-silence α | Steal term, not the paper’s α or task. |
| AT-LIF NeurIPS 2025 | Adaptive threshold `{0,θ}` | Training stabilizer. Frozen inference = static θ. |
| TMA ICCV 2023 / E-RAFT / BAT / EVA-Flow / IDNet | Event OF accuracy methods | Would unfreeze C12 if imported as correlation/anytime islands. |
| PSN NeurIPS 2023 | Parallel spatial-temporal, no reset | GPU sim speed, not proven cheaper than sequential T=10 on 28 nm. |

## Local PDFs / notes Codex may already have

- `SDformer/hw_autoresearch_nts07/docs/228_CICC2026光流芯片借鉴与FAED本土化设计_20260801.md`
- Zhang CICC PDF under hw docs (local)
- `docs/525_公开机制迁移包装与24小时同资源门_20260827.md`
- `reviews/c2_migration_fastkill_r1_20260827/`
- `rtl_h67/h67_motionxor_score_q7.sv` (synopsys_date_dual and/or hw tree)
