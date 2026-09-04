# Exact Product Capture and Context-Safe Weight Broadcast for Event-Driven Spiking Optical Flow

Anonymous submission. Markdown writing draft aligned with the ISCAS 2027 four-page component paper (`paper/iscas2027/main.tex`). If this file and the TeX diverge, **TeX and sealed evidence win**.

This is a component paper on typed-source execution islands. It is **not** a whole-network accelerator claim.

**Target.** ISCAS 2027, Bordeaux. Regular deadline historically 13 October 2026. Planning constraint: four technical pages plus references (recheck the 2027 author kit). Tracks: Digital IC / Neuromorphic.

---

## Abstract

Low firing activity does not automatically accelerate event-driven optical flow: finite storage, single-port state conflicts, and repeated weight delivery can dominate useful arithmetic. We present typed-source execution islands for a frozen binary-event optical-flow network.

**C1** combines exact repeated-product capture with finite single-port parent state and dead-write suppression. On 51.84 million source rows from ten `zurich_city_09_a` samples, its same-ledger cycle model reduces component time by 40.99% (1.6945×). A mapped 28-nm nine-SRAM component meets 3-ns setup/hold and passes 16,549 mapped-to-mapped Formality compare points.

**C2** shares Acc24 state across eight typed-signed sources. Across five directed workloads, against equal-bandwidth K1×8, K8 takes 1,913 versus 1,945 directed VCS cycles (1.0167×) and yields 4.5411× directed-throughput/logic-area with 77.61% less synthesized logic.

**Token-set broadcast (TSBG)** then reuses weight delivery without reusing signed products. Across 1,920 fixed ep34 workloads spanning 40 samples and four DSEC sequences, same-port/cache RTL reduces post-load VCS cycles from 12,522,876 to 5,124,365 (2.4438×; 59.08%) and weight requests by 64.25%. A matched logic-only DC ablation adds 0.0118% area. Matched hold and power remain open.

**C3** closes exact fixed-\(T{=}10\) temporal service at 3 ns and is coverage, not a third speedup. An operator-local INT8 bridge matches 1,280 integer-oracle probes and stays within Acc24; a separately evaluated mixed-precision deployment candidate passes a predefined AEE compatibility gate on an 825-frame local DSEC validation set.

We keep model, RTL, and physical evidence separate and make no whole-network speedup claim.

---

## 1. Introduction

Event cameras make inactivity visible, but sparse arithmetic is only one term in execution time. In our optical-flow deployment, nonzero sources still trigger parent-state traffic, accumulator ownership, and weight-row fetches. A design that reports only operation sparsity can therefore be slower than an equal-resource zero-skipping baseline.

Prosperity and Phi show that useful SNN speedups come from *executable* reuse or patterns, not firing rate alone. FireFly-T shows the value of multi-spike service. Our problem differs in three constraints that those works do not jointly close:

1. a finite 240-KiB-class working set for parent products;
2. polarity-bearing typed sources that forbid coalescing equal indices into one signed product;
3. mixed fixed-\(T\) neuron phases that must preserve software commit order.

We therefore treat the frozen network as an evaluation workload, not as a parallel algorithmic novelty claim, and map three exact invariants onto execution islands that share a typed-source protocol but have **not** been measured as a monolithic top.

**Contributions.**

1. **C1** captures exact repeated products under a finite-capacity, single-1RW parent-store contract, with residual reconstruction and atomic completion.
2. **C2** shares Acc24 and endpoint state across eight typed sources and embeds token-set broadcast (TSBG), which reuses a weight *row* while preserving independent signed products and accumulator contexts.
3. **C3** provides exact fixed-\(T{=}10\) temporal service as network-coverage evidence rather than a performance claim.

Attention and decoder traces remain part of the frozen identity. They are not additional performance contributions: attention occupies about 0.6% of a historical cycle envelope, and the decoder has no closed full-network table in this paper.

---

## 2. Workload and Execution Contract

### 2.1 Frozen network

The frozen Motion C12 checkpoint (epoch 34) obtains **AEE 1.199514** on an 825-frame local DSEC validation split (18 sequences, 48,152,523 valid pixels) and has a **5.6709%** global firing rate. Geometric parameters consumed by hardware are \(T_{\mathrm{snn}}=10\), attention window \(T_w=2\), \(15\times 15\) spatial windows, and \(N_{\mathrm{tok}}=450\). The encoder uses twelve unified attention blocks on a multi-scale spiking Swin-style backbone with 96-channel bottleneck convolutions and a convolutional decoder.

Of 105 configured ATLIF wrappers, 12 `sn2_q` wrappers are runtime-bypassed and never called, so the capture invokes **93** (48 \(T{=}2\), 45 \(T{=}10\)). A separate graph audit finds that 12 invoked `attn_sn` return values have no consumer under fixed normal inference, leaving **81** graph-live services (36 \(T{=}2\), 45 \(T{=}10\)). These two groups of twelve must not be conflated. All 93 captured ATLIF outputs are binary. Signed source codes are therefore a downstream execution-protocol property (polarity, correction traffic), not evidence that captured ATLIF tensors are analog.

The frozen software path uses Motion-XOR scoring with \(\alpha{=}0.125\) and **K as its value carrier** (\(\mathrm{attn}=\mathrm{gate}\odot K\)), but hardware quantization is disabled. Only a separately evaluated deployment candidate enables Q7 round-to-nearest scores, a next-power-of-two integer Shiftmax row sum, and Q1.7 gates, plus dyadic-INT8 QDQ on eight retained convolutions. Attention is not a performance contribution, and fractional gates are not called integer powers of two.

### 2.2 Evidence classes

| Class | Meaning | May be called RTL speedup? |
|---|---|---|
| Cycle model | Same trace, same charged resources, software replay | No, unless all three classes close on one workload |
| VCS | Directed or real-activity RTL function/protocol/cycles | Only together with matched physical proof on that workload |
| DC/PT/Formality | Area, timing, equivalence | Physical anchor only |

This paper reports the three classes side by side. Prelayout physical points use a commercial 28-nm flow at 3.0 ns, ideal clocks, ZeroWireload, and no SPEF.

### 2.3 Three exact invariants

- Bottleneck convolution exposes repeated products **only while parent and destination state remain resident** → C1.
- FC tokens can share a weight-row identity while retaining different signs, destinations, and accumulators → C2/TSBG reuses **delivery**, not products.
- Fixed-\(T{=}10\) neurons require a deterministic state and commit order → C3.

Exact fallbacks: if C1 finds no resident legal parent, it issues the original source row; if token contexts disagree on a weight-row identity, C2 emits separate fetches; sign, destination, and Acc24 state are never coalesced. Backpressure changes only service order, because the typed-source terminal bit and atomic commit keep ownership explicit. Thus sparsity affects delivered work without changing the operator result.

**Figure (overview).** Two exact reuse objects under one typed-source protocol. C1 retains only legal live parents in a finite 1RW store; C2/TSBG broadcasts one delivered weight row while keeping products and Acc24 contexts private. C3 is coverage, not a third performance claim; the branches are not a measured monolithic top. (Canonical figure: `paper/iscas2027/figures/typed_source_islands.pdf`.)

---

## 3. Microarchitecture

### 3.1 C1: Capacity-constrained product capture

C1 builds a ping-pong directory for each 64-row, 16-source mask task. A row may select only an earlier **exact-subset** row as parent; it then issues the XOR residual and reconstructs the 96-lane signed product row by adding the parent:

\[
\mathbf{P}_r = \mathbf{P}_p + \sum_{s \in \mathcal{M}_r \setminus \mathcal{M}_p} \mathbf{p}_s.
\]

Exactness follows from the disjoint residual rather than value prediction. Stable population-count ordering and a live-parent bitmap make the relation deterministic and identify final values that no later row can observe. Such dead writes are elided exactly.

The execution side combines:

- one earliest-parent lookahead;
- a two-entry reserved response queue;
- forwarding under a single-1RW deadline-aware arbiter;
- atomic final psum write and row completion, so backpressure cannot expose a partially reconstructed row.

The scheduler uses nine \(128\times 128\) 1RW SRAM macros (logical capacity about 215 KiB inside a 240 KiB budget) and never assumes the unphysical concurrent 1R1W ceiling that produced the higher reuse bound.

**Object difference versus unbounded product sparsity.** Reuse must survive capacity, port, parent lifetime, and completion. Prosperity isolates product-over-bit sparsity as an opportunity; C1 is the finite-store, single-port, atomically completing realization of that object on this ledger.

### 3.2 C2: Typed K8 and token-set broadcast

The K8 fabric accepts at most eight nonzero typed sources and shares the Acc24 array, endpoint, and control rather than replicating eight K1 engines. The fair baseline is therefore **K1×8 at equal source bandwidth**, not one K1 lane. FireFly-T’s multi-spike service motivates that equal-service comparison.

TSBG changes only weight delivery. Four token contexts expose a common source-group row; one fetched row is broadcast to four independent signed products and Acc24 contexts:

\[
\operatorname{fetch}(W_g)=1,\quad A_c \leftarrow A_c + v_c W_g,
\]

with \((v_c, d_c, A_c)\) private. It never reuses a product because equal source indices can carry different signs or destinations. Inactive sources do not issue and never update an accumulator. Negative sources use an exact nine-bit two’s-complement negation before Acc24 so that INT8 \(-128\) maps to \(+128\) without eight-bit wraparound.

Both ordinary and TSBG modes use the same 192-bit row-live map and hierarchical priority encoder. Generate-time wiring orders that bitmap token-major (`SCHEDULE_MODE=0`) or source-group-major (`SCHEDULE_MODE=1`). Empty descriptors never set the live map, so the finder skips them combinationally without a bubble. The synthesizable scheduler contains no runtime division, remainder, or two-dimensional active-array read.

The production engine reported here is the G48 configuration: 48 source groups, 16 sources per group, four cache rows, six output slices, and an eight-bank weight protocol whose ready/response channels are independent per bank. Bundle width \(B{=}4\) is the selected point; \(B{=}2\) was dropped; \(B{=}8\) remains an unphysicalized upper bound.

ELSA and SpikeX establish bundle/Gustavson and cross-window weight-reuse priors. The narrower contribution is the mapping onto a polarity-bearing stream with four private destination/Acc24 contexts under the same finite row cache and public-port contract.

### 3.3 C3: Exact fixed-\(T\) coverage

C3 schedules the deployed \(T{=}10\) ATLIF recurrence and preserves integer state and commit order. It closes the temporal service boundary so that C1/C2 numbers are not those of a truncated network. It is **not** a standalone speedup and is not allocated a contribution bullet or a speedup column.

---

## 4. Evaluation

Physical results use a commercial 28-nm flow at 3.0 ns, prelayout, ideal clocks, ZeroWireload, no SPEF. C1 area includes nine SRAM macros; C2 and TSBG/C3 companions reported here are logic-only unless noted.

### 4.1 Admitted component table

| Line | Performance | Physical anchor | Boundary |
|---|---|---|---|
| C1 | 1.6945×; −40.99% time **[model]** | 166,514 µm²; PT setup/hold +27.9/+1.8 ps; 16,549 mapped-to-mapped Formality | One-sequence model + separate mapped anchor; one real-mask tile calibrates event counts only |
| C2 | 1,913 vs 1,945 cycles (1.0167×) **[VCS]**; 4.5411× directed-throughput/logic-area | 131,086 vs 585,479 µm²; −77.61% | K8 vs equal-bandwidth K1×8, one campaign; hold/power open |
| C2/TSBG | 12,522,876 vs 5,124,365 (2.4438×) **[VCS]**; −59.08% time; −64.25% weight requests | 249,710 vs 249,740 µm²; +0.0118%; setup met | 1,920 fixed ep34 real-activity workloads; G48 engine; 383-cycle preloads excluded; no full-FC/system claim |
| C3 | 17 cycles/tile | 63,756 µm²; DC/PT setup+hold; 11,180 compare points | Exact service; Formality is hold-repaired mapped-to-mapped; no speedup/energy |

### 4.2 C1 same-ledger ladder

C1 model cycles replay ten `zurich_city_09_a` ep34 samples and 51.84 million source rows from four bottleneck Conv3×3 layers.

| Execution point | Cycles (M) | vs. zero | Boundary |
|---|---:|---:|---|
| Strong zero | 648.741 | 1.000× | baseline |
| Same-coordinate bit | 646.619 | 1.003× | baseline |
| C1 finite 1RW | 382.849 | **1.6945×** | cycle model |
| Concurrent-access | 341.058 | 1.902× | ceiling only |

Plain bit skipping barely changes time. Exact parent reuse creates the material reduction. The remaining **12.25%** gap to the concurrent-access ceiling is the price of the physical single-port contract, not an omitted claimed speedup. Versus same-coordinate bit the model is 1.689×. Per-sample ratios span 1.661–1.723× (geometric mean 1.6946×).

C1 mapped power: 29.08 mW and 22.07 nJ over a 64-row, 253-cycle directed window (mixed corner: standard-cell TT 0.9 V 25 C + SRAM SSG 0.9 V 125 C; no SPEF; not energy/frame). Parent scratch is 36.1% of that window power.

A sealed 64-row real-mask VCS tile matches 196 issues, 58 parent edges, 31 dead-write elisions, 54/33 macro R/W, four forwards, six deadline holds, 14 stalls, and 64 commits. Lane values are synthetic signed-12 with zero prior psum and **do not upgrade 1.6945×**.

### 4.3 C2 equal-bandwidth fabric

Across five frozen directed workloads, K8 versus K1×8 takes 1,913 versus 1,945 VCS cycles (1.0167×). Matched logic-only synthesis occupies 131,086 versus 585,479 µm² (−77.61%). Directed-throughput/logic-area is

\[
\frac{1945\times 585479}{1913\times 131086}=4.5411\times.
\]

Cycle count and area efficiency must appear in the same sentence. K8 is not a sparsity speedup against a single K1 lane. Both axes meet 3-ns setup; independent matched hold and power remain open. Historical hold diagnostics are on the order of −20 ps, dominated by array Q→D self-feedback; a global hold-buffer repair previously exceeded a 5% area cap and is not reapplied.

### 4.4 TSBG distribution (1,920 workloads)

Selection is fixed *before* measuring performance: all 40 captured samples, all 12 FC1 layers, the four FC2 layers whose inputs fit the same G48 engine, and first/middle/last aligned B4 quartets. Eight FC2 layers above G48 are excluded. Identical 383-cycle preloads are excluded from the ratio. The sealed population is a 1,917+3 same-`simv` cross-attempt result; the failed parent attempt is not citable.

| DSEC trace | Ordinary cycles | TSBG cycles | Speedup |
|---|---:|---:|---:|
| interlaken_01_a | 3,120,326 | 1,225,652 | 2.5458× |
| thun_01_b | 2,921,236 | 1,182,417 | 2.4706× |
| zurich_city_09_a | 3,204,120 | 1,345,486 | 2.3814× |
| zurich_city_12_a | 3,277,194 | 1,370,810 | 2.3907× |
| **aggregate** | **12,522,876** | **5,124,365** | **2.4438×** |

Scalar weight requests fall from 8,774,304 to 3,136,608 (−64.25%). FC1/FC2 weighted ratios are 2.5466× / 1.9680×. The population retains 286 all-zero workloads; 1,343 improve, 570 tie, **seven are slower**, median 1.8373×, worst nonempty **0.9935×**. Natural nonzero activity codes in this measured set are \(+1\); directed INT8 weights exercise arithmetic but do not affect scheduling. The same runs check bank stalls/reordering and stale responses; a subsequent synthetic recovery phase checks signed products, reset, and the \(-128\) corner.

Software screening of 960 FC1/FC2 pairs (2.534× modeled cycles, −65.20% traffic, per-sequence minimum 2.517×) remains a CPU premodel and is **not** the headline RTL result.

Physical companion: same B4 RTL, `SCHEDULE_MODE` 0/1 only; ordinary 249,710 µm² vs TSBG 249,740 µm² (+0.0118%); setup met; diagnostic hold −16.4 ps; macros/power/hold-closure open. This is a schedule-mode ablation, **not** TSBG versus thin K8 (~131 kµm²). State is implemented with standard cells in this experiment.

A directed VCS protocol check separately reduces weight bundles 576→144 and scalar-bank requests 4608→1152 (−75%). That check does not replace the 1,920-workload real-activity cycle result.

FC2 layers above G48 remain a CPU/source model only (about 1.87× on 960 pairs; about 2.23× when combined with the measured 24-layer mix). They are not RTL.

### 4.5 Numerical binding and task gate

Eight retained Conv3×3/ConvTranspose weights are quantized per output channel to narrow-range signed INT8 with a power-of-two scale. An independently checked export has zero pre-clip violations. Over 160 full calls and 3,271,680,000 output elements, versus FP32: MAE 0.009106, RMSE 0.012407, maximum absolute error 0.239204, cosine similarity 0.9999789. All 1,280 independently sampled Python-integer probes match. Static accumulator bounds are 200,219 (C1) and 87,136 (decoder), both below signed Acc24. The observed final-accumulator range is \([-29680, 27619]\). These are operator-local numerical results, not task AEE, downstream-neuron equivalence, or a hardware speedup.

**Task accuracy** uses the same 825 frames / 18 sequences / 48,152,523 valid pixels. Candidate = hardware-order attention on all 12 blocks + dyadic-INT8 QDQ on the eight weights. Historical baseline enabled TF32/cuDNN benchmarking; candidate disables both. Because the backends differ, the table demonstrates accuracy-gate compatibility, **not** a causal accuracy improvement from quantization.

| Metric | Baseline | Candidate | Δ |
|---|---:|---:|---:|
| AEE | 1.199514 | 1.197367 | −0.002147 |
| AAE | 5.400641 | 5.412808 | +0.012167 |
| AAE-Bench. | 5.106363 | 5.121619 | +0.015256 |
| DSEC-Fl | 5.313360 | 5.328834 | +0.015475 |

The contracted gate is candidate−baseline AEE \(\le +0.02\) (pass). Ten of eighteen sequences regress in AEE. Auxiliary errors all rise. This is frozen-population compatibility, **not** full-network INT8.

AEE is average endpoint error (the preselected gate). AAE is legacy 2-D angular error. AAE-Bench. is Middlebury/Barron 3-D angular error. DSEC-Fl is the percentage with endpoint error above 3 pixels and 5% of ground-truth flow magnitude.

---

## 5. Related Work and Comparison Discipline

Prosperity isolates product-over-bit sparsity; that ablation, not its PTB system speedup, is the relevant comparison for C1. Phi chooses a storage point by DSE and compares a complete hierarchical-pattern architecture with published baselines. We adopt the useful evaluation pattern—cycle simulation plus RTL/physical anchors and a baseline ladder—without relabeling their numbers as ours.

FireFly-T motivates equal-service comparison for C2. ELSA/SpikeX establish bundle and weight-reuse priors; TSBG is claimed only as a typed-signed, finite-context delivery specialization inside C2. SpinalFlow and SNE establish event-driven execution but do not subsume finite-parent product reuse or typed cross-token delivery. A 28-nm optical-flow chip supplies a metric template (operations, traffic, latency, energy, AEE); we do not rank silicon against prelayout islands.

Official Prosperity artifact opportunities on a related checkpoint (Conv product-vs-bit \(\approx 2.46\times\), FC1 \(\approx 2.37\times\), decoder exact subset \(\approx 3.09\times\)) remain **external / not ours**.

Three comparison levels, never multiplied: (i) same-network component baselines; (ii) iso-workload public artifacts; (iii) published technology-normalized metrics.

| Work | Prior object | Boundary here |
|---|---|---|
| Prosperity | repeated products | C1 adds finite 1RW parent lifetime, exact residual, atomic completion |
| Phi | hierarchical patterns | no substitute for dynamic parents or signed contexts |
| FireFly-T | multi-spike service | equal-bandwidth K8, not one K1 lane |
| ELSA / SpikeX | bundle / weight reuse | TSBG keeps private sign, destination, Acc24 |
| SpinalFlow / SNE | event-stream execution | do not subsume finite-parent product reuse or typed cross-token delivery |
| CICC optical flow | silicon system metrics | metric template only; no cross-platform speedup against prelayout islands |
| This work | exact parent + typed delivery | same-ledger model, VCS, and 28-nm islands; no system speedup |

Lossy pruning, N:M sparsity, integer-power gate skipping, and accumulator-width reduction are outside the scope of this exact component paper.

---

## 6. Limitations

- C1’s 1.6945× is a **cycle model** on one sequence, calibrated by one real-mask RTL tile; the nine-SRAM mapped component does not integrate the full 240-KiB common-charge ledger.
- C2 area efficiency is logic-only; hold and power are open.
- TSBG RTL cycles cover 1,920 fixed B4 workloads but exclude eight FC2 layers above G48 and the full token population; seven nonempty cases are marginally slower (worst 0.9935×). Full-capture external weight-service time remains modeled.
- The VCS population is a double-sealed 1,917+3 same-image result; the failed parent attempt is non-citable.
- Matched-macro C2 hold/power, full-network FPS/energy, decoder-complete system tables, and monolithic island integration remain open and are **not** derived from component ratios.
- The deployment-accuracy row is referenced to a historical baseline with different GPU backend flags.

---

## 7. Conclusion

The principal hardware result is not firing sparsity, but its **executable capture** under finite storage and service constraints. C1 provides bounded exact product reuse. C2 turns typed multi-source service into area efficiency and uses TSBG to suppress repeated weight delivery. C3 closes exact temporal service. Separating model, VCS, and physical evidence yields a compact ISCAS component paper without an unsupported whole-network claim.

---

## References

1. C. Wei et al., “Prosperity: Accelerating Spiking Neural Networks via Product Sparsity,” *Proc. IEEE HPCA*, 2025.
2. C. Wei et al., “Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency Spiking Neural Networks,” *Proc. ACM/IEEE ISCA*, 2025.
3. T. Li et al., “FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer Acceleration with Dual-Engine Overlay Architecture,” *IEEE Trans. Comput.*, vol. 75, no. 6, pp. 2185–2199, 2026.
4. K. You et al., “ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing,” *Proc. ACM/IEEE ISCA*, pp. 2550–2566, 2026.
5. B. Xu, R. Boone, and P. Li, “SpikeX: Exploring Accelerator Architecture and Network-Hardware Co-Optimization for Sparse Spiking Neural Networks,” arXiv:2505.12292, 2025.
6. S. Narayanan et al., “SpinalFlow: An Architecture and Dataflow Tailored for Spiking Neural Networks,” *Proc. ACM/IEEE ISCA*, pp. 349–362, 2020.
7. A. Di Mauro et al., “SNE: An Energy-Proportional Digital Accelerator for Sparse Event-Based Convolutions,” *Proc. DATE*, pp. 825–830, 2022.
8. T. Zhang et al., “A 28-nm Optical Flow Estimation Accelerator with Redundancy Speculation, Bit-Width-Aware Compression and Similarity Detection,” *Proc. IEEE CICC*, 2026.

Canonical BibTeX: `paper/iscas2027/references.bib`.

---

## Appendix A — Claim boundaries for later TeX compression

Keep these sentences if the four-page budget tightens. Do **not** restore weak mechanisms to fill space.

- Do not multiply 1.6945×, 1.0167×, 2.4438×, and 4.5411×.
- Do not write K8 versus a single K1 as sparsity speedup.
- Do not promote the TSBG CPU premodel 2.53× into the abstract or main hardware table.
- Do not call C3 a third speedup.
- Do not call Q1.7 gates integer powers of two.
- Do not claim causal accuracy improvement from the mixed-precision candidate.
- Do not publish 0.0118 **W**; the physical tax is **+0.0118%** area.
- Reproduce layout with `tectonic main.tex` and run `python3 paper/iscas2027/check_claim_boundaries.py` before changing headline numbers.
