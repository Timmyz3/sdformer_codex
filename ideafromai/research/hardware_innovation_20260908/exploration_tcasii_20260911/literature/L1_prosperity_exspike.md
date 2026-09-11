# L1 deep-read: Prosperity, ExSpike, SpikeX vs THIS net

Freeze used for THIS net: `exploration_tcasii_20260911/PROBLEM.md` (2026-09-11).
Session gate from `SCOPE.md` is cited only as a kill-number, not as extra measurements.

**File-title check.** All three local texts match the labels in the task:

| Local file | Printed title | Venue line in file |
|---|---|---|
| `.../p0_txts/2503.03379.txt` | **Prosperity: Accelerating Spiking Neural Networks via Product Sparsity** | 2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA) |
| `.../p0_txts/2606.20414.txt` | **ExSpike: A General Full-Event Neuromorphic Architecture for Exploiting Irregular Sparsity with Event Compression** | PREPRINT – To appear at FPL 2026 |
| `.../p0_txts/2505.12292.txt` | **SpikeX: Exploring Accelerator Architecture and Network-Hardware Co-Optimization for Sparse Spiking Neural Networks** | arXiv:2505.12292v1 [cs.NE] 18 May 2025. **This is SpikeX**, not a different paper. |

No citations are invented. Every mechanism claim below is from the local full text of that paper or from the freeze.

THIS net (frozen, not a novelty claim): event-camera 2D optical flow, DSEC valid825 AEE; student Motion C12 / H67 / ep34; **ATLIF with continuous threshold amplitude θg, not binary spikes**; expensive region historically patch-embed residual r1 (two convs) plus T10 PSN; **dual consumers** after source (spike/gate path **and** continuous residual/PED path); native projection conv + BN + residual add; **noncausal T10** at this PSN; native projection BN uses **actual batch statistics over the full 10×96×120×160 domain**. Ordinary dense-source AEE = 1.219801338. Lifting T10 AEE = 1.232979368 (relative +0.005 fails). Same two-stage SIMD source: always-ready slots 6938 → 5354 (−22.83%) but **long backpressure both 8088**. Integer consumer model (different resource point): 758777 → 714889 (−5.78%); do not add the two tables.

---

## Cross-walk (so the three papers are not collapsed into one slogan)

All three papers accelerate **binary-spike × multi-bit-weight** work. They are not the same mechanism:

- **Prosperity / ProSparsity:** combinatorial **reuse of already-computed inner products** of binary GeMM *rows* that are equal or subsets. Extra sparsity *beyond* zero-skipping.
- **ExSpike / APEC:** combinatorial **reuse of overlap of adjacent spatial positions’ active-channel sets** (the intersection case Prosperity explicitly refuses for GeMM rows). Plus a full-event EConv dataflow that zero-skipping already implies.
- **SpikeX:** **do not compute / do not fetch** for inactive time-windows and inactive input channels; **share multi-bit weights** across time and across postsynaptic neurons on a systolic array. Plus **train the net to be sparser**. This is bit-sparsity + time-batching + weight-stationary dispatch, **not** product-of-rows reuse.

Renaming any of {skip zeros, share weights in a time window, AND-overlap of binary supports, subset-prefix psum} as a new TCAS-II title is exactly what reviewers will call a reskin.

---

# Paper 1 — Prosperity (HPCA 2025), arXiv:2503.03379

## 1. Exact mechanism (equations, dataflow, what is reused)

**Workload they accelerate.** Layer-by-layer SNN inference whose dominant op is **Spiking GeMM**: a binary spike matrix times a multi-bit weight matrix. Time is **not** mixed by a dense T×T matrix. Time steps are **unrolled and concatenated as extra rows** of one binary matrix that **share the same weights**.

Quoted identity of the op (Background §II-A):

> “we can unroll and concatenate all spike matrices in different time steps to get a single binary spike matrix for an SNN layer. The major computation of the SNN model is the matrix multiplication of the binary spike matrix and floating point weight matrix. We define this unique operation in SNNs as Spiking GeMM. […] In spiking GeMM, the binary spike matrix only consists of 0s and 1s, i.e., bit sparsity (BitSparsity). Therefore, the computation becomes a sparse addition. If an element is 1, then the corresponding weight is straightly accumulated to the result; if an element is 0, then the corresponding weight is skipped.”

They claim “more than 98% percent of operations in SNNs are spiking GeMM.” CNN conv is lowered by im2col; transformer Q/K/V/out/FFN are native GeMM of shape `(T×L, di) × (di, do)`.

**Bit sparsity (what they start from).** For spike matrix `M ∈ {0,1}^{M×K}`, weight `W ∈ R^{K×N}`, output `Y ∈ R^{M×N}`:

- row `i` of `Y` is the sum of those rows of `W` whose column index is 1 in spike row `i`.
- Zero bits skip the corresponding weight row. That is PTB/SATO/MINT-class skipping.

**Product sparsity (the new reuse).** Treat spike row `i` as a set of firing columns:

```
Si = { j | M[i,j] = 1, j ∈ Column IDs }
```

For two rows `i, j`, intersection `A = Si ∩ Sj` with `A ≠ ∅`. Three cases (§III-B):

1. **Partial Match (PM):** `A = Sj` and `A ≠ Si` — `Sj` is a **proper subset** of `Si`. Compute inner product of the prefix row once; suffix **reuses that psum** and only accumulates the remainder `Si \ Sj`.
2. **Exact Match (EM):** `A = Si = Sj`. Copy the entire output row; skip all spikes of the duplicate.
3. **Intersection (neither is subset):** they **do not use this**. It would require materializing a new row `A` and computing it first.

Worked example from Fig. 1 / Fig. 2: Row 1 = `1001`, Row 4 = `1101` share sub-combination `1001`. Reuse Row 1’s product; only add the weight for the extra `0100`. Row 5 = Row 4: copy Row 4’s result with **zero** extra adds.

**What is actually reused.** Not weights (those are already shared by GeMM). Reused object is the **already-computed output-row partial sum** of a **prefix spike combination**. Remainder pattern is

```
ProSparsityPattern = Sq XOR Sp    (= Sq \ Sp when Sp ⊂ Sq)
```

Processor dataflow (§V-E, “ProSparsity Row-wise Dataflow”):

1. Issue spike rows in a **topology order** of the ProSparsity Forest so the prefix output row already sits in the output buffer.
2. Load that prefix output row into the PE-array psum register.
3. Bit-scan-forward on the XOR pattern; for each remaining 1, fetch that **weight row** and accumulate onto the psum.
4. Write the completed output row back. Across K-tiles, results further accumulate onto the output tile.

**Spatial vs temporal relation.**

- Spatial: which row is whose prefix (subset / equal).
- Temporal: **execution order**. PM prefix is the **smaller set** (fewer ones). EM prefix is the **smaller row index**. If you process a superset first, you cannot reuse a subset that has not been computed (Fig. 2 Row 0 vs Row 3).

**Complexity cut they actually implement (§III-D, §V).**

- Naive n-row combinatorial search is `O(m^n)`. They **only** consider **two-row** relations (`O(m^2)`).
- Graph of prefix edges is pruned to a **forest**: each node keeps **one** prefix, the one with the **largest common subset**; ties take the **largest index**.
- Reason: a second prefix costs “30% performance drop and larger area”; Table II, second prefix usable on `<6%` of nodes; first prefix already takes most of the density drop (e.g. SpikingBERT SST-2 bit 20.49% → one-prefix Pro 2.98% → two-prefix 2.30%).
- Meta information stored, not the graph: (i) temporal vector of length `m` (issue order); (ii) per-row `{Prefix index, ProSparsity XOR pattern}`. Space `O(m)`, not `O(m^2)`.

**Architecture (Prosperity / PPU), Fig. 4–6.**

- **Detector:** preload `m×k` spike tile into **TCAM**. Query a row with 1s masked to `X` (don’t-care); TCAM returns all subset indices in **one cycle**. Popcount gives number-of-ones (NO) as a cheap temporal key.
- **Pruner:** proper-subset filter (drop EM candidates with larger index) + Argmax over remaining SI to pick one prefix; XOR to form the pattern.
- **Dispatcher:** product-sparsity table `O(m)`; **does not** store suffixes. Execution order is **not** DFS/BFS of the forest (that needs `O(m^2)` suffix matrix). Instead: **stable sort by NO** (PM: fewer ones first; EM: same NO, smaller index first because sort is stable). Bitonic sorter `O(log² m)` time, `O(m)` space.
- **Processor:** 128× 8-bit **adders** (not MACs). Row-wise, n=128. Address decoder = bit-scan-forward on the XOR pattern. Unstructured zero skip **and** prefix psum reuse.
- After a layer GeMM: **Spiking Neuron Array** (32 LIF cells) produces binary spikes for the next layer. **SFU** (AND/OR, MUL, EXP, DIV) for softmax / LayerNorm of spiking transformers. PPU is reused for “spiking-GeMM-like operations in spiking attention.”
- **Tiling:** `m×n×k` with chosen `m=256, n=128, k=16`. ProSparsity **only exists inside a tile**. `m=1` kills reuse. Larger `m` → more prefix candidates, super-linear TCAM/buffer cost. Larger `k` → sets more unique, fewer subsets; `k=4` makes 0–1-spike rows whose reuse is “meaningless.” `n` does not affect sparsity.
- **Pipeline (§VI):** intra-tile, Detector/Pruner/Dispatcher are 5-stage, `m+4` cycles for spatial info. Inter-tile: **ProSparsity processing of tile n overlaps computation of tile n−1** via double-buffered sparsity table and TCAM. They claim the processing phase is always shorter than compute, hence “overhead-free” except the first tile.

**Cost model they use to justify CAM (§VII-G).** Dominant overhead is TCAM bitwise matches `m² × k`. Saved work is `ΔS × m × k × n` floating-point adds, each add counted as 45× one TCAM bit-op. Benefit if `ΔS > 4.4%` at their tile; they report average `ΔS = 13.35%`, ratio ~3.0×. This is **their** 8-bit-add vs TCAM accounting, not a port-limited SIMD schedule.

**Evaluation (what was measured).** Cycle-accurate **simulator** on extracted PyTorch spike maps. RTL SystemVerilog synthesized with Design Compiler, ARM 28 nm std cells, 500 MHz; buffers via CACTI 7.0 28 nm; DRAM via DRAMsim3. Iso-accuracy because the method is lossless on GeMM. Models: VGG-16, ResNet-18, Spikformer, SDT, SpikeBERT, SpikingBERT; CIFAR10/100, CIFAR10-DVS, GLUE/SST/MR. Table IV VGG-16: 128 PEs, 0.529 mm², 390.10 GOP/s, 299.80 GOP/J. Geometric mean vs PTB **7.4× speedup, 8.0× energy**; vs A100 **1.8× / 193×**. SpikeBERT: bit density 13.19% → Pro 1.23%, “11×” computation. Ablation Fig. 9: unstructured bit-skip 2.28× over PTB; ProSparsity with heavy dispatcher 2.16× more; overhead-free order 1.49× more. EM still costs **one cycle** even at 100% pattern sparsity. LoAS weight-pruned nets: ProSparsity still cuts **activation** density ~4.1× (orthogonal to weight pruning). Area: Dispatcher (sparsity table) dominates except buffers; on-chip power: TCAM Detector dominates because every CAM cell fires every cycle. DRAM is a large fraction of 915 mW on Spikformer/CIFAR10.

## 2. Identity assumptions (binary spikes? product of binary? 1RW?)

| Assumption | In Prosperity? |
|---|---|
| Activations are **binary {0,1}** | **Required.** Intro: “binary spikes (1 for spike and 0 for non-spike).” GeMM inner product **is** “accumulation of multiple weight values, which are ‘selected’ by the 1-values.” |
| Output of a combination depends **only on the set of 1-positions**, not on an amplitude | **Required.** Identical binary rows ⇒ identical products. Subset ⇒ product is a **prefix** of the superset product. If the “1” carried a per-row continuous θg, identical supports would **not** share products. |
| Product of binary × multi-bit weight = **conditional add**, no multiplier in the PE | **Yes.** ALU is “8-b Add” (Table IV). |
| Time is extra independent GeMM rows sharing W, **not** a noncausal mix of the T samples of one site | **Yes.** Explicit unroll/concat. |
| One consumer of Y: the next LIF / next GeMM | **Yes.** Dual residual/PED consumers are not in the dataflow. Residual in transformers is not modeled as a continuous second reader of the source. |
| **1RW / same-port SRAM** | **No.** TCAM is a parallel associativity structure, not 1RW. Output buffer must supply **prefix psums** while also accepting writes. Double-buffered spike/weight/output/sparsity tables. 128-wide PE row. |
| Lossless / algorithm-agnostic | **Claimed.** “ProSparsity is an algorithm-agnostic and lossless method for the spiking GeMM.” They do **not** retrain. Contrast Stellar (FS neuron, algorithm change). |
| Two-row, one-prefix, tile-local | **Hardwired.** Intersection off. Second prefix off. Cross-tile prefix off. |
| Weights 8-bit; neuron = LIF after GeMM | Default config Table III. |

## 3. What they explicitly do NOT do

Direct refusals / absences in the text:

- **Do not use the intersection case.** §III-B: “leveraging this scenario requires creating a new row A and compute the result of A first. This will significantly increase the complexity of the architecture design. Hence, this study will only consider the former two relationships, EM and PM.”
- **Do not keep two prefixes per row.** §III-D: second prefix “30% performance drop and larger area”; `<6%` of nodes can use it.
- **Do not run n-row (n≥3) subset search.** “when n ≥ 3, the complexity of the identification will be unacceptable.”
- **Do not search the forest from suffixes.** Suffix matrix is “more than 10 times” on-chip area; they replace it by NO-sort.
- **Do not modify the SNN algorithm / neuron to create sparsity** (that is Stellar). Table I: Stellar = “Specific Neuron”, Prosperity = “Algorithm-agnostic.”
- **Do not process analog / CIM.** Digital ASIC story, 8-bit add.
- **Do not claim foundry-signoff PPA.** 28 nm DC + CACTI + DRAMSim3 + cycle-accurate simulator. End-to-end GPU numbers are wall-clock; ASIC numbers are simulated.
- **Do not support prior ASICs’ missing transformer attention as GeMM alone**; they add an SFU. They still only **reuse PPU for GeMM-like** pieces of attention, not a second continuous residual algebra.
- **Do not skip the one cycle of an exact-match row.** “EM ProSparsity has 100% sparsity but still takes one cycle to process.”
- **Do not apply ProSparsity across tiles.**
- **Do not treat activations as continuous amplitudes.** No BN-over-full-map, no optical-flow AEE, no dual PED consumer, no noncausal T10 PSN matrix.

Related-work boundary they draw (§VIII-C): GNN redundancy-removal is **not** transferred, because SNN spike patterns “exhibiting random distributions due to dynamic input activation” and SNN density is 60–90%, not ≥99.99% graphs.

## 4. Hole vs THIS net

THIS net breaks **every** load-bearing identity of ProSparsity.

1. **Continuous θg, not binary 1.** Prosperity’s reuse theorem is: same binary combination ⇒ same inner product. ATLIF here emits **continuous threshold amplitude θg**. Two sites with the same support and different θg do **not** share a psum. Factoring “support prefix + scale by θg” is already a different algebra (real multiply or per-row scale), and the freeze forbids changing identity to binary ATLIF.
2. **Dual consumers.** Source is read by (a) spike/gate path **and** (b) continuous residual/PED, plus native projection conv + BN + residual add. Prosperity retires a spike row when **one** GeMM output row is done. Skipping / reusing the gate GeMM **does not** cancel source production the PED still needs. The historically expensive region is **r1 two convs + T10 PSN**, not “98% spiking GeMM of binary rows.”
3. **Noncausal T10 PSN.** Prosperity concatenates time as independent binary rows. THIS PSN is a **noncausal 10-point mix**. A “prefix in time” is not a subset of firing columns; you cannot issue T-rows in NO-order without changing the T10 function. Official CSE on the T10 graph is already in the freeze (ordinary 260 add/sub per T10 vector → lifting 159 add/sub + 35 RNE/sat). That is **compiled linear-form reuse**, not product sparsity of binary rows.
4. **Full-domain BN.** Prosperity has no consumer that waits for batch mean/var over `10×96×120×160`. Local-window replay with free mean/var “undercharges wait/storage” (freeze). Prefix psum reuse does not shorten that wait.
5. **Same-port / backpressure, not op count.** Lifting already cut the T10 add graph and **always-ready slots −22.83%**, and **long backpressure stayed 8088**. Prosperity’s speedup story is “fewer 8-bit adds in an adder array whose issue width is the XOR-pattern popcount,” plus CAM overlapped for free. On this SIMD source, **fewer adds can be entirely absorbed by backpressure**, which is exactly what lifting already showed. Their ΔS>4.4% CAM-vs-add model does not include 1RW prefix-psum traffic, dual-consumer live ranges, or BN wait.
6. **TCAM is not a free detector here.** Detector power is the largest on-chip compute power in their own breakdown *because every CAM cell is active every cycle*. Mapping that onto a port-limited student as “overhead-free” is false under the freeze.

## 5. Honest complete-transfer checklist to copy A

To **actually copy** Prosperity, not cite it, you must implement **all** of the following on the **binary GeMM they defined**. Anything less is a rename.

1. Lower the target layer to **binary** spike matrix `M` and multi-bit `W` such that `Y[i,:] = sum_j M[i,j] W[j,:]`.
2. Tile `M` to `m×k` with `m` large enough for subsets (they needed m=256, k=16). Pay the **on-chip spike tile + TCAM m×k**, weight tile `k×n`, output tile `m×n` (they: 8/32/96 KB).
3. Detector: subset match of every row against the tile (they: TCAM with 1s→X, 1 cycle/row) **and** popcount.
4. Pruner: one prefix, largest subset, tie → largest index; XOR remainder.
5. Dispatcher: stable sort by popcount; issue prefix-before-suffix; store only one prefix per row; **no** suffix matrix.
6. Processor: load prefix **output row**, accumulate only remainder weight rows, write Y; bit-scan skip zeros.
7. Double-buffer so detection of tile n overlaps compute of tile n−1; prove the detect phase is hidden **on this schedule**, not by slogan.
8. After Y: LIF (or THIS neuron) to produce the **next binary** spike map. If you cannot produce binary M, you have not copied A.
9. Keep numeric equality on Y (they are lossless on GeMM). On THIS net that means **integer 0-diff** on gates / I24 / PED q24, not “approx sparse.”
10. Report **same-port, same-state, same-backpressure** service of the **real** consumers (gate **and** PED **and** BN), not `ΔS × m × k × n` add counts, not CAM-vs-add 45×, not GOP/s from a 128-add array.

If step 1 is false (continuous θg), the transfer is already incomplete. You may still copy the **control structure** (prefix table, issue order, remainder bitmap) onto some **other** binary mask (e.g. gate support), but that is a **partial** copy and must be named as such; the psum-reuse theorem does not carry.

## 6. Candidate X that is NOT a rename of product sparsity

Illegal X (reskins): “reuse common binary sub-combinations of spike rows”; “XOR remainder then add”; “CAM to find subset prefixes”; “exact-match row copy”; “tile-local combination reuse.” That **is** ProSparsity.

A non-reskin X has to change the **reused object** and the **legality condition**:

- **Dual-completion certificate, not prefix psum.** A source slot may be dead for the gate (θg already decided) while **still live for PED / projection / residual add**. The token you store is `{gate_done, ped_done, bn_stats_ready}`, not `{prefix_row_Y}`. Prosperity has one consumer and one Y.
- **Noncausal T10 DAG occupancy under 1RW.** Reuse of **compiled T10 intermediate nodes** already exists in the freeze (260→159). X would be a **port-correct issue order** that reduces the **8088 backpressure**, not a further subset-psum of binary rows. If it does not move backpressure, it is dead even if it is pretty.
- **BN-domain live range.** Full-map batch stats are a barrier Prosperity never had. X is a storage/wait contract for running mean/var over `10×96×120×160`, not combination sparsity.

If the only hardware story is “binary-mask prefix of the gate GeMM,” it is still ProSparsity applied to a mask, and the PED/BN holes remain.

## 7. Strongest controls

Must beat these **on the same ports / state / backpressure**, not in a 128-add CAM accelerator:

1. **Bit-sparsity only** (their own ablation: unstructured skip-zeros, Fig. 9 “Prosperity 5.97”). If prefix reuse adds CAM/table/prefix-read and does not beat skip-zeros on **this** SIMD source, A is not worth copying.
2. **Official CSE T10 graph** already in the freeze (260 add/sub). Product-sparsity-of-rows is the wrong graph.
3. **Learnable 40-coeff lifting T10** (AEE 1.232979368, slots −22.83%, backpressure **unchanged 8088**). This is the strongest “reuse linear combinations, lose on ports” control. Any ProSparsity-like reuse that only cuts adds is predicted to die the same way.
4. **One-prefix vs two-prefix vs intersection materialization** — they already measured two-prefix as not worth it; intersection they refused. If THIS net’s “X” is intersection of supports, that is **ExSpike APEC**, not Prosperity, and must be controlled against APEC not against ProSparsity.
5. **Gate-only skip, PED still dense.** If you copy A onto the binary gate mask, the control is “same source production for PED.” Dual-consumer accounting cannot be omitted.
6. **Full-domain BN vs local-window free μ/σ.** Freeze: local replay undercharges.
7. **Binary-ATLIF ablation** is a **forbidden identity change**, but as a *control* it shows whether any reported gain required binarizing θg. If yes, the paper identity is dead.
8. Do **not** add the SIMD-slot table to the integer-consumer table (−22.83% and −5.78% are different resource points).

## 8. Kill gate

Copy A completely (checklist §5) onto the frozen student **without** switching ATLIF to binary.

**Kill if any:**

- valid825 AEE > **1.259** absolute, or **> ordinary 1.219801338 + 0.005** relative (lifting already at +0.013178, relative fail);
- integer gates / I24 / PED q24 not **0-diff** vs model;
- **long backpressure stays ~8088** (or same-port/same-state net service of the dual-consumer chain does not move) even if add counts or CAM-model ΔS look good;
- always-ready slot cut is **only** generic round→sat fusion (freeze: part of −22.83% already is);
- the method requires binary spikes to make prefix psams well-defined.

Session SCOPE extra: complete-chain same-resource net service **≥15%** after those controls. Lifting-style −5.78% on a different integer point does not pass.

## 9. Quotes relied on (section-located)

- Abstract: “a novel sparsity paradigm called Product Sparsity, which leverages combinatorial similarities within matrix multiplication operations to reuse the inner product result and reduce redundant computations.” SpikeBERT “density of only 1.23% and reduces computation by 11×, compared to bit sparsity, which has a density of 13.19%.”
- Abstract: vs PTB and A100, “average speedup of 7.4× and 1.8× … energy efficiency improvements of 8.0× and 193×.”
- §I: “SNN’s neurons only react to information encoded as binary spikes (1 for spike and 0 for non-spike).”
- §I: spatial ID complexity “O(m^n)”; temporal: if Row 0 processed first, cannot reuse Row 3.
- §II-A: unroll/concat time steps; “computation becomes a sparse addition”; “>98% … spiking GeMM.”
- §II-B: CNN via im2col; transformer linear layers are spiking GeMM `(T×L, di)×(di, do)`; “operations in spiking attention are not efficiently supported by existing SNN ASICs.”
- §III-A: “the common binary sub-combination will generate the same inner product result”; “algorithm-agnostic and lossless.”
- §III-B: Si set definition; PM / EM / Intersection; intersection refused because it “requires creating a new row A.”
- §III-D: one prefix; “30% performance drop”; Table II second-prefix `<6%`.
- §IV: layer-by-layer; Detector / Pruner / Dispatcher / Processor; SFU for exp/mul in softmax or LN.
- §V-A: tile `m×n×k`; m=1 invalidates ProSparsity; n “has no impact on ProSparsity.”
- §V-B: TCAM, 1s masked to X, “all subset indices … single clock cycle.”
- §V-E: row-wise; “Prefix row in the output matrix is fetched and serves as a starting point of the partial sum.”
- §VI-B: “ProSparsity processing phase of a tile is perfectly overlapped by the computation phase of the previous tile.”
- §VII-A: 8-bit weights; DC 28 nm; CACTI; DRAMSim3; cycle-accurate simulator; iso-accuracy because lossless.
- §VII-B: chosen tile **m=256, k=16**.
- §VII-D: EM 100% sparse still 1 cycle.
- §VII-G: ΔS threshold 4.4%; average ΔS 13.35%; ratio 3.0×; TCAM `m²×k`.
- Table III: 128 PEs 8-bit add; 32 LIF cells; 8/32/96 KB buffers.
- Table I vs PTB/Stellar: unstructured ProSparsity vs structured bit sparsity / specific neuron.

---

# Paper 2 — ExSpike (FPL 2026 preprint), arXiv:2606.20414

## 1. Exact mechanism (equations, dataflow, what is reused)

**Goal.** “Full-event” SNN inference: **every** main conv/FC is triggered by **valid binary spike events**, not by dense TConv over `Ho×Wo`. FPGA, DSP-free, LIF `τ=0.5`, 8-bit weights, 16-bit membrane.

**Cost model, Fig. 1.** Traditional conv vs event conv:

```
MAC_traditional = Ho · Wo · Co · Ci · k²
MAC_event       = α · Ci · Hi · Wi · Co · k²
MAC_t / MAC_e   = (Ho · Wo) / (α · Hi · Wi)
```

`α` = active ratio. One input spike at a spatial location updates a **fixed k×k neighborhood across all Co**. “Every operation directly contributes to valid updates.” Fig. 2 VGG11/CIFAR-10 direct-coded: EConv vs TConv layer latency −50% to −97%, average −88%, correlated with input sparsity.

**Three dataflow optimizations so intermediates stay spike-based (Algorithm 1).**

**OPT1 Direct coding (first layer is multi-bit, not spikes).** Quantize input to signed fixed-point, **bit-slice**; **duplicate-and-shift weights** offline to match bitwise multiplication. Execution is **shift-and-accumulate**, no runtime multiplier. Preprocessed shifted weights live in Weight SRAM.

**OPT2 Event-driven convolution.** Not “one PE = one output neuron time-multiplexed” (imbalance under irregular sparsity). Partition PE array into blocks; each block owns a kernel target region. For each spatial `(h,w)`, collect valid events across Ci; `psum ← W.Cal(e)`; `MP += psum`. Channel-parallel: 32 EPE clusters update 32 output channels of the same region. If Co > P, group `G = ⌈Co/P⌉` and reuse. After a spatial row `h`, `MP.CheckAndFire()` emits output spikes.

**OPT3 Event-driven average-pool + FC (EAFC).** Average-pool would emit **non-binary** values. They **do not** run W2TTFS counters (NEURAL). Instead: **scale FC weights offline** by `1/pooling_size²`, then event-accumulate those scaled weights into `y_fc`. Fused, still event-triggered, still no divide at runtime.

**Architecture, Fig. 3–4.**

- **Sparse Core:** Spike SRAM (64 KB) + Residual Spike SRAM (64 KB) for shortcut **spike** maps. Fast event filter: one valid event position **per cycle** via lowest-set-bit one-hot + LUT. Addresses into **AER FIFO**. Non-empty FIFO triggers EPE.
- **EPE Core (OPT2):** 32 clusters. Each: 3×3 WPE block (weight accumulate) → elastic FIFO → MPE (membrane add) → FPE (bias + threshold compare → spikes). 32 Co in parallel.
- **Attention Core (SDSA):** two stages. (1) While writing V spikes, read corresponding K; element-wise **AND** → KV mask; column-wise **OR** → KV status vector in **registers** (`Co` bits), no big intermediate tensor. (2) Row-wise **AND** of Q spikes with that shared KV status. Binary in, binary out.
- **EAFC Core:** event-triggered FC with offline-scaled weights.
- Instruction SRAM (~1 KB) programs kernel/channel; layer-by-layer.

**APEC — Adjacent-Position Event Compression (§III-A2, Fig. 5).** This is the extra reuse **beyond** zero-skipping.

- Group `g` **spatially adjacent** positions. For each group, bit-wise **AND** of their spike sequences (across channels) = **overlap** `OG`. Non-overlap sequences are disjoint from `OG`.
- Compute overlap **once**, cache its partial sums; each position then adds only its **unique** events.
- Example Fig. 5: 14 events → 8 (−6). For 3×3, Co=64: `6 × 64 × 9 = 3456` accumulations saved.

Equations:

```
OG = ∩_{i=1..g} S_i                         (1)  S_i = active-channel set of position i
ΔN_event^(g) = (g−1) |OG|                   (2)
ΔC^(g) = (g−1) |OG| Co k²                   (3)
M_ov ≈ Co k² w_acc                          (4)  extra storage for cached overlap psums
```

**What is reused:** the **convolution contribution Φ(c) of channels in the intersection of adjacent positions’ supports**, i.e. a **shared partial sum over a common binary channel-set**. Numerically equivalent to executing every event (“APEC only reorganizes the execution order of overlapped and non-overlapped events”).

Group-size tradeoff: larger `g` raises the `(g−1)` reuse factor but **higher-order |OG| collapses**. Measured on SpikingFormer-4-256 CIFAR-10: average |OG| **19.08 (G2) → 6.82 (G4) → 2.92 (G8)**. G2 wins every benchmark. Throughput vs no-APEC: +10.9% VGG11, +13.8% ResNet18, +14.5% SF-4-256, +11.2% SF-2-512; event reduction 1.62× / 1.35× / 1.36× / 1.38×.

**Not always compute-bound.** Fig. 8 SpikingFormer-2-512: APEC-2 cuts **calculation** cycles in Enc0.SSA/FFN but **Weight-ready cycles increase** and can wipe the win. “APEC-G2 is beneficial when the saved calculation cycles exceed the additional weight/buffer cycles; otherwise … more suitable for computation-bound layers with strong adjacent-position spike overlap.”

**Evaluation.** Verilog → Synplify → Vivado, AMD Virtex-7 XC7V2000T, **200 MHz**. Power from post-synthesis netlist + **benchmark-specific SAIF**. Table I: Baseline vs APEC-2, EPE LUT 19k→25k, FF 21k→26k, power 1.593→1.700 W (1.07×), throughput on SF-4-256 1.14× ⇒ energy-efficiency ~1.07×. Peak **479.15 GOPS, 281.85 GOPS/W, 0.80 GOPS/W/PE** (~10× FireFly-T on GOPS/W/PE). Acc: VGG11 93.89%, ResNet18 94.98%, SF-4-256 94.45% (match FireFly-T), SF-2-512 **75.81%** (below FireFly-T 78.38% / SpikeTA 78.40%), SegNet 98.70%. DSP=0. vs Xeon 8470Q: 30× latency, 7046× energy; vs RTX PRO 6000 batch-32: **higher latency**, 33.6× energy. Workloads: CIFAR classification + MLND_Capstone segmentation. **Not** DSEC optical flow.

## 2. Identity assumptions (binary spikes? product of binary? 1RW?)

| Assumption | In ExSpike? |
|---|---|
| Hidden layers are **binary spikes** | **Required** for “pure event-driven.” OPT1 is the exception at the **first** layer, and even there bits are 0/1 slices selecting shifted weights. |
| Event contribution **Φ(c)** depends only on **which channels fired**, not on a continuous amplitude | **Required** for APEC: `S_i ⊆ {1,…,Ci}`, overlap is set intersection, reuse is identical conv contribution. |
| Product of binary event × weight = **accumulate W**, no multiplier | **Yes.** “without using multipliers.” PE = accumulators + threshold. |
| Residual / shortcut is a **spike map** | Residual Spike SRAM holds **source spike feature maps**. Not a continuous PED tensor. |
| Attention is **binary SDSA** (AND/OR of Q,K,V spikes) | Yes. No softmax in the core path they implement. |
| Neuron = **LIF**, fire by threshold on MP | Yes. Causal MP update then CheckAndFire. |
| **1RW** | **No.** Spike SRAM + Residual SRAM + Weight SRAM + AER FIFO + eFIFO + 32 parallel clusters + overlap-psum buffers for APEC. Fast filter is combinational one-hot, not 1-port. |
| APEC lossless w.r.t. full-event execution | **Claimed.** Reorder only. |
| Spatial adjacency ⇒ correlated binary supports | Empirical; G2 is the useful point. |

## 3. What they explicitly do NOT do

- **Do not leave the first layer on the host.** They criticize prior designs that encode off-chip; OPT1 pulls it on-chip as bit-slice + shifted weights.
- **Do not use NEURAL’s W2TTFS counters** for pooling; they fold the scale into FC weights.
- **Do not assign one PE per output neuron** for conv (workload imbalance).
- **Do not use DSPs** (contrast FireFly-T, SpikeTA, DeepFire2). Table III: DSP-free ✓.
- **Do not claim APEC gain is monotonic in g.** They show the opposite; G2 default.
- **Do not claim event-count reduction ⇒ latency.** Fig. 8: weight/buffer cycles can dominate.
- **Do not run analog CIM.**
- **Do not evaluate optical-flow AEE / DSEC.** Gen1 mAP is cited as *other people’s* detector result in the intro, not an ExSpike number.
- **Do not handle continuous θg, noncausal T10 PSN, or full-domain BN.** Average-pool is the non-binary intermediate they **eliminate** by folding, not a BN over `10×H×W`.
- **Do not implement Prosperity-style row-subset search over a GeMM tile.** APEC is **local adjacent groups**, typically g=2, AND-overlap (intersection), not PM/EM forest + TCAM.
- FPGA power is Vivado+SAIF, not foundry PPA. Do not treat 0.80 GOPS/W/PE as an ASIC number.

## 4. Hole vs THIS net

1. **Event token is a binary spike.** THIS source is consumed as **continuous θg / I24 / PED q24**. EConv’s “each valid event updates k²×Co” is the right *shape* for a binary conv, wrong *algebra* for ATLIF θg and for PED. You cannot legally skip a “zero spike” if the residual path still needs the continuous activation.
2. **APEC overlap is set-intersection of binary channel-sets.** Prosperity refused this intersection for GeMM rows because it materializes a new combination; ExSpike does it **only** because adjacent positions share a **binary** AND and Φ(c) is identical. With continuous θg, adjacent positions with the same support still have **different amplitudes**; AND-then-reuse-psum is false unless you also share the amplitude (they don’t).
3. **Dual consumers.** Residual SRAM is a **spike** shortcut, not PED. APEC caches one overlap psum for the **same** MP update. HERE an overlap legal for the gate may be illegal for PED (or vice versa).
4. **Noncausal T10.** CheckAndFire is after accumulating events into LIF MP (spatial loops in Algorithm 1). T10 PSN is a **dense noncausal mix of 10 frames**, already CSE’d. Event-by-event MP add is not that graph. Concatenating DVS events over long T (they train LIF in PyTorch/SpikingJelly) is not T10 PSN.
5. **Full-domain BN.** No analog of waiting for batch stats on `10×96×120×160`. Folding `1/16` into FC weights is a **static** scale, not dynamic BN.
6. **Port / backpressure.** APEC **increases** weight-ready cycles when it reorders events (Fig. 8). THAT is the same family of failure as lifting: arithmetic ↓, **memory/issue occupancy not ↓**. Freeze already saw backpressure 8088 unchanged.
7. **Task mismatch.** CIFAR/SegNet FPS and GOPS/W/PE do not transfer to DSEC valid825 AEE. Their SF-2-512 accuracy is already **below** FireFly-T; not a quality story to import.

## 5. Honest complete-transfer checklist to copy A

To copy ExSpike, not cite it:

1. Represent every hidden activation as a **binary spike map** stored as “all Ci bits at one spatial address” (their Spike SRAM layout ❶).
2. Fast event filter: one event/cycle, AER FIFO, compute **only** on valid events (OPT2).
3. PE = weight accumulate + MP + threshold fire; **no** multiplier in the hidden path.
4. OPT1 if the first layer is multi-bit: offline duplicate-shift weights, bit-slice activations, shift-and-add.
5. OPT3 if there is average-pool before FC: **offline** `W_fc /= pool²`, then event-accumulate. Do **not** emit pooled real values.
6. Shortcut = **spike** Residual SRAM, not a second continuous tensor.
7. If SDSA exists: AND/OR KV status in registers, AND with Q; no softmax core.
8. APEC: adjacent groups (start at **g=2**), AND overlap, cache overlap psum (`M_ov ≈ Co k² w_acc`), unique remainder, **prove numeric equality** with uncompressed events.
9. Sweep g; **do not** assume larger groups are better. Measure **weight/buffer cycles**, not only event counts.
10. On THIS net, additionally: keep dual-consumer 0-diff (gates **and** PED), full-domain BN wait, noncausal T10 function, same-port backpressure. If you binarize to satisfy (1), you have **changed paper identity**.

## 6. Candidate X that is NOT a rename of product sparsity

Illegal X: “compress adjacent positions by AND-overlap and reuse the common accumulations.” That **is** APEC, and it is the **intersection** case of product sparsity on channel-sets. Calling it “spatial ProSparsity” is still a rename.

Non-reskin X:

- **Two-token event legality.** A spatial site produces a **gate-event** and a **PED-demand** with different skip/compress rules. Overlap is legal only when **both** algebras agree. APEC has one Φ(c).
- **Compress BN-wait or T10-DAG occupancy across adjacent pixels**, where what is shared is a **statistic or compiled node**, not a binary channel-set psum. Must still pay full-domain BN; local windows are forbidden as a free lunch.
- **Port-aware event issue** whose figure of merit is the freeze’s **8088 backpressure / always-ready slots**, matching Fig. 8’s lesson (calculation ↓ can be offset by weight-ready ↑). If X only reduces `ΔC = (g−1)|OG|Co k²`, it is APEC’s own metric.

## 7. Strongest controls

1. **EConv without APEC** (their Baseline in Fig. 7 / Table I). APEC is only +11–15% throughput on their FPGA, +1.07× power; easy to lose on a 1RW source.
2. **TConv** (Fig. 2). If THIS net’s projection/r1 is already a dense conv with a continuous residual, TConv is the honest dense control, not EConv.
3. **G2 vs G4 vs G8** (they: G2 wins because |OG| collapses).
4. **Compute-bound vs weight-bound layers** (Fig. 8). Force the control on r1 + T10 + PED, which are memory/live-range heavy, not on a 3×3 EPE cluster.
5. **Spike residual vs continuous PED.** Copying Residual Spike SRAM without PED is an identity break.
6. **Offline weight-scale vs actual full-map BN.** Folding `1/N` is not BN.
7. **Prosperity one-prefix PM/EM** as a control if someone “invents” adjacent subset reuse: APEC is intersection, ProSparsity is subset/equal; they are different, both still binary-product.
8. Lifting T10 on THIS net (ops ↓, backpressure flat). Predicts APEC-style reorder.

## 8. Kill gate

Complete A (event filter + EConv + APEC G2 + residual spike SRAM) on the frozen student **without** binarizing ATLIF.

**Kill if:**

- AEE fails 1.259 / +0.005 vs 1.219801338;
- integer dual-consumer 0-diff fails;
- APEC-style overlap is only well-defined after binarizing θg;
- **weight/buffer/backpressure** does not fall (Fig. 8 pattern, or freeze 8088);
- reported win is GOPS/W/PE or FPGA FPS, or component speedups multiplied into FPS (forbidden);
- SegNet/CIFAR accuracy used as a substitute for valid825.

APEC’s own best case is ~1.1–1.15× throughput at 1.07× power on a DSP-free FPGA. Even a clean copy is below the SCOPE **15% same-resource net service** unless the dual-consumer chain here is far more overlap-rich **and** port-bound in the opposite direction of Fig. 8. Measure, don’t assume.

## 9. Quotes relied on (section-located)

- Abstract: “full-event neuromorphic architecture that fully exploits irregular sparsity”; “adjacent-position event compression to reduce redundant accumulations across spatially adjacent spike sequences”; “up to 10× higher PE-normalized energy efficiency than … FireFly-T.”
- §I: TConv vs EConv; `MAC_event = α·Ci·Hi·Wi·Co·k²`; “full-event execution refers to … convolution and fully connected computation, are triggered by valid spike events.”
- §I challenges: direct coding multi-bit MAC; average pooling “non-binary intermediate results”; PE-per-neuron imbalance.
- §II: LIF; OPT1 bit-slice + DuplicateShift; OPT2 event conv + CheckAndFire; OPT3 `W_fc ← W_fc / pooling_size²`.
- §II end: “executed in a pure event-driven manner without using multipliers.”
- §III: Residual Spike SRAM “source spike feature maps used by shortcut or residual connections”; 32 EPE clusters; Attention Core AND/OR SDSA.
- §III-A2: OG intersection (1); `ΔN=(g−1)|OG|`; `ΔC=(g−1)|OG|Co k²`; “preserves numerical equivalence.”
- §III-A2: overlap psum storage `M_ov ≈ Co k² w_acc`; “overall gain of APEC does not necessarily increase monotonically with group size.”
- §IV-A: G2 best; |OG| 19.08→2.92; Fig. 8 weight cycles can offset calculation savings.
- Table II / §IV-B: DSP-free; SF-2-512 acc 75.81% vs FireFly-T 78.38%; 0.80 GOPS/W/PE.
- Table III: only ExSpike ticks Event conv + Full-event + Event compression among the listed FPGA designs.
- Fig. 9: GPU comparison uses batch 32; ExSpike latency **worse** than GPU, energy better.

---

# Paper 3 — SpikeX (arXiv:2505.12292, 18 May 2025)

Title confirmed from the file: **“SpikeX: Exploring Accelerator Architecture and Network-Hardware Co-Optimization for Sparse Spiking Neural Networks”** (Xu, Boone, Li). Not a mislabeled PDF.

## 1. Exact mechanism (equations, dataflow, what is reused)

**Neuron / op they implement.** Discretized **LIF**, binary in/out, multi-bit weights. Per timestep:

```
C[t] = Σ x[t] × W                         (1)   x binary ⇒ conditional accumulate, not MAC
u[t] = H(Vth − (λ u[t−1] + C[t])) · (λ u[t−1] + C[t])    (2)
S[t] = H(Vth − (λ u[t−1] + C[t]))         (3)
```

“Instead of requiring a full multiply and accumulate … the summation can be done by a simple conditional accumulate.”

**Architecture, Fig. 1.** Systolic array, **default 8×8**. PE: one **reusable accumulator**, comparator, scratchpad Vmem, registers, PE controller. Three-level memory: DRAM / GLB **54 KB** / L1 LBUF **2 KB**, all **double-buffered**. DRAM bandwidth modeled **30 GB/s**. Tech for energy: **CACTI 32 nm**. Simulator is **analytic / cycle-level traces**, “adapting techniques similar to [PTB]”; **not** a synthesized 28 nm core like Prosperity, **not** an FPGA like ExSpike.

Per PE, a postsynaptic neuron over a **time window (TW)** of `TWS` time points, three steps (§III-A):

1. Integrate activations × weights across input channels **and all time points in the TW** (weights from left PE / filter buffer). **Same W reused for every t in the TW.**
2. **Sequentially** update `Vmem[t]` from `Vmem[t−1]` (causal leak). Cannot parallelize this step across t inside the PE.
3. Compare to `Vth`, emit binary spike per t.

`TWS` is **reconfigurable per layer** (the HAS knob).

**Four temporal granularities, Fig. 2.** Time point ⊂ TW ⊂ Time Block (TB, sized to a cache level) ⊂ Time Stride (whole inference horizon). **Activity tag** at each grain: 0 iff no presynaptic firing in that unit. Hierarchical tags = bitwise OR of children. Zero tag ⇒ **do not load, do not process**.

**NTWU.** `NTWU(n, tw)` = work to run LIF for postsynaptic neuron `n` over time window `tw`. Spatial grain = one neuron; temporal grain = one TW. This packing is why they can skip at a **coarser** grain than per-spike: spikes “cluster” in time, so most TWs are empty (Fig. 5).

**Agile SpatioTemporal Dispatch — three levels of weight reuse (Fig. 1c, Fig. 3).**

1. **Inside a TW, inside a PE:** one weight load, integrate all t in the window (PTB’s idea; they say PTB “does not fully tap” sharing and dies on PE utilization when temporal sparsity is high).
2. **High temporal NTWU density mode (default):** NTWUs of the **same postsynaptic neuron** (different TWs) sit on the **same systolic row**; weights enter from the left and are **shared across columns**. Used when active NTWUs per neuron ≳ array width. Fills columns ⇒ utilization.
3. **High spatial NTWU density mode:** map NTWUs of **different neurons that share the same filter** onto the same row (conv weight reuse across postsynaptic neurons). “Particularly effective for convolutional layers.”

Vs PTB baseline: PTB is time-domain packing **without** weight reuse across postsynaptic neurons, and uses one globally searched TWS. SpikeX claims 1.3×–7.11× latency over PTB on CONV2 DVS-Gesture Medium as firing rate goes 5% → 0.01% (Fig. 10).

**Activation-induced Weight Tailoring (§III-D, Fig. 4).** Separate storage: 1-bit spatiotemporal activations vs multi-bit static weights. Activations packed in SP-MBs (postsynaptic set × time blocks ×, for conv, **one input channel**). SP-MB tag = OR of TB tags. When moving down the memory hierarchy, **only fetch weight kernels whose input channel has a nonzero tag.** “No redundant data movements.” This is **zero-activation skip of weight traffic**, not product reuse of computed psams.

**Network/hardware co-optimization (§IV).**

Hardware-aware training SpikeX-HT:

```
L_tot = L_acc + β L_HW                    (4)
L_HW(W) = EDP( S_p(W) )
```

`S_p` is **time-window sparsity** (fraction of TWs with ≥1 spike), **not** cycle-accurate EDP (too slow to put in the training loop). Piecewise-linear fit of EDP per filled TW from the simulator. Insert L_HW at each layer output. Two-stage train: pretrain with warm-up (no HW loss) → drop warm-up, add HW loss. β sweep `5e−5 … 1e−1`. Fig. 5: TW sparsity +28.4% / +29.9% / +11.3% on DVSG-medium / DVSG-large / NMNIST; “33.2× / 26.6× / 11.2× less high-density neurons.” Accuracy in Fig. 7 can **rise** 0.35–2.08% or drop ~0.69% depending on β. **This changes the network** (not lossless).

SpikeX-HAS: two-level opt over discrete architectural knobs `α` (chiefly **per-layer TWS**), continuous relaxation `α̂`, hypernet of L_HW as mixture of mappings, pick argmax selection at the end.

```
min_α L_tot(α, W*(α), S_p(α, W*(α)))      (5)
s.t. W*(α) = argmin_W L_tot(...)          (6)
```

**Evaluation.** Datasets: **DVS-Gesture** (11 gestures, event camera, T=300) two CNNs; **N-MNIST** T=30. **Classification, not optical flow.** Baseline = PTB [26] with **exhaustively searched** best global TWS, same 8×8 and memory hierarchy. Combined SpikeX+HT+HAS vs that PTB: energy **24.38× / 4.91× / 2.07×**, latency **10.29× / 5.19× / 7.25×** on the three nets; EDP **15.1×–150.87×**. HT alone already 11.13× / 4.89× / 1.21× energy. CONV1 **cannot** be sparsified by HT (“predefined inputs in the validation set”). Memory is 62% of FC energy at TWS=4; weight access 40.1%. Larger TWS trades weight energy down vs activation energy up (Fig. 11–12). Synthetic firing-rate sweep Fig. 9: energy −46.6% to −68.2% as rate 0.5%→0.01%, latency ~−90%, **optimal TWS flips with rate** (hence HAS).

## 2. Identity assumptions (binary spikes? product of binary? 1RW?)

| Assumption | In SpikeX? |
|---|---|
| x and S are **binary**; W multi-bit | **Required.** “activations are one bit (binary) while the bit-width may vary from 8 to 32 bits.” |
| `x[t]×W` is **conditional add** | **Yes.** PE has no multiplier. |
| **Causal** LIF: `u[t]` needs `u[t−1]` | **Required.** Step ❷ is sequential in t. Time-parallelism is only on **integration** inside a TW, not on the leak chain. |
| Unstructured binary firing, but **exploited at TW grain** (OR-tag), not at combination-of-rows grain | Yes. “Coarse-to-grain granularity” in Table I. |
| One activity tag per work unit (no dual consumers) | Yes. Tag is a single bit: any spike in the unit. |
| Weight reuse is the expensive thing to maximize | Stated as the data-disparity thesis. |
| Co-design **may change** sparsity and TWS; accuracy is a tradeoff via β | Yes. Opposite of Prosperity’s lossless claim. |
| **1RW** | **No.** Systolic edges, 3-level double-buffered caches, per-PE Vmem scratchpad, 8×8 PE array. |
| Simulator + CACTI 32 nm = energy | No RTL/FPGA in the evaluation methodology section. |

## 3. What they explicitly do NOT do

- **Do not** reuse **computed inner products of shared binary sub-combinations** (Prosperity). They reuse **weights** and **skip tagged-empty NTWUs**.
- **Do not** AND-compress adjacent spatial event sets (ExSpike APEC).
- **Do not** process time-serial one-t-at-a-time if TWS>1; also **do not** fully parallelize Vmem across t (causal leak).
- **Do not** keep PTB’s poor utilization under high temporal sparsity; that is the hole they claim vs PTB.
- **Do not** put real EDP in the inner training loop; proxy is TW sparsity.
- **Do not** sparsify CONV1 / dataset-defined inputs by HT.
- **Do not** evaluate spiking transformers, optical flow, DSEC, or full-domain BN.
- **Do not** claim analog CIM (they even cite ADC-less IMC [5] as *other* work).
- **Do not** publish a foundry PPA; energy is CACTI × trace counts + energy-per-add.
- **Do not** handle a second continuous residual reader of the same source. Shortcuts in their CNNs are not described as dual-algebra PED.
- Table I: TrueNorth/Loihi “no” parallel processing as they define it; SpinalFlow “low” applicability / spatial only; PTB temporal + limited arch opt; SpikeX = agile spatial+temporal, coarse-to-grain sparsity, **yes** HW-aware train, **yes** arch search.

## 4. Hole vs THIS net

1. **Binary LIF vs continuous θg ATLIF.** Conditional accumulate and a 1-bit activity tag both die if the activation is θg. A “silent” site for a binary spike can still be a **nonzero continuous residual**. Weight tailoring keyed on `OR(spikes in TB)` will **drop weights the PED still needs** or, if you OR in residual-liveness, you **tailor nothing**.
2. **Causal TW integration vs noncausal T10 PSN.** SpikeX’s win is: pack TWS causal timesteps, share W, then walk Vmem in order. THIS PSN is a **noncausal 10-point linear mix** (ordinary 260 add/sub, lifting 159). There is no “empty TW” inside a full-rank T10: every output depends on the whole vector. Hierarchical OR-tags over time windows **do not apply** to a dense T10 matrix.
3. **Dual consumers vs one NTWU tag.** SpikeX retires work when the tag is 0. HERE gate and PED have different completion times. A high-spatial-density dispatch that shares W across postsynaptic neurons does not retire the **source** until **both** consumers and **BN** are done.
4. **Full-domain BN.** No analog. Their SP-MB is a cache tile of activations, not a reduction over `10×96×120×160` that **blocks** all consumers.
5. **HW loss proxy is TW sparsity.** Freeze: “old activity-weighted-dot ledger … Proxy ≠ new student's cycle share.” Training against TW sparsity / EDP(S_p) is the **same class of wrong proxy**. Lifting already cut adds and **not** backpressure. HAS searching TWS does not search 1RW issue order of a dual-consumer T10.
6. **Task / scale.** DVS-Gesture **classification** T=300 and N-MNIST T=30, tiny CNNs (Table II: e.g. DVSG-medium last conv 8×10, 16 ch). THIS is DSEC 2D flow, 10×96×120×160 BN domain, patch r1 + T10. Their 8×8 / 54 KB story does not contain this map.
7. **EDP 15.1×–150.87×** is vs PTB **in their simulator**, after **changing the net** (HT). Forbidden to multiply into FPS; forbidden to quote as foundry PPA; cannot be used as a transferable speedup on ep34.

## 5. Honest complete-transfer checklist to copy A

To copy SpikeX:

1. Binary spike I/O, multi-bit W, LIF (1)–(3) with **causal** Vmem.
2. Systolic (or equivalent) PE with accumulate + compare + Vmem scratch; **no** hidden multiplier.
3. Pack work as **NTWU = neuron × TW**; hierarchical activity tags; skip zero tags at every memory level.
4. Implement **both** dispatch modes: temporal-density (share W across TWs of one neuron) and spatial-density (share filter across neurons).
5. Activation-induced weight tailoring: **don’t fetch W for inactive input channels / SP-MBs**.
6. Reconfigurable **per-layer TWS**.
7. If claiming HT: pretrain + `L_acc + β EDP(S_p)` with S_p = **TW** sparsity, two-stage warm-up off; report β vs AEE (not just vs classification acc).
8. If claiming HAS: joint search of TWS with W, not a hand-picked global window.
9. Baseline must be **PTB with best TWS**, plus “SpikeX arch without HT,” plus “HT without HAS” (their Fig. 13 ablations).
10. On THIS net: **also** keep continuous θg, dual consumers, noncausal T10, full-domain BN, integer 0-diff, same-port backpressure. If (1) or (3) requires dropping those, the copy is identity-breaking.

Weight tailoring **alone** (skip W for zero binary inputs) is **not** a full copy; it is the SNN version of zero-input skipping, already assumed by bit-sparsity controls.

## 6. Candidate X that is NOT a rename of product sparsity

SpikeX itself is **already not** product sparsity. The rename risk is different: reskinning **skip zeros + share weights in a time window + train for sparsity**.

Illegal X: “agile dispatch of time windows”; “don’t load weights for silent channels”; “hardware loss = f(sparsity)”; “search TWS per layer.” That is SpikeX.

Non-reskin X (relative to SpikeX **and** to ProSparsity/APEC):

- **Consumer-asymmetric work units.** Split NTWU into `NTWU_gate` and `NTWU_ped` with **different** tags, **different** retirement, **same** source live range until max(done). SpikeX has one tag. ProSparsity has one Y row. APEC has one Φ(c).
- **Search/train against measured backpressure and valid825 AEE**, not TW sparsity and not add counts. That is the freeze’s actual figure of merit. SpikeX-HAS on TWS is the wrong α.
- **Noncausal T10 as the work unit**, with compiled CSE/lifting DAG as the inner op, **causal Vmem packing forbidden**. If X “packs T10 like a TW,” it is a rename of PTB/SpikeX and **changes the PSN**.

## 7. Strongest controls

1. **PTB + exhaustive best TWS** (their own baseline). Any “time-window packing” paper that cannot beat this is not SpikeX.
2. **SpikeX architecture, no HT, no HAS** vs **HT only** vs **HAS only** (Fig. 13).
3. **Per-time-point skip vs TW-OR skip.** They pack TWs because unstructured per-spike skip is hard; on T10 length 10, TW packing has almost no room (TWS would be ≤10 and the mix is dense).
4. **Weight tailoring off** (always fetch all IC kernels).
5. **Fixed global TWS vs per-layer HAS.**
6. **Do not sparsify by HT without reporting AEE** on valid825; classification acc on DVS-Gesture is the wrong metric.
7. **Lifting T10 + CSE** on THIS net (ops ↓, backpressure flat) as the “wrong proxy / wrong object” control.
8. **Dual-consumer: tailor weights that are dead for the gate only.** Control = PED still fetches. If tailoring uses a single OR-tag, it is SpikeX and it is wrong here.
9. CONV1-like **uncensorable input** (they already show HT cannot touch it). Event-camera input to THIS net is analogously not a free sparsity knob.

## 8. Kill gate

Full A (NTWU tags + 3-level weight reuse + tailoring + optional HT/HAS) on Motion C12/H67/ep34 **without** binary ATLIF and **without** replacing noncausal T10 by causal TW-LIF.

**Kill if:**

- AEE > 1.259 or > 1.219801338 + 0.005;
- integer 0-diff on gates/I24/PED fails (tailoring dropped a live residual);
- TW/tag skip is only legal after binarizing or after making T10 causal;
- same-port backpressure / dual-consumer service does not move (SCOPE ≥15% net service);
- EDP or energy-delay from CACTI×traces, or DVS-Gesture latency, is quoted as the student result;
- HT improves TW sparsity but AEE or BN wait gets worse (wrong proxy, freeze already warned).

Their own NMNIST energy gain with full SpikeX is only **2.07×** in Fig. 13 — and that is vs PTB on a tiny binary CNN after changing the net. Do not treat the 150.87× EDP end of the abstract as a transferable number.

## 9. Quotes relied on (section-located)

- Abstract: “unstructured spatial and temporal firing sparsity”; “Agile SpatioTemporal Dispatch and Activation-induced Weight Tailoring”; “co-optimization methodology … hardware-aware SNN training [and] hardware accelerator architecture search”; “15.1×−150.87× in energy-delay-product(EDP) without comprising model accuracy.”
- §I: “Input and output activations of spiking neurons are binary while weight data are multi-bit.”
- §II: equations (1)–(3); “simple conditional accumulate.”
- §II-B Challenges: “sparse firing patterns in an SNN are highly irregular”; CONV3 DVS-Gesture “only 0.0001% … fire > 150 spikes over 300 time points.”
- §II-B Proposed: three-level weight sharing; tailoring “avoiding loading weights associated with zero-valued input activations”; co-opt “without compromising model accuracy.”
- §III-A: PE = accumulator + comparator + scratchpad Vmem; TWS reconfigurable; steps ❶ integrate over TW, ❷ sequential Vmem, ❸ spike.
- §III-B: NTWU definition; hierarchical activity tags; “data or work with a zero-valued activity tag are not loaded or processed.”
- §III-C: high temporal vs high spatial density modes; third-level reuse “between PEs processing different post-synaptic neurons.”
- §III-D: SP-MB tags; “only the weight data corresponding to the input channel of an active SP-MB are loaded.”
- §IV-A: `L_tot = L_acc + β L_HW`, `L_HW = EDP(S_p(W))`, S_p = time-window sparsity; “exact EDP measurements … far too complex to run simultaneously with network training.”
- §IV-B: HAS two-level opt (5)–(6); α includes per-layer TWS.
- §V: 8×8; GLB 54 KB; LBUF 2 KB; DRAM 30 GB/s; CACTI 32 nm; double-buffered; baseline PTB with optimal fixed TWS.
- §VI: CONV1 unchanged under HT; combined energy/latency 24.38×/10.29× (DVSG-medium), 4.91×/5.19× (large), 2.07×/7.25× (NMNIST); EDP 15.1×–150.87× vs [26].
- Table I: sparsity handling “Coarse-to-grain granularity.”

---

## What this means for a TCAS-II 5-pager on THIS net

- **Prosperity A** is prefix-psum reuse of **binary GeMM rows** with CAM + forest scheduling, lossless. It does not survive continuous θg, dual consumers, noncausal T10, full-domain BN, or the already-measured backpressure wall.
- **ExSpike A** is full-event EConv + **g=2 adjacent AND-overlap psum** (intersection product sparsity on channel sets) + spike residuals + binary SDSA, FPGA-measured. Same identity breaks; their own Fig. 8 already shows **reorder can raise weight-ready cycles**.
- **SpikeX A** is TW-tagged skip + **weight** reuse/tailoring + train/search on TW sparsity, simulator+CACTI, **lossy** co-design, causal LIF, classification DVS. Wrong neuron, wrong time algebra, wrong proxy, wrong task.

**Do not** write a paper whose X is any of: product sparsity, APEC, NTWU/TW packing, skip-zero weights, CAM subset match, exact-match row copy. Those are A, and they miss this net.

A live X has to be about **objects this net actually has**: continuous θg, **two** consumers with different algebras, **noncausal T10** compiled graph, **full-domain BN wait**, and **same-port backpressure** (the 8088 number). Kill it with AEE 1.259 / +0.005, integer 0-diff, and a same-resource service number that is not an add-count.
