# R2L1 — Prosperity after AT-LIF absorb (not the round-1 L1)

Date: 2026-09-11. Literature remap only. Freeze: `round2_absorb/PROBLEM.md` + `IDENTITY_ATLIF.md`. Full local texts: `p0_txts/2503.03379.txt` (Prosperity, HPCA 2025), `p0_txts/2606.20414.txt` (ExSpike, FPL 2026 preprint). No invented citations.

**This note supersedes round-1 `literature/L1_prosperity_exspike.md` on the neuron identity.** L1 treated AT-LIF as a continuous θg that broke ProSparsity’s reuse theorem. The locked identity is the opposite: \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\), layer-shared \(\theta\) is absorbed into the next \(W\) at inference, and the **spike path is binary \(\{0,1\}\times W\)**. Prosperity / Gustav / FireFly become **legal complete priors A to copy on that path**. They do not become the letter. The letter, if any, lives on objects those papers never had.

| Local file | Printed title | Venue in file |
|---|---|---|
| `.../p0_txts/2503.03379.txt` | Prosperity: Accelerating Spiking Neural Networks via Product Sparsity | 2025 IEEE International Symposium on High-Performance Computer Architecture (HPCA) |
| `.../p0_txts/2606.20414.txt` | ExSpike: A General Full-Event Neuromorphic Architecture for Exploiting Irregular Sparsity with Event Compression | PREPRINT – To appear at FPL 2026 |

---

## 0. Identity lock (what changed vs L1)

Official AT-LIF (this freeze, not a novelty claim):

\[
o_i[t]=\theta_i\cdot H(m_i[t]-\theta_i)=\theta_i s_i[t],\quad s\in\{0,1\},\quad o\in\{0,\theta\}.
\]

At inference, layer-shared \(\theta\) is folded \(W\leftarrow\theta W\). After that fold, **inter-layer tokens on the spike path are 0/1 pulses**. They are not analog payloads and not non-absorbable int8.

Two objects stay distinct after the fold:

1. **Spike GeMM after absorb** — binary select-add. This **is** Prosperity’s Spiking GeMM.
2. **Residual / PED / I24** — a different continuous tensor. Dual consumers: **A = binary gate after absorb**, **B = continuous residual path**. Do not write “continuous AT-LIF amplitude is shared by two MACs.”

**Before** the AT-LIF threshold, this net has a noncausal T10 mix (PSN / lifting CSE). That mix is continuous arithmetic, **then** threshold, **then** absorb.

Round-1 L1’s kill “prefix psums are ill-defined because θg is continuous” is **retired**. The remaining kills are the objects listed in §3.

---

## 1. What Prosperity’s EM/PM forest REQUIRES

The reuse theorem is not “SNN sparsity.” It is a **three-part object**. If any part is missing, you do not have ProSparsity; you have a slogan.

### 1.1 Algebra (the theorem)

Workload they name **Spiking GeMM** (§II-A): unroll every time step as extra **rows** of one binary matrix that **share the same weight matrix**.

\[
Y[i,:] \;=\; \sum_{j:\,M[i,j]=1} W[j,:],\qquad M\in\{0,1\}^{m\times k},\; W\in\mathbb{R}^{k\times n}.
\]

- Activations are **binary {0,1}**. A “1” selects a weight row; a “0” skips it. Inner product is accumulation, not MAC. PEs are 8-bit **adders** (Table III/IV).
- The product of a row depends **only on the set of firing columns**, not on a per-row amplitude. After absorb, this is exactly \(s\times(\theta W)\): \(\theta\) has already been baked into \(W\), so two spatial/time sites with the same support share the same product.
- All rows of a tile share **one** \(W\). That is why a prefix product is a legal starting psum for a suffix.

Spike row as a set (§III-B):

\[
S_i \;=\; \{j \mid M[i,j]=1\}.
\]

For two rows with \(A=S_i\cap S_j\neq\emptyset\), three cases:

| Case | Set condition | What is reused | Prosperity |
|---|---|---|---|
| **EM** (exact match) | \(A=S_i=S_j\) | Copy the entire output row | **Used.** Prefix = smaller row index. |
| **PM** (partial match) | \(A=S_j\subsetneq S_i\) | Prefix psum of the **proper subset**; suffix adds only \(S_i\setminus S_j\) | **Used.** Prefix = the smaller set (fewer ones). |
| **Intersection** | \(A\neq S_i\) and \(A\neq S_j\) | Would need a **new** row \(A\), compute it first | **Refused.** §III-B: “this will significantly increase the complexity of the architecture design.” |

Worked example (Fig. 1/2, also Abstract): Row 1 = `1001`, Row 4 = `1101` share sub-combination `1001` (PM). Reuse Row 1’s product; add only the extra `0100`. Row 5 = Row 4 (EM): copy with **zero** extra adds.

Remainder pattern they actually issue:

\[
\text{ProSparsityPattern} \;=\; S_q \oplus S_p \qquad (= S_q\setminus S_p \text{ when } S_p\subset S_q).
\]

### 1.2 Graph (the forest, not the full DAG)

- They **only** consider **two-row** relations. Naive n-row search is \(O(m^n)\); \(n\ge 3\) is “unacceptable” (§III-B).
- Prefix graph is pruned to a **forest**: each node keeps **one** prefix, the one with the **largest common subset**; ties take the **largest index** (§III-D).
- Second prefix is refused: “30% performance drop and larger area”; Table II, second prefix usable on \(<6\%\) of nodes. First prefix already takes most of the density drop (SpikingBERT SST-2: bit 20.49% → one-prefix Pro 2.98% → two-prefix 2.30%).
- Meta information stored, **not** the graph: (i) temporal vector of length \(m\) (issue order); (ii) per-row `{Prefix index, XOR pattern}`. Space \(O(m)\), not \(O(m^2)\). Suffix matrix would be “more than 10 times” on-chip area (§V-D).
- Temporal law, not DFS/BFS: **stable sort by number-of-ones**. PM: fewer ones first. EM: same NO, smaller index first (sort is stable). If you process a superset first, you cannot reuse a subset that has not been computed (Fig. 1 Row 0 vs Row 3).

### 1.3 One consumer of one Y

Prosperity retires a spike row when **one** output row \(Y[i,:]\) is written. The consumer of \(Y\) is the **Spiking Neuron Array** (32 LIF cells) that emits the **next binary** spike map, or the SFU for softmax/LN of spiking transformers (§IV). Residual / PED / dual last-use are not in the dataflow. Transformer residual is not modeled as a second continuous reader of the same source.

CNN conv is lowered by **im2col** to the same GeMM. Transformer Q/K/V/out/FFN are native GeMM of shape \((T\times L, d_i)\times(d_i,d_o)\). Time is extra independent rows sharing \(W\), **not** a dense noncausal \(T\times T\) mix of one site.

They claim “more than 98% percent of operations in SNNs are spiking GeMM” (§II-A). That claim is about **their** VGG/ResNet/Spikformer/BERT classifiers, not about this optical-flow student.

### 1.4 Architecture that makes the forest cheap (A includes this, or the copy is partial)

PPU = Detector / Pruner / Dispatcher / Processor (§IV–V). Chosen tile \(m=256,\; n=128,\; k=16\) (Table III, §VII-B).

- **Detector:** preload \(m\times k\) spike tile into **TCAM**. Query a row with 1s masked to `X`; TCAM returns **all subset indices in one cycle**. Popcount = NO, used as the temporal key. \(m=1\) kills reuse. Large \(k\) makes sets unique (fewer subsets); \(k=4\) makes 0–1-spike rows whose reuse is “meaningless.” \(n\) does not affect sparsity.
- **Pruner:** proper-subset filter (drop EM candidates with larger index) + Argmax over remaining SI + XOR remainder.
- **Dispatcher:** product-sparsity table \(O(m)\); **does not** store suffixes. Bitonic sorter \(O(\log^2 m)\) time, \(O(m)\) space. High-overhead suffix search is a 32% slowdown in their own ablation (§V-D, §VII-D).
- **Processor:** 128× 8-bit adders. Row-wise. Load **prefix output row** into psum; bit-scan-forward on the XOR pattern; accumulate remainder weight rows; write \(Y\). Across K-tiles, further accumulate onto the output tile.
- **Pipeline (§VI):** Detector/Pruner/Dispatcher are 5-stage, \(m+4\) cycles per tile. Inter-tile: processing of tile \(n\) overlaps compute of tile \(n-1\) via **double-buffered** sparsity table and TCAM. They claim the detect phase is always shorter than compute ⇒ “overhead-free” except the first tile.
- **Not 1RW.** TCAM is a parallel associativity structure. Output buffer must supply prefix psums while accepting writes. Spike/weight/output/sparsity tables are double-buffered. 128-wide PE row.

Cost model they use to justify CAM (§VII-G): TCAM bitwise matches \(m^2\times k\); saved work \(\Delta S\times m\times k\times n\) floating-point adds, each add counted as **45×** one TCAM bit-op. Benefit if \(\Delta S>4.4\%\) at their tile; they report average \(\Delta S=13.35\%\), ratio ~3.0×. This is **their** 8-bit-add vs TCAM accounting, not a port-limited SIMD schedule. Detector power dominates on-chip compute **because every CAM cell fires every cycle** (Fig. 10, §VII-E).

Lossless / algorithm-agnostic on GeMM (§III-A, Table I vs Stellar’s FS neuron). They do **not** retrain. EM at 100% pattern sparsity still costs **one cycle** (§VII-D). ProSparsity does **not** apply across tiles.

**Required object, compressed:**

> binary rows \(M\), one shared \(W\), product = select-add of weight rows, two-row EM/PM only, one prefix per node, one consumer of \(Y\), tile-local, prefix-before-suffix issue, lossless on \(Y\).

Intersection is ExSpike’s object, not Prosperity’s. Weight sharing across time is SpikeX/PTB, not product sparsity. Bitmap AND of spike×weight-mask is FireFly-S, not the forest.

---

## 2. After absorb: which layers of THIS net match that object

Frozen student: Motion C12 / H67 / ep34, event-camera 2D optical flow, DSEC valid825 AEE. Historically expensive region: patch-embed residual **r1** (two convs) plus T10 PSN. Native projection conv + BN + residual add exist.

Captured residual/projection chain (local graph, not a novelty claim):

```
r1:   conv1 → fixed norm1 → sn2/PSN(T10) → AT-LIF → conv2 → fixed norm2 + shortcut
then: r1out → proj.sn → AT-LIF → proj.conv (stride-2) → proj.norm_layer (BN) → residual add
```

T10 PSN occupancy is a tensor at r1 (spatial ~240×320 before downsample). Projection BN sits **after** stride-2 `proj.conv` on domain **10×96×120×160**.

### 2.1 Layers that match Spiking GeMM after absorb (Prosperity’s A is legal here)

After the fold \(W\leftarrow\theta W\), any layer whose **input is the previous AT-LIF spike map** and whose **op is conv or linear with one shared kernel**, lowered to

\[
Y[i,:]=\sum_j s[i,j]\,W'[j,:],\quad s\in\{0,1\},
\]

is Prosperity’s object.

| Layer | Why it matches | Lowering |
|---|---|---|
| **r1 `conv2`** | Input = `sn2` spikes after absorb. Kernel shared across space and the T=10 maps. Product depends only on which input channels/pixels fired. | im2col → binary GeMM, same \(W'\) for every (t,h,w) row. |
| **`proj.conv`** (native projection, 3×3 stride-2) | Input = `proj.sn` spikes after absorb. Same select-add. | im2col → binary GeMM. This is the freeze’s “native projection convolution” **on the spike/gate consumer**. |
| **Any later ConvBN(SN(·))** or linear Q/K/V/out/FFN that reads absorbed spikes | Same as Prosperity’s CNN/transformer lowering (§II-B). | im2col or native GeMM. |
| **r1 `conv1`**, **if and only if** its input is an absorbed binary spike map from the previous block | Same. If `conv1` instead reads a continuous residual/shortcut, it does **not** match (that input is object 2). | Check the captured graph; do not assume. |

On those layers, EM/PM is well-defined: two (t,h,w) sites with equal / subset supports share / prefix-share the inner product against the **same** \(W'\). T=10 is already materialized by noncausal PSN **after** fire, so concatenating the ten binary maps as extra rows (Prosperity §II-A) is even more natural than in a causal LIF that would have to wait for future ticks. That is a mapping convenience, **not** X.

Prosperity’s “>98% spiking GeMM” slogan is still false as a **net** statement here: T10-before-threshold, residual/PED, and full-domain BN are not GeMM of binary rows. Copy A on the matching layers; do not retitle the net as Prosperity.

### 2.2 What absorb does *not* turn into Spiking GeMM

Absorb is a **fold across the neuron boundary**. It does not rewrite operators that sit **before** the threshold, and it does not rewrite a **different tensor**.

---

## 3. What still does NOT match (four load-bearing holes)

These holes survive the identity lock. They are the only legal sources of X. Renaming EM/PM/XOR/CAM onto the matching layers is not X.

### 3.1 T10 mix **before** threshold (continuous, then fire, then absorb)

Prosperity concatenates **already-binary** time steps as independent GeMM rows. THIS PSN is a **noncausal 10-point mix** of continuous pre-threshold values, then AT-LIF, then absorb.

Freeze: ordinary CSE 260 add/sub per T10 vector; lifting 159 add/sub + 35 intermediate RNE/sat. That is **compiled linear-form reuse** of a dense \(T\times T\) mix, not a subset relation on firing columns.

Illegal transfers:

- Issue T-rows in NO-order **inside** the mix. A “prefix in time” is not a subset of firing columns; reordering changes the T10 function.
- Treat the T10 matrix as Spiking GeMM. Its operands are not binary.
- Claim lifting/CSE as product sparsity. CSE reuses **compiled intermediate nodes** of a constant matrix; ProSparsity reuses **runtime prefix psums of binary combinations**. Different reused object, different legality condition.

After fire, the ten **output** spike maps may enter `conv2` / `proj.conv` as Prosperity rows. That is §2.1. The mix that **produced** those spikes is still not A.

### 3.2 Residual / PED continuous (second consumer, different tensor)

Prosperity: one consumer, one \(Y\), then LIF. ExSpike: Residual Spike SRAM holds **source spike feature maps** for shortcuts — still binary.

THIS freeze: consumer A is the **binary gate after absorb** (Prosperity may apply). Consumer B is the **continuous residual / PED / I24** path, plus native residual add after projection BN. Integer gates / I24 / PED q24 are already **0-diff** vs model on two captured windows.

Skipping or prefix-reusing the **gate GeMM** does not cancel source production that PED still needs. A zero in the absorbed spike map is a legal skip for consumer A and can be a **nonzero live residual** for consumer B. Last-use of Z vs PED U is not an architectural signal in Prosperity.

Do not write “dual-side sparsity.” LoAS dual-side is **sparse unary spikes × pruned weights**. FireFly-S dual-side is **Bitmap AND of spike vector × weight mask**. Neither is “binary gate + continuous PED of the same source.”

### 3.3 Full-domain BN (live actual batch stats, not folded running stats)

Native projection BN on captured students uses **actual batch statistics over the full 10×96×120×160 domain**, not frozen running stats. Local-window replay given free mean/var **undercharges** wait/storage (freeze).

Prosperity has no BN consumer. Their SFU does LayerNorm/softmax of spiking transformers, not a reduction barrier over `T×H×W` that **blocks** residual add. ExSpike OPT3 folds a **static** `1/pooling_size²` into FC weights — a compile-time scale, not live batch moments.

Prefix psum reuse of `proj.conv` does not shorten the wait for domain \(\mu,\sigma\). Affine of any tile is illegal until that barrier. This is a completion condition Prosperity never scheduled.

### 3.4 1RW / same-port SIMD vs TCAM + 128-add array

THIS student (measured): two-stage SIMD source, always-ready slots 6938 → 5354 (−22.83%, part generic round→sat fusion), **long backpressure both 8088**. Venue gate: same-port / same-state / same-backpressure net service ≥15%.

Prosperity’s Detector is TCAM; their “overhead-free” claim is detect(tile \(n\)) hidden behind compute(tile \(n-1\)) on **double-buffered CAM + a 128-wide adder array whose issue width is the XOR popcount**. Their ΔS>4.4% model does not include:

- 1RW contention between subset search, prefix-psum read, weight fetch, and dual-consumer live ranges;
- BN wait;
- the already-measured fact that **fewer adds can be entirely absorbed by backpressure** (lifting already showed this).

A 1RW serial subset scan of an \(m=256\) tile is **not** their Detector. Shipping CAM onto this student as “free” is false under the freeze. Replacing CAM with a 1RW scan **and still calling the paper Product Sparsity** is an incomplete copy of A, not X (see §5).

ExSpike’s own Fig. 8 is the same family of failure: APEC-2 cuts **calculation** cycles and **raises Weight-ready cycles**, wiping the win on some blocks. Arithmetic ↓ does not imply occupancy ↓.

---

## 4. ExSpike, in one paragraph, so it is not mixed into Prosperity

ExSpike (FPL 2026 preprint) is **full-event EConv** of **binary** spikes: every hidden conv/FC is triggered by valid events; PE = accumulate + LIF; no hidden multiplier. OPT1 bit-slices the multi-bit first layer; OPT3 folds average-pool into FC weights. **APEC** (g=2 default) is adjacent-position **AND-overlap** of channel-sets:

\[
O_G=\bigcap_{i=1}^{g} S_i,\quad \Delta N=(g-1)|O_G|,\quad \Delta C=(g-1)|O_G|C_o k^2.
\]

That is the **intersection** case Prosperity **explicitly refused** for GeMM rows, done only because adjacent spatial positions share a **binary** Φ(c). After absorb, APEC becomes a legal **A** on the same matching layers as Prosperity (binary spike maps at `sn2` / `proj.sn`), with a **different** reused object (spatial-neighbor intersection psum, not PM/EM prefix). Residual SRAM is still a **spike** shortcut, not PED. FPGA, DSP-free, CIFAR/SegNet — not DSEC AEE. Peak 0.80 GOPS/W/PE vs FireFly-T is **their** PE-normalized FPGA number; forbidden as a transferable student result.

After absorb, **do not** retitle APEC as “spatial ProSparsity.” Intersection ≠ PM/EM forest. Copy both As separately if you copy them.

---

## 5. Complete-transfer checklist (copy A, do not cite A)

To **actually copy** Prosperity onto the matching layers of §2.1, implement **all** of the following. Anything less is a rename. Absorb makes step 1 **true** on those layers; it does not waive 2–10.

1. **Lower only the matching layers** to binary \(M\) and absorbed \(W'\) with \(Y[i,:]=\sum_j M[i,j]W'[j,:]\). Do **not** lower T10-before-threshold, PED, or BN to this form.
2. Tile \(M\) to \(m\times k\) with \(m\) large enough for subsets (they needed \(m=256\), \(k=16\)). Pay on-chip spike tile + **TCAM \(m\times k\)** (or a documented equivalent associativity), weight tile \(k\times n\), output tile \(m\times n\) (they: 8/32/96 KB). If you drop \(m\) to 1 or to a SIMD vector width that has no prefix candidates, A is empty.
3. Detector: subset match of every row against the tile **and** popcount. One cycle/row is **their** CAM; on 1RW you must **measure** detect occupancy, not slogan it.
4. Pruner: one prefix, largest subset, tie → largest index; XOR remainder. No second prefix. No materialized intersection row (that is APEC).
5. Dispatcher: stable sort by popcount; issue prefix-before-suffix; store only one prefix per row; **no** suffix matrix.
6. Processor: load prefix **output row**, accumulate only remainder weight rows, write \(Y\); bit-scan skip zeros. ALU is add, not MAC.
7. Double-buffer so detection of tile \(n\) overlaps compute of tile \(n-1\); **prove the detect phase is hidden on this 1RW dual-consumer schedule**. Their inter-tile overlap proof does not transfer.
8. After \(Y\): AT-LIF (this neuron) to produce the **next binary** spike map, then absorb again. Residual/PED is **not** this step.
9. Numeric equality on \(Y\) (they are lossless on GeMM). Here: integer **0-diff** on gates / I24 / PED q24, not “approx sparse.” Prefix reuse that perturbs PED is not A.
10. Report **same-port, same-state, same-backpressure** service of the **real** consumers (gate **and** PED **and** BN), not \(\Delta S\times m\times k\times n\) add counts, not CAM-vs-add 45×, not GOP/s from a 128-add array, not FPGA GOPS/W/PE.

If you implement 1–6 on `conv2`/`proj.conv` and leave 7, 9, 10 on the dual-consumer chain unmeasured, you have a **component** copy. Freeze: component speedups must not be multiplied into FPS; source-only always-ready is not service.

**Partial copy that must be named as such:** control structure (prefix table, NO-order, remainder bitmap) on the **binary gate mask only**, while PED stays dense. Legal as a **control**, illegal as the title. The psum-reuse theorem carries only for consumer A.

---

## 6. Candidate X that is NOT renaming product sparsity

**Illegal X (reskins of Prosperity after absorb):**

- “reuse common binary sub-combinations of spike rows”
- “XOR remainder then add”
- “CAM / 1RW scan to find subset prefixes”
- “exact-match row copy”
- “tile-local combination reuse”
- “unroll T=10 as extra binary rows sharing W”
- “overhead-free detect/compute overlap”

That **is** ProSparsity (or ProSparsity with a worse Detector). After absorb it is **A**, which you must copy, not invent.

**Illegal X (reskins of ExSpike / FireFly after absorb):**

- adjacent AND-overlap psum (APEC / intersection)
- Bitmap AND of spike × weight-mask (FireFly-S dual-side)
- EConv “only valid events update \(k^2\times C_o\)” on the binary path

Also A, on the same matching layers.

A non-reskin X has to change the **reused object** and the **legality condition** to something §1 does not define:

| Candidate X (objects this net has, Prosperity does not) | Reused / scheduled object | Why it is not EM/PM |
|---|---|---|
| **Dual-completion last-use** | Token `{gate_done, ped_done, bn_stats_ready}`. A source slot may be dead for the absorbed binary gate while **still live for PED / projection / residual add**. | Prosperity stores `{prefix_row_Y}`. One consumer. |
| **Noncausal T10 DAG occupancy under 1RW** | Compiled T10 intermediate nodes already exist (260→159). X is a **port-correct issue order** that reduces the **8088 backpressure**, not a further subset-psum of binary rows. | T10 operands are continuous pre-threshold. NO-order is illegal. |
| **BN-domain live range** | Storage/wait contract for running mean/var over `10×96×120×160` as a **barrier**, not combination sparsity. Local windows with free \(\mu,\sigma\) are forbidden as a claim. | Prosperity has no such consumer. |
| **Consumer-asymmetric skip** | Zero on the absorbed spike map skips **only** consumer A; consumer B still issues. Stolen slot, if any, is PED work on the **same** two ports. | Prosperity’s skip retires the whole row. |

If the only hardware story is “binary-mask prefix of the gate GeMM,” it is still ProSparsity applied to a mask, and the PED / T10-before-threshold / BN / 1RW holes remain. That can be a **control** showing A on `proj.conv` is real; it cannot be the TCAS-II increment.

**1RW vs TCAM is not automatically X.** Building a 1RW Detector for the same forest is how you **finish A** on this student (checklist step 3/7). X begins only if the thing you schedule is no longer prefix \(Y\) of binary rows.

---

## 7. Strongest controls (same ports / state / backpressure)

Must beat these **on this student**, not in a 128-add CAM accelerator:

1. **Bit-sparsity only** on the matching layers (their Fig. 9: unstructured skip-zeros, “Prosperity 5.97”). If prefix reuse adds CAM/table/prefix-read and does not beat skip-zeros on **this** SIMD source, A is not worth copying.
2. **Official CSE T10 graph** (260 add/sub). Product-sparsity-of-rows is the wrong graph for the mix.
3. **Learnable 40-coeff lifting T10** (AEE 1.232979368, slots −22.83%, backpressure **unchanged 8088**). Strongest “reuse linear combinations, lose on ports” control. Any ProSparsity-like reuse that only cuts adds is predicted to die the same way.
4. **One-prefix vs two-prefix vs intersection.** They already measured two-prefix as not worth it; intersection they refused. If THIS net’s “X” is AND-overlap of adjacent supports, the control is **ExSpike APEC G2**, not Prosperity.
5. **Gate-only ProSparsity, PED still dense.** Dual-consumer accounting cannot be omitted. Source production for PED is the control.
6. **Full-domain BN vs local-window free \(\mu/\sigma\).** Freeze: local replay undercharges.
7. **Prosperity-class tile vs SIMD vector as tile.** \(m=\) vector width with \(k\) large is predicted to have almost no PM/EM (their §VII-B: large \(k\) → unique sets; \(m=1\) invalidates ProSparsity).
8. Do **not** add the SIMD-slot table to the integer-consumer table (−22.83% and −5.78% are different resource points).
9. Do **not** quote their 7.4× / 8.0× vs PTB, 1.8× / 193× vs A100, or ExSpike 0.80 GOPS/W/PE as this student’s result.

---

## 8. Kill gate

Copy A completely (checklist §5) onto the matching post-absorb layers of the frozen student **without** touching T10-before-threshold identity, **without** dropping PED, **without** replacing live full-domain BN by running-stat fold.

**Kill if any:**

- valid825 AEE > **1.259** absolute, or **> ordinary 1.219801338 + 0.005** relative (lifting already at +0.013178, relative fail);
- integer gates / I24 / PED q24 not **0-diff** vs model;
- **long backpressure stays ~8088** (or same-port/same-state net service of the dual-consumer chain does not move ≥15%) even if add counts or CAM-model \(\Delta S\) look good;
- always-ready slot cut is **only** generic round→sat fusion (freeze: part of −22.83% already is);
- the method requires **not absorbing** \(\theta\), or requires a per-token analog payload, to make prefix psums well-defined (identity break in the other direction);
- the title is EM/PM/XOR/CAM/unroll-T on `conv2`/`proj.conv` (that is A after absorb);
- T10 mix is reordered as if it were binary rows;
- BN uses gifted \(\mu/\sigma\) or frozen running stats as the **claimed** service baseline;
- OpenROAD-as-PPA, FPS products, CIM title (venue).

Session SCOPE: complete-chain same-resource net service **≥15%** after those controls. Lifting-style −5.78% on a different integer point does not pass. A clean copy of Prosperity on `proj.conv` that leaves 8088 untouched is a **successful A** and a **dead letter**.

---

## 9. Section quotes relied on

### Prosperity (HPCA 2025), `2503.03379.txt`

- **Abstract:** “a novel sparsity paradigm called Product Sparsity, which leverages combinatorial similarities within matrix multiplication operations to reuse the inner product result and reduce redundant computations.” SpikeBERT “density of only 1.23% and reduces computation by 11×, compared to bit sparsity, which has a density of 13.19%.” vs PTB and A100: “average speedup of 7.4× and 1.8× … energy efficiency improvements of 8.0× and 193×.”
- **§I:** “SNN’s neurons only react to information encoded as binary spikes (1 for spike and 0 for non-spike).” Spatial ID complexity “\(O(m^n)\)”; temporal: if Row 0 processed first, cannot reuse Row 3. “multiple rows of the binary spike matrix usually contain a common binary sub-combination, resulting in identical inner product results when multiplied by the shared weight matrix.”
- **§II-A:** unroll/concat time steps sharing the same weight; “computation becomes a sparse addition”; “more than 98% percent of operations in SNNs are spiking GeMM.”
- **§II-B:** CNN via im2col; transformer linear layers are spiking GeMM \((T\times L,d_i)\times(d_i,d_o)\); “operations in spiking attention are not efficiently supported by existing SNN ASICs.”
- **§III-A:** “the common binary sub-combination will generate the same inner product result”; “algorithm-agnostic and lossless method for the spiking GeMM.” Inner product = “accumulation of multiple weight values, which are ‘selected’ by the 1-values.”
- **§III-B:** \(S_i\) set definition; PM / EM / Intersection; intersection refused because it “requires creating a new row A and compute the result of A first.” Two-row only; \(n\ge 3\) “unacceptable.”
- **§III-C:** PM prefix = smaller set; EM prefix = smaller index.
- **§III-D:** one prefix; “30% performance drop and larger area”; Table II second-prefix \(<6\%\); forest + meta information \(O(m)\).
- **§IV:** layer-by-layer; Detector / Pruner / Dispatcher / Processor; Spiking Neuron Array then next spikes; SFU for exp/mul in softmax or LN; “reuse the PPU for spiking-GeMM-like operations in spiking attention.”
- **§V-A:** tile \(m\times n\times k\); \(m=1\) invalidates ProSparsity; \(n\) “has no impact on ProSparsity.”
- **§V-B:** TCAM, 1s masked to X, “all subset indices … single clock cycle.”
- **§V-D:** suffix matrix “more than 10 times” area; stable sort by NO instead of DFS/BFS.
- **§V-E:** row-wise; “Prefix row in the output matrix is fetched and serves as a starting point of the partial sum.” Bit-scan-forward on the XOR pattern.
- **§VI-A/B:** five-stage detect, \(m+4\) cycles; “ProSparsity processing phase of a tile is perfectly overlapped by the computation phase of the previous tile.”
- **§VII-A:** 8-bit weights; DC 28 nm; CACTI; DRAMSim3; cycle-accurate simulator; iso-accuracy because lossless. Models: VGG-16, ResNet-18, Spikformer, SDT, SpikeBERT, SpikingBERT; CIFAR / CIFAR10-DVS / GLUE — **not** DSEC optical flow.
- **§VII-B:** chosen tile **\(m=256\), \(k=16\)**; large \(k\) → harder prefixes; \(k=4\) reuse “meaningless.”
- **§VII-D / Fig. 9:** unstructured bit-skip 2.28× over PTB; ProSparsity with heavy dispatcher 2.16× more; overhead-free order 1.49× more. EM 100% sparse still 1 cycle.
- **§VII-E / Fig. 10:** Dispatcher (sparsity table) dominates area except buffers; Detector/TCAM dominates on-chip power “because every cell in TCAM is activated … every cycle.” Spikformer/CIFAR10 total 915 mW, DRAM a large fraction.
- **§VII-F:** vs LoAS weight-pruned nets, ProSparsity still cuts **activation** density ~4.1× (orthogonal to weight pruning).
- **§VII-G:** \(\Delta S\) threshold 4.4%; average \(\Delta S=13.35\%\); ratio 3.0×; TCAM \(m^2\times k\); add counted 45× one TCAM bit-op.
- **Table I:** vs PTB/Stellar: unstructured ProSparsity vs structured bit sparsity / specific neuron.
- **Table III:** 128 PEs 8-bit add; 32 LIF cells; 8/32/96 KB buffers; 1 KB TCAM.
- **Table IV VGG-16:** 128 PEs, 0.529 mm², 390.10 GOP/s, 299.80 GOP/J.
- **§VIII-C:** GNN redundancy-removal is **not** transferred: SNN patterns “exhibiting random distributions due to dynamic input activation”; SNN density 60–90%, not ≥99.99% graphs.

### ExSpike (FPL 2026 preprint), `2606.20414.txt` — only the contrast quotes

- **Abstract:** “adjacent-position event compression to reduce redundant accumulations across spatially adjacent spike sequences.”
- **§I:** `MAC_event = α·Ci·Hi·Wi·Co·k²`; full-event = conv/FC “triggered by valid spike events.”
- **§II:** LIF; OPT3 `W_fc ← W_fc / pooling_size²`; “executed in a pure event-driven manner without using multipliers.”
- **§III:** Residual Spike SRAM = “source spike feature maps used by shortcut or residual connections.”
- **§III-A2:** \(O_G\) intersection (1); \(\Delta C=(g-1)|O_G|C_o k^2\); “preserves numerical equivalence”; gain “does not necessarily increase monotonically with group size.”
- **§IV-A / Fig. 8:** G2 best; calculation ↓ can be offset by Weight-ready ↑.

### This freeze (not literature)

- **IDENTITY_ATLIF.md:** \(o\in\{0,\theta\}\); “推理时层共享的 \(\theta\) **吸进下一层 \(W\)**”; after absorb, “层间传递的是 **0/1 脉冲**”; Prosperity/Gustav/FireFly “都变成合法的 **A**”; residual/PED “另一条连续张量”; dual consumers = “门（吸完后的二值）” vs “残差连续路径.”
- **round2_absorb/PROBLEM.md:** T10 mix “is continuous arithmetic, then threshold, then absorb.” Native projection BN: actual batch stats over `10×96×120×160`. Ordinary AEE 1.219801338; lifting 1.232979368; backpressure both 8088; integer 0-diff on gates/I24/PED q24.

---

## 10. One-sentence verdict for Round-2 ideation

After absorb, **r1 `conv2` and `proj.conv` (and any later SN→conv/linear) are Prosperity’s binary GeMM**; copy the EM/PM forest there as **A**. **T10-before-threshold, residual/PED, full-domain BN, and 1RW dual-consumer occupancy are not that object**; X must be one of those, or it is a reskin. A letter whose mechanism is product sparsity on the absorbed spike path has no increment, even if the copy is complete and lossless.
