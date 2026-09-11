# R2L4 — PSN mix is before threshold; T10→Prosperity is da4ml+Prosperity (A+A)

Date: 2026-09-11. Round-2 freeze only: `round2_absorb/PROBLEM.md` + `IDENTITY_ATLIF.md`.  
Local full text: Fang et al., *Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies*, NeurIPS 2023, arXiv:2304.12760v4 (file `survey_ab_fusion_20260910/p0_txts/2304.12760.txt`).  
Prosperity and da4ml are named as **A** from already-read local texts / in-tree compiler; this note does not re-derive their microarchitecture.

**Question.** After AT-LIF `{0,θ}` with layer-shared `θ` absorbed into the next `W`, is the letter’s **X** “the compiled T10 mix (continuous add graph) feeding a binary Prosperity engine”? Is that just **da4ml + Prosperity stacked (A+A, not X)**? What hole remains if both are copied fully?

**Verdict.** Yes: that pipeline is PSN’s own cut (continuous mix → binary `S`) implemented with two complete priors. It is **A+A, not X**. Prosperity never sees the T10 mix. da4ml never sees a binary GeMM. Stacking them does not create a third reuse theorem. After both copies, the remaining hole is **dual last-use and 2-port occupancy across the threshold cut** (binary gate after absorb vs continuous residual/PED, plus full-domain projection BN) — not product sparsity of compiled adds.

---

## 0. The cut this freeze already named

From `IDENTITY_ATLIF.md` / `PROBLEM.md`:

- Official AT-LIF: \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\). At inference, layer-shared \(\theta\) is absorbed into the next \(W\). After absorption the **spike path is binary** \(\{0,1\}\times W\).
- Prosperity / Gustav / FireFly apply **there**, as complete priors to copy.
- Residual / PED / I24 is a **different** continuous tensor. Not AT-LIF carrying analog amplitude.
- **Before** the AT-LIF threshold this net has a noncausal T10 mix (PSN / lifting CSE). That mix is **continuous arithmetic, then threshold, then absorb**.

Frozen CSE numbers (`PROBLEM.md`, do not remeasure here):

| Student | After official CSE | Intermediate RNE/sat |
|---|---|---|
| ordinary dense T10 | 260 add/sub per T10 vector | 0 |
| learnable 40-coeff lifting T10 | 159 add/sub | 35 |

Same two-stage SIMD: always-ready 6938 → 5354 (−22.83%); **long backpressure both 8088**. Part of −22.83% is generic round→sat fusion. Separate integer consumer 758777 → 714889 (−5.78%); do not add the tables. Ordinary AEE 1.219801338; lifting 1.232979368 (relative +0.013178; abs 1.259 pass, +0.005 fail).

The pipeline the question names is therefore:

```
X (continuous T10 currents)
    --[H = W_T10 X, compiled adder graph / da4ml CSE]-->
H  (continuous hidden; PSN Eq. 9)
    --[AT-LIF threshold]-->
s ∈ {0,1},  o = θ s
    --[absorb θ into next W]-->
binary GeMM  s × W'     --[Prosperity / Gustav / FireFly]-->
next layer

PED / residual / I24: a second continuous tensor, not o.
proj.BN: actual batch stats over 10×96×120×160, not this mix.
```

Two different reuse theorems sit on two different sides of **one** Heaviside. They do not compose into X.

---

## 1. PSN paper: the mix is intra-neuron; the output after threshold is binary S

File-title check: local txt header is *Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies* (NeurIPS 2023). No invented venue line.

### 1.1 Inter-layer object is already a spike, not H

§1:

> The SNNs use discrete spikes to communicate between layers, which enable the event-driven computational paradigm

Fig. 1 caption:

> X[t], S[t] are the input current and the output spike, and H[t], V[t] are hidden states at time-step t.

Vanilla firing, Eq. (2):

> \(S[t]=\Theta(H[t]-V_{th})\)

\(\Theta(x)=1\) iff \(x\ge 0\), else \(0\). The paper’s layer-to-layer token is \(S\), never \(H\).

### 1.2 PSN family: continuous linear mix, then binary S

§3.2, Eqs. (9)–(10) — the identity of this island:

> \(H = WX,\qquad W\in\mathbb{R}^{T\times T},\; X\in\mathbb{R}^{T\times N}\)
>
> \(S = \Theta(H-B),\qquad B\in\mathbb{R}^{T},\; S\in\{0,1\}^{T\times N}\)
>
> \(B\) is the learnable threshold, and **\(S\) is the binary output spike sequence**.

The T-order mix is Eq. (9): each \(H[t]\) is a **linear combination of input currents** \(X[i]\), fully parallel, no reset. The spike is Eq. (10): **element-wise threshold**. Complexity they advertise is the **matrix–matrix multiply** of Eq. (9) on GPU BLAS, then an element-wise \(\Theta\). They do not send \(H\) to the next layer.

Masked PSN, §3.3: \(H=(W\cdot M_k)X\). After the mask,

> \(S[t]\) can be computed and **sent to the next layer** once \(X[t]\) is received.

The thing that leaves the neuron is still \(S[t]\), not the mix.

Sliding PSN, §3.4, Eqs. (14)–(15):

> \(H[t]=\sum_{i=0}^{k-1} W_i\cdot X[t-k+1+i]\)
>
> \(S[t]=\Theta(H[t]-V_{th})\)

Same cut: k-tap continuous FIR, then Heaviside, then binary \(S[t]\).

Noncausal full PSN (this net’s T10) is the unmasked Eq. (9): **all 10 inputs live until all 10 \(H[t]\) exist**, then \(\Theta\). §3.3 states the latency cost explicitly: \(S\) of a full PSN “can only be generated when all \(X[t]\) have arrived.” That is a **barrier on the mix**, not a binary-row GeMM.

### 1.3 What PSN does *not* give the letter

- No product-sparsity of spike rows. Next-layer GeMM is out of scope.
- No CMVM compiler. Eq. (9) is a dense \(T\times T\) matmul on MKL/cuBLAS for **simulation speed**, not an adder graph on two write slots.
- No dual residual consumer of \(H\). Communication is \(S\in\{0,1\}\).
- Related-work TDBN (§2.3) “eliminates the variance introduced by thresholds.” That is a **normalization identity**, not a live full-map batch barrier, and not this net’s captured `proj.norm_layer` over `10×96×120×160`.

PSN already wrote the sentence a stacked letter would pretend is X: **continuous mix inside the neuron, binary \(S\) after threshold, spikes between layers.**

After this freeze’s absorb, AT-LIF agrees with that inter-layer object: \(o=\theta s\) becomes \(W\leftarrow\theta W\), and the next GeMM sees \(s\in\{0,1\}\). PSN never had a \(\theta\) payload on \(S\); absorb is the extra AT-LIF step that **makes** the two papers meet. It does not move the mix onto Prosperity’s side of the cut.

---

## 2. da4ml owns the pre-threshold mix. Prosperity owns the post-absorb GeMM.

### 2.1 da4ml (TRETS 2026 / arXiv:2507.04535) — A for Eq. (9)

In-tree official 0.6.0 (`psn/cmvm_20260909/da4ml_official/docs/cmvm.md`): \(y=Wx\) with **constant** \(W\).

1. Stage 1: columns of \(W\) as vertices, distance = nnz of sum/difference, Prim MST, \(W=W_1W_2\).
2. Stage 2: CSD, greedy bitwidth-weighted CSE of \(a\pm(b\ll s)\).
3. Exact arithmetic. FPGA LUT/adder graph, optional delay constraint (min-delay **increases** nodes).

That is exactly PSN Eq. (9) / this net’s T10 (ordinary 10×10 or 40-coeff lifting) as a **constant-matrix–vector multiply**. The freeze’s 260 vs 159+35 **is** that compiler on this student. Node count ≠ cycles: da4ml S0 10×10 already showed −49% nodes ≠ −49% cycles; this freeze already showed always-ready −22.83% with **long-BP 8088=8088**.

da4ml activations are **multi-bit**. It has no spike tile, no TCAM, no prefix psum, no product sparsity. It does not know AT-LIF, dual consumers, or two writeback slots.

### 2.2 Prosperity (HPCA 2025, arXiv:2503.03379) — A for \(s\times W'\) after absorb

Workload they accelerate is **Spiking GeMM after spikes exist**:

> the information (activation) propagating in the network is a binary spike event … we can unroll and concatenate all spike matrices in different time steps to get a single binary spike matrix … The major computation of the SNN model is the matrix multiplication of the binary spike matrix and floating point weight matrix. … more than 98% percent of operations in SNNs are spiking GeMM. … If an element is 1, then the corresponding weight is straightly accumulated; if an element is 0, then the corresponding weight is skipped.

Reuse theorem (product sparsity): two **binary** rows, same 1-set or subset ⇒ same inner product or prefix psum. Intersection refused. One prefix. Tile-local. Detector = TCAM. Processor = 128× 8-bit **adders**, not MACs.

**Neuron is after GeMM, not a T×T mix before it.** §IV:

> After that, the results are sent to the Spiking Neuron Array to generate spikes for the next layer, which is a general and necessary unit for SNNs. … 32 Cells LIF Neuron.

They unroll time as **independent extra rows sharing \(W\)**. That is the opposite of PSN Eq. (9), where time is mixed **inside** \(H=WX\) **before** \(S=\Theta(H-B)\). You cannot issue T10 rows in popcount order without changing the mix. You cannot CAM-match **continuous** \(H\) rows: identical supports with different amplitudes are not equal products, and after absorb the mix is no longer on the spike path anyway.

Prosperity’s “98% GeMM / cheap LIF array” is the accounting that **deletes this net’s T10 source**. Copying Prosperity leaves 8088 untouched.

### 2.3 Two reuse theorems, not one

| | da4ml CSE | Prosperity ProSparsity |
|---|---|---|
| Object | constant \(W_{T10}\), multi-bit \(X\) | runtime binary spike rows \(M\) |
| Reused thing | subexpression \(a\pm(b\ll s)\) | already-computed **output-row psum** of a subset pattern |
| When | compile time | runtime, after \(S\) exists |
| Time | mixed **inside** \(H=WX\) | unrolled as extra **independent** rows |
| This freeze | 260 / 159+35 already emitted | legal **after absorb** on the spike GeMM |

Calling both “combination reuse” is a rename that confuses a constant adder graph with a runtime prefix forest. They do not share an IR.

---

## 3. “Compiled T10 mix feeding a binary Prosperity engine” is A+A

### 3.1 The mix does not feed Prosperity

PSN §3.2–3.3: \(H\) is hidden; **\(S\) is sent to the next layer**. Prosperity’s Detector input is a **spike matrix**. The T10 add graph produces \(H\), not \(M\). The only legal edge is:

**da4ml(Eq. 9) → threshold → absorb → Prosperity(Spiking GeMM).**

“Mix feeding Prosperity” is a category error unless it means that trivial cascade.

### 3.2 The cascade is PSN’s 2023 pipeline plus two published tools

1. PSN already specified mix-then-binary-\(S\).
2. da4ml already compiles the mix to an adder graph (and this repo already ran it).
3. After absorb, the spike path **is** the binary GeMM Prosperity defined. `IDENTITY_ATLIF.md` says copy it **as A**.
4. DATE 2025 hybrid SNN (dense core on the multi-bit first layer, event cores after) is the **shape** prior for “dense arithmetic then sparse binary engine.” Classification VGG, not T10, not OF — still A for the stacking cartoon.

A letter whose X is the cascade is three stacked priors. Reviewer one-liner: *you ran da4ml on PSN’s \(W\) and Prosperity on \(S\).*

### 3.3 What a complete copy of both actually is (checklist, not a title)

**Copy da4ml on T10 (A):** official two-stage MST+CSE (or the same graph already in the freeze); bit-exact vs the integer model; **same two write slots**; report always-ready **and** long-BP separately; ordinary 260+cutoff+generic fusion as the control, not unfused lifting.

**Copy Prosperity on post-absorb spike GeMM (A):** binary \(M\), tile \(m\times k\), TCAM subset, one-prefix forest, NO-sort issue, remainder XOR, prefix **output row** reuse, LIF/AT-LIF after \(Y\), lossless on that GeMM. Gustav NRV/CPTB and FireFly Bitmap AND are **sibling A** on the same binary path, not extra X.

**Do not** apply Prosperity’s detector to T10 nodes, CSE commons, or PED values. Those are not binary rows.

If either copy is incomplete, the letter is not “A+X”; it is an incomplete A.

---

## 4. Hole that remains after both copies (the only legal B)

After da4ml has compiled Eq. (9) and Prosperity has taken \(s\times W'\), reviewers subtract both. What is still billed on **this** net:

### 4.1 The threshold cut is not a free register (2-port occupancy of the mix)

da4ml’s figure of merit is adder/LUT count. This machine is **two-stage SIMD, two writeback slots**. Freeze: CSE already cut the mix; always-ready −22.83%; **long-BP 8088=8088**. Prosperity’s 128-add + TCAM array is a **different** machine and sits **after** \(S\) exists. Copying it does not drain the T10 file. Copying da4ml “harder” (bit-serial DA, min-delay, extra CSE) is still A and is predicted to die the same way if it only cuts adds.

Remaining B: **issue/live-range of the pre-threshold DAG under two ports**, including the 35 lifting RNE/sat writes. Not “product sparsity of compiled adds.”

### 4.2 Dual last-use: binary gate vs continuous PED

Prosperity retires a row when **one** GeMM output row is done, then 32 LIF cells. da4ml emits **one** \(y=Wx\).

This freeze keeps two objects:

1. Spike / gate after absorb — binary, Prosperity’s algebra.
2. Residual / PED / I24 — continuous, **not** AT-LIF \(o\) with analog amplitude.

A skip, prefix, or last-use that is legal for (1) does not cancel production still required by (2). Source parents of the T10 DAG stay live until \(\max(\text{gate done}, \text{PED done}, \text{BN stats ready})\). Prosperity has no such token. da4ml has no second sink type.

**Illegal rewrite:** “continuous AT-LIF amplitude shared by two MACs.” Absorb already removed amplitude from the spike path. The second consumer is the **residual tensor**, not \(o\).

**Narrow remaining X (must be named as this, not as T10→Prosperity):** one compiled mix, **two typed exits** (Heaviside/gate vs q24 PED) without cloning the 159-node cone, last-use = the union, same two ports. Control: ordinary CSE with the **same** two-output postprocess. If that already matches 8088, this X is dead.

### 4.3 Prosperity cannot touch the mix; da4ml cannot touch product sparsity

Noncausal T10: every output \(H[t]\) uses the whole vector (PSN §3.2–3.3). Hierarchical empty-window skip (SpikeX) and prefix-psum of **time-unrolled binary rows** (Prosperity) are both illegal on Eq. (9). CSE of the constant matrix is the legal reuse, and it is already A.

### 4.4 Full-domain projection BN

Neither compiler nor ProSparsity computes batch mean/var over `10×96×120×160`. Local-window replay with free \(\mu,\sigma\) is forbidden by the freeze. RISCSparse-style fold \(Y=aX+B\) assumes **frozen** running stats — a different student, new AEE. This hole is **hygiene on the chain**, not a reason to retitle the cascade.

### 4.5 AEE

Prosperity is lossless on binary GeMM: it cannot repair lifting +0.013178 vs ordinary. da4ml is exact: it cannot either. Relative gate +0.005 already fails for the lifting student. A stacked letter that keeps unconstrained 40-coeff lifting as the mix still dies on AEE even if slots later move.

### 4.6 What is *not* a remaining hole

- “Need a binary engine after threshold” — Prosperity/Gustav/FireFly, must copy, not X.
- “Need to compile the 10×10” — da4ml, already copied in the freeze.
- “PSN is parallel / no reset” — the PSN paper.
- “Fewer adds in the mix” — already measured; absorbed at 8088.
- Generic round→sat fusion — already part of −22.83%.

---

## 5. A / B / X map for ideation (not a chosen title)

**A (copy fully, then subtract):**

1. PSN Eqs. (9)–(10): noncausal (or masked) mix, **binary \(S\) after \(\Theta\)**.
2. AT-LIF absorb: spike path \(=\{0,1\}\times W'\).
3. da4ml two-stage CMVM + the freeze’s 260 / 159+35 graphs + cutoff fold + generic fusion.
4. Prosperity complete (TCAM, one-prefix, NO-sort, remainder XOR, lossless GeMM). Gustav and FireFly-S on the **same** binary path.
5. DATE 2025 hybrid dense-then-event cores, as the stacking **shape**.

**B (this net only):**

- T10 mix **before** threshold, 2-slot occupancy, 8088 tied after CSE.
- Dual consumers with **different algebras** after the cut.
- Full-domain proj.BN.
- Lifting relative AEE fail.
- Do not add 8088 to 758777.

**X that is still a reskin (illegal):**

| Claimed X | Why it is A+A |
|---|---|
| Compiled T10 feeding Prosperity | da4ml then Prosperity; PSN already drew the cut |
| Product sparsity of CSE nodes | Wrong object; CSE is constant-matrix, ProSparsity is binary rows |
| “After absorb we can use Prosperity, that’s the contribution” | Freeze already assigned Prosperity to A |
| Hybrid dense mix + sparse spike core as title | DATE 2025 shape; still two copied engines |
| Unrolling T10 as extra GeMM rows | Changes PSN Eq. (9); Prosperity §II-A, not this mix |

**X that is not a reskin of the stack** (still has to beat the controls; not declared a title here):

- Dual-typed last-use **across** \(\Theta\): gate sink may be binary after absorb; PED sink stays continuous; one DAG; two ports; BN charged.
- Write-slot-optimal schedule of the **pre-threshold** graph (cost = committed writes, not add count), with Prosperity running only on the binary sink as A.

If the only sentence is “continuous add graph then binary Prosperity,” there is no X.

---

## 6. Controls and kill-gate

**Controls (same ports / state / backpressure):**

1. Ordinary 260-node CSE + terminal cutoff + generic fusion (no Prosperity).
2. Lifting 159+35 + same fusion (current freeze).
3. (1) or (2) + **complete** Prosperity/Gustav/FireFly **only** on post-absorb \(s\times W'\).
4. Two-output da4ml cutoff (gate int vs PED q24) without a new schedule — kills typed-sink X if 8088 does not move.
5. Gate-only skip, PED still dense.
6. Frozen-BN student vs actual-batch `10×96×120×160`.
7. Do not quote Prosperity 7.4× / da4ml LUT−1/3 / PSN GPU speedup as local service.

**Kill if:**

- valid825 AEE > 1.259 or > ordinary + 0.005;
- long-BP stays ~8088 (or complete-chain same-resource net service < 15%);
- win is add-count, CAM \(\Delta S\), or always-ready fusion already in −22.83%;
- method requires feeding \(H\) (or PED) to a binary prefix engine;
- method requires **not** absorbing \(\theta\) (conflicts with `IDENTITY_ATLIF.md`);
- the two tables are added;
- title is Prosperity, da4ml, PSN-parallel, or “binary after threshold.”

---

## 7. Two-sentence prior statement (for a 5-page letter, not a pitch)

PSN already emits **binary \(S=\Theta(H-B)\)** after a dense T-order mix \(H=WX\); da4ml already compiles that mix to a CSE adder graph (local 260 vs 159+35); after AT-LIF absorb the spike path is the binary GeMM Prosperity accelerates. A brief whose X is “compiled T10 feeding a binary Prosperity engine” is those three priors stacked. The hole they do not cover is dual last-use and two-port occupancy **at the threshold cut**, plus full-domain projection BN — measure those, or do not write the cascade as the contribution.

**Sources (absolute):**

- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2304.12760.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/round2_absorb/PROBLEM.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/IDENTITY_ATLIF.md`
- Prosperity quotes: `.../p0_txts/2503.03379.txt` (§II-A, §IV)
- da4ml method: `.../psn/cmvm_20260909/da4ml_official/docs/cmvm.md`
