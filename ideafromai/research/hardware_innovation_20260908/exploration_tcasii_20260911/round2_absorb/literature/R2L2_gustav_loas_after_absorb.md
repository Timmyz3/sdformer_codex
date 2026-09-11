# R2L2 — GustavSNN + LoAS remapped onto AT-LIF **after absorb**

Date: 2026-09-11. Transfer contract, not a novelty claim. Round-2 identity only.

Freeze: `round2_absorb/PROBLEM.md` + `../IDENTITY_ATLIF.md`.  
Do **not** use round-1 `literature/L2_gustav_loas.md` as the identity (that file is locked to continuous θg). This note supersedes L2 wherever the two disagree.

Sources read in full:

- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/GustavSNN_HPCA2026_public_mirror.txt` (Hwang, Lee, Koo, Kung; HPCA 2026; DOI 10.1109/HPCA68181.2026.11408587)
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2407.14073.txt` (Yin, Kim, Wu, Panda; MICRO 2024; arXiv:2407.14073v3)

Forbidden sentence: *“GustavSNN = Prosperity + Gustavson.”* Gustav cites Prosperity only as an energy comparison (§VI-C4, simulator + CACTI). It does not implement product-sparsity reuse. LoAS never mentions Prosperity.

---

## 0. Identity cut this remap is allowed to use

Official AT-LIF: \(o_i[t]=\theta_i\cdot H(m_i[t]-\theta_i)=\theta_i s_i[t]\), \(s\in\{0,1\}\), \(o\in\{0,\theta\}\).  
At inference, **layer-shared** \(\theta\) is absorbed: \(W\leftarrow\theta W\). After that cut, **inter-layer spike traffic is binary** \(\{0,1\}\times W\). Not a per-event analog payload, not a non-absorbable int8 contract.

Two objects stay distinct (freeze):

| Object | Algebra | Who may copy it as A |
|---|---|---|
| **Spike GeMM after absorb** | binary select-add \(s\in\{0,1\}\times W\) | Gustav CPTB/NRV, LoAS FTP/fiber, Prosperity product-sparsity, FireFly bitmap AND — **complete priors, must copy** |
| **Residual / PED / I24** | a **different** continuous tensor | **neither** paper. Not AT-LIF \(o\) carrying amplitude |

**Before** the AT-LIF threshold this net has a **noncausal T10 mix** (PSN / lifting CSE). That mix is continuous arithmetic, **then** threshold, **then** absorb.

Pipeline the two papers never ran:

```
T10 mix (noncausal, compiled add/sub; ordinary 260 or lifting 159+35 RNE/sat)
        │
        ├─► AT-LIF threshold  ──► o={0,θ} ──► absorb θ into next W ──► binary GeMM  ══ spike path
        │
        └─► residual / PED / I24  (continuous; 0-diff q24 on captured windows)     ══ other path
+ native projection conv + BN (actual batch stats over 10×96×120×160) + residual add
```

Round-1 L2 blocked Gustav/LoAS with “θg is an amplitude, NRV is a bit.” **That block is dead on the spike path.** After absorb, binary \(B\) is the official identity of the GeMM. The remaining mismatch is **where** Gustav puts causal in-situ LIF, and **what** LoAS packs into a T-bit unary fiber.

---

## 1. How the two papers relate (still not a stack)

They are **parallel priors**. LoAS (MICRO 2024) is IP + fully temporal-parallel for **dual-sparse** unary SNNs and **rejects** Gustavson/OP for that setting (§III: extra `t` multiplies psum traffic). GustavSNN (HPCA 2026) is in-situ GP + column-parallel tick-batch for **spike-sparse, typically dense-weight** SNNs and treats LoAS as the **TP baseline**, not a layer it includes.

| Axis | LoAS | GustavSNN |
|---|---|---|
| SpMM family | IP; Gust/OP “not suitable” for dual-sparse SNN | In-situ GP; IP-FTP and OP lose on potential traffic |
| Time | FTP: `t` innermost, spatially unrolled | Tick-batch sequential `t`; column-major time-second |
| Sparsity identity | Dual-sparse (LTH ~97–98% W, packed silent-neuron A) | Spike sparsity + quantization; **explicitly not** dual-sparse as the main path (§VI-A) |
| Skip unit | Silent neuron across **all** `T` (spatial) | All-zero **P-column row at one tick** (temporal) |
| Neuron | Hard-reset LIF, unrolled over small `T` (default 4) | Sequential LIF (subtractive reset) **in the PE** |
| Role of the other | N/A (2024) | LoAS = TP [47]; Prosperity = PR **comparison** [42] |

---

## 2. GustavSNN — objects that exist in the paper

### 2.1 Loop nest (Alg. 1 / Fig. 3 / Fig. 5)

- Weight \(A\in\mathbb{R}^{M\times D}\) (eval: 8-bit)
- Spike \(B\in\{0,1\}^{D\times N\times T}\)
- Output spike \(C\in\{0,1\}^{M\times N\times T}\)
- Column partition: \(P=N/K\), \(B=[B_0|\cdots|B_{K-1}]\), \(C_k=A\cdot B_k\), \(C\) is **concatenation** along columns. Columns of \(B\) are independent.

```
parallel-for m in [0, M):          # tile-level, one weight row
  parallel-for k in [0, K):        # PE-level, one column partition
    for t in [0, T):
      for d in [0, D]:
        for n in [p, p+P):
          O_k[m,n,t] += A[m,d] * B_k[d,n,t]
      C_k[m, p:p+P-1, t] = LIF(O_k[m, p:p+P-1, t])
```

Loop order is **column-major time-second** (§V-B): a PE sweeps the entire current-tick column partition (all `d` of that tick) so the `P` local potentials are complete, **then LIF**, then the next tick. Timestep-major is the rejected alternative at high sparsity (Fig. 13: >95% sparsity, column-major spike-register utilization >60% vs timestep-major <25%). Occupancy of an L-1D jumps from \(1-\rho^P\) to \(1-\rho^{PT}\) under timestep-major.

### 2.2 CPTB

Problem CPTB exists to solve: in-situ GP wants a \(1\times N\) **potential register inside the PE**. For \(N=1024\), 8-bit, that is ≥1 KiB/PE and \(N\) varies by layer (§IV-B). CPTB partitions \(B\) along **output-neuron columns**. Each PE holds only \(P=N/K\) potentials in local REG and updates them **in-situ** (no global membrane buffer, no ex-situ merge). Tick-batch **layer-wise** semantics are kept: all ticks of this layer before the next layer (Fig. 1 right vs SpinalFlow left vs temporal-parallel middle).

Default silicon (Table II): 8 tiles, 8 PEs/tile, **\(P=8\)**, 8-bit W, 1 GHz, 1.1 V, 28 nm FDSOI, dual-port shared W L-BUF 8 KiB/tile, spike G-BUF 8×1 KiB, weight G-BUF 128 KiB, LPDDR3 6.4 GB/s, **1.34 mm²** post-layout+CACTI. No inter-tile reduction network (concatenation, not merge).

### 2.3 NRV

After CPTB, a submatrix has \(P\) columns. If a row is all zeros **at this tick**, skip the fetch. Probability a row is all-zero scales as \(\rho^P\) (§IV-D). Fig. 6 on ONE/ResNet-18: **\(P=8\) skips ~80% of rows; \(P=32\) skips ~58%.**

NRV payload per kept row of submatrix \(k\): **row index + \(P\)-bit naive \(\{0,1\}\) vector** (not COO column indices). Two skip sites in the PE (§IV-E): (1) weights whose row index is absent from NRV are not fetched; (2) if the fetched weight is 0, skip that spike row.

### 2.4 NR4 PE and in-situ LIF (the assumption that will break)

PE is fetch ‖ exec. **`NR` = merger-tree width = number of L-1Ds; Fig. 9 uses `NR=4`.** L-1Ds emit nonzero column indices; merger emits the smallest; selected bit is cleared (`SPIKE -= (1 << Col_idx)`); after the current-tick submatrix row is consumed: **compare `P` potentials to \(V_\theta\), emit spikes, neuron-model update**.

Implemented neuron (§II-A, subtractive reset):

```
v_j(t) = λ v_j(t-1) + Σ_i w_ij S_i(t)
S_j(t)=1 and v_j(t) ← v_j(t)-V_θ   when v_j(t) ≥ V_θ
```

That membrane **is** the in-situ REG. Commit is `LIF` after the current-tick `d`-sweep. §V-B: “the neuron computation logic inside the PE can be replaced with any target model without altering the overall architecture” — **architectural compatibility sentence, not a measurement of AT-LIF, T10, or optical flow.**

Same-ID barrier (§V-A, §VII-A): PEs with the same partition id **across tiles** share one NRV submatrix and **synchronize** before the next. Unpruned idle ~1%; W sparsity <80% idle <10%. Two-pointer intersection (§VII-B) is **not** the default; it recovers dual-sparse and **hurts dense W**. Do not transfer two-pointer as always-on A.

### 2.5 What Gustav evaluates (do not export as this student’s result)

SEW ResNet-18 rate T=4 87.5%; ONE ResNet-18 temporal T=24 95.2%; SDT ViT rate T=4 88.9%; SPA Swin-T temporal T=40 98.4%. Abstract **up to 11.8×** GOPS/W vs naïve GP; **1.43×** vs SOTA SNN accelerators; vs TP/LoAS **1.52–2.25×**; vs Prosperity **1.18–1.71×** (comparison only). Fig. 4 is **flat-memory, unbounded bandwidth, no bank conflicts, no timing**. Neuromorphic chips are explicitly not compared.

---

## 3. CPTB / NRV after absorb — 1-1 vs not

After absorb, the **spike GeMM** is legally \(B\in\{0,1\}\times W\). That is necessary for Gustav, **not sufficient**. Gustav’s PE is not a GeMM engine with a detachable LIF. It is **in-situ causal membrane + binary B + LIF-after-this-tick**.

### 3.1 Maps 1-1 (legal A on the **post-absorb spike GeMM only**)

| Gustav object | This net, after absorb | Why 1-1 |
|---|---|---|
| \(B\in\{0,1\}^{D\times N\times T}\) | Inter-layer spikes \(s\in\{0,1\}\) | Official identity. \(\theta\) already sits in \(W\). Inner kernel is select-add, no multiplier. |
| \(A\) multi-bit weight row, written once to shared L-BUF | Next-layer \(W\leftarrow\theta W\) | Still multi-bit. GP’s reason to exist (read 1-bit sparse rows, not multi-bit \(W\)/potentials) still applies **to this GeMM**. |
| \(C_k=A\cdot B_k\), concatenation not reduction | Column-tiling of the binary GeMM | Output columns of a GeMM are independent. CPTB’s partition algebra does not need LIF to be true. |
| NRV = row index + \(P\)-bit occupancy **of binary \(s\) at one tick** | Skip a weight row when none of the \(P\) sites spiked this tick | \(\rho^P\) math is well-defined on **this** binary tensor. Recompute \(\rho\) on **this student**, do not import ONE Fig. 6. |
| Dual skip: missing NRV index, and \(W=0\) | Same two predicates on the GeMM | Skip-2 is idle if this student’s \(W\) is dense — report that; do not import LTH. |
| Fetch ‖ exec, L-1D, merger width `NR` (Fig. 9: 4), bit-clear loop | Binary spike rows of width \(P\) | Hardware of the GeMM front-end. |
| Weight-row tile, dual-port 8 KiB L-BUF, \(K'\) PEs/tile | Same reuse of one \(W\) row across column partitions | Independent of neuron model. |
| Same-ID barrier sharing one NRV submatrix | Spike G-BUF reuse across tiles | Valid **if** the only live consumer of that submatrix is the GeMM. |
| Column-major vs timestep-major as a **control** | Issue order of binary rows into L-1Ds | Fig. 13 axis still exists on the GeMM. |
| In-situ **GeMM psum** of \(P\) outputs (not yet LIF) | Holding \(P\) select-add accumulators locally vs spilling them | The **register-size** argument of CPTB (§IV-B) applies to any output-stationary \(1\times P\) psum. |

Until the checklist in §3.3 exists on the **same** port/state/backpressure contract as ordinary dense-source, “we do Gustavson” is a rename of (at best) a GP inner kernel.

### 3.2 Does **not** map 1-1 (do not smuggle these in as A)

| Gustav object | This net | Break |
|---|---|---|
| In-situ REG **= causal LIF membrane** \(v(t)=\lambda v(t-1)+\sum w S(t)\) | T10 mix sits **before** threshold and is **noncausal** | Completing the current-tick `d`-sweep of the GeMM is **not** producing \(m\). \(m\) is a compiled mix of all ten ticks (ordinary 260 add/sub, or lifting 159+35 RNE/sat). You cannot LIF after tick \(t\) and then proceed to tick \(t+1\) as if \(v\) were sequential. |
| Commit = `C = LIF(O)` after this tick’s sweep | AT-LIF is \(s=H(m-\theta)\) **after** T10, then absorb | Replacing the LIF ALU with an AT-LIF comparator does **not** insert T10, does not absorb \(\theta\) into \(W\), and does not retire residual/PED. §V-B “any neuron model” was not measured on this graph. |
| Tick-batch: all \(T\) of layer \(\ell\) before layer \(\ell+1\), sequential \(t\) | T10 is one compiled source; BN waits on **full** \(10\times 96\times 120\times 160\) stats | Layer-wise tick-batch is the wrong completion unit for r1. Projection-BN cannot see “this tick’s membrane is done.” |
| Temporal-coding skip: already-fired neurons skip remaining ticks (§V-B) | Noncausal mix uses **future** ticks of the same site | Illegal on T10. Rate-coded “fire many times” skip is also not the T10 CSE. |
| Single consumer of \(O\): neuron writes binary \(C\) | Dual consumers: **binary gate after absorb** **and** **continuous residual/PED** | Finishing NRV-skipped GeMM is not finishing r1. A site with \(s=0\) this tick can still be live for PED/I24. |
| NRV skip predicate = “P-column row all-zero **this tick**” | Residual liveness is not that bitmap | Using NRV as last-use of the **source** under-counts the residual. Union occupancy is a different predicate (candidate X, not Gustav). |
| Same-ID sync to cut repeated spike loads | Two consumers, finite ports, measured long backpressure **8088** on both source arms | A barrier that only helps the GeMM can extend PED/BN wait. Always-ready slots already moved 6938→5354 (−22.83%) with **no** long-backpressure change. |
| Potential traffic ≥80% of ex-situ GP (Fig. 4 proxy) | Residual/I24 is a **dense or quantized continuous** write, not a membrane refill | Fig. 4 is not this student’s energy, and it does not contain PED. |
| \(P=8\) chosen on ONE/SDT/SPA spike sparsity | This student’s post-absorb \(\rho\) and W density are **unmeasured in the freeze** | Importing \(P=8\) and Fig. 6 skip ratios is a different network. \(P=4\) already lost to shared-W conflicts below ~85% sparsity **in their** sweep. |
| Two-pointer dual-sparse patch | Freeze does not state W sparsity | Default A is dense-W NRV scan. Two-pointer **hurts** dense W (§VII-B). |

One-sentence summary of the mismatch:

> **CPTB/NRV copy 1-1 onto the binary GeMM after absorb; they do not copy onto the T10 mix before threshold, and they do not copy onto the residual continuous path.** Gustav’s in-situ object is a **causal LIF membrane of the GeMM’s outputs**. This net’s membrane-like object is a **noncausal compiled mix of the source**, consumed by a **binary gate** and a **continuous residual**.

### 3.3 Complete-transfer checklist (A, spike path only)

A copy that may be called “Gustav-complete” **after absorb** has to include **all** of the following, on the **same** port/state/backpressure contract as ordinary dense-source, **and** a written statement that T10 and residual are **outside** this A:

1. CPTB: \(C_k=A\cdot B_k\), \(P=N/K\) local **GeMM psums**, concatenation not reduction.
2. Those psums held in-situ in the PE (or the local equivalent), **not** ex-situ global as the GP baseline they beat — **without** renaming them “LIF membranes.”
3. NRV: row index + \(P\)-wide packed occupancy of **post-absorb \(s\)**, with measured \(P\) and \(\rho^P\) recomputed on **this** source.
4. Dual skip: missing NRV row **and** zero weight, with **this student’s** real \(W\). If \(W\) is dense, skip-2 is idle and must be reported as such.
5. Fetch‖exec PE: L-1D + merger of width `NR` (Fig. 9: 4), bit-clear loop, pipeline fill of `NR` rows.
6. Weight-row tile: dual-port shared W-BUF, one row written once, \(K'\) PEs per tile.
7. Same-ID barrier and its idle-cycle cost vs weight sparsity (Fig. 20) — **and** its interaction with the residual port.
8. Column-major **time-second** loop order on the GeMM, with timestep-major as a **control**, not the design.
9. **No** LIF-after-tick-sweep as a substitute for T10. Neuron/threshold sits **after** the compiled mix, not inside the GP PE as Gustav drew it.
10. Scheduler that slices \(D\times N\) into `NR`\(\times P\) submatrices and double-buffers G-BUFs.
11. Two-pointer **only if** \(W\) is actually sparse; dense-W config retained as control.
12. Energy/service accounting that is **not** Fig. 4: ports, bank conflicts, backpressure. Complete r1 = GeMM **and** T10 **and** gate **and** PED/BN/add.

Until 1–12 exist, the title is not GustavSNN.

---

## 4. LoAS — FTP on dual-sparse unary vs this residual path

### 4.1 Exact dataflow (Alg. 1)

- Input spikes \(A\in\{0,1\}^{M\times K\times T}\)
- Weights \(B\in\mathbb{Z}^{K\times N}\) (sparse, 8-bit in eval; Table II AvSpB = 98.2 / 98.2 / 96.8%)
- Current \(O\in\mathbb{Z}^{M\times N\times T}\), output spikes \(C\in\{0,1\}^{M\times N\times T}\)

```
for m in M:
  for n in N:
    for k in K:
      parallel-for t in T:          # spatially unrolled
        O[m,n,t] += A[m,k,t] * B[k,n]
    parallel-for t in T:            # spatially unrolled P-LIF
      C[m,n,t] = LIF(O[m,n,t])
```

**Inner-product**, `t` **innermost**, then **spatially unrolled**. Not Gustavson, not tick-batch, not PTB (PTB still sequences `t` inside a window; even `time-window=1` puts `t` between `m` and `n`, §VII).

Three goals (§III): (1) no extra refetch across timesteps; (2) few psums on `t`; (3) no \(T\times\) latency in sparsity hardware. Observation that kills GP/OP here: they produce \(T\times\) psum matrices/rows.

LIF they unroll (hard reset, scalar \(v_{th}\)):

```
O[t] = Σ_k A[t] B
X[t] = O[t] + U[t-1]
C[t] = 1 if X > v_th else 0
U[t] = τ X (1-C)                 # hard reset to 0
```

Default **T=4** (1× 12-bit pseudo-acc + 4× 10-bit correction accs). Fig. 16: TPPE area/power at T=16 is 1.37×/1.25× vs T=4; “very likely to have fewer silent neurons when T>8.”

### 4.2 Unary pack, inner-join, speculation

Pack all \(T\) bits of one pre-synaptic index into one word (Fig. 8). Silent neuron = packed `0000` (never fires in \(T\)); dropped. Bitmask + pointer **fiber** (the PDF never says “CSF”). Weights: same bitmask fiber, **column-wise**. FiberCache unified G-BUF; one compressed row fiber of \(B\) broadcast to all TPPEs.

Inner-join: AND bitmasks; **fast** prefix-sum on \(B\) (one cycle, n=128); **laggy** prefix-sum on \(A\) (16 adders, 8 cycles). Speculate packed \(A\) is **all 1s**, stream \(B\) into the pseudo-accumulator; when laggy prefix finishes, **correct** timesteps where \(A\) is 0. **Final \(O[t]\) is exact** if correction runs. Gustav’s paraphrase “approximates” refers to the speculative first step, not an allowed numerical error.

Preprocessing: mask neurons that fire **only once** in \(T\), fine-tune <5 epochs. Hardware compressor drops outputs with 0 or 1 spike. **Workload mutation**, not a free skip. Intro **lists** optical flow [28] Spike-FlowNet as an application of the **field**, **not** a LoAS benchmark.

Hardware identity of their numbers: 16 TPPEs, 256 KB 16-bank FiberCache, 32 nm, 800 MHz, 2.08 mm², 188.9 mW; global cache 65.9% power; fast prefix-sum 51.8% power / 66.7% area of a TPPE. Fig. 17: 98.2%→25% W sparsity loses ~88% performance.

### 4.3 After absorb: what FTP may copy, what the residual path forbids

**Unary identity on the spike path is now true.** After absorb, the gate/GeMM operand is \(\{0,1\}\). LoAS’s \(A\in\{0,1\}\) and “accumulate or discard the weight” are legal **on that tensor**. That is the opposite of round-1 L2.

| LoAS object | Spike path after absorb | Residual / PED / I24 | Verdict |
|---|---|---|---|
| Packed **T-bit unary** fiber | Gate bitmap across T10 **can** be packed | I24 / q24 PED are amplitudes | Fiber is A **only** on the gate. Packing the residual is identity-breaking. |
| Silent = never fires in \(T\) | Well-defined on post-absorb \(s\). At **T=10**, \(N_{\mathrm{silent}}\propto 1-\rho^T\) is the attack Gustav already used, and LoAS itself flags T>8 | A site silent for the gate can be **nonzero** on PED | FTP’s main skip (Table II packed silent 71–80% after FT, T=4 classification) need not exist here, and **does not** skip residual work. |
| Inner-join two bitmasks (spike fiber × weight fiber) | Legal **iff** \(W\) is sparse | Residual is not a bitmask | Freeze does **not** state W sparsity. LoAS identity is LTH ~97–98% W. Dense \(W\) ⇒ nothing to join; Fig. 17 says performance collapses. Keep a dense-W control. |
| Pseudo-acc all-1s + correction to exact \(O[t]\) | Legal on unary GeMM **with** correction (local bar: integer 0-diff) | Speculation on a continuous residual has no “all-1s” meaning | Dropping correction is a **kill**, not an optimization. Residual must not enter the pseudo-acc. |
| P-LIF unroll of **causal** hard-reset LIF, T=4 | Wrong operator even on the spike path: threshold is AT-LIF **after T10**, not unrolled LIF of GeMM currents | Residual is not a LIF | Unrolling LIF does not evaluate the compiled T10 graph (260 or 159+35 RNE). |
| One TPPE → one output neuron → P-LIF | One binary GeMM column is still one consumer | Second consumer + projection BN over full-domain stats | Finishing \(O[t]\) is not finishing r1. |
| Mask 1-spike neurons + FT | Optical-flow AEE gates (abs 1.259, rel +0.005) | Would mutate PED too if applied at the source | Different network. Cannot import Table II silent ratios. |
| `t` innermost of IP | May be A on the **GeMM after absorb** | T10 is a **horizontal compiled** mix, already CSE’d | Re-introducing an IP `k` loop **around T10** can undo 260→159. Local already-seen fragment: 8-bank T-parallel accumulate without fiber/correction **cut adds and raised cycles**. |
| FiberCache holding packed A as long as possible | Gate fibers | Residual needs a dense or quantized write port | One buffer format cannot be both. |

**FTP vs residual, one paragraph.** LoAS’s efficiency story is: dual-sparse unary × sparse \(W\), pack \(T\) bits, skip neurons that are 0 for **all** \(T\), speculate all-1s, correct, unroll a short causal LIF. This net’s residual path is a **continuous** tensor with a **0-diff** integer contract, a **noncausal** T10 producer, and a **second** last-use after projection-BN. There is no LoAS object that is “FTP on I24.” Applying FTP to the residual is not a variant; it is a different algebra. Applying FTP **only** to the post-absorb binary GeMM, with dense \(W\) and T=10, is a **control** that LoAS’s own Fig. 16b / Fig. 17 and Gustav’s Fig. 22 already predict will lose the silent-neuron skip.

### 4.4 Complete-transfer checklist (A, LoAS on the spike path only)

1. IP loop with **`t` innermost**, spatially unrolled over **actual T=10**, not 4.
2. Packed-across-T fiber for the **unary/gate** view only; silent-drop **only** where the numerical function allows (not on PED).
3. Matching column-wise weight fiber **if** \(W\) is sparse; dense \(W\) keeps a dense control.
4. Inner-join: AND, **fast** prefix on \(W\), **laggy** prefix on source, FIFOs as in Fig. 10.
5. **Pseudo-acc + T correction accs**, proof that corrected \(O\) matches the integer model (0-diff on gates; residual not in this unit).
6. Operator after \(O\): **not** P-LIF. T10 remains the compiled graph **before** threshold. Writing “unrolled LIF = T10” is a failed copy.
7. FiberCache (or iso-capacity banked scratchpad) for **gate** fibers; residual/PED still has a dense or q24 port.
8. Dual-consumer completion: gate fiber **and** PED/BN/add. Not in LoAS; required before claiming this net is “running LoAS.”
9. No 1-spike masking unless a new student is trained and passes AEE gates.
10. Iso-16-PE (or iso-port) sequential-`t` IP/OP/Gust **controls**, which is how LoAS itself claims speedup.
11. Service model with ports and backpressure, not only AC-count.

Until 1–11 exist, “time-parallel accumulators” is not LoAS.

---

## 5. Candidate X (after A is complete; not a reskin)

A is now **legal** on the spike path. Reviewers will treat “we also CPTB/NRV” or “we also FTP” as missing relative prior. X has to be an object **neither** paper has, using **only** this freeze.

Honest X (pick **one** mechanism, not a zoo):

**X1. Dual-retire across the absorb cut.**  
Keep CPTB/NRV (or LoAS fiber) as A on the **binary gate GeMM**. Change the commit: a column / fiber / source slot stays live until **gate consumer and residual/PED/BN** have both completed. Gustav commits at `LIF(V_θ)` after one tick sweep. LoAS commits at P-LIF of one output neuron. The differential is the **retire contract**, not Gustavson and not FTP.

**X2. Union occupancy, two algebras.**  
Compile a P-wide (or T-wide) occupancy from \(\mathrm{support}(s_{\mathrm{after\,absorb}})\cup\mathrm{support}(\mathrm{PED})\) on the **already-compiled** T10 source graph (ordinary 260 or lifting 159+35 RNE), not from unary spikes alone. Same NRV/fiber machinery as A; the **predicate** is the X. If the union is dense, skip dies — that is a kill, not a reason to rename GP.

**X3. Noncausal T10 as the operator between GeMM and threshold.**  
Do not put causal in-situ LIF in the PE, and do not unroll P-LIF. Evaluate the CSE’d T10 graph so that T-parallelism (if any) does **not** reopen CSE, and so RNE/sat intermediates stay on the same two-stage writeback that already moved 6938→5354 always-ready slots with **8088** long backpressure unchanged. The X is **preserving CSE under a dual-consumer last-use**, with post-absorb binary GeMM as the copied A.

Not X (these are A, or another paper, or a rename):

- In-situ psum, NRV, NR4 merger, shared W-BUF, same-ID barrier, column-major order, two-pointer
- FiberCache, laggy prefix, pseudo-acc, silent-neuron pack, LTH dual-sparse, “fully temporal parallel”
- Prosperity pattern reuse / prefix psum (parallel prior; Gustav **compared** against it, did not include it)
- “Compatible with any neuron” / “replace LIF with AT-LIF”
- “Gustav + Prosperity,” “Gustav + LoAS,” “FTP + CPTB” as a title-level fusion
- Non-absorbable int8 payload (forbidden identity)

---

## 6. Controls

- Ordinary dense-source/raw vs lifting, **same** CPTB/NRV **or** same FTP, **same** ports/state/backpressure. Lifting raw AEE already **fails** relative +0.005 (1.232979368 vs 1.219801338).
- Column-major vs timestep-major vs ordinary time-row (Gustav Fig. 13) **on the GeMM only**.
- In-situ \(P\)-psum vs ex-situ global (their Fig. 4/18 reason for existing) — still a control for the GeMM, not a claim about T10.
- Gustav CPTB vs LoAS FTP as **parallel** A’s, never parent/child.
- Prosperity product-sparsity as a **third** parallel control on the same binary GeMM, never a hidden layer of Gustav.
- Single-consumer (gate only, or PED only) vs dual-retire (X1).
- Binary NRV vs union occupancy vs dense (no skip).
- Dense-\(W\) default vs two-pointer / vs LoAS bitmask-\(W\) (only if \(W\) is actually sparse).
- Sequential-`t` IP (SparTen-SNN analogue) vs FTP, same ports.
- Complete pseudo-acc+correction vs “parallel acc without correction.”
- With vs without 1-spike masking (must re-measure AEE).
- Frozen-running-BN vs full-domain batch stats (local BN fact; absent from both papers).
- Same-state budget: do not buy extra W banks to hide CPTB conflicts (\(P=4\) lesson).
- PTB-style `t` between `m` and `n` vs LoAS innermost `t` (§VII).

---

## 7. Kill gates

Use `PROBLEM.md` numbers, not Gustav GOPS/W or LoAS 8.51×.

- **Accuracy:** valid825 AEE absolute ≤ 1.259; relative vs ordinary dense-source/raw (1.219801338) **≤ +0.005**. Lifting raw is already +0.013178 (relative fail). Any mapping that changes the numerical function re-enters this gate. Integer gates / I24 / PED q24 must stay **0-diff** vs model on captured windows.
- **Service:** complete r1 → gate + PED + projection-BN, **same port / same state / same backpressure**, net service **≥ 15%**. Do not add the two historical tables (SIMD source −22.83% at 6938→5354; integer consumer −5.78% at 758777→714889). Long backpressure already 8088 on both source arms — a skip that only helps always-ready slots is not a pass.
- **Novelty kill:** if the only sentence is GP / NRV / in-situ membrane / “we also Gustavson” / “we also FTP” / “Gustav+Prosperity,” reject as reskin (A, not X).
- **Mechanism kill:** if complete A (Gustav §3.3 or LoAS §4.4) plus ordinary order already matches the candidate; if union-NRV density makes skip ≈ 0; if dual-retire state grows the **same-state** budget used by the ordinary arm; if innermost-`t` undoes CSE (159+35 → more ops) or needs extra banks vs the frozen two-stage SIMD point — stop that X, keep A as denominator.
- **Identity kill:** putting residual/PED into a T-bit unary fiber, dropping LoAS correction, treating T10 as P-LIF or as causal in-situ LIF, or reviving non-absorbable int8.
- **Do not** multiply PE GOPS/W into FPS; do not quote 28 nm 1.34 mm² or 32 nm 2.08 mm² as this student’s PPA; do not quote Fig. 4 as measured chip energy.

---

## 8. Parallel priors (do not write Gustav = Prosperity)

```
                    product sparsity (pattern reuse)
                    Prosperity HPCA'25  ────────┐
                                                │  compared in Gustav §VI-C4
                                                │  (simulator, not a layer)
Gustavson GP ──── in-situ + CPTB + NRV + NR4 ── GustavSNN HPCA'26
                                                │  compared as TP baseline
IP + FTP + fiber + pseudo-acc/correction ────── LoAS MICRO'24
                     (explicitly: Gust unsuitable)
```

| Claim | Supported by text? |
|---|---|
| GustavSNN **cites** Prosperity | Yes, [42], §VI-C4 |
| GustavSNN **implements** Prosperity | **No** |
| GustavSNN **=** Prosperity + Gustavson | **No** |
| GustavSNN **cites** LoAS | Yes, [47], as TP / FTP LIF |
| GustavSNN **includes** LoAS FTP | **No**; CPTB is Design Choice #2, the alternative |
| LoAS **uses** Gustavson | **No**; §III says Gust is not suitable |
| LoAS **evaluates** Gamma-SNN (sequential Gust) | Yes, as a **baseline it beats** |
| After absorb, Prosperity **and** Gustav **and** LoAS are all legal **A** on the binary GeMM | **Yes** — three **parallel** complete priors, not a stack |
| Local title “Gustav = Prosperity + GP” | **Forbidden** |

Transfer implication: complete **A** copies **one** denominator (or runs them as **controls of each other**). Title-level X cannot be “we fused the names.” After absorb, the reviewer objection is no longer “your spikes are analog.” It is “you copied a binary-GeMM accelerator and left T10 + residual + BN + 8088 unaddressed.”

---

## 9. Quoted evidence

### GustavSNN

> “we propose a novel scheduling approach along with its hardware architecture for Gustavson product (GP)-based SNN acceleration. We introduce a column-parallel tick-batch (CPTB) dataflow that partitions the spike matrix into multiple submatrices and processes each submatrix of a timestep in parallel while maintaining tick-batch semantics.”
> — Abstract

> “our design leverages the overlooked advantages of Gustavson product (GP), which repeatedly reads sparse 1-bit spike rows but reduces the data traffic on fetching multi-bit weights and neuron potentials.”
> — §I

> “without imposing any constraints on the choice of SNN architectures or neuron models.”
> — §I (compatibility claim, not a measured AT-LIF / T10 result)

> “LoAS [47] uses a spike compression format by identifying silent neurons (that never fire) and approximates the computation by initially assuming that spikes occur at every tick, enabling FTP execution for LIF neuron-based SNNs. Then, it is followed by a correction stage for the true neuron potential update.”
> — §III-B2 (Gustav’s paraphrase; LoAS’s own text is speculative all-1s inner-join, then **exact** correction, then unrolled LIF)

> “If more complex neuron models (e.g., adaptive ones) are used, they can encounter difficulties due to long critical paths”
> — §III-B2 (reason to reject FTP, not an AT-LIF benchmark)

> “Therefore, instead of pursuing dual-sparse compression, we focus on efficiently supporting high spike sparsity with quantization”
> — §VI-A

> “In addition, we also compare with Prosperity (PR) [42], which efficiently reuses computations exploiting product sparsity. […] For a fair comparison, we use its publicly available simulator of PR and CACTI-7.0 for memory modeling.”
> — §VI-C4 (**comparison**, not a stacked layer)

> “We achieved 1.18–1.71× higher energy efficiency than PR.”
> — §VI-C4

> “By using NRV, our proposal skips entire spike row vectors when all P neurons in a row remain inactive at a given timestep (temporal sparsity). In contrast, the TP-based LoAS architecture … captures spatial sparsity by skipping only neurons that are silent across the entire timestep.”
> — §VII-C

> “we implement LIF neurons, but the neuron computation logic inside the PE can be replaced with any target model without altering the overall architecture.”
> — §V-B

Prosperity is bibitem [42] Wei et al., HPCA 2025. No sentence composes Gustavson **with** that product-sparsity engine.

### LoAS

> “The main challenge is that processing timesteps, a natural property of SNNs, introduces an extra loop to ANN spMspM, leading to longer latency and more memory traffic. To address the problem, we propose a fully temporal-parallel (FTP) dataflow”
> — Abstract

> “Our second observation is that both OP and Gust dataflow are not suitable for dual-sparse SNNs since they oppose goal (2).”
> — §III

> “We first choose to position the t-dim at the innermost of the IP dataflow, as given in Algorithm 1. … we fully parallelize the t-dim”
> — §III

> “We call this compressed row: a fiber [34], [62].”
> — §IV-A (no “CSF” string)

> “This mechanism opportunistically presumes the matched non-zero value of fiber-A is all 1s (pre-synaptic neuron fires at all timesteps) … we need a mechanism to ensure that the accumulation results are correct.”
> — §IV-C

> “we spatially unroll the LIF operations so that the output spikes for all timesteps will be generated at once.”
> — §IV-D

> “We set the default timesteps T to 4 across all experiments.”
> — §V

> “it is very likely to have fewer silent neurons when we have even larger timesteps (> 8). This is one of the challenge that LoAS needs to face when scaling up on the number of timesteps.”
> — §VI-B

> “SNNs have been widely used in computer vision tasks, such as image classification [46], [56], optical flow estimation [28], …”
> — §I (application list; **not** LoAS eval)

---

## 10. What this remap changes relative to round-1 L2

| Round-1 L2 (continuous θg) | This remap (absorb identity) |
|---|---|
| NRV cannot skip because θg is an amplitude | NRV **can** skip on post-absorb \(s\in\{0,1\}\). That is now **A**, not a hole. |
| Inner kernel is not AND/add | After absorb, it **is** AND/add on the spike path. |
| LoAS pack is illegal on θg | LoAS pack is legal on the **gate** bitmap; still illegal on residual. |
| Hole = “binary accelerators do not apply” | Hole = **T10-before-threshold + dual-retire of binary gate and continuous residual + full-domain BN + 8088 backpressure** |
| X lived in “make occupancy work on amplitudes” | X lives in last-use / union occupancy / CSE-preserving T10 **after** a complete binary-GeMM prior is copied |

**Strongest shared control:** ordinary dense-source/raw, same ports, same state, same 8088-class backpressure, **both** consumers retired, actual BN stats, integer 0-diff, **plus** a complete Gustav **or** LoAS A on the binary GeMM. A local win against a sequential scatter that those papers would also have beaten is a weak denominator.

Do not quote 11.8×, 8.51×, 3.68×, 1.34 mm², 2.08 mm², 28 nm, or 32 nm as this student’s result.
