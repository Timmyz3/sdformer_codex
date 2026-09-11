# L2 deep-read: GustavSNN (HPCA 2026) and LoAS (MICRO 2024)

Date: 2026-09-11. Local texts only. This note is a transfer contract, not a novelty claim.

Frozen local identity (`PROBLEM.md`): event-camera 2D optical flow, DSEC valid825 AEE; student Motion C12 / H67 / ep34; **ATLIF continuous threshold amplitude θg, not binary spikes**; dual consumers after source (spike/gate path and continuous residual/PED path) plus native projection conv + BN + residual add; **noncausal T10** at this PSN; expensive region historically patch-embed residual r1 + T10 PSN.

Sources read in full:

- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/GustavSNN_HPCA2026_public_mirror.txt` (1275 lines; Hwang, Lee, Koo, Kung; HPCA 2026; DOI 10.1109/HPCA68181.2026.11408587)
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2407.14073.txt` (1109 lines; Yin, Kim, Wu, Panda; MICRO 2024; arXiv:2407.14073v3, 1 Sep 2024)

---

## 0. How the two papers relate (do not stack them)

**They are parallel priors, not a layered stack.**

- LoAS (MICRO 2024) is an **inner-product + fully temporal-parallel** accelerator for **dual-sparse** (sparse spike **and** sparse weight) LIF SNNs. It **rejects** outer-product and Gustavson for this setting because the extra `t` loop multiplies partial-sum traffic (§III).
- GustavSNN (HPCA 2026) is a **Gustavson-product + column-parallel tick-batch** accelerator for **spike-sparse, typically dense-weight** SNNs. It **rejects** FTP as the right answer for long `T` / temporal coding, and treats LoAS as the **TP baseline**, not as a layer it includes.
- Prosperity (HPCA 2025) appears in GustavSNN **only as an energy-efficiency comparison** (product-sparsity reuse; public simulator). Gustav does **not** implement Prosperity’s pattern reuse, does **not** inherit its pruning model, and does **not** say “Gustav = Prosperity + Gustavson.”

Forbidden sentence unless a later paper actually writes it: *“GustavSNN = Prosperity + Gustavson.”* The GustavSNN text never says that.

Disagreement that matters for transfer:

| Axis | LoAS | GustavSNN |
|---|---|---|
| SpMM family | IP; Gust/OP called unsuitable for dual-sparse SNN | In-situ GP; IP-FTP and OP lose on potential traffic |
| Time | FTP: `t` innermost, spatially unrolled | Tick-batch sequential `t`; column-major time-second |
| Sparsity identity | Dual-sparse (LTH ~97–98% W, packed silent-neuron A) | Spike sparsity + quantization; **explicitly not** dual-sparse as the main path |
| Skip unit | Silent neuron across **all** `T` (spatial) | All-zero **P-column row at one tick** (temporal) |
| Neuron | Simple LIF, hard reset, unrolled over small `T` | Sequential LIF (subtractive reset) in PE; claims swappable neuron logic |
| Role of the other | N/A (2024; no GustavSNN) | LoAS = TP [47]; Prosperity = PR comparison [42] |

---

## 1. GustavSNN — exact dataflow

Title: *GustavSNN: Unleashing the Power of Gustavson’s Algorithm on SNN Acceleration with Column-Parallel Tick-Batch Dataflow.*

### 1.1 Loop nest and matrices

Matrices as in Alg. 1 / Fig. 3 (SNN-GP):

- Weight `A ∈ R^{M×D}` (multi-bit; eval uses 8-bit)
- Spike `B ∈ {0,1}^{D×N×T}`
- Output spike `C ∈ {0,1}^{M×N×T}`
- Column partition: `P = N/K`, `B = [B_0 | … | B_{K-1}]`, `C_k = A · B_k`, `C` is concatenation along columns. Columns of `B` are **independent**.

**Algorithm 1 — Column-Parallel Tick-Batch Gustavson Dataflow** (reconstructed from §IV-C; `t` is outside `d`, LIF after the `d` sweep of that tick):

```
parallel-for m in [0, M):          # tile-level, one weight row
  parallel-for k in [0, K):        # PE-level, one column partition
    for t in [0, T):
      for d in [0, D]:
        for n in [p, p+P):
          O_k[m,n,t] += A[m,d] * B_k[d,n,t]
      C_k[m, p:p+P-1, t] = LIF(O_k[m, p:p+P-1, t])
```

This is **not** FTP and **not** timestep-major. §V-B names it **column-major time-second**: a PE sweeps the entire current-tick column partition (all `d` of that tick) so the `P` local potentials are complete, then LIF, then the next tick; after all `T`, the tile takes the next column partition.

Timestep-major (explicitly rejected for high sparsity): process all `T` for given `NR` spike rows before the next rows, so each weight row is loaded once over `T`. Occupancy of an L-1D then jumps from `1-ρ^P` to `1-ρ^{PT}` (§V-B). Fig. 13: at sparsity >95%, column-major spike-register utilization >60% vs timestep-major <25%.

### 1.2 CPTB (column-parallel tick-batch)

Problem CPTB exists to solve: in-situ GP wants a `1×N` potential register **inside the PE**. For `N=1024`, 8-bit, that is ≥1 KiB/PE and `N` varies by layer, so a fixed `N'` wastes area or spills (§IV-B).

CPTB: partition `B` along **columns** (output neurons). Each PE holds only `P = N/K` potentials in local REG and updates them **in-situ** (no global membrane buffer, no ex-situ merge). Tick-batch **layer-wise** semantics are kept: finish all ticks of this layer before the next layer (Fig. 1 right vs SpinalFlow left vs temporal-parallel middle).

In-situ vs ex-situ (Fig. 4, flat-memory energy count, **not** the system result): ex-situ GP spends ≥80% of that proxy on moving potentials every tick. In-situ GP is lowest across 80–100% sparsity at `T=4` and `T=16` when weights are dense. Dual-sparse is a second panel; extreme dual sparsity is called “uncommon in typical SNNs without explicit weight pruning.”

### 1.3 NRV (non-zero row vector)

After CPTB, a submatrix has `P` columns. If a row is all zeros, skip the fetch. Probability a row is all-zero scales as `ρ^P` (§IV-D). Fig. 6 on ONE/ResNet-18: **P=8 skips ~80% of rows; P=32 skips ~58%.**

NRV payload per kept row of submatrix `k`:

- row index
- **P-bit naive {0,1} vector** (not COO column indices)

Compared to COO/CSR/bitmap: better than COO/CSR except at very low sparsity; 28% less memory than COO and 42% than CSR in their sweep; SPA embedding at 76% sparsity prefers naive bitmap, later stages >98% prefer NRV (up to 5× vs naive) (Fig. 8).

Two skip sites in the PE (§IV-E):

1. **NRV construction / weight skipping:** weights whose row index is absent from NRV are not fetched.
2. **Row skipping at fetch:** if the fetched weight is 0, skip that spike row.

### 1.4 NR4 PE (fetch | exec, L-1D, merger)

§IV-E / Fig. 9: PE is a two-stage pipeline (fetch, execution). **`NR` = merger-tree width = number of L-1Ds.** Fig. 9 uses **`NR = 4`**. Fig. 12 walk-through uses `NR=2` only as a drawing. Default silicon (Table II) restates tiles/PEs/`P`, not `NR`; the designed PE in Fig. 9 is NR=4.

Fetch: take up to `NR` NRV rows, skip missing indices and zero weights, fill `NR` P-bit registers.

Exec:

- each of `NR` L-1Ds emits a non-zero column index in parallel (Fig. 10)
- merger tree emits the **smallest** column index; L-1D id selects which weight in `W REG` to add
- selected bit is cleared (`SPIKE -= (1 << Col_idx)`); repeat until the `NR` vectors are empty
- if the new col-idx equals the previous, accumulate into the same potential; else write out and start a new partial sum
- after the current-tick submatrix row is consumed: compare `P` potentials to `V_θ`, emit spikes, neuron-model update

Same-ID barrier (§V-A, §VII-A): PEs with the same partition id **across tiles** share one NRV submatrix from spike G-BUF and **synchronize** before the next submatrix, to cut repeated spike loads. Cost: idle cycles if weight sparsity unbalances tiles. Unpruned: ~1% idle. Weight sparsity <80%: idle <10%.

### 1.5 Weight-row tiled microarchitecture (default config)

Table II / Fig. 11 / Fig. 17:

- 8 tiles, 8 PEs/tile, **P = 8**, 8-bit weights, 1 GHz, 1.1 V, 28 nm FDSOI
- per tile: dual-port shared weight L-BUF **8 KiB** (one weight row, written once)
- spike G-BUF: 8 × 1 KiB; weight G-BUF: 128 KiB; LPDDR3 6.4 GB/s
- total **1.34 mm²** (post-layout tile + CACTI buffers)
- no inter-tile / inter-PE reduction network for psums (concatenation, not merge)

`P` trade-off (§VI-B): smaller `P` → more row skips and more PEs per area, but more NRV fetches and **shared W-BUF conflicts**. `P=4` loses to `P=8` below ~85% spike sparsity; `P=4` energy efficiency drops because of those conflicts (Fig. 16). Chosen `P=8`.

Two-pointer intersection (§VII-B) is **not** the default pipeline. Naive NRV-index scan becomes skip-dominant when weight density <40%, and naive skip is **worse than LoAS** above 80% weight sparsity. Two-pointer (advance the smaller of NRV-row-index pointer and nonzero-W-index pointer) recovers dual-sparse; **on dense weights it is slower than the dense-optimized design.** Future work: a flexible engine that switches skip scheme. Do not transfer two-pointer as if it were Gustav’s always-on A.

### 1.6 Neuron in the PE

Implemented LIF (§II-A, subtractive reset):

```
v_j(t) = λ v_j(t-1) + Σ_i w_ij S_i(t)
S_j(t)=1 and v_j(t) ← v_j(t)-V_θ   when v_j(t) ≥ V_θ
```

§V-B: “we implement LIF neurons, but the neuron computation logic inside the PE can be replaced with any target model without altering the overall architecture.” That is an **architectural compatibility sentence**, not an evaluation of ATLIF, continuous θg, noncausal T10, or optical flow.

### 1.7 What GustavSNN evaluates (identity of the numbers)

Table I:

| Model | Arch | Coding | T | Sparsity | Accuracy |
|---|---|---|---|---|---|
| SEW | ResNet-18 | rate | 4 | 87.5% | 63.18% |
| ONE | ResNet-18 | temporal | 24 | 95.2% | 68.73% |
| SDT | ViT | rate | 4 | 88.9% | 77.07% |
| SPA | Swin-T | temporal | 40 | 98.4% | 80.0% |

Headline numbers (do not mix “up to” and “average”):

- Abstract: **up to 11.8×** GOPS/W vs naïve GP; **1.43×** vs “state-of-the-art SNN accelerators”
- §VIII: **11×** vs direct GP SpMM deployment; **1.43× on average** vs SOTA
- vs TP/LoAS: **1.52–2.25×** GOPS/W
- vs Prosperity (comparison only): **1.18–1.71×** GOPS/W (Fig. 19 callouts 1.18 / 1.63 / 1.21 / 1.71 on SEW / ONE / SDT / SPA)

Baselines implemented as TS (SpinalFlow tick-batch), TP (LoAS-style four ticks in parallel), OP, naïve GP (GAMMA-like, **ex-situ** potentials), plus Prosperity simulator+CACTI. All iso-64-PE, same global buffer. Neuromorphic chips (Loihi, SpiNNaker) are **explicitly not** compared.

Fig. 4 energy is **flat-memory, unbounded bandwidth, no bank conflicts, no timing** — they say so, and point to §VI for system results. Do not quote Fig. 4 as measured chip energy.

### 1.8 Identity assumptions (must hold for a faithful copy)

1. Activations are **binary spikes {0,1}**. Weights are multi-bit. The inner kernel is `O += A[m,d] * B[d,n,t]` with `B∈{0,1}` (AND/add, no multiplier required).
2. One **causal** membrane per output neuron; LIF (or a drop-in sequential neuron) after the current-tick `d`-sweep. Tick-batch: all `T` of layer ℓ before layer ℓ+1.
3. **Single consumer** of `O`: the neuron model that writes binary `C`. No second continuous residual/PED consumer, no BN, no shortcut add in the PE contract.
4. Skip predicate is **binary**: NRV row exists iff some of `P` columns spiked **this tick**; second skip iff `W=0`.
5. Primary sparsity is **spike** sparsity. Dual-sparse is a discussion-section patch (two-pointer), not the evaluated identity. “Instead of pursuing dual-sparse compression, we focus on efficiently supporting high spike sparsity with quantization” (§VI-A).
6. `T` may be 4–40 and coding may be rate or temporal; FTP’s silent-neuron skip is argued to collapse as `T` grows (`N_silent ∝ 1-ρ^T`).
7. Classification / converted-transformer SNNs (SEW, ONE, SDT, SPA), 8-bit weights, 28 nm FDSOI 1 GHz RTL+PnR of **their** tile.

### 1.9 What GustavSNN does **not** claim

- Does not claim optical-flow, event-camera, DSEC, or any dense residual reconstruction task.
- Does not claim **continuous-valued** neuron output or threshold **amplitude** θg. Spikes are binary events; `V_θ` is a fire threshold.
- Does not claim dual-consumer lifetime (gate vs residual/PED), native projection BN over a full spatial-temporal domain, or noncausal T10 PSN.
- Does not claim to **stack** Prosperity product-sparsity, LoAS FTP, or LTH dual-sparse pruning. Prosperity and LoAS are **comparisons**.
- Does not claim that “any neuron model” was **measured**. Adaptive neurons are used as a **reason to reject FTP** (long critical path), not as a GustavSNN benchmark.
- Does not claim dual-sparse is free: naive skip loses to LoAS at high weight sparsity; two-pointer is a special case; unpruned SNNs are the main tables.
- Does not claim Fig. 4 flat-memory counts are system energy; does not compare Loihi/SpiNNaker; 11.8× is vs **naïve GP**, 1.43× is vs SOTA SNN accelerators.
- Does not claim a foundry tapeout; 1.34 mm² is 28 nm FDSOI post-layout + CACTI.

### 1.10 Hole vs local continuous-θg dual-consumer T10 residual

Local source is not `B∈{0,1}^{D×N×T}` feeding a LIF. It is a **compiled T10 PSN** (ordinary 260 add/sub per vector, or lifting 159 add/sub + 35 RNE/sat) whose output is **continuous θg**, then **two consumers** (gate path and residual/PED path) plus projection+BN+add. Integer gates / I24 / PED q24 already match the model with **0 difference** on captured windows. BN on captured students uses **actual batch statistics over 10×96×120×160**, not frozen running stats.

Where Gustav’s objects stop matching:

| Gustav object | Local object | Break |
|---|---|---|
| NRV = P-bit {0,1} row | θg is an amplitude, not a bit | `ρ^P` all-zero-row skip is not the skip predicate of a continuous vector, and “no spike this tick” does not imply “PED does not need this source” |
| In-situ REG = LIF membrane of P neurons | State that must survive until **both** consumers finish, including residual add | Retiring after `LIF(V_θ)` drops the PED/BN consumer |
| Column-major time-second + causal LIF | **Noncausal T10** | Cannot use “already-fired neurons skip remaining ticks” (§V-B temporal-coding skip). T10 is not a sequential `v(t)=λv(t-1)+…` chain |
| Source ∩ W skip | Dual-consumer **union** of live sources | A source that is zero for the gate can still be live for PED (and conversely after lifting) |
| Single binary `C` to next layer | Gate + continuous residual + projection BN | Completing SpMM/GP is not completing r1 |
| Same-ID sync to reuse one spike submatrix | Two consumers, finite ports, already-measured long backpressure 8088 | Extra sync that only helps one consumer can extend the other consumer’s wait |
| “Replace neuron logic” | ATLIF continuous θg + T10 PSN | Replacing LIF ALU with ATLIF does not give T10, dual lifetime, or BN stats |

Already-seen local failure modes that are **not** Gustav’s fault: category-name layouts, private byte indices, static share that cuts adds but raises cycles. Incomplete local 2×4 GP slice ≠ complete transfer of §IV–V.

### 1.11 Complete-transfer checklist (A, before any X)

A copy that may be called “Gustav-complete” on this net has to include **all** of the following, on the **same** port/state/backpressure contract as ordinary dense-source:

1. CPTB: `C_k = A·B_k`, `P=N/K` local state, concatenation not reduction.
2. In-situ state in the PE (or the local equivalent of potential REG), **not** ex-situ global membrane as the GP baseline they beat.
3. NRV: row index + P-wide packed occupancy, with measured `P` (their default 8) and the `ρ^P` skip math recomputed on **this** source, not ONE’s Fig. 6.
4. Dual skip: missing NRV row **and** zero weight, with **this student’s** real W traffic. Do not import historical all-nonzero W1/W2 from an older FP32 student as if it were the freeze. If W is dense, skip-2 is idle and must be reported as such.
5. Fetch\|exec PE: L-1D + merger of width `NR` (Fig. 9: 4), bit-clear loop, pipeline fill of `NR` rows.
6. Weight-row tile: dual-port shared W-BUF, one row written once, K′ PEs per tile.
7. Same-ID barrier and its idle-cycle cost vs weight sparsity (Fig. 20).
8. Column-major **time-second** loop order, with timestep-major as a **control**, not the design.
9. Neuron/commit **after** the current-tick (or current T10) sweep of assigned columns — plus a **written** mapping of that commit onto dual consumers (this mapping is **not** in the paper; without it the copy is still incomplete for **this net**).
10. Scheduler that slices `D×N` into `NR×P` submatrices and double-buffers G-BUFs.
11. If and only if weights are actually sparse: two-pointer as a **separate** config, with the dense-W config retained as control (paper: two-pointer **hurts** dense W).
12. Energy/service accounting that is **not** Fig. 4 flat-memory: ports, bank conflicts, backpressure. Their own §III-A / §VI split.

Until 1–12 exist, “we do Gustavson” is a rename of (at best) a GP inner kernel.

### 1.12 Candidate X that is **not** “we also do Gustavson”

After A is complete, the only hole large enough for a TCAS-II mechanism is **dual-lifetime continuous state on noncausal T10**, not another GP PE.

Honest X (one mechanism, not a zoo):

- **X-G1. Dual-retire P-column context.** Keep CPTB’s `P` local registers as A. Change the commit rule: a column is live until **gate consumer and residual/PED consumer** have both completed, including projection-BN add. Gustav commits at `LIF(V_θ)` after one tick sweep. The differential is the **retire contract**, not Gustavson.
- **X-G2. Union occupancy vs binary NRV.** Compile a P-wide occupancy from `support(gate) ∪ support(PED)` on the **already-compiled** T10 source graph (ordinary 260 or lifting 159+35 RNE), not from `{0,1}` spikes. Same NRV machinery as A; the predicate is the X. If the union is dense, NRV skip dies — that is a kill, not a reason to rename GP.

Not X: in-situ psum, NRV, NR4 merger, shared W-BUF, same-ID barrier, column-major order, two-pointer, “compatible with any neuron,” or Prosperity pattern reuse.

### 1.13 Controls

- Ordinary dense-source/raw vs lifting, **same** CPTB/NRV/ports/backpressure.
- Column-major vs timestep-major vs ordinary time-row (paper’s own Fig. 13 axis).
- In-situ P-state vs ex-situ global (their Fig. 4/18 reason for existing).
- Single-consumer (gate only, or PED only) vs dual-retire (X-G1).
- Binary NRV vs union occupancy vs dense (no skip).
- Dense-W default vs two-pointer (only if W is actually sparse).
- Prosperity **as a parallel control** (product-sparsity reuse), never as a hidden layer of Gustav.
- Frozen-running-BN vs full-domain batch stats (local BN fact; absent from Gustav).
- Same-state budget: do not buy extra W banks to hide CPTB conflicts (`P=4` lesson).

### 1.14 Kill gate

Use `PROBLEM.md` numbers, not Gustav’s GOPS/W.

- **Accuracy:** valid825 AEE absolute ≤ 1.259; relative vs ordinary dense-source/raw (1.219801338) **≤ +0.005**. Lifting raw already at 1.232979368 (**+0.013178**, relative fail). Any Gustav mapping that changes the numerical function re-enters this gate.
- **Service:** complete r1→gate+PED+projection-BN chain, **same port / same state / same backpressure**, net service **≥ 15%**. Do not add the two historical tables (SIMD source −22.83% at 6938→5354 and integer consumer −5.78% at 758777→714889). Long backpressure already 8088 on both source arms — a skip that only helps always-ready slots is not a pass.
- **Novelty kill:** if the only sentence is GP / NRV / in-situ membrane / “we also Gustavson,” reject as reskin (A, not X).
- **Mechanism kill:** if complete A (checklist 1–12) plus ordinary order already matches the candidate, or union-NRV density makes skip ≈ 0, or dual-retire state grows the **same-state** budget used by the ordinary arm, stop that X and keep A as denominator.
- **Do not** multiply PE GOPS/W into FPS; do not quote 28 nm 1.34 mm² as this student’s PPA.

### 1.15 Section-quoted evidence (GustavSNN)

> “we propose a novel scheduling approach along with its hardware architecture for Gustavson product (GP)-based SNN acceleration. We introduce a column-parallel tick-batch (CPTB) dataflow that partitions the spike matrix into multiple submatrices and processes each submatrix of a timestep in parallel while maintaining tick-batch semantics.”
> — Abstract

> “our design leverages the overlooked advantages of Gustavson product (GP), which repeatedly reads sparse 1-bit spike rows but reduces the data traffic on fetching multi-bit weights and neuron potentials.”
> — §I

> “without imposing any constraints on the choice of SNN architectures or neuron models.”
> — §I (compatibility claim, not a measured ATLIF result)

> “LoAS [47] uses a spike compression format by identifying silent neurons (that never fire) and approximates the computation by initially assuming that spikes occur at every tick, enabling FTP execution for LIF neuron-based SNNs. Then, it is followed by a correction stage for the true neuron potential update.”
> — §III-B2 (Gustav’s **paraphrase** of LoAS; LoAS’s own text is speculative all-1s **inner-join**, then exact correction, then unrolled LIF — see §2.15)

> “FTP dataflow is efficient at running rate-coded SNNs with a short timestep. … LoAS [47] selects simple LIF neurons specifically to enable synchronized spike generation over multiple timesteps … If more complex neuron models (e.g., adaptive ones) are used, they can encounter difficulties due to long critical paths”
> — §III-B2

> “Therefore, instead of pursuing dual-sparse compression, we focus on efficiently supporting high spike sparsity with quantization”
> — §VI-A

> “In addition, we also compare with Prosperity (PR) [42], which efficiently reuses computations exploiting product sparsity. It identifies repeating spike patterns or sub-patterns to eliminate repeated computations in SNNs. For a fair comparison, we use its publicly available simulator of PR and CACTI-7.0 for memory modeling.”
> — §VI-C4 (**comparison**, not a stacked layer)

> “We achieved 1.18–1.71× higher energy efficiency than PR.”
> — §VI-C4

> “By using NRV, our proposal skips entire spike row vectors when all P neurons in a row remain inactive at a given timestep (temporal sparsity). In contrast, the TP-based LoAS architecture … captures spatial sparsity by skipping only neurons that are silent across the entire timestep.”
> — §VII-C

> “we implement LIF neurons, but the neuron computation logic inside the PE can be replaced with any target model without altering the overall architecture.”
> — §V-B

Prosperity reference is bibitem [42] Wei et al., HPCA 2025, “Prosperity: Accelerating spiking neural networks via product sparsity.” No sentence in the body composes Gustavson **with** that product-sparsity engine.

---

## 2. LoAS — exact dataflow

Title: *LoAS: Fully Temporal-Parallel Dataflow for Dual-Sparse Spiking Neural Networks.* Venue: MICRO 2024. Artifact note in paper: `https://github.com/RuokaiYin/LoAS`.

### 2.1 Loop nest and matrices

§II-A, Alg. 1:

- Input spikes `A ∈ U^{M×K×T}`, `U={0,1}`
- Weights `B ∈ Z^{K×N}` (sparse, 8-bit in eval)
- Current `O ∈ Z^{M×N×T}`, output spikes `C ∈ U^{M×N×T}`

**Algorithm 1 — FTP:**

```
for m in M:
  for n in N:
    for k in K:
      parallel-for t in T:          # spatially unrolled
        O[m,n,t] += A[m,k,t] * B[k,n]
    parallel-for t in T:            # spatially unrolled P-LIF
      C[m,n,t] = LIF(O[m,n,t])
```

This is **inner-product** with `t` **innermost** and then **spatially unrolled**. It is **not** Gustavson, **not** tick-batch SpinalFlow, **not** PTB time-window (partial-T).

§III three goals: (1) no extra refetch across timesteps; (2) few psums on `t`; (3) no `T×` latency in sparsity hardware. Observations:

- Any dataflow that does **not** put `t` innermost refetches the loops below by `T`.
- **OP and Gust are “not suitable” for dual-sparse SNNs** because they produce `T×` psum matrices/rows (goal 2).
- Sequential `t` always costs `T×` latency (goal 3).

So LoAS’s A is **IP+FTP**, and **Gustavson is a rejected alternative** in this paper.

Table I identity: spike sparsity **and** weight sparsity, **S + fully-T**, **LIF**. SpinalFlow: S, LIF, no W sparsity. PTB: S+partial-T, LIF. Stellar: S+fully-T, **FS** (not LIF).

### 2.2 FTP vs PTB vs Stellar vs SpinalFlow

- SpinalFlow (§II-E): tick-batch sequential `t`, temporal-coded bias, no W sparsity.
- PTB: time-windows across systolic columns; **inside** a window, `t` still sequential. Even `time-window=1` still differs: PTB puts `t` between `m` and `n`; LoAS puts `t` innermost (§VII).
- Stellar: fully-T but **FS neurons** with detached accumulate/fire, so no LIF temporal dependence at accumulate. LoAS: LIF dependence exists; FTP still unrolls it.
- LoAS: TPPEs compute full `O[n, :]` across all `t`; P-LIF emits all `T` output spikes “at once” (Fig. 7 purple box).

LIF they actually unroll (§II-A, **hard reset**):

```
O_{m,n}[t_i] = Σ_k A_{m,k}[t_i] B_{k,n}                 (1)
X_{m,n}[t_i] = O_{m,n}[t_i] + U_{m,n}[t_{i-1}]
C_{m,n}[t_i] = 1 if X > v_th else 0                     (2)
U_{m,n}[t_i] = τ X (1-C)                                (3)  # hard reset to 0
```

Footnote 2: they **focus on hard reset**; other resets “will not lose generality in the hardware design” — again a compatibility sentence, not a sweep.

`v_th` is “a pre-defined **scalar** value.” Not a per-neuron continuous amplitude output.

Temporal dependence of `U[t-1]` is **not** removed. FTP parallelizes the **spMspM** over `t`. P-LIF is a **short combinational/unrolled chain of length T** after all `O[t]` are ready. Default **T=4** (5 accumulators: 1 pseudo + 4 correction). Fig. 16: TPPE area/power at T=16 is 1.37×/1.25× vs T=4; silent-neuron ratio likely falls for T>8.

### 2.3 Spike compression — packed T-bit fiber (local nickname “CSF”)

**The LoAS PDF never uses the acronym CSF.** Local notes use “CSF” for this fiber. The paper’s objects:

1. **Pack** all T bits of one pre-synaptic index into one word (Fig. 8 step 2). Example T=4: `a0,0` fires at t0 and t2 → packed `1010`.
2. **Silent neuron:** packed value `0000` (never fires in T). Dropped from storage.
3. **Bitmask + pointer fiber** for a row of A (and column of B): bitmask marks non-silent positions; pointer locates packed nonzero values. “We call this compressed row: a fiber [34], [62].”
4. **FiberCache** unified global buffer ([62] Gamma): line = bitmask+pointer, then values; NULL pointer if the line holds all values; banked so many TPPEs read concurrently; one compressed **row fiber of B** broadcast to all TPPEs.

Why not CSR/COO: for 128×128, CSR needs two 7-bit coordinates per 1-bit spike, **per timestep** (§IV-A). Packing + silent-drop is the FTP-friendly format so `t` unrolling sees a **contiguous** T-bit word.

Weights: same bitmask fiber, **column-wise**.

Preprocessing (§V, Fig. 11, Table II): mask neurons that fire **only once** in T, fine-tune <5 epochs, recover accuracy. Raises packed silent density (e.g. AlexNet origin 81.2% spike sparsity → packed 71.3% silent → 76.7% after FT). Hardware compressor also drops outputs with 0 or 1 spike. This is **workload mutation**, not a free skip.

Table II (T=4): AvSpB = 98.2 / 98.2 / 96.8% on AlexNet / VGG16 / ResNet19. Direct-coded, LTH, 15 search rounds.

### 2.4 Inner-join, pseudo-accumulator, correction

Not a merger-tree GP PE. Inner-join on two bitmasks (SparTen-style):

- AND bitmasks → match positions
- **Fast** prefix-sum on **B** (one cycle, tree, n=128)
- **Laggy** prefix-sum on **A** (16 adders, 128-bit buffer, **8 cycles** = `len(bm-A)/#adders`)

SNN-specific trick (§IV-C): a match implies fiber-A is **non-silent**, so B will be accumulated **at least once**. Speculate that packed A is **all 1s**, stream B into the **pseudo-accumulator** at fast-prefix rate. When laggy prefix finishes, inspect packed A:

- if A is all 1s: discard buffered B (pseudo-acc already correct)
- else: send B into **correction accumulators** for the timesteps where A is 0; subtract from the all-1s pseudo sum

Fig. 10: A=`1111` → discard; A=`1010` → correct t1 and t3.

**Final `O[t]` is exact** if correction runs. “Approximate” in Gustav’s paraphrase refers to the **speculative first step**, not an allowed numerical error. Each TPPE: **1× 12-bit pseudo-acc + 4× 10-bit correction accs** for T=4 (Table III / §V).

Then duplicate corrected `O[t]` into spatially unrolled P-LIF.

Output compressor: inverted **laggy** prefix-sum (need not be fast), bitmask of output spikes.

Scheduler: swizzle-switch crossbar [47 in LoAS bib = Sewell swizzle-switch, **not** GustavSNN].

### 2.5 Hardware identity of the numbers

Table III: 16 TPPEs, 8-bit W, 256 KB 16-bank 16-way FiberCache, 16×16 swizzle crossbars, 128 GB/s HBM (16×64-bit), **32 nm, 800 MHz**.

Table IV: 2.08 mm², 188.9 mW. Global cache 0.80 mm² / 124.5 mW (**65.9%** power). Per TPPE, **fast prefix-sum is 51.8% power / 66.7% area**. Laggy prefix is the saving vs two fast prefix-sums (SparTen).

Baselines: SparTen-SNN (IP), GoSPA-SNN (OP), Gamma-SNN (Gust) — ANN spMspM chips with multipliers removed, **`t` sequential innermost**, 16 PEs, same SRAM. Also ScaleSim estimates of **dense** PTB (16×4) and Stellar.

Results (do not export as this net’s FPS):

- vs dual-sparse sequential: average **6.79× / 5.99× / 3.25×** speedup vs SparTen / GoSPA / Gamma; **up to 8.51×** vs SparTen on ResNet19 (lowest A sparsity)
- FT preprocessing: **+20%** more performance
- energy (with FT): up to **3.68×** vs SparTen on AlexNet (abstract “up to 3.68×”)
- vs dense PTB/Stellar on dense VGG16 T=4: ~6× energy vs PTB, ~2.5× vs Stellar, **46.9×** speedup vs PTB (Fig. 19) — mixed dense vs dual-sparse, not an iso-sparsity PE contest
- Fig. 17: performance **highly sensitive to weight sparsity**; 98.2%→25% W sparsity loses ~88% performance. T×2 loses ~14%.

Workloads: AlexNet, VGG16, ResNet19 classification; SpikeTransformer HFF layer only in a scalability bar. Intro **lists** optical flow [28] Spike-FlowNet as an SNN *application of the field*, **not** a LoAS benchmark.

### 2.6 Identity assumptions

1. Dual-sparse **spMspM**: unary spikes **and** LTH-pruned weights (~97–98% W).
2. Direct-coded rate SNN, **small T (default 4, discussed ≤8)**, BPTT + surrogate gradient.
3. LIF, **hard reset**, scalar `v_th`, binary C. Temporal dependence handled by unrolled P-LIF of length T, not by tick-batch membrane REG.
4. IP dataflow, `t` innermost, fully unrolled. Inner-join bitmasks, not GP merger.
5. Skip unit = **silent across all T**. Neurons that fire once still occupy a fiber unless the FT mask removes them.
6. Pseudo-acc + correction is a **throughput** speculation that must be corrected to exact AC.
7. Classification CNNs (plus one transformer HFF shape). Single-core dataflow; neuromorphic multi-core **out of comparison** (footnote 1).
8. FiberCache + HBM 128 GB/s, 16 TPPEs, 32 nm 800 MHz.

### 2.7 What LoAS does **not** claim

- Does not claim event-camera optical-flow **results** (citation [28] is background).
- Does not claim continuous θg, ATLIF, residual dual-consumer, projection BN, or noncausal T10.
- Does not claim Gustavson is the right SNN dataflow; it claims the opposite for dual-sparse SNNs (§III).
- Does not claim FTP remains efficient at large T or low silent-neuron ratio (Fig. 16b; Gustav later uses this as the attack).
- Does not claim the all-1s pseudo-acc is usable **without** correction.
- Does not claim PTB-reconfigured-to-T-parallel equals LoAS (loop order differs).
- Does not claim Stellar-style FS neurons; Table I distinguishes LIF vs FS.
- Does not claim 46.9× vs PTB is an iso-sparse, iso-neuron comparison.
- Does not provide a dual-lifetime completion contract.

### 2.8 Hole vs local continuous-θg dual-consumer T10 residual

| LoAS object | Local object | Break |
|---|---|---|
| Packed **T-bit unary** fiber | Continuous θg / I24 / q24 PED | Packing bits is not packing amplitudes; 0-diff integer path forbids dropping correction or treating θg as {0,1} |
| Silent = never fires in T | Residual/PED is a **continuous** path; lifting T10 is 159 add/sub + 35 RNE/sat, not a silent-neuron bitmap | FTP’s main skip (60–70% silent, Table II) need not exist |
| P-LIF unroll of **causal** LIF T=4 | **Noncausal T10 PSN** | `C=LIF(O[t])` is the wrong operator; unrolling LIF does not evaluate the compiled T10 graph |
| One TPPE → one output neuron → P-LIF | Two consumers + projection BN over **full 10×96×120×160 stats** | Finishing O[t] for LIF is not finishing r1 |
| Dual-sparse LTH W | Freeze does not state W sparsity; LoAS identity is LTH ~97–98% W. If this student’s W is dense, a weight bitmask has nothing to join | Do not import Table II AvSpB; inner-join of two bitmasks is then the wrong skip |
| Mask 1-spike neurons + FT | Optical-flow AEE gates (abs 1.259, rel +0.005) | That preprocessing is a different network; cannot import Table II silent ratios |
| Speculate all-1s then subtract | Integer gates/I24/PED **0 vs model** | Speculation is legal only with a proven exact correction; “almost no throughput penalty” is not a license for AEE drift |
| T innermost of IP | T10 is a **horizontal compiled FIR-like** source, already CSE’d | Re-introducing an IP `k` loop around T10 can undo the 260→159 CSE |

Local already-tried fragment (from the 2026-09-10 line, not from LoAS): 8-bank T10 parallel accumulate **without** pseudo-acc/correction/fiber/BN2/shortcut — adds down, cycles up. That was an incomplete copy, not a LoAS falsification.

### 2.9 Complete-transfer checklist (A, before any X)

A copy that may be called “LoAS-complete” on this net:

1. IP loop with **`t` innermost**, spatially unrolled over the **actual** T (here T10, not 4).
2. Packed-across-T fiber for the **unary/gate** view, bitmask+pointer, silent-drop **only where the numerical function allows**.
3. Matching column-wise weight fiber **if** W is sparse; dense W must keep a dense control.
4. Inner-join: AND, **fast** prefix on W, **laggy** prefix on source, FIFOs as in Fig. 10.
5. **Pseudo-acc + T correction accs**, with a proof that corrected O matches the integer model (local bar: 0 difference).
6. Operator after O: **not** blindly P-LIF. Either (a) an unrolled causal LIF — which is the **wrong** local operator — or (b) the **compiled T10 PSN** as the thing FTP is feeding. (b) is adaptation; without a written mapping, the copy is not complete for this net.
7. FiberCache (or iso-capacity banked scratchpad) with the paper’s replacement intent: hold fiber-A chunks as long as possible; broadcast one fiber-B.
8. Output/gate compressor if the gate path is bitmask-worthy; residual/PED path still needs a **dense or quantized** write port.
9. Dual-consumer completion: gate fiber **and** PED/BN/add. Not in LoAS; required before claiming this net is “running LoAS.”
10. No 1-spike masking unless a new student is trained and passes AEE gates.
11. Iso-16-PE (or iso-port) sequential-`t` IP/OP/Gust controls, which is how LoAS itself claims speedup.
12. Service model with ports and backpressure, not only AC-count.

Until 1–12 exist, “time-parallel accumulators” is not LoAS.

### 2.10 Candidate X that is **not** “we also do FTP / we also do Gustavson”

- **X-L1. Exact dual-path pack, no LIF unroll.** Use FTP’s innermost-T pack + inner-join **as A** on the **gate** bitmask, and a **second** exact (non-speculative, or speculated-and-corrected-to-0-diff) path for continuous PED. The X is the **coupled retire** of packed-gate and continuous residual under one port contract. LoAS has one P-LIF consumer.
- **X-L2. Noncausal compiled-T10 as the unrolled operator.** Replace `parallel-for t: O += A[t]*B; C=LIF(O)` with the **already-CSE’d** T10 graph (ordinary 260 or lifting 159+35 RNE) evaluated so that T-parallelism does not reopen CSE, and so RNE/sat intermediates stay on the same two-stage writeback that already moved 6938→5354 always-ready slots. The X is **preserving CSE under T-unroll with dual consumers**, not FTP itself.

Not X: FiberCache, laggy prefix, pseudo-acc, silent-neuron pack, LTH dual-sparse, “fully temporal parallel,” or Gamma/Gustavson (LoAS rejected Gust).

### 2.11 Controls

- Sequential-`t` IP (SparTen-SNN analogue) vs FTP, **same** ports.
- Complete pseudo-acc+correction vs “parallel acc without correction.”
- Gate-only vs PED-only vs dual-path (X-L1).
- Ordinary 260-op T10 vs lifting 159+35, both under FTP-style unroll (X-L2).
- With vs without 1-spike masking (must re-measure AEE).
- Dense W vs bitmask-W (local W may be dense).
- PTB-style `t` between `m` and `n` vs LoAS innermost `t` (§VII).
- Gustav CPTB tick-batch as **parallel prior control**, not a parent layer.

### 2.12 Kill gate

Same AEE and ≥15% full-chain same-resource gates as §1.14.

- If T10 unroll **without** fiber/correction already matches cycles (local 8-bank lesson: adds −8.96%, cycles **+0.23454%**), complete A still required; if complete A still does not beat sequential innermost-`t` on **this** net, stop FTP-as-title, keep it as control.
- If silent-neuron ratio on θg/gate is too low (Gustav’s `1-ρ^T` attack at T=10), FTP skip dies — kill X that depended on silent pack; do not switch the title to Gustavson as a consolation prize without running §1.11.
- If pseudo-acc is kept but correction is dropped, **integer 0-diff is the kill** (must stay 0 vs model on I24/PED q24).
- If innermost-`t` IP **undoes** CSE (159+35 → more ops) or needs extra banks vs the frozen two-stage SIMD point, fail.
- Novelty kill: “we also FTP” or “we also inner-join.”

### 2.13 Section-quoted evidence (LoAS)

> “The main challenge is that processing timesteps, a natural property of SNNs, introduces an extra loop to ANN spMspM, leading to longer latency and more memory traffic. To address the problem, we propose a fully temporal-parallel (FTP) dataflow”
> — Abstract

> Table I: LoAS = spike sparsity ✔, weight sparsity ✔, parallel support S+fully-T, neuron LIF.
> — Table I

> “Our second observation is that both OP and Gust dataflow are not suitable for dual-sparse SNNs since they oppose goal (2).”
> — §III

> “We first choose to position the t-dim at the innermost of the IP dataflow, as given in Algorithm 1. … we fully parallelize the t-dim”
> — §III

> “We call this compressed row: a fiber [34], [62].”
> — §IV-A (no “CSF” string)

> “This mechanism opportunistically presumes the matched non-zero value of fiber-A is all 1s (pre-synaptic neuron fires at all timesteps) … Since the non-zero value in fiber-A is not always all 1s, we need a mechanism to ensure that the accumulation results are correct.”
> — §IV-C

> “After the computation of the pseudo-accumulator completes, its accumulation results are duplicated and sent to each correction accumulator. The correction value inside each accumulator will be subtracted from the pseudo accumulation results for each timestep. … we spatially unroll the LIF operations so that the output spikes for all timesteps will be generated at once.”
> — §IV-D

> “We adopt a FiberCache design [62].”
> — §IV-D

> “We set the default timesteps T to 4 across all experiments.”
> — §V

> “it is very likely to have fewer silent neurons when we have even larger timesteps (> 8). This is one of the challenge that LoAS needs to face when scaling up on the number of timesteps.”
> — §VI-B

> “SNNs have been widely used in computer vision tasks, such as image classification [46], [56], optical flow estimation [28], semantic segmentation [21], and object detection [20].”
> — §I (application list; **not** LoAS eval)

> “We are not comparing with those systems due to our focus on single-core dataflow SNN accelerator designs.”
> — footnote 1, neuromorphic multi-core

---

## 3. Prosperity, LoAS, GustavSNN — three parallel priors

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
| GustavSNN **includes** LoAS FTP | **No**; CPTB is posed as the alternative (Design Choice #2) |
| LoAS **uses** Gustavson | **No**; §III says Gust is not suitable |
| LoAS **evaluates** Gamma-SNN (sequential Gust) | Yes, as a **baseline it beats** |
| Local “Gustav = Prosperity + GP” | **Forbidden** unless a new citation appears |

Transfer implication: complete **A** may copy **one** of these denominators (or run them as **controls of each other**). Title-level X cannot be “we fused all three names.”

---

## 4. Joint hole, one X budget, TCAS-II

Local B (from freeze, not from these papers): r1 residual chain, **continuous θg**, **two consumers**, native projection+BN+add, **noncausal T10**, same-port/same-backpressure, AEE abs 1.259 / rel +0.005, full-chain net service ≥15%. Proxy “patch ~35%” is **not** the new student’s cycle share.

What a complete prior copy can legally be:

- **A-Gustav:** CPTB + NRV + NR4 PE + dual-port W tile + same-ID barrier + column-major time-second, with dual-consumer mapping written down (checklist §1.11).
- **A-LoAS:** FTP innermost-T + bitmask fiber + inner-join + pseudo-acc **and** correction + FiberCache, with T10/dual-consumer mapping written down (checklist §2.9).
- **A-Prosperity:** not in these two PDFs; do not smuggle it in via Gustav’s comparison paragraph.

What can still be X after that copy (pick **one**):

1. Dual-retire of continuous θg-gate **and** residual/PED on P-column (or fiber) state — neither paper’s commit rule.
2. Occupancy / pack compiled from **union of two consumers** on the CSE’d T10 graph — neither paper’s skip predicate.
3. Noncausal T10 evaluated under T-parallel **without** breaking CSE or 0-diff integer — LoAS unrolls LIF, Gustav sequences LIF.

Kill the title if the sentence is only “Gustavson on SNN,” “FTP on SNN,” “NRV skip,” or “Prosperity+Gustav.” Reviewers will treat those as missing **relative prior** plus missing **measured advantage** on this identity.

**Strongest shared control:** ordinary dense-source/raw, same ports, same state, same 8088-class backpressure, both consumers retired, actual BN stats, integer 0-diff. Gustav and LoAS papers never ran that control; a local win against a sequential scatter that they also would have beaten is a weak denominator.

**Do not quote** 11.8×, 8.51×, 3.68×, 1.34 mm², 2.08 mm², 28 nm, or 32 nm as this student’s result.
