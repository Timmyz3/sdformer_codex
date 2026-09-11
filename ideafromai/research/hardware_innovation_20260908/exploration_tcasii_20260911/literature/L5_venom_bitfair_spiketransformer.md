# L5 · VENOM / BitFair / SpikePack / T-parallel Spiking Transformer / SpAtten

Freeze: 2026-09-11. Full local texts, not cards:

- `survey_ab_fusion_20260910/p0_txts/2310.02065.txt`
- `survey_ab_fusion_20260910/p0_txts/2607.05445.txt`
- `survey_ab_fusion_20260910/p0_txts/2501.14484.txt`
- `survey_ab_fusion_20260910/p0_txts/2503.19643.txt`
- `survey_ab_fusion_20260910/p0_txts/2012.09852.txt`

Question for this sheet: on **continuous-θg dual-consumer T10 PSN** (gate/spike path **and** continuous residual/PED path, noncausal T10, same-port/same-state/same-backpressure), which of {N:M format, bit-serial early terminate, spike packing, reconfigurable T-parallel transformer} is a **CONTROL (A, copy whole)** vs a **title-level X**.

**One-line answer:** all four are **CONTROL**. None is title-level X on this identity. SpAtten (the fifth file) is also CONTROL, not “generic importance pruning.” Candidate increments, if any, sit *on top of* VENOM or BitFair after dual-consumer union is charged; they are not supplied by these papers.

Contract from `PROBLEM.md`: ATLIF emits **continuous threshold amplitude θg**, not binary spikes. Dual consumers after source. Do not change identity to binary ATLIF. Do not sell analog CIM / foundry PPA / multiplied FPS.

---

## 0. Title confirmation

| arXiv | Confirmed title | What it actually is |
|---|---|---|
| 2310.02065 | **VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores** (SC 2023; Castro, Ivanov, Andrade, Ben-Nun, Fraguela, Hoefler) | GPU V:N:M format + Spatha SpMM + second-order pair-wise OBS pruning. Not a neuromorphic paper. |
| 2607.05445 | **BitFair: A 12-nm Bit-Serial CNN Accelerator with Learnable Early Termination and Adaptive Bit Ordering for Ultra-Low-Power XR Vision** (JETCAS 2026; Li & Gao) | Weight-bit-serial CNN; learnable ReLU-zero early stop + greedy bit order. Event *bins* as CNN input, not an SNN. |
| 2501.14484 | **SpikePack: Enhanced Information Flow in Spiking Neural Networks with High Hardware Compatibility** (Shen, Li, Li, Zhao, Zeng) | **Confirmed SpikePack.** Rank-1 global membrane + integer zip of T spikes. Neuron model, not an ASIC. |
| 2503.19643 | **Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing** (Chen & Chang, NYCU) | 28 nm Spike-IAND-Former: IAND residual + unrolled LIF T-parallel tick-batching. Short paper (~298 lines of extract). |
| 2012.09852 | **SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning** (HPCA 2021; Wang, Zhang, Han) | **Not** N:M / weight-importance pruning. On-the-fly **cascade token/head pruning** from cumulative attention mass + progressive MSB/LSB. |

---

## 1. N:M formats — VENOM (2310.02065) → **CONTROL**

### What the paper actually does

NVIDIA Sparse Tensor Cores only accept **row-wise 2:4** (`mma.sp`, half: k32/k16). VENOM adds an abstraction **V:N:M**:

1. Partition A into blocks of **V × M**.
2. **Vector-wise:** keep 4 shared columns out of M (column-loc of size `R/V × K/M × 4`).
3. **Row-wise 2:4** inside those 4 columns (native SPTC).
4. Storage: dense nonzeros `R × K/M × 2`, 2-bit **m-indices**, plus **column-loc**. m-indices index the *selected* 4, not the original M.

Example in Fig. 4: **4:2:8** is 2:8 on the original row, mapped as 2:4 onto SPTCs; B rows not in column-loc are never loaded. 50% is `1:2:4`; higher ratios `128:2:10` … `128:2:100` (80–98%). Peak kernel claim **37× vs cuBLAS** is at **98% / 2:100**, not 50%. At 50%, Spatha is ~2× dense and up to **1.38× cuSparseLt**. GPT-3 encoder: GEMM time **11×** at 2:32; end-to-end much smaller.

Spatha is a template SpMM: TB tile `BSr=V`, async GMEM→SMEM, 128-bit coalesced stores, bank-conflict padding, m-indices in RF. column-loc overhead is small except at 2:100.

Pruning is **not** magnitude. §6 uses SparseGPT-style group OBS:

`ρ_Q = 1/2 (E_Q w*)^T (E_Q F̂^{-1} E_Q^T)^{-1} E_Q w*`

Inside V×M they drop inter-row correlations and use **pair-wise** `E_Q = {[1,0],[0,1],[1,1]}` instead of all `C(M,N)` subsets. **Gradual N decay** (start N⁰ ≫ N^β, then tighten) because one-shot 2:16 already hurts. BERT SQuAD: ~2% F1 drop at 2:16.

### What must be copied as A (CONTROL)

- Two-level format: shared column-loc **and** per-row 2:4, not “a 2:4 mask.”
- Indirect gather of B (or source) by column-loc; the index traffic is part of the method.
- Pair-wise / group second-order score + gradual schedule + recovery, not W².
- Same physical word as the skip unit. GPU 128-bit vector load ≠ local 64-bit source bank word.

### Why this is not X on dual-consumer θg PSN

VENOM prunes **one sparse operand** (weights) for **one dense consumer** (activations). This net has:

- one **shared source word** feeding **two consumers** (gate/θg path and continuous residual/PED);
- noncausal **T10** (future ticks still consume the same source);
- already-measured fact: ordinary row 2:4 deletes ~50% W but **does not drop P4 64-bit source-word demand**; shared H8×C16 can drop ~40% service and still lose to hidden50 on AEE.

A co-mask on H8 that saves W slots is not VENOM. A format that still reads the full 64-bit word is not VENOM. GPU 37× is the 2:100 regime and is not a same-port number.

**Possible increment on top of this CONTROL (not supplied by VENOM):** choose V:N:M groups so that **both** consumers’ T10/PED error plus **touched physical source words** go down, with one compact source layout. That is a HiNM/VENOM *interface* hypothesis already named in `sparsity_transfer_followup_20260908.md`. It is not a new format.

**Strong controls if anyone trains it:** full VENOM/HiNM (column-loc + 2:4 + OBS + gradual), ordinary row 2:4, same-GT hidden50 / R16 narrow-dense, same source-word budget. Kill if source words still read full or AEE loses to hidden50.

**Forbidden to move:** cuBLAS/Spatha speedups, SPTC `mma.sp` shapes, “energy” = retained |w| ratio.

---

## 2. Bit-serial early terminate — BitFair (2607.05445) → **CONTROL**

### What the paper actually does

CNN, not SNN. Motivation §II-A is explicit: SNNs have a training/accuracy gap on DVSGesture (ReckOn 87.3%, CICC’25 90.5% vs their CNN 96.5%); on dense RGB, spike-rate saturation kills SNN efficiency. They run **CNN on event bins** (SpikingJelly equal-count 16 bins).

Mechanism (Fig. 1, §III):

- Weights: 8-bit **sign-magnitude** (1 sign + 7 magnitude). Activations: 8-bit 2’s complement, **bit-parallel**. Partial sum: 16-bit 2’s complement.
- **Reverse loop nest:** for each bit position j in order ω, reduce **across all spatial/channel inputs**, then next bit. Conventional bit-serial does all bits of one weight then next input — that cannot early-stop on the *output sign*.
- After k bit-planes: `P_k = Σ_{j=0}^{k} Σ_i 2^{ω(j)} s_i b_{i,ω(j)} a_i`.
- **Hard stop:** `P_k ≤ θ^l` ⇒ predict ReLU(y)=0, write 0, skip remaining bits.
- `θ^l = θ_0^l + θ_x^l`, `θ_0` from fused BN (`−(β/γ)√(σ²+ε)+μ`) or `−bias`.
- Training: soft gate `G_k = σ(−(P_k−θ)/T)`, survival `S_k = Π_{m≤k}(1−G_m)`, output `Â = S_K ⊙ ReLU(P_K + b̃)`. Loss `L_CE + λ_bit L_bit` where `L_bit` is the expected processed-bit fraction. T anneals 1.0 → 0.05.
- **Adaptive bit order (Algo 1):** greedy per layer, score = ETR / (AccLoss+ε), remaining bits filled MSB-first for the probe. Offline, once.

Hardware: 16×16 **output-stationary** PEs; each PE has its own comparator vs θ^l; FSM **aggregates terminate** to suppress further operand fetch for that output; clock-gate idle PEs. GF 12 nm post-layout: 0.34 mm², 104 KB, 0.55–0.70 V, 0.12–1.55 ms, 0.8–13.7 mW, 234.0 BTOPS/W, 0.07 pJ/SOP. DVSGesture 96.5%, N-MNIST 97.7%. Cycle “speed-up” vs vanilla 8-bit bit-serial is **2.12×** on DVSGesture with ABO (not 22×). 4.0–22.1× is energy vs **fabricated SNN XR chips**, a different denominator.

§VI limitation, in their own words: speed-up is **bounded by the ReLU-negative fraction**; GELU/smooth activations “rarely produce exact zeros and would benefit less.” Dense RGB SVHN (55% ReLU sparsity) already drops to 1.92×.

### What must be copied as A (CONTROL)

- Bit-plane-across-all-inputs, then threshold, then skip remaining planes.
- Learnable θ + survival regularizer + temperature anneal — not a hand BN threshold (BitSET).
- ABO as an *offline prefix order*, not a title.
- Per-PE terminate + **group/FSM gather** that stops **further fetches** for completed outputs. The gather is part of the prior (SparseInfer/BitFair family).
- Continue path when the prefix is uncertain: full remaining bits, no “certificate.”

### Why this is not X on dual-consumer θg PSN

BitFair’s stop **zeros a ReLU output**. This identity has **no ReLU-zero consumer**:

- Gate path still emits **continuous θg**, including **non-zero** amplitudes. Stopping because “this will be zero” is the wrong predicate.
- Continuous residual / PED / native conv+BN+add still need the **value**, not a {0, ReLU} decision. A prefix that is enough to know “gate is silent” is **not** enough to know the residual.
- Noncausal T10: a prefix that retires one tick does not retire the vector. Shared source: seven heads finishing at C192 and one still live at C384 still read the C384 word (already measured on GP suffix bounds).
- Weight bit-planes ≠ lifting RNE/sat intermediates. BitFair never mentions residual dual-issue.

**Possible increment on top of this CONTROL (not BitFair):** train a prefix accept/continue on the **union of unfinished dual consumers** (full T10 θg **and** PED/residual), charge the last pending consumer’s physical request, keep non-zero θg on accept. That sentence is already in `conditional_execution_followup_20260908.md` / `sparsity_hardware_next_20260908.md`. BitFair supplies the **control skeleton**, not the X.

**Strong controls:** vanilla bit-serial (all bits), BitSET static BN θ + MSB-first, SparseInfer predict-then-skip, exact-width header / whole-word, BitFair-style *independent* per-neuron stop with the **same** group-close logic (to isolate the union objective). Kill if union objective does not beat independent BitFair+same closer, or AEE > 1.259 / +0.005 vs same-budget dense.

**Forbidden to move:** 4.0–22.1×, 0.07 pJ/SOP, 12 nm PPA, DVSGesture accuracy, “SNN-like event-driven CNN” as our identity.

---

## 3. Spike packing — SpikePack (2501.14484) → **CONTROL**, identity-hostile if used as the neuron

### What the paper actually does

Diagnoses LIF as (i) recursive `v_t = v_{t-1}/τ + W s_{t-1} − θ s_{t-1}` so each spike sees only a partial prefix of the sequence; (ii) O(T) time/space and wasted GeMM because each tick is a 1-bit × W.

SpikePack:

- Global membrane **`v_g^l = w^l S^{l-1} q`** with `q = [τ^{T-1}, …, τ^0]^T`. Rank-1 temporal fold.
- Decode: `v_0 = v_g`, `v_t = v_{t-1} − θ_t s_{t-1}`, `θ_t = θ / τ^{t-T}`, spike if `v_t > θ_t`.
- Packed integer: `s_zip = S q` (they also write `s_zip = ⌈v_g / (θ/τ)⌋`); each bit of `s_zip` is a tick. **O(1) in T for *word count***, not O(1) bits.
- `τ=2` ⇒ uniform bit quantization; `τ≠2` ⇒ nonuniform. Default τ=2 on GPU “to avoid exponentiation.”
- Backward: `∂L/∂s_zip ≈ (∂L/∂s_zip_out)(w/θ)` with `∂⌈x⌋_τ/∂x ≈ 1`. No BPTT unroll.
- Mutual-information argument vs LIF (App. A); empirical MI higher in Fig. 4.

Experiments: ImageNet SEW-ResNet / Spikeformer; they **plot PSN (Fang et al., NeurIPS 2024 parallel spiking neuron)** as a competing *neuron*, not as this project’s T10 PSN. Spikeformer-8-512 T=8: 80.1%, memory/time **flat in T**. ANN-to-SNN conversion near-lossless at T=8. FPGA xczu3eg 300 MHz neuromorphic PE (64 syn, 16 neuron, 16-input spike detector): ResNet-34 23 ms / 18.6 mJ vs LIF 29.1 ms / 23.8 mJ. Binary spike I/O is a selling point for neuromorphic reuse.

### What must be copied as A (CONTROL)

- Rank-1 pack `v_g = W (S q)` + integer zip + threshold decode. This is the **time-fold baseline** for any “compress T10 into fewer words.”
- O(1) *vectors* vs T; still pay bit-width of the packed integer.
- Direct STE through the packed scalar (no BPTT) as a training-template control.
- Serial mode on existing SNN engines vs parallel GeMM: they already claim both.

### Why this is not X — and why replacing ATLIF with SpikePack is out of identity

- Local **single-projection + uniform 8-level quant** (explicit SpikePack Eq. 8–9 control) already ran valid825 **AEE 1.310533** and failed the 1.259 gate (`README.md`). That does **not** kill the whole SpikePack family, but it kills “pack T10 to one scalar then decode” as a free lunch on this student.
- **O(1) words ≠ O(1) bits.** A T10 zip is a wider integer; dual consumers still need enough bits for **amplitude θg** and for the residual path.
- Rank-1 `q` is **not** full-rank T10 PSN / lifting. Independent time rows, learnable 40-coeff lifting, and dual PED all assume the T axis is not a single geometric leak.
- Output of SpikePack is a **binary spike train** (plus an integer count). This paper’s ATLIF output is **continuous θg**. Using SpikePack as the neuron **changes the paper identity** (forbidden).
- Dual-consumer residual/PED is a **dense multi-bit** path. Zip-and-threshold does not produce a residual addend.
- FPGA “compatible with neuromorphic processors” is binary-spike hardware. Not a same-port T10 PSN schedule.

**X?** No. Keep SpikePack as the **mandatory time-fold / rank-1 / zip control** whenever anyone proposes packing T10. Do not retitle lifting or PSN as SpikePack. Do not claim O(1) service.

**Strong controls:** ordinary time-row T10, full-rank PSN, lifting T10, low-rank/CMVM, raw dense. Kill a pack layout if AEE fails 1.259 or if packed width × dual-consumer replay ≥ ordinary T10 words.

---

## 4. Reconfigurable T-parallel transformer — 2503.19643 → **CONTROL** for T-unroll / membrane-free; **IAND is an X-killer**

### What the paper actually does

Short 28 nm accelerator paper. Two coupled moves:

**Model (Spike-IAND-Former).** Spikformer residual **add** makes values leave {0,1}, so later conv cannot stay AND-gates. They replace residual with **element-wise IAND** `x ⊙ (1 − ConvBN(x))` (Fang et al. SEW/IAND prior). Whole net spike I/O. LIF θ=0.5, leak=0.25. Tokenizer first layer encodes 8-bit images into T spikes (bitplanes reuse the spike PEs). ImageNet 8-384 T=4: 70.32% (Spikformer 70.24%). CIFAR-10 T=4 95.69; progressive T reduction T=2/1: 92.93 / 91.34.

**Hardware.** Fully **parallel tick-batching** vs SpinalFlow serial tick-batching:

- MAC across T has **no inter-tick dependence**; only LIF membrane does.
- **Unroll LIF** over T=4 so all ticks of one channel emit together; **membrane SRAM deleted**.
- Weight SRAM fetched **once**, shared across T (they quote **−43.2%** weight SRAM accesses vs a serial-T spatial-temporal design).
- Reconfigure T=4/2/1 by three muxes (select 111 / 101 / 000) that chain neighbor unrolled neurons.
- 12 PE blocks × 4 T × 8×9 PE = 3456 PEs. Vector dataflow: 3×3 diagonal psum, 1×1/matmul horizontal psum.
- TSMC 28 nm, 500 MHz, 198.46 kGE, 139.25 KB SRAM, 90.153 mW, **3.456 TSOPS**, **38.334 TSOPS/W**. CIFAR-10 T=4: 46.72 FPS. Activation sparsity 73.88%. Memory 43% of power, logic 57%.

They contrast SpinalFlow (serial T, repeated W and membrane), systolic spatial-temporal (needs membrane buffers, T still chained), Sibrain (parallel membrane *values* but still sequential spike emit).

### What must be copied as A (CONTROL)

- Unroll T, share W once, delete membrane if the neuron is a causal LIF chain.
- Reconfigurable T-parallelism (4/2/1 mux) as a **state-residency / occupancy** control, not a accuracy method.
- Vectorized 3×3 / 1×1 / matmul on one array as a **denominator** for “one PE kind.”
- Bitplane split of 8-bit pixels to reuse spike PEs — a control for “multi-bit input on spike datapath,” **not** a way to implement continuous θg.

### Why this is not X on dual-consumer θg PSN

- **IAND residual is identity-illegal.** It exists to *destroy* the continuous residual that this net’s PED/native conv+BN+add **is**. Applying IAND would convert ATLIF/residual into binary spike I/O, which `PROBLEM.md` forbids.
- Unrolled LIF solves **causal** membrane dependence. This PSN is **noncausal T10**: future source samples affect the first output. Unrolling a causal LIF is **not** a free schedule for lifting/PSN. Gustav already notes that full Y or all pending U must stay; P membrane registers do not cover it for free.
- Dual consumers: even if T-parallel source MACs are legal (no T-dependence in the *projection*), the **residual consumer is multi-bit** and the **gate consumer is continuous θg**. A spike-only PE array does not issue both.
- 3456 GSOPS / 38 TSOPS/W / 46.72 FPS are **not** same-port service and must not be multiplied into a paper FPS.

**X?** No. Use T-unroll + single W fetch + membrane-free as a **CONTROL** when arguing state lifetime of T10 (F5-class). Never use IAND as a residual for PED. If unroll increases RF/ports enough to eat the W-fetch win under the same-port contract, the layout dies; that is not a family kill.

**Strong controls:** serial tick-batching + membrane SRAM (SpinalFlow), spatial-temporal systolic with membrane buffers, time-parallel *supply* keeping the **current continuous residual**, ordinary dense T10. Kill IAND layouts on AEE/identity immediately.

---

## 5. Fifth file — SpAtten (2012.09852): pruning/importance? **Yes, but not weight N:M**

Confirmed: **cascade token and head pruning + progressive quantization** for BERT/GPT-2 attention. No trainable weights inside attention; pruned objects are **tokens and heads**, selected **on the fly**, **never restored** in later layers.

- Token score: accumulate `attention_prob` over queries/heads/layers (GPT-2 also over generation steps).
- Head score: accumulate `|attention_out|` over tokens/dims.
- Algo 2 + hardware **quick-select top-k** (O(n) average, order-preserving, 16-way comparators, zero-eliminator).
- Local V pruning after softmax (this head only).
- Progressive quant: fetch MSB, softmax; if `max p < thres` (flat) fetch LSB and recompute. BERT is compute-bound so **static** quant only. Claim: 3.8× DRAM from tokens, 1.1× from heads, 5.1× from progressive quant; 5.9% of GPT-2 samples need LSB.

### CONTROL vs X on this net

| Piece | Verdict |
|---|---|
| Cumulative importance + top-k ranking | **CONTROL** (ranking template). Replace attention mass with **lifting-source activity + dual-PED/T10 loss**. |
| Cascade “once gone, gone forever” | **Too aggressive** for noncausal T10 and for a residual that may need a token later. Use as a **negative control**, not a method. F6-style cancellable columns already named as the reversible alternative. |
| Progressive MSB then LSB | **CONTROL** for bitwidth-vs-error, close in spirit to BitFair prefixes. Gate-only MSB is **not** enough for continuous θg/PED. Dual-consumer rule: fetch LSB if **either** consumer is uncertain. |
| HBM top-k / 162× GPU | Not a same-port number. |

**X?** No. NLP attention mass ≠ optical-flow AEE. Permanent cascade on shared source words can break the residual consumer. Ranking-by-dual-loss is a template, not a new mechanism.

---

## 6. CONTROL vs X matrix (the required extract)

Identity: **continuous θg**, **two consumers** (gate/θg and residual/PED), **noncausal T10**, shared source, same-port/state/backpressure.

| Mechanism | Paper | Copy-as-A CONTROL? | Title-level X on this net? | Dual-consumer failure mode |
|---|---|---|---|---|
| **V:N:M / N:M format** (column-loc + 2:4 + OBS + gradual) | VENOM | **Yes. Strong.** Must copy two-level index and physical gather, not a co-mask. | **No.** | One sparse W × one dense activation. Shared 64-bit source word can remain full after 2:4. Dual error (θg **and** PED) is not in ρ_Q. |
| **Bit-serial early terminate** (bit-plane prefix, learnable θ, ABO, PE+FSM stop) | BitFair | **Yes. Strong** for prefix / survival-loss / group-close. | **No.** | Predicate is **ReLU = 0**. Continuous θg is not 0/ReLU; residual still wants the value; T10 union may keep the source live. |
| **Spike packing** (`v_g = W(Sq)`, zip, dynamic-θ decode) | SpikePack | **Yes** as time-fold / rank-1 / zip **baseline**. | **No. Identity-hostile** if it replaces ATLIF. | Rank-1 fold ≠ full T10; zip is binary+integer, not dual multi-bit consumers. Local 8-level pack already AEE 1.310533. |
| **Reconfigurable T-parallel transformer** (unrolled LIF, T=4/2/1 mux, W once, no membrane) | 2503.19643 | **Yes** for T-unroll / W-once / membrane-free **residency**. | **No.** | Causal LIF unroll ≠ noncausal T10. **IAND residual deletes the continuous consumer.** Spike-only PE cannot issue PED. |
| Cascade token/head importance + progressive MSB | SpAtten | **Yes** as ranker + MSB/LSB template. | **No.** | Attention mass ≠ dual-PED loss. Permanent cascade vs noncausal T10. MSB-only may starve the residual. |

### Which of the four can be X?

**None of the four, as stated by their papers.**

The only *candidate* X sentences that these priors *support but do not contain*:

1. **On VENOM (format CONTROL):** group / permute so that the **union of dual-consumer T10+PED error** and the **touched physical source words** drop together, one compact layout. Still must beat HiNM+VENOM+hidden50. Not a new N:M.
2. **On BitFair (terminate CONTROL):** prefix accept/continue on **last unfinished dual consumer**, emit predicted **non-zero θg**, charge shared requests. Still must beat independent BitFair with the same closer. Not “learnable θ.”

SpikePack and T-parallel **do not even yield that kind of candidate**. SpikePack is a neuron swap. T-parallel+IAND is a binary-residual swap. Both fight `PROBLEM.md`.

---

## 7. What a TCAS-II reviewer would punish

- Renaming V:N:M, ABO, zip, unrolled LIF, or cascade top-k as the contribution.
- Quoting 37× / 22.1× / 38 TSOPS/W / 162× GPU as this student’s service.
- Changing ATLIF into binary SpikePack / IAND-Former to “make the hardware match.”
- Early-stopping the gate and silently dropping the residual.
- Claiming O(1) in T when the packed integer widened, or claiming 2:4 saves source words when the 64-bit bank still fills.

Relative prior that **must** appear if any of these families is used: VENOM+HiNM+ordinary 2:4+hidden50; BitFair+BitSET+SparseInfer+whole-word; SpikePack+ordinary T10+PSN+lifting; SpinalFlow/serial LIF+membrane vs unroll with **continuous** residual kept.

---

## 8. Kill-gates (numbers already on the freeze)

- AEE: absolute ≤ 1.259; vs same-budget strong control ≤ +0.005. Ordinary dense-source/raw = **1.219801338**. Lifting T10 = **1.232979368** (relative +0.005 already fails). SpikePack-style single projection = **1.310533** (dead layout).
- Service: full-chain same-resource net ≥ 15%. Do not add the SIMD −22.83% table to the integer-consumer −5.78% table.
- Identity: any method that needs binary spikes, IAND residual, or ReLU-zero as the *output contract* is a **stopped layout**, not a family death.

End of L5.
