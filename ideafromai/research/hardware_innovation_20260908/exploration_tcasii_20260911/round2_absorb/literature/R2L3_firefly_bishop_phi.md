# R2L3 FireFly-S / Bishop / Phi — remap onto AT-LIF absorb identity

Date of freeze: 2026-09-11. Independent of round-1 `literature/L3_firefly_scnn_bishop.md`.  
Sources: this session `PROBLEM.md` + `../IDENTITY_ATLIF.md` + the three local full texts. No invented citations; venue/page not in a local txt is not filled in.

| Paper | Local txt | Identity in this txt |
|---|---|---|
| FireFly-S | `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2408.15578.txt` | Title *FireFly-S: Exploiting Dual-Side Sparsity for Spiking Neural Networks Acceleration with Reconfigurable Spatial Architecture*; authors Tenglong Li, Jindong Li, Guobin Shen, Dongcheng Zhao, Qian Zhang, Yi Zeng; arXiv:2408.15578v3 [cs.AR] 29 Jan 2026; manuscript created 29 August 2024. Header is the generic “JOURNAL OF LATEX CLASS FILES”. Session label TCAS-I is **not** a volume/page found in this txt. |
| Bishop | `.../p0_txts/2505.12281.txt` | Title *Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning*; authors Boxun Xu, Yuxuan Yin, Vikram Iyer, Peng Li. arXiv:2505.12281v1 [cs.NE] 18 May 2025. Footer: ISCA ’25, June 21–25, 2025, Tokyo, Japan. DOI in txt: https://doi.org/10.1145/3695053.3731063 |
| Phi | `.../p0_txts/2505.10909.txt` | **Local.** Not missing; HTML was not fetched. Title *Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency Spiking Neural Networks*; authors Chiyue Wei, Bowen Duan, Cong Guo, Jingyang Zhang, Qingyue Song, Hai Li, Yiran Chen. arXiv:2505.10909v1 [cs.AR] 16 May 2025. Footer: ISCA ’25, June 21–25, 2025, Tokyo, Japan. DOI in txt: https://doi.org/10.1145/3695053.3731035 |

Numbers below (85–95% sparsity, 5.91×, 3.45×, 128 patterns, …) belong to those papers’ own networks. They are not this-net cycle share.

---

## 0. Identity used for this remap (locked)

Official AT-LIF (`IDENTITY_ATLIF.md`):

\[
o_i[t]=\theta_i\cdot H(m_i[t]-\theta_i)=\theta_i s_i[t],\quad s\in\{0,1\},\quad o\in\{0,\theta\}.
\]

At inference, **layer-shared \(\theta\) is absorbed into the next \(W\)**: \(W\leftarrow\theta W\). After absorption, the **spike path is binary \(\{0,1\}\times W\)** (select-add / Bitmap AND / pattern×W). It is not a per-event analog amplitude and not an unabsorbable int8 payload.

Two objects stay distinct (`PROBLEM.md`):

1. **Spike GeMM after absorb** — binary select-add. FireFly-S Bitmap AND, Bishop SAC/AAC/ECP, Phi L1/L2 hierarchy apply **here as complete priors A**. Prosperity product sparsity and Gustav NRV/CPTB are the same class (named in IDENTITY; not re-read in this sheet).
2. **Residual / PED / I24** — a **different continuous tensor**. Not AT-LIF carrying analog amplitude. If dual consumers exist: consumer A is the **binary gate after absorb**; consumer B is the **continuous residual path**. Dual-side sparsity and pattern hierarchy **do not** apply to B.

**Before** the AT-LIF threshold, this net has a **noncausal T10 mix** (PSN / lifting CSE). That mix is continuous arithmetic, **then** threshold, **then** absorb. T10 is therefore **not** a binary spike×W object.

Flip versus round-1 L3 (do not reuse that sheet’s kill list):

| Round-1 (continuous \(\theta_g\) as delivered amplitude) | Round-2 (absorb identity) |
|---|---|
| FireFly-S CLK0 AND is the wrong ALU | CLK0 AND is the **right** ALU on the **spike** path |
| Bishop AAC/ECP lemmas fail | lemmas **hold** on binary Q/K/V and binary MLP/projection inputs |
| Phi L1 alphabet is binary, so “not our neuron” | Phi L1/L2 is **complete A** on post-absorb spike rows |
| “θg not absorbed is the X” | **forbidden** as contribution (`SCOPE.md`) |

What does **not** flip: residual/PED, native projection BN on live full-domain batch stats, pre-threshold T10, same-port dual-path last-use, event-camera 2D OF / DSEC AEE.

---

## 1. FireFly-S — dual-side Bitmap skip

### 1.1 What the paper actually does

Software: **joint** gradient-rewiring pruning and an LSQ variant **during training**, targeting >85% **weight** sparsity and 4-bit integer W/bias/Vth. Spike sparsity is treated as already there. Hardware: spatial FPGA mapping, one core per layer, Bitmap dual-side detector that ANDs a 1-bit spike vector with a 1-bit weight mask and walks only matching non-zero pairs into IF/LIF. Dataflow is rebuilt so the temporal loop sits inside the output-channel loop (membrane storage-free consecutive V) and channel loops sit inside pixel loops (inter-layer pipeline). Models SCNN5/7/9 are stacks of **3×3** convs (`c3p1`) + maxpool, T=4 LIF, MNIST / DVS-Gesture / CIFAR-10.

It is an SNN **classifier** FPGA overlay-replacement. Not an event-flow transformer, not a residual dual-consumer, not a noncausal PSN.

### 1.2 Section quotes

**Abstract (dual-side claim + training + Bitmap detector).**

> While current SNN hardware accelerators often prioritize temporal spike sparsity, exploiting sparse synaptic weights offers significant untapped potential for even greater efficiency. … On the software side, we propose a novel algorithmic optimization framework that combines gradient rewiring for pruning and modified Learned Step Size Quantization (LSQ) tailored for SNNs, achieving a remarkable weight sparsity exceeding 85% and enabling efficient 4-bit quantization with negligible accuracy loss. On the hardware side, we present an efficient dual-side sparsity detector employing a Bitmap-based sparse decoding logic to pinpoint the positions of non-zero weights and input spikes. The logic allows for direct bypassing of redundant computations.

**§I contributions (what must exist together).**

> 1) FireFly-S proposes an algorithmic optimization framework that combines gradient rewiring-based pruning and a Learned Step Size Quantization (LSQ) variant specifically tailored for SNNs, achieving over 85% weight sparsity with 4-bit quantization during training for various SNN models.
> 2) FireFly-S employs a Bitmap-based sparse decoding logic in the dual-side sparsity detector, enabling efficient exploitation of sparsity found in both activation spikes and synaptic weights. This approach directly skips all computations involving zero values …
> 3) FireFly-S leverages a fully unfolded spatial architecture with inter-layer pipelining, eliminating off-chip memory access and concurrently processing multiple layers.

**§II.A neuron (binary spike).** Equations (2)–(4): \(I_i(t)=\sum_j w_{ij}s_j(t)+b_i\); leaky V update; \(V_i\leftarrow V_{\mathrm{reset}}\) if \(V_i>V_{\mathrm{th}}\). The synaptic input is **weight × binary spike** \(s_j(t)\). After AT-LIF absorb, **this is exactly the spike-path identity**.

**§III.B LSQ: their absorb is quant-scale into Vth, not AT-LIF \(\theta\) into next \(W\).**

> By sharing the quantization scale across weights within the same channel, along with the corresponding bias and neuronal threshold, our approach ensures consistent scaling … enabling efficient integer-only inference. Additionally, due to the unique properties of spiking neurons, there is no need to rescale per-channel activations during inference compared to ANN quantization, as the per-channel quantization scales are absorbed into the membrane thresholds.

Two different scalars: FireFly-S folds **quant scale \(S\)** into **this layer’s \(V_{\mathrm{th}}\)**; AT-LIF folds **\(\theta\)** into **next \(W\)**. After the identity lock both yield a **binary spike operand**. They are not the same fold, and FireFly-S’s fold does **not** replace native projection BN on this net.

**§V.B Dual-side Sparsity Detector (the circuit that is now A).**

> After passing through the Dataflow Orchestrator, the activation spikes are buffered and aligned with their corresponding weight mask data, which is represented in Bitmap format.
> At CLK0, a bitwise AND operation is executed between the vector pair, yielding a result denoted as x, which identifies active events.
> … a one-hot encoded output, y, is generated through the logical expression `y = x ∧ ¬(x−1)` … The register holding x is then updated using `x = x ∧ ¬y` …
> Finally, at CLK6, the weight data is fetched from the RAM and used to perform IF/LIF neuron model computation. … the voltage for each time step is temporally stored in the register and reused for the next time step.

Idle-cycle fill: if the AND is all zeros, they **inject bias as a fake pair** (`s_j=1` with `w=bias`) rather than stall.

**§III.B Algorithm 1 silent-channel prune.** For each output channel, for `t = 0 .. T-1`: `V[co][t] ← V[co][t-1] + b[co]`; if `V > Vth[co]` mark channel active. Channels never crossing Vth are dropped. This is a **causal leaky/IF threshold test with no synaptic input**, and the **only** consumer it considers is “does this channel ever spike”.

**§VI models.** Table II: SCNN5/7/9 from `c3p1` = “a 3x3 convolution kernel with padding of 1”; “we utilized LIF neurons in all experiments, with a time step of 4.” Their Bitmap study uses `(T, Co, Ci, Fho, Fwo, Kh, Kw) = (4, 64, 32, 16, 16, 3, 3)`. Spike sparsity they measured: 70–90%. Table VII (in txt): Quant+Prune sparsities 93.82% / 93.26% / 86.12% at 4-bit.

### 1.3 What MUST be trained (FireFly-S)

| Item | Trained in the paper? | If skipped |
|---|---|---|
| Weight sparsity via gradient rewiring + Laplace/ℓ2 on `θ`, including conv **and** FC **and** bias | **Yes, during training** | Detector still skips *existing* W zeros, but 85–95% W sparsity and the Bitmap vs COO/CSR operating region are not free. Dual-side is then almost only spike-side. |
| Joint 4-bit LSQ of **W + bias + Vth** with **shared per-channel scale** | **Yes, during training** (STE) | Integer-only IF/LIF path and “scale absorbed into threshold” break. |
| Silent-channel Algo 1 | Evaluation on the trained IF/LIF (bias vs Vth over T), then drop channels | Legal only if the **only** consumer is “does this channel ever spike”. |
| Spike sparsity | **Not** a training loss. “Natural sparsity of spikes inherent to SNNs.” | Inference-time, from thresholding. |

They explicitly reject prune-then-quant or quant-then-prune as a separate post-process.

### 1.4 Map after absorb

| FireFly-S piece | Spike GeMM after absorb (A?) | Residual / PED (A?) | Pre-threshold T10 |
|---|---|---|---|
| CLK0 Bitmap AND of 1-bit spike × 1-bit W mask; one-hot peel; fetch W; add | **Yes. Complete A.** After \(W\leftarrow\theta W\), the next layer’s operand is \(s\in\{0,1\}\). Their own Bitmap study is a 3×3 conv — native geometry for r1. | **No.** PED / native `conv_res` / BN / residual add are multi-bit continuous. AND of a gate bitmap does not skip a non-zero residual term. | Detector dumps matching pairs into **causal** IF/LIF V-reg reused “for the next time step”. T10 is a 10×10 mix of **all** t **before** the spike exists. |
| Joint rewiring + LSQ | A **if** one actually trains it on the two r1 convs (and any fused scale that they would absorb into Vth). Their Vth-absorb ≠ this net’s native projection BN using **actual batch statistics over 10×96×120×160**. | Training that only maximizes spike/W zeros can **increase** disagreement with PED. | PSN coefficients / τ are not their (W, b, Vth) triple. |
| Algo 1 silent channels | Legal on the **spike** consumer. | **Function change** if a never-spike channel still feeds PED/BN/add. | Wrong dynamics (causal T=4, no synaptic input). |
| Spatial inter-layer pipeline, membrane-free V | Two r1 convs **can** be two pipeline stages **on the spike path**. Halo/padding/im2col are ordinary 3×3 costs they already pay. | Residual identity and PED are extra **live continuous tensors** between stages. Their Sbuf formula is a **spike** line buffer. | Table I Vbuf shrinks because T is innermost-after-Co and V is a scalar recurrence. T10 state is a 10-vector, not `PCo` scalars. |
| Bias-as-bubble fill | Spike-path trick: idle cycle → add bias, maybe spike. | PED q24 / residual add have no such fake pair. | — |
| 70–90% spike sparsity they measured | Their LIF classifiers. Not a license to quote as r1 gate density. | — | — |

**Complete prior A (copy, not X):** Bitmap pair-walk on \(\{W=0\}\cup\{s=0\}\) for a 3×3 after absorb; joint prune+quant training if induced W sparsity is claimed; spatial pipeline of two spike-path conv layers.

**Hole that is still B, not A:** (i) second consumer is continuous residual/PED; (ii) T10 is noncausal and **pre-threshold**; (iii) Algo 1 / Vth-absorb / bias-bubble are spike-only.

### 1.5 What no longer fails (and what still does)

Round-1 kills that **die under absorb**:

1. “CLK0 AND of spike vector × weight mask is the wrong ALU for \(\theta_g\).” After absorb the activation **is** a 1-bit Bitmap.
2. “Synaptic current \(I=\sum w s\) with \(s\in\{0,1\}\) is not our neuron.” It **is** the spike path.
3. “Need a multiplier for \(\theta_g\).” Not on the spike path.

Kills that **remain**:

1. **Algo 1.** A channel that never spikes can still be required as a continuous residual/PED source.
2. **Scale-into-Vth** does not implement native projection BN, and does not hide a continuous residual from its consumer.
3. **Bias-as-bubble** is not a PED retirement rule.
4. **Consecutive-t V register** is not a T10 certificate.
5. Dual-consumer last-use / same-port backpressure is not in the paper.

---

## 2. Bishop — TTB / BSA / ECP / AAC / SAC

### 2.1 What the paper actually does

First (their claim) HW/SW co-design for **spiking transformers**. Work unit = **Token-Time Bundle (TTB):** \(BS_n\) tokens × \(BS_t\) time points × one feature. Tag a bundle **active** iff it contains at least one spike (L0). BSA trains a **bundle-level** L0 loss on MLP, projection, Q, and K. ECP prunes whole Q/K **bundle rows** using a bound that holds **because Q and K are binary**. Heterogeneous cores: stratifier sends high-density TTB features to a dense SAC systolic array, low-density to a SIGMA-like sparse core; attention core is **multiplier-less AAC** (AND+accumulate) then SAC for S×V. LIF: \(S[t_k]\in\{0,1\}\). They **move LIF before the last SSA linear** so \(O_{\mathrm{attn}}\) is \(W\) times **binary** activations.

Tokenizer conv exists (\(O(T H W C^2 K^2)\)) but they explicitly punt it to prior spiking-CNN accelerators. Dominant targets are MLP / QKVO projection / SSA.

### 2.2 Section quotes

**Abstract.**

> Specifically, we introduce the concept of Token-Time Bundle (TTB), a container that bundles spiking data of a set of tokens over multiple time points. … Bishop utilizes a stratifier, a dense core array, and a sparse core array to process MLP blocks and projection layers. The stratifier routes high-density spiking activation workload to the dense core and low-density counterpart to the sparse core … we introduce a novel Bundle Sparsity-Aware (BSA) training pipeline that enhances not only the overall but also structured TTB-level firing sparsity. Moreover, the processing efficiency of self-attention layers is boosted by the proposed Error-Constrained TTB Pruning (ECP), which trims activities in spiking queries, keys, and values both before and after the computation of spiking attention maps with a well-defined error bound. Finally, we design a reconfigurable TTB spiking attention core to efficiently compute spiking attention maps by executing highly simplified “AND” and “Accumulate” operations.

**§2.1 LIF is binary; they reposition LIF to kill spike-residuals.**

> A spiking output is generated if the membrane potential `Vm[tk]` exceeds `Vth`, setting `S[tk]` to 1 and resetting `Vm[tk]` to 0.

> In contrast to the method described in [64] which extensively employs multipliers due to spike residuals, we reposition the final LIF neuron layer to precede the last linear layer of each SSA block [53] as in (7). This adjustment allows for efficient multiplication-free computation of attention output `Oattn` based on the product of the weights `Wo` with the binary LIF activations `Otemp` in (8).

AT-LIF absorb is the **official** way to get “next linear sees binary” **without** deleting this net’s residual/PED tensor. Copying Bishop’s **move** is still an identity change on the continuous consumer.

**§2.2 they do not claim 3×3 is their problem.**

> The tokenizer’s computational complexity is `O(T H W C² K²)`, where `K` is the size of the employed spiking convolutional (CONV) filters. It is typically not the dominant complexity, and there has been much work targeting hardware acceleration of spiking CNNs [27, 37].

**§3.2 TTB + active/inactive + W reuse.**

> A TTB is called active if there exists at least one active spike in the bundle. Otherwise, we call it inactive. … skipping inactive bundles can be efficiently accommodated in the dataflow and avoids large overheads resulting from skipping computations at a much smaller granularity, e.g., at the spike or token level.

> Because the same multi-bit weights are used for different time points and different tokens for each feature, the 1×D weight data in one row … is shared within each TTB for processing `BSn` tokens across `BSt` time points.

**§4.1 BSA (trained, structured L0).** Without BSA, Model 1: 29% of bundles active. Eq. (9): \(Z=\|X_{\text{tokens in bundle, times in bundle, feature }d}\|_0\). Eq. (10): \(L_{bsp}=\sum Z\). \(L_{tot}=L_{CE}+\lambda L_{bsp}\) on MLP, projection, Q, K.

**§5.1 ECP — the bound is 100% only for binary K/Q. After absorb, Q/K on the spike path *are* binary, so the lemma holds there.**

> In ANNs, bounding the scores (S) given the queries (Q) and keys (K) is difficult because Q and K are continuous-valued floating point numbers.
> Differently, we explore the binary nature of the spiking Q and K … If `nab` is less than a threshold `θp,Q`, due to the binary nature of K, it is certain that any activation in a bundle that is in the corresponding row of the scores (S) tensor would be less than `θp,Q` per `S = QKᵀ`.

Figure 7 caption: “Densely Compute correlation between strong Q bundles and K bundles with **100% confidence** that all pruned attention values smaller than `min(uQ, uK)`.”

> ECP is also integrated into the training pipeline, leading to ECP-aware training to maintain high accuracy.

**§5.4 SAC / §5.5 AAC.** After absorb these ALUs are legal on the spike path:

> Leveraging the binary nature of spiking inputs, each PE executes “Select ACcumulation” (SAC) operations … A “SAC” operation is efficiently implemented by one MUX and one accumulator.

> Mode 1. … This setup executes “AND” operations between binary queries (Q) and keys (K), followed by the accumulation of partial sums in the S register …

**§6.** \(\lambda\in\{1,0.5,0.3,1.0\}\) by dataset; ECP thresholds 10 (DVS-Gesture-128) / 6 (others). \(T\in\{4,8,10,20\}\). RTL: 28 nm, die 2.96 mm², peak 627 mW, 500 MHz. Abstract vs prior SNN accelerators: 5.91× speedup, 6.11× energy. **Not** this-net numbers.

### 2.3 What MUST be trained (Bishop)

| Item | Trained? | If skipped |
|---|---|---|
| TTB packing `(BSt, BSn)` | Architectural hyperparameter. Not a loss. | Packing/reuse still work; **structured skip** is only as good as natural bundle L0 (Model 1: 29% active). |
| BSA \(L_{tot}=L_{CE}+\lambda L_{bsp}\) on MLP, projection, Q, K | **Yes.** | “Restricted opportunities for computation skipping.” |
| ECP-aware training | **Yes.** Too-large \(\theta_p\) “could degrade model performance”. | Inference bound still **true** if Q/K stay binary; they do not claim accuracy without the training hook. |
| Binary LIF / moving LIF before last SSA linear | Architectural + training choice vs Spikformer-style spike residuals. | If residuals stay continuous, they say multipliers come back. |
| Stratifier threshold \(\theta_s\) | HW hyperparameter, not trained. | |

BSA is **not** Han W-prune; it is a **firing-structure** loss. PAFT/rewiring of FireFly-S/Phi are different losses.

### 2.4 Map after absorb

Bishop’s native object is **SSA/MLP tokens**, not a 3×3 sliding window. Mapping r1 is still a **retarget**; the paper itself sends conv to SNN-CNN accelerators (FireFly-S class).

| Bishop piece | Spike GeMM after absorb (A?) | Residual / PED (A?) | Pre-threshold T10 |
|---|---|---|---|
| TTB packing + intra/inter-bundle W reuse | **A on spike MLP/projection/SSA.** Same multi-bit W reused across tokens and ticks. | One L0 tag on **spikes** cannot retire a continuous residual that is live on an empty-gate bundle (native `conv_res` even more so). | Natural axis for `BSt` **packing**. **Mismatch:** Bishop time is **independent LIF ticks** with the **same** W. This net’s T10 is a **noncausal dense mix before threshold**. Packing 10 times reuses spatial W; it does **not** skip PSN coefficients. |
| BSA \(L_{bsp}\) | A **if** retrained on **this** net’s spike work units (not CIFAR Q/K). | Loss on gates ≠ loss on PED. Must include **both** consumers or the continuous path stays dense. | A T-group L0 loss can zero time groups; noncausal PSN still needs surviving times as a dense 10-vector unless the mix itself is retrained. |
| ECP 100% bound | **A on binary SSA Q/K after absorb.** Round-1 “bound collapses” is **false** under this identity. Still **does not map to r1 conv** (no attention map). “Prune this 3×3 if nab < θp” is **not** ECP. | No. Residual is not \(S=QK^T\). | No. |
| AAC (AND-accumulate) | **A on binary Q·K after absorb.** | No. | No. |
| SAC (MUX + acc) | **A on spike-path select-add.** Same class as FireFly-S “fetch W if s=1”. | Fails for PED (need \(W\cdot x_{\mathrm{cont}}\)). | No. |
| Stratifier dense/sparse cores | A as **density split of binary spikes**. Load-balance vs \(\theta_s\) is ordinary. | Stratifier splits **density**, not gate-vs-residual. Would need a **third** consumer class they do not have. | — |
| Move LIF before last linear | How **they** get binary \(O_{\mathrm{attn}}\). AT-LIF absorb already gives binary next-GeMM **without** this move. Copying the move **destroys** the continuous residual consumer. | **Forbidden identity change.** | — |

**Complete prior A:** structured spatiotemporal packing for W reuse; BSA if one trains structured skip on **spike** units; heterogeneous dense/sparse dispatch on binary spike density; AAC/ECP **as related work for binary attention after absorb**, not for r1, not for PED.

**Hole B:** r1 is 3×3 + pre-threshold PSN + dual continuous consumer; tokenizer/3×3 is explicitly out of their scope; one TTB tag ≠ two consumers.

### 2.5 What no longer fails (and what still does)

Round-1 kills that **die under absorb** (spike path only):

1. “AAC AND(Q,K) ≠ Q·K.” After absorb Q,K are \(\{0,1\}\).
2. “ECP \(nab<\theta_p\Rightarrow|S|<\theta_p\) collapses because Q/K carry \(\theta_g\).” After absorb they do not. The paper’s own ANN contrast no longer describes the **spike** path. It **does** still describe residual/PED (continuous).
3. “SAC MUX is wrong for \(W\cdot\theta_g\).” Right for \(W\cdot s\).

Kills that **remain**:

1. **One L0 bundle tag ≠ two consumers.** PED can be live when gates are dead.
2. **ECP does not map to r1 3×3.**
3. **Reposition LIF** is the opposite of keeping residual/PED.
4. **T10 is not independent LIF ticks.**
5. Heterogeneous cores are not dual-consumer cores.

---

## 3. Phi — pattern-based hierarchical sparsity

### 3.1 What the paper actually does

SNN (CNN **and** transformer) accelerator that treats **binary activation rows** as a small codebook. Two-level hierarchy:

- **Level 1 (vector-wise):** each K-tile row is replaced by one of \(q\) pre-defined 16-bit patterns (default \(k=16\), \(q=128\)). Pattern × weight tile is a **Pattern-Weight Product (PWP)**, computed **offline**. Runtime L1 is index lookup + add.
- **Level 2 (element-wise):** residual of pattern vs actual spike row, encoded in \(\{1,-1\}\) (add the missing 1, or subtract a false 1). Unstructured, packed, processed by a reconfigurable adder tree.

Calibration: k-means on Hamming distance of binary rows (Alg. 1), independent per model/dataset/layer/K-partition; filter all-zero and one-hot rows. Optional **Pattern-aware Fine-tuning (PAFT):** few-epoch Hamming regularizer \(R=\sum_\ell N_\ell\sum H(\mathrm{Act},\mathrm{Pattern})\), \(\mathrm{Loss}=\mathrm{Loss}_{orig}+\lambda R\). Without PAFT they claim **lossless** vs bit-sparsity; with PAFT ~1.26× extra runtime, minor accuracy drop.

Hardware: Preprocessor (systolic pattern matcher + compressor + conflict-aware packer), L1 processor (16-to-8 PWP crossbar + adder tree + PWP prefetcher), L2 processor (8-channel 32-SIMD reconfigurable adder tree), LIF array. Tiling K-first so each output tile feeds LIF then the next layer’s matcher. 28 nm, 0.662 mm², 346.6 mW, 500 MHz. Vs Stellar: **3.45×** speedup, **4.93×** energy (their Table 2 / §5.3). Models: VGG-16, ResNet-18, Spikformer, SDT, SpikeBERT, SpikingBERT; CIFAR / CIFAR10-DVS / SST / MNLI.

### 3.2 Section quotes

**Abstract (the hierarchy that is now A on the spike path).**

> Phi introduces a two-level sparsity hierarchy: Level 1 exhibits vector-wise sparsity by representing activations with pre-defined patterns, allowing for offline pre-computation with weights and significantly reducing most runtime computation. Level 2 features element-wise sparsity by complementing the Level 1 matrix, using a highly sparse matrix to further reduce computation while maintaining accuracy.

**§1 binary identity + why patterns exist.**

> SNNs employ spiking neurons … where neurons react to binary coded spike activations (0 or 1) from previous layers …
> We hypothesize that this phenomenon arises from the binary nature of SNN activations, consisting solely of 0s and 1s, which enforces a more structured distribution.

After absorb, **this hypothesis applies to the spike path**. It does **not** apply to residual/PED (continuous) or to pre-threshold T10 (continuous mix).

**§2.1 bit sparsity = skip zeros, accumulate on ones.** Same class as FireFly-S / Prosperity bit-sparsity; Phi’s increment is **patterns beyond zeros**.

**§3.1 L2 is a signed bit correction, not residual/PED.**

> 1. One to zero mismatch: … we set a 1 at the corresponding position in the Level 2 matrix as a correction term
> 2. Zero to one mismatch: … we set a -1 at the corresponding position to ensure correctness
> … the summation of activations from Level 1 and Level 2 precisely matches the original activation matrix.

If L2 sparsity would be worse than raw bit sparsity, they **assign no pattern** and fall back to ordinary bit sparsity. L2 \(\{1,-1\}\) is **not** this net’s residual tensor. Renaming L2 → “PED residual” is a reskin.

**§3.2 calibration; §3.3 PAFT.** Hamming distance = number of L2 nonzeros. PAFT is optional; “only a few epochs (e.g., 5)”. Fine-tunes a **pre-trained** SNN — that is a **student change** relative to frozen Motion C12 / H67 / ep34.

**§4 architecture.** Matcher broadcasts a spike row to 128 pattern units, popcount of difference vs popcount of the row, pick min; compressor drops all-zero L2 rows; packer packs 8 units (label / index / value) with bank-conflict windows; L1 skips unused PWP (average 27.73% of PWPs used per tile; prefetcher loads only those). Output of L1+L2 → **LIF neuron array** (one consumer).

**§5.4 / Table 4.** Example: VGG16-CIFAR10 bit density 8.7%, L1 7.5%, L2 +1 1.4%, L2 −1 0.1%. Claimed theoretical 4.5× over bit sparsity / 38× over dense (averaged). Random binary matrices still show patterns but **weaker** speedup than SNN activations.

**§6.2 they themselves refuse DNN-continuous as the native case.**

> Phi uniquely leverages the binary nature of SNNs … CGNet / SIGMA … target zero elements only. In contrast, Phi takes advantage of pattern-based sparsity to skip significant computation related to **nonzeros** in activations, thanks to the unique binary features of SNNs.

They mention bit-sliced DNNs as a **possible extension**, not as the evaluated design.

### 3.3 What MUST be trained / calibrated (Phi)

| Item | Trained? | If skipped |
|---|---|---|
| Pattern codebook (k-means, \(k=16\), \(q=128\), per layer/partition) | **Calibration** on a train subset. Not a gradient step. | Codebook is wrong for this net’s spike rows; L2 density explodes; may fall back to bit sparsity (FireFly-S class). |
| Offline PWP = pattern × **absorbed** \(W\) | Compile. Must rebuild if \(W\leftarrow\theta W\) changes \(W\). | Stale PWPs are a **function error**. |
| PAFT Hamming regularizer | **Optional** fine-tune. | Lossless Phi (w/o PAFT) still A; their extra 1.26× is not free. PAFT **moves** the frozen student; AEE gates apply. |
| Matcher / packer / L1-L2 split | Hardware. Not trained. | |

### 3.4 Map after absorb

| Phi piece | Spike GeMM after absorb (A?) | Residual / PED (A?) | Pre-threshold T10 |
|---|---|---|---|
| L1 16-bit binary patterns + offline PWP | **Yes. Complete A.** After absorb, spike rows are \(\{0,1\}^k\). PWP is compile of `pattern × (θW)`. | **No.** Residual is not a 16-bit binary pattern. Hamming k-means on PED is a different alphabet (their own DNN contrast). | T10 mix is **before** spikes exist. Compiling a **constant 10×10** is a cousin of “offline product”, but the alphabet is **continuous mix coefficients**, not 128 Hamming centers of spike rows. Calling T10 “Phi L1” is a reskin. |
| L2 \(\{1,-1\}\) correction | **A on spike-path mismatch to codebook.** | **Not PED.** Signed bit to restore \(s=\mathrm{pattern}+L2\). Continuous residual does not add/subtract a weight row; it is a second tensor. | No. |
| Runtime matcher + packer + L1/L2 units | A as the HW for the above. | Would need a dense (or separately sparse) continuous pipeline they do not build. | Matcher expects 1-bit rows. |
| PAFT | A **only if** one accepts a student change and re-gates AEE. Frozen ep34 does not include PAFT. | Hamming-to-pattern on gates can **hurt** PED. | Hamming on spikes does not compile T10. |
| One LIF consumer after L1+L2 | Matches **spike** next-layer. | Dual last-use not in the paper. | LIF-after-GeMM is **after** threshold; T10 is before. |

**Complete prior A:** k-means binary codebook, offline PWP on absorbed \(W\), L2 \(\{1,-1\}\) pack, matcher/packer/L1-L2 split, optional PAFT as a **named** training hook — **on the spike path**.

**Hole B:** residual/PED; pre-threshold T10; PAFT vs frozen student; one consumer; event-OF / r1 3×3 geometry is not their evaluated object (they do GeMM tiles, conv via lowering).

### 3.5 What no longer fails (and what still does)

Round-1 / W1 “weak X if T10 is not Phi L1 on a different alphabet” **collapses as an X story for the spike path**: the alphabet **is** binary after absorb, so Phi is A there.

Remaining:

1. **T10 lifting is not Phi L1.** Different time, different alphabet (continuous pre-threshold mix vs 16-bit spike patterns), different consumer count.
2. **L2 \(\{1,-1\}\) is not residual/PED.**
3. **PAFT is not free** on a frozen student.
4. **Dual last-use** of a source word that also feeds PED is not in Phi.
5. Their 3.45× / 4.93× vs Stellar is not a same-port number on DSEC.

---

## 4. Synthesis after absorb

### 4.1 One table

| Mechanism | Must train / calibrate | HW primitive | Spike path after absorb | Residual / PED | Pre-threshold T10 |
|---|---|---|---|---|---|
| FireFly-S rewiring + joint LSQ | **Yes** | Induced W zeros + 4-bit integer LIF | A if copied as training | Can hurt PED | No |
| FireFly-S Bitmap dual-side walk | Spike side not trained; W side needs prune | AND mask, one-hot peel, fetch W, acc V | **Complete A** | No | No consecutive-t V |
| FireFly-S Algo 1 silent channel | Eval on IF/LIF | Drop never-spike channels | Spike-only | **Illegal** if PED needs the channel | Wrong dynamics |
| Bishop TTB packing | Not trained | Pack \(BS_n\times BS_t\), skip inactive, reuse W | **A** on MLP/SSA | One L0 tag ≠ two consumers | Packing ≠ PSN math |
| Bishop BSA \(L_{bsp}\) | **Yes** | Structured TTB L0 | A if **redefined** on this net’s spike units | Must put both consumers in the sum | Only if T-groups are the work unit |
| Bishop ECP | **Yes** (aware) + binary Q/K | Row prune by \(nab<\theta_p\) | **A on SSA**; **not** r1 3×3 | No | No |
| Bishop AAC / SAC | Binary spike (now true after absorb) | AND+acc / MUX+acc | **Complete A** on spike/attention | Not PED | No |
| Bishop move-LIF-before-last-linear | Arch. choice | Kill spike-residuals | How they get binary next-GeMM; **absorb already does this** | **Destroys** residual identity | — |
| Phi L1 PWP | Calibrate codebook; compile PWP on **absorbed** W | Index → precomputed pattern×W | **Complete A** | No | Not T10 |
| Phi L2 \(\{1,-1\}\) | Follows assignment; optional PAFT | Packed signed bit add/sub of W rows | **A** as pattern residual | **Not** PED | No |
| Phi PAFT | Optional fine-tune | Hamming regularizer | Student change; AEE gates | Can hurt PED | No |

### 4.2 What is complete A (must copy, not rename)

On the **post-absorb spike path**, a TCAS-II related-work / denominator that skips any of these is incomplete:

1. **FireFly-S Bitmap dual-side** (and, if W sparsity is claimed, their joint rewiring+LSQ, not a post-hoc magnitude cut).
2. **Bishop TTB + SAC/AAC**, and ECP **if** the letter talks about binary SSA.
3. **Phi two-level pattern hierarchy** (L1 PWP + L2 \(\{1,-1\}\) + matcher). Do not call Bitmap skip “Phi”, and do not call Phi “dual-side zeros”.
4. IDENTITY’s other binary GeMM priors (Prosperity product sparsity, Gustav NRV/CPTB, LoAS) — same consumer class; not re-derived here.

Copying HW skip **without** the paper’s training/calibration is not “the method”: W-side dual-side needs prune+recover; BSA needs \(L_{bsp}\) on **this** work unit; Phi needs a codebook on **this** net’s spike rows and PWPs rebuilt after \(W\leftarrow\theta W\); ECP-aware and PAFT are optional accuracy hooks, not free FPS.

### 4.3 What is **not** A — remaining X

After absorb, **dual-side sparsity and pattern hierarchy apply to binary spike×W. Residual/PED does not.** The increment, if any, can only sit on objects those papers do not cover (`SCOPE.md` / `PROBLEM.md`):

1. **Pre-threshold noncausal T10 mix (PSN / lifting CSE).** Continuous arithmetic on a 10-vector **before** \(H(m-\theta)\). FireFly-S consecutive-t V, Bishop per-tick LIF, Phi LIF-after-GeMM all run **after** a spike exists. Compiling T10 is **not** Phi L1 (different alphabet), **not** Bitmap AND, **not** TTB skip. Lifting AEE already fails the relative +0.005 gate (1.232979368 vs 1.219801338); any T10 X must re-gate, not quote their 3.45×.

2. **Residual / PED / I24 as a second continuous tensor.** Dual consumers: binary gate after absorb **and** continuous residual. FireFly-S detector, Bishop bundle L0, Phi Hamming patterns are defined on \(\{0,1\}\). A skip legal for the gate can be illegal for PED/BN/add. None of the three trains a dual-consumer last-use. Phi L2 \(\{1,-1\}\) is a **signed bit correction of spike rows**, not PED.

3. **Native projection conv + BN + residual add**, with BN on **actual batch statistics over the full 10×96×120×160 domain.** FireFly-S “scale absorbed into Vth” and AT-LIF “\(\theta\) absorbed into next \(W\)” are both **scalar folds on the spike path**. Neither is live full-domain BN on the residual path.

4. **Same-port / same-state / same-backpressure dual-path service.** Measured on this net: always-ready 6938→5354 (−22.83%, part generic fusion); long backpressure **both 8088**. FireFly-S/Bishop/Phi each have **one** accumulate/LIF consumer. TCAS-II gate is net service ≥15% under that discipline, not GOP/s vs Stellar/PTB.

5. **Event-camera 2D optical flow, DSEC valid825 AEE**, student Motion C12 / H67 / ep34. Their models are classifiers / DVS-Gesture / speech. Bishop **punts** the tokenizer conv. Frozen AEE 1.219801338; abs ≤1.259 and relative ≤+0.005. Integer gates/I24/PED q24 already 0-diff vs model on two windows — that does **not** make residual binary.

6. **Training hooks that move the frozen student** (BSA, ECP-aware, PAFT, rewiring+LSQ) are **not** X by themselves; they are A’s software half. Using them on this net is a new training run under the AEE kill-gate, not a circuit title.

**Forbidden as X under this freeze:** “continuous unabsorbable \(\theta_g\)”; HBG-RP int8 payload; renaming Bitmap→Phi or L2→PED; Bishop-ECP on r1 3×3; T10-as-Phi-L1; OpenROAD-as-PPA; FPS products; CIM.

### 4.4 Relative prior vs this session’s X (one paragraph)

FireFly-S, Bishop, and Phi are **complete A** for post-absorb binary spike×W: dual-side zero skip (FireFly-S), structured token-time packing and binary AND/MUX ALUs (Bishop), pattern codebook + offline PWP + signed bit correction (Phi). After absorb those binary lemmas **hold**, so they are no longer kill-gates on the spike path. They remain kill-gates if applied to **residual/PED** or if T10 is described as their dataflow. The only honest X left is on the objects they do not build: **pre-threshold noncausal T10**, **continuous residual/PED dual-consumer last-use**, **full-domain native BN**, **same-port service** on an event-flow r1 chain. Anything that is only Bitmap AND, TTB skip, AAC, or Phi L1/L2 on the spike path is a reskin the venue will punish.

---

## 5. Quotes not to launder as this-net results

Do not write as if measured here:

- FireFly-S 10,047 / 3,683 / 2,327 FPS/W; 85–95% sparsity; 4-bit; 333 MHz; KV260; 70–90% spike sparsity.
- Bishop 5.91× / 6.11× vs prior SNN accelerators; 2.96 mm², 627 mW, 500 MHz, 28 nm; ECP thresholds 6/10; 29% bundles active without BSA.
- Phi 3.45× / 4.93× vs Stellar; 0.662 mm², 346.6 mW; \(k=16\), \(q=128\); L2 density 3.05% vs bit density 16.37%; 1.26× from PAFT; 4.5× theoretical over bit sparsity.

Do not cite volume/issue for FireFly-S from this txt. Do not cite Han/LSQ/gradient-rewiring/PTB/Spikformer/Stellar as if they were read in this R2L3 pass; they appear only as **those papers’** references.

Phi arXiv 2505.10909 **was local** (`p0_txts/2505.10909.txt`). HTML `https://arxiv.org/html/2505.10909v1` was not used.

End of R2L3. Sources: the three local txts + this session `PROBLEM.md` / `../IDENTITY_ATLIF.md` / `SCOPE.md` only.
