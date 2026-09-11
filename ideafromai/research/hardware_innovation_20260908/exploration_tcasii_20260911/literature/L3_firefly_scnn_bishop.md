# L3 FireFly-S / SCNN / Bishop

Date of freeze: 2026-09-11. Full local texts only. No invented citations; venue/page not in a local txt is not filled in.

| Paper | Local txt | Identity in this txt |
|---|---|---|
| FireFly-S | `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2408.15578.txt` | Title *FireFly-S: Exploiting Dual-Side Sparsity for Spiking Neural Networks Acceleration with Reconfigurable Spatial Architecture*; authors Tenglong Li, Jindong Li, Guobin Shen, Dongcheng Zhao, Qian Zhang, Yi Zeng; arXiv:2408.15578v3 [cs.AR] 29 Jan 2026; manuscript created 29 August 2024. Header is the generic “JOURNAL OF LATEX CLASS FILES”. Session label TCAS-I is **not** a volume/page found in this txt. |
| SCNN | `.../p0_txts/1708.04485.txt` | Title *SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks*; authors Angshuman Parashar, Minsoo Rhu, Anurag Mukkara, Antonio Puglielli, Rangharajan Venkatesan, Brucek Khailany, Joel Emer, Stephen W. Keckler, William J. Dally. arXiv:1708.04485v1 [cs.NE] 23 May 2017. Footnote: “A version appears in the 44th IEEE/ACM International Symposium on Computer Architecture (ISCA-44), 2017.” |
| Bishop | `.../p0_txts/2505.12281.txt` | Title *Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning*; authors Boxun Xu, Yuxuan Yin, Vikram Iyer, Peng Li. arXiv:2505.12281v1 [cs.NE] 18 May 2025. Footer: ISCA ’25, June 21–25, 2025, Tokyo, Japan. DOI in txt: https://doi.org/10.1145/3695053.3731063 |

This-net identity used for mapping is **only** `PROBLEM.md` / `SCOPE.md` of this session: event-camera 2D optical flow; frozen student Motion C12 / H67 / ep34; **ATLIF continuous threshold amplitude θg, not binary spikes**; expensive region patch-embed residual r1 (two convs) plus noncausal T10 PSN; dual consumers after source = spike/gate path **and** continuous residual/PED path; native projection convolution + BN + residual add also exist. Do not change paper identity to binary ATLIF.

Numbers below that are 2.7×, 85% sparsity, 5.91×, etc. belong to those papers’ own networks. They are not this-net cycle share.

---

## 0. Mapping target (this net, not theirs)

- **r1:** two 3×3 convs in the patch residual chain.
- **T10 PSN:** noncausal 10-step time mix at this neuron, not causal LIF leak-and-reset over T.
- **Dual consumers of the same source:** (i) spike/gate path, (ii) continuous residual/PED path. A skip that is legal for gates can still be illegal for PED / BN / residual add.
- **θg:** when the neuron fires, the delivered amplitude is continuous θg, not `{0,1}`. Zeros still exist when it does not fire. Integer gates / I24 / PED q24 are already bit-exact on two captured windows (`PROBLEM.md`); that does not make the neuron binary.

Three mechanisms extracted, then mapped:

1. FireFly-S: dual-side sparsity **training + Bitmap HW**.
2. SCNN: Cartesian-product sparse conv dataflow.
3. Bishop: Token-Time Bundle + BSA + ECP + AAC/SAC.

---

## 1. FireFly-S — dual-side sparsity training + hardware

### 1.1 What the paper actually does

Software: **joint** gradient-rewiring pruning and an LSQ variant **during training**, targeting >85% **weight** sparsity and 4-bit integer W/bias/Vth. Spike sparsity is treated as already there. Hardware: spatial FPGA mapping, one core per layer, Bitmap dual-side detector that ANDs a 1-bit spike vector with a 1-bit weight mask and walks only matching non-zero pairs into IF/LIF. Dataflow is rebuilt so the temporal loop sits inside the output-channel loop (membrane storage-free consecutive V) and channel loops sit inside pixel loops (inter-layer pipeline). Models SCNN5/7/9 are stacks of **3×3** convs (`c3p1`) + maxpool, T=4 LIF, MNIST / DVS-Gesture / CIFAR-10.

It is an SNN **classifier** FPGA overlay-replacement, not an event-flow transformer, not a residual dual-consumer, not a noncausal PSN.

### 1.2 Section quotes

**Abstract (dual-side claim + training + Bitmap detector).**

> While current SNN hardware accelerators often prioritize temporal spike sparsity, exploiting sparse synaptic weights offers significant untapped potential for even greater efficiency. … On the software side, we propose a novel algorithmic optimization framework that combines gradient rewiring for pruning and modified Learned Step Size Quantization (LSQ) tailored for SNNs, achieving a remarkable weight sparsity exceeding 85% and enabling efficient 4-bit quantization with negligible accuracy loss. On the hardware side, we present an efficient dual-side sparsity detector employing a Bitmap-based sparse decoding logic to pinpoint the positions of non-zero weights and input spikes. The logic allows for direct bypassing of redundant computations.

**§I contributions (what must exist together).**

> 1) FireFly-S proposes an algorithmic optimization framework that combines gradient rewiring-based pruning and a Learned Step Size Quantization (LSQ) variant specifically tailored for SNNs, achieving over 85% weight sparsity with 4-bit quantization during training for various SNN models.
> 2) FireFly-S employs a Bitmap-based sparse decoding logic in the dual-side sparsity detector, enabling efficient exploitation of sparsity found in both activation spikes and synaptic weights. This approach directly skips all computations involving zero values …
> 3) FireFly-S leverages a fully unfolded spatial architecture with inter-layer pipelining, eliminating off-chip memory access and concurrently processing multiple layers.

**§II.A neuron (binary spike in `{0, Vreset}` semantics).** Equations (2)–(4): `Ii(t) = Σ_j wij sj(t) + bi`; leaky V update; `Vi ← Vreset` if `Vi > Vth` else keep V. The synaptic input is **weight × binary spike** `sj(t)`, not a continuous amplitude.

**§II.B why pruning is the software half of dual-side.**

> These pruning techniques provide two significant benefits for hardware implementation. 1) Lower memory footprint. The reduction of synaptic connections can lead to fewer spikes and inactive neurons … 2) Enhanced computational efficiency. Fewer operations are needed for pruned connections … For a network with 75% sparsity, the dense method processes 75% of computations that are ineffectual on average. However, by incorporating sparsity detection logic, the same performance can be achieved with just a quarter of the processing elements, eliminating wasted computations.

> However, while pruning can induce dual-side sparsity in both weights and activations, and quantization can further boost the compression ratio, there is limited work on effectively combining these two techniques during training, especially in a hardware-friendly manner that considers the efficiency constraints presented by neuromorphic hardware for SNNs, such as fully integer arithmetic within neuron computations.

**§III.A compression sequence (joint, inside training; not post-hoc).**

> An alternative strategy is to jointly optimize pruning and quantization, where weight values and connections are simultaneously refined. … By embedding pruning and quantization within the training process, this strategy not only enhances the balance between compression and accuracy but also eliminates the need for additional post-training steps, such as retraining or fine-tuning. This joint optimization is precisely the strategy employed by FireFly-S.

**§III.B gradient rewiring (what is trained for W sparsity).**

> The pruning algorithm we adopt is based on gradient rewiring [19]. In this method, each synaptic weight `w` is represented as `w = s ⊙ ReLU(θ)`, where `θ` represents the synaptic parameter … During training, `θ` is influenced by a Laplace prior … This penalty pushes some `θ` values towards non-positive numbers, thereby eliminating the corresponding connections.
> Our approach builds on this by applying weight decay to `θ` … we apply this pruning strategy to both convolutional and fully connected layers, including bias parameters …

**§III.B Algorithm 1 silent-channel prune (binary threshold test on bias-only V, over T).**

> Due to the adoption of fine-grained pruning, there is a likelihood that some weights within an output channel dimension might be entirely zero. In ANNs, even if the weights are zero, the presence of a bias can still result in a non-zero output for a layer. … However, SNNs exhibit different behavior due to the presence of a neuron threshold. In SNNs, even the accumulated bias across the temporal dimension might not exceed the threshold, particularly when the neuron model includes leaky currents. Consequently, if the neuron remains inactive (silent) throughout, it indicates that the kernel of that output channel could potentially be pruned.

Algo 1 (quoted structure): for each output channel, for `t = 0 .. T-1`: `V[co][t] ← V[co][t-1] + b[co]`; if `V > Vth[co]` mark channel active and break. Channels never crossing Vth are dropped. This is a **causal leaky/IF threshold test with no synaptic input**, not a T10 matrix.

**§III.B LSQ variant (what is trained for 4-bit integer neuron).**

> During training, weights, biases, and thresholds are quantized and dequantized, with the dequantized values used for forward and backward propagation. In contrast, inference uses only quantized parameters for forward propagation …
> 1) We employ learnable per-channel scales … 2) … our method jointly optimizes shared quantization scales for weights, biases, and neuronal thresholds within each channel during training.
> By sharing the quantization scale across weights within the same channel, along with the corresponding bias and neuronal threshold, our approach ensures consistent scaling … enabling efficient integer-only inference. Additionally, due to the unique properties of spiking neurons, there is no need to rescale per-channel activations during inference compared to ANN quantization, as the per-channel quantization scales are absorbed into the membrane thresholds.

**§IV.B Bitmap vs COO/CSR (why 1-bit presence, and the 3×3 experiment).**

> Bitmap representation, however, allows for simultaneous decoding and direct identification of matching non-zero pairs, mitigating these performance bottlenecks.
> This analysis … simulating a convolutional operation with parameters `(T, Co, Ci, Fho, Fwo, Kh, Kw) = (4, 64, 32, 16, 16, 3, 3)`.
> Experiments indicate that the sparsity of activation spikes ranges from 70% to 90%. Therefore, the sparsity of the weights must be significantly higher to reach the threshold where Bitmap becomes less efficient than COO or CSR.

**§V.B Dual-side Sparsity Detector (the actual circuit).**

> After passing through the Dataflow Orchestrator, the activation spikes are buffered and aligned with their corresponding weight mask data, which is represented in Bitmap format.
> At CLK0, a bitwise AND operation is executed between the vector pair, yielding a result denoted as x, which identifies active events.
> … a one-hot encoded output, y, is generated through the logical expression `y = x ∧ ¬(x−1)` … The register holding x is then updated using `x = x ∧ ¬y` … This process repeats until all non-zero pairs in x are processed …
> Finally, at CLK6, the weight data is fetched from the RAM and used to perform IF/LIF neuron model computation. … the voltage for each time step is temporally stored in the register and reused for the next time step.

Idle-cycle fill: if the AND is all zeros, they **inject bias as a fake pair** (`sj=1` with `w=bias`) rather than stall. That only makes sense if the consumer is “add bias then compare Vth then emit a binary spike”.

**§VI models are 3×3 conv stacks, T=4 LIF.** Table II: SCNN5/7/9 configs built from `c3p1` = “a 3x3 convolution kernel with padding of 1”; “we utilized LIF neurons in all experiments, with a time step of 4.” Table VII: Quant+Prune sparsities 93.82% / 93.26% / 86.12% at 4-bit.

**§VII conclusion (training is not optional for the HW story).**

> On the software side, our approach includes a quantization-aware and pruning-aware training method, which enables the training of SNN models at 4-bit quantization and 85–95% sparsity with competitive accuracy loss. … On the hardware side, we capitalize on the natural sparsity of spikes inherent to SNNs and the induced sparsity of pruned weights … effectively exploited by our Dual-side Sparsity Detector.

### 1.3 What MUST be trained (FireFly-S, as stated)

| Item | Trained in the paper? | If skipped |
|---|---|---|
| Weight sparsity via gradient rewiring + Laplace/ℓ2 on `θ`, including conv **and** FC **and** bias | **Yes, during training** | Detector still skips *existing* W zeros, but 85–95% W sparsity and the Bitmap vs COO/CSR operating region are not free. Dual-side is then almost only spike-side. |
| Joint 4-bit LSQ of **W + bias + Vth** with **shared per-channel scale** | **Yes, during training** (STE) | Integer-only IF/LIF path and “scale absorbed into threshold” break. Post-stat min-max on W applied to bias/Vth is exactly what they say degrades accuracy. |
| Silent-channel Algo 1 | Evaluation on the trained IF/LIF (bias vs Vth over T), then drop channels | Legal only if the **only** consumer is “does this channel ever spike”. |
| Spike sparsity | **Not** a training loss. “Natural sparsity of spikes inherent to SNNs.” | Inference-time, from thresholding. |

They explicitly reject prune-then-quant or quant-then-prune as a separate post-process.

### 1.4 Map onto r1 3×3 + T10 PSN + dual consumer

| FireFly-S piece | r1 two 3×3 | T10 PSN | Dual consumer |
|---|---|---|---|
| Bitmap dual-side detector | **Closest map.** Their own Bitmap study is a 3×3 conv `(Kh,Kw)=(3,3)`. Orchestrator is implicit im2col over a 3×3 window. Can AND a **gate** bitmap with a **W** bitmap on each r1 conv. | Not a PSN. Detector dumps matching pairs into **causal** IF/LIF V-reg reused “for the next time step”. Noncausal T10 is a 10×10 mix of **all** t; you cannot stream t and drop V. | Detector’s output is “update V, compare Vth, emit spike”. PED / native `conv_res` / BN / residual add are **not** this consumer. Skipping a zero **gate** does not skip a non-zero continuous residual/PED term. |
| Joint rewiring+LSQ | Must be run on **both** r1 convs (and any fused BN/scale that they would absorb into Vth). Absorbing per-channel scale into **Vth** is not the same as this net’s native projection BN using **actual batch statistics over the full 10×96×120×160 domain** (`PROBLEM.md`). | PSN coefficients / τ / θg are not their (W, b, Vth) triple. Sharing one scale across W, b, Vth does not give you a compiled T10 integer PSN. | A channel that never spikes can still feed PED/BN/add. Algo 1 would delete it; that is a **function change** on the continuous consumer. |
| Spatial inter-layer pipeline, membrane-free V | Two r1 convs **can** be two pipeline stages. Halo/padding and im2col orchestrator are ordinary 3×3 costs they already pay. | Their Table I Vbuf shrinks because T is innermost-after-Co and V is a scalar recurrence. T10 PSN state is a 10-vector (or a compiled time mix), **not** `PCo` scalars. | Residual identity and PED are extra live tensors between stages. Their Sbuf formula `((Kh−1)×Fwo+Kw)×T×Ci` is a **spike** line buffer, not a continuous PED buffer. |
| 70–90% spike sparsity they measured | Only a property of their LIF classifiers. Not a license to quote as r1 gate density. | — | — |

**Complete prior A (copy, not X):** Bitmap pair-walk on `{W=0} ∪ {gate=0}` for a 3×3; joint prune+quant training if you want induced W sparsity; spatial pipeline of two conv layers.

**Hole B on this net:** (i) source amplitude is θg, so the “activation” side of the AND is not a 1-bit spike vector unless you **throw away θg**; (ii) second consumer is continuous; (iii) T10 is noncausal.

### 1.5 What fails if θg is continuous amplitude, not binary

Still legal: skip a MAC when the **source value is exactly 0** or W is 0; store a W bitmap; train rewiring on W.

Fails or changes meaning:

1. **CLK0 bitwise AND of spike vector × weight mask.** A continuous θg vector is not a Bitmap of events. Zero-detect on multi-bit θg is a different comparator; non-zero θg needs a **multiplier** (or scale-and-add), not “fetch W and add it as if `s=1`”.
2. **Synaptic current (2): `Ii = Σ wij sj` with `sj ∈ {0,1}`.** Replace `sj` by θg and the arithmetic, the 4-bit datapath, and “no DSP, LUT-only 4-bit” story all move.
3. **“No need to rescale per-channel activations … scales are absorbed into the membrane thresholds.”** That sentence is for **binary spikes**. Continuous θg **is** the activation that later layers multiply; it cannot be absorbed only into this layer’s Vth if PED/residual read the amplitude.
4. **Algo 1 silent channels.** Tests whether **bias-only causal V** ever crosses Vth. It does not test whether a continuous residual/PED consumer still needs that channel, and it is not a T10 PSN certificate.
5. **Bias-as-bubble fill.** Idle cycle is replaced by “add bias, maybe spike”. PED q24 / residual add have no such fake pair.
6. **Integer-only IF/LIF with V in a register reused next t.** Noncausal T10 does not have a next t in that sense.

If one **redefines** the gate path as binary `g∈{0,1}` and keeps θg only for the amplitude consumer: FireFly-S HW covers **only the gate MAC into a LIF**. The continuous consumer is then a second, dense (or separately sparse) pipeline that the paper does not build. Training that only maximizes spike sparsity / silent channels can increase the **disagreement** between the two consumers.

---

## 2. SCNN — Cartesian-product sparse conv

### 2.1 What the paper actually does

Inference accelerator for **pruned CNNs**. Dual sparsity = **static W zeros from training-time pruning** + **dynamic ReLU zeros at inference**. Dataflow **PT-IS-CP-sparse**: fetch a compressed vector of F non-zero weights and I non-zero activations; compute the **full Cartesian product** F×I (every non-zero W times every non-zero A is a useful partial sum); derive output (k,x,y) from the **embedded coordinates**, not from dense loop indices; **scatter** into a **dense** accumulator because products are not contiguous and density recovers before ReLU. Weights grouped as compressed blocks `Kc×R×S`; activations as `Wt×Ht` per input channel. PE array planar-tiles the activation plane; **output halos** exchanged. Encoding: 16-bit value + 10-bit / 4-bit run-length of zeros.

Filters “usually 1×1 or 3×3”. Not an SNN, not a PSN, not a residual dual-consumer.

### 2.2 Section quotes

**Abstract.**

> … exploiting the zero-valued weights that stem from network pruning during training and zero-valued activations that arise from the common ReLU operator applied during inference. Specifically, SCNN employs a novel dataflow that enables maintaining the sparse weights and activations in a compressed encoding … the SCNN dataflow facilitates efficient delivery of those weights and activations to the multiplier array, where they are extensively reused. In addition, the accumulation of multiplication products are performed in a novel accumulator array.

**§I Cartesian product + scatter (the mechanism).**

> At the heart of the SCNN design is a processing element (PE) with a multiplier array that accepts a vector of weights and a vector of activations. Unlike previous convolutional dataflows … the SCNN dataflow only delivers weights and activations to the multiplier array that can all be multiplied by one another in the manner of a Cartesian product. … Finally, only non-zero weights and activations are fetched from the input storage arrays and delivered to the multiplier array. … However, since the products generated by the multiplier array cannot be directly summed together, SCNN tracks the output coordinates associated with each multiplication and sends the coordinate and product to a scatter accumulator array for summing.

**§II Motivation: 3×3 is the common case; sparsity sources are asymmetric.**

> The convolutional layers … are characterized by a set of filters that are usually 1×1 or 3×3, and occasionally 5×5 or larger.

> The primary technique for creating weight sparsity is to prune the network during training. Han, et al. developed a pruning algorithm that operates in two phases [15]. First, any weight with an absolute value that is close to zero (e.g. below a defined threshold) is set to zero. … Second, the remaining network is retrained, to regain the accuracy lost through naïve pruning. … The process can be iteratively repeated …

> Activation sparsity occurs dynamically during inference and is highly dependent on the data being processed. Specifically, the rectified linear unit (ReLU) function that is commonly used as the non-linear operator in CNNs forces all negatively valued activations to be clamped to zero. … 50–70% of the activations are clamped to zero.

> Typical layers can reduce work by a factor of 4, and can reach as high as a factor of ten. [product of per-layer W density and A density]

**§III.A PT-IS-CP-dense inner core (still Cartesian, still 3×3 coordinates).**

> To exploit the parallelism of many multipliers within a PE, we fetch a vector of F filter-weights from the weight buffer and a vector of I inputs from the input activation buffer. These values are delivered to an array of F×I multipliers to compute a full Cartesian product of output partial-sums. Each product yields a useful partial sum such that no extraneous fetches or computations are performed. PT-IS-CP-sparse will exploit this same property …

Figure 4 inner: `acc_buf[k][x][y] += in[i]*wt[f]` with `k,x,y` from `Kcoord/Xcoord/Ycoord` of the **dense** R×S geometry.

**§III.B sparse form + dense accumulator until ReLU (critical for residual).**

> weights are grouped into compressed-sparse blocks at the granularity of an output-channel group, with of `Kc × R × S` weights encoded into one compressed block. Likewise, input activations are encoded at the granularity of input channels, with a block of `Wt × Ht` encoded into one compressed block. … the multiplier array computes the full cross-product of F×I partial sum outputs, with no extraneous computations. Unlike a dense architecture, output coordinates are not derived from loop indices in a state machine but from the coordinates of non-zero values embedded in the compressed format.

> Even though calculating output coordinates is trivial, the multiplier outputs are not typically contiguous as they are in PT-IS-CP-dense. Thus the F×I multiplier outputs must be scattered to discontiguous addresses within the `Kc × Wt × Ht` output range. Because any value in the output range can be non-zero, the accumulation buffer must be kept in a dense format. In fact, output activations will probabilistically have high density even with a very low density of weights and input activations, until they pass through a ReLU operation.

**§III inter-PE 3×3 halo (ordinary, must be copied).**

> strictly partitioning both input and output activations into `Wt × Ht` tiles does not work because the sliding-window nature of the convolution operation introduces cross-tile dependencies at tile edges. These dependencies are called halos. [input-halo replication vs output-halo partial-sum exchange] Our PT-IS-CP-dense dataflow uses output halos …

**§IV PE:** 4×4 multiplier array, 16×32 scatter crossbar, 32 accumulator banks, `A = 2×F×I` to cut bank conflicts, double-buffered acc, PPU does halo exchange **then** ReLU/pool/dropout **then** compress into OARAM. Run-length: “Four bits per index allows for up to 15 zeros to appear between any two non-zero elements.”

**§VII vs Eyeriss/Cnvlutin/Cambricon-X/EIE:** SCNN keeps **both** W and A compressed **through almost the entire computation**, and does not even issue a multiply if either operand is zero. EIE is called out as FC-only; SCNN is conv.

**§VIII:** beats dense when W and A are each **<85% dense**; on AlexNet/GoogLeNet/VGGNet, 2.7× performance and 2.3× energy vs their dense CNN. **Not** this-net numbers.

### 2.3 What MUST be trained (SCNN, as stated)

| Item | Trained? | If skipped |
|---|---|---|
| W sparsity (Han two-phase prune + **retrain**, optionally iterate) | **Yes** | Cartesian product still correct on whatever zeros exist, but the 20–80% W elimination and the <85%-dense crossover vs dense are not free. First layers with “100% input activation density” already hurt them. |
| Activation sparsity | **No.** ReLU at inference. No sparsity loss. | If the nonlinearity is not ReLU, the A-side of the Cartesian product is a different random variable. They measured it by instrumenting Caffe **after** prune-and-train. |
| Cartesian product / scatter / halo / compressed format | Hardware dataflow. Not trained. | |

They do **not** train a structured pattern that the PE would like (no BSA-style bundle loss). Load imbalance and 1×1 fragmentation (GoogLeNet later inceptions, `Kc=8` with 1×1) are accepted as hardware tax.

### 2.4 Map onto r1 3×3 + T10 PSN + dual consumer

| SCNN piece | r1 two 3×3 | T10 PSN | Dual consumer |
|---|---|---|---|
| PT-IS-CP Cartesian F×I on non-zero W and non-zero source | **Direct dataflow prior for each 3×3.** R=S=3 is their common case. Must keep: compressed W blocks `Kc×3×3`, compressed source tiles, coordinate functions, scatter crossbar, acc-bank overprovision, **output or input halos**, Kc blocking, double-buffer drain. θg, if non-zero, is just a multi-bit activation value in the 16-bit slot. | PSN is **after** the spatial Cartesian product. SCNN has no time loop. T10 is either (a) 10 independent spatial passes with W reused (Bishop-like), or (b) a dense time GEMM on the 10 collected spatial sums. Cartesian product does **not** skip PSN rows. Noncausal mix means you cannot ReLU/compress **per t** before all 10 spatial sums exist. | The paper’s compress point is **after ReLU**. Residual add / BN / PED are **pre-ReLU dense consumers**, i.e. exactly the regime they warn about: “accumulation buffer must be kept in a dense format … until they pass through a ReLU”. Gate-path compress is legal only **after** the binary/gate decision. PED cannot wait for that compress if it reads the continuous pre-gate tensor. |
| Input-stationary reuse of an activation against Kc×R×S weights | A source θg at one (p,c) fans out to a 3×3 of outputs × Kc — same geometry. | Stationary source across T10 is extra reuse, not in SCNN. | Same source must be delivered to **two** consumer dataflows (gate conv vs PED/projection). SCNN has one consumer: next layer after ReLU. |
| Dense acc until nonlinearity | Maps to: r1 conv accumulators stay dense. | T10 PSN acc stays dense (full 10). | Dual consumer ⇒ **two** drain conditions. Cannot use “PPU ReLU then compress” as the single retirement. |

**Complete prior A:** compressed non-zero W and non-zero source, Cartesian F×I, coordinate scatter, halo, dense acc until a true zeroing nonlinearity.

**Hole B:** this net’s zeroing nonlinearity is ATLIF **gate**, but the amplitude is θg and a second continuous consumer never sees a ReLU. First-layer-style “100% dense input” is also the residual identity / PED story.

### 2.5 What fails if θg is continuous amplitude, not binary

SCNN never claimed binary activations. ReLU outputs are already continuous **when non-zero**. So Cartesian `in[i]*wt[f]` **does not fail** merely because θg ∉ `{0,1}`.

What **does** fail if one pretends ATLIF is ReLU, or pretends θg is a spike:

1. **Activation sparsity source.** Their 50–70% zeros are “ReLU clamps all **negative** activations to zero”. ATLIF zeros are **did not fire**, not `x<0`. Density is a firing rate, not a ReLU histogram. Quoting 50–70% is a false citation for this net.
2. **“Until they pass through a ReLU” compress point.** There is no ReLU after r1. Compressing as if post-ReLU **drops PED/BN/residual**.
3. **PPU order: halo → ReLU → compress.** Halo exchange of **dense partial sums** is still required for 3×3. The ReLU/compress tail is not.
4. **Multiplier-array utilization vs 1×1.** Irrelevant to 3×3 r1, but relevant if someone maps PSN or 1×1 PED into the same F×I array with tiny F.
5. If someone **further** treats activations as bits (AND instead of multiply), that is **not** SCNN. SCNN’s inner op is a 16-bit × 16-bit multiply into a 24-bit acc.

Training implication: W prune+retrain **must** still be done to get W-side Cartesian sparsity. One cannot “train ReLU sparsity” into ATLIF. A sparsity loss on **gates** does not automatically zero the **PED** operand at the same coordinate.

---

## 3. Bishop — TTB bundle / BSA / ECP / AAC

### 3.1 What the paper actually does

First (their claim) HW/SW co-design for **spiking transformers**. Work unit = **Token-Time Bundle (TTB):** `BSn` tokens × `BSt` time points × one feature. Tag a bundle **active** iff it contains at least one spike (L0). BSA trains a **bundle-level** L0 loss on MLP, projection, Q, and K. ECP prunes whole Q/K **bundle rows** using a bound that holds **because Q and K are binary**. Heterogeneous cores: stratifier sends high-density TTB features to a dense SAC systolic array, low-density to a SIGMA-like sparse core; attention core is **multiplier-less AAC** (AND+accumulate) then SAC for S×V. LIF: `S[tk]∈{0,1}`. They **move LIF before the last SSA linear** so `O_attn` is W times **binary** activations.

Tokenizer conv exists (`O(T H W C² K²)`) but they explicitly punt it to prior spiking-CNN accelerators. Dominant targets are MLP / QKVO projection / SSA.

### 3.2 Section quotes

**Abstract (bundle, BSA, ECP, AAC).**

> Specifically, we introduce the concept of Token-Time Bundle (TTB), a container that bundles spiking data of a set of tokens over multiple time points. … Bishop utilizes a stratifier, a dense core array, and a sparse core array to process MLP blocks and projection layers. The stratifier routes high-density spiking activation workload to the dense core and low-density counterpart to the sparse core … we introduce a novel Bundle Sparsity-Aware (BSA) training pipeline that enhances not only the overall but also structured TTB-level firing sparsity. Moreover, the processing efficiency of self-attention layers is boosted by the proposed Error-Constrained TTB Pruning (ECP), which trims activities in spiking queries, keys, and values both before and after the computation of spiking attention maps with a well-defined error bound. Finally, we design a reconfigurable TTB spiking attention core to efficiently compute spiking attention maps by executing highly simplified “AND” and “Accumulate” operations.

**§2.1 LIF is binary; SSA is binary Q/K/V.**

> A spiking output is generated if the membrane potential `Vm[tk]` exceeds `Vth`, setting `S[tk]` to 1 and resetting `Vm[tk]` to 0.

Equations (3)–(8): `Q = LIF(X WQ)` etc.; `O = (Q Kᵀ · s) · V`; they **reposition** the last LIF:

> In contrast to the method described in [64] which extensively employs multipliers due to spike residuals, we reposition the final LIF neuron layer to precede the last linear layer of each SSA block [53] as in (7). This adjustment allows for efficient multiplication-free computation of attention output `Oattn` based on the product of the weights `Wo` with the binary LIF activations `Otemp` in (8).

**§2.2 they do not claim 3×3 is their problem.**

> The tokenizer’s computational complexity is `O(T H W C² K²)`, where `K` is the size of the employed spiking convolutional (CONV) filters. It is typically not the dominant complexity, and there has been much work targeting hardware acceleration of spiking CNNs [27, 37].

**§3.2 TTB definition + active/inactive + weight reuse.**

> A TTB is a container that bundles spiking data for a set of spatial tokens over a number of time points … we split the spiking activities of N tokens, D features, across T time points into multiple token-time bundles. Each TTB packs `BSn` tokens across `BSt` timepoints for a given output feature. Accordingly, we have `⌈T/BSt⌉ × ⌈N/BSn⌉` token-time bundles.

> A TTB is called active if there exists at least one active spike in the bundle. Otherwise, we call it inactive. … skipping inactive bundles can be efficiently accommodated in the dataflow and avoids large overheads resulting from skipping computations at a much smaller granularity, e.g., at the spike or token level.

> Because the same multi-bit weights are used for different time points and different tokens for each feature, the 1×D weight data in one row … is shared within each TTB for processing `BSn` tokens across `BSt` time points. In addition, the same weight data is reused across `⌈T/BSt⌉ × ⌈N/BSn⌉` TTBs.

**§4.1 BSA — this is trained, and it is structured L0 not spike-level.**

> Without applying BSA, spiking transformers exhibit moderate TTB-level sparsity. For instance, in Model 1 of Table 2, 29% of the bundles are active across all layers, offering some restricted opportunities for computation skipping.
> Unlike suppressing activities at the individual spike level as in [13], the proposed Bundle-Sparsity Aware Training (BSA) algorithm sparsifies spiking activities at the TTB level, i.e., by reducing the number of active TTBs …

Eq. (9): bundle tag `Z = ||X_{tokens in bundle, times in bundle, feature d}||_0`  
Eq. (10): `Lbsp = Σ Z` over layers, bundle-token, bundle-time, features.  
> When forming the above `Lbsp` loss, we consider activations from all MLP and linear projection layers as well as the bundles associated with the queries (Q) and keys (K) in the attention layers. BSA optimizes the model weight parameters `θ` by minimizing a total loss: `Ltot = LCE + λ Lbsp`

**§5.1 / intro AAC — binary is the circuit.**

> Taking advantage of the binary nature of spiking TT-bundled queries and keys, we design a reconfigurable multiplier-less And-ACcumulate (AAC) array. This array efficiently processes spiking attention layers by executing highly simplified “AND” and “Accumulate” operations …

**§5.1 ECP — the bound that is 100% only for binary K/Q.**

> In ANNs, bounding the scores (S) given the queries (Q) and keys (K) is difficult because Q and K are continuous-valued floating point numbers.
> Differently, we explore the binary nature of the spiking Q and K for a more straightforward bounding of S without computing them. The total number of active bundles `nab` in a particular bundle row across all features of the Q tensor can be easily obtained from the active bundle tags. If `nab` is less than a threshold `θp,Q`, due to the binary nature of K, it is certain that any activation in a bundle that is in the corresponding row of the scores (S) tensor would be less than `θp,Q` per `S = QKᵀ`. We prune out this row from the Q tensor entirely while limiting the error to be no greater than `θp,Q`.

Figure 7 caption: “Densely Compute correlation between strong Q bundles and K bundles with **100% confidence** that all pruned attention values smaller than `min(uQ, uK)`.”

> ECP is also integrated into the training pipeline, leading to ECP-aware training to maintain high accuracy.

**§5.4 SAC (MLP/projection) is MUX×W, still binary.**

> Leveraging the binary nature of spiking inputs, each PE executes “Select ACcumulation” (SAC) operations to effectively multiply synaptic weights with spiking inputs. A “SAC” operation is efficiently implemented by one MUX and one accumulator.

**§5.5 AAC Mode 1 / SAC Mode 2.**

> Mode 1. We configure the core into multiple And-ACcumulate (AAC) units to efficiently compute the accumulated attention … This setup executes “AND” operations between binary queries (Q) and keys (K), followed by the accumulation of partial sums in the S register …
> Mode 2. … In each “SAC” operation, the binary V input selects the right S data to be accumulated into the partial sum …

**§6 training hyperparameters (must be trained, values are theirs).** `λ ∈ {1, 0.5, 0.3, 1.0}` by dataset; ECP thresholds “10 for models trained on DVS-Gesture-128 and 6 for other models without compromising accuracy.” T in `{4,8,10,20}` — they argue PTB-style **time-only** packing is weak at short T, which is why TTB adds the token axis.

**§8.** BSA + ECP + dense/sparse/attention cores. Average vs prior SNN accelerators in the abstract: 5.91× speedup, 6.11× energy. **Not** this-net numbers.

### 3.3 What MUST be trained (Bishop, as stated)

| Item | Trained? | If skipped |
|---|---|---|
| TTB packing `(BSt, BSn)` | Architectural hyperparameter (they sweep 4–8 as a good volume). Not a loss. | Packing still works; reuse still works; **structured skip** is only as good as natural bundle L0 (Model 1: 29% bundles active). |
| BSA `Ltot = LCE + λ Lbsp` on MLP, projection, Q, K | **Yes.** This is the software half of structured skip and of the stratifier’s dense/sparse split. | “Restricted opportunities for computation skipping.” Sparse-core utilization “is enhanced considerably by the proposed BSA”. |
| ECP-aware training | **Yes.** “ECP is also integrated into the training pipeline.” Thresholds chosen “without compromising accuracy”; too-large `θp` “could degrade model performance”. | You can still **apply** the inference bound if Q/K stay binary, but they do not claim accuracy without the training hook. |
| Binary LIF / moving LIF before last SSA linear | Architectural + training choice vs Spikformer-style spike residuals. | If residuals stay continuous, they say multipliers come back ([64] contrast). |
| Stratifier threshold `θs` | HW hyperparameter, not trained. Balance dense vs sparse core. | |

They cite weight pruning, temporal pruning, quantization as existing for **spiking CNNs**, and say a systematic co-design was missing for **spiking transformers**. BSA is **not** Han W-prune; it is a **firing-structure** loss.

### 3.4 Map onto r1 3×3 + T10 PSN + dual consumer

Bishop’s native object is **SSA/MLP tokens**, not a 3×3 sliding window. Mapping is a **retarget**, and the paper itself sends conv to SNN-CNN accelerators (FireFly-S/SCNN class).

| Bishop piece | r1 two 3×3 | T10 PSN | Dual consumer |
|---|---|---|---|
| TTB = tokens × time | **Not native.** A forced map: treat spatial patches (or 3×3 neighborhoods) as “tokens” and T10 as `BSt`. Intra-bundle W reuse **does** match “same 3×3 W at every t”. Inter-bundle W reuse matches broadcasting W across patches. Sliding-window **halos** are SCNN/FireFly-S, not TTB. Inactive-bundle skip at 3×3-window granularity is a **new packing**, not a copy. | Natural axis for `BSt`. **Mismatch:** Bishop time is **independent LIF ticks** with the **same** W. This net’s T10 is a **noncausal dense mix**. Packing 10 times into one bundle still reuses spatial W; it does **not** skip PSN coefficients. | Bundle tag Z is L0 of **spikes**. PED/residual can be non-zero on a bundle whose **gates** are empty, or vice versa. One tag cannot retire both consumers. |
| BSA `Lbsp` | Would have to be redefined on **r1 work units** (e.g. 3×3×T10 bundles, or channels). Training on SSA Q/K L0 does not sparsify r1. | A T10-bundle L0 loss can increase **all-zero time groups**. Noncausal PSN still needs the surviving times as a dense 10-vector unless the mix itself is retrained. | Loss on gates ≠ loss on PED. Must include **both** consumers in any copied `Lbsp`, or the continuous path remains dense. |
| ECP | **Does not map to r1 conv.** ECP is a bound on `S=QKᵀ` rows. r1 has no attention map. Applying “prune this 3×3 if nab < θp” is **not** ECP; it has no 100% score bound. | — | — |
| AAC (AND-accumulate) | **Does not map** unless r1 operands are binary and the op is a popcount-like correlation. 3×3 conv is `Σ W·θg`, not `AND`. | — | — |
| SAC (MUX + acc) | Maps to **gate path only** if gate∈{0,1}: select W or 0. Fails for θg amplitude (need W·θg). Fails for PED. | — | Same. |
| Stratifier dense/sparse cores | Could split r1 **channels** by gate density. Load-balance vs `θs` is ordinary. Does not create a PED core. | — | Would need a **third** consumer class (continuous dense), which they do not have. |
| Move LIF before last linear to kill spike-residuals | Opposite of this net’s identity: **keep** continuous θg and residual/PED. Copying this move **is** the binary-ATLIF identity change `PROBLEM.md` forbids. | — | Explicitly destroys the continuous consumer. |

**Complete prior A:** structured spatiotemporal packing for W reuse; a bundle-L0 training loss if you want structured skip; heterogeneous dense/sparse dispatch on **binary** spike density; AAC/ECP **only** as related work for **binary attention**, not for r1.

**Hole B:** r1 is 3×3+PSN+dual continuous consumer, not SSA; θg is not `S[tk]=1`.

### 3.5 What fails if θg is continuous amplitude, not binary

Bishop is the paper that **fails hardest**. They repeatedly use “binary nature” as the enabling lemma.

1. **AAC.** AND(Q,K) equals Q·K only for `{0,1}`. Continuous θg requires a multiplier (or at least an AND-plus-scale). Their 2.66× / 4.27× vs PTB on the attention core is this circuit.
2. **ECP 100% bound.** `S = QKᵀ ≤ nab` (per row) uses `K∈{0,1}` and `Q∈{0,1}`. If Q or K carry amplitudes, `S` can be `Σ θg_q θg_k`, **unbounded by nab**. Their own contrast: ANN Q/K are “continuous-valued floating point numbers” and bounding S “is difficult”. **Continuous θg is that ANN case.** Pruning a row because `nab < θp` is then an **unbounded** approximation, not ECP.
3. **SAC as MUX.** `spike? W : 0` is wrong for `W·θg`. Need a multiplier or a shift/scale of W by θg.
4. **`Oattn = Wo · binary Otemp`.** If Otemp is θg, the last linear is a real GEMM; their “multiplication-free” SSA tail is gone. This is the same reason they moved LIF in (7)–(8).
5. **Bundle L0 tag Z.** Still defined for any tensor (`||·||_0` counts non-zeros). Skipping an “inactive” bundle is legal **iff every consumer** sees zeros there. Continuous PED can be live when gates are dead (native `conv_res` even more so: it does not go through `proj.sn`).
6. **Stratifier on spike density.** Density of continuous amplitudes is a different statistic; a feature can be “sparse as spikes” and dense as PED.
7. **ECP-aware training** does not repair (2). Training can hide accuracy loss; it cannot restore the **proof**.

Session identity already forbids “Bishop-ECP-bound” as a binary island. The local txt is why: the bound **is** the binary lemma.

---

## 4. Synthesis: must-train vs continuous-θg kill

### 4.1 One table

| Mechanism | Must train to get the paper’s sparsity | HW primitive | Maps to r1 3×3? | Maps to T10 PSN? | Dual-consumer legal? | If θg continuous, not binary |
|---|---|---|---|---|---|---|
| FireFly-S rewiring + joint LSQ | **Yes** (W/bias/Vth, during training) | Induced W zeros + 4-bit integer LIF | Yes, on both convs; Vth-absorb ≠ this BN | No. Causal T=4 V-reg ≠ noncausal T10 | No. Algo 1 / detector feed IF/LIF only | AND-detector, `I=Σ W s`, scale-into-Vth, bias-bubble, silent-channel all assume binary `s` |
| FireFly-S Bitmap dual-side walk | Spike side **not** trained; W side needs prune | AND mask, one-hot peel, fetch W, acc V | Yes as **gate×W** skip on 3×3 im2col | No consecutive-t V | Only the gate MAC | AND of 1-bit spikes fails; need zero-detect + multiply by θg |
| SCNN Han prune+retrain | **Yes** for W; A is ReLU, not trained | Compressed W/A, Cartesian F×I, scatter | **Yes** (R×S=3×3 is native) | No time; PSN is a later dense mix | Acc **dense until ReLU**; residual/PED **are** that dense region | Multiply `θg×W` is fine; ReLU density and post-ReLU compress are **not** |
| SCNN Cartesian product | Not trained | F non-zero W ⊗ I non-zero A | Yes | Does not skip PSN | Scatter into dense acc is the residual-like consumer | Still valid if “non-zero” means `θg≠0`; invalid if rewritten as AND |
| Bishop TTB packing | Not trained | Pack `BSn×BSt`, skip inactive, reuse W | Forced retarget only; halo missing | `BSt` matches packing, not PSN math | One L0 tag ≠ two consumers | Packing still works; skip-if-no-spike may drop PED |
| Bishop BSA `Lbsp` | **Yes** | Structured TTB L0 | Only if loss is **redefined** on r1 units | Only if T-groups are the work unit | Must put **both** consumers in the sum | L0 still defined; does not make AAC/ECP true |
| Bishop ECP | **Yes** (ECP-aware) + binary Q/K | Row prune by `nab < θp` | **No** (SSA only) | No | No | **Bound collapses** (paper’s own ANN contrast) |
| Bishop AAC | Binary Q/K (architecture+train) | AND + acc | **No** | No | No | AND ≠ θg multiply |
| Bishop SAC | Binary spike | MUX + acc | Gate path only | No | Not PED | Need `W·θg` |

### 4.2 What must be trained **on this net** if one actually copies the prior

Copying HW skip **without** the paper’s training is not “the method”:

1. **Any W-side dual-side / Cartesian story (FireFly-S, SCNN):** must run a **prune + recover** training on the two r1 3×3 (FireFly-S: joint with quant and with their rewiring parameterization; SCNN: Han two-phase). Inference-only magnitude cut is exactly SCNN’s “naïve pruning” that they then **retrain**.
2. **Any structured bundle skip (Bishop BSA):** must train `Lbsp` on the **actual r1/T10 work unit**, not on SSA Q/K. λ and bundle shape are new hyperparameters; their CIFAR/DVS numbers do not transfer.
3. **ECP/AAC:** do **not** train on this net as a copy. The lemma is false for continuous θg. Training cannot restore a binary product bound.
4. **Dual consumer:** none of the three trains it. A copied spike/bundle loss can **hurt** PED. If a sparsity pattern is to be legal, the training objective must include **gate consumer and continuous residual/PED** (and native projection/BN/add), or the continuous path stays dense and the skip is only a gate-side A.
5. **T10 PSN:** none of the three trains a noncausal time mix. FireFly-S Algo 1 / consecutive V and Bishop per-tick LIF are the wrong neuron. PSN coefficients stay a dense (or separately compiled) consumer unless **this** net trains otherwise.
6. **Do not** copy Bishop’s “reposition LIF before last linear” / FireFly-S “output is a spike”. That is a binary-ATLIF identity change.

### 4.3 What fails, in one sentence each, if θg is continuous amplitude not binary

- **FireFly-S detector:** the activation operand is no longer a 1-bit Bitmap; CLK0 AND + “add W as event” is the wrong ALU for θg, and Vth-absorbed per-channel scale does not hide θg from PED.
- **FireFly-S Algo 1:** a channel that never **spikes** can still be required as a continuous residual/PED source; the test is also the wrong dynamics for noncausal T10.
- **SCNN Cartesian:** the multiply is still right; the **ReLU compress/retire** and the **50–70% activation** citation are wrong. Dense acc until a real zeroing is the residual/PED case they already pay for.
- **Bishop TTB skip:** L0-of-spikes is not L0-of-PED.
- **Bishop BSA:** trainable, but does not resurrect binary circuits.
- **Bishop ECP:** `nab < θp ⇒ |S| < θp` is false once Q/K carry θg; this is the failure the paper attributes to ANNs.
- **Bishop AAC/SAC:** AND/MUX are `{0,1}` ALUs; continuous θg returns multipliers (and returns spike-residual multipliers they deliberately removed in (7)–(8)).

### 4.4 Relative prior vs this session’s X

All three are **complete A** for a TCAS-II related-work / denominator:

- dual-side **zero skip** on 3×3 (FireFly-S Bitmap, SCNN Cartesian);
- W sparsity **must be trained** (both);
- structured time-token packing and bundle-L0 training (Bishop), **if** retargeted and if both consumers are in the loss;
- dense scatter-acc until nonlinearity (SCNN), which is the honest residual/PED cost.

None of them is an X on **continuous-θg dual-consumer noncausal T10 r1**. Renaming TTB→“T10 bundle”, ECP→“error-constrained gate”, or AAC→“AND-popcount PSN” is a reskin the venue will punish. The binary lemmas (AND, ECP bound, SAC MUX, scale-into-Vth, post-ReLU compress) are **kill-gates** under this-net identity, not knobs.

---

## 5. Quotes not to launder as this-net results

Do not write as if measured here:

- FireFly-S 10,047 / 3,683 / 2,327 FPS/W; 85–95% sparsity; 4-bit LUT-only; 333 MHz; KV260.
- SCNN 2.7× / 2.3× vs dense; 7.9 mm²; 1024 multipliers; 85% density crossover.
- Bishop 5.91× / 6.11× vs prior SNN accelerators; 2.96 mm², 627 mW, 500 MHz, 28 nm; ECP 65.79× attention-layer speedup on ImageNet-100.

Do not cite volume/issue for FireFly-S from this txt. Do not cite Han/LSQ/gradient-rewiring/PTB/Spikformer as if they were read in this L3 pass; they appear only as **those papers’** references.

End of L3. Sources: the three local txts + this session `PROBLEM.md`/`SCOPE.md` only.
