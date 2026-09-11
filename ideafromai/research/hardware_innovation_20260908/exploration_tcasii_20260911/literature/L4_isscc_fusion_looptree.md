# L4 — ISSCC ConvFormer / CFMP, LoopTree retain–recompute, RISCSparse coordinate traffic → r1 `conv–PSN–conv2–BN–proj`

Date: 2026-09-11. Literature check for `exploration_tcasii_20260911`. Frozen observations only from [`PROBLEM.md`](../PROBLEM.md). This note is a prior-to-copy + hole map, not an idea catalog and not a claim of novelty.

**Read in full (local texts):**

| File | Role |
|---|---|
| `literature/ISSCC2025_23_2_ConvFormer_author.txt` | ISSCC 2025 digest, Session 23.2 (author two-column layout) |
| `survey_ab_fusion_20260910/p0_txts/2512.17555.txt` | same digest posted as arXiv:2512.17555 |
| `literature/LoopTree_TCASAI2024_author.txt` | IEEE TCASAI 2024 fused-layer DSE model |
| `literature/RISCSparse_ICCAD2024_author.txt` | ICCAD 2024 sparse-conv mapping / GMS / operator fusion |

**Focus axes:** layer fusion; retain vs recompute; cascaded pruning; hybrid attention; sparse coordinate traffic. **Local object:** patch-embed residual r1 `conv–PSN–conv2–BN–proj` with **full-domain projection BN statistics**. Local window is **not** a closed hardware schedule.

---

## 0. Identity freeze (do not rewrite the paper)

From `PROBLEM.md`:

- Task: event-camera 2D optical flow, DSEC valid825 AEE. Student family Motion C12 / H67 / ep34.
- Neuron: ATLIF with **continuous threshold amplitude θg**, not binary spikes. Noncausal **T10** at this PSN.
- Expensive region (historical proxy, ≠ new cycle share): patch-embed residual r1 (two convs) plus T10 PSN.
- Dual consumers after source: spike/gate path **and** continuous residual/PED path. Native projection convolution + BN + residual add also exist.
- Native projection BN on captured students uses **actual batch statistics over the full `10×96×120×160` domain**, not frozen running stats. **Local-window replay given free mean/var undercharges wait/storage.**
- Venue: TCAS-II Express Briefs, one mechanism, circuits+systems. Do not change identity to binary ATLIF. Do not sell analog CIM. Do not quote OpenROAD/Yosys as foundry PPA. Do not multiply component speedups into FPS.

r1 chain used below (software graph, not a new claim):

```
sn1 → conv1 → norm1 → sn2/PSN(T10, θg) → conv2 → norm2 + identity  =  r1out
r1out ─┬─ proj.conv_res (1×1, stride 2)                    ─┐
       └─ proj.sn → proj.conv → proj.norm_layer → ADD  ─────┴─ PED
```

`proj.norm_layer` is the **unfrozen** BN. Residual `norm1`/`norm2` on captured parents may already be fixed; the freeze that kills closed-window HW is the **projection** BN.

---

## 1. Title confirmation: arXiv:2512.17555 **is** ISSCC 2025 23.2 / CFMP

The two local texts are the same ISSCC digest (same title, DOI, session line, author list, figure captions, and body). `2512.17555.txt` is not a different CFMP paper.

**Title (page header + session line, both files):**

> 23.2 A 28nm 0.22μJ/Token Memory-Compute-Intensity-Aware CNN-Transformer Accelerator with Hybrid-Attention-Based Layer-Fusion and Cascaded Pruning for Semantic-Segmentation

**Authors:** Pingcheng Dong*, Yonghao Tan* (ECAs), Xuejiao Liu, Peng Luo, Yu Liu, Luhong Liang, Yitong Zhou, Di Pang, Man-To Yung, Dong Zhang, Xijie Huang, Shih-Yang Liu, Yongkun Wu, Fengshi Tian, Chi-Ying Tsui, Fengbin Tu, Kwang-Ting Cheng.

**Venue / DOI:** 2025 IEEE International Solid-State Circuits Conference (ISSCC), Session 23 / AI-Accelerators / 23.2. DOI `10.1109/ISSCC49661.2025.10904499`. IEEE Xplore authorized-use footer on the local author text.

**Corresponding authors (acknowledgement):** Kwang-Ting Cheng, Fengbin Tu (HKUST / ACCESS).

CFMP in this paper is **Cascaded Fmap Pruner** (also “Cascaded Feature-Map Pruner” in Fig. 23.2.5 caption), not a standalone algorithm paper. Hybrid-attention layer-fusion is **HAPU + LFS (DR-LFS)**; cascaded pruning is **CFMP = FMS + DRU**.

Workload in the paper: SegFormer-B0, PVTv1-Ti, PVTv2-B0 on **Cityscapes** semantic segmentation. Token length cited as **>16K**, attention example **64K TL × 32 channels**. This is ANN ConvFormer, not event-SNN optical flow.

---

## 2. ISSCC 23.2 — quotes by unit, then r1 map

ISSCC digest body is two-column. Quotes below follow the **logical** reading order (left column then right), not the OCR column-interleave.

### 2.1 Problem statement: three ConvFormer bottlenecks

> Recently, hybrid models integrating a CNN and a Transformer (ConvFormer) … have achieved significant advancements in semantic segmentation tasks [1-4] … but the large token length (TL) demand of semantic segmentation (>16K TL) incurs significant computation and memory overheads. Prior NN accelerators [5-12] demonstrate that sparse computing and pruning can effectively reduce computation and weight storage, but most of them focus on pure CNN or Transformer models in simpler vision or language-processing tasks (1-4K TL). Moreover, the performance bottlenecks of ConvFormers stem from their memory-intensive Backbone and compute-intensive Segmentation Head (Seg. Head), raising three challenges for hardware acceleration:
>
> 1) Conventional sparse attention [5-9] fails to buffer the attention feature map (Fmap) on-chip when the TL exceeds 16K, even at 90% sparsity, resulting in massive external memory access (EMA).
>
> 2) While Layer-Fusion (LF) [13-18] is a common technique to reduce Fmap EMA, it is infeasible to buffer key (K), value (V), and convolution weights on-chip simultaneously. Moreover, different fused attention-convolution layers may cover various vanilla attention (VA) tiles, leading to enormous redundant KV and weight EMA.
>
> 3) In the Seg. Head, the Fmap sparsity is extremely low, thereby limiting the effectiveness of conventional zero-skipping strategies [11,12] designed to reduce computational work.

**r1 map.** Challenge (1) is **attention-map storage O(N²)**. r1’s expensive object is **T10 PSN + two 3×3 convs + dual consumers**, not a 64K-token attention Fmap. Challenge (2) is the useful LF prior: **you cannot co-resident every live tensor of a fused pair**. On r1 the analogous live set is `{source θg/gates, W_conv2, residual identity, conv2 psums, BN2 state, proj.conv_res state, proj.sn state, proj.conv weights, **full-domain BN moments**}`. Challenge (3) is **dense-head zero-skipping fails**. r1’s continuous PED / `conv_res` path is dense by construction; gate sparsity does not make the projection BN sparse.

### 2.2 Three features they actually built (complete A, not X)

> To tackle these challenges, we propose a ConvFormer accelerator with three key features:
>
> 1) We propose a Hybrid Attention Processing Unit (HAPU) that utilizes memory-efficient linear attention (LA) [19-23] for most query (Q) tiles in Backbone, reducing Fmap storage from O(N²) to O(C²) by reordering the computation from QKT-first to KTV-first, while a few Q tiles perform VA to ensure accuracy by their global receptive field. Here, N is the TL and C is the channel dimension with C≪N. This reordering allows the HAPU to buffer tiny KTV Fmaps entirely on-chip, saving 60.2-78.6% EMA.
>
> 2) We develop an LF Scheduler (LFS) with KV-weight reuse to mitigate redundant EMA overhead of LF in Backbone. The LFS first reuses on-chip buffered KV to compute all VA tiles, then replaces the KV with off-chip convolution weights. Afterward, convolution layers are sequentially fused with each VA output tile and LA input tile, reusing both KTV and convolution weights. This approach significantly alleviates redundant EMA, reducing overall EMA by 86.8-96.2%.
>
> 3) A Cascaded Fmap Pruner (CFMP) is designed to decompose each convolution of Seg. Head into two sub-convolutions: the first injects redundancy by expanding the intermediate Fmap, which is further pruned using a pre-trained mask, while the second restores density using the same mask, reducing 91.10% of computation in the Seg. Head.

Closing hedge (do not import as “applies to SNN flow”):

> Notably, while this work focuses on accelerating ConvFormer for semantic segmentation tasks, the proposed solutions are not limited to ConvFormer and can be also applied to pure CNN or Transformer models.

That last sentence is an **author claim of architectural generality**, not a measurement on ATLIF / T10 / AEE.

### 2.3 SoC / dataflow (Fig. 23.2.2)

> It consists of a SIMD core, a top controller, a PLL, 2 HAPUs, a LFS, a CFMP, a 64KB ISA buffer, a global buffer (GB) including a 2MB left matrix buffer (LMB), a 1MB right matrix buffer (RMB) and an LMB-RMB Router (LR2).
>
> In the Backbone stage, the HAPUs prioritize KT, V, and KTV generation, where KT is further routed from LMB to RMB via LR2. Then, LFS clusters the VA and LA tiles in the attention cluster unit (ACU), scheduling HAPUs to reuse KV for parallel VA tile processing. Once completed, the LFS replaces the KV with subsequent convolution weights, directing HAPUs to perform fused convolution on each VA output tile. Then, the remaining convolution layers could reuse these weights to fuse with their associated LA input tiles.
>
> In the Seg. Head stage, the top controller configures CFMP using pre-trained sparsity masks. The feature map sparsiﬁer (FMS) decodes the RMB IDs of unpruned column tiles, which are sent to HAPUs for sparse convolution. Then, the density recovery unit (DRU) converts column tile IDs in FMS to row tile IDs, guiding HAPUs to recover the density of sparse Fmap via row-wise accumulation.

**r1 map.** Copyable A is **buffer-slot sequencing**: occupy on-chip SRAM with tensor family A, finish all consumers of A, then **replace the slot** with tensor family B. Not copyable as PPA: 2 MB + 1 MB GB, 28 nm, two HAPUs. On r1, a legal A is “reuse on-chip source/KV-analog (θg words + identity) for **both** gate and continuous consumers, then replace the slot with projection weights / BN accumulators.” Illegal A: treating the 3 MB GB as local capacity, or multiplying HAPU×LFS×CFMP speedups into FPS.

### 2.4 HAPU hybrid attention, RMPI, ATM (Fig. 23.2.3)

> Figure 23.2.3 illustrates HAPU that leverages a hybrid attention mechanism for EMA reduction. This hybrid attention allows most Q tiles to employ LA, replacing the exponential function of VA with separable kernel functions, such as identity, ReLU, etc. The hybridization pattern is learned during training and exhibits a layer-wise distribution. However, since KT serves as a right matrix for QKT in VA and left matrix for KTV in LA, it incurs storage conflicts that require KT transfers through DDR. To address this issue, the right matrix prioritized initializer (RMPI) ﬁrst computes KT, V, and KTV, then decodes the LR2 offset to route KT from LMB to RMB on-chip via LR2.
>
> In addition, certain layers may retain a high proportion of VA, leading to large VA tiles that could still cause EMA issues. To mitigate this, the Attention Tiling Manager (ATM) identiﬁes the VA tile size and speculates potential LMB overﬂows. If overﬂows are detected, the ATM subdivides each Q tile into smaller segments, which are processed sequentially and combined together. Compared to a layer-wise architecture, the HAPU achieves a reduction in EMA and energy consumption by 22.05× and 8.93×, respectively, for an attention layer with a 64K TL and 32 channel dimensions.

**r1 map — hybrid attention is A for attention chips, not X for r1 PSN.**

| HAPU mechanism | What it actually is | r1 legality |
|---|---|---|
| Most tiles LA, few tiles VA | **Trained model change** (hybridization pattern learned) | If used on r1, it is a **new student**, must re-run valid825. Cannot be sold as a lossless schedule of the frozen ATLIF T10 PSN. |
| KTV-first, buffer O(C²) instead of O(N²) | Algebraic reordering of **linear** attention | T10 PSN is a dense 10×10 mix over time per spatial site, **not** QKT. CSE of lifting vs ordinary is already in `PROBLEM.md` (260 vs 159 add/sub). Reordering PSN is a different algebra. |
| RMPI: compute KT/V/KTV first, route KT LMB→RMB | Avoids a tensor playing **both left and right** matrix roles | Dual-consumer analog: r1out is left operand of `conv_res` **and** input of `proj.sn`. Slot conflict is real; DDR-via-KT is not the r1 bottleneck. |
| ATM overflow → subdivide Q, sequential combine | **Tile-size speculation against a named SRAM** | Legal control for any fused window. On r1 the overflow that ATM does **not** model is **full-domain BN moments**, which are not an LMB tile of Q. |

Do **not** retitle r1 PSN as “hybrid attention.” LA vs VA is a different function. Local few-VA tiles “to ensure accuracy by their global receptive field” is the same class of **lossy RF recovery** as the non-overlapped LF below.

### 2.5 LFS / DR-LFS: KV-then-weight fusion, non-overlapped LF (Fig. 23.2.4)

> Figure 23.2.4 depicts the LFS that consists of an ACU, a KV-reused vanilla attention-convolution fuser (VACF), and a weight-reused linear attention-convolution fuser (LACF). Owing to the signiﬁcant reduction in VA proportion by HAPU, all VA tiles can be computed in parallel for most layers. The ACU initially re-orders the VA and LA tiles into two fusion groups, integrating them with the same fused convolution (FC). Subsequently, the VACF decodes the LMB ID for each VA tile to generate Q and reuses the KV prepared by RMPI for parallel VA execution. The VACF then fetches off-chip FC weights to replace KV, and sequentially schedules each VA output tile for its corresponding FC. Since the KTV is prepared in advance by RMPI and remains on-chip due to its small size, and the FC weights are already buffered on-chip by VACF, the LACF can reuse them to perform LF from each LA input tile.
>
> However, the convolution cannot fuse with boundary tiles between VA and LA in VACF because the LA output tiles are not yet ready. While the previous slice-based LF method [16] can handle this with overriding, it incurs extra storage and computation overhead for each VA tile. To address this, we propose a non-overlapped LF processing scheme wherein the FC is broken into several non-overlapped FCs. The subsequent attention layer recovers the broken receptive ﬁeld by its inherent long-range dependency. The boundary fusion issue is then resolved by zero-padding the unavailable boundary tiles, resulting in 50% GB usage and a 20% reduction in operations, with <0.5% accuracy drop. Moreover, EMA and energy consumption are reduced by 3.91× and 1.45×, respectively, for a ConvFormer sub-block with a 64K TL and a 50% VA ratio.

**r1 map — this is the dangerous prior.**

- **Legal A (complete, not X):** (i) fuse producer tile into consumer tile **without writing the full intermediate fmap off-chip**; (ii) **time-multiplex a physical buffer** between “left” tensors (KV / source) and “right” tensors (conv weights / proj weights); (iii) if a fusion group is not ready, you must **either retain halo, recompute halo, override/slice, or change the model**.
- **Illegal copy onto residual r1:** non-overlapped LF + **zero-pad missing boundary** + “later attention recovers RF” is an **explicitly lossy model change** (`<0.5%` Cityscapes, not AEE). r1 residual identity and 3×3 conv2 **need the halo**. Projection BN needs the **entire** `10×96×120×160` domain, which is not a halo that attention can hallucinate.
- Hiddenite-style slice override [16] is named as the storage-heavy alternative. LoopTree (below) is the systematic retain/recompute language for that alternative. ISSCC **chose the lossy pad** to cut GB 50%. That choice is **not** available as a silent HW optimization of the frozen student.

### 2.6 CFMP cascaded pruning (Fig. 23.2.5)

> Figure 23.2.5 introduces the workﬂow and architecture of CFMP, which contains an FMS and a DRU. The CFMP decomposes the convolution weights into two cascaded components W0 and W1, injecting redundancy by enlarging the intermediate Fmap Z, which is then pruned using pre-trained tiled masks. Since Z is the only sparse Fmap, with the input and output Fmaps remaining dense, CFMP can be generalized to support sparse VA by substituting the input Fmap, W0, W1, and the output Fmap in convolution with Q, KT, V, and the output Fmap in VA.
>
> The FMS begins by decoding the mask and splitting it into multiple parts for pipeline processing. Tiled column (TC) offsets are generated by ﬂattening each part and extracting the positions of non-zero values. Once all valid indices are decoded, the FMS halts the current mask decoding and fetches the next one, allowing for an early stop. The TC offsets are combined with the base IDs from the LMB/RMB and sent to the HAPU, where the unpruned TCs are fetched, and the sparse Z is stored in a dense format.
>
> Then, the DRU needs to obtain unpruned Tiled Rows (TRs) corresponding to each Z tile and its associated mask to recover the Fmap density. However, the physical storage scheme of the RMB maps each TC across different SRAM banks consecutively, resulting in interleaved storage of various TR slices within the same bank from a row-wise perspective. To address this, the DRU converts TC offsets into TR form and recovers density by accumulating the multiplication results from different Z tiles and TR slices. Compared to zero-skipping approaches [11,12], the CFMP improves sparsity by 6× in the Seg. Head and reduces energy consumption by 2.03× when pruning both VA and convolution.

**r1 map.**

| CFMP piece | Must copy if you borrow CFMP | Cannot call `conv1–PSN–conv2` “CFMP two factors” |
|---|---|---|
| One **linear** conv → `W0, W1`, expand Z, train tiled mask | Joint train both factors + mask under the **same** AEE gates | PSN/θg sits **between** r1’s two convs. `Conv1→threshold→Conv2` is not `X W0` then `Z W1`. A linear factorization does not commute with ATLIF. |
| I/O dense, only Z sparse | Matches r1 **continuous** consumer (PED / `conv_res`) which wants dense r1out | Gate path can be sparse; using CFMP then **DRU-restoring density** before full-domain BN is extra traffic, not a saving, unless the mask also reduces BN’s reduction domain (it does not: BN is over the dense Y). |
| FMS: decode mask → TC offsets → early-stop when valid tiles done | Index/metadata tax must be charged | “Early stop” here is **mask decode complete**, not ATLIF membrane early-exit. |
| DRU: TC→TR, bank-interleave repair, accumulate to dense Y | Coordinate conversion is the real HW | Same tax appears in RISCSparse GMS (below). |
| Same mask on produce and consume | Strong A for any structured prune | Does not create a closed local window for projection BN. |

CFMP numbers (91.10% Seg.Head ops, 6× sparsity vs zero-skip, 2.03× energy) are **Cityscapes ConvFormer head** under trained masks. They are not r1 cycle share.

### 2.7 Silicon numbers — cite as prior, never as local PPA

> The chip works at 200-625MHz with a supply voltage of 0.65-1.0V. The peak energy efﬁciency is 52.90TOPS/W at 0.65V and 200MHz. … The memory-intensity-aware HAPU and LFS and compute-intensive-aware CFMP obtain 4.66-to-7.71× speedup and 4.39-to-7.10× energy savings compared to the baseline, with negligible accuracy loss. … our chip consumes 0.22μJ/token for SegFormer-B0, achieving 3.86-to-10.91× system-level energy reduction.

Fairness hedge in the paper (they peak-assume priors):

> we include a DDR3 interface similar to [10] and assume all prior state-of-the-art accelerators [5-8, 10] work at peak energy efﬁciency with their reported technical conﬁgurations, such as pruning ratio, sparse attention patterns, etc.

**Do not** multiply 22.05× HAPU × 3.91× LFS × 2.03× CFMP into FPS. `PROBLEM.md` forbids it. The 0.22 μJ/token is a **system-level token energy on SegFormer-B0**, not a TCAS-II local metric.

LF citations they name as A: Alwani MICRO’16 fused-layer CNN [13]; Lee VLSI’19 selective-cache LF [14]; DepFiN [15]; Hiddenite ISSCC’22 slice LF [16]; VISTA [17]; NVE [18]. Hybrid LA citations [19-23] (Katharopoulos, Performer, EfficientViT, FLatten, Castling-ViT). Those are the relative priors TCAS-II reviewers will expect if anyone writes “layer fusion” or “hybrid attention” in the brief.

---

## 3. LoopTree (TCASAI 2024) — quotes by section, retain vs recompute

**Title:** LoopTree: Exploring the Fused-layer Dataflow Accelerator Design Space.

**Authors:** Michael Gilbert, Yannan Nellie Wu, Joel S. Emer, Vivienne Sze. IEEE Transactions on Circuits and Systems for Artificial Intelligence, DOI `10.1109/TCASAI.2024.3461716`. arXiv:2409.13625v4.

LoopTree is a **model + taxonomy**, not a chip. It is the correct language for ISSCC’s “retain halo vs pad vs recompute” choice, and for r1’s “what stays on-chip between conv, PSN, conv2, BN, proj.”

### 3.1 Abstract / intro — the contract

> Fused-layer accelerators reuse data across operations in different layers by retaining intermediate data in on-chip buffers, which has been shown to reduce energy consumption and latency. Moreover, the intermediate data is often tiled (i.e., broken into chunks) to reduce the on-chip buffer capacity required to reuse the data. Because on-chip buffer capacity is frequently more limited than computation units, fused-layer dataflow accelerators may also recompute certain parts of the intermediate data instead of retaining them in a buffer. Achieving efficient trade-offs between on-chip buffer capacity, off-chip transfers, and recomputation requires systematic exploration of the fused-layer dataflow design space.

Footnote precision they insist on:

> When the precision is required, we will refer to retaining entire intermediate fmaps on-chip as **untiled fusion**, and fusing with inter-layer tiling as **tiled fusion**.

Layer-by-layer vs fused (Fig. 1):

> Because operations from a layer are scheduled close together, layer-by-layer processing is efficient at exploiting intra-layer reuse opportunities. On the other hand, reusing the intermediate fmap between layers requires retaining the entire intermediate fmap in a buffer because the entire intermediate fmap is produced before it is used in the next layer. … More typically, layer-by-layer accelerators do not exploit inter-layer fmap reuse on-chip. Rather, intermediate fmaps are streamed to and from off-chip buffers between the processing of different layers.

> The amount of intermediate fmap that needs to be retained to exploit inter-layer reuse can be reduced by tiling the layers and scheduling the processing of the tiles such that an intermediate fmap tile … can be computed and immediately used between layers. When the intermediate fmap tile is not needed anymore, it is released from the buffer.

**r1 map.** “Immediately used between layers” is **true** for `conv1 tile → PSN tile → conv2 tile` **if and only if** every consumer of that tile is local. It is **false** for `proj.norm_layer`, whose mean/var are reductions over the **full** `10×96×120×160` domain. That BN is an **untiled (or two-pass) fusion constraint** even if conv/PSN are tiled.

### 3.2 Why compute is the cheap side of retain/recompute

> In DNN accelerators, on-chip buffers often occupy a large share of the chip’s area [11], [12], [20]. In contrast, compute units are abundant and use less energy compared to reading from buffers [11], [12], [20]. Fused-layer dataflows can take advantage of this fact by recomputing instead of retaining intermediate data [16], [21], [22].

This is the ISSCC non-overlapped-LF alternative: **pay MAC to avoid SRAM**, instead of zero-padding a broken RF.

### 3.3 Four design-space axes (Table I) — what prior LF missed

> • Partitioned ranks. … most prior work only supports a limited set of ranks to partition.
> • Recomputation. … Most prior works do not support or support only a limited set of recomputation choices.
> • Per-intermediate-fmap recomputation. … prior work that supports extensive recomputation choices is limited to applying the same recomputation choice for all intermediate fmaps.
> • Per-tensor retention. Some prior works are limited in choosing the shape of tensor tiles that are retained.

Table I (framework vs this work): TANGRAM, DeFiNES, ConvFusion, Optimus, SET, FLAT, TileFlow — **none** simultaneously support any-rank partition + all recomputation + per-intermediate-fmap recompute + per-tensor retain. LoopTree claims to be first that does.

**r1 map.** Dual consumers **are** two tensors with different reuse: gate bitmap vs continuous r1out vs identity vs BN moments vs proj weights. Uniform retain (same tile for all of them) is the thing LoopTree shows is up to ~9–10× worse in buffer. **Per-tensor retain is A**, not X.

### 3.4 Tiling × retain-recompute interaction (II-C, Fig. 3)

> Fused-layer dataflows tile layers in order to retain only tiles of intermediate fmaps … Fig. 3(a) shows an example of tiling by partitioning the output row of the second layer (i.e., rank P2 …).
>
> Fused-layer dataflows can also reduce required on-chip buffer capacity through recomputation. For example, note that tiles in iterations 0 and 1 in Fig. 3(a) overlap. The overlap contains activations in Fmap2 that are computed and used in iterations 0 and 1. At iteration 0, we have two choices: (1) retain the common activations in a buffer to reuse in iteration 1, or (2) do not retain them to save buffer capacity but recompute them later. This retention-recomputation choice trades off the buffer capacity required for the intermediate fmap tile with extra computation.
>
> Note that the tiling choice determines the space of retention-recomputation choices. For example, in Fig. 3(b), tiles of Fmap2 are created by partitioning channels of Fmap2. Because the tiles of Fmap2 in different iterations do not overlap, this tiling choice results in no retention-recomputation choice.

**r1 map.**

- Partition **spatial P/Q** of conv2: 3×3 halo overlap → **retain vs recompute halo** is a real choice (ISSCC’s pad is a third, lossy, choice).
- Partition **channel C** of conv2: no spatial overlap → no halo recompute, but **full spatial fmap** of those channels must still live until projection BN stats exist.
- Partition **time T** of noncausal T10 PSN: tiles do **not** form a closed causal prefix; every output time uses all 10 inputs. T-tiling without retaining the other times is a **recompute of the PSN**, not a fusion win.
- Projection BN’s reduction ranks are **T, C, H, W of the whole domain**. Partitioning any of them for a “local BN” **changes the function** unless moments are the true full-domain moments (which requires either retaining the activations or a first statistics pass).

### 3.5 Mapping choices (Table IV) and “recompute is not a separate knob”

> we first observe that although prior work has proposed recomputation as a separate design choice [16], [21], recomputation can be seen as a consequence of our processing schedule and retention choice. Specifically, if we specify a processing schedule and a retention choice such that an operation needs to access an intermediate fmap activation that is not retained in on-chip nor off-chip buffers, then we have to recompute that activation. Note the resemblance with non-intermediate fmap tensors where data that is not retained in on-chip buffers need to be refetched from off-chip buffers.

Retention specification:

> In LoopTree, we make a retention choice for each tensor by choosing the last rank partitioned to form the retained tile, which can be one or none of the partitioned ranks, and a buffer in the architecture that retains the data.

Schedule constraint they **always** keep:

> Our scheduling follows the constraint that the output from one layer is immediately consumed by the next layer.

**r1 map.** Projection BN **violates “immediately consumed by the next layer”** unless the “next layer” is defined as a **global reduction** whose output (μ, σ²) is not available until the last spatial/time/channel sample. After that, every activation must be **retained or recomputed** for the affine `γ(x−μ)/√(σ²+ε)+β`. LoopTree’s own representation can encode this: BN stats are a tensor that must be **fully retained** (untiled) before any normalized tile is legal. A local-window engine that is handed **free μ/σ** is silently assuming that tensor is already in a buffer of size 1 — `PROBLEM.md` forbids that accounting.

### 3.6 Analysis pipeline (IV) — what the model will and will not charge

Tile-shape walk (Fig. 10), last layer backward:

> Step 3) From the data tile of the fmap, we subtract the amount that is retained from previous iterations … only a subset of the Fmap3 tile needs to be computed.
> Step 4) The part of the fmap tile that is not retained from previous iterations has to be produced … This may include the recomputation of certain operations.

Pipeline latency (Fig. 12): sequential latency minus min(hidden). **Assumptions (do not import as free):**

> LoopTree assumes explicit data orchestration using Buffets [41] such that pipeline stalls can be assumed to be negligible.
> … this analysis assumes that data layout reordering is performed during processing such that the reordering does not increase latency.

Validation: worst-case **4%** vs DepFin, Fused-layer CNN, ISAAC, PipeLayer, FLAT. Table V: **all five validation designs use “Fully retain”** — LoopTree’s recompute axis is **not** validated against a silicon recompute engine in that table.

**r1 map.** Same-port / same-backpressure in `PROBLEM.md` is the **anti-Buffet** constraint. If a fusion idea’s predicted win exists only under “stalls negligible + free layout reorder + free μ/σ,” it is not a TCAS-II measured advantage.

### 3.7 Case-study takeaways (VI) — numbers as prior, not r1

Quoted / paraphrased only as LoopTree’s own CNN/transformer fusion-set results:

- Partitioned ranks + schedule can change required on-chip capacity by **up to 10×** for the same algorithmic-min off-chip traffic (VI-B, Fig. 14). Takeaway 1: *the schedule that fully reuses the **smallest** tensors usually wins.*
- Recompute can cut buffer **~2×** at **~+10%** compute (intro / II-D). Takeaway 2: *retain-recompute, ranks, and schedule must be searched together.*
- Per-tensor retain vs uniform: **up to 9×** buffer (VI-D, Fig. 16; intro says 10×). Takeaway 3: *match retain shape to each tensor’s reuse.*
- Per-fmap retain-recompute (conv+conv+conv): mixing choices beats uniform; **recomputing a later fmap forces more retain/recompute of earlier fmaps** (compounding, citing Alwani [16]). Takeaway 4.
- **Takeaway 5 (VI-F, Fig. 18):** at buffer capacities **much lower than required for algorithmic-min off-chip transfers, fused-layer dataflows are often less efficient than layer-by-layer**, because intra-layer reuse is more abundant (e.g. 3×3×64 = 576 reads per activation) than the one extra read+write saved by inter-layer reuse.

fc+fc fusion set:

> we note that the fc+fc fusion set does not have retention-recomputation choices because all partitioned ranks and schedule choices for fc+fc result in intermediate fmap tiles that do not overlap.

**r1 map of takeaway 5.** r1 already has **heavy intra-layer reuse** in two 3×3×96 convs. Fusing into proj does **not** automatically beat layer-by-layer if the on-chip budget is small — unless the fusion **also** removes off-chip r1out. Projection BN **re-opens** that off-chip (or on-chip full-domain) traffic. A TCAS-II “we fused r1” brief that only shows a local 8×8 window with gifted μ/σ is exactly the accounting LoopTree’s takeaway 5 + `PROBLEM.md` both reject.

Fusion-set selection is **exogenous**:

> Methods for finding the optimal fusion sets have been explored extensively in prior work (see Section VII). … Orthogonally, LoopTree is a model to find the optimal design choices **for a fusion set**.

You must **first** name the r1 fusion set `{conv1, PSN, conv2, BN2, proj.conv_res, proj.sn, proj.conv, proj.BN}`. LoopTree will not tell you to drop proj.BN from the set.

---

## 4. RISCSparse (ICCAD 2024) — sparse coordinate traffic

**Title:** RISCSparse: Point Cloud Inference Engine on RISC-V Processor.

**Authors:** Shangran Lin, Xinrui Zhu, Baohui Xie, Tinghuan Chen, Cheng Zhuo, Qi Sun, Bei Yu. ICCAD’24.

Not an SNN paper. It is the cleanest local text on **(coordinate mapping tax) + (gather/scatter tax) + (BN fused as frozen affine)**.

### 4.1 Bottlenecks they named (intro + §2.1, Fig. 4)

> We address three critical bottlenecks of SSCNs — Rule Map Construction (Mapping), Gather-MatMul-Scatter (GMS), and uncombined operation …
>
> After maps are generated, SSC adopts a Gather-Matrix Multiply-Scatter (GMS) operation … a gather operation consolidates input data linked to identical weights, as directed by the hash table. This data is then routed to the Matrix Multiplication unit … Finally, the outcomes are scattered into the appropriate positions in the output features.
>
> … matrix multiplication and mapping search constitute the primary computational tasks, accounting for approximately 60% to 70% of the total runtime.

CPU breakdown (Fig. 4 caption + bars): Mapping ~36%, GMS ~40% on CPU; on GPU Mapping ~38%, GMS ~29.5%. Scatter/gather **inside** GMS are large (GPU scatter ~35% of GMS).

Submanifold vs dense (Fig. 2):

> In dense convolution, the output feature will progressively dilate … In contrast, SSCN only calculates on the non-zero input site, leading to substantial computational savings and avoiding the dilation issue.

**r1 map.** If the gate path turns conv2 into “only fire at sn2-active sites,” you have inherited SSC’s **mapping + GMS**, not only its MatMul. `PROBLEM.md` integer consumer model (−5.78%) is a **different resource point** from the SIMD source table; coordinate traffic would land in the consumer model, not in the 260→159 add/sub CSE.

### 4.2 Mapping: hash of coordinates, then (pin, δ, pout) tuples (§3.2)

> During the Mapping phase of SC, we generate a kernel map that systematically indexes the necessary GEMM operations. This map includes the corresponding weights and feature vectors as a tuple (p_in, p_out, δ_k) required for each GEMM.

Vectorized hashing: transpose `N×(x,y,z,b)` → per-dimension vectors; SIMD bitwise hash; linear probing with conflict detection (scatter unique lane IDs, gather, check). Insertion may **reorder**; query uses a nested loop to **preserve input-to-output order** because

> Order preservation is essential for mapping calculations in SC spaces, where the sequential integrity of the input-to-output relationship must be maintained.

**r1 map.** Dual consumers need a **stable order** of the same source coordinates: gate path wants compressed active sites; `conv_res` wants even/even dense sites; proj.BN wants **every** site in the `10×96×120×160` domain for moments. A hash-order GMS that is legal for sparse conv2 can be **illegal** for BN reduction unless you re-sort (RISCSparse already paid this). Coordinate metadata (hash table, kernel map, TC/TR offsets in CFMP) is **not** free relative to r1’s 40 lifting coefficients (80 B).

### 4.3 GMS offload (§3.3, Fig. 8)

> The kernel map execution primarily involves a sequence of GEMM operations of **varying sizes**. … Only positions with valid indices, i.e., non-negative, are considered …
>
> With GEMM operations now efficiently offloaded to the systolic array, we turn our attention to optimizing the channel processing through advanced gather and scatter techniques. … The process begins by using a vector mask operation to extract valid positions. Positions P_in in the kernel map that is negative are considered invalid and are masked out. These valid positions are then consolidated into a contiguous array.

Fig. 8: mask-filter indices → input buffer → systolic GEMM → output buffer with **vector accumulate** on Cout.

**r1 map.** This is the hardware of “sparse conv2 from sn2 gates.” It does **not** remove:

- halo of 3×3 (mapping still emits `δ ∈ Δ(K)` neighbors);
- identity path (dense, not in the kernel map);
- projection BN (needs dense reconstructed Y, i.e. a DRU/scatter into the **full** domain).

### 4.4 Operator fusion of BN — **frozen** μ, σ² (§3.4)  [central conflict with PROBLEM.md]

> During inference, Batch Normalization (BatchNorm) operates with **fixed parameters: mean (μ) and variance (σ²)**, which can be efficiently computed as a GEMM operation suitable for Systolic Arrays. Given the BatchNorm operation
>
> `y_i = γ (x_i − μ) / √(σ²+ε) + β`
>
> … Defining `a = γ/√(σ²+ε)` and `b = β − γμ/√(σ²+ε)`, simplifies to `Y = aX + B` …
> This configuration allows the operation to be efficiently executed using `Y = A · X + B`.
>
> This computational approach effectively utilizes the capabilities of Systolic Arrays, accelerating BatchNorm computation latency while enabling ReLU operations post-BatchNorm integration. … we utilize the ONNX Runtime’s graph optimization capabilities to fuse BatchNorm + ReLU and Add + ReLU operators …

**This is the opposite of captured r1 projection BN.** RISCSparse’s fusion A is: **fold running stats into affine, fuse into SA, never look at the batch again.** `PROBLEM.md`: native projection BN uses **actual batch statistics over the full domain**. You **cannot** legally emit `Y = aX + B` with compile-time `a,b`. Local-window replay that **is given** `μ,σ` is using RISCSparse’s inference assumption **without paying** the reduction.

If a student **does** freeze projection BN (running stats / train32 calibration), RISCSparse §3.4 becomes ordinary A and the full-domain hole **closes**. Until that student exists and passes AEE gates, treat unfrozen proj.BN as a **global barrier** in the fusion set.

### 4.5 Scaling: mapping dominates at large N (Fig. 12, §4.3)

> RISCSparse’s acceleration rate tends to decrease with larger data sizes. … the mapping phase becomes critical with larger input sizes and predominates the SC operations. This trend diminishes the perceived benefits of leveraging the SA to accelerate GEMM operations. Consequently, RISCSparse is particularly effective for small to moderate workloads, but its benefits are less noticeable for larger ones.

Domain here is `10×96×120×160 = 18,432,000` activations per projection-BN reduction (one captured tensor; batch axis may already be 1). That is **not** a 512–4096 point cloud. Coordinate-structure ideas that win at MinkUNet-1k **lose** when mapping/hash/GMS scale with occupied sites, and they **still** lose if the consumer is a dense full-domain BN.

WS systolic + larger accumulator helped modest GEMMs; larger scratchpad did not (Fig. 13–14). Do not import Gemmini SRAM configs as r1 PPA.

---

## 5. Joint map: r1 `conv–PSN–conv2–BN–proj` is not a closed local window

### 5.1 Stage-by-stage

| Stage | Live tensors | ISSCC 23.2 analog | LoopTree analog | RISCSparse analog | Closed local window? |
|---|---|---|---|---|---|
| **conv1** | input (r0out / identity later), W1 3×3, Y1 | fused conv tile (VACF/LACF FC) | producer tile; partition P/Q → halo retain/recompute; partition C → no overlap | dense GEMM unless input already sparse | **Yes, spatially**, modulo 3×3 halo of sn1 |
| **PSN T10 noncausal** | 10-vector per site, A 10×10, θg | **not** LA/VA; do not rename | T is a rank with **full reuse across outputs**; no sliding-window overlap in T | coordinate is `(t,c,h,w)` not `(x,y,z,b)` | **No in time.** All 10 inputs live for all 10 outputs. Lifting CSE (159+35 RNE vs 260) is intra-layer, not LF. |
| **conv2** | θg source, W2 96×96×3×3, psums | FC on VA/LA output tiles; KV-vs-weight slot replace | Fmap2 retain vs recompute halo; per-tensor retain of W2 vs fmap | optional GMS if skipping zeros of sn2; mapping tax | **Spatially yes with halo policy.** Dual consumers of **source** start here. |
| **BN2 + identity** | conv2 out, identity (r0out, already live), affine or frozen stats | CFMP restores **dense** Y before any later head | if frozen: pointwise, fuse; if not: global reduce | §3.4 frozen `Y=aX+B` | Frozen parent: yes. Unfrozen: **no.** Captured parents often fix residual BNs; **do not assume** without the freeze file. |
| **proj.conv_res** | r1out even/even, 1×1 stride2 | extra dense consumer of the same fmap (like a second FC) | second consumer tensor: cannot drop r1out after gate-only last-use | gather even/even sites | Needs r1out **values**, not only gates. |
| **proj.sn** | full `[T10,C96,H240,W320]` r1out | “few VA tiles for global RF” is a **different function** | full spatial consume of r1out | full-domain gather | Not a local 8×8. |
| **proj.conv → proj.norm_layer** | sn θg, 3×3 W, **batch μ,σ over `10×96×120×160`**, then ADD with conv_res | LF **does not** contain a global BN; non-overlapped pad cannot stand in for moments | **untiled retain of all activations** *or* two-pass (stats then recompute affine) *or* off-chip stream (layer-by-layer) | **forbidden** to fold `a,b` at compile time | **No. This is the barrier.** |

### 5.2 Why “local-window replay + free mean/var” undercharges

LoopTree accounting identity: missing data is either **retained**, **refetched**, or **recomputed**. RISCSparse inference BN **deletes** μ,σ from the live set by freezing them. ISSCC LF **deletes** missing halo by **zero-pad + later attention**, which is a new network.

A local r1 window that is given μ,σ is taking RISCSparse’s deletion **and** LoopTree’s “tensor already in buffer” **without a line in the schedule**. Wait undercharge: the window cannot emit a legal `proj.norm_layer` output until the **last** of 18.432M sites has contributed to moments (or a running-stat student is used). Storage undercharge: either the activations stay live until pass 2, or pass 2 recomputes `conv–PSN–conv2` (and possibly proj.sn/conv) after moments exist. Dual consumers make the recompute **more** expensive than a single dense head: you would recompute both gate and continuous paths unless you retained gates.

### 5.3 Dual consumers vs LFS slot replace

ISSCC LFS: finish all VA that need KV → **replace KV slot with FC weights** → fuse conv on VA tiles → reuse those weights on LA tiles.

r1 legal analog (A, not X): finish all **gate and continuous** uses of a source word / r1out tile that **do not need proj.BN**, then replace that SRAM with **projection weights**. Illegal analog: replace the slot **before** `conv_res` and `proj.sn` have both consumed, or before BN moments are known if the affine still needs the activations.

Per-tensor retain (LoopTree VI-D) is the right vocabulary: retain a **compact gate bitmap** longer than the **wide θg vectors** if conv2 is sparse-GMS, while `conv_res` may still need dense values at even/even. Uniform retain of full FP/INT feature volumes is the 9–10× buffer mistake.

### 5.4 Cascaded prune vs r1

CFMP’s `W0/W1` + DRU density restore is **A** for a **single linear** conv (or for a new trained student). It is **not** a factorization of `conv1–PSN–conv2`. If applied to **proj.conv** (post-sn, linear), I/O stay dense and Z is sparse — then **proj.BN still reduces over dense Y**, so the prune must hit MAC of W0/W1, not BN traffic. If the mask is spatial, moments change (new student, AEE gate). If the mask is channel-tiled as in Fig. 23.2.5, BN over remaining channels is still full H×W×T.

### 5.5 Sparse coordinate traffic vs r1

CFMP FMS/DRU and RISCSparse mapping/GMS are the same family: **indices are first-class traffic**.

On r1:

- Occupied-site maps for sn2→conv2: pay hash/kernel-map **or** a dense bitmask of size `T×C×H×W` bits (LoopTree “retain the bitmap tensor”).
- Scatter into conv2 psums: bank conflicts, like SCNN / DRU interleaved TR slices.
- Projection BN: a **reduction tree over coordinates**, then a **broadcast of a,b** back to every coordinate — the opposite of submanifold skip. Submanifold “do not dilate empty sites” does **not** apply to a batch-norm that includes those sites in μ,σ.

Fig. 12 of RISCSparse is the kill-gate shape: as the occupied set approaches the full domain, mapping dominates and SA GEMM wins vanish. Full-domain BN **forces** the occupied set for the stats pass to be 100%.

---

## 6. A / B / X / kill-gate (for independent ideation; not a chosen title)

Use this as **relative prior**. Do not treat the three papers as one mechanism.

### A — complete prior to copy (not the contribution)

1. **Fused-layer dataflow** (Alwani [ISSCC ref 13] + LoopTree taxonomy): tile last consumer, infer producer tiles, **retain or recompute** overlap, per-tensor retain, sequential vs pipeline. Fusion **set is an input**.
2. **LFS slot multiplexing** (ISSCC Fig. 23.2.4): finish consumers of tensor family A, **replace the physical buffer** with family B; do not assume K, V, and W fit together (ISSCC challenge 2).
3. **CFMP** (ISSCC Fig. 23.2.5): train `W0/W1` + tiled mask, FMS indices, compact Z, DRU TC→TR density restore; I/O dense. Copy **training + both factors + index HW**, or do not use the name.
4. **Hybrid attention HAPU**: trained LA/VA mix, KTV-first O(C²), RMPI left/right routing, ATM overflow split. Copy only if the student **is** that attention.
5. **Non-overlapped LF + zero-pad + later global op recovers RF**: copy only as a **new network** with AEE re-measure. ISSCC reports `<0.5%` Cityscapes, 50% GB, −20% ops.
6. **RISCSparse GMS stack**: rule map, gather-by-weight, SA GEMM, scatter, order-preserving query, operator fusion **for frozen-BN** `Y=aX+B`.
7. **LoopTree takeaway 5**: small buffers → layer-by-layer can beat tiled fusion; intra-layer reuse ≫ one extra inter-layer read/write.

### B — hole in **this** net (from `PROBLEM.md`, not from ISSCC energy tables)

- Dual consumers after r1 source: gate **and** continuous PED/`conv_res`. Last-use of a tile for one path is not last-use for the other.
- Noncausal T10 PSN: no causal prefix tile.
- Native **projection BN = actual batch stats over `10×96×120×160`**. Local fused window is **not** closed HW.
- Same-port / same-state / same-backpressure; long backpressure already **8088** on both source arms (fusion gain absorbed). Separate integer consumer table must **not** be added to the SIMD source table.
- Lifting vs ordinary: AEE +0.013178 vs +0.005 relative gate (fails); absolute 1.233 < 1.259 (passes). CSE 260→159+35 RNE is **not** cycle share.
- Historical patch ~35% proxy ≠ new student’s cycles.

None of the three papers measures this B.

### X — why a reskin is not a TCAS-II brief

| Reskin | Why it is not X |
|---|---|
| “r1 layer fusion” | Alwani + ISSCC LFS + LoopTree already name it. Need a **measured** same-port win **after** charging halo and full-domain BN. |
| “hybrid attention PSN” | HAPU is trained LA/VA. T10 ATLIF is a different Einsum. |
| “CFMP on residual” | Two r1 convs have PSN between them; CFMP factors **one linear** conv. |
| “sparse conv2 like SSCN” | Mapping+GMS 60–70% is the prior; full-domain BN forces dense consume. |
| “fuse BN into conv” | RISCSparse does this **iff μ,σ frozen**. Captured proj.BN is not. |
| “non-overlapped tiles, pad halo” | ISSCC already did it **with accuracy drop** and later attention RF repair. Residual + BN stats cannot use that repair. |
| Chip TOPS/W, μJ/token, 22.05× | Wrong identity, wrong workload, forbidden multiplier. |

A true increment, if any, would have to be a **schedule/state mechanism that charges full-domain moments and dual consumers on the same ports**, not a rename of HAPU/LFS/CFMP/GMS.

### Strongest controls (if anyone proposes fusion HW)

- LoopTree-complete: same fusion set, **explicit** retain vs recompute vs refetch for **each** of {Y1, θg, W2, identity, r1out, BN moments, proj W}.
- Frozen-BN student (RISCSparse §3.4) **and** unfrozen full-domain student; AEE gates on both.
- Layer-by-layer **and** untiled fusion (full r1out on-chip) as LoopTree VI-F baseline — tiled fusion must beat **both** at the **same** buffer/port/backpressure.
- Dense zero-skip conv2 **and** bitmask-retain conv2 **and** GMS conv2, all paying mapping/index bytes.
- ISSCC non-overlapped pad student, **re-trained**, valid825, not a silent pad in HW of the frozen net.
- Same-port source table and consumer table **reported separately** (do not add −22.83% to −5.78%).

### Kill-gate numbers (local, not ISSCC)

- valid825 AEE **> 1.259** absolute, or **> +0.005** vs ordinary dense-source/raw (ordinary = 1.219801338).
- Full-chain same-port net service **< 15%** after charging BN wait/storage (gifted μ/σ **invalidates** the experiment).
- Index/mapping/DRU traffic ≥ arithmetic saved (RISCSparse Fig. 12 shape).
- Long-backpressure remaining **8088** with no consumer-side closed chain.
- Any FPS/PPA obtained by multiplying HAPU×LFS×CFMP or quoting 52.90 TOPS/W / 0.22 μJ/token as local.

### Two-sentence TCAS-II prior statement (for reviewers, not a pitch of a new chip)

ISSCC 2025 23.2 already combines hybrid-attention tiling, KV-then-weight layer-fusion, and cascaded dense-I/O pruning with explicit **lossy** boundary padding; LoopTree already makes retain/recompute/per-tensor occupancy a complete mapspace; RISCSparse already fuses **frozen** BN as `Y=aX+B` and shows mapping/GMS dominating sparse conv. A brief on r1 fusion that does not (i) name those three as relative prior and (ii) keep projection BN’s **full-domain** moments in the live set will read as a rename.

### Biggest objection

The fusion set is not closed at a spatial window: **proj.norm_layer is a global reduction**. Every ISSCC/LoopTree/RISCSparse trick that assumes “tile produced ⇒ tile consumable ⇒ tile releasable” fails at that node unless the student freezes BN (new AEE) or the schedule **pays** a stats pass plus retain-or-recompute of the activations. Dual consumers make the cheap “drop the fmap, keep a gate bit” retain illegal for `conv_res`.

---

## 7. What this note does **not** do

- Does not choose a title X or add modules.
- Does not treat 28 nm / TOPS/W / μJ/token as transferable.
- Does not treat `2512.17555` as a second paper.
- Does not assume residual `norm1/norm2` are unfrozen; the freeze file only names **native projection BN**.
- Does not add the two resource tables in `PROBLEM.md`.
- Does not claim LoopTree’s 4% model error holds under finite backpressure.

**Sources (absolute):**

- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/ISSCC2025_23_2_ConvFormer_author.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2512.17555.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/LoopTree_TCASAI2024_author.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/literature/RISCSparse_ICCAD2024_author.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/PROBLEM.md`
