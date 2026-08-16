# Deep Idea Mining Update: Unified Binary DSEC Attention (2026-07-12)

## 0. Scope and decision boundary

This note is a formula/code/hardware audit for the parallel idea-mining track. It does not register an H number, modify the model, or start an experiment.

Frozen constraints:

- DSEC only; standard crop `288x384`.
- All 12 encoder attention blocks use one formula. Different learned weights are allowed, but there is no S2-only path, no stage-wise TX/SC selection, and no alternating NA/DiNA schedule.
- All 105 neuron wrappers remain one-sided binary ATLIF, emitting `{0,+theta}`.
- No native QKFormer carrier `K*sn2_q(sum_channel(Q))` before a second gate. The first three candidates below also avoid SC entirely.
- An attention-block-local redesign is allowed. Encoder/decoder topology and the existing hardware schedule outside the attention block stay fixed.
- H60 dyadic deployment is the reference: AEE `1.5016`, AAE `9.8431`, total spikes `23.2439G`. A candidate is not a mainline result until independent full30 plus standard valid825 is complete.

The current window geometry is `T=2`, `H=W=9`, `N=162`, with `D=32` per head. Let token `i=(t,y,x)`, paired time `bar(t)=1-t`, and binary Q/K be `q_i,k_j in {0,1}^32`. Define

```text
n11(q,k) = popcount(q & k)
n00(q,k) = popcount((~q) & (~k)) over the valid D bits
AX_alpha(q,k) = n11(q,k) + alpha*n00(q,k)
```

Deployment uses `alpha=1/64`, so `64*AX = 64*n11+n00` is exact integer shift/add. `Shiftmax_R` below means the repository's dyadic normalization over candidate axis `R`, not floating softmax.

## 1. Source-level findings, not abstract-level analogies

### 1.1 EEMFlow DFC is a cross-frame cost descriptor

[EEMFlow, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_Efficient_Meshflow_and_Optical_Flow_Estimation_from_Event_Cameras_CVPR_2024_paper.pdf) defines displaced feature correlation

```text
C_i(u,d) = V_tk^i(u) dot V_t{k-1}^i(u+d) / M,  d in N.
```

The [official implementation at commit 1ba77c4](https://github.com/boomluo02/EEMFlow/blob/1ba77c4dcb45d22b175f835da5a4a0f310cb863b/model/EEMFlow/EEMFlow.py#L84-L100) instantiates `Correlation(4)`, hence a 9x9 correlation, and hard-codes 49 zero-based channels. Those offsets are exactly the Manhattan-distance rings `0,1,2,3,5,7`, with counts `1,4,8,12,16,8`. At [lines 160, 167 and 173](https://github.com/boomluo02/EEMFlow/blob/1ba77c4dcb45d22b175f835da5a4a0f310cb863b/model/EEMFlow/EEMFlow.py#L158-L176), the code computes the full correlation first and then `index_select`s the 49 channels. It concatenates the selected cost channels with 16 feature channels and sends them directly to a decoder. Thus the cost vector itself is a useful motion representation; EEMFlow does not first turn it into `weights@K`.

Hardware correction: copying that PyTorch order would calculate 81 and discard 32. A defensible circuit must address only the selected offsets and never generate the omitted channels.

### 1.2 EEMFlow CDC is not encoder attention

The paper's completion rule is

```text
F_tilde = alpha*W(F_bar,DeltaF) + (1-alpha)*(A tensor-product F_bar)
F_up = W_conf .* F_bar + (1-W_conf) .* F_tilde.
```

The [official `cdc_model`](https://github.com/boomluo02/EEMFlow/blob/1ba77c4dcb45d22b175f835da5a4a0f310cb863b/model/EEMFlow/cdc_utils.py#L147-L174) warps feature 2 with an initial flow, concatenates two features, runs a five-layer dense convolutional estimator, predicts 2-D intermediate flow plus a sigmoid confidence mask, and blends warped and initial flow. Its [CFP path](https://github.com/boomluo02/EEMFlow/blob/1ba77c4dcb45d22b175f835da5a4a0f310cb863b/model/EEMFlow/cdc_utils.py#L179-L209) additionally forms an `N x N` self-correlation and multiplies it by flow. Direct CDC transplantation is therefore rejected. Only the principle "use an expensive completion path when the direct match is uncertain" is transferable, and any such adaptation must be named CDC-inspired rather than CDC reproduction.

### 1.3 Pixel-level token reduction requires full-resolution compensation

[DAR-TR-PEFT, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lei_Rethinking_Token_Reduction_with_Parameter-Efficient_Fine-Tuning_in_ViT_for_Pixel-Level_CVPR_2025_paper.pdf) reports that token reduction homogenizes head attention and loses high-frequency information in pixel-level tasks. Its method is not a free attention regularizer. In the [official router](https://github.com/AVC2-UESTC/DAR-TR-PEFT/blob/ea7ef525201b26395c5e8c65e902e81a15b18b0d2/src/Models/finetunes/msk_gen.py#L10-L70), a learned 3x3 convolution feeds Gumbel-sigmoid and a hard threshold. In the [block forward](https://github.com/AVC2-UESTC/DAR-TR-PEFT/blob/ea7ef525201b26395c5e8c65e902e81a15b18b0d2/src/Models/backbones/dinov2_vit_l_ft_dar.py#L101-L153), training masks MLP output while inference gathers selected tokens and scatters them back. The [compensator](https://github.com/AVC2-UESTC/DAR-TR-PEFT/blob/ea7ef525201b26395c5e8c65e902e81a15b18b0d2/src/Models/finetunes/dar.py#L202-L240) is `1x1 -> 3x3 depthwise -> 1x1 -> activation -> 1x1` at full spatial resolution.

Conclusion: dynamic token deletion is a poor fit here. The useful requirement is to preserve one output per DSEC pixel/token and retain local displacement diversity. All candidates below do so; none gathers or drops tokens.

### 1.4 Neighborhood attention supports fixed-cardinality locality, not our schedule

[Neighborhood Attention Transformer, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Hassani_Neighborhood_Attention_Transformer_CVPR_2023_paper.pdf) defines local attention for query `i` over its nearest `k` keys, preserving a sliding receptive field and linear complexity. The [official DiNAT configurations](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/d334f5bb0bd73cad05e5aeaf7c14cce3876a9650/segmentation/configs/dinat/upernet_dinat_tiny_512x512_160k_ade20k.py#L14) alternate layer-specific dilation factors. That alternating schedule violates this project's all12 identity and is not copied. What remains useful is the evidence that a fixed number of spatial neighbors can replace an all-pairs map while retaining pixel topology.

### 1.5 Integer attention papers separate algorithm from implementation

[FQ-ViT, IJCAI 2022](https://arxiv.org/abs/2111.13824) uses integer polynomial exponential and log2-quantized attention. Its [official LIS code](https://github.com/megvii-research/FQ-ViT/blob/5daf5915c715dc21d6afcd64e661652030e640ab3/models/ptq/layers.py#L237-L288) computes integer exponential, then stores `qlog=round(log2(sum_exp/exp_i))` and reconstructs a weight as `2^(-qlog)`. This supports exponent-coded gates, but it does not add motion semantics and is not an accuracy idea by itself.

[ITA](https://arxiv.org/abs/2307.03493) is an independently auditable integer-attention accelerator. Its [official RTL](https://github.com/pulp-platform/ita/blob/ba96519becce195d64e85eb9a5302e8a1d5487e7/src/ita_softmax.sv#L136-L183) updates a row maximum, right-shifts the old denominator when that maximum changes, and accumulates `256 >> shift`. It then [replays the row](https://github.com/pulp-platform/ita/blob/ba96519becce195d64e85eb9a5302e8a1d5487e7/src/ita_softmax.sv#L211-L230) and produces normalized values by shifts. This is directly useful for an exact streaming Shiftmax microarchitecture, described in Section 4.

## 2. Candidate A, P0: MC49 direct cross-time Match-Code attention

### 2.1 Migrated formula and tensor semantics

Use EEMFlow's exact 49-offset set `Omega49` in every block, but compute binary alpha-XNOR directly rather than dense dot products:

```text
c[b,h,i,r] = AX_1/64(q[b,h,t,y,x], k[b,h,bar(t),y+dy_r,x+dx_r]) / D
a[b,h,i,:] = Shiftmax_Omega49(c[b,h,i,:] + boundary_mask)
z[b,h,i,:] = C[l,h] * a[b,h,i,:],       C in Z8^(32x49)
Y[b,i,:] = W_o[l] * concat_h(z[b,h,i,:]).
```

`c` and `a` are `[Bwin, heads, 162, 49]`; `z` is `[Bwin, heads, 162, 32]`. Invalid offsets are assigned the minimum representable score before Shiftmax; they must not clamp to an edge or wrap across a Swin window. `C` is a static learned codebook local to the attention block and is quantized to INT8 for deployment. It converts a displacement-probability vector into the existing 32 feature lanes. There is no V tensor, no K value reread, no `gate*K`, and no SC term.

This is an adaptation, not EEMFlow reproduction: EEMFlow uses dense real-valued features, three pooled scales, and a decoder; MC49 uses binary Q/K inside each existing encoder block.

### 2.2 All12 identity and hardware cost

- Identity: all 12 blocks use the same `Omega49`, alpha, boundary rule, Shiftmax axis, and codebook operation. Stage resolution changes physical receptive field naturally, but the formula does not change.
- Arithmetic per token/head: 49 32-bit AND/popcount pairs for `n11/n00`, one 49-lane Shiftmax, and a `32x49` static INT8 projection. H60 uses one token-local score and `32` gate-value products, so MC49 is an accuracy-first block, not a lower-operation block.
- Storage: one paired-time 9x9 K window (`81x32` bits/head) or equivalent banked SRAM, at most 49 INT8 scores (`392` bits/query/head), and `49x32x8` codebook bits per head/block unless weights are compressed. No V/K-value buffer is needed after score generation.
- Bandwidth: without a local K window, reads are prohibitive (`49x32` bits/query/head). With a loaded opposite-time window, external bandwidth is one K window per head/window; the 49 accesses become SRAM reads. The implementation must report bank conflicts and cycles, not count them as free.
- Control: fixed ROM offsets, boundary-valid mask, 49-cycle scalar lane or multiple replicated match lanes, Shiftmax row state. No data-dependent routing.

### 2.3 Risk and non-duplication

- Main risk is compute/latency and the newly initialized codebook, not irregularity. A 49-lane cost descriptor may also over-focus on window-relative displacement at deep stages.
- It is not H66a: H66a compares all 162 tokens and returns `weights@K`; MC49 compares only cross-time fixed offsets and returns the score descriptor itself.
- It is not H66c/H66d: those return weighted K from 2 or 5 self/same-time candidates and discard displacement-channel identity.
- It is not H67: H67 adds one same-position temporal XOR scalar to H60.
- It is not H68-H71: no training-only matrix, score temperature, event-density selector, or window-mean broadcast.
- It is not H63: H63 broadcast one/group scalar gate into feature channels. MC49 assigns each lane a stable displacement basis before a learned static codebook, so channel rank carries motion semantics.

### 2.4 Minimal verifiable full30 protocol

1. One frozen point only: exact `Omega49`, `alpha=1/64` deployment, no offset/count sweep and no SC/Kmag fallback.
2. Warm-start from the same H60 TTX epoch2 checkpoint used by H67-H71. Load all compatible parameters. Expected new keys are exactly 12 `match_code.weight` tensors; zero other missing keys, zero unexpected keys, and zero unresolved overlay-owned keys.
3. Initialize each codebook with a deterministic orthogonal/Xavier matrix in software; train it normally and quantize to INT8 only in deployment evaluation. This initialization is part of the protocol, not a sweep.
4. DSEC `288x384`, batch 8, workers 8, AMP/cupy, 30 epochs, warmup 720, milestones 20/25. Save/evaluate epochs `0/4/9/14/19/24/28/29` on standard valid825.
5. Required audits: ATLIF `105`, patched attention `12`, Q/K shape and paired-time index checks, no boundary wrap, checkpoint key audit, float plus dyadic valid825, total spikes, and attention-inclusive operations/SRAM traffic.
6. Promotion: AEE below H60 `1.5016` is the competitive target. At minimum it must remain within `1.05*NB0` and retain the NB0-relative 20% spike reduction. AAE regression greater than `0.3` requires an explicit flow-boundary breakdown.

## 3. Candidate B, P0/P1: DE9 dual-evidence Match-Code attention

### 3.1 Why split the alpha-XNOR evidence

[Alpha-XNOR SSA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.pdf) defines spike-spike agreement and alpha-weighted silence-silence agreement as one score. Its Eq. 25 then linearly transforms the score before V aggregation. For sparse one-sided DSEC events, fixing `alpha=1/64` before all downstream mixing may suppress useful static/background evidence too early. DE9 keeps the two already-computed integer counts separate until the output projection.

### 3.2 Migrated formula

Let `Omega9={-1,0,1}x{-1,0,1}` at the paired time:

```text
u_r = n11(q_i,k_pair(i,r)) / D
v_r = n00(q_i,k_pair(i,r)) / D
a_plus = Shiftmax_Omega9(u)
a_zero = Shiftmax_Omega9(v)          # not SC; this is the second alpha-XNOR term
d_i = concat(a_plus,a_zero)          # 18 displacement-evidence lanes
z_i = C[l,h] * d_i,                  # C in Z8^(32x18)
Y_i = W_o[l] * concat_h(z_i).
```

Both branches are TX evidence from the same Q/K comparison. There is no TX/SC mixture, no K/V carrier, and no token removal. Deployment can optionally fold `alpha=1/64` into the initialized/scaled `C` columns for the silence branch; it must not add a runtime multiplier.

### 3.3 Cost, risks, and overlap audit

- Arithmetic: 9 binary comparisons/head/token, two 9-lane Shiftmax contexts, and one `32x18` static INT8 codebook. `n11` and `n00` come from the same 32-bit words.
- Storage: paired-time 3-row K line buffer plus halo, 18 INT8 descriptor values, two max/denominator states, and `32x18x8` codebook bits/head/block.
- Bandwidth: 9 K reads from the local buffer per query. No second K read for value aggregation.
- Control is fixed and regular. Compared with MC49, it reduces comparisons by 81.6% and codebook MACs by 63.3%.
- Accuracy risk: splitting evidence can make the silence branch nearly uniform; two independent normalizations remove the original relative mass between `n11` and `n00`. The output codebook must learn to reject uninformative silence lanes.
- Non-duplication: H60/H67 collapse evidence to one token-local scalar; H69/H70 change scalar temperature; H71 broadcasts a mean. H66 variants aggregate K. None exposes a displacement-indexed two-part alpha-XNOR descriptor.

### 3.4 Full30 protocol

Use the same full30 schedule, start checkpoint, valid825 epochs, and load audits as MC49. Freeze `Omega9`, two Shiftmax contexts, and codebook width 18. Expected missing keys are exactly 12 codebooks. Do not sweep alpha or neighborhood size. Compare DE9 directly against MC49 and H60 on AEE/AAE, attention latency, codebook SRAM, and per-stage boundary AEE. DE9 should run before MC49 only if hardware turnaround is the immediate bottleneck; for pure accuracy discovery MC49 remains first.

## 4. Candidate C, P1: XD13 cross-time dilated alpha-XNOR aggregation

### 4.1 Migrated formula

This is the conservative value-aggregation alternative. Use the symmetric cross-time stencil

```text
Omega13 = {-1,0,1}x{-1,0,1} union {(-2,0),(2,0),(0,-2),(0,2)}.
s_r = AX_1/64(q_i,k_pair(i,r)) / D
a = Shiftmax_Omega13(s + boundary_mask)
Y_i = sum_r a_r * k_pair(i,r).
```

The formula is identical in all12 and uses only alpha-XNOR. It retains one output per token and 32 lanes per head. It has no native carrier, but it does retain dynamic weighted-K aggregation; therefore it is less clean than MC49/DE9 for a design that wants to eliminate all dynamic value scaling.

### 4.2 Cost and risk

- Arithmetic: 13 32-bit comparisons, one 13-lane Shiftmax, 13 gated 32-lane K accumulations per query/head.
- Storage: paired-time five-row K buffer or full 9x9 K window, 13 scores/gates, and a 32-lane accumulator. No new learned parameter or checkpoint key.
- Bandwidth: the same K words are used for score and value. A score-stationary engine should retain/replay them locally; otherwise K traffic doubles.
- Control is fixed; invalid candidates are masked. No top-k, sorting, hash, or dynamic lane count.
- Accuracy risk: H66c valid825 (`1.6567/10.4283`) shows that same-position temporal K aggregation alone is insufficient, while H66d's same-time local-5 short result was also weak. XD13 is still non-duplicate because every non-center candidate is a cross-time spatial displacement, the semantic primitive needed for flow. Nevertheless, those results lower its priority below direct Match-Code outputs.

### 4.3 Full30 protocol

One frozen `Omega13` point from H60 epoch2; no codebook and therefore all checkpoint keys must load with zero missing/unexpected. Use the common full30/valid825 schedule. Audit 105/12 counts, no wrap, 13 candidates in every block, and both float/dyadic paths. Run only after one direct Match-Code candidate, unless codebook integration is blocked. A valid825 result no better than H66c stops larger K-aggregation stencils; it does not trigger a 25/49-offset sweep.

## 5. Candidate D, P2 conditional: CE-MC confidence-expanded Match-Code

### 5.1 CDC-inspired but not CDC

CE-MC uses a fixed local descriptor for every query and expands to EEMFlow's far rings only when the local match is ambiguous. It copies neither CDC's warp nor its convolutional confidence network.

```text
Compute DE9 local scores s over Omega9.
m1,m2 = largest and second-largest valid s values.
uncertain = 1[(m1-m2) < Delta], with fixed integer Delta.
If uncertain, additionally compute offsets with Manhattan radius 3,5,7.
Missing far lanes are zero-masked; a fixed 49-lane Match-Code projection produces z_i.
```

All12 use the same `Delta`, offset order, and projection. This is still pure TX. It adds a top-2 comparator tree, a one-bit uncertainty map, pending-query state, and variable K-buffer accesses. Worst-case cost equals MC49; best-case cost is 9 comparisons. Workload divergence and score-lane masking are the primary hardware risks.

### 5.2 Relation to H67-H71 and full30 protocol

H70 selects temperature from event density; CE-MC selects extra displacement candidates from match margin. H67 adds temporal change to a scalar score; CE-MC searches actual cross-time offsets. H71 always broadcasts context; CE-MC is query-local. It is not a first-run candidate because `Delta` would otherwise become an unconstrained sweep.

Only run CE-MC if MC49 beats H60 and profiling shows most queries have a clear local margin. Pre-register `Delta` as the single 75th-percentile integer margin measured on the frozen MC49 training split, without labels and without trying alternatives. Then train independently for full30 from H60 epoch2 using the common schedule. Report average, P95, and worst-case candidate counts in addition to AEE/AAE/spikes. A hardware claim must use worst-case provisioned latency and separately report average energy.

## 6. Independent hardware line A: exact online Shiftmax/match engine

This line changes scheduling, not the network formula. For a stream of integer scores `s_r`, maintain row maximum `m`, denominator `Z`, and, for XD13, a 32-lane weighted numerator `U`:

```text
m_new = max(m, max(block))
Delta = m_new-m
Z_new = (Z >> Delta) + sum_r (Q >> (m_new-s_r))
U_new = (U >> Delta) + sum_r ((Q >> (m_new-s_r))*K_r)
```

`Q` is the fixed Shiftmax numerator scale. This is the ITA max-rescale recurrence specialized to the repository's integer score scale. For Match-Code outputs, retain/replay candidate scores and emit `a_r=(Q/Z)>>(m-s_r)`; for XD13, `U/Z` directly avoids materializing all gates. Exactness requires the software golden model and RTL to share saturation width, round-half rule, shift clipping, reciprocal/divider behavior, and candidate order.

Audit units:

- Arithmetic: comparator tree, variable right shifters, 19-bit-or-derived denominator accumulator, reciprocal/divider, optional 32 numerator accumulators.
- State per active row: `m`, `Z`, candidate counter, boundary mask; plus `U[32]` for weighted K or a score replay FIFO for Match-Code.
- Storage tradeoff for MC49: at 8-bit scores, full replay storage is 392 bits/query/head. Recomputing scores saves that FIFO but rereads K and repeats popcounts; both options must be synthesized.
- Control: fixed 49/18/13 candidate loops; no token-level dynamic FSM except CE-MC.
- Verification: bit-exact random and recorded DSEC vectors, boundary cases, all-equal scores, maximum changes in the last candidate block, saturation, and backpressure.

There is no separate full30 for this exact line. Its executable protocol is: train the selected algorithm with its full30 protocol, export integer Q/K and software Shiftmax outputs for all 12 blocks, replay them through RTL, require zero mismatches, then compare post-layout PPA against the current H60 attention block under identical SRAM and clock assumptions. Calling this a new accuracy result would be incorrect.

## 7. Independent hardware line B: log-exponent gate storage

FQ-ViT LIS suggests storing each normalized gate as an exponent `e_r=clip(round(-log2 a_r),0,15)`, so multiplication by binary K becomes a right shift. Applied to XD13:

```text
a_hat_r = 2^(-e_r)
Y_i = sum_r (K_r >> e_r).
```

This removes a general gate multiplier and stores 4 bits/gate, but adds leading-one/log-round logic and changes numerical output. It is distinct from H69: H69 shifts the input score temperature before Shiftmax; this line quantizes the output gate after normalization. It is a B-class approximate hardware candidate, not an automatic optimization.

If used, run one full30 QAT point with 4-bit exponent gates from H60 epoch2, the common DSEC schedule, and standard valid825. No bit-width sweep. Promotion requires matching the float XD13/selected Match-Code AEE within `0.02` while reducing synthesized gate SRAM plus gate-application energy. It cannot be claimed from a frozen-checkpoint PTQ result alone.

## 8. Explicitly rejected transfers

| Transfer | Evidence-based reason not to run now |
|---|---|
| Full EEMFlow CDC/CFP | Warp, five dense conv layers, sigmoid flow mask, dense `N x N` CFP, and flow state alter the decoder/dataflow rather than only the attention block. |
| DAR token router | 3x3 learned router, Gumbel control, gather/scatter, adapter, and depthwise compensator target MLP reduction, not the H60 attention deficiency. |
| Sanger low-bit predictor | Sanger predicts sparse candidates with 4-bit QK before exact higher-precision attention. Q/K here are already 1 bit, so the predictor is not cheaper than the exact alpha-XNOR check. Mask packing/dataflow remains useful, but not as a new algorithm. |
| ELSA SRP hash prefilter | Hash projection, hash SRAM, Hamming threshold, and K norm are extra work before an already-cheap 32-bit binary comparison. It becomes plausible only for much larger candidate sets than 49. |
| DIP iterative inverse PatchMatch | [DIP, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_DIP_Deep_Inverse_Patchmatch_for_High-Resolution_Optical_Flow_CVPR_2022_paper.pdf) uses propagation, warp, local search and GRU aggregation over iterations. That recurrent state and variable iteration count do not fit an all12 local replacement. Its fixed radius-2 search supports `Omega13`, not the iterative module. |
| Alternating NA/DiNA | Official DiNAT deliberately alternates dilation by layer/stage. It is good external evidence for sparse neighborhoods but violates the required all12 formula identity. |
| ViTCoD/Sanger dynamic sparse maps | Their benefit targets large token-token matrices. MC49/DE9 already use fixed bounded candidate sets; adding map prediction or polarization creates control/storage without removing the dominant 32-bit local comparisons. |

## 9. Recommended order and kill rules

| Priority | Candidate | Accuracy rationale | Hardware rationale | Decision |
|---:|---|---|---|---|
| 1 | **MC49 direct Match-Code** | Closest transfer of EEMFlow's actual motion descriptor; preserves displacement identity and removes the failed scalar-broadcast bottleneck | Regular fixed offsets, no K value reread; high but bounded cost | First accuracy-discovery full30 after the frozen H67-H71 queue |
| 2 | **DE9 dual-evidence Match-Code** | Retains local motion identity and lets the model separate event and silence evidence | 9 comparisons, no dynamic value scaling, fixed two-context Shiftmax | Best software/hardware balance; may precede MC49 if implementation cost dominates |
| 3 | **XD13 cross-time aggregation** | Conservative local matching with existing 32-lane output semantics | No new weights; regular 13-candidate engine, but keeps weighted K | Fallback/control, not the preferred no-gate-K mainline |
| 4 | CE-MC | Can approach MC49 accuracy at lower average work | Dynamic margin control and worst-case MC49 provisioning | Conditional only after positive MC49 profiling |

Kill rules:

1. Do not sweep 9/13/25/49 stencils. MC49, DE9 and XD13 test different output semantics, not a neighborhood-size sweep.
2. If MC49 does not beat H60 after full30, do not run CE-MC; pruning a weaker teacher cannot establish the desired accuracy line.
3. If DE9's silence branch entropy is within 1% of uniform in all 12 blocks and its valid825 does not beat H60, do not tune alpha or add a third branch.
4. If XD13 is no better than H66c valid825, stop all larger `weights@K` stencil variants.
5. Keep ITA-style exact streaming as an independently auditable hardware contribution regardless of which fixed candidate wins. Keep log-exponent gates conditional on a measured multiplier/SRAM bottleneck.

## 10. What would constitute a DATE-ready claim

The strongest coherent claim is not "many attention variants were tried." It is:

1. **Algorithm:** one-sided binary ATLIF plus one all12 cross-time binary matching formula, with a displacement-indexed Match-Code output that removes native carrier and TX/SC mixture.
2. **Ablation:** H60 scalar selector versus DE9 local dual evidence versus MC49 dilated motion descriptor; XD13 isolates whether gains come from displacement scores themselves or from K aggregation.
3. **Hardware:** a separately verified fixed-offset match engine and online integer Shiftmax recurrence, with bit-exact software/RTL vectors and attention-inclusive SRAM, latency, area, and energy.
4. **Evidence:** standard DSEC full30/valid825, float and dyadic deployment, 105 ATLIF/12 attention/load audits, NB0-relative accuracy and spike constraints, and no proxy-only energy claim.

Until those runs exist, MC49/DE9/XD13 are proposals with source-level support, not measured improvements over H60.
