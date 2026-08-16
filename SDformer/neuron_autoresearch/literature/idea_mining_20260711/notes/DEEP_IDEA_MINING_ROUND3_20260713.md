# Deep Idea Mining Round 3: Unified Binary TTX Attention (2026-07-13)

## 0. Scope, evidence bar, and current target

This is a read-only algorithm sidecar audit. It does not register a new H number, modify the
training code, or start an experiment. The deployment boundary is frozen as follows:

- DSEC only, with the existing `288x384` crop and standard `valid825` evaluation.
- All 12 encoder attention blocks use one formula. Per-block learned weights are allowed, but
  stage-dependent formulas, S2-only replacement, TX/SC mixing, and pure SC are excluded.
- All 105 neuron wrappers remain one-sided binary ATLIF, emitting `{0,+theta}`.
- The old native QKFormer carrier is absent. The preferred candidates also avoid dynamic
  `gate*K` and produce a displacement descriptor followed by a static projection.
- The encoder/decoder topology and the hardware schedule outside the attention block remain
  unchanged. Only a locally replaceable attention block is in scope.

The accuracy target is now stricter than the original H60 screen. The current software leader is
H67 dyadic epoch 19: AEE `1.4626`, AAE `9.3949`, total spikes `26.3948G`. H60 dyadic is AEE
`1.5016`, AAE `9.8431`, total spikes `23.2439G`; NB0 is AEE about `1.4872`, AAE about `9.93`,
total spikes about `44.05G`. A new accuracy candidate should beat H67 on standard `valid825`, not
merely pass the old 5% baseline window. A candidate that does not beat H67 can still be a hardware
Pareto point only if its complete attention PPA is materially lower.

This round intentionally avoids another 9/17/49-offset sweep. H73 DE9, H74 MC49, and H75 AX17
already isolate local dual evidence, a 49-offset descriptor, and an axial 17-offset descriptor.
The candidates below change the matching function, evidence rank, or cost-volume readout.

## 1. Common notation and frozen full30 protocol

The current window is `T=2`, `H=W=9`, `N=162`, with head dimension `D=32`. Let
`i=(t,y,x)`, `bar(t)=1-t`, and

```text
q_i, k_i in {0,1}^32
Omega9 = {(-1,-1),...,(-1,1),(0,-1),(0,0),(0,1),(1,-1),...,(1,1)}
k_bar(i,delta) = k[bar(t), y+dy_delta, x+dx_delta]

n11(q,k) = popcount(q & k)
n10(q,k) = popcount(q & ~k)
n01(q,k) = popcount(~q & k)
n00(q,k) = popcount(~q & ~k) over the valid 32 lanes
AX(q,k) = n11(q,k) + n00(q,k)/64
```

`Shiftmax_R` means the repository's dyadic normalization over candidate axis `R`. Every full30
candidate below uses the same protocol unless explicitly marked ineligible:

1. Warm-start independently from the same frozen H60 TTX epoch-2 checkpoint used by H67-H75.
2. DSEC crop `288x384`, batch 8, workers 8, AMP/cupy, 30 epochs, warmup 720, milestones
   20/25; evaluate epochs `0/4/9/14/19/24/28/29` on standard `valid825`.
3. A 120-step run is only an implementation/finite-gradient health check. It is not an accuracy
   kill gate and must not replace full30.
4. Audit ATLIF count `105`, replaced attention count `12`, candidate-local tensor shapes, boundary
   masking, and float/dyadic deployment. Warm-start must report only the explicitly new tensors as
   missing, with zero unexpected, zero non-candidate missing, and zero unresolved overlay-owned
   keys. A trained candidate checkpoint must restore with zero missing/unexpected keys.
5. Report AEE, AAE, PE1/PE2/outlier, total spikes, and an attention-inclusive operation/SRAM/
   control estimate. `total_spikes*pJ` is not a complete energy number for these candidates.
6. No candidate-size, group-count, patch-size, temperature, alpha, or score-weight sweep is allowed
   before the single pre-registered point completes full30.

## 2. Decision summary

| Priority | Candidate | New expression, not an offset sweep | H67-beating potential | Deployment regularity | Decision |
|---:|---|---|---|---|---|
| 1 | **PC9 patch-consistent Match-Code** | 3x3 corresponding-patch evidence for each of 9 displacements | High: adds local motion coherence and noise rejection | Fixed score planes, fixed dyadic spatial filter | First Round3 full30 |
| 2 | **LC4 learned-contingency Match-Code** | Learns asymmetric costs for `11/10/01/00` binary outcomes | High: one-sided events make `10` and `01` non-equivalent | Fixed popcount/subtract/shift-add | Second full30 |
| 3 | **G4 grouped multi-channel Match-Code** | Preserves four 8-lane matching hypotheses instead of one scalar | Medium-high: avoids scalar evidence bottleneck | Same 32 bit comparisons, four small Shiftmax rows | Third full30 |
| 4 | **BDAP9 displacement mixing** | Mixes displacement scores before nonlinear normalization | Medium: can suppress multimodal score maps | Fixed `9x9` low-bit matrix | Conditional after a plain Omega9 result |
| 5 | **SMM36 max/mean marginals** | Separates 2-D score volume into horizontal/vertical robust marginals | Medium; strong inductive bias but lower function rank | Same H74 score generator plus reductions | Conditional after H74 |
| 6 | **BI9 mutual temporal score** | Requires forward and reverse Q/K agreement per displacement | Uncertain: may reject ambiguous matches but hurt occlusion | Fixed two-pass score planes | Profile first, then at most one full30 |
| 7 | **CM8 coordinate-moment readout** | Replaces a wide codebook with motion moments/confidence | Low as an accuracy idea; useful compression control | Very small static projection | Do not run before a Match-Code winner |
| 8 | **TC3 temporal-group consensus** | Uses three event sub-interval feature banks | Potentially high in another architecture | Changes input/feature interface | Ineligible under the frozen boundary |

The top three are the only unconditional Round3 accuracy experiments. They are independent
alternatives, not components to stack. BDAP9 must not be combined with LC4 in its first run.

## 3. Candidate 1, P0: PC9 patch-consistent Match-Code

### 3.1 Full-paper and code evidence

[KPA-Flow, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Luo_Learning_Optical_Flow_With_Kernel_Patch_Attention_CVPR_2022_paper.pdf)
defines a local kernel operator in Eqs. 1-4: motion values are aggregated over a kernel region,
with context-derived normalized weights and a distance-dependent scale. The paper's ablation is
important: a `3x3` patch-kernel shape outperforms both `1x1` and `5x5`, and enabling the scale
function improves KITTI validation EPE from `4.58` to `4.46`. This is evidence for bounded local
motion coherence, not evidence for global attention.

The [official implementation at commit `98213dc`](https://github.com/megvii-research/KPAFlow/blob/98213dc77fa7ef7cf0e9507ebe03f902acc62d78/core/module.py#L63-L147)
uses `nn.Unfold` to form a three-patch-wide kernel, computes projected Q/K affinities at lines
128-138, applies the distance scale, normalizes over kernel candidates, and aggregates projected
motion features at lines 140-146. The full operator is too expensive and uses a dynamic value
carrier; it is not transplanted. The transferable mechanism is corresponding-patch consistency.

### 3.2 Migrated formula and tensor semantics

First compute nine cross-time base score planes, then apply the same fixed `3x3` dyadic spatial
filter to every plane:

```text
m_delta(i) = 64*n11(q_i,k_bar(i,delta)) + n00(q_i,k_bar(i,delta))

w(0,0)=4; w(axis-neighbor)=2; w(corner)=1
Z(i,delta) = sum_epsilon w(epsilon)*valid(i,delta,epsilon)
p_delta(i) = round[ sum_epsilon w(epsilon)*m_delta(i+epsilon) / Z(i,delta) ]

a_i = Shiftmax_Omega9(p_i + displacement_boundary_mask)
z_i = C[l,h] * a_i,                 C in Z8^(32x9)
Y_i = W_o[l] * concat_h(z_i)
```

`m` and `p` are `[Bwin,heads,162,9]`; `a` has the same shape and `z` is
`[Bwin,heads,162,32]`. The coordinate-only `Z` values are a tiny ROM; no data-dependent divide is
required. The implementation should generate the nine match planes once and spatially filter them,
not recompute 81 popcounts per query.

### 3.3 Uniformity and hardware cost

- All12 identity: same `Omega9`, patch kernel, boundary normalization, Shiftmax axis, and codebook
  shape in all 12 blocks. Stage resolution changes physical receptive field but not the formula.
- Arithmetic per token/head: nine 32-bit alpha-XNOR comparisons, then 81 small score additions/
  shifts across the nine planes, one Shiftmax9, and one static `32x9` INT8 projection.
- State/storage: opposite-time 3-row K halo for score generation, three rows of nine score planes,
  a coordinate-class normalization ROM, nine normalized scores, and the codebook. Adjacent queries
  reuse both Q/K words and score-plane rows.
- Control: fixed nested loops. Window edges select a ROM normalization and valid mask; there is no
  top-k, gather/scatter, learned offset, or variable candidate count.

### 3.4 Non-duplication and risk

- H67 adds a same-position K-XOR scalar but does not test whether a displacement remains consistent
  over neighboring pixels. PC9 is a patch match, not a motion-density bonus.
- H73 evaluates one Q/K pair at each of nine offsets and separates `n11/n00`; PC9 evaluates the
  spatial coherence of the combined alpha-XNOR score plane.
- H74 evaluates 49 point displacements; PC9 evaluates only nine displacements but each score is a
  corresponding `3x3` patch statistic. It is not a smaller H74 stencil.
- H75 keeps axial point matches; it has no patch consistency. H68 is a training-only matrix and
  H69-H71 change training/temperature/output context, not the matching support.

Accuracy risk: sparse event boundaries can be oversmoothed, and incorrect boundary normalization
can make edge tokens systematically weak. Hardware risk: nine concurrent score-plane line buffers
may dominate a very small H60 block unless streamed and banked carefully.

DATE novelty risk is **medium**. Patch matching and KPA are prior art. A defensible claim is the
specific binary alpha-XNOR score-plane reuse, dyadic spatial kernel, unified all12 descriptor, and
measured hardware implementation. It must be called KPA-inspired, not KPA reproduction.

### 3.5 Minimal falsification experiment

Run exactly one PC9 full30 point with the fixed `{4,2,1}` spatial kernel and `Omega9`. Initialize
the `9x32` codebook deterministically; do not sweep patch size or weights. Required diagnostics are
per-boundary-class AEE, score entropy before/after patch filtering, and score-plane SRAM traffic.
Kill PC9 if it does not beat H67 and does not improve flow-boundary/outlier metrics; do not enlarge
to `5x5`.

## 4. Candidate 2, P0: LC4 learned-contingency Match-Code

### 4.1 Full-paper and code evidence

[DICL, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/add5aebfcb33a2206b6497d53bc4f309-Abstract.html)
argues that a fixed dot product/cosine score discards useful high-dimensional evidence. Its Eq. 2
concatenates source and displaced target features, and Eq. 3 applies the same learned matching
network to every displacement. In its ablation, a learned DICL matcher improves over dot product,
cosine, and an MLP; the full spatial CNN is the strongest but is not hardware-compatible here.

The [official matcher](https://github.com/jytime/DICL-Flow/blob/0b34967ba4b7333bba37969ae152ac7b4eb6a0f1/models/DICL.py#L23-L38)
is shared over displacements. The [cost construction](https://github.com/jytime/DICL-Flow/blob/0b34967ba4b7333bba37969ae152ac7b4eb6a0f1/models/DICL.py#L332-L369)
concatenates source and shifted-target feature maps for every hypothesis and applies that shared
network independently. LC4 keeps this displacement-invariant learned-cost principle while replacing
the CNN with a complete binary contingency statistic.

### 4.2 Migrated formula and tensor semantics

For each `delta in Omega9`, retain all four binary outcomes:

```text
c_delta = [n11, n10, n01, n00]
r_delta = beta11*n11 + beta10*n10 + beta01*n01 + beta00*n00 + b
a_i = Shiftmax_Omega9(r_i + boundary_mask)
z_i = C[l,h] * a_i,                  C in Z8^(32x9)
Y_i = W_o[l] * concat_h(z_i)
```

`beta` is shared across all nine displacements within one head/block, preserving DICL's
displacement invariance. Training uses constrained signed dyadic weights; deployment stores a small
signed mantissa plus shift. Initialize `[beta11,beta10,beta01,beta00]=[1,0,0,1/64]`, so the initial
score is exactly the current alpha-XNOR score.

Only one AND-popcount per candidate is fundamentally required if `popcount(q)` is computed once
and `popcount(k)` is available with each K word:

```text
n10 = pop_q-n11
n01 = pop_k-n11
n00 = 32-pop_q-pop_k+n11
```

### 4.3 Uniformity, cost, and overlap

- All12 use the same four-outcome formula and nine offsets. Learned coefficients may differ by
  block/head, as ordinary projection weights already do.
- Compared with a plain AX Omega9 score, LC4 adds three subtract/add paths and four signed dyadic
  weighted terms per candidate. It adds four score coefficients plus bias and a `32x9` codebook per
  head/block; K buffering and candidate control remain fixed.
- `n10` and `n01` are directionally distinct under one-sided sparse spikes. H73 keeps only `n11`
  and `n00` as two independently normalized descriptors; it cannot express a direct penalty or
  reward for query-only versus key-only events before normalization.
- H67's K temporal XOR is independent of Q and has no displacement index. H69/H70 only change
  score temperature. H74/H75 use a fixed AX metric. H68 does not deploy a learned matcher.

Accuracy risk: `n10` and `n01` may be nearly collinear with per-token activity, making LC4 learn a
density prior rather than correspondence. Quantization risk arises if the trained coefficients are
not stable on a small dyadic grid.

DATE novelty risk is **medium-high**. Learned matching costs are established by DICL, and four-bin
binary contingency tables are elementary. Novelty depends on proving that asymmetric mismatch is
the missing event-flow signal and that the dyadic circuit is cheaper than recovering accuracy with a
larger offset volume. Do not claim a new general similarity metric from one DSEC result.

### 4.4 Minimal falsification experiment

Run one `Omega9` LC4 full30 point, with baseline-equivalent initialization and a pre-registered
dyadic coefficient grid. Log the four count distributions and learned coefficient trajectory in all
12 blocks. Expected warm-start missing keys are only the LC4 coefficients/biases and 12 codebooks;
all overlay-owned H60 weights must load. Kill the line if `beta10` and `beta01` converge to the same
quantized value and AEE does not beat H67; do not sweep alpha or offset radius.

## 5. Candidate 3, P0: G4 grouped multi-channel Match-Code

### 5.1 Full-paper and code evidence

[VCN, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/bbf94b34eb32268ada57a3be5062fe7d-Paper.pdf)
states that a scalar cost volume is an information bottleneck. Section 3.3 records `K` similarities
between jointly trained embeddings, forming a multi-channel volume
`R^(K x U x V x H x W)`, and regresses multiple hypotheses with uncertainty rather than collapsing
matching evidence immediately.

The [official correlation path](https://github.com/gengshan-y/VCN/blob/00c4befdbdf4e42050867996a6f686f52086e01a/models/VCN.py#L294-L313)
keeps the channel-wise feature product in a `B,C,U,V,H,W` tensor. The processed volume produces
16 cost channels per displacement and separate hypotheses before soft fusion in
[the official forward path](https://github.com/gengshan-y/VCN/blob/00c4befdbdf4e42050867996a6f686f52086e01a/models/VCN.py#L355-L380).
The volumetric U-Net and hypothesis selector are not transplanted; the transferable point is that
matching should retain multiple evidence subspaces.

### 5.2 Migrated formula

Split each 32-lane head into four fixed contiguous 8-lane groups. Q/K projections can learn how to
place evidence into the groups; no runtime router or permutation is added.

```text
s[g,delta] = 64*n11(q_i[g],k_bar(i,delta)[g]) + n00(q_i[g],k_bar(i,delta)[g])
a[g,:] = Shiftmax_Omega9(s[g,:] + boundary_mask),  g=0..3
d_i = concat_g a[g,:]                    # 36 lanes
z_i = C[l,h] * d_i,                      C in Z8^(32x36)
Y_i = W_o[l] * concat_h(z_i)
```

### 5.3 Uniformity and hardware cost

- Same `G=4`, 8-lane partition, `Omega9`, and output mapping in all12.
- The number of compared Q/K bits remains `9x32` per query/head, identical to a scalar Omega9
  matcher. The popcount produces 36 four-bit partial counts instead of nine six-bit counts.
- Four Shiftmax9 contexts are required. The descriptor/codebook grows from H73's 18 lanes to 36
  lanes, but no K value reread or dynamic gate multiplication is present.
- A byte- or nibble-sliced popcount datapath naturally exposes four groups. The main cost is four
  normalization states and a `36x32` static projection; groups can be processed serially if latency
  permits.

This is not H63's grouped scalar broadcast: H63 generated one scalar per group without
displacement semantics. It is not H73, which groups evidence type (`n11/n00`) after a full 32-lane
popcount. It is not H74/H75 because it changes channel rank, not search support. H67-H71 never
preserve multiple channel-subspace match distributions.

Accuracy risk: fixed channel groups can be redundant, and four independent normalizations discard
relative mass across groups. Hardware risk: the larger static projection can outweigh savings from
eliminating dynamic K scaling.

DATE novelty risk is **medium-high** because groupwise correlation is established. The useful claim
would be a measured co-design result: byte-sliced binary evidence avoids scalar collapse at fixed
comparison count and maps efficiently to the attention engine. It is not a standalone algorithmic
novelty claim without the hardware evidence.

### 5.4 Minimal falsification experiment

Pre-register `G=4` only, because four 8-bit groups match the 32-lane datapath. Run one full30 from
H60 epoch2 and audit a new `36x32` codebook in each block/head. Report per-group entropy and
codebook column norms. Kill grouped matching if two or more groups collapse to statistically
indistinguishable distributions and AEE does not beat H67; do not sweep `G=2/8`.

## 6. Candidate 4, P1: BDAP9 pre-normalization displacement mixing

### 6.1 Evidence and migrated formula

DICL Eq. 4 introduces displacement-aware projection (DAP): every displacement cost is a learned
linear combination of all raw displacement costs before soft-argmin. The
[official implementation](https://github.com/jytime/DICL-Flow/blob/0b34967ba4b7333bba37969ae152ac7b4eb6a0f1/models/DICL.py#L95-L116)
is a `1x1` convolution over flattened displacement channels. The paper reports a consistent roughly
`0.1`-pixel EPE improvement and attributes it to suppressing multimodal cost distributions.

The binary adaptation is deliberately small:

```text
s_delta = 64*n11(q_i,k_bar(i,delta)) + n00(q_i,k_bar(i,delta)), delta in Omega9
s_tilde = W_dap[l,h] * s + b_dap[l,h], W_dap in Z4^(9x9)
a = Shiftmax_Omega9(s_tilde + boundary_mask)
z = C[l,h] * a,                         C in Z8^(32x9)
```

Initialize `W_dap` to identity and `b_dap=0`. Training uses signed low-bit/dyadic constrained
weights so the deployed block is a fixed score-mixing matrix, not a floating convolution.

### 6.2 Cost, overlap, and risk

- All12 identity is exact. Every block uses one fixed nine-score mixer, Shiftmax9, and codebook.
- Added arithmetic is 81 low-bit score products/adds per token/head. With 4-bit signed dyadic
  weights, products become select/shift/add; storage is 324 bits per head/block plus biases.
- H74's codebook acts after nonlinear normalization and maps probabilities to feature lanes. BDAP9
  acts before Shiftmax and changes competition between displacement hypotheses, so it cannot in
  general be folded into the H74/H73 codebook.
- H68's matrix is a training-only token-token teacher; BDAP9 is a deployed displacement-domain
  transform. H69/H70 scale scores but cannot mix neighboring or competing modes.

Accuracy risk: a full `9x9` matrix may memorize window-relative displacement and overfit DSEC.
Hardware risk is the added low-bit matrix engine, although its dimensions are fixed and small.

DATE novelty risk is **high** as a standalone idea because this is a direct low-bit DAP adaptation.
It is useful as a mechanism ablation if it beats plain Omega9 and H67, but should not be the paper's
headline without a stronger binary/hardware contribution.

### 6.3 Minimal falsification experiment

Run only after one plain Omega9 Match-Code result exists. Use plain AX scores, identity-initialized
`W_dap`, and no LC4/PC9 combination. Run one full30 and compare score peak count/margin against
the plain matcher. Kill if the quantized matrix remains within a small norm of identity or if
multimodality falls without AEE/AAE improvement.

## 7. Candidate 5, P1: SMM36 separable max/mean marginal Match-Code

### 7.1 Full-paper and code evidence

[Separable Flow, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Separable_Flow_Learning_Motion_Cost_Volumes_for_Optical_Flow_Estimation_ICCV_2021_paper.pdf)
defines horizontal mean and max marginals in Eqs. 2-3, then learns additional adaptive marginal
channels. Its ablation shows both mean and max alone improve the RAFT baseline, while combining all
separation channels is strongest. Eqs. 8-10 show that coordinate expectation from a 2-D probability
map is naturally separable into horizontal and vertical marginals.

The [official `CorrBlock.separate`](https://github.com/feihuzhang/SeparableFlow/blob/f04796f64d8c9b1edf839b2cf33c5151457c67ef/core/corr.py#L35-L74)
computes max and mean along each spatial axis at every correlation-pyramid level and concatenates
the results. Its non-local filter, 3-D convolutions, and iterative RAFT refinement are not copied.

### 7.2 Migrated formula

Use the exact H74 `Omega49` score generator, but replace the unconstrained 49-lane readout by robust
horizontal and vertical marginals. Let `V_x(dx)` and `V_y(dy)` be the valid selected offsets in each
column/row of `Omega49`:

```text
s(dy,dx) = 64*n11(q_i,k_bar(i,(dy,dx))) + n00(q_i,k_bar(i,(dy,dx)))

h_mean(dx) = mean_{dy in V_x(dx)} s(dy,dx)
h_max(dx)  = max_{dy in V_x(dx)} s(dy,dx)
v_mean(dy) = mean_{dx in V_y(dy)} s(dy,dx)
v_max(dy)  = max_{dx in V_y(dy)} s(dy,dx)

d = concat(Shiftmax9(h_mean), Shiftmax9(h_max),
           Shiftmax9(v_mean), Shiftmax9(v_max))       # 36 lanes
z = C[l,h] * d,                                       C in Z8^(32x36)
```

The varying valid counts are fixed by offset index and boundary class; reciprocal constants are ROM
entries. No missing location is clamped or included in the mean.

### 7.3 Uniformity, cost, and overlap

- All12 use the same `Omega49`, four marginals, four Shiftmax9 units/contexts, and `36x32`
  projection.
- Score cost equals H74: 49 alpha-XNOR comparisons. Added work is four streamed max/mean
  reductions; codebook work drops from 49 to 36 input lanes.
- H75 only evaluates the horizontal/vertical axes. SMM36 includes off-axis/diagonal evidence and
  marginalizes it, so a diagonal match contributes to both motion coordinates.
- H74 preserves arbitrary 49-lane displacement identity; SMM36 imposes separable structure and is
  lower rank. It can improve generalization but cannot represent every H74 readout.

Accuracy risk is the loss of x-y coupling, particularly for diagonal motion and repeated patterns.
Hardware risk is four normalization contexts and max/mean state, though all access patterns are
static and the 49 scores can be consumed as a stream without materializing a dense `9x9` grid.

DATE novelty risk is **high** standalone: max/mean cost separation is prior art. Its role is a clean
hardware/accuracy ablation against H74, not a headline contribution unless the binary streaming
marginal engine creates a measured PPA advantage while also beating H67.

### 7.4 Minimal falsification experiment

Run only after H74 MC49 has a standard full30 result. Reuse exactly `Omega49`; do not change the
search set. Run one full30 from H60 epoch2 with the four fixed marginal types. Compare H74 and
SMM36 on AEE/AAE, diagonal-motion bins, codebook SRAM, and cycles. Kill if H74 is already below
H60 and score-map analysis does not show multimodality; do not add adaptive marginal attention.

## 8. Candidate 6, P2: BI9 mutual temporal score

### 8.1 Full-paper and code evidence

[BAT, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/38100/42062)
splits both event streams into temporal groups. Its Eqs. 3-4 compute forward correlations from the
reference group to future groups and backward correlations from the reference group to past groups;
Table 3 reports that bidirectional temporal correlation improves its baseline before adaptive
sampling or SATMA. BAT then uses flow-guided bilinear sampling and deformable value attention,
which are outside this project's attention-block boundary.

The advertised [official repository at commit `f8f15bd`](https://github.com/gangweiX/BAT/blob/f8f15bd3cc910ac58e6cacdefeb818467dbc5cbd/README.md)
currently contains only `README.md` with “Code will be available soon.” Therefore the paper
equations are auditable, but no executable BAT implementation was available for code-level
verification on 2026-07-13.

### 8.2 Hardware-bounded adaptation

Literal BAT requires extra event groups and warping. BI9 instead tests only whether reciprocal Q/K
agreement improves a two-time binary matcher:

```text
s_f(i,delta) = AX(q[t,p], k[bar(t),p+delta])
s_r(i,delta) = AX(q[bar(t),p+delta], k[t,p])
s_bi(i,delta) = min(s_f(i,delta), s_r(i,delta))
a_i = Shiftmax_Omega9(s_bi + boundary_mask)
z_i = C[l,h] * a_i,                    C in Z8^(32x9)
```

`min` is a comparator, and the reverse score for `(i,delta)` is paired with the opposite-time query
at `p+delta`. This is BAT-inspired reciprocal consistency, not a reproduction of BAT's temporal
group correlations.

### 8.3 Cost, overlap, and risk

- The formula is identical in all12 and uses fixed `Omega9`; there is no SC, native carrier, warp,
  deformable offset, or dynamic K value path.
- It needs two 32-bit match evaluations per candidate, or storage/reuse of forward and reverse score
  planes, plus one comparator. Q and K for both time slices must be co-resident.
- H67 reads opposite-time K only to form a same-position XOR prior. BI9 compares displaced Q/K in
  both directions and suppresses a hypothesis unless both projected directions agree.
- H73-H75 are one-direction displacement descriptors. H68-H71 do not couple reciprocal scores.

Accuracy risk is substantial: optical flow occlusion is inherently asymmetric, and the minimum may
discard exactly the one-sided evidence needed at boundaries. Hardware cost is almost twice the
matching work unless reverse planes are reused across queries.

DATE novelty risk is **high**. Forward-backward consistency is established, and this adaptation is
not BAT's mechanism. It can be a useful negative/positive ablation but must not be branded as BAT.

### 8.4 Minimal falsification experiment

Before training, run a frozen H73/H74 score trace to measure the fraction of top-1 forward matches
rejected by reciprocal consistency and its correlation with valid flow boundaries. Only if rejection
is selective rather than near-global should one BI9 full30 run be authorized. Use `min`, not a
mean/min sweep. Kill if boundary AEE worsens or attention matching energy exceeds two times PC9
without beating H67.

## 9. Candidate 7, P3 control: CM8 coordinate-moment readout

### 9.1 Full-paper and code evidence

[GMFlow, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)
defines correlation `C=F1*F2^T/sqrt(D)`, matching distribution `M=softmax(C)`, expected target
coordinates `G_hat=M*G`, and flow `V=G_hat-G` in Eqs. 1-4. The
[official global implementation](https://github.com/haofeixu/gmflow/blob/b5123431164d01ec14526a1c3d22218aecb62024/gmflow/matching.py#L7-L36)
multiplies the normalized correspondence matrix by a fixed coordinate grid. Its
[local implementation](https://github.com/haofeixu/gmflow/blob/b5123431164d01ec14526a1c3d22218aecb62024/gmflow/matching.py#L39-L83)
does the same over a bounded window.

The corresponding Match-Code compression control is:

```text
a = Shiftmax_Omega49(s)
r = [sum a*dx, sum a*dy,
     sum a*dx^2, sum a*dy^2, sum a*dx*dy,
     sum a*abs(dx), sum a*abs(dy), max(a)]
z = W_m[l,h] * r,                       W_m in Z8^(32x8)
```

The first seven terms use fixed small coordinate constants. `max(a)` is a confidence term. This
retains one output per token and eliminates the wide `49x32` H74 codebook.

### 9.2 Why it is not an accuracy priority

All12 uniformity and control are excellent: fixed coordinate ROM, eight accumulators, one max, and
an `8x32` projection. However, the first seven moment terms are linear combinations of the same 49
probabilities and are therefore representable by H74's unconstrained codebook. CM8 is a structured
low-rank restriction, not a more expressive attention function. It may regularize or compress, but
there is weak reason to expect it to beat H67 before a richer Match-Code model succeeds.

It is distinct from H75 because it observes all H74 off-axis scores before compression, but it is
partly functionally contained in H74. It does not overlap H67-H71's scalar temperature/context
changes.

DATE novelty risk is **very high** as an algorithm claim because coordinate expectation is GMFlow's
core operation and moment compression is straightforward. It is only useful as a hardware readout
ablation.

### 9.3 Minimal falsification experiment

Do not run CM8 before H74 or another 49-score candidate beats H67. If codebook SRAM/energy then
dominates, run one full30 using the fixed eight moments and compare against the winning checkpoint's
score support. Promotion requires AEE within `0.02` of the winner and a synthesized readout-area/
energy reduction; it need not beat the winner.

## 10. Candidate 8, rejected under current boundary: TC3 temporal-group consensus

### 10.1 Full-paper and code evidence

[TMA, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_TMA_Temporal_Motion_Aggregation_for_Event-based_Optical_Flow_ICCV_2023_paper.pdf)
splits an event interval into `g` segments, computes `C_i=F_0*F_i^T/sqrt(D)` for every segment in
Eq. 4, linearly scales lookup coordinates by `i/g` in Eqs. 5-6, and cross-attends each intermediate
motion feature to the final one in Eqs. 7-8.

The official code confirms that this is not an attention-only edit. `TMA.__init__` fixes five splits;
[`TMA.forward`](https://github.com/ispc-lab/TMA/blob/905e47ee219b65d639da7d6d6f06ecdae8618394/model/TMA.py#L43-L87)
chunks each 15-bin voxel tensor into five inputs, runs a shared feature extractor, builds five full
correlation volumes, performs flow-dependent lookup, and aggregates motion features. The
[official MPA](https://github.com/ispc-lab/TMA/blob/905e47ee219b65d639da7d6d6f06ecdae8618394/model/aggregate.py#L28-L108)
uses full projected QK softmax and dynamic V aggregation.

### 10.2 Possible formula and rejection

A nominal three-group binary adaptation would be

```text
s[j,delta] = AX(q_ref(i), k_group[j](i+delta)), j=1..3, delta in Omega9
d = concat_j Shiftmax_Omega9(s[j,:])        # 27 lanes
z = C[l,h] * d
```

It would remain all12-uniform, but it requires three separately encoded temporal feature banks in
every block. The current TTX representation has only `T=2`; manufacturing `k_group[1..3]` requires
changing voxel partitioning, feature extraction, checkpoint semantics, SRAM allocation, and the
encoder-to-attention interface. This is not an attention-block-local replacement.

Additional cost would be three K banks, three times the score comparisons, three Shiftmax states,
and a `27x32` projection. Literal TMA also adds flow state, bilinear lookup, MPA, and dynamic V.
It is not H67's cheap temporal XOR and not H73-H75's two-time descriptor, but its non-overlap comes
from violating the frozen architecture rather than from a cleaner attention expression.

DATE novelty risk is **high and strategically poor**: TMA already establishes temporal splitting,
and a binary port would force the hardware team to redesign more than the attention block.

No full30 should be generated under the current boundary. The minimal falsification is architectural:
if three temporal K banks cannot be produced from the existing Q/K projection interface without
changing the backbone checkpoint and SRAM contract, the candidate is rejected. That condition is
already met.

## 11. Experiment order and kill rules

The minimal Round3 addition is three independent full30 runs, not an eight-way queue:

1. PC9 patch-consistent Match-Code.
2. LC4 learned-contingency Match-Code.
3. G4 grouped multi-channel Match-Code.

BDAP9 is authorized only after a plain Omega9 score distribution demonstrates persistent
multimodality. SMM36 is authorized only after H74 provides a 49-score baseline. BI9 requires a
frozen score-trace screen. CM8 is a post-winner hardware compression control. TC3 is rejected.

Global kill rules:

1. Do not combine PC9, LC4, G4, BDAP9, or Motion-XOR in a first run. Each tests a different missing
   variable; stacking destroys the DATE ablation.
2. If none of PC9/LC4/G4 beats H67 dyadic AEE `1.4626`, retain H67/H68 as the software mainline
   and move attention effort to exact hardware optimization. Passing the old NB0 5% window is not
   enough to claim an algorithm improvement.
3. If a direct Match-Code candidate beats H67 but has more spikes, it remains eligible because
   neuron spikes do not include its attention cost. Promotion requires complete attention SRAM,
   cycles, and synthesized energy against H67, not spike proxy alone.
4. Only the best two standard `valid825` candidates receive independent-seed replication. Do not
   spend seeds on all proposals.
5. Any trained run with attention count other than 12, ATLIF count other than 105, unresolved
   overlay keys, non-candidate missing keys, or boundary wrap is invalid regardless of AEE.

## 12. Source and implementation ledger

All papers were read beyond the abstract, including the cited equations and ablations. Official code
was inspected at the pinned commit when available:

| Work | Paper | Inspected official source |
|---|---|---|
| DICL, NeurIPS 2020 | [paper/abstract page](https://proceedings.neurips.cc/paper_files/paper/2020/hash/add5aebfcb33a2206b6497d53bc4f309-Abstract.html) | [`models/DICL.py`, commit `0b34967`](https://github.com/jytime/DICL-Flow/blob/0b34967ba4b7333bba37969ae152ac7b4eb6a0f1/models/DICL.py) |
| VCN, NeurIPS 2019 | [full paper](https://proceedings.neurips.cc/paper/2019/file/bbf94b34eb32268ada57a3be5062fe7d-Paper.pdf) | [`models/VCN.py`, commit `00c4bef`](https://github.com/gengshan-y/VCN/blob/00c4befdbdf4e42050867996a6f686f52086e01a/models/VCN.py) |
| KPA-Flow, CVPR 2022 | [full paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Luo_Learning_Optical_Flow_With_Kernel_Patch_Attention_CVPR_2022_paper.pdf) | [`core/module.py`, commit `98213dc`](https://github.com/megvii-research/KPAFlow/blob/98213dc77fa7ef7cf0e9507ebe03f902acc62d78/core/module.py) |
| Separable Flow, ICCV 2021 | [full paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Separable_Flow_Learning_Motion_Cost_Volumes_for_Optical_Flow_Estimation_ICCV_2021_paper.pdf) | [`core/corr.py`, commit `f04796f`](https://github.com/feihuzhang/SeparableFlow/blob/f04796f64d8c9b1edf839b2cf33c5151457c67ef/core/corr.py) |
| GMFlow, CVPR 2022 | [full paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf) | [`gmflow/matching.py`, commit `b512343`](https://github.com/haofeixu/gmflow/blob/b5123431164d01ec14526a1c3d22218aecb62024/gmflow/matching.py) |
| TMA, ICCV 2023 | [full paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_TMA_Temporal_Motion_Aggregation_for_Event-based_Optical_Flow_ICCV_2023_paper.pdf) | [`model/TMA.py`, commit `905e47e`](https://github.com/ispc-lab/TMA/blob/905e47ee219b65d639da7d6d6f06ecdae8618394/model/TMA.py) and [`model/aggregate.py`](https://github.com/ispc-lab/TMA/blob/905e47ee219b65d639da7d6d6f06ecdae8618394/model/aggregate.py) |
| BAT, AAAI 2026 | [full paper](https://ojs.aaai.org/index.php/AAAI/article/download/38100/42062) | [official repository, commit `f8f15bd`; README-only on audit date](https://github.com/gangweiX/BAT/tree/f8f15bd3cc910ac58e6cacdefeb818467dbc5cbd) |

## 13. Final recommendation

The strongest new accuracy line is **PC9**, because it changes the semantics from point matching to
corresponding-patch matching while retaining a fixed nine-offset engine and enabling score-plane
reuse. **LC4** is the cleanest one-sided-event-specific metric experiment: it tests whether query-only
and key-only events should have different costs. **G4** is the cleanest rank experiment: it preserves
four matching subspaces without increasing compared Q/K bits.

These three are more likely to exceed H67 than CM8/SMM36-style compression, and they remain
local attention-block replacements. None should be combined until independent full30 and standard
deployment audits identify a winner.
