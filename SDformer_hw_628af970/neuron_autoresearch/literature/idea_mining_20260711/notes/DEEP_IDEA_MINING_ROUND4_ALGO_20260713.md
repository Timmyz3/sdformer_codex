# Deep Idea Mining Round 4: All12 Binary Match Attention (2026-07-13)

## 0. Scope and decision

This is an algorithm-only literature and source-code audit. It does not reserve H numbers, change
training code, modify the DATE portfolio, or start an experiment. The deployment contract remains:

- DSEC only, standard `288x384` crop and `valid825` evaluation.
- All 12 encoder attention blocks use one formula. No stage-specific replacement, TX/SC mixture,
  pure SC, or native QKFormer carrier is allowed. The global contract still permits H60 `gate*K`;
  this Round4 deliberately uses static codebook output to isolate local assignment semantics.
- All 105 neuron wrappers remain one-sided binary ATLIF with `{0,+theta}` output.
- The encoder/decoder and the hardware schedule outside the attention block remain unchanged.
- A deployed attention output must come from binary matching, Shiftmax/integer normalization, and
  a static codebook. Training-only heads are allowed only when they are removed and audited.

The current target to beat is H67 dyadic epoch 19: AEE `1.4626`, AAE `9.3949`, total spikes
`26.3948G`. H67/H68 have completed full30 and standard evaluation. H76-H78 have generated configs,
unit/load-chain audits, and a serial watcher, but the watcher is still waiting for the H73-H75
completion marker. Therefore this round proposes four **provisional** candidates, `R4-A` through
`R4-D`; it does not allocate `H79+` and does not enter the existing queue.

### Round4 recommendation

| Priority | Provisional candidate | Missing capability tested | Deployed change | Decision |
|---:|---|---|---|---|
| 1 | **R4-A CF10 conflict-free null Match-Code** | Explicitly represent unmatchable/occluded event tokens | One extra score and a fixed-zero null code | Best first full30 |
| 2 | **R4-B DN9 destination-normalized Match-Code** | Test whether target-side competition resolves ambiguous local matches | Incoming-edge Shiftmax and gate product | Mechanistic counterpoint to R4-A |
| 3 | **R4-D AMM9 multimodal offset supervision** | Do not force one displacement target at motion boundaries | Training-only target construction | Low deployment risk, support contribution |
| 4 | **R4-C BSMR9 masked score reconstruction** | Force local score representation to infer missing correspondence evidence | Training-only mask/reconstruction branch | Higher training complexity; run last |

R4-A and R4-B must not be combined. Their scientific value is the contradiction: RCM argues that
dense matching should preserve independent many-to-one assignments, while LightGlue/Efficient
LoFTR use source-and-target competition. DSEC determines which assumption is correct here.

## 1. Frozen notation and common deployment base

For every block, the window has `T=2`, `H=W=9`, `N=162`, and head dimension `D=32`. Let
`i=(t,y,x)`, `bar(t)=1-t`, and let `Omega9` be the fixed row-major `3x3` displacement set. A valid
edge connects query `i` to `j(i,delta)=(bar(t),y+dy_delta,x+dx_delta)`.

```text
q_i, k_j in {0,1}^32
n11(i,delta) = popcount(q_i & k_j)
n00(i,delta) = popcount(~q_i & ~k_j)
s(i,delta)   = (n11(i,delta) + n00(i,delta)/64) / 32
```

Out-of-window edges are masked before normalization. `SM_R` denotes the existing hardware-bounded
Shiftmax over axis/set `R`. A plain carrier-free Match-Code block is

```text
p(i,:) = SM_Omega9(s(i,:))
Y_i[h,d] = sum_delta p(i,delta) * C[h,delta,d],  C in Q1.7^(9x32)
```

`C` is static after training; no `K` or `V` tensor is read by the output path. All four candidates
start from this score semantics. They are changes to assignment/readout or training, not another
`9/17/49` support sweep.

## 2. Source audit ledger

All repository commits below were cloned and read on 2026-07-13. Line links identify the audited
implementation rather than a README claim.

| Work | Full-paper evidence | Official implementation snapshot | Audited mechanism |
|---|---|---|---|
| RCM, ECCV 2024 | [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05862.pdf), Eqs. 4-5 | No official repository is linked by the paper or arXiv metadata, and none was found in the audit | Learnable dustbin plus row-only softmax for independent many-to-one assignment |
| LightGlue, ICCV 2023 | [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.pdf), Eqs. 6-8, 11 | [`cvg/LightGlue@eb42fee`](https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/lightglue/lightglue.py#L265-L299) | Learned matchability, double softmax, and explicit unmatched rows/columns |
| Efficient LoFTR, CVPR 2024 | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Efficient_LoFTR_Semi-Dense_Local_Feature_Matching_with_Sparse-Like_Speed_CVPR_2024_paper.pdf) | [`zju3dv/efficientloftr@ffd4a46`](https://github.com/zju3dv/efficientloftr/blob/ffd4a4644064354468eb1f0c7a3e732233cb732f/src/loftr/utils/coarse_matching.py#L94-L168) | Product of source/target softmaxes and mutual match extraction; fine path uses a `3x3` correlation heatmap |
| FlowFormer++, CVPR 2023 | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Shi_FlowFormer_Masked_Cost_Volume_Autoencoding_for_Pretraining_Optical_Flow_Estimation_CVPR_2023_paper.pdf), Eqs. 1-3 | [`FlowFormerPlusPlus@c33de90` mask loading](https://github.com/XiaoyuShi97/FlowFormerPlusPlus/blob/c33de90f35af3fac1a55de6eac58036dd8ffb3b3/core/pretrain_maemask_datasets.py#L89-L95), [`decoder.py`](https://github.com/XiaoyuShi97/FlowFormerPlusPlus/blob/c33de90f35af3fac1a55de6eac58036dd8ffb3b3/core/FlowFormer/PerCostFormer3/decoder.py#L346-L439) | Block-shared cost masking, normalized larger-patch reconstruction, training-only head |
| ADL, CVPR 2024 | [paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Adaptive_Multi-Modal_Cross-Entropy_Loss_for_Stereo_Matching_CVPR_2024_paper.pdf), Eqs. 3-5 | [`xxxupeng/ADL@46811d1` target/loss](https://github.com/xxxupeng/ADL/blob/46811d14c42b0ef8ae9ecc646d7b1ef2e6209b4c/losses/gt_distribution.py#L10-L77), [`DME`](https://github.com/xxxupeng/ADL/blob/46811d14c42b0ef8ae9ecc646d7b1ef2e6209b4c/disparity_estimators/disparity_estimator.py#L65-L122) | Local clustering, mixture-of-Laplacians target, dominant-modal inference |

Two recent event-flow papers were also checked and excluded. EDCFlow (CVPR 2025) uses five event
windows, flow-conditioned bilinear warping, depthwise 3-D convolutions, adaptive feature fusion,
and a GRU; its Eqs. 4-6 do not fit a local attention-only replacement. EMatch (ICCV 2025) uses TRN
and spatial contextual attention and was already audited in Round2/Match-Code work. Neither is a
new all12 candidate under the frozen boundary.

## 3. R4-A, P0: CF10 conflict-free null Match-Code

### 3.1 Original mechanism and transferable evidence

RCM appends a learnable dustbin feature to the dense target set. Its Eq. 4 computes the score
matrix including the dustbin, and Eq. 5 applies **row softmax only**. Each source feature therefore
chooses one target or the dustbin independently; multiple sources may select the same target. The
paper explicitly contrasts this with dual softmax, whose target-side constraint discards valid
many-to-one correspondences under scale changes.

LightGlue provides executable evidence that matchability is not redundant with similarity. Its
paper predicts `sigma_i=sigmoid(Linear(x_i))` and multiplies two matchability terms with row/column
softmaxes in Eq. 8. The official code computes two log-softmax terms and explicit unmatched
entries at `lightglue.py:265-277`; the paper's ablation reports that removing matchability damages
the ability to separate good and bad matches. We transfer the explicit null hypothesis, not
LightGlue's dynamic pruning or global assignment matrix.

### 3.2 Migrated formula

For each query, obtain the largest and second-largest valid local scores:

```text
s1_i = max_delta s(i,delta)
s2_i = secondmax_delta s(i,delta)
rho_i = popcount(q_i) / 32

e_i = s1_i - 1
      + Q_1/64(beta_m) * (s1_i - s2_i)
      + Q_1/64(beta_q) * (rho_i - 1/2)

p_i = SM_10([s(i,delta_1), ..., s(i,delta_9), e_i])
Y_i[h,d] = sum_delta p_i[delta] * C[h,delta,d]
C[h,empty,d] = 0 exactly
```

`beta_m` and `beta_q` are per-head signed dyadic coefficients, clipped to `[-1,1]`, initialized to
zero. The fixed `s1-1` initialization makes the null candidate subordinate without removing it;
there is no null/codebook parameter sweep. The null row is hard zero rather than learned, so a
high null probability suppresses unreliable token output instead of introducing a new carrier.

Tensor shapes are `s:[Bwin,Hd,162,9]`, `e:[Bwin,Hd,162,1]`, `p:[...,10]`, and
`C:[Hd,10,32]` with the last row constant zero. The same formula and two coefficients are used in
all 12 blocks.

### 3.3 Hardware and state

- Existing work: nine alpha-XNOR scores and one Shiftmax row.
- Added arithmetic: top-2 over nine values, one subtract for margin, one query popcount already
  available from contingency accounting, two signed dyadic multiply/add terms, and Shiftmax10
  instead of Shiftmax9. General multiplication is not required.
- Added state: two low-bit coefficients per head per block. The codebook stores nine effective
  rows because the null row is wired to zero.
- Added control: a fixed tenth candidate and top-2 compare network. No destination scatter,
  token-token SRAM, dynamic route, K carrier, or stage-dependent mode exists.
- Primary risk: DSEC valid supervision may not label all occlusions as unmatched; the null can
  become either permanently inactive or an easy output-suppression path. Log null occupancy by
  stage, event density, and flow-boundary distance.

### 3.4 Difference from H67-H78

- H67 adds a same-position temporal XOR prior but still has no unmatchable state.
- H68 is a training-only rich teacher; CF10 changes deployed assignment semantics.
- H69-H71 modify temperature/density/window context, not the candidate set.
- H73-H78 always distribute mass among valid displacement hypotheses. CF10 is the first candidate
  that may explicitly select none, and it does not change offset support, patch kernel,
  contingency score, or channel grouping.
- It is not Round3 BI9: BI9 recomputes reciprocal Q/K evidence and applies a minimum; CF10 adds no
  reverse matching pass.

DATE novelty risk is **medium**. Dustbins and matchability are established. The defensible claim
would be a carrier-free binary local null assignment co-designed with fixed-zero hardware readout,
not invention of unmatched matching.

### 3.5 Single full30 falsification protocol

Run one config from the frozen H60 TTX epoch-2 checkpoint. Keep the common batch 8, workers 8,
AMP/cupy, 30 epochs, warmup 720, milestones 20/25, saved/evaluated epochs
`0/4/9/14/19/24/28/29`, and standard `valid825`. No coefficient, null bias, or support sweep.

Required audits are ATLIF `105`, attention `12`, CF10 modules `12`, overlay keys `210`, zero
unexpected/non-candidate missing, fixed-zero null code, and strict same-mode checkpoint reload.
The candidate is falsified if best valid825 AEE does not beat `1.4626`, if null occupancy collapses
below `0.5%` or above `50%` in at least 10 of 12 blocks, or if total spikes exceeds H67 without an
AEE/AAE improvement. A positive result must also improve low-event and flow-boundary AEE; a global
gain caused only by output suppression is insufficient.

## 4. R4-B, P1: DN9 destination-normalized Match-Code

### 4.1 Original mechanism and code evidence

LightGlue Eq. 8 forms partial assignment with the product of a source-normalized and a
target-normalized similarity. Its official `sigmoid_log_double_softmax` implements this as two
log-softmaxes before adding matchability. Efficient LoFTR's official coarse matcher independently
confirms the operator: `softmax(sim,1) * softmax(sim,2)` at
`coarse_matching.py:119-123`, followed by row/column mutual maxima at lines 165-168. The
transferable mechanism is destination competition, not global attention, adaptive pruning, or
mutual hard selection.

### 4.2 Migrated local-edge formula

Each score belongs to a directed local edge `e=(i,delta)` ending at `j=j(i,delta)`. Define the
valid incoming edge set

```text
E_in(j) = {(i',delta') | j(i',delta') = j, delta' in Omega9, valid(i',delta')}

r(i,delta) = SM_{delta in Omega9}(s(i,delta))
c(i,delta) = SM_{e' in E_in(j(i,delta))}(s(e')) evaluated at edge (i,delta)
a(i,delta) = Q_1.7(r(i,delta) * c(i,delta))
Y_i[h,d] = sum_delta a(i,delta) * C[h,delta,d]
```

There is deliberately no final row renormalization. Thus `sum_delta a(i,delta) <= 1` is a local
matching-confidence signal, as in a partial assignment, and low-confidence tokens produce lower
output magnitude. The product is the literal source mechanism. A hardware implementation may
replace the Q1.7 multiplier with addition of Shiftmax exponent classes only if bit-exact tests show
the same `a`; that is an implementation optimization, not a different algorithm.

The destination normalization is local. For a `9x9` spatial plane, each destination has at most
nine incoming edges from the opposite time; it never constructs an `N x N` matrix. Boundary edge
sets use exact validity counts.

### 4.3 Hardware and state

- Added arithmetic: a second Shiftmax of at most nine elements per destination and nine Q1.7 gate
  products per source token.
- Added storage: one score/gate plane of `2*9*9*9` entries per head if source and destination
  passes are decoupled. A streamed scatter-reduce implementation can use nine banks and retain only
  the current destination stripe, but that claim requires an RTL schedule and bank-conflict trace.
- Added control: fixed coordinate scatter from each `(source,delta)` to one destination, boundary
  masks, and a second normalization phase. No learned runtime state is added beyond the `9x32`
  static codebook.
- Primary risk: optical flow admits occlusion and many-to-one motion. Destination competition may
  reject valid edges, exactly the failure RCM identifies. It may also increase attention latency
  enough to erase spike-energy gains.

### 4.4 Non-duplication

DN9 does not alter H67 temporal evidence, H69/H70 temperature, H71 broadcast context, H76 spatial
patch filtering, H77 contingency coefficients, or H78 channel grouping. It differs from BI9 more
substantially than the names suggest: BI9 computes a second reverse Q/K score for the same pair and
takes `min`; DN9 computes each edge score once, then compares it against **other source edges that
terminate at the same destination**. Reciprocal consistency and assignment competition are not the
same operator.

DATE novelty risk is **medium-high** because dual normalization is standard. Its role is a clean
mechanistic ablation against conflict-free row-only CF10. It becomes a contribution only if the
bounded local graph and bit-exact two-pass hardware show a measured accuracy/PPA advantage.

### 4.5 Single full30 falsification protocol

Use the frozen common full30 protocol and the same plain Omega9 alpha-XNOR score/codebook as R4-A,
but no null candidate. No product-vs-log, renormalization, or hard-mutual sweep is allowed. Before
training, test edge-index bijection, exact boundary incoming sets, and equality to a dense reference
on random `2x9x9x32` binary tensors.

Falsify DN9 if best AEE fails to beat H67, boundary AEE or valid-edge recall worsens by more than
`2%` relative to the plain Omega9 control, or attention-inclusive energy/latency is more than
`1.5x` CF10 without at least `0.01` lower AEE. Record row mass, destination collision count, and
occlusion/boundary bins; overall AEE alone cannot determine why it passed or failed.

## 5. R4-C, P2: BSMR9 block-shared masked score reconstruction

### 5.1 Original mechanism and code evidence

FlowFormer++ observes that neighboring source pixels have highly correlated cost maps, so random
independent masking leaks the answer. Its block-sharing mask gives neighboring source pixels the
same missing target region. The paper's Eq. 1 masks features during cost tokenization; Eqs. 2-3
decode a normalized larger cost patch from a smaller query patch with MSE. The reconstruction head
is discarded for fine-tuning. Official code loads pre-generated shared masks, normalizes target
patches, predicts them with `pretrain_head`, and accumulates squared error in
`decoder.py:346-439`.

Literal FlowFormer++ is ineligible: it uses a 4-D full cost volume, a transformer cost encoder, and
separate pretraining data. The transferable hypothesis is narrower: block-shared missing local
scores can regularize a carrier-free Omega9 descriptor without changing deployment.

### 5.2 Migrated training formula

The main flow forward is always unmasked. In a training-only auxiliary branch, partition each
`9x9` spatial window into fixed clipped `3x3` source blocks. Sample exactly three of nine candidate
indices as masked; the same mask is used by both time slices and every source token in a block.

```text
p_i = SM_9(s_i)                                      # unmasked main path

M_b in {0,1}^9, sum_delta (1-M_b[delta]) = 3
s_mask(i) = min_valid_delta s(i,delta) - 1
s_tilde(i,delta) = M_b[delta]*s(i,delta) + (1-M_b[delta])*s_mask(i)
p_tilde_i = SM_9(s_tilde_i)
y_tilde_i = sum_delta p_tilde(i,delta) * C[delta,:]
g_i = exact_boundary_mean_{epsilon in Omega9}(y_tilde_{i+epsilon})
s_hat_i = R_h^T concat(y_tilde_i,g_i),               R_h in R^(64x9)

L_rec = mean_{i,delta:M_b[delta]=0}
        |Norm(s(i,delta)).detach() - Norm(s_hat(i,delta))|^2
L = L_flow + lambda(e)*L_rec
lambda(e) = (1/4)*max(0,1-e/20)
```

`R_h` and the auxiliary spatial mean exist only in training. The deployed block is exactly plain
Omega9 Match-Code with static `C`; epochs 20-29 therefore train the actual deployment graph with
`lambda=0`. The fixed mask ratio, block size, and schedule are one pre-registered point, not a
sweep.

### 5.3 Cost, uniformity, and risks

- Deployment arithmetic/state/control: no change from plain Omega9 Match-Code.
- Training-only cost: a second nine-score normalization, one boundary-normalized `3x3` mean,
  `64x9` reconstruction head per head/block, and masked MSE. Peak training memory and time will
  increase; inference/PPA claims must exclude the removed branch.
- All12 identity: same mask count, source block geometry, reconstruction formula, and annealing in
  every block. Resolution changes the number of windows, not semantics.
- Risk 1: unlike FlowFormer++, the local codebook has no deep cost-memory encoder; the auxiliary
  may teach only average score priors.
- Risk 2: a train/deploy mismatch remains during epochs 0-19 despite the unmasked main path.
- Risk 3: this is self-reconstruction, not extra ground-truth information, so gains may be smaller
  than FlowFormer++'s reported benchmark gains.

It does not repeat H68. H68 supplies a rich `N x N` matrix teacher and anneals back to H60's
deployed formula. BSMR9 reconstructs only nine local binary scores, uses no teacher attention, and
deploys static Match-Code. It does not repeat H76: H76 retains a spatial patch filter at inference;
BSMR9 removes all spatial reconstruction machinery.

DATE novelty risk is **high** as a headline algorithm because masked cost reconstruction is known.
It is valuable as a training-only ablation if it improves a strictly fixed binary deployment.

### 5.4 Single full30 falsification protocol

Run exactly one full30 from H60 epoch 2 with the common protocol. The resume checkpoint must retain
`R`; the exported deployment checkpoint must remove `R` through an explicit converter and then
strict-load a deploy-only config with zero missing/unexpected keys. Report both training and deploy
parameter counts so the removed head is auditable.

Falsify if epoch 20 shows a discontinuity after `lambda` reaches zero, if masked reconstruction
improves while valid825 does not beat H67, or if epoch-29 strict deployment export changes outputs
relative to the training model with `lambda=0`. No alternate mask ratio or reconstruction head is
authorized from this run.

## 6. R4-D, P1: AMM9 adaptive multimodal offset supervision

### 6.1 Original mechanism and code evidence

ADL argues that forcing a unimodal correspondence target at object boundaries is wrong. Its Eq. 4
models the target as a mixture of discretized Laplacians, and Eq. 5 gives the central cluster at
least half the mass while distributing the remainder according to local support. The official code
uses a local window, disparity clustering, Laplace scale `b=0.8`, central weight `0.8`, remaining
weight `0.2`, and cross entropy. Its DME code additionally finds modes and chooses the one with the
largest cumulative probability.

The DME inference is **not** transferred: its ordered 1-D disparity axis has no canonical mapping
to a `3x3` 2-D offset grid, and connected-component/mode control would complicate hardware. Only
the training target is migrated; deployment remains unchanged.

### 6.2 DSEC/all12 migration

For block `l`, let `r_l` be its known feature stride. Convert valid DSEC flow vectors in a `3x3`
ground-truth neighborhood into feature-offset coordinates `v=f_gt/r_l`. Apply the auxiliary only
when the central vector satisfies `||v_i||_inf <= 1.5`; larger motions are outside Omega9 and are
not clipped into a false boundary class. Cluster valid neighboring 2-D vectors with fixed L1
threshold `epsilon=1` in feature-offset units. Let cluster 1 contain the central pixel.

```text
mu_k = mean_{v in cluster k}(v)
L_k(delta) = exp(-||delta-mu_k||_1 / 0.8)
             / sum_{delta' in Omega9} exp(-||delta'-mu_k||_1 / 0.8)

w_1 = 0.8 + (|cluster_1|-1) * 0.2/(n_valid-1)
w_k = |cluster_k| * 0.2/(n_valid-1),  k != 1
q_i(delta) = sum_k w_k * L_k(delta)

p_i = SM_9(s_i)
L_amm = -mean_i sum_delta q_i(delta) * log(max(p_i(delta),eps))
L = L_flow + lambda(e)*mean_{all 12 blocks}(L_amm)
lambda(e) = (1/8)*max(0,1-e/20)
```

If only one valid cluster exists, `w_1=1` and the target degenerates to a single Laplacian, exactly
matching ADL's limiting case. Target maps may be cached offline for training speed. No clustering,
exponential, target SRAM, or extra state exists in inference.

### 6.3 Difference, cost, and risk

- H73-H78 change deployed score evidence/readout. AMM9 changes only the loss placed directly on
  the displacement distribution.
- H68 distills a rich learned matrix; AMM9 uses DSEC ground-truth geometry and no teacher.
- It is more specific than the generic Round3 note “add GT displacement supervision”: the target
  is intentionally multimodal at boundaries, has a central-dominant weighting rule, excludes
  out-of-support vectors, and has a fixed full30 schedule.
- Deployed hardware is bit-identical to plain Omega9 Match-Code. Training adds neighborhood gather,
  clustering, target generation, and CE only.
- Main risk: early-stage physical displacement can exceed Omega9, causing too few supervised
  samples; report eligible fraction separately for each of 12 blocks. A second risk is that DSEC
  sparse GT neighborhoods may produce unstable clusters.

DATE novelty risk is **high** as a primary contribution because the source is a stereo training
loss. It is a low-deployment-risk accuracy tool and a useful boundary ablation, not the paper title.

### 6.4 Single full30 falsification protocol

Run one plain Omega9 Match-Code full30 from H60 epoch 2 with the frozen protocol and the exact
`epsilon/scale/weights/lambda` above. Do not add DME inference, alter support, or combine AMM9 with
CF10/DN9/BSMR9 in the first run. Validate target normalization, single-cluster degeneration,
central-cluster dominance, sparse-GT masking, and stage-stride conversion before training.

Falsify if fewer than `10%` of valid training tokens are eligible in at least 8 of 12 blocks, if
flow-boundary AEE does not improve, or if global valid825 AEE fails to beat H67. A positive result
must retain H67-level spikes and show that the gain survives epochs 20-29 after the auxiliary loss
has vanished.

## 7. Common full30 and loading contract

Every authorized Round4 experiment is an independent alternative, never a stack. Use the frozen
H60 TTX epoch-2 checkpoint and exactly:

```text
DSEC crop 288x384
batch=8, workers=8, AMP, cupy
epochs=30, warmup=720, milestones=20,25
save/evaluate epochs=0,4,9,14,19,24,28,29
standard valid825: AEE, AAE, PE1/PE2/outlier, total_spikes, energy
```

A short smoke test may establish finite forward/backward execution only and may not reject a
candidate. Before full30, audit `ATLIFTernaryPSN=105`, attention modules `=12`, candidate modules
`=12`, `checkpoint_overlay_keys=210`, `unexpected=0`, and an exact candidate-only missing-key
allowlist. Trained same-mode checkpoints must strict-load with zero missing/unexpected. Training-
only candidates additionally require a deploy export and bitwise/close output comparison after the
auxiliary weight reaches zero.

The ranking gate is not the old 5% baseline window. A software winner must beat H67 AEE `1.4626`
on standard valid825, retain competitive AAE/PE metrics, and avoid a total-spike regression above
`26.3948G`. For R4-A/R4-B, report complete attention operations, local SRAM, cycles, and control;
`total_spikes*pJ` is not a sufficient energy claim.

## 8. What not to run from this round

1. Do not mix CF10 and DN9; that loses the row-only versus target-competition causal comparison.
2. Do not combine any Round4 candidate with H67 Motion-XOR, H76 patch smoothing, H77 contingency
   learning, or H78 grouping before its independent full30 result.
3. Do not sweep null bias, mask ratio, block size, clustering threshold, Laplace scale, auxiliary
   weight, offset support, temperature, or group count.
4. Do not port EDCFlow's five-window warping/GRU or EMatch's TRN/SCA into the attention block.
5. Do not claim RCM, LightGlue, FlowFormer++, or ADL as reproduced. The formulas above are bounded
   transfers whose novelty and failure modes must be stated explicitly.

## 9. Final recommendation

If only one Round4 experiment can be afforded, run **R4-A CF10**. It adds a capability absent from
H67-H78, has the smallest deployed increment, preserves many-to-one dense flow, and directly tests
whether low-event/occluded tokens are hurting H67. If two can be afforded, add **R4-B DN9** as the
contradictory assignment experiment; its result decides whether target competition is beneficial or
fundamentally wrong for DSEC. AMM9 is the safest deployment-neutral accuracy side line. BSMR9 is
last because its source mechanism relies on a stronger cost-memory encoder than this network has.
