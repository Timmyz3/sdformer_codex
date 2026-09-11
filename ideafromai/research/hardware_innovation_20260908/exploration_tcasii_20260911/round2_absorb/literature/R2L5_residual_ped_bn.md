# R2L5 — Membrane-shortcut vs SEW vs this net’s I24 residual / PED / live projection BN

Date: 2026-09-11. Literature check for `round2_absorb`. Frozen observations only from [`PROBLEM.md`](../PROBLEM.md) and [`../../IDENTITY_ATLIF.md`](../../IDENTITY_ATLIF.md). This note is a prior-to-copy + hole map, not an idea catalog and not a novelty claim.

**Question.** After AT-LIF \(\{0,\theta\}\) with layer-shared \(\theta\) absorbed into the next \(W\), the spike consumer is binary. Residual / PED / I24 and live full-domain projection BN are **not** that absorbed spike. Map Spike-driven Transformer membrane-shortcut vs SEW residual vs this net. What dual-path X is **not** an SDT membrane-shortcut rename.

**One-line answer.** Membrane-shortcut (SDT / MS-ResNet restated by SDT / SDformerFlow) is **A** for “add on the continuous residual stream, then SN, so spike tensors stay \(\{0,1\}\).” SEW ADD is **A** as a negative identity: spike+spike → integer spikes, which SDT already rejected. After absorb, this net’s **spike GeMM is exactly that binary contract** — Prosperity / Gustav / FireFly copy there. I24 skip, PED \(1\times1\) on continuous `r1out`, and live `proj.norm_layer` over `10×96×120×160` are a **different tensor family**. Dual-path X, if any, is last-use / occupancy / join of `{binary gate after absorb, continuous residual/PED, BN-stat}`, not “we rearranged the shortcut before SN.”

---

## 0. Identity freeze (do not rewrite)

From `IDENTITY_ATLIF.md` / round-2 `PROBLEM.md`:

- Official AT-LIF: \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\). Inference: layer-shared \(\theta\) **absorbed** into next \(W\). Transmitted spikes are then \(\{0,1\}\).
- **Two objects stay distinct.** (1) Spike GeMM after absorb = binary select-add. (2) Residual / PED / I24 = **a different continuous tensor**. Not AT-LIF carrying analog amplitude. Dual consumers, if they exist: consumer A = **binary gate after absorb**; consumer B = **continuous residual path**. Do **not** write “continuous AT-LIF amplitude shared by two MACs.”
- Before threshold: noncausal T10 mix (PSN / lifting CSE) is continuous arithmetic, then threshold, then absorb.
- Task: event-camera 2D optical flow, DSEC valid825 AEE. Student Motion C12 / H67 / ep34. Native projection conv + BN + residual add. Captured projection BN uses **actual batch statistics** over `10×96×120×160`.
- Integer gates / I24 / PED q24: 0-diff vs model on two captured windows.
- Venue: TCAS-II, 5 pages. Same-port / same-state / same-backpressure net service \(\ge 15\%\). AEE abs \(\le 1.259\) and relative \(\le +0.005\) vs ordinary \(1.219801338\).

**Forbidden in this sheet:** retitling AT-LIF \(o\) as the I24 skip; retitling PED as “membrane of the next SN”; retitling live proj BN as tdBN / FrozenBN; selling IAND residual as a hardware win.

**Read in full (local texts):**

| File | Role |
|---|---|
| `survey_ab_fusion_20260910/p0_txts/2307.01694.txt` | Spike-driven Transformer (SDT), NeurIPS 2023 — membrane shortcut + SDSA |
| `survey_ab_fusion_20260910/p0_txts/2102.04159.txt` | Fang et al., *Deep Residual Learning in SNNs* (SEW ResNet), NeurIPS 2021 |
| `survey_ab_fusion_20260910/p0_txts/2409.04082.txt` | SDformerFlow — MS + deformed PED shortcut (Eq. 13–14) on this family |
| `exploration_tcasii_20260911/literature/L4_isscc_fusion_looptree.md` | ISSCC 23.2 / LoopTree / RISCSparse vs r1 fusion set and **unfrozen** proj BN |

Hu et al. MS-ResNet (arXiv:2112.08954, SDT ref. [26]) is **not** in the local `p0_txts`. MS equations below are quoted from SDT’s restatement and from this tree’s `MS_ResBlock`, not from Hu’s PDF. QKFormer SPEDS (arXiv:2403.16552) is not in `p0_txts`; deformed-shortcut algebra is quoted from SDformerFlow Eq. 13–14, which this student’s `SpikingPEDLayer` implements.

---

## 1. Title confirmation

| arXiv | Printed title | What it actually is |
|---|---|---|
| 2102.04159 | **Deep Residual Learning in Spiking Neural Networks** (Fang, Yu, Chen, Huang, Masquelier, Tian; NeurIPS 2021) | SEW residual: \(O=g(\mathrm{SN}(F(S)),S)\) with \(g\in\{\mathrm{ADD},\mathrm{AND},\mathrm{IAND}\}\). Identity mapping + gradient analysis vs “replace ReLU with SN.” ImageNet / DVS Gesture / CIFAR10-DVS. Not optical flow, not a chip. |
| 2307.01694 | **Spike-driven Transformer** (Yao, Hu, Zhou, Yuan, Tian, Xu, Li; NeurIPS 2023) | SDSA (Hadamard/mask + column sum) **and** membrane shortcut so **all spike tensors stay binary**. ImageNet-1K 77.1%. Energy model is \(R\cdot T\cdot\)FLOPs, not same-port service. |
| 2409.04082 | **SDformerFlow** (local full text) | Event OF on DSEC. Uses **MS** in residual blocks (Fig. 4) **and** a **deformed \(1\times1\) stride-2 shortcut** on patch embedding (Eq. 13–14). Eval: “disable the tracking of running states for batch normalization layers.” |

---

## 2. Three residual algebras (quotes, then one table)

### 2.1 Vanilla Spiking ResNet (SEW paper Fig. 1(b); SDT “vanilla Res-SNN”)

SEW Eq. (7):

\[
O^l[t]=\mathrm{SN}\bigl(F^l(S^l[t])+S^l[t]\bigr).
\]

Shortcut **adds a spike tensor into the next membrane**, then fires. Identity mapping \(\mathrm{SN}(S)=S\) is **neuron-model-dependent** (works for IF with \(0<V_{th}\le 1\) and reset-to-0; fails for learnable-\(\tau\) LIF). Gradient through a chain of SNs vanishes/explodes (SEW Eq. (8)).

SDT’s name for the same class: “Vanilla Res-SNN … performs a shortcut between membrane potential and spike.”

### 2.2 SEW (Fang et al. §3.3, Fig. 1(c), Table 1)

\[
O^l[t]=g\bigl(\mathrm{SN}(F^l(S^l[t])),\,S^l[t]\bigr)=g(A^l[t],S^l[t]),
\]

with **both operands spike tensors**. Element-wise \(g\):

| \(g\) | Expression | Identity by | Output alphabet |
|---|---|---|---|
| ADD | \(A+S\) | \(A\equiv 0\) (zero last BN in \(F\)) | **integer**, \(\le k+1\) after \(k\) blocks |
| AND | \(A\land S=A\cdot S\) | \(A\equiv 1\) (BN bias large enough to fire) | binary |
| IAND | \((\neg A)\land S=(1-A)\cdot S\) | \(A\equiv 0\) | binary |

Downsample SEW (Fig. 2(b)): **SN on the shortcut conv**, so the skip is also a spike.

SEW’s own relief for ADD: “the output of \(k\) sequential SEW blocks will be no larger than \(k+1\).” That sentence is the integer-spike contract.

**SDT’s objection (2307.01694 §1 and §3.2), quoted:**

> “existing spiking Transformers usually follow the SEW-SNN residual design, whose shortcut is spike addition and thus outputs **multi-bit (integer) spikes**. This shortcut can satisfy event-driven, but **introduces integer multiplication**.”

> “Only binary spikes can support the spike-driven function. However, the values in the spike tensors are multi-bit (integer) spikes, as the SEW shortcut builds the addition between binary spikes.”

AND / IAND keep \(\{0,1\}\) I/O. IAND is the residual used by Spike-IAND-Former (local L5, arXiv:2503.19643) to *destroy* integer ADD so later conv can stay AND-gates. That is identity-illegal on **this** net’s continuous PED (L5 already: “IAND residual deletes the continuous consumer”).

### 2.3 Membrane shortcut (SDT §3.1–3.2, Eq. 7–11)

SPS / encoder (symbols as printed):

\[
\begin{aligned}
U_0 &= u + \mathrm{RPE}, \\
S_0 &= \mathrm{SN}(U_0), \\
U'_l &= \mathrm{SDSA}(S_{l-1}) + U_{l-1}, \\
S'_l &= \mathrm{SN}(U'_l), \\
S_l &= \mathrm{SN}\bigl(\mathrm{MLP}(S'_l)+U'_l\bigr).
\end{aligned}
\]

“Residual connections are applied to **membrane potentials**.” Fig. 2 caption: “the shortcut is constructed **before** the spike neuron layer … residual connections **between membrane potentials** to make sure that the values in the **spike matrix are all binary**.”

Four reasons SDT gives (§3.2): (i) binary spike-driven AC; (ii) MS-Res-SNN accuracy > SEW-Res-SNN (their Table 6: `+ MS` is the accuracy win, SDSA alone is a small drop); (iii) bio-plausibility as membrane-distribution opt; (iv) dynamical isometry cited from MS-Res-SNN.

**What MS is, mechanically.** One residual **stream** \(U\) (continuous). Linear ops see only \(S=\mathrm{SN}(U)\in\{0,1\}\). Add happens **on \(U\)**, then SN. The continuous tensor is the **same neuron family’s membrane / pre-activation stream**. There is **not** a second dense MAC whose operand is that \(U\) *instead of* going through SN.

ANN ancestor (must copy if anyone writes “pre-activation residual”): He, Zhang, Ren, Sun, *Identity Mappings in Deep Residual Networks* (ECCV 2016) — add **before** the nonlinearity. SDT cites this as [68]. MS is that placement under a spike SN, plus the **spike-driven** claim that follows from binary \(S\).

### 2.4 Side-by-side

| | Vanilla Spiking ResNet | SEW | Membrane shortcut (SDT / MS) |
|---|---|---|---|
| Addends | \(F(S)\) (current / membrane) **+** spike \(S\), then SN | spike \(A=\mathrm{SN}(F(S))\) **+** spike \(S\) (or AND/IAND) | continuous \(U_{l-1}\) **+** block output **before** SN |
| What the **next linear** sees | spikes (if identity SN works) | ADD: **integer spikes**; AND/IAND: binary | **binary spikes only** |
| Identity mapping | neuron-restricted | easy for ADD/IAND via \(A\equiv 0\) | \(F\equiv 0\Rightarrow U\) unchanged; SN still binarizes the *communication* |
| Spike-driven AC (SDT’s criterion) | intended, fragile | ADD **breaks** it | **the point of the paper** |
| Continuous tensor that leaves the block | no (except internal \(V\)) | ADD: integer spike train | **yes: \(U\)**, but it is the **next SN’s membrane**, not a second conv’s dense operand |

---

## 3. This net’s objects (software graph, not a new claim)

Student family uses `MS_PED_Spiking_PatchEmbed_Conv_sfn` (`Spiking_modules.py`). Two residual *kinds* sit on the same r1 chain. They must not be collapsed.

### 3.1 r1 MS residual (`MS_ResBlock`, comment: “Spike-driven Transformer NIPS 2023”)

```
identity = x                          # continuous residual stream in
x = sn1(x);  conv1;  norm1
x = sn2(x);  conv2;  norm2
out = out + identity                  # ADD on continuous tensors
```

`cnt_fun='ADD'`. This **is** SDT membrane-shortcut on patch residual r1. After AT-LIF absorb, `conv1` / `conv2` see \(\{0,1\}\). The skip `identity` is **not** AT-LIF \(o\); it is the block input residual stream (I24-class in the integer capture).

L4’s r1 sketch remains valid:

```
sn1 → conv1 → norm1 → sn2/PSN(T10) → conv2 → norm2 + identity  =  r1out
```

Noncausal T10 sits **before** the AT-LIF threshold of `sn2` (round-2 freeze). That mix is continuous; absorb does not make T10 a binary GeMM.

### 3.2 PED (`SpikingPEDLayer.forward`, lines 817–825)

```
x_res = conv_res(x.flatten(0,1))      # 1×1, stride 2, on CONTINUOUS r1out
x     = sn(x)                         # AT-LIF; after absorb, conv sees {0,1}
x     = conv(x.flatten(0,1))          # 3×3
x     = norm_layer(x)                 # BatchNorm2d, live stats
x     = (x + x_res).reshape(...)      # ADD after BN
```

SDformerFlow Eq. 13–14 (same algebra; they call it SPE):

\[
z_{\mathrm{res}}=\mathrm{Conv}_{\mathrm{deformed}}(I),\qquad
z=\mathrm{BN}\bigl(\mathrm{Conv}(\mathrm{SN}(I))\bigr)+z_{\mathrm{res}}.
\]

“A convolutional layer \(\mathrm{Conv}_{deformed}\) with kernel size of \(1\times 1\) and stride size of 2 is applied to the residual.”

So PED is **not** “MS under another name.” MS adds **inside** the residual stream then SN. PED **forks** continuous `r1out` / \(I\):

- **Path A (gate / spike after absorb):** \(\mathrm{SN}\to 3\times 3\) conv. After absorb this is binary GeMM. Prosperity / Gustav / FireFly **A**.
- **Path B (continuous residual / PED):** \(1\times 1\) stride-2 `conv_res` on the **same** continuous tensor. Dense multi-bit MAC. Integer capture: PED q24, 0-diff.
- **Then** live BN on path A’s conv-out, **then** ADD with `x_res`.

Even/even spatial sites (stride 2) need the **value** on path B; other sites may be gate-only. Union of the two supports is the occupancy that binary NRV on gates alone does not retire (round-1 F1 hole; still true after absorb, because path B never became a spike).

### 3.3 I24 is the skip tensor, not AT-LIF \(o\)

Integer capture names **three** 0-diff objects: gates, **updated I24**, **PED q24**. I24 is the residual stream / skip that is written and later reread for PED and for the post-BN ADD. After absorb:

- gates = binary (or packed 0/1) consumer of the thresholded path;
- I24 / PED q24 = **continuous residual family**.

Writing “AT-LIF analog payload fans out to two MACs” is the identity the freeze forbids. The producer of the fork is **`r1out` / residual stream**, a post-add continuous tensor. AT-LIF only binarizes **path A** after absorb.

### 3.4 Live full-domain projection BN is not MS and not FrozenBN

Captured `proj.norm_layer`: eval, `track_running_stats=False`, reduction over `[10,96,120,160]` = \(18\,432\,000\) values (`full_chain/README.md`; L4 §4.4 / §5). Local-window replay with **free** \(\mu,\sigma\) undercharges wait/storage.

SDformerFlow eval sentence: “we disable the tracking of running states for batch normalization layers.” That is consistent with **not using FrozenBN**, but the paper does **not** name a hardware completion barrier, Welford sidecar, or retain/recompute of activations for affine.

**tdBN** (Zheng et al., AAAI 2021; not re-read here as a local `p0_txt`; round-1 workflow already mapped it): training normalizes over a joint \(T\times N\times H\times W\) domain; **inference uses dataset moving-average fused into conv**. RISCSparse ICCAD 2024 (L4 §4.4) is explicit: inference BN is \(Y=aX+B\) with **fixed** \(\mu,\sigma^2\). Both are **A as negative controls**. They **close** the barrier by freezing. This student’s captured proj BN does **not**.

LoopTree (L4 §3): a global reduction is an **untiled** (or two-pass) fusion constraint. Tile produced \(\not\Rightarrow\) tile consumable \(\not\Rightarrow\) tile releasable until moments exist. ISSCC 23.2 LFS slot-replace and non-overlapped pad **cannot** stand in for moments (L4 §2.5, §5.1).

BN-stat is therefore a **third use** (narrow \((\mu,M_2)\) or a domain-sized pin of activations), not a membrane shortcut and not AT-LIF \(o\).

---

## 4. After absorb: what became binary, what did not

```
                 noncausal T10 mix (continuous)     threshold      absorb θ into W
r1 source  ──►  PSN / lifting CSE  ──────────────►  AT-LIF     ──►  {0,1} × W     = spike GeMM
                                                                      │
r1out / I24 ──────────────────────────────────────────────────────────┼──► conv_res 1×1     = PED continuous
  (MS add of identity + conv2/BN2; continuous skip)                   │
                                                                      └──► sn → conv → BN  = gate + live proj BN
                                                                                 │
                                                                                 └── ADD ← conv_res
```

| Tensor | After absorb | Which paper already named it |
|---|---|---|
| AT-LIF communication into `conv1`/`conv2`/`proj.conv` | **binary** \(\{0,1\}\times W\) | SDT’s *goal* for MS; now the **locked neuron identity**, independent of shortcut placement |
| T10 mix before threshold | still continuous CMVM | PSN / lifting; **not** spike GeMM |
| r1 `identity` / I24 skip | still continuous | MS residual stream (SDT), **plus** a later dense reread this net actually does |
| PED `conv_res` input | still continuous `r1out` | SDformerFlow Eq. 13 \(z_{\mathrm{res}}=\mathrm{Conv}_{deformed}(I)\) — **algebra A** |
| `proj.norm_layer` moments | live full-domain, not fused \(a,b\) | **not** SDT, **not** SEW, **not** RISCSparse inference BN |

Prosperity / GustavSNN NRV / FireFly-S Bitmap AND become **legal complete A on the binary spike GeMM only**. They do not retire I24, do not compute `conv_res`, and do not pay BN moments (round-2 SCOPE).

---

## 5. Dual-path X that is **not** an SDT membrane-shortcut rename

### 5.1 Reskins a TCAS-II reviewer will subtract

| Claim | Why it is a rename / identity break |
|---|---|
| “We use membrane shortcut so spikes stay binary” | SDT §3.2 + SDformerFlow Fig. 4 + this student’s `MS_ResBlock`. After absorb, binary communication is **the neuron identity**, not a residual invention. |
| “Pre-activation residual” | He et al. 2016 + MS. Placement is A. |
| “SEW residual for AT-LIF” | ADD reintroduces integer spikes (SDT already forbade). AND/IAND **delete** the continuous PED addend. Spike-IAND-Former is the hardware paper that *wanted* that deletion. |
| “Deformed \(1\times1\) skip / PED block” | SDformerFlow Eq. 13–14; `SpikingPEDLayer` **is** that block. Algebra is A. |
| “Dual residual like ResNet projection downsample” | He et al. downsample conv on the skip. Names the **graph**, not last-use of two dtypes on one SRAM. |
| “Fuse BN into conv” | RISCSparse / TensorRT FrozenBN. Illegal while proj BN is live full-domain. |
| “Continuous AT-LIF amplitude, two MACs” | Conflicts with `IDENTITY_ATLIF.md`. Path A is binary after absorb; path B is the **residual stream**, not \(o\). |

MS answers: *where do we add so that the spike matrix stays \(\{0,1\}\)?* After absorb, that answer is already “yes” on path A.

### 5.2 What dual-path would have to be (hole, not a title)

SDT MS: **one** continuous stream \(U\), **one** binary communication \(S=\mathrm{SN}(U)\), **one** class of linear ops (spike-driven add). Last-use of \(U\) is “until the next SN/add in the same stream.”

This net after absorb:

1. **Two MAC consumers of one producer tensor `r1out`**, different dtypes: binary GeMM (path A) vs dense I24/`conv_res` (path B). Last-use(A) is not last-use(B). Gate-silent does **not** retire the word if PED still needs it.
2. **I24 reread** of a skip that MS would treat as “the membrane already in the SN.” Here the skip is a **named SRAM object** (0-diff I24) that is written, then reread for PED and for post-BN ADD. Phantom third consumer if last-use is not architected (round-1 P05-I2 class; still a hole after absorb).
3. **Live proj BN** is a **domain barrier** on a **later** tensor (`proj.conv` out at `10×96×120×160`), not a membrane. Affine+ADD cannot legally complete from a local window with gifted \(\mu,\sigma\). LoopTree retain/recompute/refetch must be charged (L4).
4. **T10 is pre-threshold.** Binary GeMM reuse (Prosperity EM/PM, Gustav NRV at one tick, FireFly Bitmap) does not schedule the noncausal mix.

That four-tuple is **not** “membrane-shortcut optical-flow Transformer.” SDformerFlow already did MS + deformed skip + DSEC OF **as a model**. It did not measure same-port last-use of `{gate, PED, BN-stat}` or move the frozen **8088** long-backpressure class.

### 5.3 Honest leftover X sentence (for ideation; not chosen)

> After \(\theta\)-absorb, copy Prosperity/Gustav/FireFly **in full** on the binary spike GeMM. The increment, if any, is a **typed dual-consumer completion** of a continuous skip (I24 / PED q24) whose last-use is **not** the binary gate’s last-use, with **paid** full-domain proj-BN moments as a third conjunct — occupancy \(\mathrm{support}(\mathrm{gate})\cup\mathrm{support}(\mathrm{PED})\) plus a stats barrier — without renaming SDT’s membrane add and without treating AT-LIF \(o\) as the skip.

If union density \(\approx 1\) (PED almost always live), skip-on-gate is empty and the sentence collapses to **last-use / reread elimination / BN accounting**. Those are still not MS. They may still die as register-allocation / Welford-on-fill (L4 A). Kill-gates below decide; this sheet does not pick a letter.

---

## 6. A / B / X / controls / kill-gates

### A — complete prior to copy

1. **SEW block + \(g\in\{\mathrm{ADD},\mathrm{AND},\mathrm{IAND}\}\)** (Fang NeurIPS 2021): identity-mapping table, downsample SN on skip, ADD integer-spike bound \(\le k+1\). Copy as **relative prior and negative identity** for “spike+spike residual.”
2. **Membrane shortcut** (SDT §3.2, Eq. 7–11; SDformerFlow Fig. 4; local `MS_ResBlock`): add on continuous \(U\) before SN so spike tensors stay binary. Copy the **placement** and the **spike-driven claim**. After absorb, the claim is already the neuron contract.
3. **He et al. identity-mapping / pre-activation ResNet** (ECCV 2016): ANN ancestor of MS.
4. **Deformed downsample skip** (SDformerFlow Eq. 13–14 = `SpikingPEDLayer`): \(z_{\mathrm{res}}=\mathrm{Conv}_{1\times1,s=2}(I)\), \(z=\mathrm{BN}(\mathrm{Conv}(\mathrm{SN}(I)))+z_{\mathrm{res}}\). Copy the **graph**. Do not call the graph X.
5. **Binary spike GeMM accelerators** (Prosperity, GustavSNN, FireFly-S, LoAS): complete A on **path A after absorb only**.
6. **Frozen / moving-average BN fold** (RISCSparse §3.4, tdBN inference, TensorRT): complete A as **numeric control**. Illegal as silent HW of the captured student.
7. **LoopTree retain/recompute/per-tensor retain** (L4): language for I24 vs BN moments vs `r1out`. Fusion set is an input; proj BN stays in the set.

### B — hole in **this** net (freeze only)

- After absorb, path A is binary; path B (I24 / PED q24) is still continuous and 0-diff on captured windows.
- Dual consumers after source: binary gate **and** continuous residual/PED. Native proj conv + BN + residual add exist.
- Proj BN = actual batch stats over `10×96×120×160`, not running stats. Free \(\mu/\sigma\) undercharges.
- Same two-stage SIMD: always-ready \(6938\to 5354\) (\(-22.83\%\), part generic fusion); **long backpressure both 8088**. Source CSE does not move the join.
- Lifting AEE \(1.232979368\) fails relative \(+0.005\); ordinary \(1.219801338\) is the strong control.
- Do not add integer-consumer \(-5.78\%\) to the SIMD table.

None of SEW, SDT, or SDformerFlow measures this B as same-port service.

### X — why a dual-path letter is not an MS reskin

MS = one stream, binary communication, add-before-SN. Dual-path X must be a **schedule/state contract** that (i) copies binary GeMM A on path A, (ii) **keeps** path B dense and bit-true, (iii) **pays** BN moments, (iv) retires I24 only at last-use of the **union**, and (v) moves **8088** / \(\ge 15\%\) complete-chain service. If the figure is SDT Fig. 2 with “optical flow” in the caption, it is a reskin.

### Strongest controls

- SEW-ADD student (integer spikes into proj.conv) — expect spike-driven A broken; AEE may still pass; **identity fail** for “only sparse add.”
- SEW-IAND / Spike-IAND-Former residual — expect PED 0-diff **fail**.
- MS-only projection (`SpikingEmbeddingLayer._forward_MS`: sn→conv→BN, **no** `conv_res`) — different graph; valid825 vs ordinary.
- Frozen-BN student (RISCSparse \(Y=aX+B\)) **and** unfrozen full-domain student; AEE on both.
- Binary-NRV / gate-only last-use **without** PED consume — occupancy control; should **not** move 8088 if PED is live.
- LoopTree layer-by-layer vs untiled full `r1out` vs tiled fusion, **same** ports, **paid** \(\mu,\sigma\).
- Ordinary T10 at AEE \(1.219801338\); do not use lifting as a passing prior.

### Kill-gates (local numbers, not ImageNet %)

- AEE abs \(> 1.259\) or vs ordinary \(> +0.005\).
- Complete-chain same-port net service \(< 15\%\).
- Long-backpressure remains **8088** (always-ready-only movement is insufficient).
- Integer gates / I24 / PED q24 miss 0-diff on the two windows (unless the idea is an explicitly new student that re-runs valid825).
- Gifted \(\mu/\sigma\) or FrozenBN used to claim service.
- Title is MS / SEW / “binary spikes” / Prosperity-on-path-A without a dual-consumer figure.
- IAND residual, or “AT-LIF analog shared by two MACs.”

### Two-sentence prior statement (reviewer, not a pitch)

SEW already defined spike-element-wise ADD/AND/IAND and showed ADD leaves integer spikes; Spike-driven Transformer already moved the residual onto the membrane so that **spike tensors stay binary**, and SDformerFlow already used that MS plus a deformed \(1\times1\) stride-2 skip as patch embedding on DSEC. After AT-LIF \(\theta\)-absorb, binary GeMM on the spike path is the locked identity — the remaining object is a **typed continuous skip** (I24/PED) plus **live full-domain proj BN**, which those papers do not schedule as dual last-use.

### Biggest objection

PED Eq. 13–14 **is** the dual-path *graph*. A reviewer can say: “you implemented SDformerFlow SPE and called last-use a circuit.” The only defense is a **measured** same-port occupancy split \(\{\mathrm{wait\_gate},\mathrm{wait\_PED},\mathrm{wait\_BNstat},\mathrm{wait\_I24\_reread}\}\) that moves 8088, with MS/SEW/FrozenBN as published controls. If the split is simultaneous ready, dual-path is empty and MS was always enough.

---

## 7. What this note does **not** do

- Does not choose a letter X or add modules.
- Does not treat ImageNet 77.1%, SEW-ResNet-152 accuracy, or SDT \(87.2\times\) SDSA energy as local PPA.
- Does not quote Hu et al. 2112.08954 primary text (not in `p0_txts`).
- Does not claim QKFormer SPEDS equations beyond SDformerFlow Eq. 13–14.
- Does not add the two resource tables in `PROBLEM.md`.
- Does not revive non-absorbable int8 \(\theta_g\) as the skip.
- Does not treat r1 `norm1`/`norm2` freeze files as a substitute for `proj.norm_layer`.

**Sources (absolute):**

- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2307.01694.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2102.04159.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/survey_ab_fusion_20260910/p0_txts/2409.04082.txt`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/literature/L4_isscc_fusion_looptree.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/IDENTITY_ATLIF.md`
- `/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/exploration_tcasii_20260911/round2_absorb/PROBLEM.md`
- `/home/zhumd/work/sdformer_codex/SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py` (`MS_ResBlock`, `SpikingPEDLayer`, `MS_PED_Spiking_PatchEmbed_Conv_sfn`)
