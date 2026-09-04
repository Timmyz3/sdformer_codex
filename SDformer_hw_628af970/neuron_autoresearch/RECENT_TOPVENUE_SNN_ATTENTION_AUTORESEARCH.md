# Recent Top-Venue SNN Attention Autoresearch

Date: 2026-05-19

Scope: only recent top venues from 2024-2026. SDSA-style attention is treated as
background, not as a main innovation candidate.

## Search Criteria

- Venue: CVPR, ICLR, NeurIPS, ICML, ECCV, AAAI.
- Time: 2024-2026.
- Topic: spiking transformer attention, spike-driven attention, spatio-temporal
  spiking attention, event/video spiking attention, hardware-friendly spike
  similarity.
- Priority: mechanisms that can modify SDformerFlow's attention without
  rewriting the whole backbone.

## Source Library

Local source roots:

| Paper / Project | Venue | Local source |
|---|---|---|
| STAtten | CVPR 2025 | `/root/private_data/work/optimization_sources/attention_optimization/CVPR2025_STAtten/` |
| SpikeVideoFormer | ICML 2025 | `/root/private_data/work/optimization_sources/attention_optimization/ICML2025_SpikeVideoFormer/` |
| Spikingformer | AAAI 2026 | `/root/private_data/work/optimization_sources/attention_optimization/AAAI2026_Spikingformer/` |
| QSD-Transformer | ICLR 2025 | `/root/private_data/work/optimization_sources/attention_optimization/ICLR2025_QSD-Transformer/` |
| Spike-driven Transformer V2 | ICLR 2024-era source | `/root/private_data/work/optimization_sources/attention_optimization/ICLR2024_Spike-Driven-Transformer-V2/` |
| Spiking Wavelet Transformer | ECCV 2024 | `/root/private_data/work/optimization_sources/attention_optimization/ECCV2024_Spiking-Wavelet-Transformer/` |
| STAA-SNN candidate clone | older SCTFA source, not confirmed official STAA-SNN | `/root/private_data/work/optimization_sources/attention_optimization/CVPR2025_STAA-SNN_candidate/` |

Important note: `CVPR2025_STAA-SNN_candidate` is **not** confirmed to be the
official STAA-SNN repository. It is a related spatial-channel-temporal attention
SNN implementation and should not be cited as the STAA-SNN code unless verified.

## Full Candidate Pool From Local + Web Survey

This table merges this pass with other local agent notes in
`H9_bipolar_self_attention/docs/design.md`, `H13_SERIES_REVIEW.md`,
`THREE_AXIS_NEXT_PLAN_20260518.md`, `autoresearch.ideas.md`, and
`PAPER_CO_DESIGN_PROPOSAL.md`.

| Candidate | Venue | Main idea | Fit to SDformerFlow | Status |
|---|---|---|---|---|
| α-XNOR SSA | CVPR 2025 | spike-tailored Q/K similarity that rewards spike matches and lightly rewards non-spike matches | high for Q/K ternary attention | prioritize |
| A2OS2A | CVPR 2025 | accurate addition-only SSA with binary Q, ReLU/nonnegative K, ternary V, no softmax/scaling | high as a separate attention paradigm | prioritize |
| SpikeVideoFormer SDHA | ICML 2025 | video-oriented Hamming attention with joint temporal-spatial tokens | high for event optical flow | prioritize |
| Saccadic SSSA | ICLR 2025 | spike-distribution relevance plus saccadic temporal focus | high, especially temporal gate | prioritize after simpler gates |
| STAA-SNN | CVPR 2025 | spatial-temporal attention aggregation, step attention, timestep dropout | high for temporal aggregation | prioritize after Q/K score fixes |
| TIM | IJCAI 2024 | convolution-based temporal interaction module inserted into spiking transformer | high, low code risk | add as temporal plugin |
| TTFSFormer | ICML 2025 | time-to-first-spike conversion of Transformer attention/nonlinearities | medium, more conversion than direct training | use for time-gating ideas |
| SEMM/EMSA | NeurIPS 2024 | spike-driven MoE routing for attention heads/experts | medium-high, good sparse story | later |
| QSD-Transformer/SID | ICLR 2025 | spike information distortion, multi-bit training, distillation | medium, best as analysis/loss | profiler first |
| QP-SNN | ICLR 2025 | quantization + structured pruning using spike-activity singular values | medium, pruning/quantization not attention | later for pruning |
| Spike2Former | AAAI 2025 | spike-driven deformable transformer encoder, normalized integer LIF | medium for decoder/refinement/complex modules | later |
| SpikingResformer | CVPR 2024 | Dual Spike Self-Attention bridging ResNet and ViT | medium, older but top venue | baseline/reference |
| OST | IJCAI 2024 | one-step spiking transformer with time-domain compression and spiking linear transform | medium, architecture-level | later |
| Spikingformer | AAAI 2026 | spike-driven residual+self-attention, hardware-clean residuals | medium, hardware cleanliness | audit path |
| SWformer | ECCV 2024 | attention-free wavelet/frequency token mixer with negative spikes | medium, event-flow edge/frequency story | fallback/FFN mixer |
| STAtten | CVPR 2025 | spatiotemporal chunk attention | medium-high, but heavier than gates | later if H20/H19 fail |

## Ranked Attention Candidates

### Rank 1: CVPR 2025 α-XNOR Spiking Self-Attention

Paper:

- `Rethinking Spiking Self-Attention Mechanism: Implementing α-XNOR Similarity
  Calculation in Spiking Transformers`, CVPR 2025.

Core mechanism:

```text
For binary spikes:
α-XNOR(x, y) =
  1, if x = y = 1
  α, if x = y = 0
  0, otherwise
```

Why it matters:

- Directly addresses the exact problem in spiking attention: dot product treats
  `0*0` and mismatches poorly.
- Hardware-friendly: XNOR-style matching plus weighted non-spike matches.
- Paper is newer and more targeted than SDSA.

Fit to our ternary ATLIF setting:

Our Q/K are not binary `{0,1}`; they are threshold-scaled ternary
`{-theta, 0, +theta}`. A faithful adaptation should use signs for similarity
and keep theta for value amplitude/sparsity:

```text
q_sign, k_sign in {-1, 0, +1}
score =
  +1        if q_sign == k_sign and both nonzero
  +alpha0   if q_sign == k_sign == 0
  -beta     if q_sign == -k_sign and both nonzero
   0        otherwise
```

First experiment:

- **H18a α-XNOR ternary matrix attention**
- Score matrix uses ternary α-XNOR.
- Value path uses threshold-scaled K, because SDformerFlow has no V.
- Normalize with Shiftmax first for stability, then try L1/pow2 after it works.

Expected story:

`α-XNOR` provides paper-backed spike similarity; our novelty is extending it to
signed ternary ATLIF spikes and event-flow QK attention.

Risk:

- For sparse event data, too much `0-0` reward can cause attention collapse to
  silent regions. α must be small or scheduled.

### Rank 2: CVPR 2025 A2OS2A Addition-Only Spiking Self-Attention

Paper:

- `Spiking Transformer: Introducing Accurate Addition-Only Spiking
  Self-Attention for Transformer`, CVPR 2025.

Core mechanism:

- Hybrid spike roles instead of forcing all Q/K/V to be binary.
- Q is binary, K is ReLU/nonnegative, V is ternary.
- Removes softmax and scaling.
- Targets addition-only attention with better accuracy.

Why it matters:

- This was already flagged in
  `neuron_experiments/H9_bipolar_self_attention/docs/design.md` as H9d, but was
  missing from the previous survey ranking.
- It is extremely aligned with our current mixed mechanism:
  attention Q/K are special, FFN/downsample can stay binary, and ternary should
  be used where polarity matters.

Fit to SDformerFlow:

High, but it should be a separate branch rather than mixed into H13/H14:

```text
Q path: binary or sign-only event gate
K path: nonnegative/ReLU-like carrier, possibly abs(K) or positive ATLIF branch
V path: threshold-scaled ternary, initially reuse K because SDformerFlow has no V
normalization: none or L1/pow2, no Shiftmax first
```

First experiment:

- **H18b A2OS2A-QK adapter**
- Preserve baseline carrier at first:

```text
main = K * sn2_q(sum(Q))
aux_gate = addition-only score(Q_binary, K_nonnegative, V_ternary)
out = main * aux_gate
```

Second experiment:

- **H18c A2OS2A direct adapter**
- Replace the attention carrier only after H18b is stable.

Expected story:

This is the cleanest top-venue support for a **hybrid binary/nonnegative/ternary
attention** story instead of pure BSA/Shiftmax.

Risk:

- Our module lacks an independent V projection. Reusing K as V is less
  paper-faithful; adding V changes checkpoint compatibility.

### Rank 3: ICML 2025 SpikeVideoFormer Hamming Attention

Paper:

- `SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming
  Attention and O(T) Complexity`, ICML 2025.

Official code:

- `ICML2025_SpikeVideoFormer/classification/models/metaspikformer.py`

Relevant source pattern:

```text
if sim_mode == "hamming":
    x = (2 * k - 1).transpose(-2, -1) @ v
    x = (2 * q - 1) @ x
    x = x / (2 * dim)
```

Why it matters:

- It is recent, top venue, video-oriented, and explicitly uses Hamming
  attention.
- It treats temporal-spatial tokens jointly (`T*H*W`), which is closer to event
  optical flow than static SNN classification.

Fit to SDformerFlow:

High, but with a caveat: their formula assumes binary spikes. We should adapt
to ternary signs:

```text
q_h = sign(q)      # {-1,0,+1}
k_h = sign(k)
score/channel_mix = hamming-style signed agreement
value = K or learned V later
```

First experiment:

- **H21a SDformer Hamming-QK**
- Use Hamming-style `2*event-1` only after converting silence carefully.
- For ternary, avoid mapping zero directly to `-1`; instead test:

```text
event = q_sign != 0
polarity = q_sign
```

and compare spike-active Hamming vs signed Hamming.

Expected story:

Hamming similarity is a stronger top-venue alternative to our current signed
consensus, with a video/event rationale.

Risk:

- Naively mapping ternary zero to `-1` can make silence dominate. This is the
  same failure mode as α-XNOR if silence reward is too high.

### Rank 4: ICLR 2025 Saccadic Spike Self-Attention

Paper:

- `Spiking Vision Transformer with Saccadic Attention`, ICLR 2025.

Core mechanism:

- Spatial relevance is based on spike distribution rather than dot product.
- Temporal saccadic interaction dynamically focuses on selected visual areas at
  each timestep.
- Reported as linear-complexity SNN-ViT attention.

Fit to SDformerFlow:

Very high. SDformerFlow has explicit temporal event windows, and optical flow is
more temporal than static classification.

First experiment:

- **H19a Saccadic temporal gate**
- Do not rewrite the whole attention at first.
- Add a timestep-wise gate over the existing Q/K path:

```text
time_score[t] = distribution_relevance(Q_t, K_t)
time_gate = sparse/sigmoid or L1-normalized temporal weights
K_t = K_t * time_gate[t]
```

Second experiment:

- **H19b SSSA distribution relevance**
- Replace QK score with spike distribution cross-entropy/relevance.

Expected story:

Saccadic focus is especially natural for event optical flow: not every timestep
contains equally useful motion evidence.

Risk:

- No local official code was found during this pass. Implementation must follow
  the paper equations carefully, with small smoke tests before full training.

### Rank 5: CVPR 2025 STAA-SNN

Paper:

- `STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks`,
  CVPR 2025.

Core mechanism:

- Spike-driven self-attention for SNNs.
- Positional encoding for latent temporal relationships.
- Step attention to selectively amplify features at different timesteps.
- Time-step random dropout.

Fit to SDformerFlow:

High for two reasons:

- SDformerFlow already processes event time bins.
- Step attention can be added around attention outputs without changing Q/K
  projections.

First experiment:

- **H20a Step Attention Gate**
- Add a lightweight learned or spike-derived gate over the temporal dimension
  after Q/K attention but before projection:

```text
attn_out: [T, B, H, W, C]
step_gate: [T, B, 1, 1, C] or [T, B, 1, 1, 1]
out = attn_out * step_gate
```

Second experiment:

- **H20b Time-Step Random Dropout**
- During training only, randomly drop/reweight timesteps in the sparse attention
  path to reduce overfitting to a few bins.

Expected story:

The improvement is temporal aggregation, not just a different QK score.

Risk:

- Need to locate or verify official code. The current cloned candidate is a
  related SCTFA repository, not confirmed as STAA-SNN.

### Rank 6: IJCAI 2024 TIM Temporal Interaction Module

Paper:

- `TIM: An Efficient Temporal Interaction Module for Spiking Transformer`,
  IJCAI 2024.

Core mechanism:

- A lightweight convolution-based temporal interaction module.
- It is designed to plug into existing spiking transformer attention and improve
  temporal information use with minimal parameters.

Fit to SDformerFlow:

Very high as a low-risk temporal plugin because SDformerFlow has fixed 10-bin
event tensors and current H13/H14 attention mostly treats time as reshape
structure rather than explicit temporal memory.

First experiment:

- **H20c TIM-QK temporal prefilter**
- Add a depthwise temporal convolution on Q/K or attention output:

```text
Q_tilde = Q + DWConv1d_time(Q)
K_tilde = K + DWConv1d_time(K)
```

Expected story:

TIM is a minimal temporal interaction add-on, safer than full saccadic attention
or full STAtten replacement.

Risk:

- Not CVPR/ICLR/NeurIPS/ICML, but IJCAI main track and highly relevant.

### Rank 7: ICML 2025 TTFSFormer

Paper:

- `TTFSFormer: A TTFS-based Lossless Conversion of Spiking Transformer`,
  ICML 2025.

Core mechanism:

- Time-to-first-spike coding for Transformer conversion.
- Each neuron can spike at most once, reducing temporal energy.
- Addresses attention and nonlinear layers in TTFS conversion.

Fit to SDformerFlow:

Medium. SDformerFlow is directly trained and event-flow-specific, so full
conversion is not the right move. But TTFS gives a useful time gate:

```text
first_spike_time(Q/K) -> reliability or priority gate
```

First experiment:

- **H20d TTFS reliability gate**
- Use earlier/later first-spike timing as a temporal reliability score for the
  auxiliary gate, without changing baseline carrier.

Risk:

- TTFS assumptions may conflict with optical-flow event bins where later motion
  evidence can be more useful than first events.

### Rank 8: NeurIPS 2024 SEMM / EMSA

Paper:

- `Spiking Transformer with Experts Mixture`, NeurIPS 2024.

Core mechanism:

- Replaces dense MoE routing with spiking router sequences.
- EMSA performs head-wise spiking expert allocation.
- EMSP performs channel-wise spiking expert allocation.

Fit to SDformerFlow:

Medium. It is top-venue and current, but it is more of a conditional-computation
architecture than a QK similarity fix.

First experiment:

- **H22a Head-wise sparse attention expert gate**
- Keep our current attention variants as experts:
  `compat`, `signed_consensus`, `α-XNOR`, maybe `Hamming`.
- A simple spiking router selects/weights heads or blocks.

Expected story:

Dynamic sparse conditional attention: not every block/head needs the same
attention rule.

Risk:

- More moving parts, harder to debug than α-XNOR/Hamming/step attention.

### Rank 9: QSD-Transformer / QP-SNN Analysis Track

Papers:

- `QSD-Transformer`, ICLR 2025.
- `QP-SNN: Quantized and Pruned Spiking Neural Networks`, ICLR 2025.

Core mechanisms:

- QSD diagnoses spike information distortion and uses information-enhanced LIF
  plus fine-grained distillation.
- QP-SNN combines quantization with structured pruning and uses singular values
  of spatiotemporal spike activities as a pruning criterion.

Fit to SDformerFlow:

These are not first attention replacements, but they are important for deciding
where attention modifications hurt:

```text
SID-like profile: compare baseline PSN Q/K distribution vs sparse ternary Q/K.
SVS-like score: rank Q/K, FFN, and downsample modules by spatiotemporal spike singular values.
```

First experiment:

- **H17a SID/SVS profiler**
- No training. Profile H9a, H13n, H14, H18/H21 checkpoints.

Risk:

- More analysis than immediate metric gain, but it can prevent random
  experiment drift.

### Rank 10: AAAI 2026 Spikingformer

Paper:

- `Spikingformer: A Key Foundation Model for Spiking Neural Networks`, AAAI
  2026.

Core mechanism:

- Merges MS residual connection with self-attention to remove non-spike
  residual computations while preserving global modeling.

Fit to SDformerFlow:

Medium-low for attention replacement, medium for hardware story.

Experiment:

- **H23a spike-driven residual audit**
- Audit whether our attention overlay introduces non-spike residual paths.
- Optionally gate residual additions with spike-compatible scaling.

Expected story:

This is a hardware-cleanliness improvement, not the main attention innovation.

### Rank 11: AAAI 2025 Spike2Former

Paper:

- `Spike2Former: Efficient Spiking Transformer for High-performance Image
  Segmentation`, AAAI 2025.

Core mechanisms:

- Spike-driven deformable transformer encoder.
- Spike-driven mask embedding.
- Normalized integer LIF for stable training in complex architectures.

Fit to SDformerFlow:

Medium. It is less about QK attention and more about complex dense prediction
architecture stability. This matters because optical flow is also dense
prediction.

Experiment:

- **H23b NI-LIF style attention-output stabilizer**
- Borrow normalized integer activation only around attention/projection outputs
  if H18/H21 cause unstable magnitude.

Risk:

- Task/domain mismatch: segmentation masks vs event optical flow.

### Rank 12: CVPR 2024 SpikingResformer

Paper:

- `SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural
  Networks`, CVPR 2024.

Core mechanism:

- Dual Spike Self-Attention and residual-convolution integration.

Fit to SDformerFlow:

Medium-low as a new experiment because it is older and less event-flow-specific,
but it is a useful reference for not over-replacing the transformer carrier.

Experiment:

- Use as related-work/context, not priority code.

### Rank 13: ECCV 2024 Spiking Wavelet Transformer

Paper:

- `Spiking Wavelet Transformer`, ECCV 2024.

Core mechanism:

- Attention-free Frequency-Aware Token Mixer.
- Sparse wavelet transform plus negative spike dynamics.

Fit to SDformerFlow:

Useful but not an attention paper. It is highly relevant to event optical flow
because motion edges/high-frequency changes matter.

Experiment:

- **H24a wavelet FFN/token-mixer branch**
- Add a small frequency branch in high-SOP FFN blocks or as an attention
  alternative.

Expected story:

If attention changes fail, SWformer gives a top-venue attention-free route:
event flow may need frequency/motion-edge modeling more than global attention.

## Recommended Next Experiments

Do not start with SDSA. The next attention series should be:

| Priority | Experiment | Paper basis | Why first |
|---:|---|---|---|
| 1 | H18a ternary α-XNOR attention | CVPR 2025 α-XNOR | direct QK similarity fix, top venue, hardware-friendly |
| 2 | H18b A2OS2A-QK adapter | CVPR 2025 A2OS2A | hybrid binary/nonnegative/ternary attention, addition-only, already flagged by local H9d notes |
| 3 | H21a ternary Hamming-QK attention | ICML 2025 SpikeVideoFormer | video/event attention, official code available |
| 4 | H20a temporal step attention gate | CVPR 2025 STAA-SNN | temporal aggregation, low integration risk |
| 5 | H19a saccadic temporal gate | ICLR 2025 SSSA | strongest event-flow intuition, but no local code |
| 6 | H22a head-wise attention experts | NeurIPS 2024 SEMM/EMSA | good paper story, higher implementation risk |

## Concrete Integration Plan

### H18a: Ternary α-XNOR Matrix Attention

Use this first because it most directly fixes Q/K similarity.

Formula for ternary ATLIF:

```text
q = sign(Q), k = sign(K)
same_pos = (q == +1 and k == +1)
same_neg = (q == -1 and k == -1)
same_zero = (q == 0 and k == 0)
opposite = (q == -k and both nonzero)

score = same_pos + same_neg + alpha0 * same_zero - beta * opposite
```

Start with:

```text
alpha0 = 0.05
beta = 0.5
normalization = Shiftmax
value = threshold-scaled K
scope = H13n/H14 attention scope
```

Then ablate:

```text
alpha0 in {0.0, 0.02, 0.05, 0.1}
beta in {0.0, 0.25, 0.5, 1.0}
normalization in {Shiftmax, L1, pow2}
```

### H21a: Ternary Hamming-QK

Use SpikeVideoFormer's Hamming logic, but do not map silence to `-1` blindly.

Variants:

```text
H21a-active: score counts active agreement/disagreement only.
H21b-alpha-silent: score also gives small alpha to same-zero.
H21c-polarity-only: score uses sign agreement and ignores magnitude.
```

### H20a: Step Attention Gate

Add temporal step attention around the existing H13/H18/H21 attention output:

```text
gate_t = sigmoid(MLP(global_pool(attn_out_t)))
out_t = attn_out_t * gate_t
```

Keep it training-only light at first:

```text
gate hidden dim small
no new V projection
no full architecture rewrite
```

### H19a: Saccadic Temporal Gate

Implement after H20a because the paper mechanism is stronger but more complex.

```text
top-k or sparse temporal focus over event bins
distribution-based Q/K relevance
use sparse gate to mask K/value path
```

## Decision

H15/H16 from the previous note are demoted:

- STAtten remains relevant, but H20/H19 are better recent top-venue temporal
  attention paths.
- SDSA/H16 is too old to be a main contribution.
- QSD remains an analysis/distillation tool, not the first attention mechanism.

The next code work should implement H18a and H21a as modular attention modes
inside the experiment overlay, with guard configs only. Full training should be
promoted only if valid40 AEE/AAE and SOPs beat or approach H9a/H13n.

After merging the local backlog, H18b A2OS2A should be implemented immediately
after H18a because it is a strong CVPR 2025 paper-backed hybrid binary/ReLU/
ternary attention baseline.
