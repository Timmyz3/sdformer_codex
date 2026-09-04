# SNN Attention Autoresearch Notes

Date: 2026-05-19

Update: this note is superseded for experiment prioritization by
`RECENT_TOPVENUE_SNN_ATTENTION_AUTORESEARCH.md`. SDSA-style attention is useful
historically, but it is no longer a priority mechanism because the project needs
recent top-venue attention ideas from 2024-2026.

## Question

How should SDformerFlow attention be improved by borrowing real SNN attention
mechanisms from recent papers, instead of only tuning the current H13/H14
variants?

## Local Inventory

Existing local attention research:

| Location | Content | Status |
|---|---|---|
| `neuron_autoresearch/ATTENTION_DESIGN_SPACE.md` | broad ternary-native attention brainstorm | useful, but mostly self-designed |
| `neuron_autoresearch/ATTENTION_AAE_RESEARCH.md` | AAE protection and ternary attention ideas | useful for constraints |
| `neuron_autoresearch/attention/soc_attention.py` | sign-only consensus gate | implemented as probe, not paper-faithful |
| `neuron_experiments/H9_bipolar_self_attention/overlay/.../bsa_attention.py` | compat Shiftmax, signed consensus, strict BSA, ShiftNorm, popcount L1 | active experiment code |
| `neuron_experiments/H9_bipolar_self_attention/ideas/ternary_native_gate.py` | hardware-friendly ternary gate sketches | discussion-only |

Source code collected under:

| Source | Local path |
|---|---|
| BSA paper PDF | `/root/private_data/work/optimization_sources/attention_optimization/BSA_BipolarSelfAttention/` |
| STAtten CVPR 2025 | `/root/private_data/work/optimization_sources/attention_optimization/CVPR2025_STAtten/` |
| Spike-driven Transformer V2 | `/root/private_data/work/optimization_sources/attention_optimization/ICLR2024_Spike-Driven-Transformer-V2/` |
| QSD-Transformer ICLR 2025 | `/root/private_data/work/optimization_sources/attention_optimization/ICLR2025_QSD-Transformer/` |

## Paper Mechanisms Worth Migrating

### 1. BSA: Bipolar Self-attention

Reference idea:

```text
Q, K become ternary spike events.
Score = ternary matrix product Q @ K^T.
Gate = Shiftmax(Score).
Output = Gate @ V.
```

Why it matters:

- Fixes binary SSA's polarity blind spot.
- Restores row-stochastic attention behavior with Shiftmax.
- Closest paper match for our PSN+ATLIF+ternary story.

Fit to SDformerFlow:

- Current block has Q and K but no independent V.
- H14 already implements the first faithful adaptation by reusing K as V:

```text
Qe = sign(Q)
Ke = sign(K)
Score = Qe @ Ke^T
Gate = Shiftmax(Score)
Output = Gate @ K
```

Experiment status:

- H14a/b/c configs exist and are queued after H13n.
- This is still the most important paper-faithful baseline.

Risk:

- Full matrix attention changes SDformer/QKFormer's original linear-token gate
  into an `N x N` relation. Accuracy may improve, but memory and runtime can
  worsen.

### 2. STAtten: Spatial-Temporal Attention

Official implementation:

`CVPR2025_STAtten/module/ms_conv.py`

Core code pattern:

```text
q, k, v: [T, B, heads, N, D]
split T into chunks
merge each chunk into [chunk_size * N] tokens
attn = K^T @ V
out = Q @ attn
```

Important paper claim:

- Existing SNN transformers mostly model spatial attention and underuse the
  temporal dimension.
- STAtten integrates temporal context by block-wise temporal chunks while
  keeping the same order of computational complexity as spike-driven attention.

Fit to SDformerFlow:

- Strong fit because SDformerFlow has explicit time dimension `T`.
- Current QK attention already folds time/spatial tokens in some paths, but it
  does not deliberately compute temporal chunk interactions.
- We can adapt STAtten in two levels:

H15a, low-risk:

```text
Use existing Q and K.
Set V = K because SDformer has no V projection.
For each temporal chunk:
    A = K_chunk^T @ K_chunk
    out = Q_chunk @ A
Apply existing attn_sn/proj path.
```

H15b, paper-faithful:

```text
Add experiment-local V projection cloned from K projection shape.
Use official STAtten q/k/v formula.
```

Recommendation:

- Start with H15a guard because it preserves SDformer module shape.
- Only try H15b if H15a is promising; adding V changes checkpoint loading and
  may require careful initialization.

### 3. Spike-driven Transformer V2 / SDSA

Official implementation:

`ICLR2024_Spike-Driven-Transformer-V2/classification/models.py`

Core code pattern:

```text
q = LIF(q_conv(x))
k = LIF(k_conv(x))
v = LIF(v_conv(x))
x = k.transpose(-2, -1) @ v
x = q @ x
x = attn_lif(x)
```

Why it matters:

- No Softmax.
- Spike-driven, multiplication order is `Q @ (K^T @ V)`, avoiding direct
  token-token Softmax attention.
- Very close to STAtten's computational form.

Fit to SDformerFlow:

- Similar issue: SDformer has no independent V.
- Low-risk adaptation can set `V=K`.
- This gives a clean H16 baseline:

```text
A = K^T @ K
out = Q @ A
```

This differs from H14:

| Method | Relation |
|---|---|
| H14 strict BSA | token-token: `Q @ K^T`, then `@ K` |
| H16 SDSA-KV | channel-channel: `K^T @ K`, then `Q @` |

H16 is likely cheaper and more compatible with SDformerFlow's existing QK block.

### 4. QSD-Transformer / SID

Official implementation:

`ICLR2025_QSD-Transformer/classification/models.py`

Core mechanism:

- QSD keeps the SDSA attention structure.
- It introduces multi-bit spikes during training with `Multispike`.
- The paper diagnoses spike information distortion (SID) in quantized
  spike-driven self-attention.
- It uses information-enhanced LIF and fine-grained distillation to align
  attention distributions.

Fit to current project:

- This is not the best first attention replacement.
- It is useful as an analysis/training layer after H14/H15/H16:

```text
measure attention distribution distortion between baseline and sparse attention
add a light distillation loss if AAE drifts
try multi-bit attention spikes only for attention, not FFN
```

Recommendation:

- Do not run QSD as first attention experiment.
- Use its SID idea to explain and measure why AAE changes when attention is
  modified.

## Proposed New Experiment Series

### H15: STAtten-Inspired Temporal Chunk Attention

Goal:

Use STAtten's paper mechanism to make attention explicitly temporal.

Variants:

| Run | Change | Risk | Priority |
|---|---|---|---|
| H15a | STAtten with `V=K`, chunk size 2 | low | first |
| H15b | add V projection, official q/k/v form | medium | only if H15a works |
| H15c | H15a plus ternary sign events in score path | medium | after H15a |

Prediction:

- Should help AAE because event flow is temporal.
- SOPs may increase unless chunking is kept small.

### H16: SDSA-KV Attention

Goal:

Port Spike-driven Transformer V2's SDSA computation into SDformerFlow.

Formula:

```text
A = K^T @ K
out = Q @ A
```

Variants:

| Run | Change | Risk | Priority |
|---|---|---|---|
| H16a | threshold-scaled Q/K | low | first |
| H16b | sign-only Q/K | medium | if H16a dense/expensive |
| H16c | ternary Q/K with post-attention binary/ternary spike | medium | later |

Prediction:

- More compatible with SDformerFlow than strict token-token BSA.
- Hardware story is strong: no Softmax/Shiftmax, channel-channel accumulation.

### H17: SID / Distortion Analysis

Goal:

Use QSD's SID framing to explain AAE and attention failures.

Measurements:

```text
attention entropy
attention distribution KL/MSE vs baseline
Q/K firing distribution
positive/negative ternary balance
per-stage distortion vs AEE/AAE
```

This should be a profiling script before it becomes a training loss.

## Queue Recommendation

Keep current active H13n run and queued H14 guards. Then prioritize the newer
top-venue series from `RECENT_TOPVENUE_SNN_ATTENTION_AUTORESEARCH.md`:

1. H18a guard: ternary α-XNOR attention, based on CVPR 2025 α-XNOR SSA.
2. H21a guard: ternary Hamming-QK attention, based on ICML 2025 SpikeVideoFormer.
3. H20a guard: temporal step-attention gate, based on CVPR 2025 STAA-SNN.
4. H19a guard: saccadic temporal gate, based on ICLR 2025 SSSA.
5. H17 profiler on H9a, H13n, H14/H18/H21/H20 checkpoints.

Do not prioritize angular loss unless the attention mechanism is already stable.
Previous H9a+i14 results suggest angular loss can conflict with compat QK gating.

## Implementation Boundary

- Keep all code under experiment-local overlays.
- Do not modify `third_party/SDformerFlow`.
- Source repositories stay under `/root/private_data/work/optimization_sources/attention_optimization/`.
- New trainable parameters, if any, must live only in H15b+ experiment modules.
