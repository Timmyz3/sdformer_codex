# Hardware Transfer Audit for Unified TTX (2026-07-12)

## Scope and evidence rule

This note asks whether an architecture-paper mechanism transfers to the frozen DSEC TTX datapath. A citation is not evidence of savings in this model. A mechanism is promoted only if its source workload assumptions match binary Q/K, two timesteps, 162 tokens per Swin window and 32 lanes per head, or if a new profile/RTL experiment validates the mismatch.

## 1. LoAS (MICRO 2024): transfer the temporal layout, not dual-sparse weight claims

Source: *LoAS: Fully Temporal-Parallel Dataflow for Dual-Sparse Spiking Neural Networks*, arXiv:2407.14073 / MICRO 2024.

LoAS places timestep `t` in the innermost spatially unrolled loop. Its stated reason is concrete: placing `t` outside inner loops causes repeated weight-row fetches. It also uses spike compression designed for contiguous single-bit access and a cheaper inner-join circuit for dual-sparse spMspM.

TTX transfer:

- Applicable: temporal-pair co-residency. The measured TTX shape is `T=2`, `tokens=162=2x81`, `head_dim=32`. Therefore Q0/Q1 for one spatial token/head fit exactly in one 64-bit word; K0/K1 fit another. A single fetch exposes both ordinary TTX operands and temporal XOR/delta metadata.
- Applicable: schedule the two timesteps together so Q/K weights and positional state are not refetched across `t`.
- Not yet applicable: LoAS's dual-sparse weight inner join. Current SDformer weights have not been structurally pruned and no bitmap/CSR weight format has been frozen.
- Claim boundary: cite LoAS for temporal-loop placement and packed single-bit access; do not reuse its speedup or energy numbers for TTX.

Classification: **A-class exact hardware optimization**, compatible with H60/H67/H69/H70 and the pairwise candidates. It requires a cycle/traffic model but no retraining.

## 2. Bishop (ISCA 2025): profile bundle sparsity before instantiating two cores

Source: *Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning*, ISCA 2025.

Bishop's Token-Time Bundle, dense/sparse stratifier and error-constrained pruning are three separate ideas. For this project:

- TTB layout supports the same temporal-pair co-residency as above.
- Dense/sparse heterogeneous cores are not automatically justified. They duplicate area/control and only pay off if bundle density is strongly bimodal. The deferred Delta-locality v2 profile must report empty bundle4/8 ratios and changed-lane histograms first.
- ECP changes attention values unless an exact bound proves the deployment gate code unchanged. It therefore belongs to the software/B-class line and needs valid825, not just an RTL estimate.

Classification: TTB is A-class; heterogeneous routing is conditional A-class; ECP is B-class.

## 3. PADE (2025 preprint): mostly a mismatch for one-bit TTX

Source: *PADE: A Predictor-Free Sparse Attention Accelerator via Unified Execution and Stage Fusion*, arXiv:2512.14322.

PADE progressively reads INT8 K bit planes, computes uncertainty intervals for unfinished dot products, and prunes keys against a threshold. Its out-of-order scoreboard hides irregular bit-plane memory latency. The paper itself reports nontrivial overheads and that theoretical compute reduction converts to a much smaller practical efficiency gain.

TTX transfer verdict:

- H60 Q/K are already one bit, so there is no lower bit plane to defer. Direct BUI-GF/bit-serial transfer is invalid.
- H66a matrix attention may use partial accumulation over 32 binary lanes, but pruning a key changes row Shiftmax unless a deployment-code bound is proved. This is only relevant if H66a wins full30 accuracy and remains a hardware candidate.
- The general lesson is retained: predictor and scoreboard power, imbalance and memory layout must be charged against sparsity savings.

Classification: **not an H60 optimization**; conditional B-class support for H66a only.

## 4. ICCAD 2024 3D spiking-transformer accelerator

Source: *Spiking Transformer Hardware Accelerators in 3D Integration*, ICCAD 2024.

The paper maps spiking attention and synaptic integration onto memory-on-logic and logic-on-logic face-to-face stacks, reporting large memory-access latency/power reductions relative to its own 2D implementation. This validates that attention data movement can dominate and that physical integration matters.

Transfer verdict: useful for PPA discussion and a future physical-design option, but it is a technology/mapping choice, not a new TTX algorithm. Without a 3D PDK, bonding assumptions and matched 2D/3D flow, this project must not quote its savings as achieved results.

Classification: discussion/future-work evidence only.

## 5. Hardware line after deep read

| Priority | Mechanism | Exact? | Required evidence | Decision |
|---:|---|---|---|---|
| H0 | 64-bit temporal-pair Q/K co-residency | yes | traffic/cycle model and RTL address trace | proceed |
| H0 | Exact Delta-TTX changed-lane update | yes for alpha0=1/64 | locality v2 raw counts and RTL equivalence | proceed |
| H1 | bundle4/8 zero-activity clock gating | yes | empty-bundle profile, ICG coverage and scheduler cost | profile first |
| H1 | dense/sparse dual path | yes | density bimodality plus area/power crossover | do not instantiate yet |
| H2 | error-bounded score/gate pruning | no unless code-equivalent bound | valid825 plus error proof | software/B-class |
| H2 | PADE-style partial key pruning | no for H60 | H66a win and row-Shiftmax bound | conditional only |
| H3 | 3D memory-on-logic | mapping choice | 3D PDK and matched physical flow | discussion only |

The hardware story should therefore lead with **one data layout and one exact sparse update mechanism**, not a collection of unrelated accelerators: temporal-pair packed TTX supplies contiguous operands; Delta-TTX suppresses unchanged lanes; bundle gating suppresses fully inactive groups if the measured locality supports it.
