# Orchestra Phase 1 — Diverge

Skill: `brainstorming-research-ideas` (Orchestra Research).  
Starting point (grill-me, locked): H67/ep34 event-camera SNN Transformer optical flow; one TCAS-II 5-page co-design object; algorithm may move; spike-driven Q-K required; valid825 AEE ≤ 1.259 and better than SDformerFlow PSN 1.5848; TSMC 28HPC+; ZCU102 optional; 1RW not required; no kill-list.

**Goal of this phase (skill text):** produce 10–20 candidate ideas *without filtering*. Filtering is Phase 2.

**Deviation (recorded, not hidden):** this round is a structured re-generation from the six required lenses. The 32 mechanism cards in `../README.md` already existed (grill + S1 screen). Those cards are *inputs to the lenses*, not a substitute for the lenses. Candidates below are labeled `R01`–`R20`. Mapping to cards/objects is provenance, not a score.

**Examples shown before this round (anchoring risk):** grill-me contract; Motion C12 ep34 AEE 1.199514; old C1/C2 RTL; ep35 QK census; Prosperity/LoAS/FireFly-T names. Skill F1–F10 walkthrough is therefore *not* a blank-page session.

---

## F1 — Problem-first vs solution-first (session mode)

One-sentence current idea: *Make a TCAS-II brief whose circuit face is the same object as a spike-driven event-OF algorithm change.*

Classification: **both modes are live**.

- Problem-first: the H67 Motion-XOR score has no published ASIC; existing accelerators implement AND-PopCount or linear Q-K, not temporal-peer XOR. Who suffers: the authors of this brief, who otherwise have only a Prosperity/LoAS copy. Community: event-OF on-chip designers who cannot reuse FireFly-T as-is.
- Solution-first: leftover C1/C2 RTL (product forest, temporal shared-sum, late-BN packet) is looking for a problem that still exists under the new contract.

Gap that cannot be done today: a 28 nm digital island whose *reason for existing* is the Motion-XOR temporal term, with a skip rule that is true at the actual Shiftmax grain — **or**, if that leaf is <~2% of cycles, a mixed-T FC island whose difference from LoAS is the T=2/T=10 split.

Self-check:

- [x] Named community: TCAS-II circuits readers + this project's authors.
- [x] Unsolved, not just under-marketed: no located Motion-XOR temporal-peer popcount ASIC (bounded search; not “never studied”).
- [x] Solution-first leftovers (C1/C2) must name two problems they still address: (1) Conv/FC share of the envelope; (2) mixed-horizon packing LoAS does not publish.

---

## Step 1 — Scan for tensions (F3)

Desiderata everyone wants: DSEC AEE, spike-driven Q-K identity, 28 nm PPA, system FPS, “looks new to a circuits editor”.

| # | Tension pair | Artifact or fundamental? | Research opportunity |
|---|---|---|---|
| T1 | AEE ↔ hardware-friendly sparsity | Artifact of PAFT-ep4 (AEE ~1.47) under a now-lifted freeze | Retry sparsity only if ep34 retraining clears 1.259 |
| T2 | System speedup ↔ mechanism novelty | Unknown until T0; historically attention ~0.59% | Leaf paper vs system paper is a *measurement*, not a preference |
| T3 | Copy-complete prior ↔ “looks new” | Social/review, not physics | Copy is a **baseline**, not a title |
| T4 | Keep Motion-XOR ↔ replace with public SDSA/QKFormer | Empirical | 10-frame overlay is the cheap discriminator |
| T5 | Token skip ↔ window-level Shiftmax correctness | Mechanistic; ep35 preview says window-clean 9.9% | Token enable is legal; window memo is not, unless M2 says otherwise |

Papers that optimize each side independently (located, not exhaustive): FireFly-T / QKFormer (simple Q-K hardware); Prosperity / LoAS (system MAC reuse); SDformerFlow (public OF accuracy without Motion-XOR); Zhang CICC 2026 (temporal skip on feature maps, not scores).

---

## Step 2 — What changed (F5)

Abandoned or previously-killed assumptions, re-tested under 2026-09-07 contract:

1. **Compute / training:** algorithm identity unfrozen; one A800 allowed. PAFT/kernel-swap no longer auto-killed by “cannot retrain”.
2. **Tooling / SRAM:** 1RW not required; TSMC 28 dual-port compilers exist. Dual-port only matters if the stall is memory, not compute (A1 official model mem stall ≈ 0).
3. **Failure of “no novelty”:** old C1/C2 judged unoriginal *as titles*. New contract allows remeasurement; it does **not** restore title status.

Frame: “X was impractical because Y, but Z changed” applies to *retrying measurements*, not to claiming the old title.

---

## Step 3 — Probe boundaries (F6)

Two internally strong methods, and where they break:

**Prosperity full chain (A1).**  
Implicit eval assumption: product/bit speedup is the paper.  
Violate: require *title novelty after naming Prosperity HPCA 2025*. Breaks. Product/bit 2.33× still true as a comparator. Dual-port does not rescue a compute-bound outer product.

**Time-dirty skip sold as Shiftmax skip (B2).**  
Implicit eval assumption: token-equal scores ⇒ skip the row.  
Violate: denominator is a 15×15 (or 450) window. ep35 preview: window OR-dirty 90.1%, S3 clean 0%. Breaks as a lossless row memo. Survives as token ALU/energy gating.

Root cause of the second failure: grain mismatch, not “skip is fake”. Constructive path: enable the leaf per token; do not memoize Shiftmax rows unless M2 shows a cleaner grain.

---

## Step 4 — Cross-pollinate (F4)

Domain-agnostic problem: *do not recompute a three-term popcount when the two binary vectors did not change, and do not skip a reduction whose support is larger than the unchanged set.*

Adjacent fields:

- Computer vision / graphics: DeltaCNN, CBinfer — skip unchanged pixels. Structural map: token/lane, not RGB pixel. Testable prediction: if Q and K bits equal t0, the three popcounts equal the t0 values bit-exactly.
- Circuits: Zhang CICC 2026 event-OF DLSS — temporal similarity skip on **feature maps**. Structural map is *not* Motion-XOR. Testable prediction: encoder activation skip rate ≠ attention token skip rate.
- Neuroscience-inspired SNN: SpiLiFormer (ICCV 2025, arXiv:2503.15986) — lateral inhibition to suppress irrelevant tokens. Structural map: training on co-silence (`same_zero`), not a new ALU.

Validity checks: DeltaCNN mapping has structural fidelity at “skip unchanged”; SpiLiFormer is a training analogy (weaker fidelity to the ALU); CICC DLSS is a sideways analog, easy to over-claim.

---

## Step 5 — Compose / decompose (F9)

Components in this area: Motion-XOR score leaf, Q/K projection, Shiftmax, ATLIF membrane, BN/PSN, bottleneck conv, FC1/FC2, ConvTranspose decoder, T=2 attention fibers, T=10 neuron fibers.

**Compose:** Motion-XOR 32-lane ALU + token enable (same leaf). Emergent capability: a skip that is honest at token grain and still uses the temporal XOR datapath.

**Compose:** LoAS-style inner-join + mixed T=2/T=10 fiber split. Emergent capability: the *difference from LoAS* is the packing, not the join.

**Decompose:** “attention” ≠ score leaf. T0 must split score / proj / Shiftmax. If proj dominates, I001 is the wrong circuit face.

**Decompose:** “FC speedup” ≠ add reduction. Shared-sum without the Y port is not a cycle.

---

## Step 6 — Abstraction ladder (F2)

Current sentence: *Give the H67 spike OF Transformer a 28 nm circuit brief.*

| Direction | Statement | Publishable on its own? |
|---|---|---|
| Up | Which skip rules on sparse event tensors remain bit-exact on foundry 1RW/2RW SRAM | Framework; too big for TCAS-II 5 pages |
| Down | 32-lane triple-popcount with `K_peer` as the t0 shadow | Yes, as a leaf brief **if** T0 allows a leaf |
| Side | Temporal-similarity skip on encoder feature maps (CICC DLSS family) vs score-lane skip | Yes as a negative/ablation; easy to confuse with I001 |

For every raw candidate below, up/down/side is one line in the table.

---

## Raw candidates (20, unfiltered)

Generated from the six steps. Kill/keep is Phase 2.

| ID | One sentence | Lens | Up / down / side | Maps to |
|---|---|---|---|---|
| R01 | 32-lane Motion-XOR triple-popcount leaf with `K_peer` t0 shadow | F3-T2, F2-down | Up: skip theory; down: this leaf; side: FireFly-T AND-PopCount | O1 / B1 / I001 |
| R02 | Token-level leaf enable when K is zero both T or Q/K clean vs t0 | F3-T5, F4-DeltaCNN, F6 | Up: lossless skip calculus; down: per-token enable; side: window memo | O2 / B2-B3 / I002 |
| R03 | Replace one stage-2 block with public SDSA or QKFormer linear Q-K | F3-T4 | Up: which Q-K terms OF needs; down: one block overlay; side: classification QKFormer | O3 / B4 E1 E2 / I003 |
| R04 | Mixed-horizon T=2/T=10 FC join; LoAS FTP is the baseline not the title | F3-T2, F5, F9 | Up: mixed-T packing; down: one FC2 capture; side: uniform-T LoAS | O4 / A6 D1 / I004 |
| R05 | TSBG B8 same-IO weight-row broadcast as a **comparator** | F5 retry | Up: broadcast family; down: B8 vs LRU; side: Eyeriss/Gustavson | O5 / A4 / I005 |
| R06 | Copy Prosperity product forest onto H67 conv (baseline, not title) | F3-T3, F6 | Up: parent-value reuse; down: 3000×6912×768 layer; side: SumMerge | O6 / A1 / I006 |
| R07 | Keep ATLIF membrane in a mixed-T island; late-BN θ packet at the boundary | F9 decompose | Up: neuron/BN contract; down: existing A8 RTL + real BN; side: Gist | O7 / A8 C1 / I007 |
| R08 | Sparse ConvTranspose decoder island with a complete Table-A | F9 decompose | Up: sparse deconv; down: D0/D2/D3 rows; side: FireFly-S | O8 / D3 / I008 |
| R09 | Retry PAFT Hamming sparsity under AEE 1.259 on **ep34** | F5 | Up: train-for-hardware; down: running-BN valid825; side: Phi | A3 / I011 |
| R10 | Dual-port SRAM parent cache for C1 (1RW no longer required) | F5 | Up: port vs stall; down: TS1N28 2RW; side: still stall≈0 | A1 variant |
| R11 | FireFly-T-style byte SRAM permute for T×15×15 attention layout | F4 / literature reopen | Up: layout vs foundry macros; down: bitplane write; side: KV260 overlay | B11 / I009 |
| R12 | SpiLiFormer-style lateral inhibition training on Motion-XOR co-silence | F4 | Up: inhibit irrelevant tokens; down: `same_zero` regularizer; side: ICCV 2025 classification | B12 / I010 |
| R13 | Hardware-foldable running-BN / LN-free inference | F5 / F7 | Up: remove BN from the island; down: ep34 running vs frozen; side: NF-SpikingVTG | C5 / I011 |
| R14 | Independent skip of overlap / same_zero / motion_xor **terms** when that popcount is known-zero | F9 decompose, T1 census | Up: term-wise skip; down: overlap almost always empty (mean 0.013); side: not window skip | new I012 |
| R15 | Encoder feature-map temporal skip (CICC DLSS analog), not score skip | F2 side, F4 | Up: temporal similarity; down: one conv activation; side: Zhang 28 nm OF | new I013 |
| R16 | Q/K projection island, if T0 shows proj ≫ score leaf | F9 decompose | Up: linear spike-weight; down: one proj layer; side: FireFly-T sparse decode | unassigned until T0 |
| R17 | Shiftmax-row alignment experiment (measurement, not a paper) | F6 | Down: M2 grain; side: softmax accelerators | M2 |
| R18 | Window 15×15 / T_w=2 retuned to 8×8 or T_w=4 | F2 down | Up: geometry vs hardware; down: retrain from SDformerFlow; side: Swin window papers | E4 |
| R19 | SLI neighborhood path + SSA when Motion-XOR is sparse | F4 / F9 | Up: local motion complement; down: one depthwise bypass; side: arXiv:2608.19238 | B6 |
| R20 | Parent-value in-register promotion (+1.58% adds) retry under new layout | F5 | Down: already measured; side: Prosperity parent | A2 |

No filtering in this file. Rejected-later items stay on the list so they can recombine.

---

## F7 preview (not a kill yet)

Simplest cores:

- Motion-XOR reduced to AND-PopCount (FireFly-T) — that *is* I003’s hardware limit.
- Skip reduced to “K=0 skip” without dirty-vs-t0 — cheaper, weaker.
- FC reduced to “copy LoAS” — I004 without mixed-T is empty.

---

## Phase 1 output

20 raw ideas. Next: Phase 2 filters (F10, F1, F7, F8, feasibility) → 3–5. The 32 cards and 8 objects remain in `../README.md` and `../CODESIGN_OBJECTS.md`.
