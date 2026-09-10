# 02 — Ranked new mechanisms

Scoring: novelty vs priors × fit to frozen ep34 (binary ATLIF, Motion-XOR, mixed T, event OF) × 28 nm 1RW implementability before ISCAS 13 Oct 2026.

**Rank 1 and Rank 4 are one island.** Do not split them into two contribution bullets.

## Rank 1 — Motion-XOR triple-popcount score ALU (MX3P)

**One sentence:** Combinational datapath computes `pop(Q∧K)`, `pop(K⊕K_peer)`, `pop(co-silence)` and integer `round_even` into Q7. No multiplier in the score leaf. `attn = gate ⊙ K`.

**Steal:**
- FireFly-T AND-PopCount as the **QK term only** (baseline, not claim).
- α-XNOR SSA (CVPR 2025, Xiao et al.): spike–spike = 1, silence–silence = α∈(0,1). Map `α → same_zero/64`. α-XNOR tables use α∈[0.2,0.4] (paper tables 0.3/0.5), **never 0.125**. Our 0.125 is frozen H67 identity, not their accuracy peak.
- ITA (PULP): integer shift/round leaf for `round_even(128·(·)/32)`. Do **not** steal softmax; Motion-XOR has none.

**Modify:** Add temporal-peer **K port** (SDformerFlow never had it). Published SDformerFlow attention is `BN(Linear((Qs Ks^T+PE)Vs))` or SDSA `A_t = SN(sum_i Qs^{i,j})` then `z' = A_t ⊗ Ks`, **not** temporal XOR of tokens. Energy table is 45 nm theoretical MAC/AC, not SRAM island.

**Not a copy:** No surveyed ASIC maps Motion-XOR. α-XNOR is classification-only, no temporal-peer XOR, no RTL. FireFly-T is FPGA LUT6 QK^T. Comperity XOR is GeMM residual, not a score.

**Primitive:** 1RW Q/K tiles + K_peer shadow + three mask popcounts + shift-add Q7.

**Story:** First 28 nm 1RW Motion-XOR attention island for event-OF spikeformers (H67 operator, not SDformerFlow’s published SDSA).

## Rank 4 — Temporal-peer similarity / dirty-lane gate (same island as Rank 1)

**One sentence:** `dirty = OR(Q_t ⊕ Q_{t-1}, K_t ⊕ K_{t-1})`. If `pop(K ⊕ K_peer) ≈ 0` and Q also quiet, skip the three-popcount leaf and reuse t0 Q7 / Shiftmax class.

**Steal:** Zhang CICC 2026 DLSS: skip deep Hybrid U-Net levels when consecutive feature maps similar, `AS = 1-(1/N)sum|FM_i-FM_{i+δ}|`. MVSEC stream similarity >90% at δ>10. **Steal the temporal-similarity idea only.** Do not steal MaxPool/ReLU speculation (breaks Shiftmax) or BWAC (expands our INT8 weights).

**ANN analog:** CBinfer / DeltaCNN / Delta networks — recompute only changed activations. Retarget from RGB pixels to **attention score lanes**.

**Not empty-tile:** Q/K empty still produces silent/silent scores that enter Shiftmax. Profile text: only Delta score reuse and K-zero value gating have lossless skip proofs.

**Directional evidence (ep35 one-sample P0, not ep34 paper RTL):** TTX paired scores equal 97.79%; H67 97.54%; Q∨K update density 3.70%; t1 ideal lane skip 96.3%; K-motion zero 76%. **Must remeasure row-level dirty on frozen ep34.** If a single dirty lane forces whole-row Shiftmax denom, 97% token equality can collapse.

**Story:** OF temporal redundancy skip of attention **depth**, using K_peer already fetched for Motion-XOR. Spatial locality is taken by ASNA-Flow.

## Rank 2 — LoAS FTP inner-join remapped to mixed-T Motion-XOR fibers

**Steal:** LoAS MICRO 2024: fully temporal-parallel inner-product, T-bit CSF fibers of non-silent neurons, fast+laggy prefix-sum, pseudo-acc then per-timestep correction accs (16 TPPEs, 800 MHz, 32 nm). APEX only swaps the neuron (PASC-IF); do not restory APEX as a new join.

**Modify:** Pack **T=2 attention fibers separately from T=10 neuron fibers**. Join Q/K for overlap; second join/XOR against K_peer; co-silence from silent-drop packing. Dual-sparse **weights** are not frozen — this is spike-side join only.

**Not a copy:** LoAS/APEX join spike-pack GEMM, not Motion-XOR + mixed-T OF attention.

**Use as:** execution fabric under Rank 1, not the title.

## Rank 3 — Mixed-horizon binary ATLIF (membrane-private, spike-public)

**One sentence:** T=10 membrane / leak / threshold never leave the neuron island. Island boundary carries 1-bit fire (threshold compare). Attention T=2 and FC see binary streams. Downstream is add/sub, not MAC.

**Steal:**
- Spike-driven Transformer (NeurIPS 2023): mask+add, no multiply — **downstream contract**, not the neuron.
- Chen/Chang 28 nm Spike-IAND-Former: mux-unroll T=4/2/1, eliminates Vmem SRAM. **Template only.** ATLIF adaptive threshold + T=10 may be unable to drop Vmem. Safer: keep membrane SRAM **inside** the island (firewall), do not advertise membrane-less T=10.
- ASTER membrane persistence: steal the **stay-put** idea, not analog PIM.
- APEX: organization pattern “existing dataflow + one neuron circuit”. PASC-IF ≠ ATLIF. Preprint, not a recorded top venue.

**Modify:** One mux/schedule for T_snn=10 neuron service, one for T_w=2 attention wrappers. Outputs stay binary. Do not claim “first mixed-T ASIC”; claim “first mixed-horizon ATLIF for this event-OF encoder (T=10 neurons vs T=2 windows)”.

**C3 upgrade:** today’s C3 is exact T=10 coverage. Rank 3 makes it a **rate converter** (coverage remains).

**C1 consequence:** if post-ATLIF tensors are 1-bit, C1 product-capture of signed analog products is the wrong object. Optional remake: **add-forest** (parent + residual sources as add/sub), or kill C1 as a contribution.

## Rank 5 — Bishop ECP-style bound without computing S

**Steal:** drop a Q or K bundle when activity < θ_p because binary K (resp. Q) bounds every score in S=QK^T without computing S. AAC = AND+accumulate.

**Modify:** bound `overlap + same_zero/64 + motion_xor/4` from popcount proxies. **Do not** sell inactive-TTB skip. BSA/ECP **training** would unfreeze ep34 — inference-only bound is OK if lossless vs full leaf.

**Use as:** ablation in front of Rank 1.

## Rank 6 — K-zero / K-as-V gating (lossless, already named in P0)

Per-token K zero ~84% (ep35 P0). Frozen contract: K reused as V. `K==0 ⇒ skip score ALU and V gather`. Orthogonal to dirty: all-zero K is often not dirty.

Not empty-source skip: this is **key-silence implies value-not-fetched**.

## Rank 7 — Spatial dirty-run scheduler (OF data, not ASNA spatial-locality claim)

Dirty tokens have mean run length ~4.14 (ep35 P0); empty 4-token update bundles ~52%. Emit MX3P for contiguous dirty runs, bypass contiguous clean runs.

**Priors to cite, not copy:** ASNA-Flow spatial locality (already claimed); ExSpike adjacent-position event compression; Zhang similarity (already used in Rank 4).

**Differentiator:** run-length on **token-time dirty bits of Motion-XOR**, not flow-field spatial locality of a neuromorphic OF core.

## Rank 8 — Inter-frame event dirty tile (CBinfer event edition)

Event camera is a change detector. Tiles with no new events vs last **output frame** can skip encoder windows. Distinct from intra-frame empty window. Decoder/prediction heads care. Prove lossless or open AEE Pareto.

## Weak / do not headline

| Item | Why weak |
|---|---|
| ELSA bundled AER / spine-wise pipeline | After discarding Gustavson, remainder is packing. Thin vs empty-tile. |
| SpikeX activity tags | Keep only if they gate Motion-XOR tiles without collapsing to empty-tile. |
| Bishop asymmetric TTB | TTB already owned; only mixed width 2 vs 10 is incremental. |
| 4-bit W / dual-side sparsity | Unfreeze + AEE. |
| Scene-adaptive θ | Inference θ frozen in checkpoint. |
| Polar ON/OFF dual-rail | Current binary pos_rate, neg_rate=0 on captured ATLIF. Do not revive ternary Q/K. |

## C1 / C2 remake mapping

| Old island | New job | Mechanism | Old numbers |
|---|---|---|---|
| C1 product capture | Delete as contribution, or binary add-forest / dirty-conv | Rank 3 add-not-MAC, or Rank 8 dirty conv | 1.6945× out of abstract |
| C2 + TSBG | Attention score island | Rank 1+4 MX3P + dirty gate + Rank 6 K-zero | K8 area-efficiency may stay as PE org, not claim |
| C3 T=10 coverage | Dual-rate neuron island | Rank 3 membrane firewall | Coverage remains |

**Paper sentence:**

> Event-OF spikeformer attention is T=2 Motion-XOR three-popcount, not GEMM; neurons are T=10 membrane-private and binary-fire-public. Hardware is a dirty-lane score memo plus triple-popcount ALU plus membrane firewall, not product sparsity or weight broadcast.
