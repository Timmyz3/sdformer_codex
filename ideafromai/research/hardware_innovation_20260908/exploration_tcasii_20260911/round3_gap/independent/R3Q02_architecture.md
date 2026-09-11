# R3Q02 — architecture / overlay / dual-path last-use join (independent candidates)

Focal: after official AT-LIF \(\{0,\theta\}\) with \(\theta\) absorbed into next \(W\) (spike path is binary GeMM), what **one** architecture-level mechanism could be a TCAS-II 5-page letter for event-camera 2D optical-flow SNN-Transformer co-design?

Locked (do not restate as title): residual/PED/I24 is a **separate continuous tensor**; FireFly-T dual-engine is **A** for overlay control of sparse-conv vs binary-attention, **not** dual last-use of the same producer; DATE hybrid is **layer-split** dense-input / sparse-rest, **not** typed last-use; ESTU (same journal) is binary-SSA skip classification mW; ASNA-Flow already takes OF **spatial** locality; CIM is not a title; event-OF silicon already exists (SENECA, SNE, ERAFT FPGA, …); Spike-IAND **deletes residual ADD**. G1 typed last-use remains revise-not-title until 8088 wait-class is split — these ideas must name a **join/overlay mechanism**, not “typed last-use” as the claim. BN is hygiene. Prosperity/Gustav/FireFly-S bitmap-AND are **A** on the post-absorb binary path.

Problem: the OF SNN-Transformer block simultaneously owns a **binary spike GeMM live range** and a **continuous residual/PED/I24 live range**. Overlay that only switches layer-class engines, or layer-splits dense vs sparse, does not retire that pair.

---

## R3Q02-I1

**One sentence.** Intra-block **dual-tensor retire overlay**: a single controller issues an event-OF transformer/SNN block as two live ranges (post-absorb binary spike GeMM vs continuous PED/I24 residual) and retires the block only at their **join**, rather than switching engines by layer class (sparse-conv vs attention).

- **Assumptions:** After \(\theta\) absorb, the spike consumer is 0/1 GeMM; residual ADD still exists as a distinct continuous consumer (Spike-IAND’s deletion of ADD is A, not ours); FireFly-T already covers dual-engine overlay by operator class; DATE already covers inter-layer dense/sparse split.
- **Predicted observation:** Cycle-accurate overlay occupancy of an OF block is gated by `max(T_bin_last_use, T_ped_last_use)`, not by engine-switch latency; stall traces show residual ADD waiting after binary GeMM bus-release (or the reverse) on the same block.
- **Disconfirming evidence:** If a FireFly-T-style engine scheduler already inserts an equivalent join barrier for residual tensors, or if residual ADD is fused away so only one live range remains, the overlay has no second retire event.
- **Uncertainties:** Whether commercial event-OF cores (SENECA/SNE) already scoreboard residual vs spike inside a block; whether 5 pages can isolate the join controller from a full dual-engine story.
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q02-I2

**One sentence.** Make **residual ADD the only architectural join instruction**: two producers (binary spike bitmap after absorb, continuous I24/PED) keep **independent last-use pointers** and become architecturally visible only at the ADD that Spike-IAND would delete.

- **Assumptions:** Identity forbids treating AT-LIF \(o\) as a shared analog payload; the two tensors are different producers; keeping ADD is the co-design choice that creates a join site for 2D OF residual/flow update.
- **Predicted observation:** Killing the ADD (IAND-style) removes the join and collapses occupancy to binary GeMM alone; restoring ADD as a scored join increases SRAM residency of PED until spike last-use, with a measurable wait-class distinct from 8088 “generic last-use.”
- **Disconfirming evidence:** If OF accuracy or energy is unchanged when ADD is deleted or moved off the critical overlay, the join instruction is not the letter mechanism; if ADD is already a first-class scored op in ESTU/SENECA, it is A.
- **Uncertainties:** Exact I24/PED tensor shape in the target OF net; whether ADD join is circuit (TCAS-II) or ISA (too thick for 5 pages).
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q02-I3

**One sentence.** **Intra-layer dual-path overlay** (not DATE’s layer-split): inside one OF attention/MLP block, binary AND-GeMM lanes and continuous residual-MAC lanes share issue but **do not share last-use**, with a barrier only at block writeback.

- **Assumptions:** DATE hybrid is A for “first layers dense, rest sparse”; the remaining hole is same-layer concurrent tensor types after AT-LIF absorb; CIM / analog crossbar is out of title.
- **Predicted observation:** Utilization traces of the two lane classes desynchronize inside a layer (binary lanes idle while PED MAC drains, or vice versa); a DATE-style layer partition cannot reproduce that intra-layer occupancy hole.
- **Disconfirming evidence:** If the OF net’s residual is only between layers (no intra-layer continuous tensor), DATE already covers the split; if a single time-multiplexed MAC with one last-use matches both paths, dual lanes are unnecessary.
- **Uncertainties:** Whether event-OF transformers actually keep PED/I24 live inside attention vs only at block residual; PE-area tax of two lane types in a letter-length design.
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q02-I4

**One sentence.** Treat the **2D correlation / cost volume** as the **sole legal join site** of post-absorb binary event features and the continuous OF residual field — an overlay that retires a spatial tile only when both the binary matching GeMM and the continuous residual update have last-used that tile.

- **Assumptions:** Event-OF silicon already exists, so “an OF accelerator” is not the claim; ASNA-Flow already claimed OF **spatial** locality, so locality itself is A; the unclaimed mechanism is **typed join at the volume**, not sparse addressing.
- **Predicted observation:** Tile-retire time tracks the later of binary matching last-use and residual-field last-use; skipping spatial reuse (ASNA-Flow A) does not remove the dual-retire stall, while collapsing residual into binary matching does.
- **Disconfirming evidence:** If SENECA/ERAFT/SNE already barrier correlation vs residual with the same two wait events; if the target net has no cost volume (pure transformer OF) so the join site does not exist.
- **Uncertainties:** Mapping of SDFormer-class OF to an explicit volume vs implicit attention; whether the letter can cite existing OF HW as A without becoming a survey.
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q02-I5

**One sentence.** **Temporal dual-retire** of an event frame vs a persistent continuous flow residual: overlay holds the residual tensor across event packets and joins last-use on **time**, because spatial locality of OF is already taken.

- **Assumptions:** ASNA-Flow A = spatial locality; events arrive as sparse packets; after absorb, packet features are binary GeMM; the flow residual is a long-lived continuous tensor whose last-use is not the packet’s last-use.
- **Predicted observation:** Packet-done (binary GeMM bus idle) is not residual-done; energy/latency of OF updates is dominated by residual keep-alive across empty inter-event gaps, a stall class spatial tiling does not explain.
- **Disconfirming evidence:** If residual is rewritten every packet so temporal keep-alive is nil; if SENECA-class cores already implement inter-packet residual scoreboard as the published mechanism.
- **Uncertainties:** Event rate vs residual refresh rate on the chosen 2D OF benchmark; whether “keep-alive SRAM” reads as memory paper rather than architecture overlay.
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q02-I6

**One sentence.** Overlay **scoreboard that refuses to treat binary-spike bus-occupancy release as net service** of the continuous PED path: two done bits (`bin_lu`, `ped_lu`) so early release of the post-absorb 0/1 tensor cannot retire the OF block.

- **Assumptions:** Codex observation that bus occupancy released is **not** net service (early_V96 vs late_U32) generalizes from BN-hygiene to dual tensor types; BN itself stays hygiene and is not the title; 8088 wait-class must be split **by tensor kind**, not by a generic last-use flag.
- **Predicted observation:** A controller that retires on binary bus-idle reports false completion while PED/I24 still occupies SRAM; splitting done-bits recovers the true `max` occupancy and changes measured block latency without changing arithmetic count (arithmetic_saving can stay 0).
- **Disconfirming evidence:** If `bin_lu` and `ped_lu` always coincide on the OF net, the extra done-bit is dead; if reviewers read this as the delayed-V/BN buffer result (hygiene, integer 0-diff), it is not an architecture letter.
- **Uncertainties:** Need a wait-class split that is **not** the blocked G1 title; whether TCAS-II accepts a scoreboard-only mechanism without a new datapath.
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q02-I7

**One sentence.** **Single-array, dual-issue overlay**: one PE array time-multiplexes post-absorb binary AND-GeMM and continuous residual MAC, with **typed issue tags** and **two last-use counters**, explicitly **not** dual last-use of one AT-LIF amplitude and **not** FireFly-T’s two engines.

- **Assumptions:** Identity: spike path is binary GeMM; residual is another tensor; dual-engine control of conv vs attention is A; analog/CIM titles are forbidden; letter budget prefers one array over two engines.
- **Predicted observation:** Issue traces alternate tag-B (bitmap AND) and tag-C (PED MAC) on the same PEs; occupancy hole equals context-switch of tags, not engine-to-engine NoC; removing either tag collapses one of the two last-use counters to zero.
- **Disconfirming evidence:** If FireFly-T overlay control already time-multiplexes residual MAC on the attention engine with a published tag; if dual engines remain cheaper than dual-issue on one array for the OF mix.
- **Uncertainties:** Preview-V-scale microkernel occupancy (ordinary corner already thousands of slots on FMA lanes) may drown the tag-switch story; 5-page limit on hazard logic.
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate

---

## R3Q02-I8

**One sentence.** **Warp-sync last-use for 2D OF**: binary SSA/GeMM warps (event tokens) and continuous residual-warp (flow field / warped PED) share an event-tile issue group but **commit independently**, with a warp-level join that is not ESTU skip-classification and not dual last-use of the same producer.

- **Assumptions:** ESTU A = skip vs compute on binary SSA (mW classification); skip-not-compute does **not** authorize skip-not-join on the continuous residual; event-OF already exists, so the claim is the **warp join**, not “SNN-Transformer for OF.”
- **Predicted observation:** When ESTU-style skip fires on binary SSA, residual warp still last-uses the tile; energy saved on skipped AND-GeMM does not equal energy saved on the block unless the join is also skipped (which would break OF residual).
- **Disconfirming evidence:** If skip classification already gates residual writeback in ESTU or ASNA-Flow; if the OF net has no warped continuous field (residual is identity/zero).
- **Uncertainties:** Whether “warp” language maps cleanly onto SNN timesteps vs transformer tokens in 5 pages; risk of being read as another skip-classifier letter (ESTU).
- **Origin:** ai-assisted
- **Stage:** independent
- **Status:** candidate
