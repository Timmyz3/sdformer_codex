# F1 — Problem-first vs solution-first (shortlist)

Applied to the Phase 2 shortlist. Skill self-check: who suffers, is it unsolved, does a solution-first idea create new capability.

| ID | Mode | Who suffers today | Unsolved vs under-marketed | New capability? |
|---|---|---|---|---|
| I001 | Problem-first | Authors cannot point a circuits editor at a published Motion-XOR temporal-peer ALU | Bounded search: AND-PopCount and co-silence exist *separately*; temporal K_peer XOR as an ASIC leaf was not located | Yes: three popcounts including motion XOR, not a copy of FireFly-T |
| I002 | Problem-first | Recomputing a silent token wastes energy; selling that as row skip is a correctness bug | Token skip is known in ANN accelerators; **grain** vs Shiftmax is the unsolved part here | Only if enable is at the true grain; otherwise it is ordinary clock-gate |
| I003 | Solution-first (public kernels seeking a home on H67) | Training budget / hardware simplicity vs AEE | Public SDSA/QKFormer are solved; *whether OF needs the XOR term* is not | Discriminating ablation, not a new kernel |
| I004 | Mixed: problem-first (FC share) + solution-first (existing A6 RTL) | System FPS if T0 kills the attention leaf | LoAS FTP is solved; mixed T=2/10 packing on this net is not published as an ASIC | Only the mixed-horizon split is new capability |
| I005 | Solution-first (B8 RTL/CPU looking for a title) | Weight-row traffic | Broadcast is a solved family | Unlikely; keep as comparator |
| I007 | Solution-first (A8 RTL) meeting a real BN-boundary problem | Late θ after BN | Mixed-T neuron islands exist (Chen/Chang mux T=4/2/1); late-BN packet on ATLIF is local | Incremental unless real BN intervals show a new recovery law |

I005 and I007 fail a strict “new capability” reading; they still pass Phase 2 feasibility and are kept for information value / contingent circuit face.
