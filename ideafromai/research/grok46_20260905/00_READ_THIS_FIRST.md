# 00 — Read this first (Codex / human)

Date: 2026-09-05. Pack: Grok 4.6 under canonical `/home/zhumd/work/ideafromai/`.

## One-line verdict

C1 and C2/TSBG are not new mechanisms. The only reviewer-honest new circuit object on the frozen Motion C12 ep34 workload is a **Motion-XOR score island** (three popcounts + temporal-peer K + dirty/similarity gate) plus a **mixed-horizon binary ATLIF island** (T=10 membrane private, 1-bit fire public). Do not rebuild product sparsity or weight-row broadcast.

## Frozen identity (do not unfreeze without a new AEE Pareto)

| Item | Value |
|---|---|
| Checkpoint | Motion C12 ep34, SHA prefix `4bbaf7fc` |
| Task quality | valid825 AEE **1.199514**, firing **5.6709%** |
| ATLIF | 105 installed; 93 invoked; 81 graph-live; **all 93 captured outputs binary** |
| Mixed T | 48 wrappers T=2 (attention), 45 wrappers T=10 (neurons) |
| Attention score | Motion-XOR, α=0.125: `round_even(128*(overlap + same_zero/64 + motion_xor/4)/32)` |
| overlap | `popcount(Q AND K)` |
| motion_xor | `popcount(K XOR K_peer)` |
| same_zero | co-silence |
| attn | `gate ⊙ K` (K reused as V) |
| Quant | `hardware_quant_enabled=false` |
| SRAM contract | foundry TS1N28 **1RW**, not dual-port / byte-write / analog CIM |
| C3 | exact Fixed-T10 **coverage**, not a third speedup |

Signed `load_source_sign` is a downstream polarity/correction protocol, **not** evidence that ATLIF tensors are analog.

## Two research workflows in this session (both Partial)

| Workflow | Conclusion | Use as |
|---|---|---|
| deep-research-2 | Only honest 28 nm 1RW story is Motion-XOR + mixed-T binary ATLIF. Steal LoAS FTP join, Chen mux-unroll (scoped), Bishop ECP bound, Zhang DLSS temporal similarity, ITA shift-round, α-XNOR co-silence term. | **Mechanism ranking** |
| deep-research | After excluding product-sparsity and broadcast, nothing leftover is both novel **and** implementable under exact/light co-design of frozen C12. Attention historically ~0.59% of cycle envelope (infinite attention speedup ≈1.006×). CICC/DLSS/MaxPool speculation NO-GO on Shiftmax. | **System-speedup gate** |

**Resolution for Codex:** these do not cancel. If the user wants **novelty of mechanism**, build the Motion-XOR island (object = attention leaf, not whole-net FPS). If the user wants **system cycle speedup ≥1.20× on frozen ep34**, first remeasure attention/FC/conv share on ep34; do not spend a month on an island that cannot move the envelope.

## Conflict with Grok Bot folder `/home/zhumd/work/ideafromai/`

That pack (OP-STW, HBG-RP, int8 ATLIF payload, pyramid residual, occlusion gate) assumes **real-valued ATLIF amplitudes not absorbed into W**. Frozen ep34 capture is **binary**. Do not implement HBG-RP as if it were the deployed contract. Canonical NeurIPS’25 AT-LIF is `{0, θ}`; inference threshold is checkpoint-static (no online homeostasis).

## Hard do-nots

- Do not claim C1 exact-subset forest as original (Prosperity HPCA’25).
- Do not claim TSBG as original broadcast (ELSA Gustavson, Eyeriss, SpikeX, FireFly-T).
- Do not claim empty-tile / inactive-TTB / zero-source skip as the new idea.
- Do not claim Shiftmax “scores are integer powers of two, therefore only shifts” — deploy is Q7 + finite exp LUT; gating codes are not integer 2^k.
- Do not restory ASNA-Flow spatial locality as our primitive.
- Do not copy Comperity DSE (AND-base + XOR-diff for GeMM reuse) and call it Motion-XOR. Their XOR is product reuse; ours must be a **score term**.
- Do not copy α-XNOR (CVPR’25) co-silence as invention. Map `α → same_zero/64`; the new term is **temporal-peer XOR**.
- Do not say “first mixed-T hardware” without scoping: Chen/Chang 28 nm already mux-unrolls T=4/2/1 LIF. Ours is T_snn=10 ATLIF vs T_w=2 attention on event-OF.
- Do not analog CIM / ASTER PIM / SpiDR dual-port IFspad.
- Do not multiply C1 × TSBG × anything.
- Do not edit `docs/359`, H81 RTL, or resume Codex via `codex exec resume` / websocket from this pack.
- Do not cite M-numbers, 0.0118 W, 1.770× encoder, ep44 AEE 1.2819, or 4.76× C2 vs single K1.
- IEEE “not submitted elsewhere”: TCAS-II XOR ISCAS, not both.

## Recommended pick (user has not yet locked)

**One attention island = Rank 1 + Rank 4 of `02_ranked_mechanisms.md` (do not split into two contribution bullets).**  
**One neuron island = Rank 3 dual-rate ATLIF (membrane stays; do not bet on membrane-less T=10).**  
**C1/C2 demoted** to execution context, not headline.

Next physical action: `07_next_stats_rtl_gates.md` (ep34 dirty census). No new production RTL until those tables exist.
