# F8 — Stakeholder rotation

Skill: spend time in each role; list top concerns; unaddressed concern with broadest impact becomes a research question. System in one paragraph:

H67 is a spike-driven Transformer for event-camera 2D optical flow (DSEC). Attention score is Motion-XOR (overlap + co-silence + temporal-peer K XOR), T_w=2, window 15×15, ATLIF z=θ×g, neuron horizon T=10. The brief must be one co-design object, algorithm + circuit, TSMC 28HPC+, AEE ≤ 1.259.

| Stakeholder | Top 3 concerns | Addressed by existing work? |
|---|---|---|
| **End user** (robot / driving stack consuming OF) | (1) AEE on DSEC, especially fast scenes (S3 is the busy attention stage). (2) Latency / FPS of the whole net, not a leaf. (3) Silent failure when skip is approximate. | (1) ep34 1.20 is good; (2) **unaddressed** until T0; (3) contract forbids lossy skip unless AEE gated |
| **Developer** (Codex / RTL) | (1) Reuse `h67_motionxor_score_q7.sv` vs new microarch. (2) Trace identity ep34 vs ep35. (3) Measurement contracts before production RTL. | (1) leaf exists; (2) **unaddressed** (no ep34 QK pack); (3) written in `MEASUREMENT_CONTRACTS.md` |
| **Theorist** | (1) Why XOR-in-time helps OF vs SDSA. (2) Formal lossless skip condition. (3) Mixed-T packing vs uniform-T join. | All three **under-addressed**; I003 and M2/M4 are the probes |
| **Adversary** (TCAS-II reviewer) | (1) “This is FireFly-T / LoAS / Prosperity.” (2) Leaf without system PPA. (3) ep35 table labeled ep34. | (1) named priors in SOURCE_LEDGER; (2) T0 gate; (3) explicit prohibition |
| **Ethicist** | (1) Unpublished traces in external tools. (2) Dual-use of event cameras / accelerators. (3) Energy of A800 retrains. | (1) keep traces local; (2) standard digital OF chip, not a bioweapon path; (3) one-A800 cap already in contract |
| **Regulator** | (1) Export-control on 28 nm compilers / PDK. (2) Not clinical. (3) Audit of AI-assisted ideation. | (1) institutional, out of this session; (2) N/A; (3) this register |
| **Operator** | (1) A800 queue. (2) Foundry SRAM compiler licenses. (3) ZCU102 optional, not blocking. | (1) 10-frame before full valid825; (2) PPA node frozen TSMC 28HPC+; (3) contract |

**Unaddressed concern with the broadest impact:** the reviewer’s “leaf without performance” plus the end-user’s FPS — both collapse to **M0/T0: what share is the score leaf on ep34**. That is why Phase 3’s winner is a *measurement next action*, not a locked title.

Missing voices (also in the K-Dense register): external circuits editor; lived-experience driver. Do not treat silence as agreement.
